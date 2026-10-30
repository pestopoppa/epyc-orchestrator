import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.autopilot.process_verification import (
    binary_score,
    parse_step_verdicts,
    selected_results,
    verification_prompt,
    verify_results,
)
from scripts.server.eval_process_verifier import NativeProcessVerifier, ProcessVerifierConfig
from src.autopilot_core.verification_families import cross_family_status


def row(qid="a", confidence=0.5, route="gemma-3-1b-it"):
    return SimpleNamespace(
        qid=qid,
        question_id=qid,
        prompt="2+2?",
        expected="4",
        answer="Step 1: 2+2=4",
        error=None,
        disposition="scored",
        eval_partition="core",
        confidence=confidence,
        confidence_source="binary_correctness_proxy",
        route_used=route,
        details={},
        correct=True,
    )


def test_known_family_and_guard_precedes_backend_load(tmp_path):
    assert cross_family_status("Qwen3-35B", "ThinkPRM-1.5B") == (False, "same_family")
    assert cross_family_status("gemma-3-1b-it", "ThinkPRM-1.5B") == (True, "cross_family")
    assert cross_family_status("unknown", "ThinkPRM-1.5B") == (False, "unverified")

    class Backend:
        model_identity = "ThinkPRM-1.5B"

        def __enter__(self):
            pytest.fail("same family model must not load")

    with pytest.raises(ValueError, match="same_family"):
        verify_results([row(route="Qwen3-35B")], Backend(), n=1, artifact_path=tmp_path / "x")


def test_probability_math_and_missing_candidate_is_not_zero():
    response = {
        "completion_probabilities": [
            {"top_logprobs": [{"id": 1, "logprob": -1000}, {"id": 2, "logprob": -1001}]}
        ]
    }
    measured = binary_score(response, {"Yes": 1, "No": 2})
    assert measured["score"] == pytest.approx(0.73105857863)
    for shift in (-100, 0, 100):
        shifted = {
            "probs": [
                {
                    "top_logprobs": [
                        {"id": 1, "logprob": -3 + shift},
                        {"id": 2, "logprob": -4 + shift},
                    ]
                }
            ]
        }
        assert binary_score(shifted, {"Yes": 1, "No": 2})["score"] == measured["score"]
    assert (
        binary_score({"probs": [{"top_logprobs": [{"id": 1, "logprob": 0}]}]}, {"Yes": 1, "No": 2})[
            "score"
        ]
        is None
    )


def test_exact_prompt_native_boxed_verdict_and_selection():
    prompt = verification_prompt("2+2?", "Step 1: 4")
    assert "[Math Problem]\n\n2+2?" in prompt and "[Solution]\n\nStep 1: 4" in prompt
    steps = parse_step_verdicts("Step 1 is \\boxed{correct}\nStep 2 is \\boxed{incorrect}")
    assert [s["verdict"] for s in steps] == ["correct", "incorrect"]
    assert not parse_step_verdicts("I think Step 1 looks correct")
    rows = [row("z", 0.5), row("a", 0.5), row("x", 0.1)]
    bad = row("excluded")
    bad.error = "offline"
    assert [r.qid for r in selected_results(rows + [bad], 2)] == ["a", "z"]
    with pytest.raises(ValueError):
        selected_results(rows, 101)


def test_durable_attribution_preserves_original_verdict(tmp_path):
    original = row()

    class Backend:
        model_identity = "ThinkPRM-1.5B"

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def verify(self, *_):
            return {"steps": [{"step": 1, "verdict": "incorrect"}], "probability": {"score": 0.1}}

    path = tmp_path / "result.jsonl"
    result = verify_results([original], Backend(), n=1, artifact_path=path)
    assert original.correct is True
    assert result["binary_proxy_ties_are_not_uncertainty"]
    saved = [json.loads(line) for line in path.read_text().splitlines()]
    assert saved[0]["details"] == original.details
    assert saved[-1] == {"kind": "process_verification_complete", "n": 1}
    assert result["artifact_fsync_returned"]


def test_sequential_boundary_required_without_spawning(tmp_path):
    config = ProcessVerifierConfig(
        tmp_path / "model", tmp_path / "server", tmp_path, tmp_path / "log"
    )
    with pytest.raises(ValueError, match="generation_drained"):
        with NativeProcessVerifier(config):
            pytest.fail("not admitted")


def test_deadline_returns_partial_and_still_drains(tmp_path):
    import time

    class Backend:
        model_identity = "ThinkPRM-1.5B"
        drained = False

        def __enter__(self):
            return self

        def __exit__(self, *_):
            self.drained = True

        def verify(self, *_):
            pytest.fail("no request after cutoff")

    backend = Backend()
    result = verify_results(
        [row()],
        backend,
        n=1,
        artifact_path=tmp_path / "partial",
        deadline_monotonic=time.monotonic() - 1,
    )
    assert result["partial"] and result["completed_n"] == 0 and backend.drained
    assert json.loads((tmp_path / "partial").read_text())["kind"] == "process_verification_partial"


def test_attribution_survives_later_row_failure(tmp_path):
    class Backend:
        model_identity = "ThinkPRM-1.5B"
        calls = 0

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def verify(self, *_):
            self.calls += 1
            if self.calls == 2:
                raise TimeoutError("deadline")
            return {"steps": [], "probability": {"score": 0.5}}

    path = tmp_path / "partial"
    with pytest.raises(TimeoutError):
        verify_results([row("a"), row("b")], Backend(), n=2, artifact_path=path)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert records[0]["qid"] == "a"
    assert records[-1]["kind"] == "process_verification_interrupted"


def test_resume_exact_identity_reuses_once_and_rejects_torn_foreign_duplicate(tmp_path):
    from scripts.benchmark.eval_t2_process_verification import restore_attribution

    original = row("a")
    identity = {"input": "sha", "source": "sha", "model": "sha", "plan": "sha"}
    assert restore_attribution([original], tmp_path, identity) == set()
    feedback = {"verification_cot": "Step 1 is \\boxed{correct}", "generator": original.route_used}
    record = {"qid": "a", "details": {"process_verification": feedback}}
    artifact = tmp_path / "process-attribution.1.jsonl"
    artifact.write_text(json.dumps(record) + "\n")
    assert restore_attribution([original], tmp_path, identity) == {"a"}
    assert original.details["process_verification"] == feedback
    with pytest.raises(ValueError, match="plan changed"):
        restore_attribution([original], tmp_path, {**identity, "model": "different"})
    artifact.write_text(json.dumps(record) + "\n" + json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="duplicate or foreign"):
        restore_attribution([original], tmp_path, identity)
    artifact.write_text(json.dumps({**record, "qid": "foreign"}) + "\n")
    with pytest.raises(ValueError, match="duplicate or foreign"):
        restore_attribution([original], tmp_path, identity)
    artifact.write_text(json.dumps(record) + '\n{"qid":')
    with pytest.raises(json.JSONDecodeError):
        restore_attribution([original], tmp_path, identity)


def test_cli_native_transport_timeout_returns_partial75(tmp_path, monkeypatch):
    import httpx
    from scripts.benchmark import eval_t2_process_verification as cli
    from scripts.server import eval_process_verifier as native
    import src.config

    model = tmp_path / "models/ThinkPRM-1.5B/ThinkPRM-1.5B-Q4_K_M.gguf"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"fixture-model")
    server = tmp_path / "kernels/production/cpu/llama-server"
    server.parent.mkdir(parents=True)
    server.write_bytes(b"fixture-server")
    monkeypatch.setattr(
        src.config,
        "get_config",
        lambda: SimpleNamespace(
            paths=SimpleNamespace(llm_root=tmp_path, models_dir=tmp_path / "models")
        ),
    )

    class Backend:
        model_identity = "ThinkPRM-1.5B"
        lifecycle = {"started": False, "drained": True}

        def __init__(self, _config):
            pass

        def __enter__(self):
            raise httpx.ReadTimeout("bounded request timeout")

        def __exit__(self, *_):
            pass

    monkeypatch.setattr(native, "NativeProcessVerifier", Backend)
    payload = {
        "question_id": "a",
        "suite": "math",
        "prompt": "2+2?",
        "expected": "4",
        "answer": "4",
        "route_used": "gemma-3-1b-it",
        "confidence": 0.5,
        "confidence_source": "operational_control",
        "disposition": "scored",
    }
    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps([payload]))
    output = tmp_path / "output"
    assert (
        cli.main(
            [
                "--input",
                str(input_path),
                "--output-dir",
                str(output),
                "--uncertain-n",
                "1",
                "--generation-drained",
            ]
        )
        == 75
    )
    result = json.loads((output / "RESULT.json").read_text())
    assert result["partial"] and result["error"] == "ReadTimeout"
    assert result["lifecycle"]["drained"]


def test_eval_t2_default_off_and_durable_postpass(tmp_path, monkeypatch):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/autopilot"))
    import eval_tower
    from eval_tower import EvalTower, QuestionResult

    tower = EvalTower()
    question = {"id": "a", "suite": "math", "prompt": "2+2?", "expected": "4"}
    monkeypatch.setattr(tower, "_load_pool", lambda: {"math": [question]})
    monkeypatch.setattr(tower, "_t1_core_exclusion_qids", lambda *a, **k: (set(), {}))
    monkeypatch.setattr(tower, "_load_tool_sentinels", lambda: [])
    result_row = QuestionResult(
        "a", "math", "2+2?", "4", answer="Step 1: 4", route_used="gemma-3-1b-it", correct=True
    )
    monkeypatch.setattr(tower, "_eval_batch", lambda *a, **k: [result_row])
    monkeypatch.setattr(eval_tower, "_stamp_eval_instrument", lambda result, **kwargs: result)
    tower.set_question_artifact_dir(tmp_path)
    baseline = tower.eval_t2(n=1)
    assert "process_verification" not in baseline.details
    assert result_row.details == {}

    class Backend:
        model_identity = "ThinkPRM-1.5B"

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def verify(self, *_):
            return {"steps": [], "probability": {"score": 0.75}}

    verified = tower.eval_t2(n=1, process_verifier=Backend(), verification_n=1)
    assert verified.quality == baseline.quality
    assert (
        verified.question_results[0]["details"]["process_verification"]["probability"]["score"]
        == 0.75
    )
    assert Path(verified.details["process_verification"]["artifact_path"]).exists()
