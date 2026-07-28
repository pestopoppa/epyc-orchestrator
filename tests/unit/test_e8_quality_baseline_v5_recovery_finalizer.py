"""Focused contract tests for the separate E8 recovered-r2 validator context."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VALIDATOR_PATH = PROJECT_ROOT / "scripts/benchmark/validate_e8_quality_baseline_v5.py"
FINALIZER_PATH = PROJECT_ROOT / "scripts/benchmark/finalize_e8_quality_baseline_v5_recovery_r2.py"
spec = importlib.util.spec_from_file_location("e8_recovery_context_validator", VALIDATOR_PATH)
assert spec is not None and spec.loader is not None
validator = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = validator
spec.loader.exec_module(validator)
finalizer_spec = importlib.util.spec_from_file_location("e8_recovery_finalizer", FINALIZER_PATH)
assert finalizer_spec is not None and finalizer_spec.loader is not None
finalizer = importlib.util.module_from_spec(finalizer_spec)
sys.modules[finalizer_spec.name] = finalizer
finalizer_spec.loader.exec_module(finalizer)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _context(tmp_path: Path) -> tuple[dict, dict]:
    root = tmp_path / "bundle"
    snapshot = root / "intermediate/source_snapshot"
    _write_json(snapshot / "question_vector.T2.json", {"fixed": "source"})
    source_hashes = {"question_vector.T2.json": _sha(snapshot / "question_vector.T2.json")}
    _write_json(
        snapshot / "source_binding.json",
        {"source_sha256": source_hashes, "source_tree_sha256": validator.canonical_hash(source_hashes)},
    )
    reuse = list(range(59))
    replay = [59, 60, 61]
    generation = list(range(62, 500))
    plan = {
        "schema": validator.RECOVERY_R2_PLAN_SCHEMA,
        "protocol_id": "e8_quality_full_pool_tier_baseline.v5",
        "source_sha256": source_hashes,
        "source_tree_sha256": validator.canonical_hash(source_hashes),
        "tier": 2,
        "repetition": 2,
        "n": 500,
        "generation_concurrency": 3,
        "reuse_ordinals": reuse,
        "scorer_replay_ordinals": replay,
        "generation_ordinals": generation,
    }
    _write_json(root / "intermediate/partial_r2_plan.json", plan)
    claim = {"claims": [{"payload": {"request_tag": "e8", "region": "q0"}}], "global_claims": [{"region": "q0"}]}
    proposal = {
        "schema": validator.RECOVERY_R2_PROPOSAL_SCHEMA,
        "status": "observation_only",
        "protocol_id": "e8_quality_full_pool_tier_baseline.v5",
        "source_tree_sha256": plan["source_tree_sha256"],
        "generation_concurrency": 3,
        "generation_ordinals_sha256": validator.canonical_hash(generation),
        "scorer_replay_ordinals_sha256": validator.canonical_hash(replay),
        "instrument": {"commit": "c", "runner_sha256": "r", "measurement_source_sha256": {"/a": "1" * 64, "/b": "2" * 64, "/c": "3" * 64}},
        "output_namespace": "/tmp/recovery",
        "region_claim": {"tag": "e8", "regions": ["q0"]},
        "frontdoor_capacity": {"capacity": 3},
        "application": "requires_separate_human_finalizer",
    }
    _write_json(root / "intermediate/recovery_proposal.json", proposal)
    response = root / "responses.T2.r2.jsonl"
    sidecar = root / "eval_sidecars/question_results.e8-t2-r2.jsonl"
    trace = root / "judge_traces.T2.r2.jsonl"
    raw = root / "raw.T2.r2.json"
    _write_jsonl(response, [])
    _write_jsonl(sidecar, [])
    _write_jsonl(trace, [])
    _write_json(raw, {"q": 0.0})
    watcher_path = root / "intermediate/runtime_watch.r2.jsonl"
    watcher_rows = [
        {"ok": True, "started_at": "2026-01-01T00:00:00Z", "active_load": {"tier": 2, "repetition": 2}, "api_probe_urls": {}, "runtime_artifacts": {}},
        {"ok": True, "started_at": "2026-01-01T00:00:05Z", "active_load": None, "api_probe_urls": {}, "runtime_artifacts": {}},
    ]
    _write_jsonl(watcher_path, watcher_rows)
    complete = {
        "schema": validator.RECOVERY_R2_COMPLETE_SCHEMA,
        "status": "intermediate_r2_complete",
        "plan_sha256": _sha(root / "intermediate/partial_r2_plan.json"),
        "responses_sha256": _sha(response),
        "sidecar_sha256": _sha(sidecar),
        "trace_sha256": _sha(trace),
        "raw_sha256": _sha(raw),
        "watcher": {
            "path": str(watcher_path),
            "sha256": _sha(watcher_path),
            "samples": 2,
            "claim_before": claim,
            "claim_after": claim,
            "proposal_sha256": _sha(root / "intermediate/recovery_proposal.json"),
            "binding_sha256": validator._monitor_binding_sha256(watcher_rows[0]),
            "observed_gap_count_over_7s": 0,
            "observed_max_gap_s": 5.0,
        },
        "claim": claim,
    }
    _write_json(root / "intermediate/r2_complete.json", complete)
    journal = [
        {"ordinal": ordinal, "source": source, "response": {"qid": f"q{ordinal}"}}
        for source, ordinals in (("reuse", reuse), ("scorer_replay", replay), ("generation", generation))
        for ordinal in ordinals
    ]
    _write_jsonl(root / "intermediate/recovery_rows.T2.r2.jsonl", journal)
    complete["journal_sha256"] = _sha(root / "intermediate/recovery_rows.T2.r2.jsonl")
    _write_json(root / "intermediate/r2_complete.json", complete)
    context = {
        "schema": validator.RECOVERY_R2_CONTEXT_SCHEMA,
        "recovery_runner": {"path": "/reviewed/recovery.py", "sha256": "a" * 64},
        "finalizer_runner": {"path": "/reviewed/finalizer.py", "sha256": "b" * 64},
        "dependency_sha256": {"v5": "c" * 64, "resume": "d" * 64, "recovery": "a" * 64},
        "source_binding": str(snapshot / "source_binding.json"),
        "source_binding_sha256": _sha(snapshot / "source_binding.json"),
        "source_tree_sha256": plan["source_tree_sha256"],
        "plan_path": str(root / "intermediate/partial_r2_plan.json"),
        "plan_sha256": _sha(root / "intermediate/partial_r2_plan.json"),
        "proposal_path": str(root / "intermediate/recovery_proposal.json"),
        "proposal_sha256": _sha(root / "intermediate/recovery_proposal.json"),
        "complete_path": str(root / "intermediate/r2_complete.json"),
        "complete_sha256": _sha(root / "intermediate/r2_complete.json"),
        "watcher_path": str(watcher_path),
        "watcher_sha256": _sha(watcher_path),
        "response_path": str(response),
        "sidecar_path": str(sidecar),
        "trace_path": str(trace),
        "raw_path": str(raw),
        "journal_path": str(root / "intermediate/recovery_rows.T2.r2.jsonl"),
        "journal_sha256": _sha(root / "intermediate/recovery_rows.T2.r2.jsonl"),
    }
    return root, context


def _validate(root: Path, context: dict) -> dict:
    return validator.validate_recovery_r2_context(
        {"recovery_r2": context}, evidence_root=root, expected_recovery_runner_sha256="a" * 64,
        expected_finalizer_runner_sha256="b" * 64,
    )


def test_recovery_r2_context_accepts_hash_bound_59_3_438_bundle(tmp_path: Path) -> None:
    root, context = _context(tmp_path)
    accepted = _validate(root, context)
    assert accepted is not None
    assert len(accepted["plan"]["generation_ordinals"]) == 438


@pytest.mark.parametrize("field", ["schema", "plan_sha256", "source_tree_sha256"])
def test_recovery_r2_context_rejects_schema_or_hash_drift(tmp_path: Path, field: str) -> None:
    root, context = _context(tmp_path)
    context[field] = "wrong" if field == "schema" else "0" * 64
    with pytest.raises(ValueError):
        _validate(root, context)


def test_recovery_r2_context_rejects_ordinal_allowlist_drift(tmp_path: Path) -> None:
    root, context = _context(tmp_path)
    plan_path = Path(context["plan_path"])
    plan = json.loads(plan_path.read_text())
    plan["generation_ordinals"][-1] = 61
    _write_json(plan_path, plan)
    context["plan_sha256"] = _sha(plan_path)
    with pytest.raises(ValueError, match="allowlist"):
        _validate(root, context)


def test_recovery_r2_context_rejects_source_watcher_and_claim_drift(tmp_path: Path) -> None:
    root, context = _context(tmp_path)
    source = Path(context["source_binding"]).parent / "question_vector.T2.json"
    source.write_text("tampered\n")
    with pytest.raises(ValueError, match="source binding"):
        _validate(root, context)
    root, context = _context(tmp_path / "watcher")
    complete_path = Path(context["complete_path"])
    complete = json.loads(complete_path.read_text())
    complete["watcher"]["sha256"] = "0" * 64
    _write_json(complete_path, complete)
    context["complete_sha256"] = _sha(complete_path)
    with pytest.raises(ValueError, match="watcher"):
        _validate(root, context)
    root, context = _context(tmp_path / "claim")
    complete_path = Path(context["complete_path"])
    complete = json.loads(complete_path.read_text())
    complete["watcher"]["claim_after"] = {"changed": True}
    _write_json(complete_path, complete)
    context["complete_sha256"] = _sha(complete_path)
    with pytest.raises(ValueError, match="claim"):
        _validate(root, context)


def test_finalizer_plan_accepts_the_preserved_completed_r1_source() -> None:
    source = Path(
        "/mnt/raid0/llm/epyc-root/artifacts/operator/"
        ".e8_quality_baseline_v5_partial_resume_promptfix_20260728.staging-b0d7ce62d6e04509a1cec7849aa68832"
    )
    plan = finalizer.build_plan(source)
    assert plan["banked"] == {"tiers": [1], "t2_r1": True}
    assert plan["fresh_collection"] == [{"tier": 2, "repetition": 3}]


def test_install_recovered_r2_replaces_only_hash_bound_partial_source_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging = tmp_path / "staging"
    snapshot = staging / "source_snapshot"
    recovered = tmp_path / "recovered"
    partial = {
        "eval_sidecars/question_results.e8-t2-r2.jsonl": b"partial-sidecar\n",
        "judge_traces.T2.r2.jsonl": b"partial-trace\n",
    }
    hashes: dict[str, str] = {}
    for relative, payload in partial.items():
        path = snapshot / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        (staging / relative).parent.mkdir(parents=True, exist_ok=True)
        (staging / relative).write_bytes(payload)
        hashes[relative] = _sha(path)
    _write_json(snapshot / "source_binding.json", {"source_sha256": hashes})
    replacement = {
        "responses.T2.r2.jsonl": b"recovered-responses\n",
        "eval_sidecars/question_results.e8-t2-r2.jsonl": b"recovered-sidecar\n",
        "judge_traces.T2.r2.jsonl": b"recovered-trace\n",
        "raw.T2.r2.json": b"recovered-raw\n",
    }
    for relative, payload in replacement.items():
        path = recovered / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    _write_json(staging / "question_vector.T2.json", {"core_id": "core"})
    _write_json(
        staging / "scoring_vector.T2.json",
        {"questions": [{"qid": f"q{ordinal}"} for ordinal in range(500)]},
    )
    monkeypatch.setattr(finalizer.RESUME, "_pristine_reference", lambda **_: {"artifacts": {}})
    monkeypatch.setattr(finalizer.RESUME, "_questions", lambda *_: [])
    monkeypatch.setattr(
        finalizer.RESUME,
        "_banked_observation_and_detail",
        lambda **_: ({"q": 1}, {"scorer_tail_replay": ["old"], "scorer_sidecar_replacement_ordinals": [1]}),
    )
    observation, detail = finalizer._install_recovered_r2(
        {"root": recovered, "plan": {"scorer_replay_ordinals": [1, 2, 3]}}, staging, tmp_path / "published", object()
    )
    assert observation == {"q": 1}
    assert detail["scorer_tail_replay"] == [
        {"ordinal": ordinal, "qid": f"q{ordinal}", "outcome": "recovered"}
        for ordinal in (1, 2, 3)
    ]
    assert all((staging / relative).read_bytes() == payload for relative, payload in replacement.items())


def test_install_recovered_r2_rejects_mutated_partial_source_file(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    snapshot = staging / "source_snapshot"
    relative = "eval_sidecars/question_results.e8-t2-r2.jsonl"
    source = snapshot / relative
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("sealed\n")
    _write_json(snapshot / "source_binding.json", {"source_sha256": {relative: _sha(source)}})
    target = staging / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("tampered\n")
    for name in ("responses.T2.r2.jsonl", "judge_traces.T2.r2.jsonl", "raw.T2.r2.json"):
        path = tmp_path / "recovered" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("new\n")
    (tmp_path / "recovered" / relative).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "recovered" / relative).write_text("new\n")
    with pytest.raises(ValueError, match="pre-existing partial r2 artifact"):
        finalizer._install_recovered_r2(
            {"root": tmp_path / "recovered"}, staging, tmp_path / "published", object()
        )


def test_validate_intermediate_rejects_lexical_symlink(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="must not be a symlink"):
        finalizer.validate_intermediate(link)
