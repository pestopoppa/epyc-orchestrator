"""Adversarial eligibility tests for the E8 r2 race-only second successor."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "scripts/benchmark/prepare_e8_quality_baseline_v5_partial_r2_race_retry.py"
SPEC = importlib.util.spec_from_file_location("e8_r2_race_retry_test", PATH)
assert SPEC and SPEC.loader
RETRY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RETRY)


QUESTION = {"qid": "q0"}
RACE = "[ERROR: placement timeout role=frontdoor reason=race_lost holders=[0, 1, 2] after 90.0s]"
FAILED_C1_SOURCE = Path(
    "/mnt/raid0/llm/epyc-root/artifacts/operator/"
    "e8_quality_baseline_v5_partial_r2_mixed_tail_c1_successor_20260728T194407Z"
)
FAILED_C1_TREE_SHA256 = "4b7e66bec01c4eb2f65e10b75b9b1219ff74afda79f02873972194eefca2e286"


def _row(*, error: str = RACE, tokens: int = 0, answer: str | None = None) -> dict:
    return {
        "row_type": "question_result", "ordinal": 0,
        "answer": error if answer is None else answer,
        "result": {
            "qid": "q0", "question_id": "q0", "error": True,
            "error_detail": error, "tokens_generated": tokens, "route": "frontdoor",
        },
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _watcher(started_at: str, *, binding: str = "a") -> dict:
    return {
        "ok": True,
        "active_load": {"tier": 2, "repetition": 2},
        "started_at": started_at,
        "api_probe_urls": {"frontdoor": binding},
        "runtime_artifacts": {"server": {"identity": binding}},
    }


def test_copy_tree_rejects_destination_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    artifact = source / "artifact.json"
    artifact.write_text("sealed\n")
    expected = {"artifact.json": RETRY.sha256_path(artifact)}
    real_copy = RETRY.shutil.copyfile

    def corrupt_copy(origin: Path, destination: Path) -> None:
        real_copy(origin, destination)
        Path(destination).write_text("corrupt\n")

    monkeypatch.setattr(RETRY.shutil, "copyfile", corrupt_copy)
    with pytest.raises(ValueError, match="copy differs"):
        RETRY._copy_tree(source, tmp_path / "destination", expected)


def test_exact_race_lost_requires_zero_tokens_and_error_sentinel() -> None:
    assert RETRY._race_lost(_row(), QUESTION)
    assert not RETRY._race_lost(_row(tokens=1), QUESTION)
    assert not RETRY._race_lost(_row(answer="model output"), QUESTION)


def test_non_race_error_is_not_retry_eligible() -> None:
    assert not RETRY._race_lost(_row(error="timed out", answer=""), QUESTION)


def test_duplicate_sidecar_ordinal_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "sidecar.jsonl"
    _write_jsonl(path, [_row(), _row()])
    with pytest.raises(ValueError, match="duplicate"):
        RETRY._rows(path)


def test_terminal_failure_ledger_validates_sidecar_completion_order(tmp_path: Path) -> None:
    first = _row()
    first["ordinal"] = 279
    second = _row()
    second["ordinal"] = 97
    sidecars = {279: first, 97: second}
    expected = [
        {"ordinal": ordinal, "sidecar_sha256": RETRY.canonical_hash(row)}
        for ordinal, row in sidecars.items()
    ]
    _write_jsonl(
        tmp_path / "generation_failed_attempts.T2.r2.jsonl",
        [{"failures": expected, "disposition": "failed_closed_no_automatic_retry"}],
    )

    path, digest = RETRY._terminal_failure_ledger(tmp_path, sidecars, [97, 279])

    assert path.name == "generation_failed_attempts.T2.r2.jsonl"
    assert digest == RETRY.sha256_path(path)


def test_terminal_failure_ledger_rejects_ordinal_sorted_order(tmp_path: Path) -> None:
    first = _row()
    first["ordinal"] = 279
    second = _row()
    second["ordinal"] = 97
    sidecars = {279: first, 97: second}
    ordinal_sorted = [
        {"ordinal": ordinal, "sidecar_sha256": RETRY.canonical_hash(sidecars[ordinal])}
        for ordinal in (97, 279)
    ]
    _write_jsonl(
        tmp_path / "generation_failed_attempts.T2.r2.jsonl",
        [{"failures": ordinal_sorted, "disposition": "failed_closed_no_automatic_retry"}],
    )

    with pytest.raises(ValueError, match="exact sidecar failures"):
        RETRY._terminal_failure_ledger(tmp_path, sidecars, [97, 279])


def test_tree_pin_rejects_source_tamper_before_any_artifact_read(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    item = source / "immutable.txt"
    item.write_text("before")
    expected = RETRY.canonical_hash(RETRY.source_hashes(source))
    item.write_text("after")
    with pytest.raises(ValueError, match="explicit terminal tree hash"):
        RETRY.build_plan(source, expected)


def test_contaminated_predecessor_watcher_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "watcher.jsonl"
    rows = [_watcher("2026-01-01T00:00:00Z"), _watcher("2026-01-01T00:00:05Z")]
    rows[0]["ok"] = False
    _write_jsonl(path, rows)
    with pytest.raises(ValueError, match="watcher is contaminated"):
        RETRY._require_clean_predecessor_watcher(path)


def test_predecessor_watcher_rejects_gap_over_ratified_limit(tmp_path: Path) -> None:
    path = tmp_path / "watcher.jsonl"
    _write_jsonl(
        path,
        [_watcher("2026-01-01T00:00:00Z"), _watcher("2026-01-01T00:00:07.001Z")],
    )
    with pytest.raises(ValueError, match="watcher is contaminated"):
        RETRY._require_clean_predecessor_watcher(path)


def test_predecessor_watcher_rejects_immutable_binding_drift(tmp_path: Path) -> None:
    path = tmp_path / "watcher.jsonl"
    _write_jsonl(
        path,
        [_watcher("2026-01-01T00:00:00Z", binding="one"), _watcher("2026-01-01T00:00:05Z", binding="two")],
    )
    with pytest.raises(ValueError, match="watcher is contaminated"):
        RETRY._require_clean_predecessor_watcher(path)


def test_saved_rows_rejects_conflicting_ordinal_collision(tmp_path: Path) -> None:
    base, predecessor = tmp_path / "base", tmp_path / "predecessor"
    (base / "eval_sidecars").mkdir(parents=True)
    (predecessor / "eval_sidecars").mkdir(parents=True)
    base_row = _row(answer="base")
    predecessor_row = _row(answer="predecessor")
    _write_jsonl(base / "eval_sidecars/question_results.e8-t2-r2.jsonl", [base_row])
    _write_jsonl(
        predecessor / "eval_sidecars/question_results.e8-t2-r2-recovery.jsonl",
        [predecessor_row],
    )
    with pytest.raises(ValueError, match="sources conflict"):
        RETRY._saved_rows(predecessor, base)


def _clean_row(ordinal: int) -> dict:
    return {
        "row_type": "question_result",
        "ordinal": ordinal,
        "answer": f"answer-{ordinal}",
        "started_at_s": 30.0,
        "ended_at_s": 31.0,
        "result": {
            "qid": f"q{ordinal}",
            "question_id": f"q{ordinal}",
            "error": False,
            "tokens_generated": 2,
            "route": "frontdoor",
        },
    }


def _bind_tree(root: Path) -> dict[str, str]:
    hashes = {
        str(path.relative_to(root)): RETRY.sha256_path(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "source_binding.json"
    }
    RETRY.RECOVERY._write_json(
        root / "source_binding.json",
        {
            "source_sha256": hashes,
            "source_tree_sha256": RETRY.canonical_hash(hashes),
        },
    )
    return hashes


def _rebind_terminal_snapshot(root: Path) -> None:
    hashes = {
        str(path.relative_to(root)): RETRY.sha256_path(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != root / "source_binding.json"
    }
    RETRY.RECOVERY._write_json(
        root / "source_binding.json",
        {
            "source_sha256": hashes,
            "source_tree_sha256": RETRY.canonical_hash(hashes),
        },
    )


@pytest.mark.skipif(not FAILED_C1_SOURCE.is_dir(), reason="sealed E8 c1 source is host evidence")
def test_nested_terminalization_accepts_only_the_exact_enclosing_binding(tmp_path: Path) -> None:
    snapshot = tmp_path / "predecessor_snapshot"
    shutil.copytree(FAILED_C1_SOURCE / "predecessor_snapshot", snapshot)
    descriptor = RETRY.V4.load_json(FAILED_C1_SOURCE / "partial_r2_plan.json")[
        "mixed_tail_repair"
    ]["terminalization_transition"]

    assert RETRY._validate_terminalization_transition_semantically(snapshot) == descriptor


@pytest.mark.skipif(not FAILED_C1_SOURCE.is_dir(), reason="sealed E8 c1 source is host evidence")
def test_frozen_c1_source_is_directly_admitted_without_a_root_transition() -> None:
    assert not (FAILED_C1_SOURCE / RETRY.TERMINALIZATION_NAME).exists()

    plan = RETRY.build_plan(FAILED_C1_SOURCE, FAILED_C1_TREE_SHA256)

    assert plan["predecessor_tree_sha256"] == FAILED_C1_TREE_SHA256
    assert plan["generation_ordinals"] == plan["race_retry_ordinals"]
    assert plan["race_retry_ordinals"] == [97, 203, 279]
    assert (
        plan["mixed_tail_repair"]["terminalization_transition"]["sha256"]
        == "227bbd841f8fc3a4a58f2ef35d6452b63f7c34e21de4e75a407f21413d4409c6"
    )


@pytest.mark.skipif(not FAILED_C1_SOURCE.is_dir(), reason="sealed E8 c1 source is host evidence")
def test_copied_mixed_predecessor_cannot_enter_future_execution(tmp_path: Path) -> None:
    copied = tmp_path / "copied-mixed-predecessor"
    shutil.copytree(FAILED_C1_SOURCE, copied)
    with pytest.raises(ValueError, match="exact historical artifact"):
        RETRY.build_plan(copied, FAILED_C1_TREE_SHA256)


@pytest.mark.skipif(not FAILED_C1_SOURCE.is_dir(), reason="sealed E8 c1 source is host evidence")
def test_execute_binds_race_generation_targets_before_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class ProposalBound(Exception):
        pass

    captured: dict = {}
    claim = {
        "claims": [
            {
                "payload": {
                    "request_tag": "test-race-proposal",
                    "region": "q3",
                }
            }
        ]
    }
    monkeypatch.setenv("AUTOPILOT_EVAL_CONCURRENCY", str(RETRY.V4.CONCURRENCY))
    monkeypatch.setattr(RETRY.RECOVERY, "_capture_recovery_claim", lambda _args: claim)
    monkeypatch.setattr(RETRY.V4, "runtime_binding", lambda _args: {})
    monkeypatch.setattr(
        RETRY.RECOVERY,
        "preflight_frontdoor_capacity",
        lambda *_args, **_kwargs: {"capacity": RETRY.V4.CONCURRENCY},
    )
    monkeypatch.setattr(RETRY.RECOVERY, "_load_vector", lambda *_args: {})
    monkeypatch.setattr(
        RETRY.RECOVERY,
        "_reconstruct_questions",
        lambda *_args, **_kwargs: [{"qid": f"q{ordinal}"} for ordinal in range(RETRY.N)],
    )
    monkeypatch.setattr(RETRY.RECOVERY, "_instrument_identity", lambda _args: {"test": True})

    def bind_proposal(_output: Path, proposal: dict) -> None:
        captured.update(proposal)
        raise ProposalBound

    monkeypatch.setattr(RETRY.RECOVERY, "_bind_recovery_proposal", bind_proposal)
    args = SimpleNamespace(
        source_dir=FAILED_C1_SOURCE,
        expected_source_tree_sha256=FAILED_C1_TREE_SHA256,
        output_dir=tmp_path / "race-retry",
        api_url="http://127.0.0.1:8000",
        region_claim_tag="test-race-proposal",
        region_claim_regions="q3",
        region_claim_dir=tmp_path,
    )

    with pytest.raises(ProposalBound):
        RETRY.execute(args)

    plan = RETRY.V4.load_json(args.output_dir / "partial_r2_plan.json")
    assert plan["generation_ordinals"] == [97, 203, 279]
    assert captured["generation_ordinals_sha256"] == RETRY.canonical_hash([97, 203, 279])
    assert captured["race_retry_ordinals_sha256"] == captured["generation_ordinals_sha256"]


@pytest.mark.skipif(not FAILED_C1_SOURCE.is_dir(), reason="sealed E8 c1 source is host evidence")
@pytest.mark.parametrize("tamper", ["wrapper", "payload", "transition", "journal"])
def test_nested_terminalization_rejects_wrapper_or_terminal_evidence_tamper(
    tmp_path: Path, tamper: str
) -> None:
    snapshot = tmp_path / "predecessor_snapshot"
    shutil.copytree(FAILED_C1_SOURCE / "predecessor_snapshot", snapshot)
    if tamper == "wrapper":
        binding = RETRY.V4.load_json(snapshot / "source_binding.json")
        binding["source_tree_sha256"] = "0" * 64
        RETRY.RECOVERY._write_json(snapshot / "source_binding.json", binding)
    else:
        target = {
            "payload": snapshot / "generation_judge_traces.T2.r2.jsonl",
            "transition": snapshot / RETRY.TERMINALIZATION_NAME,
            "journal": snapshot / "recovery_rows.T2.r2.jsonl",
        }[tamper]
        target.write_bytes(target.read_bytes() + b"\n")
        _rebind_terminal_snapshot(snapshot)

    with pytest.raises(ValueError):
        RETRY._validate_terminalization_transition_semantically(snapshot)


def _mixed_chain_fixture(tmp_path: Path) -> tuple[Path, dict, list[dict], dict[int, dict]]:
    root = tmp_path / "mixed"
    original = root / "predecessor_snapshot"
    original.mkdir(parents=True)
    generation = [0, 1, 2, 3]
    questions = [
        {"qid": f"q{ordinal}", "scoring_method": "llm_judge"}
        for ordinal in generation
    ]
    race = {
        **_row(),
        "ordinal": 0,
        "started_at_s": 4.0,
        "ended_at_s": 6.0,
        "result": {**_row()["result"], "qid": "q0", "question_id": "q0"},
    }
    timeout = {
        **_row(
            error="[ERROR: Inference failed: chat_completions failed: timed out]",
            answer="[ERROR: Inference failed: chat_completions failed: timed out]",
        ),
        "ordinal": 1,
        "started_at_s": 30.0,
        "ended_at_s": 31.0,
        "result": {
            **_row()["result"],
            "qid": "q1",
            "question_id": "q1",
            "error_detail": "[ERROR: Inference failed: chat_completions failed: timed out]",
        },
    }
    scorer = {
        **_row(error="scoring_unavailable: synthetic", tokens=4, answer="preserved"),
        "ordinal": 2,
        "started_at_s": 30.0,
        "ended_at_s": 31.0,
        "result": {
            **_row(error="scoring_unavailable: synthetic", tokens=4)["result"],
            "qid": "q2",
            "question_id": "q2",
            "scoring_method": "llm_judge",
        },
    }
    overlap = _clean_row(3)
    overlap["started_at_s"], overlap["ended_at_s"] = 4.0, 6.0
    original_rows = {row["ordinal"]: row for row in (race, timeout, scorer, overlap)}
    (original / "eval_sidecars").mkdir()
    _write_jsonl(
        original / "eval_sidecars/question_results.e8-t2-r2-recovery.jsonl",
        list(original_rows.values()),
    )
    original_plan = {
        "schema": RETRY.SUCCESSOR.PLAN_SCHEMA,
        "generation_ordinals": generation,
    }
    RETRY.RECOVERY._write_json(original / "partial_r2_plan.json", original_plan)
    RETRY.RECOVERY._write_json(original / "recovery_proposal.json", {"sealed": True})
    watcher = [
        _watcher("1970-01-01T00:00:00Z"),
        {
            **_watcher("1970-01-01T00:00:05Z"),
            "finished_at": "1970-01-01T00:00:05Z",
            "ok": False,
            "api_failure_class": "api_transport_error",
            "api_probe_urls": {},
            "active_load": {"tier": 2, "repetition": 2},
            "binding_matches_pre": True,
            "immutable_files_match_pre": True,
            "autopilot_active": False,
        },
        {
            **_watcher("1970-01-01T00:00:10Z"),
            "finished_at": "1970-01-01T00:00:10Z",
            "ok": False,
            "api_failure_class": "api_transport_error",
            "api_probe_urls": {},
            "active_load": {"tier": 2, "repetition": 2},
            "binding_matches_pre": True,
            "immutable_files_match_pre": True,
            "autopilot_active": False,
        },
        _watcher("1970-01-01T00:00:15Z"),
        _watcher("1970-01-01T00:00:20Z"),
    ]
    for row in watcher:
        row.setdefault("finished_at", row["started_at"])
    _write_jsonl(original / "runtime_watch.r2.successor.jsonl", watcher)
    original_hashes = _bind_tree(original)
    watcher_evidence, _ = RETRY._mixed_watcher_evidence(
        original / "runtime_watch.r2.successor.jsonl"
    )
    classified = {
        "clean": [3],
        "race_lost": [0],
        "timeout": [1],
        "outer_timeout": [],
        "scorer_replay": [2],
    }
    descriptor = {
        "schema": RETRY.MIXED_REPAIR_SCHEMA,
        "repair_runner_sha256": RETRY.sha256_path(RETRY.MIXED_REPAIR_PATH),
        "predecessor": "/immutable/original",
        "predecessor_sha256": original_hashes,
        "predecessor_tree_sha256": RETRY.canonical_hash(original_hashes),
        "allowed_class_ordinals": classified,
        "allowed_class_ordinals_sha256": {
            kind: RETRY.canonical_hash(ordinals)
            for kind, ordinals in classified.items()
        },
        "classification_sha256": RETRY.canonical_hash(classified),
        "watcher_overlap_ordinals": [0, 3],
        "watcher_overlap_ordinals_sha256": RETRY.canonical_hash([0, 3]),
        "generation_retry_ordinals": [1, 3],
        "generation_retry_ordinals_sha256": RETRY.canonical_hash([1, 3]),
        "scorer_replay_ordinals": [2],
        "scorer_replay_ordinals_sha256": RETRY.canonical_hash([2]),
        "race_retry_ordinals": [0],
        "race_retry_ordinals_sha256": RETRY.canonical_hash([0]),
        "predecessor_watcher": watcher_evidence,
        "predecessor_provenance": {
            "path": "recovery_proposal.json",
            "sha256": RETRY.sha256_path(original / "recovery_proposal.json"),
        },
    }
    current_rows = {
        0: race,
        1: _clean_row(1),
        2: _clean_row(2),
        3: _clean_row(3),
    }
    plan = {**original_plan, "mixed_tail_repair": descriptor}
    RETRY.RECOVERY._write_json(root / "partial_r2_plan.json", plan)
    evidence = {
        "schema": RETRY.MIXED_REPAIR_SCHEMA,
        "descriptor_sha256": RETRY.canonical_hash(descriptor),
        **{
            key: descriptor[key]
            for key in (
                "predecessor_tree_sha256",
                "repair_runner_sha256",
                "allowed_class_ordinals",
                "allowed_class_ordinals_sha256",
                "classification_sha256",
                "watcher_overlap_ordinals",
                "watcher_overlap_ordinals_sha256",
                "generation_retry_ordinals",
                "generation_retry_ordinals_sha256",
                "scorer_replay_ordinals",
                "scorer_replay_ordinals_sha256",
                "race_retry_ordinals",
                "race_retry_ordinals_sha256",
            )
        },
        "generation_retry": [
            {
                "ordinal": ordinal,
                "before_sha256": RETRY.canonical_hash(original_rows[ordinal]),
                "after_sha256": RETRY.canonical_hash(current_rows[ordinal]),
            }
            for ordinal in (1, 3)
        ],
        "scorer_replay": [
            {
                "ordinal": 2,
                "before_sha256": RETRY.canonical_hash(original_rows[2]),
                "after_sha256": RETRY.canonical_hash(current_rows[2]),
            }
        ],
        "remaining_race_retry_ordinals": [0],
    }
    RETRY.RECOVERY._write_json(root / RETRY.MIXED_EVIDENCE_NAME, evidence)
    RETRY.RECOVERY._write_json(
        root / "recovery_proposal.json",
        {"schema": RETRY.MIXED_PROPOSAL_SCHEMA, "mixed_tail_repair": descriptor},
    )
    return root, plan, questions, current_rows


def test_mixed_predecessor_chain_recomputes_nested_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, plan, questions, current_rows = _mixed_chain_fixture(tmp_path)
    monkeypatch.setattr(
        RETRY,
        "_clean",
        lambda row, _question: row.get("result", {}).get("error") is not True,
    )

    chain = RETRY.validate_mixed_predecessor(root, plan, questions, current_rows)

    assert chain is not None
    assert chain["descriptor"]["generation_retry_ordinals"] == [1, 3]
    assert chain["descriptor"]["race_retry_ordinals"] == [0]


def test_mixed_predecessor_chain_rejects_runner_or_evidence_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, plan, questions, current_rows = _mixed_chain_fixture(tmp_path)
    monkeypatch.setattr(
        RETRY,
        "_clean",
        lambda row, _question: row.get("result", {}).get("error") is not True,
    )
    plan["mixed_tail_repair"]["repair_runner_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="runner"):
        RETRY.validate_mixed_predecessor(root, plan, questions, current_rows)
