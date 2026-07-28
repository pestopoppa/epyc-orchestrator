from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "scripts/benchmark/final_c1_retry.py"
SPEC = importlib.util.spec_from_file_location("e8_final_c1_retry_test", PATH)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNNER
SPEC.loader.exec_module(RUNNER)


@pytest.fixture(autouse=True)
def _clean_pinned_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ratifier = tmp_path / "ratifier.sh"
    ratifier.write_text("#!/bin/bash\n", encoding="utf-8")
    monkeypatch.setattr(RUNNER, "CANONICAL_RATIFIER", ratifier)
    monkeypatch.setattr(
        RUNNER,
        "_runtime_git_identity",
        lambda _repository: {
            "runtime_top": str(RUNNER.ROOT),
            "canonical_top": str(RUNNER.CANONICAL_REPOSITORY),
            "same_repository": True,
            "commit": "a" * 40,
            "tree": "b" * 40,
            "clean": True,
        },
    )


def _receipt() -> dict:
    return {
        "schema": RUNNER.RECEIPT_SCHEMA,
        "status": "ratified",
        "protocol_id": RUNNER.PROTOCOL_ID,
        "ratified_at": "2026-07-28T20:00:00Z",
        "human_attestation": RUNNER.ATTESTATION,
        "amendment_script": {
            "path": str(RUNNER.CANONICAL_RATIFIER),
            "sha256": RUNNER.sha256_path(RUNNER.CANONICAL_RATIFIER),
        },
        "failed_race_evidence": {
            "namespace": str(RUNNER.SOURCE),
            "canonical": True,
            "files": RUNNER.FAILED_RACE_FILES,
            "recorded_trees": {
                "plan_failed_source_tree_sha256": "92241f793c254dcf71dfca452f8cc50416d2fb1410698584b514ff3c14c5571a",
                "proposal_source_tree_sha256": "b821900094e866027d9a1561b21d91eb09f6a02ff92b8d91b133df57c7d5ce2d",
            },
            "failed_timeout_sidecars": (
                "97:a550c07752f8dedc0fdf5c4582b587c90f3b624405ed1454f628e523c100cae9,"
                "279:a41be1b012bb33475a5d8c9fd2e810c5b6dab651d123e3006f07cfc3f7fc835e"
            ),
        },
        "source": {
            "path": str(RUNNER.SOURCE),
            "tree_sha256": RUNNER.SOURCE_TREE_SHA256,
        },
        "instrument": {
            "repository": str(RUNNER.CANONICAL_REPOSITORY),
            "commit": "a" * 40,
            "tree": "b" * 40,
            "ratifier_interpreter": "/usr/bin/python3",
            "runner": {
                "path": "scripts/benchmark/final_c1_retry.py",
                "sha256": RUNNER.sha256_path(PATH),
            },
            "validator": {
                "path": "scripts/benchmark/final_c1_validator.py",
                "sha256": RUNNER.sha256_path(RUNNER.VALIDATOR_PATH),
            },
        },
        "authorization": RUNNER._receipt_contract(),
        "non_authorizations": {
            "no_inference_by_ratifier": True,
            "no_lineup_mutation": True,
            "no_state_write": True,
        },
    }


def _write_receipt(path: Path, receipt: dict | None = None) -> Path:
    path.write_text(json.dumps(receipt or _receipt()) + "\n", encoding="utf-8")
    RUNNER.CANONICAL_RECEIPT = path
    return path


@pytest.mark.skipif(not RUNNER.SOURCE.is_dir(), reason="sealed failed race evidence unavailable")
def test_exact_failed_race_source_is_the_only_admitted_source() -> None:
    validated = RUNNER.validate_failed_source()
    assert len(validated["hashes"]) == 806
    assert RUNNER.canonical_hash(validated["hashes"]) == RUNNER.SOURCE_TREE_SHA256
    assert sorted(set(range(500)) - set(validated["journal"])) == [97, 279]


def test_source_alias_and_unreviewed_namespace_are_rejected(tmp_path: Path) -> None:
    with pytest.raises((FileNotFoundError, ValueError)):
        RUNNER.validate_failed_source(tmp_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("generation_concurrency", 3),
        ("request_timeout_s", 301),
        ("ordinals", [279, 97]),
        ("qids", list(reversed(RUNNER.RETRY_QIDS))),
        ("region_claim_regions", ["q2"]),
        ("no_auto_retry", False),
    ],
)
def test_receipt_rejects_broadened_authorization(
    tmp_path: Path, field: str, value: object
) -> None:
    receipt = _receipt()
    receipt["authorization"][field] = value
    with pytest.raises(ValueError, match="exact authorization"):
        RUNNER.validate_receipt(_write_receipt(tmp_path / "receipt.json", receipt))


def test_receipt_rejects_source_or_instrument_drift(tmp_path: Path) -> None:
    receipt = _receipt()
    receipt["source"]["tree_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="exact authorization"):
        RUNNER.validate_receipt(_write_receipt(tmp_path / "source.json", receipt))
    receipt = _receipt()
    receipt["instrument"]["runner"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="exact authorization"):
        RUNNER.validate_receipt(_write_receipt(tmp_path / "runner.json", receipt))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("repository", "/tmp/unrelated-orchestrator"),
        ("commit", "0" * 40),
        ("tree", "0" * 40),
    ],
)
def test_receipt_rejects_wrong_repository_commit_or_tree(
    tmp_path: Path, field: str, value: str
) -> None:
    receipt = _receipt()
    receipt["instrument"][field] = value
    with pytest.raises(ValueError, match="exact authorization"):
        RUNNER.validate_receipt(_write_receipt(tmp_path / f"{field}.json", receipt))


def test_receipt_rejects_dirty_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity = {
        "runtime_top": str(RUNNER.ROOT),
        "canonical_top": str(RUNNER.CANONICAL_REPOSITORY),
        "same_repository": True,
        "commit": "a" * 40,
        "tree": "b" * 40,
        "clean": False,
    }
    monkeypatch.setattr(RUNNER, "_runtime_git_identity", lambda _repository: identity)
    with pytest.raises(ValueError, match="exact authorization"):
        RUNNER.validate_receipt(_write_receipt(tmp_path / "dirty.json"))


def test_receipt_accepts_exact_clean_linked_worktree(tmp_path: Path) -> None:
    receipt = RUNNER.validate_receipt(_write_receipt(tmp_path / "linked.json"))
    assert receipt["instrument"]["commit"] == "a" * 40
    assert receipt["instrument"]["tree"] == "b" * 40


def test_receipt_rejects_unrelated_repo_even_with_same_files_and_pins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        RUNNER,
        "_runtime_git_identity",
        lambda _repository: {
            "runtime_top": "/tmp/copied-repository",
            "canonical_top": str(RUNNER.CANONICAL_REPOSITORY),
            "same_repository": False,
            "commit": "a" * 40,
            "tree": "b" * 40,
            "clean": True,
        },
    )
    with pytest.raises(ValueError, match="exact authorization"):
        RUNNER.validate_receipt(_write_receipt(tmp_path / "unrelated.json"))


def test_receipt_rejects_copy_outside_canonical_namespace(tmp_path: Path) -> None:
    canonical = _write_receipt(tmp_path / "canonical.json")
    copied = tmp_path / "copied.json"
    copied.write_bytes(canonical.read_bytes())
    RUNNER.CANONICAL_RECEIPT = canonical
    with pytest.raises(ValueError, match="missing or unsafe"):
        RUNNER.validate_receipt(copied)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("protocol_id", "wrong"),
        ("ratified_at", "not-a-timestamp"),
    ],
)
def test_receipt_rejects_protocol_or_timestamp_mutation(
    tmp_path: Path, field: str, value: str
) -> None:
    receipt = _receipt()
    receipt[field] = value
    with pytest.raises(ValueError, match="exact authorization"):
        RUNNER.validate_receipt(_write_receipt(tmp_path / f"{field}.json", receipt))


def test_receipt_rejects_extra_top_level_or_instrument_keys(tmp_path: Path) -> None:
    receipt = _receipt()
    receipt["ignored"] = True
    with pytest.raises(ValueError, match="exact authorization"):
        RUNNER.validate_receipt(_write_receipt(tmp_path / "extra-top.json", receipt))
    receipt = _receipt()
    receipt["instrument"]["ignored"] = True
    with pytest.raises(ValueError, match="exact authorization"):
        RUNNER.validate_receipt(_write_receipt(tmp_path / "extra-instrument.json", receipt))


def _questions() -> list[dict]:
    return [{"qid": f"q{ordinal}"} for ordinal in range(500)]


def _clean(ordinal: int) -> tuple[dict, dict]:
    qid = RUNNER.RETRY_QIDS[RUNNER.RETRY_ORDINALS.index(ordinal)]
    response = {"qid": qid, "answer": "ok", "error": None}
    sidecar = {
        "row_type": "question_result",
        "ordinal": ordinal,
        "elapsed_s": 1.0,
        "result": {
            "qid": qid,
            "question_id": qid,
            "correct": False,
            "tokens_generated": 1,
            "route": "frontdoor",
        },
    }
    return response, sidecar


def _timeout(ordinal: int) -> tuple[dict, dict]:
    qid = RUNNER.RETRY_QIDS[RUNNER.RETRY_ORDINALS.index(ordinal)]
    response = {"qid": qid, "answer": "", "error": "timed out"}
    sidecar = {
        "row_type": "question_result",
        "ordinal": ordinal,
        "elapsed_s": 300.1,
        "result": {
            "qid": qid,
            "question_id": qid,
            "correct": False,
            "error": True,
            "error_detail": "timed out",
            "tokens_generated": 0,
            "latency_ms": 300100,
        },
    }
    return response, sidecar


def _run_schedule(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    outcomes: dict[int, tuple[dict, dict]],
) -> tuple[list[int], list[dict], dict[int, dict], dict | None]:
    calls: list[int] = []

    def generate(**kwargs):
        ordinal = kwargs["ordinal"]
        calls.append(ordinal)
        return copy.deepcopy(outcomes[ordinal])

    monkeypatch.setattr(RUNNER, "_generate_one", generate)
    monkeypatch.setattr(
        RUNNER.V5,
        "validate_clean_sidecar_result",
        lambda response, *_args, **_kwargs: response.get("error") is None,
    )
    questions = _questions()
    questions[97]["qid"], questions[279]["qid"] = RUNNER.RETRY_QIDS
    rows: dict[int, dict] = {}
    attempts, sidecars, failure = RUNNER._collect_schedule(
        output=tmp_path,
        watcher=object(),
        runner_args=SimpleNamespace(),
        questions=questions,
        original_sidecars={97: {}, 279: {}},
        journal_path=tmp_path / "journal.jsonl",
        rows=rows,
    )
    return calls, attempts, rows, failure


def test_schedule_runs_exactly_two_clean_rows_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls, attempts, rows, failure = _run_schedule(
        tmp_path,
        monkeypatch,
        {97: _clean(97), 279: _clean(279)},
    )
    assert calls == [97, 279]
    assert [row["ordinal"] for row in attempts] == [97, 279]
    assert set(rows) == {97, 279}
    assert failure is None


def test_repeated_timeout_stops_schedule_without_second_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls, attempts, rows, failure = _run_schedule(
        tmp_path,
        monkeypatch,
        {97: _timeout(97), 279: _clean(279)},
    )
    assert calls == [97]
    assert [row["outcome"] for row in attempts] == ["terminal_failure"]
    assert rows == {}
    assert failure["ordinal"] == 97


def test_non_timeout_failure_is_instrument_error_not_terminal_disposition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    response, sidecar = _timeout(97)
    sidecar["result"]["error_detail"] = "connection reset"
    with pytest.raises(RuntimeError, match="outside the ratified timeout"):
        _run_schedule(
            tmp_path,
            monkeypatch,
            {97: (response, sidecar), 279: _clean(279)},
        )
    assert not (tmp_path / RUNNER.TERMINAL_NAME).exists()


def test_cli_exposes_no_timeout_or_concurrency_override() -> None:
    with pytest.raises(SystemExit):
        RUNNER.parse_args(
            [
                "--amendment-receipt",
                "/tmp/receipt",
                "--evaltower-timeout-s",
                "301",
            ]
        )


def _focused_sidecar_rows(*, label: str, qid: str) -> list[dict]:
    return [
        {
            "row_type": "batch_start",
            "label": label,
            "requested_n": 1,
            "concurrency": 1,
            "complete": False,
        },
        {
            "row_type": "question_result",
            "label": label,
            "requested_n": 1,
            "ordinal": 0,
            "result": {"qid": qid},
        },
        {
            "row_type": "batch_complete",
            "label": label,
            "requested_n": 1,
            "completed_n": 1,
            "complete": True,
        },
    ]


def test_focused_sidecar_is_discovered_by_validated_content(tmp_path: Path) -> None:
    label = "e8-final-c1-t2-r2-o97"
    qid = RUNNER.RETRY_QIDS[0]
    path = tmp_path / "question_results.unexpected-upstream-name.jsonl"
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in _focused_sidecar_rows(label=label, qid=qid)),
        encoding="utf-8",
    )
    selected, row = RUNNER._discover_focused_sidecar(tmp_path, label=label, qid=qid)
    assert selected == path
    assert row["result"]["qid"] == qid


def test_focused_sidecar_rejects_ambiguous_or_wrong_contract(tmp_path: Path) -> None:
    label = "e8-final-c1-t2-r2-o97"
    qid = RUNNER.RETRY_QIDS[0]
    wrong = _focused_sidecar_rows(label=label, qid=qid)
    wrong[0]["concurrency"] = 3
    (tmp_path / "question_results.wrong.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in wrong), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="exactly one"):
        RUNNER._discover_focused_sidecar(tmp_path, label=label, qid=qid)

    rows = _focused_sidecar_rows(label=label, qid=qid)
    for name in ("first", "second"):
        (tmp_path / f"question_results.{name}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
    with pytest.raises(ValueError, match="exactly one"):
        RUNNER._discover_focused_sidecar(tmp_path, label=label, qid=qid)
