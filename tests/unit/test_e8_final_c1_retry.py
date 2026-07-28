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


def _receipt() -> dict:
    return {
        "schema": RUNNER.RECEIPT_SCHEMA,
        "status": "ratified",
        "human_attestation": RUNNER.ATTESTATION,
        "source": {
            "path": str(RUNNER.SOURCE),
            "tree_sha256": RUNNER.SOURCE_TREE_SHA256,
        },
        "instrument": {
            "repository": "/mnt/raid0/llm/epyc-orchestrator",
            "commit": "a" * 40,
            "tree": "b" * 40,
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
