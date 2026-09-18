"""Unit tests for the TD-7 routing replay (src/typed_decisions/routing_replay).

Every test runs against fake primitives and in-memory rows: no store, no
network, no real model. The native-path test injects the runner's tokenizer
seam and a synthetic v9 ``completion_probabilities`` capture, so the same
slicing code path the live run uses is exercised deterministically.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from src.llm_primitives.stat_tests import expected_calibration_error, wilson_interval
from src.typed_decisions.routing_replay import (
    LIVE_DB_PATH,
    ReplayError,
    RoutingRow,
    RowOutcome,
    aggregate,
    build_question,
    calibration_stats,
    load_options,
    main,
    prepare_state,
    require_live_primitives,
    resolve_snapshot,
    run_replay,
    run_row,
    sample_rows,
)
from src.typed_decisions.types import QuestionKind

ROLE = "frontdoor"


class _FakePrimitives:
    """Primitives-shaped stand-in; records every call, emits canned responses.

    ``mock_mode=False`` by default so a replay accepts it; ``_last_inference_meta``
    is settable by the responder for the native slicing path.
    """

    def __init__(self, responder, *, mock_mode: bool = False):
        self.responder = responder
        self.mock_mode = mock_mode
        self.calls: list[dict] = []
        self._last_inference_meta: dict = {}

    def llm_call(self, prompt: str, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        response = self.responder(prompt, **kwargs)
        if isinstance(response, dict):
            self._last_inference_meta = response.get("meta", {})
            return response.get("text", "")
        return response


def _parse_choice_candidates(prompt: str) -> tuple[str, list[str]]:
    """Extract the single question id and candidate list from a runner prompt."""
    match = re.search(r"^Q\d+ id=(\S+) kind=choice$", prompt, re.MULTILINE)
    assert match, "runner prompt carries no choice question"
    candidates_line = re.search(r"^  candidates: (.+)$", prompt, re.MULTILINE)
    assert candidates_line, "runner prompt carries no candidates line"
    return match.group(1), candidates_line.group(1).split(" | ")


def _first_candidate_responder(prompt: str, **kwargs) -> str:
    question_id, candidates = _parse_choice_candidates(prompt)
    return json.dumps(
        {
            "answers": {
                question_id: {
                    "choice": candidates[0],
                    "probabilities": {label: 1.0 / len(candidates) for label in candidates},
                    "confidence": 0.5,
                }
            }
        }
    )


def _row(index: int, action: str, outcome: str, context: str | None = None) -> RoutingRow:
    return RoutingRow(
        embedding_idx=index,
        action=action,
        outcome=outcome,
        context=context if context is not None else f"recorded context {index}",
    )


def _outcome(
    position: int,
    incumbent: str,
    action: str | None,
    label: bool,
    confidence: float | None,
    path: str,
) -> RowOutcome:
    return RowOutcome(
        position=position,
        incumbent=incumbent,
        label=label,
        state_sha256=f"state-{position}",
        state_chars=10,
        state_truncated=False,
        action=action,
        confidence=confidence,
        probabilities={action: confidence} if action is not None and confidence else None,
        path=path,
        native_failures=(),
        json_failures=(),
        prompt_sha256=f"prompt-{position}",
        elapsed_ms=1.0,
    )


# ── sampling determinism and question construction ────────────────────────


def test_sample_rows_is_deterministic_and_order_invariant():
    rows = [
        _row(
            index, ("frontdoor", "SELF", "WORKER")[index % 3], "success" if index % 2 else "failure"
        )
        for index in range(100)
    ]
    first = sample_rows(rows, 10, seed=7)
    assert first == sample_rows(rows, 10, seed=7)
    assert first == sample_rows(list(reversed(rows)), 10, seed=7)
    assert first != sample_rows(rows, 10, seed=8)
    assert len(first) == 10
    assert sample_rows(rows, 1000, seed=7) == sorted(rows, key=RoutingRow.canonical_key)


def test_sample_rows_rejects_negative_n():
    with pytest.raises(ValueError):
        sample_rows([_row(1, "frontdoor", "success")], -1, seed=0)


def test_load_options_filters_empty_and_sorts():
    rows = [
        _row(1, "frontdoor", "success"),
        _row(2, "", "success"),
        _row(3, "WORKER", "failure"),
        _row(4, "SELF", "success"),
        _row(5, "frontdoor", "failure"),
    ]
    options = load_options(rows)
    assert options == ("SELF", "WORKER", "frontdoor")
    with pytest.raises(ReplayError):
        load_options([_row(1, "", "success")])


def test_build_question_choice_over_all_options():
    options = ("SELF", "WORKER", "frontdoor")
    question = build_question(options)
    assert question.kind is QuestionKind.CHOICE
    assert question.options == options
    assert question.id
    assert question.criteria


def test_prepare_state_truncates_at_documented_budget():
    short, cut = prepare_state("abc", 10)
    assert (short, cut) == ("abc", False)
    long, cut = prepare_state("x" * 50, 10)
    assert cut is True
    assert long.startswith("x" * 10)
    assert "truncated at 10 chars" in long


# ── agreement and calibration math ────────────────────────────────────────


def test_aggregate_agreement_path_split_and_by_action():
    outcomes = [
        _outcome(0, "A", "A", True, 0.9, "native"),
        _outcome(1, "A", "B", False, 0.6, "json"),
        _outcome(2, "B", "B", True, 0.8, "json"),
        _outcome(3, "C", None, True, None, "unresolved"),
    ]
    stats = aggregate(outcomes)
    assert stats["n_rows"] == 4
    assert stats["n_decided"] == 3
    assert stats["n_unresolved"] == 1
    assert stats["agreement"]["n_agreeing"] == 2
    assert stats["agreement"]["n_disagreeing"] == 1
    assert stats["agreement"]["rate"] == pytest.approx(2 / 3)
    assert stats["agreement"]["wilson_95"] == list(wilson_interval(2, 3))
    assert stats["path_split"] == {"native": 1, "json": 2, "unresolved": 1}
    by_action = stats["agreement"]["by_incumbent_action"]
    assert by_action["A"] == {"n": 2, "agreeing": 1, "rate": 0.5}
    assert by_action["B"] == {"n": 1, "agreeing": 1, "rate": 1.0}


def test_calibration_stats_matches_hand_computed_ece_and_brier():
    pairs = [(0.25, 1.0), (0.25, 0.0), (1.0, 1.0), (1.0, 1.0)]
    stats = calibration_stats(pairs)
    assert stats["n"] == 4
    assert stats["ece"] == pytest.approx(0.125)
    assert stats["ece"] == pytest.approx(
        expected_calibration_error([p for p, _ in pairs], [y for _, y in pairs])
    )
    assert stats["brier"] == pytest.approx(0.15625)
    assert stats["base_rate"] == pytest.approx(0.75)
    assert len(stats["reliability_bins"]) == 10
    populated = [bin_ for bin_ in stats["reliability_bins"] if bin_["n"]]
    assert [bin_["n"] for bin_ in populated] == [2, 2]
    total = sum(bin_["n"] for bin_ in stats["reliability_bins"])
    assert total == 4
    recomputed = sum(
        (bin_["n"] / stats["n"]) * abs(bin_["accuracy"] - bin_["mean_confidence"])
        for bin_ in populated
    )
    assert recomputed == pytest.approx(stats["ece"])


def test_calibration_stats_empty_is_none_not_fabricated():
    stats = calibration_stats([])
    assert stats["n"] is None
    assert stats["ece"] is None
    assert stats["brier"] is None
    assert stats["reliability_bins"] == []


# ── fail-closed gates ─────────────────────────────────────────────────────


def test_require_live_primitives_refuses_mock_and_none():
    require_live_primitives(_FakePrimitives(_first_candidate_responder))
    with pytest.raises(ReplayError, match="MOCK"):
        require_live_primitives(_FakePrimitives(_first_candidate_responder, mock_mode=True))
    with pytest.raises(ReplayError):
        require_live_primitives(None)


def test_resolve_snapshot_refuses_live_database_without_reading_it():
    with pytest.raises(ReplayError, match="LIVE"):
        resolve_snapshot(LIVE_DB_PATH)


def test_main_refuses_without_live_or_dry_run(capsys):
    assert main(["--snapshot", "/nonexistent"]) == 2
    assert "refusing to run" in capsys.readouterr().err


# ── per-row replay paths ──────────────────────────────────────────────────


def test_run_row_uses_native_when_candidates_tokenize():
    def responder(prompt: str, **kwargs):
        return {
            "text": "A",
            "meta": {
                "completion_probabilities": [
                    {"id": 7, "token": "\nq1: ", "logprob": -0.01, "top_logprobs": []},
                    {
                        "id": 11,
                        "token": "A",
                        "logprob": -0.105,
                        "top_logprobs": [
                            {"id": 11, "token": "A", "logprob": -0.105},
                            {"id": 21, "token": "B", "logprob": -2.3},
                        ],
                    },
                ]
            },
        }

    def tokenize(text: str):
        return {"A": [11], " A": [12], "B": [21], " B": [22], "\nrouting_action: ": [7]}.get(text)

    primitives = _FakePrimitives(responder)
    question = build_question(("A", "B"))
    outcome = run_row(
        primitives,
        _row(1, "A", "success"),
        question,
        position=0,
        role=ROLE,
        tokenize_fn=tokenize,
    )
    assert outcome.path == "native"
    assert outcome.action == "A"
    assert outcome.confidence == pytest.approx(0.8, abs=1e-3)
    assert outcome.native_failures == ()
    assert len(primitives.calls) == 1
    assert "grammar" in primitives.calls[0]


def test_run_row_falls_back_to_json_when_native_excludes_question():
    primitives = _FakePrimitives(_first_candidate_responder)
    question = build_question(("frontdoor", "architect_general"))
    outcome = run_row(
        primitives,
        _row(1, "frontdoor", "failure", context="recorded failure"),
        question,
        position=0,
        role=ROLE,
    )
    assert outcome.path == "json"
    assert outcome.action == "frontdoor"
    assert [failure["reason"] for failure in outcome.native_failures] == [
        "native_tokenizer_unavailable"
    ]
    assert outcome.json_failures == ()
    assert len(primitives.calls) == 1
    assert "json_schema" in primitives.calls[0]


def test_run_row_reports_unresolved_when_both_paths_fail():
    primitives = _FakePrimitives(lambda prompt, **kwargs: "not json")
    question = build_question(("frontdoor", "architect_general"))
    outcome = run_row(
        primitives,
        _row(1, "frontdoor", "success"),
        question,
        position=0,
        role=ROLE,
        json_n_tokens=128,
    )
    assert outcome.path == "unresolved"
    assert outcome.action is None
    assert outcome.confidence is None
    assert outcome.native_failures and outcome.json_failures


# ── receipt shape and dry-run plan ────────────────────────────────────────


def _snapshot_stub() -> dict:
    return {
        "path": "/frozen/snapshot",
        "db_path": "/frozen/snapshot/episodic.db",
        "db_sha256": "a" * 64,
        "db_bytes": 123,
        "db_mtime_utc": "2026-04-15T09:17:02Z",
        "faiss_path": "/frozen/snapshot/embeddings.faiss",
        "faiss_sha256": "b" * 64,
        "max_created_at": "2026-04-15T09:17:02.851570+00:00",
        "max_updated_at": "2026-04-15T09:17:02.851570+00:00",
        "admissible": True,
        "admissibility_reason": "known_2026_04_15_frozen_snapshot",
    }


def test_run_replay_receipt_shape_and_json_fallback(tmp_path: Path):
    rows = [
        _row(0, "frontdoor", "success"),
        _row(1, "SELF", "failure"),
        _row(2, "frontdoor", "failure"),
        _row(3, "WORKER", "success"),
        _row(4, "", "success"),
    ]
    primitives = _FakePrimitives(_first_candidate_responder)
    receipt_path = tmp_path / "receipt.json"
    receipt = run_replay(
        rows,
        snapshot_provenance=_snapshot_stub(),
        primitives=primitives,
        n=4,
        seed=1,
        role=ROLE,
        receipt_path=receipt_path,
        timestamp="2026-09-18T00:00:00+00:00",
    )
    assert receipt["receipt"] == "td7-routing-replay"
    assert receipt["corpus"]["routing_rows"] == 5
    assert receipt["corpus"]["excluded_empty_action_rows"] == 1
    assert receipt["corpus"]["distinct_actions"] == 3
    assert receipt["label_provenance"]["status"] == "frozen-snapshot"
    assert receipt["label_provenance"]["snapshot_db_sha256"] == "a" * 64
    assert "2026-09-17 leak purge" in receipt["label_provenance"]["warning"]
    assert receipt["metric_directions"]["calibration.ece"] == "lower_is_better"
    assert receipt["config"]["mode"] == "native+json_fallback"
    assert receipt["config"]["cue_style"] == "id_only"
    assert len(receipt["rows"]) == 4
    for record in receipt["rows"]:
        assert record["path"] == "json"
        assert record["action"] in receipt["config"]["options"]
        assert record["incumbent"] in receipt["config"]["options"]
        assert isinstance(record["label"], bool)
        assert re.fullmatch(r"[0-9a-f]{64}", record["state_sha256"])
        assert re.fullmatch(r"[0-9a-f]{64}", record["prompt_sha256"])
        assert record["native_failures"]
    aggregates = receipt["aggregates"]
    assert aggregates["path_split"] == {"native": 0, "json": 4, "unresolved": 0}
    assert aggregates["calibration"]["n"] == 4
    assert set(aggregates["calibration_agreeing_only"]) >= {"ece", "brier"}
    assert receipt_path.exists()
    assert json.loads(receipt_path.read_text(encoding="utf-8")) == receipt


def test_run_replay_dry_run_prints_plan_and_writes_nothing(tmp_path: Path):
    rows = [
        _row(0, "frontdoor", "success"),
        _row(1, "SELF", "failure"),
        _row(2, "WORKER", "success"),
    ]
    receipt_path = tmp_path / "receipt.json"
    plan = run_replay(
        rows,
        snapshot_provenance=_snapshot_stub(),
        primitives=None,
        n=2,
        seed=3,
        dry_run=True,
        receipt_path=receipt_path,
    )
    assert plan["dry_run"] is True
    assert len(plan["sample"]) == 2
    assert plan["sample"][0]["state_sha256"]
    assert plan["config"]["options"] == ["SELF", "WORKER", "frontdoor"]
    assert plan["estimated_calls"] == {"native_attempts": 2, "json_fallbacks_max": 2}
    assert not receipt_path.exists()
