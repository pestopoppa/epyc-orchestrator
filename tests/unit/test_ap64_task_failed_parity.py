"""AP-64 (S3-AP-02): the OP-20 task_failed ruling holds in BOTH quality producers.

Ruling (S3-OP-01 option a, 2026-09-15; ETR-1 era 2026-09-23): a non-infra ``task_failed`` row
is WRONG and stays in the quality denominator; an infra-caused row is EXCLUDED. This pins that
one set of responses yields the same row classification and the same axis-0 quality from
``EvalTower`` (``_score_generation`` + ``_aggregate``) and from the seeding path
(``seeding_eval._build_role_result``), so a future edit to either producer cannot silently
re-open the "PRE-EXISTING: eval_tower and the seeding path DISAGREE" defect.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "scripts" / "autopilot"), str(REPO_ROOT / "scripts" / "benchmark")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import eval_tower  # noqa: E402
import seeding_eval  # noqa: E402
from src.autopilot_core.measurement_guards import (  # noqa: E402
    DISPOSITION_INFRA_FAILED,
    DISPOSITION_SCORED,
    DISPOSITION_TASK_FAILED,
    legacy_error_type,
)

# (label, resp, error, expected disposition, expected correct-for-quality)
CASES = [
    ("correct", {"answer": "4", "tokens_generated": 3}, None, DISPOSITION_SCORED, True),
    ("correct2", {"answer": "4", "tokens_generated": 3}, None, DISPOSITION_SCORED, True),
    ("wrong", {"answer": "5", "tokens_generated": 3}, None, DISPOSITION_SCORED, False),
    ("task_failed_declined", {"answer": "I refuse", "tokens_generated": 4},
     "model declined the task", DISPOSITION_TASK_FAILED, False),
    ("task_failed_agent_error", {"answer": "", "tokens_generated": 2},
     "agent_produced_error", DISPOSITION_TASK_FAILED, False),
    ("infra_http400",
     {"answer": "", "failure_reason": "http_status", "http_status": 400},
     "Client error '400 Bad Request' for url 'http://localhost:8000/chat'",
     DISPOSITION_INFRA_FAILED, None),
    ("infra_read_timeout",
     {"answer": "", "failure_reason": "read_timeout",
      "failure_provenance": {"class": "client_transport_timeout"}},
     "", DISPOSITION_INFRA_FAILED, None),
    ("infra_inband_banner",
     {"answer": "[ERROR: Backend unavailable (circuit open): http://x]"},
     None, DISPOSITION_INFRA_FAILED, None),
]


def _tower_row(resp, error, monkeypatch, expected="4"):
    """Drive the REAL generation + scoring phases of EvalTower with a stubbed transport."""
    full = dict(resp)
    if error is not None:
        full["error"] = error
    monkeypatch.setattr(eval_tower, "call_orchestrator_forced", lambda **kw: dict(full))
    tower = eval_tower.EvalTower()
    q = {"id": "q", "suite": "unit", "prompt": "2+2", "expected": expected,
         "scoring_method": "exact_match"}
    outcome = tower._generate_question(q, None)
    return tower, tower._score_generation(q, outcome, client=None)


def _seeding_row(resp, error, expected="4"):
    full = dict(resp)
    if error is not None:
        full["error"] = error
    return seeding_eval._build_role_result(
        role="worker_math", mode="direct", resp=full, elapsed=1.0, expected=expected,
        scoring_method="exact_match", scoring_config={}, allow_delegation=False,
    )


@pytest.mark.parametrize("label,resp,error,disposition,correct", CASES, ids=[c[0] for c in CASES])
def test_each_row_is_classified_identically_by_both_producers(
    label, resp, error, disposition, correct, monkeypatch
):
    _, trow = _tower_row(resp, error, monkeypatch)
    rr, error_type = _seeding_row(resp, error)
    assert trow.disposition == disposition
    assert error_type == legacy_error_type(trow.disposition)
    if disposition == DISPOSITION_INFRA_FAILED:
        # excluded in both: the tower keeps it out of the denominator, seeding marks it infra
        assert error_type == "infrastructure"
        assert eval_tower._quality_denominator([trow]) == []
    else:
        assert eval_tower._quality_denominator([trow]) == [trow]
        assert eval_tower._quality_correct(trow) is correct
        assert bool(rr.passed) is correct  # task_failed is WRONG in both


def test_one_run_yields_the_same_axis0_quality_from_either_producer(monkeypatch):
    tower = eval_tower.EvalTower()
    rows, seeding_passed = [], []
    for label, resp, error, *_ in CASES:
        _, trow = _tower_row(resp, error, monkeypatch)
        rows.append(trow)
        rr, error_type = _seeding_row(resp, error)
        if error_type != "infrastructure":
            seeding_passed.append(bool(rr.passed))
    agg = tower._aggregate(rows, tier=1)
    seeding_quality = sum(seeding_passed) / len(seeding_passed)  # 0-1
    # 2 correct of 5 in-denominator rows (correct x2, wrong, 2 task_failed); 3 infra excluded
    assert len(seeding_passed) == 5
    assert agg.details["quality_denominator"] == 5
    assert seeding_quality == pytest.approx(2 / 5)
    assert agg.quality / 3.0 == pytest.approx(seeding_quality)
    assert agg.details["task_failed"] == 2 and agg.details["infra_failed"] == 3
