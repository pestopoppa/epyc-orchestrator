"""KPF-22: bounded LPM queue ordering and reservation safety, without inference."""

import pytest

from src.inference import prefix_index as pi
from src.scheduling.kv_pool_admission import SharedKVPoolAdmission

URL = "http://localhost:8083"


@pytest.fixture
def gate(monkeypatch):
    monkeypatch.setenv(pi.FLAG_ENV, "1")
    monkeypatch.setenv(pi.LPM_MIN_TOKENS_ENV, "100")
    monkeypatch.setenv(pi.LPM_MAX_SKIPS_ENV, "1")
    pool = SharedKVPoolAdmission(cross_process=False)
    pool._waiting[1] = (900, False)
    pool._lpm_score[1] = 100
    pool._inflight[URL] = {90: 200}
    return pool


def _decision(pool, ticket, score, *, want=50, observed=None):
    pool._lpm_score[ticket] = score
    return pool._admissible(URL, ticket, [1, ticket], want, False, 1000, observed)


def test_ties_and_below_threshold_do_not_spend_fairness_budget(gate):
    assert _decision(gate, 2, 100) == "wait"
    assert _decision(gate, 3, 99) == "wait"
    assert gate._lpm_skips.get(1, 0) == 0


def test_unfittable_prefix_cannot_bypass_or_consume_skip(gate):
    assert _decision(gate, 2, 200, want=801) == "wait"
    assert gate._lpm_skips.get(1, 0) == 0
    assert _decision(gate, 2, 200, want=800) == "lpm"
    assert gate._lpm_skips[1] == 1


def test_one_bypass_then_head_wins_when_reservation_is_released(gate):
    assert _decision(gate, 2, 200) == "lpm"
    assert _decision(gate, 3, 300) == "wait"
    assert gate._admissible(URL, 1, [1, 3], 900, False, 1000, None) == "wait"
    gate._inflight[URL].pop(90)
    assert gate._admissible(URL, 1, [1, 3], 900, False, 1000, None) == "head"


def test_native_occupied_cells_and_outstanding_reservations_both_count(gate):
    # Native used=700, outstanding=200, capacity=1000. A request of 101
    # must not pass even though it fits if either side is ignored.
    observed = (700, 1, 0, None, None, 1000)
    assert _decision(gate, 2, 200, want=101, observed=observed) == "wait"
    assert gate._lpm_skips.get(1, 0) == 0
    assert _decision(gate, 2, 200, want=100, observed=observed) == "lpm"


def test_prefix_does_not_bypass_a_head_that_already_fits(gate):
    # Existing 200 + head 800 fits exactly: FCFS must win regardless of
    # the later request's better prefix score.
    gate._waiting[1] = (800, False)
    assert _decision(gate, 2, 200) == "wait"
    assert gate._lpm_skips.get(1, 0) == 0


def test_any_fitting_or_unknown_earlier_waiter_prevents_bypass(gate):
    gate._lpm_score.update({2: 150, 3: 300})
    gate._waiting[2] = (700, False)  # Fits even though head 1 does not.
    assert gate._admissible(URL, 3, [1, 2, 3], 50, False, 1000, None) == "wait"
    assert not gate._lpm_skips
    gate._waiting.pop(2)
    assert gate._admissible(URL, 3, [1, 2, 3], 50, False, 1000, None) == "wait"
    assert not gate._lpm_skips


def test_skip_budget_applies_independently_to_all_bypassed_waiters(gate):
    gate._waiting[2] = (850, False)
    gate._lpm_score.update({2: 150, 3: 300, 4: 400})
    assert gate._admissible(URL, 3, [1, 2, 3], 50, False, 1000, None) == "lpm"
    assert gate._lpm_skips == {1: 1, 2: 1}
    assert gate._admissible(URL, 4, [1, 2, 4], 50, False, 1000, None) == "wait"
