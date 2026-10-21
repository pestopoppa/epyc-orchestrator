"""TOC-SP-9 fixtures for the draft repeat-notice guard (not wired)."""
from src.graph.repeat_notice import RepeatNoticeGuard, repeat_notice_enabled


def test_identical_call_different_result_no_notice():
    g = RepeatNoticeGuard(quiet_through_turn=0)
    assert g.observe(1, "run()", "out A").notice is None
    assert g.observe(2, "run()", "out B").notice is None


def test_identical_result_different_file_revision_no_notice():
    g = RepeatNoticeGuard(quiet_through_turn=0)
    g.observe(1, "peek()", "x", file_rev="r1", file_range="0-9")
    assert g.observe(2, "peek()", "x", file_rev="r2", file_range="0-9").notice is None
    assert g.observe(3, "peek()", "x", file_rev="r2", file_range="10-19").notice is None


def test_exit_status_is_part_of_result():
    g = RepeatNoticeGuard(quiet_through_turn=0)
    g.observe(1, "t()", "same", exit_code=0)
    assert g.observe(2, "t()", "same", exit_code=1).notice is None


def test_quiet_period_then_single_factual_notice_per_key():
    g = RepeatNoticeGuard()
    g.observe(1, "a", "r")
    assert g.observe(5, "a", "r").notice is None  # inside quiet period
    n = g.observe(11, "a", "r").notice
    assert n and n.startswith("Repeat notice") and "does not establish" in n
    assert g.observe(12, "a", "r").notice is None  # one per key


def test_terminate_only_after_identical_failing_streak():
    g = RepeatNoticeGuard()
    ds = [g.observe(t, "a", "err", failed=True) for t in range(1, 9)]
    assert [d.reminder for d in ds].count(True) == 1 and ds[4].reminder
    assert [d.terminate for d in ds] == [False] * 7 + [True]
    g2 = RepeatNoticeGuard()
    for t in range(1, 8):
        g2.observe(t, "a", "err", failed=True)
    assert not g2.observe(8, "a", "different err", failed=True).terminate


def test_flag_default_off(monkeypatch):
    monkeypatch.delenv("ORCHESTRATOR_REPL_REPEAT_NOTICE", raising=False)
    assert repeat_notice_enabled() is False
    monkeypatch.setenv("ORCHESTRATOR_REPL_REPEAT_NOTICE", "1")
    assert repeat_notice_enabled() is True
