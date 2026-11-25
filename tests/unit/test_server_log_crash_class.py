"""HF-CRASH-CLASS: server_log_crash_class (default off)."""
from src.autopilot_core.measurement_guards import (
    DISPOSITION_INFRA_FAILED,
    DISPOSITION_TASK_FAILED,
    infra_failure_reason,
    measurement_disposition,
)
from src.autopilot_core.server_log_crash import server_log_crash
from src.features import Features, get_features

DROP = {"failure_reason": "connect_error"}


def _log(tmp_path, text, port=8080):
    (tmp_path / f"llama-server-{port}.log").write_text(text)
    return lambda: server_log_crash([port], log_dir=tmp_path)


def test_assert_in_log_is_candidate_crash(tmp_path):
    scan = _log(tmp_path, "srv ok\nGGML_ASSERT(x) failed\nAborted (core dumped)\n")
    assert infra_failure_reason(DROP, server_log_scan=scan) == "candidate_crash"
    assert measurement_disposition(DROP, server_log_scan=scan) == DISPOSITION_TASK_FAILED


def test_segfault_variants(tmp_path):
    for line in ("Segmentation fault", "killed by signal 11", "SIGSEGV received"):
        assert infra_failure_reason(DROP, server_log_scan=_log(tmp_path, line + "\n")) == "candidate_crash"


def test_clean_log_stays_infra(tmp_path):
    scan = _log(tmp_path, "srv ok\nslot released\n")
    assert infra_failure_reason(DROP, server_log_scan=scan) == "connect_error"
    assert measurement_disposition(DROP, server_log_scan=scan) == DISPOSITION_INFRA_FAILED


def test_missing_log_stays_infra(tmp_path):
    scan = lambda: server_log_crash([9], log_dir=tmp_path)  # noqa: E731
    assert infra_failure_reason(DROP, server_log_scan=scan) == "connect_error"


def test_non_drop_reason_not_reclassified(tmp_path):
    scan = _log(tmp_path, "GGML_ASSERT\n")
    assert infra_failure_reason({"failure_reason": "read_timeout"}, server_log_scan=scan) == "read_timeout"


def test_flag_off_unchanged(tmp_path):
    _log(tmp_path, "GGML_ASSERT\n")
    assert infra_failure_reason(DROP) == "connect_error"
    assert measurement_disposition(DROP) == DISPOSITION_INFRA_FAILED
    assert Features().server_log_crash_class is False


def test_flag_env_on(monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_SERVER_LOG_CRASH_CLASS", "1")
    assert get_features().server_log_crash_class is True
