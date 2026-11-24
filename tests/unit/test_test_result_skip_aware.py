"""HF-SKIPS: flag ``test_result_skip_aware``.

An rc==0 pytest run where nothing passed (all skipped) decided nothing. ON: it is
inconclusive with cause ``skipped``. OFF: byte-identical legacy behaviour.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.features import reset_features  # noqa: E402
from src.gate_runner import GateConfig, GateRunner  # noqa: E402
from src.tools.pytest_summary import is_inconclusive, parse_pytest_summary  # noqa: E402

ALL_SKIPPED = (
    "ssss\n"
    "=========================== short test summary info ============================\n"
    "SKIPPED [3] tests/test_a.py:4: requires GPU\n"
    "SKIPPED [1] tests/test_b.py:9: no network\n"
    "4 skipped in 0.02s\n"
)
MIXED = "..s.x\n1 failed, 3 passed, 1 skipped, 1 xfailed in 1.20s\n".replace("1 failed, ", "")
REAL_PASS = "....\n4 passed in 0.10s\n"


@pytest.fixture(autouse=True)
def _flags(monkeypatch):
    monkeypatch.delenv("ORCHESTRATOR_TEST_RESULT_SKIP_AWARE", raising=False)
    reset_features()
    yield
    reset_features()


def _on(monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_TEST_RESULT_SKIP_AWARE", "1")
    reset_features()


def test_parser_counts_and_reasons():
    c = parse_pytest_summary(ALL_SKIPPED)
    assert (c["passed"], c["skipped"], c["failed"]) == (0, 4, 0)
    assert c["skip_reasons"] == [
        {"reason": "requires GPU", "count": 3},
        {"reason": "no network", "count": 1},
    ]
    m = parse_pytest_summary(MIXED)
    assert (m["passed"], m["skipped"], m["xfailed"]) == (3, 1, 1)
    assert parse_pytest_summary("random output") is None
    assert is_inconclusive(0, c) and not is_inconclusive(0, m) and not is_inconclusive(1, c)
    assert not is_inconclusive(0, None)


def _runner(tmp_path):
    cfg = tmp_path / "gates.yaml"
    cfg.write_text("gates:\n  - name: t\n    command: pytest\n    timeout: 5\n    required: true\n")
    return GateRunner(config_path=cfg, working_dir=tmp_path)


def _fake_run(stdout, rc=0):
    return lambda *a, **k: subprocess.CompletedProcess(a, rc, stdout=stdout, stderr="")


def test_gate_runner_off_is_legacy_passed(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "run", _fake_run(ALL_SKIPPED))
    r = _runner(tmp_path)
    res = r.run_gate(GateConfig(name="t", command="pytest", timeout=5))
    assert res.passed is True and res.verdict == "pass" and res.cause is None
    assert res.test_counts is None
    assert "skipped" not in res.to_dict()


def test_gate_runner_on_all_skipped_is_inconclusive_and_blocking(tmp_path, monkeypatch):
    _on(monkeypatch)
    monkeypatch.setattr(subprocess, "run", _fake_run(ALL_SKIPPED))
    res = _runner(tmp_path).run_gate(GateConfig(name="t", command="pytest", timeout=5))
    assert res.passed is False and res.verdict == "out-of-coverage" and res.cause == "skipped"
    d = res.to_dict()
    assert d["skipped"] == 4 and d["skip_reasons"][0]["reason"] == "requires GPU"


def test_gate_runner_on_real_pass_and_non_pytest_unchanged(tmp_path, monkeypatch):
    _on(monkeypatch)
    monkeypatch.setattr(subprocess, "run", _fake_run(REAL_PASS))
    res = _runner(tmp_path).run_gate(GateConfig(name="t", command="pytest", timeout=5))
    assert res.passed is True and res.cause is None and res.test_counts["passed"] == 4
    monkeypatch.setattr(subprocess, "run", _fake_run("all good\n"))
    res = _runner(tmp_path).run_gate(GateConfig(name="t", command="ruff", timeout=5))
    assert res.passed is True and res.test_counts is None


def _run_tests_tool(monkeypatch, stdout, big=False):
    from src.tools.code import run_tests as rt

    out = ("x" * 20000 + "\n" if big else "") + stdout
    monkeypatch.setattr(rt, "_validate_path", lambda p: (True, ""))
    monkeypatch.setattr(subprocess, "run", _fake_run(out))
    return rt.run_tests(test_path="tests/", working_dir=str(REPO_ROOT))


def test_run_tests_tool_off_and_on(monkeypatch):
    off = _run_tests_tool(monkeypatch, ALL_SKIPPED)
    assert off["success"] is True and "skipped" not in off and "cause" not in off
    _on(monkeypatch)
    on = _run_tests_tool(monkeypatch, ALL_SKIPPED)
    assert on["success"] is False and on["cause"] == "skipped" and on["skipped"] == 4
    assert on["skip_reasons"][0]["count"] == 3


def test_run_tests_tool_parses_before_truncation(monkeypatch):
    _on(monkeypatch)
    # Summary sits at the end of >16k output; the legacy scan runs on truncated text.
    out = _run_tests_tool(monkeypatch, REAL_PASS, big=True)
    assert out["passed"] == 4 and out["success"] is True


def _builtin_run_tests(monkeypatch, stdout):
    from src import builtin_tools

    handlers = {}

    class Reg:
        def register_handler(self, name, **kw):
            def deco(fn):
                handlers[name] = fn
                return fn
            return deco

    builtin_tools._register_code_tools(Reg())
    monkeypatch.setattr(subprocess, "run", _fake_run(stdout))
    return handlers["run_tests"]("tests/")


def test_builtin_run_tests_off_and_on(monkeypatch):
    big = "x" * 5000 + "\n" + ALL_SKIPPED
    off = _builtin_run_tests(monkeypatch, big)
    assert off["passed"] is True and "skipped" not in off
    _on(monkeypatch)
    on = _builtin_run_tests(monkeypatch, big)
    assert on["passed"] is False and on["cause"] == "skipped" and on["skipped"] == 4
    real = _builtin_run_tests(monkeypatch, "x" * 5000 + "\n" + REAL_PASS)
    assert real["passed"] is True and real["tests_passed"] == 4
