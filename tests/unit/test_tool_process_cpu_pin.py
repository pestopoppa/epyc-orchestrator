"""tool_process_cpu_pin: flag on pins children, flag off is unchanged, overlap check fires."""

import logging
import os
import subprocess
import sys

import pytest

from src.repl_environment import tool_pin

pytestmark = pytest.mark.skipif(not hasattr(os, "sched_setaffinity"), reason="needs sched_setaffinity")


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.setattr(tool_pin, "_warned", False)
    monkeypatch.delenv("ORCHESTRATOR_TOOLS_CPU_SET", raising=False)


def _flag(monkeypatch, on):
    monkeypatch.setattr(tool_pin, "tool_pin_enabled", lambda: on)


def test_flag_off_kwargs_empty(monkeypatch):
    _flag(monkeypatch, False)
    assert tool_pin.popen_kwargs() == {}
    assert tool_pin.preexec() is None


def test_default_cpu_set_is_84_87(monkeypatch):
    assert tool_pin.tool_cpu_set_spec() == "84-87"
    assert tool_pin.parse_cpus("84-87") == {84, 85, 86, 87}


def test_flag_on_real_child_is_pinned(monkeypatch):
    allowed = sorted(os.sched_getaffinity(0))
    target = allowed[0]
    monkeypatch.setenv("ORCHESTRATOR_TOOLS_CPU_SET", str(target))
    _flag(monkeypatch, True)
    kw = tool_pin.popen_kwargs()
    assert "preexec_fn" in kw
    out = subprocess.run(
        [sys.executable, "-c", "import os;print(sorted(os.sched_getaffinity(0)))"],
        capture_output=True, text=True, **kw,
    )
    assert out.stdout.strip() == f"[{target}]"


def test_flag_off_child_affinity_unchanged(monkeypatch):
    _flag(monkeypatch, False)
    out = subprocess.run(
        [sys.executable, "-c", "import os;print(sorted(os.sched_getaffinity(0)))"],
        capture_output=True, text=True, **tool_pin.popen_kwargs(),
    )
    assert out.stdout.strip() == str(sorted(os.sched_getaffinity(0)))


def test_preexec_sets_requested_mask(monkeypatch):
    calls = []
    monkeypatch.setattr(os, "sched_setaffinity", lambda pid, cpus: calls.append((pid, set(cpus))))
    _flag(monkeypatch, True)
    tool_pin.preexec()()
    assert calls == [(0, {84, 85, 86, 87})]


CFG = {
    "frontdoor": {"instances": [("0-47", 8070, 48)]},
    "worker": {"instances": [("0-23", 8080, 24), ("48-71", 8081, 24)]},
}


def test_overlap_fires(monkeypatch, caplog):
    _flag(monkeypatch, True)
    monkeypatch.setenv("ORCHESTRATOR_TOOLS_CPU_SET", "20-25,84-87")
    with caplog.at_level(logging.WARNING):
        ov = tool_pin.check_tool_cpu_overlap(CFG)
    assert set(ov) == {"frontdoor[0]", "worker[0]"}
    assert "overlaps serving role" in caplog.text
    with pytest.raises(RuntimeError):
        tool_pin.check_tool_cpu_overlap(CFG, strict=True)


def test_disjoint_and_flag_off_silent(monkeypatch):
    _flag(monkeypatch, True)
    assert tool_pin.check_tool_cpu_overlap(CFG) == {}
    monkeypatch.setenv("ORCHESTRATOR_TOOLS_CPU_SET", "20")
    _flag(monkeypatch, False)
    assert tool_pin.check_tool_cpu_overlap(CFG, strict=True) == {}


def test_real_topology_default_overlaps_full_shape(monkeypatch):
    # 84-87 sits inside NUMA_FULL (0-95): the check must surface that.
    _flag(monkeypatch, True)
    ov = tool_pin.check_tool_cpu_overlap()
    assert ov and all(set(v) <= {84, 85, 86, 87} for v in ov.values())


def test_feature_spec_default_off():
    from src.features import Features

    assert Features().tool_process_cpu_pin is False
