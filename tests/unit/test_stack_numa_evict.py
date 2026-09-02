"""Unit tests for scripts/server/stack_numa_evict.py (INF-70/C7).

The step is PREPARED, not enabled: the tests pin that `numa_pre_evict_gib`
absent or 0 makes every entry point a no-op, that a declared value is honoured
and capped, and that the placement fold reports the skew that motivated the
work. No subprocess is ever really run — `subprocess.run` is monkeypatched
throughout, mirroring tests/unit/test_stack_prewarm.py.
"""

from __future__ import annotations

import subprocess

import pytest

from scripts.server import stack_numa_evict as ev


# --- the config gate ---------------------------------------------------------

@pytest.mark.parametrize(
    "cfg,expected",
    [
        (None, 0),
        ({}, 0),
        ({"instances": []}, 0),
        ({"numa_pre_evict_gib": 0}, 0),
        ({"numa_pre_evict_gib": -5}, 0),
        ({"numa_pre_evict_gib": "nonsense"}, 0),
        ({"numa_pre_evict_gib": None}, 0),
        ({"numa_pre_evict_gib": 40}, 40),
        ({"numa_pre_evict_gib": "40"}, 40),
        ({"numa_pre_evict_gib": 10_000}, ev.MAX_PRE_EVICT_GIB),
    ],
)
def test_pre_evict_gib_for_role(cfg, expected):
    assert ev.pre_evict_gib_for_role(cfg) == expected


def test_no_live_role_declares_it_yet():
    """PREPARED, not enabled: shipping this must not change any role's launch."""
    from scripts.server.stack_numa import NUMA_CONFIG

    declared = {r: c.get("numa_pre_evict_gib") for r, c in NUMA_CONFIG.items()
                if isinstance(c, dict) and c.get("numa_pre_evict_gib")}
    assert declared == {}, f"roles opted in without a decision: {declared}"


def test_role_field_is_accepted_by_the_topology_allowlist():
    from scripts.server import stack_numa

    assert "numa_pre_evict_gib" in stack_numa._ROLE_FIELDS


# --- pre_evict_nodes ---------------------------------------------------------

def test_zero_target_is_a_noop(monkeypatch):
    def boom(*a, **k):  # pragma: no cover - must never run
        raise AssertionError("pre_evict_nodes(0) must not spawn anything")

    monkeypatch.setattr(ev.subprocess, "run", boom)
    ok, msg = ev.pre_evict_nodes(0)
    assert ok is True
    assert "disabled" in msg


def test_missing_numactl_is_reported_not_raised(monkeypatch):
    monkeypatch.setattr(ev.shutil, "which", lambda name: None)
    ok, msg = ev.pre_evict_nodes(40)
    assert ok is False
    assert "numactl" in msg


def test_single_node_host_is_a_noop(monkeypatch):
    monkeypatch.setattr(ev.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(ev, "_node_ids", lambda: [0])
    ok, msg = ev.pre_evict_nodes(40)
    assert ok is True
    assert "single-node" in msg


def _four_nodes(monkeypatch):
    monkeypatch.setattr(ev.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(ev, "_node_ids", lambda: [0, 1, 2, 3])


def test_uses_the_research_helper_when_present(monkeypatch):
    _four_nodes(monkeypatch)
    monkeypatch.setattr(ev.os.path, "isfile", lambda p: p == ev.RESEARCH_EVICT)
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, "  OK: every requested node is at target.\n", "")

    monkeypatch.setattr(ev.subprocess, "run", fake_run)
    ok, msg = ev.pre_evict_nodes(40)
    assert ok is True
    assert seen["cmd"][:2] == ["python3", ev.RESEARCH_EVICT]
    assert seen["cmd"][2:] == ["--target-gib", "40"]
    assert "numa_evict.py" in msg


def test_falls_back_when_the_research_helper_is_absent(monkeypatch):
    _four_nodes(monkeypatch)
    monkeypatch.setattr(ev.os.path, "isfile", lambda p: False)
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, "inline eviction done\n", "")

    monkeypatch.setattr(ev.subprocess, "run", fake_run)
    ok, msg = ev.pre_evict_nodes(40)
    assert ok is True
    assert seen["cmd"][0].endswith("numactl")
    assert "inline fallback" in msg


def test_nonzero_exit_degrades_but_does_not_raise(monkeypatch):
    _four_nodes(monkeypatch)
    monkeypatch.setattr(ev.os.path, "isfile", lambda p: True)
    monkeypatch.setattr(
        ev.subprocess, "run",
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, "node 2 short\n", ""),
    )
    ok, msg = ev.pre_evict_nodes(40)
    assert ok is False
    assert "exit 1" in msg


def test_timeout_is_caught(monkeypatch):
    _four_nodes(monkeypatch)
    monkeypatch.setattr(ev.os.path, "isfile", lambda p: True)

    def fake_run(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, kw.get("timeout", 1))

    monkeypatch.setattr(ev.subprocess, "run", fake_run)
    ok, msg = ev.pre_evict_nodes(40, timeout_s=5)
    assert ok is False
    assert "timed out" in msg


# --- the placement fold ------------------------------------------------------

# The 2026-09-02 failure, as numa_maps would render it: a 98 GB model at
# 57.7/10.7/8.0/17.7 GB across 4 nodes, THP-backed (kernelpagesize_kB=2048).
def _maps(gb: tuple[float, float, float, float], page_kb: int = 2048) -> str:
    pages = [int(g * 1024 * 1024 / page_kb) for g in gb]
    ns = " ".join(f"N{i}={p}" for i, p in enumerate(pages))
    return (
        "7f0000000000 interleave:0-3 file=/model.gguf mapped=1 N0=1 kernelpagesize_kB=4\n"
        f"7f1000000000 interleave:0-3 anon={sum(pages)} dirty={sum(pages)} "
        f"{ns} kernelpagesize_kB={page_kb}\n"
    )


MEASURED_SKEW = _maps((57.7, 10.7, 8.0, 17.7))
CLEAN = _maps((23.5, 23.6, 23.4, 23.6))


def test_summary_reports_the_measured_skew():
    out = ev.summarize_numa_maps(MEASURED_SKEW)
    assert "n0=59084MB" in out          # 57.7 GiB
    assert "max=node0@61.3%" in out
    assert "even=25%" in out


def test_summary_reports_clean_placement():
    out = ev.summarize_numa_maps(CLEAN)
    assert "max=node1@25.1%" in out or "max=node3@25.1%" in out


def test_huge_pages_are_not_understated():
    """A THP mapping counts 2 MiB per page; treating it as 4 KiB understates 512x."""
    thp = ev.summarize_numa_maps(_maps((10.0, 10.0, 10.0, 10.0), page_kb=2048))
    assert "total=40960MB" in thp


def test_summary_handles_garbage_and_empty():
    assert ev.summarize_numa_maps("nothing here") == "no NUMA-resident mappings"
    assert ev.summarize_numa_maps("") == "no NUMA-resident mappings"


def test_placement_summary_on_a_missing_pid_does_not_raise():
    out = ev.placement_summary(2 ** 22 + 7)
    assert "unreadable" in out or "no NUMA-resident" in out


def test_placement_summary_reads_this_process():
    """Against a real live pid — our own — it must produce a parsable fold."""
    import os as _os

    out = ev.placement_summary(_os.getpid())
    assert "total=" in out and "max=node" in out
