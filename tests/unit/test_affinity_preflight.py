from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts/server/affinity_preflight.py"
spec = importlib.util.spec_from_file_location("affinity_preflight", MODULE_PATH)
assert spec is not None and spec.loader is not None
affinity_preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(affinity_preflight)


def test_expected_nodes_for_quarter_cpu_sets() -> None:
    assert affinity_preflight._expected_nodes(affinity_preflight._parse_cpulist("0-23,96-119")) == {0}
    assert affinity_preflight._expected_nodes(affinity_preflight._parse_cpulist("24-47,120-143")) == {1}
    assert affinity_preflight._expected_nodes(affinity_preflight._parse_cpulist("48-71,144-167")) == {2}
    assert affinity_preflight._expected_nodes(affinity_preflight._parse_cpulist("72-95,168-191")) == {3}


def test_expected_nodes_for_node_and_full_cpu_sets() -> None:
    assert affinity_preflight._expected_nodes(affinity_preflight._parse_cpulist("0-47,96-143")) == {0, 1}
    assert affinity_preflight._expected_nodes(affinity_preflight._parse_cpulist("48-95,144-191")) == {2, 3}
    assert affinity_preflight._expected_nodes(affinity_preflight._parse_cpulist("0-95")) == {0, 1, 2, 3}


def test_pages_by_node_from_line() -> None:
    line = "abc default file=/models/model.gguf mapped=30 N0=10 N1=20 kernelpagesize_kB=4"
    assert affinity_preflight._pages_by_node_from_line(line) == {0: 10, 1: 20}


def test_fmt_nodes() -> None:
    assert affinity_preflight._fmt_nodes({2, 0}) == "N0,N2"
    assert affinity_preflight._fmt_nodes(set()) == "(none)"


def test_summarize_mmap_gguf_pages_single_node_now_armed() -> None:
    summary = affinity_preflight._summarize_numa_maps(
        [
            "abc default file=/models/model.gguf mapped=100 N0=25 N1=25 N2=25 N3=25",
            "def default anon=10 dirty=10 N0=10",
        ],
        no_mmap=False,
        expected_nodes={0},
        threshold=0.85,
    )

    assert summary["required"] is True
    assert summary["match"] is False
    assert summary["arm"] == "locality"
    assert summary["signal_kind"] == "mmap_gguf_pages"
    assert summary["local_fraction"] == 0.25
    assert summary["model_files"] == ["/models/model.gguf"]


def test_summarize_no_mmap_single_node_requires_locality() -> None:
    summary = affinity_preflight._summarize_numa_maps(
        [
            "abc default anon=100 dirty=100 N0=25 N1=25 N2=25 N3=25",
            "def default anon=10 dirty=10 N0=10",
        ],
        no_mmap=True,
        expected_nodes={0},
        threshold=0.85,
    )

    assert summary["required"] is True
    assert summary["match"] is False
    assert summary["signal_kind"] == "anon_pages"
    assert summary["local_fraction"] == 35 / 110


def test_locality_threshold_is_ratified_095() -> None:
    assert affinity_preflight.LOCALITY_THRESHOLD == 0.95
    assert affinity_preflight.INTERLEAVE_TOLERANCE == 0.25


def _sm(lines, no_mmap, nodes, threshold=0.95):
    return affinity_preflight._summarize_numa_maps(
        lines, no_mmap=no_mmap, expected_nodes=nodes, threshold=threshold
    )


def test_single_node_pass_and_reject_both_mmap_modes() -> None:
    ok_mmap = _sm(["a file=/m.gguf N0=96 N1=4"], False, {0})
    assert ok_mmap["required"] and ok_mmap["match"] is True
    bad_mmap = _sm(["a file=/m.gguf N0=94 N1=6"], False, {0})
    assert bad_mmap["match"] is False
    ok_anon = _sm(["a anon=1 N0=100"], True, {0})
    assert ok_anon["match"] is True
    bad_anon = _sm(["a anon=1 N0=90 N2=10"], True, {0})
    assert bad_anon["match"] is False and bad_anon["note"] == "MEMORY LOCALITY MISMATCH"


def test_interleave_within_tolerance_passes() -> None:
    s = _sm(["a file=/m.gguf N0=26 N1=24 N2=26 N3=24"], False, {0, 1, 2, 3})
    assert s["arm"] == "interleave" and s["required"] is True
    assert s["match"] is True


def test_interleave_outside_tolerance_rejects() -> None:
    # share 0.32 > 1.25/4 = 0.3125
    s = _sm(["a file=/m.gguf N0=32 N1=22 N2=23 N3=23"], False, {0, 1, 2, 3})
    assert s["match"] is False and s["note"] == "MEMORY INTERLEAVE MISMATCH"


def test_interleave_two_nodes_bounds() -> None:
    assert _sm(["a anon=1 N0=60 N1=40"], True, {0, 1})["match"] is True  # 0.6<=0.625
    assert _sm(["a anon=1 N0=70 N1=30"], True, {0, 1})["match"] is False


def test_interleave_node_with_zero_pages_rejects() -> None:
    assert _sm(["a anon=1 N0=50 N1=50"], True, {0, 1, 2, 3})["match"] is False


def test_no_pages_fails_closed() -> None:
    assert _sm([], False, {0, 1})["match"] is False
    assert _sm([], False, {0})["match"] is False
