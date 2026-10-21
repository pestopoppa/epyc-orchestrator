"""RTE-PFX-2: probe analysis + window guard (no server)."""
import importlib.util
import sys
from pathlib import Path

_P = Path(__file__).resolve().parents[2] / "scripts" / "benchmark" / "rte_pfx2_stagger_probe.py"
_S = importlib.util.spec_from_file_location("rte_pfx2_test", _P)
M = importlib.util.module_from_spec(_S); sys.modules["rte_pfx2_test"] = M; _S.loader.exec_module(M)


def test_slot_selection_check_counts_only_cached_siblings():
    rows = [{"idx": 0, "prompt_n": 1000, "cache_n": 0, "ttft_ms": 900.0, "id_slot": 0},
            {"idx": 1, "prompt_n": 20, "cache_n": 980, "ttft_ms": 80.0, "id_slot": 0},
            {"idx": 2, "prompt_n": 1000, "cache_n": 0, "ttft_ms": 900.0, "id_slot": 2}]
    s = M.summarise(rows, expected_prefix_tokens=1000)
    assert s["siblings_on_cached_prefix"] == 1 and s["n_siblings"] == 2
    assert s["total_uncached_prompt_tokens"] == 2020


def test_refuses_without_window(monkeypatch, tmp_path):
    monkeypatch.delenv("WINDOW_CONFIRMED", raising=False)
    assert M.main(["--server", "http://x", "--prefix-file", "f", "--output", str(tmp_path / "o")]) == 3
