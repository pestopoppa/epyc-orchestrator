"""KPF-LB1: bucket + cell arithmetic of the cache-by-turn tabulation."""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "kpflb1", Path(__file__).resolve().parents[2] / "scripts/analysis/kpf_lb1_cache_by_turn.py")
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)


def test_buckets():
    assert [m.bucket(t) for t in (1, 2, 4, 5, 6, 20, 21, 500)] == ["1", "2", "4-5", "4-5", "6-10", "11-20", "21+", "21+"]


def test_cell_token_weighted_and_cold_share():
    c = m.cell([(1000, 0), (100, 900)])
    assert c["n"] == 2 and c["token_weighted_hit"] == round(900 / 2000, 4) and c["share_cache_below_10pct"] == 0.5
