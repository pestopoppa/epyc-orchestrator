"""M-15: the COMP_r probe can resolve vectors through id_map instead of embedding_idx."""
import importlib.util
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "scripts" / "analysis" / "comp_region_probe.py"
spec = importlib.util.spec_from_file_location("comp_region_probe", SRC)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_resolves_through_id_map_and_counts_unmapped():
    id_map = ["a", "b", "c"]  # faiss row 0 -> memory a, ...
    rows = [("c", "frontdoor", "success", "{}"), ("a", "frontdoor", "failure", "{}"),
            ("zzz", "frontdoor", "success", "{}")]
    out, st = mod.resolve_rows_via_id_map(rows, id_map)
    assert [r[0] for r in out] == [2, 0]
    assert st == {"unmapped": 1, "mapped": 2}


def test_int_and_str_ids_match():
    out, st = mod.resolve_rows_via_id_map([(7, "x", "success", "{}")], [3, 7])
    assert out[0][0] == 1 and st["unmapped"] == 0
