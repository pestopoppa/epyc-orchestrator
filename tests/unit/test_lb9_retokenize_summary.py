"""UFH14-LB9: the summary arithmetic of the re-tokenization comparison."""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "lb9", Path(__file__).resolve().parents[2] / "scripts/analysis/ufh14_lb9_retokenize_gap.py")
lb9 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lb9)


def test_modal_offset_absorbs_eos_and_excess_counts_noncanonical():
    rows = [(11, 10)] * 8 + [(12, 10)] * 2          # 8 canonical (+1 EOS), 2 with 1 extra token
    s = lb9.summarise(rows)
    assert s["modal_offset"] == 1 and s["rows_above_mode"] == 2 and s["share_above_mode"] == 0.2
    assert s["mean_excess_above_mode_tokens"] == 1.0 and s["rows_below_mode"] == 0


def test_empty():
    assert lb9.summarise([]) == {"n": 0}
