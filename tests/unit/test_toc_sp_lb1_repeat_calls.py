"""TOC-SP-LB1: repeat detection and the follow-on classification."""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "tocsp", Path(__file__).resolve().parents[2] / "scripts/analysis/toc_sp_lb1_repeat_calls.py")
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)


def call(name, args, out):
    return ("call", name, args, out)


def test_repeat_equal_result_then_other_tool():
    c = m.analyse_sequence([call("read", "a", "X"), call("read", "a", "X"), call("grep", "b", "Y")])
    assert c["calls"] == 3 and c["repeats"] == 1 and c["result_equal"] == 1
    assert c["next_other_tool"] == 1 and c["back_to_back"] == 1


def test_loop_changed_result_and_text_end():
    c = m.analyse_sequence([call("bash", "ls", "1"), call("bash", "ls", "2"), call("bash", "ls", "3"), ("text",)])
    assert c["repeats"] == 2 and c["result_changed"] == 2 and c["next_same"] == 1 and c["next_text"] == 1


def test_text_between_and_different_args_are_distinguished():
    c = m.analyse_sequence([call("read", "a", None), ("text",), call("read", "a", None), call("read", "b", None)])
    assert c["repeats"] == 1 and c["text_between"] == 1 and c["result_unknown"] == 1


def test_codex_volatile_lines_are_normalised():
    a = m._h("Chunk ID: 1\nWall time: 0.1 seconds\nOutput:\nhello")
    b = m._h("Chunk ID: 2\nWall time: 9.0 seconds\nOutput:\nhello")
    assert a == b


def test_polling_tools_are_excluded_by_default_set():
    seq = [call("write_stdin", "a", "1"), call("write_stdin", "a", "1"), call("read", "x", "1")]
    assert m.analyse_sequence(seq)["repeats"] == 1
    assert m.analyse_sequence(seq, m.POLLING_TOOLS)["repeats"] == 0
