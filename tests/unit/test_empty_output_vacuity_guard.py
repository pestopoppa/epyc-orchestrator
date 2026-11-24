"""empty_output_vacuity_guard (default OFF): vacuous outputs are failures. Each of a-d tested ON and OFF."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.features import _REGISTRY_BY_NAME, Features, reset_features


def _flag(monkeypatch, on: bool) -> None:
    flagset = Features(empty_output_vacuity_guard=on)
    monkeypatch.setattr("src.features.features", lambda: flagset)


def test_flag_registered_default_off():
    spec = _REGISTRY_BY_NAME["empty_output_vacuity_guard"]
    assert spec.default_test is False and spec.default_prod is False
    assert Features().empty_output_vacuity_guard is False
    reset_features()


# (a) empty generation ---------------------------------------------------------
@pytest.mark.parametrize("output", ["", "   \n\t"])
def test_a_empty_fails_fast_when_on(monkeypatch, output):
    from src.backends.llama_server import _is_empty_long_generation

    _flag(monkeypatch, True)
    assert _is_empty_long_generation(output, 0.5) is True
    assert _is_empty_long_generation(output, 0.0) is True
    assert _is_empty_long_generation("ok", 100.0) is False


def test_a_off_keeps_30s_threshold(monkeypatch):
    from src.backends.llama_server import _is_empty_long_generation

    _flag(monkeypatch, False)
    monkeypatch.delenv("LLAMA_EMPTY_GENERATION_FAILURE_AFTER_S", raising=False)
    assert _is_empty_long_generation("", 0.5) is False
    assert _is_empty_long_generation("", 30.0) is True
    assert _is_empty_long_generation("ok", 100.0) is False


# (b) batch-edit verify ---------------------------------------------------------
def test_b_no_verifiable_file(monkeypatch, tmp_path):
    from src.graph.helpers import _batch_edit_verify_fn

    (tmp_path / "notes.txt").write_text("x")
    monkeypatch.delenv("ORCHESTRATOR_BATCH_EDIT_VERIFY_CMD", raising=False)
    _flag(monkeypatch, True)
    assert _batch_edit_verify_fn(tmp_path) is False
    _flag(monkeypatch, False)
    assert _batch_edit_verify_fn(tmp_path) is True


def test_b_executed_check_still_passes_when_on(monkeypatch, tmp_path):
    from src.graph.helpers import _batch_edit_verify_fn

    _flag(monkeypatch, True)
    (tmp_path / "a.py").write_text("x = 1\n")
    assert _batch_edit_verify_fn(tmp_path, verify_cmd="") is True
    (tmp_path / "b.txt").write_text("x")
    other = tmp_path / "sub"
    other.mkdir()
    assert _batch_edit_verify_fn(other, verify_cmd="true") is True


# (c) FINAL("") -----------------------------------------------------------------
def _repl():
    from src.repl_environment import REPLConfig, REPLEnvironment

    return REPLEnvironment(context="test", config=REPLConfig(timeout_seconds=5))


@pytest.mark.parametrize("answer", ["", "  \n"])
def test_c_empty_final_refused_when_on(monkeypatch, answer):
    _flag(monkeypatch, True)
    repl = _repl()
    with pytest.raises(ValueError, match="empty answer"):
        repl._final(answer)
    assert repl.artifacts.get("_empty_final_refused") is True


def test_c_nonempty_final_still_signals_when_on(monkeypatch):
    from src.repl_environment.types import FinalSignal

    _flag(monkeypatch, True)
    with pytest.raises(FinalSignal):
        _repl()._final("42")


def test_c_off_empty_final_signals(monkeypatch):
    from src.repl_environment.types import FinalSignal

    _flag(monkeypatch, False)
    repl = _repl()
    with pytest.raises(FinalSignal):
        repl._final("")
    assert "_empty_final_refused" not in repl.artifacts


def test_c_executor_label_source_wired():
    src = Path("src/api/routes/chat_pipeline/repl_executor.py").read_text()
    assert "_empty_final_refused" in src and "empty_output_vacuity_guard" in src


# (d) path suffix ---------------------------------------------------------------
def _patchset(path: str, rename_to: str | None = None) -> str:
    fd = {"path": path, "operation": "create", "new_content": "x\n"}
    return "```patchset\n" + json.dumps({"files": [fd]}) + "\n```"


@pytest.mark.parametrize(
    "raw,clean",
    [("pkg/a.py#L12", "pkg/a.py"), ("pkg/a.py#L12-L30", "pkg/a.py"), ("pkg/a.py:45", "pkg/a.py"), ("pkg/a.py", "pkg/a.py")],
)
def test_d_suffix_stripped_when_on(monkeypatch, raw, clean):
    from src.batch_edit_parse import parse_patchset_from_model_output

    _flag(monkeypatch, True)
    ps = parse_patchset_from_model_output(_patchset(raw))
    assert ps.files[0].path == clean


def test_d_off_path_untouched(monkeypatch):
    from src.batch_edit_parse import parse_patchset_from_model_output

    _flag(monkeypatch, False)
    ps = parse_patchset_from_model_output(_patchset("pkg/a.py#L12"))
    assert ps.files[0].path == "pkg/a.py#L12"
