"""AEI-TRUNC-NOTICE: the llm_call output-cap markers say how to get the rest."""

import re
from pathlib import Path

import pytest

from src.features import reset_features
from src.llm_primitives import primitives as prim


@pytest.fixture
def flag(monkeypatch):
    def _set(on):
        monkeypatch.setenv("ORCHESTRATOR_TRUNCATION_NARROWING_HINT", "1" if on else "0")
        reset_features()

    yield _set
    monkeypatch.undo()
    reset_features()


def test_flag_off_marker_is_byte_identical(flag):
    flag(False)
    assert prim._truncation_notice(1234) == "\n[... truncated at 1234 chars]"


def test_flag_on_marker_carries_narrowing_hint(flag):
    flag(True)
    out = prim._truncation_notice(1234)
    assert out.startswith("\n[... truncated at 1234 chars;")
    assert "narrower" in out and "peek/grep" in out


def test_every_marker_goes_through_the_helper():
    src = Path(prim.__file__).read_text()
    # The only literal marker is the helper's own line; call sites use the helper.
    assert len(re.findall(r"_truncation_notice\(self\.config\.output_cap\)", src)) == 3
    assert src.count("truncated at") == 1
    line = next(ln for ln in src.splitlines() if "truncated at" in ln)
    assert "{hint}" in line
