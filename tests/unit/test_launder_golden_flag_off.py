"""AEI-LAUNDER-FIX-1 golden: with delegation_untrusted_data_field OFF every boundary-laundering
site renders byte-identically to the commit before the flag existed (ceb93ff3).

The fixture was rendered by ``launder_golden_render.render_all`` run against ceb93ff3 itself.
Do not regenerate it from current code: that would erase the proof.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.features import reset_features

from . import launder_golden_render as r

_FIXTURE = Path(__file__).parent / "fixtures" / "launder_golden_parent_ceb93ff3.json"


@pytest.fixture(autouse=True)
def _flag_off(monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_DELEGATION_UNTRUSTED_DATA_FIELD", "0")
    reset_features()
    yield
    reset_features()


def test_flag_off_prompts_are_byte_identical_to_parent():
    golden = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    now = r.render_all()
    assert set(now) == set(golden)
    diff = [k for k in golden if now[k] != golden[k]]
    assert not diff, f"flag-off prompt changed vs parent for: {diff}"
