"""REPL-EMB-3.1: extractive spill summary (flag spill_extractive_summary, default off)."""
from __future__ import annotations

import uuid
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.features import _REGISTRY_BY_NAME, reset_features
from src.repl_environment import REPLConfig, REPLEnvironment
from src.repl_environment.spill_extract import extractive_summary, top_k_chunks

ROOT = Path(__file__).resolve().parents[2]
FLAG = "ORCHESTRATOR_SPILL_EXTRACTIVE_SUMMARY"


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    reset_features()
    yield
    reset_features()


def _repl(llm=None):
    cfg = REPLConfig(output_cap=200, spill_dir=f"/mnt/raid0/llm/tmp/spill_ext_{uuid.uuid4().hex[:8]}")
    return REPLEnvironment(context="x", config=cfg, llm_primitives=llm)


def _output() -> str:
    lines = [f"row {i} filler filler" for i in range(200)]
    lines[100] = "row 100 the zebra_metric regressed to 0.42"
    return "\n".join(lines)


def test_flag_declared_default_off_and_in_spec():
    spec = _REGISTRY_BY_NAME["spill_extractive_summary"]
    assert spec.default_test is False and spec.default_prod is False
    assert "  spill_extractive_summary: baseline" in (ROOT / "orchestration/runtime_flags.spec.yaml").read_text()
    assert FLAG in (ROOT / "docs/chapters/01-runtime-environment.md").read_text()


def test_top_k_finds_relevant_chunk_and_skips_zero_score():
    out = _output()
    hits = top_k_chunks(out, "find zebra_metric", k=3, chunk_chars=200, exclude_before=600)
    assert len(hits) == 1 and "zebra_metric" in hits[0][1]
    assert top_k_chunks(out, "nothing_matches_here", k=3, chunk_chars=200) == []
    assert top_k_chunks(out, "", k=3, chunk_chars=200) == []


def test_extractive_summary_is_verbatim_head_tail_and_excerpt():
    out = _output()
    s = extractive_summary(out, "print zebra_metric", head_chars=100, tail_chars=80, chunk_chars=200)
    assert s.startswith(out[:100]) and s.endswith(out[-80:])
    assert "zebra_metric regressed to 0.42" in s
    assert "chars omitted" in s and "[@char " in s
    assert extractive_summary("short", "q", head_chars=100, tail_chars=80) == "short"


def test_flag_off_still_uses_worker_summary():
    llm = MagicMock()
    llm.llm_call.return_value = "WORKER SUMMARY"
    repl = _repl(llm)
    res = repl.execute('print("\\n".join(f"row {i} filler" for i in range(200)))')
    assert "WORKER SUMMARY" in res.output
    assert llm.llm_call.called


def test_flag_on_extractive_no_worker_call(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    reset_features()
    llm = MagicMock()
    repl = _repl(llm)
    code = (
        'rows = [f"row {i} filler filler" for i in range(200)]\n'
        'rows[100] = "row 100 the zebra_metric regressed"\n'
        'print("\\n".join(rows))  # find zebra_metric'
    )
    res = repl.execute(code)
    assert not llm.llm_call.called
    assert "zebra_metric regressed" in res.output
    assert "peek(" in res.output and "chars omitted" in res.output
