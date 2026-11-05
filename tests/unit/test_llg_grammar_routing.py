"""V11-122: GBNF -> %llguidance Lark routing for the turn-1 grammar (flag default OFF)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from src.backends import llg_grammar
from src.registry.tool_registry import Tool, ToolCategory, ToolRegistry

ROOT = Path(__file__).resolve().parents[2]


def _gbnf(shape: str) -> str:
    reg = ToolRegistry()
    reg.register_tool(Tool(name="run_shell", description="d", category=ToolCategory.WEB,
                           parameters={"command": {"type": "string"}}))
    return reg.generate_gbnf_grammar(None, shape=shape)


def test_flag_defaults_off_and_leaves_gbnf_untouched(monkeypatch):
    monkeypatch.delenv(llg_grammar.ENV_FLAG, raising=False)
    g = _gbnf("legacy")
    assert llg_grammar.route_grammar(g) == (g, "gbnf")


@pytest.mark.parametrize("val", ["0", "off", "false", ""])
def test_flag_off_spellings(monkeypatch, val):
    monkeypatch.setenv(llg_grammar.ENV_FLAG, val)
    assert not llg_grammar.llguidance_routing_enabled()


@pytest.mark.parametrize("shape", ["legacy", "call"])
def test_flag_on_translates_both_turn1_shapes(monkeypatch, shape):
    monkeypatch.setenv(llg_grammar.ENV_FLAG, "1")
    out, fmt = llg_grammar.route_grammar(_gbnf(shape))
    assert fmt == "llguidance"
    assert out.startswith("%llguidance")           # the server's own routing test
    assert "start:" in out and "::=" not in out     # Lark, not GBNF
    assert '"run_shell"' in out


def test_lark_and_llguidance_input_pass_through(monkeypatch):
    monkeypatch.setenv(llg_grammar.ENV_FLAG, "1")
    lark = "%llguidance {}\nstart: \"a\"\n"
    assert llg_grammar.gbnf_to_llguidance(lark) == lark
    assert llg_grammar.route_grammar("") == ("", "gbnf")


def test_translation_is_deterministic():
    g = _gbnf("call")
    assert llg_grammar.gbnf_to_llguidance(g) == llg_grammar.gbnf_to_llguidance(g)


def test_vendored_converter_is_the_pinned_upstream_file():
    head = (ROOT / "src/backends/_vendor/llg_gbnf_to_lark.py").read_text().splitlines()[:3]
    assert "d795912fedc7d393de740177ea9ea761e7905774" in head[0]


# Parity of the translated grammar against the GBNF on the SPEC-GRAM corpus. Needs a Python
# with llguidance+tokenizers on LLG_PARITY_PYTHONPATH, the corpus, a tokenizer and a
# test-gbnf-validator dir; skipped otherwise. LLG_PARITY_LIMIT bounds it (default 200 commands).
_CORPUS = os.environ.get("LLG_PARITY_CORPUS", "/mnt/raid0/llm/tmp/specgram3-llg/replay101/work/corpus.jsonl")
_PYPATH = os.environ.get("LLG_PARITY_PYTHONPATH", "/mnt/raid0/llm/tmp/specgram3-llg/v101/lib/python3.13/site-packages")
_VALID = os.environ.get("LLG_PARITY_VALIDATOR", "/mnt/raid0/llm/kernels/production/cpu")
_TOK = os.environ.get("LLG_PARITY_TOKENIZER", "/mnt/raid0/llm/models/turboderp/Qwen3.8-Flash-Next-exl3-3.05bpw/tokenizer.json")


@pytest.mark.skipif(not (Path(_CORPUS).is_file() and Path(_PYPATH, "llguidance").is_dir()
                         and Path(_VALID, "test-gbnf-validator").is_file() and Path(_TOK).is_file()),
                    reason="SPEC-GRAM corpus / llguidance / gbnf validator not available")
def test_translated_grammar_acceptance_parity_on_spec_gram_corpus(tmp_path):
    out = tmp_path / "parity.json"
    env = {**os.environ, "PYTHONPATH": _PYPATH}
    r = subprocess.run([sys.executable, str(ROOT / "scripts/analysis/llg_gbnf_lark_parity.py"), "--corpus", _CORPUS,
                        "--validator", _VALID, "--tokenizer", _TOK, "--limit", os.environ.get("LLG_PARITY_LIMIT", "200"),
                        "--out", str(out)], capture_output=True, text=True, env=env, timeout=1800)
    res = json.loads(out.read_text())
    for shape in ("legacy", "call"):
        s = res["shapes"][shape]
        assert "error" not in s, s
        assert s["disagree"] == 0, s["mismatches"][:3]
        assert 0 < s["gbnf_accepted"] < s["texts"]   # both positives and negatives were exercised
    assert r.returncode == 0 and res["verdict"] == "PASS"
