"""codemode-ab: the tool selector is flag-gated, default OFF, and the builder hook is inert unless on."""

from __future__ import annotations

import pytest

from src.prompt_builders import tool_selector as ts
from src.prompt_builders.builder import PromptBuilder
from src.prompt_builders.constants import DEFAULT_ROOT_LM_TOOLS

TOOLS = [
    {"name": "syn_sku_price", "description": "Look up unit price for a product SKU", "parameters": {"sku": {"type": "string", "required": True}}},
    {"name": "syn_calendar_create", "description": "Create a calendar event", "parameters": {"title": {"type": "string", "required": True}}},
    {"name": "read_file", "description": "Read a file", "parameters": {"path": {"type": "string", "required": True}}},
]


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv(ts.ENV_FLAG, raising=False)
    ts.set_selector_provider(None)
    yield
    ts.set_selector_provider(None)


def test_default_off_and_unknown_values_are_off(monkeypatch):
    assert ts.selector_variant({}) == "off"
    assert ts.selector_variant({ts.ENV_FLAG: "1"}) == "off"
    assert ts.selector_variant({ts.ENV_FLAG: "EMBED"}) == "embed"


def test_hook_is_inert_when_flag_off_even_with_provider(monkeypatch):
    ts.set_selector_provider(lambda task: "SELECTED")
    assert ts.maybe_selected_tools_block("x") is None
    out = PromptBuilder().build_root_lm_prompt(state="s", original_prompt="task")
    assert DEFAULT_ROOT_LM_TOOLS in out and "SELECTED" not in out


def test_hook_replaces_tools_only_when_on(monkeypatch):
    ts.set_selector_provider(lambda task: "SELECTED-TOOLS-BLOCK")
    monkeypatch.setenv(ts.ENV_FLAG, "embed")
    out = PromptBuilder().build_root_lm_prompt(state="s", original_prompt="task")
    assert "SELECTED-TOOLS-BLOCK" in out and DEFAULT_ROOT_LM_TOOLS not in out


def test_broken_provider_falls_back_to_static(monkeypatch):
    def boom(_):
        raise RuntimeError("x")

    ts.set_selector_provider(boom)
    monkeypatch.setenv(ts.ENV_FLAG, "oneshot")
    assert DEFAULT_ROOT_LM_TOOLS in PromptBuilder().build_root_lm_prompt(state="s", original_prompt="task")


def test_lexical_and_render():
    sc = ts.lexical_scores("what is the price of this sku", TOOLS)
    assert max(sc, key=sc.get) == "syn_sku_price"
    block = ts.render_tools_block(TOOLS, ["syn_sku_price"])
    assert "syn_sku_price(sku: string)" in block and "FINAL(" in block and "syn_calendar_create" not in block


def test_embedding_selector_with_fake_embedder():
    np = pytest.importorskip("numpy")
    from src.repl_environment.context_search import DenseResult

    class Fake:
        model_id = "fake"

        def embed(self, texts, *, timeout_s=0.0, admission_wait_s=0.0):
            v = np.array([[1.0 if "price" in t or "sku" in t.lower() else 0.0, 1.0 if "calendar" in t else 0.0, 0.1] for t in texts], dtype="float32")
            return DenseResult(v, "fake")

    r = ts.EmbeddingToolSelector(Fake(), TOOLS).select("price of a SKU", k=1)
    assert r.selected == ["syn_sku_price"] and r.fallback is None
