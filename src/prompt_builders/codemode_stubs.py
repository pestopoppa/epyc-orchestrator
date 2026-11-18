"""Codemode reliability fix 4: typed Python stubs for the picked tools (``ORCHESTRATOR_CODEMODE_STUBS=1``, DEFAULT off).

The static tools text gives the model ``CALL("name", ...)`` strings with no argument types and no return type, and
``CALL`` / ``list_dir`` return JSON *strings*. With the flag, the selected-tools block shows each picked tool as the
Python function it is bound to in the REPL (``codemode_tools.bind_stub_tools``)::

    def syn_sku_price(sku: str) -> dict:  # {"sku", "price_cents", "currency"}
        \"\"\"Look up the current unit price in cents for a product SKU.\"\"\"

Types come from the JSON schemas; a return type comes from an optional ``returns`` / ``returns_doc`` key on the tool dict
(unknown -> ``Any``). Returned values are Python objects (CM-3 envelopes are unwrapped, JSON text is parsed).
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from typing import Any

ENV_STUBS = "ORCHESTRATOR_CODEMODE_STUBS"

_PY_TYPES = {"string": "str", "integer": "int", "number": "float", "boolean": "bool", "array": "list", "object": "dict"}
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def stubs_enabled(environ: Mapping[str, str] | None = None) -> bool:
    env = os.environ if environ is None else environ
    return str(env.get(ENV_STUBS, "")).strip().lower() in ("1", "true", "on", "yes")


def _one_line(text: str, limit: int = 160) -> str:
    t = " ".join(str(text or "").split()).replace('"""', "'''")
    return t if len(t) <= limit else t[: limit - 3].rstrip() + "..."


def render_stub(tool: Mapping[str, Any]) -> str:
    """Def-line stub plus one-line docstring for one ``{name, description, parameters[, returns, returns_doc]}`` tool."""
    name = str(tool["name"])
    params = tool.get("parameters") or {}
    required, optional = [], []
    for pn, spec in params.items():
        if not _IDENT.match(str(pn)):
            continue
        spec = spec if isinstance(spec, Mapping) else {}
        ty = _PY_TYPES.get(str(spec.get("type", "string")), "Any")
        (required if spec.get("required") else optional).append(f"{pn}: {ty}" if spec.get("required") else f"{pn}: {ty} = None")
    ret = str(tool.get("returns") or "Any")
    doc = f"  # {_one_line(tool['returns_doc'], 100)}" if tool.get("returns_doc") else ""
    return f"def {name}({', '.join(required + optional)}) -> {ret}:{doc}\n    \"\"\"{_one_line(tool.get('description', ''))}\"\"\""


STUB_CORE_LINES = (
    "- `context`: str - full input. `artifacts`: dict - store intermediate results between turns.",
    "- Each tool below is a Python function: call it directly and use the returned Python object (no json.loads).",
    "  A failing call raises an exception with the reason. `CALL(name, **kw)` / `TOOL(name, **kw)` still reach any other tool.",
    "- `list_tools()`: discover ALL available tools and their schemas (always available; the picks below add to it, never replace it).",
    "  `search_tools(query)` / `describe_tool(name)` find or read one tool without listing everything.",
    "- `FINAL(answer)`: signal completion with the final answer. REQUIRED for every task.",
)


def render_stubs_block(tools: Sequence[Mapping[str, Any]], selected: Sequence[str]) -> str:
    by = {t["name"]: t for t in tools}
    body = [render_stub(by[n]) for n in selected if n in by]
    return "\n".join(["## Tools (selected for this request)", *STUB_CORE_LINES, "", "```python", *body, "```"])
