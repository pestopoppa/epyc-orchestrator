"""Export JSON schemas for registered tools from the tool manifests (read-only; no tool module is imported)."""
from __future__ import annotations

import json
from pathlib import Path

DEFAULT_TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools"


def manifest_tool_schema(params: dict) -> dict:
    props, required = {}, []
    for name, spec in (params or {}).items():
        spec = dict(spec)
        if spec.pop("required", False):
            required.append(name)
        spec.pop("default", None)
        spec.pop("description", None)
        props[name] = spec
    out = {"type": "object", "properties": props, "additionalProperties": False}
    if required:
        out["required"] = required
    return out


def export_tool_schemas(tools_dir: Path | str = DEFAULT_TOOLS_DIR) -> dict[str, dict]:
    schemas: dict[str, dict] = {}
    for mf in sorted(Path(tools_dir).glob("*/manifest.json")):
        data = json.loads(mf.read_text())
        for tool in data.get("tools", []):
            schemas[tool["name"]] = manifest_tool_schema(tool.get("parameters", {}))
    return schemas
