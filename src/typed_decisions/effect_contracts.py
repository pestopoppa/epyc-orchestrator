"""UFH-TE-4: effect contracts for tools reachable by stored or replayed calls."""

from __future__ import annotations

from pathlib import Path

import yaml

EFFECT_CLASSES = ("read-only", "mutating", "external_side_effect", "irreversible", "unknown")
_PATH = Path(__file__).resolve().parents[2] / "orchestration" / "tool_effect_contracts.yaml"


def load_contracts(path: Path = _PATH) -> dict[str, dict]:
    return yaml.safe_load(path.read_text())["contracts"]


def reuse_allowed(tool_name: str, contracts: dict[str, dict] | None = None) -> bool:
    """False for `unknown` and for any tool with no contract (absence is unknown, never inferred)."""
    entry = (load_contracts() if contracts is None else contracts).get(tool_name)
    return bool(entry) and entry.get("effect_class") in EFFECT_CLASSES[:-1] and bool(
        entry.get("reuse_allowed")
    )
