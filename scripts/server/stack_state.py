"""State persistence for orchestrator stack processes."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class ProcessInfo:
    """Information about a running process."""

    role: str
    pid: int
    port: int
    started_at: str
    model_path: str
    log_file: str


def load_state_file(state_file: Path) -> dict[str, ProcessInfo]:
    """Load process state from a JSON file."""
    if not state_file.exists():
        return {}
    try:
        with open(state_file) as file:
            data = json.load(file)
    except json.JSONDecodeError:
        return {}

    out: dict[str, ProcessInfo] = {}
    for key, value in data.items():
        if not isinstance(value, dict):
            continue
        try:
            out[key] = ProcessInfo(**value)
        except TypeError as exc:
            print(f"[load_state] dropping non-ProcessInfo entry {key!r}: {exc}")
    return out


def save_state_file(state_file: Path, state: dict[str, ProcessInfo]) -> None:
    """Persist only actionable ProcessInfo records to a JSON file."""
    state_file.parent.mkdir(parents=True, exist_ok=True)
    serializable: dict[str, dict[str, Any]] = {}
    for key, value in state.items():
        if isinstance(value, ProcessInfo):
            serializable[key] = asdict(value)
    with open(state_file, "w") as file:
        json.dump(serializable, file, indent=2)
