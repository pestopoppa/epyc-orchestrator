"""AutoPilot mutation exclusions: repo-relative targets PromptForge must not mutate (HIL-6j).

A sibling process that measures a prompt file (the HIL-6 pilot measures
``orchestration/prompts/architect_investigate.md``) needs AutoPilot's PromptForge to leave that
file's bytes alone while it runs. This module is the one shared contract for that:

* **Default empty.** With no exclusions file, :func:`load` returns an empty set and every
  PromptForge path behaves byte-for-byte as before (the guards return before doing anything).
* **One file, read at call time.** AutoPilot is a separate process, so an environment variable set
  by the pilot cannot reach it; the pilot writes a small JSON file instead. Its path is
  ``$AUTOPILOT_MUTATION_EXCLUSIONS_FILE`` when set, else :data:`DEFAULT_EXCLUSIONS_FILE` (under
  the shared clone's git-ignored ``tmp/``, so writing it never dirties a tracked tree). Because it
  is read on every call, an AutoPilot that starts or restarts while the file exists is guarded too.
* **Owner-scoped entries.** :func:`set_exclusion` adds or refreshes one entry for one owner and
  :func:`clear_exclusion` removes only that owner's entries; the file is deleted when it empties.
  Writes are atomic (temp file + ``os.replace``) under an ``flock`` on a sidecar lock file.
* **Fail closed on corruption.** An exclusions file that exists but cannot be parsed excludes
  EVERY prompt target (:data:`UNREADABLE`): AutoPilot then proposes no prompt mutation rather than
  silently mutating a file someone asked it to leave alone.
"""

from __future__ import annotations

import fcntl
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

SCHEMA = "epyc.orchestrator.autopilot_mutation_exclusions.v1"
ENV_FILE = "AUTOPILOT_MUTATION_EXCLUSIONS_FILE"
#: AutoPilot runs from the shared clone; its ``tmp/`` is git-ignored.
DEFAULT_EXCLUSIONS_FILE = Path("/mnt/raid0/llm/epyc-orchestrator/tmp/autopilot_mutation_exclusions.json")
#: Sentinel target meaning "the file exists but is unreadable: exclude every target".
UNREADABLE = "*"


def exclusions_file() -> Path:
    value = os.environ.get(ENV_FILE, "").strip()
    return Path(value) if value else DEFAULT_EXCLUSIONS_FILE


def _norm(target: str) -> str:
    p = PurePosixPath(str(target).replace("\\", "/"))
    if p.is_absolute() or ".." in p.parts or not p.parts:
        raise ValueError(f"exclusion target must be a repo-relative path, got {target!r}")
    return p.as_posix()


def _read(path: Path) -> dict[str, Any] | None:
    """The parsed document, ``{}`` when the file is absent, ``None`` when it is unreadable."""
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError:
        return None
    try:
        doc = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(doc, dict) or doc.get("schema") != SCHEMA or not isinstance(doc.get("entries"), list):
        return None
    return doc


def load(path: Path | None = None) -> frozenset[str]:
    """Excluded repo-relative targets. Empty when no file exists; ``{UNREADABLE}`` when corrupt."""
    doc = _read(path or exclusions_file())
    if doc is None:
        return frozenset({UNREADABLE})
    out: set[str] = set()
    for entry in doc.get("entries") or ():
        if not isinstance(entry, dict):
            return frozenset({UNREADABLE})
        try:
            out.add(_norm(entry.get("target", "")))
        except ValueError:
            return frozenset({UNREADABLE})
    return frozenset(out)


def excluded_reason(target: str, excluded: frozenset[str] | None = None, path: Path | None = None) -> str | None:
    """``mutation_exclusion:<target>`` when ``target`` may not be mutated now, else None."""
    excluded = load(path) if excluded is None else excluded
    if not excluded:
        return None
    if UNREADABLE in excluded:
        return "mutation_exclusion:unreadable_exclusions_file"
    try:
        norm = _norm(target)
    except ValueError:
        return None
    return f"mutation_exclusion:{norm}" if norm in excluded else None


@contextmanager
def _locked(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_name(path.name + ".lock")
    with open(lock, "a+") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def _write(path: Path, entries: list[dict[str, Any]]) -> None:
    if not entries:
        path.unlink(missing_ok=True)
        return
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps({"schema": SCHEMA, "entries": entries}, indent=1, sort_keys=True) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)


def set_exclusion(target: str, *, owner: str, reason: str, path: Path | None = None,
                  pid: int | None = None) -> Path:
    """Add (or refresh) ``owner``'s exclusion of ``target``. Returns the file written."""
    if not owner.strip():
        raise ValueError("an exclusion names its owner")
    norm = _norm(target)
    path = path or exclusions_file()
    with _locked(path):
        doc = _read(path)
        if doc is None:
            raise ValueError(f"{path} exists but is unreadable; refusing to overwrite it")
        entries = [e for e in doc.get("entries") or ()
                   if not (e.get("owner") == owner and e.get("target") == norm)]
        entries.append({"target": norm, "owner": owner, "reason": reason,
                        "set_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "pid": int(pid if pid is not None else os.getpid())})
        _write(path, entries)
    return path


def clear_exclusion(*, owner: str, target: str | None = None, path: Path | None = None) -> int:
    """Remove ``owner``'s entries (only ``target``'s when given). Returns how many were removed."""
    path = path or exclusions_file()
    norm = _norm(target) if target is not None else None
    with _locked(path):
        doc = _read(path)
        if doc is None:
            raise ValueError(f"{path} exists but is unreadable; refusing to rewrite it")
        entries = list(doc.get("entries") or ())
        keep = [e for e in entries
                if not (e.get("owner") == owner and (norm is None or e.get("target") == norm))]
        if len(keep) != len(entries):
            _write(path, keep)
        return len(entries) - len(keep)


__all__ = [
    "DEFAULT_EXCLUSIONS_FILE",
    "ENV_FILE",
    "SCHEMA",
    "UNREADABLE",
    "clear_exclusion",
    "excluded_reason",
    "exclusions_file",
    "load",
    "set_exclusion",
]
