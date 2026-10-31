"""Request-local capability for an exact graph-produced session artifact."""
import os
import tempfile
from pathlib import Path
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Callable, Iterator

_capability: ContextVar[tuple[str, Callable] | None] = ContextVar(
    "repl_session_artifact", default=None,
)


@contextmanager
def registry_artifact_read(path: str, reader: Callable) -> Iterator[None]:
    token = _capability.set((path, reader))
    try:
        yield
    finally:
        _capability.reset(token)


def is_registry_artifact(path: str) -> bool:
    cap = _capability.get()
    return cap is not None and cap[0] == path


def read_registry_artifact(path: str):
    cap = _capability.get()
    return cap[1](path, -1, 0) if cap is not None and cap[0] == path else None


def session_artifact_root() -> Path:
    try:
        from src.config import get_config
        base = get_config().paths.tmp_dir
    except Exception:
        base = tempfile.gettempdir()
    return Path(base) / "repl_session_artifacts"


def is_reserved_artifact_path(path: str) -> bool:
    root = os.path.abspath(session_artifact_root())
    try:
        real_root = os.path.realpath(root)
        absolute = os.path.abspath(path)
        resolved = os.path.realpath(absolute)
    except (OSError, ValueError):
        return True  # An unresolvable path cannot be proven outside the reserved directory.
    return (absolute == root or absolute.startswith(root + os.sep)
            or resolved == real_root or resolved.startswith(real_root + os.sep))
