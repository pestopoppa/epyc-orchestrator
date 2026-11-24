"""AEI-TOKEN-ROOT-CAPTURE: one root-token row per completed /chat request, beside the usage ledger.

``usage_provenance.jsonl`` holds the per-call rows. ``scripts/analysis/token_span_ratio.py`` divides
their sum by the request's root total, and nothing persisted that total. This module appends one row
per request to ``usage_roots.jsonl`` (a sibling of the ledger, same directory and kill switch) behind
the default-off ``usage_root_capture`` flag. Off: no file is touched.

Row: ``{"schema", "request_id", "prompt_tokens", "output_tokens_reported", "path", "ts_epoch"}``.
``path`` is ``cpu`` / ``gpu`` / ``stream`` / ``unknown``; it is never guessed (an unresolvable role
is ``unknown``, not ``cpu``). Never raises: capture must not affect serving.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

SCHEMA = "epyc.request_root_usage.v1"
PATHS = ("cpu", "gpu", "stream", "unknown")
_log = logging.getLogger(__name__)


def roots_path() -> Path | None:
    from .usage_log import log_path

    ledger = log_path()
    return ledger.with_name("usage_roots.jsonl") if ledger is not None else None


def path_tag(role: str | None, *, streaming: bool = False) -> str:
    """cpu / gpu / stream for the request's primary role; ``unknown`` when it cannot be resolved."""
    if streaming:
        return "stream"
    if not role:
        return "unknown"
    try:
        from src.scheduling.device_model import resolve_role_device

        return "gpu" if resolve_role_device(role).is_gpu else "cpu"
    except Exception:
        return "unknown"


def _count(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def record_root(
    request_id: str | None,
    prompt_tokens: Any,
    output_tokens_reported: Any,
    path: str,
    *,
    enabled: bool | None = None,
) -> bool:
    """Append one root row. Returns False when disabled, keyless or the write failed."""
    if enabled is None:
        from src.features import features

        enabled = features().usage_root_capture
    if not enabled or not request_id:
        return False
    target = roots_path()
    if target is None:
        return False
    try:
        row = {
            "schema": SCHEMA, "request_id": str(request_id),
            "prompt_tokens": _count(prompt_tokens),
            "output_tokens_reported": _count(output_tokens_reported),
            "path": path if path in PATHS else "unknown",
            "ts_epoch": time.time(),
        }
        line = (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd = os.open(target, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            os.write(fd, line)  # one O_APPEND write of a short line is atomic
        finally:
            os.close(fd)
        return True
    except Exception:
        _log.warning("usage root capture write failed", exc_info=True)
        return False


def read_roots(rows: Iterable[str]) -> dict[str, dict[str, Any]]:
    """request_id -> {prompt_tokens, output_tokens_reported, path}; the last row per request wins."""
    out: dict[str, dict[str, Any]] = {}
    for line in rows:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get("schema") == SCHEMA and row.get("request_id"):
            out[str(row["request_id"])] = {
                "prompt_tokens": row.get("prompt_tokens"),
                "output_tokens_reported": row.get("output_tokens_reported"),
                "path": row.get("path"),
            }
    return out
