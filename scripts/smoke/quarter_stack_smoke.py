"""Deterministic post-promotion smoke for the quarter production stack.

The probe deliberately targets the fourteen quarter/vision chat endpoints and
the six production BGE endpoints.  It does not probe the deprecated
qwen3.5-122B lane on port 8083.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

# Make package imports work when this file is invoked directly.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.server.stack_manifest import EMBEDDER_PORTS, PORT_MAP  # noqa: E402
from scripts.server.stack_numa import NUMA_CONFIG  # noqa: E402


EXPECTED_CHAT_PORTS = (8080, 8180, 8280, 8380, 8082, 8182, 8282, 8382, 8185, 8285, 8385, 8485, 8086, 8087)
EXPECTED_EMBEDDER_PORTS = (8090, 8091, 8092, 8093, 8094, 8095)
EMBEDDING_DIMENSION = 1024
TIMEOUT_SECONDS = 30.0

_QUARTER_ROLES = ("frontdoor", "worker_general", "ingest_long_context")
_SINGLE_CHAT_ROLES = ("worker_vision", "vision_escalation")


def _ports_for_role(role: str) -> list[int]:
    try:
        return [int(instance[1]) for instance in NUMA_CONFIG[role]["instances"]]
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise ValueError(f"invalid NUMA_CONFIG entry for {role!r}: {exc}") from exc


def topology_errors() -> list[str]:
    """Return closed-world manifest/NUMA drift errors for this fixed smoke."""
    errors: list[str] = []
    derived_chat: list[int] = []

    for role in _QUARTER_ROLES:
        try:
            ports = _ports_for_role(role)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if not ports or PORT_MAP.get(role) != ports[0]:
            errors.append(f"{role}: PORT_MAP primary does not match NUMA_CONFIG")
            continue
        derived_chat.extend(ports[1:])

    for role in _SINGLE_CHAT_ROLES:
        try:
            ports = _ports_for_role(role)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if len(ports) != 1 or PORT_MAP.get(role) != ports[0]:
            errors.append(f"{role}: expected one PORT_MAP-aligned NUMA instance")
            continue
        derived_chat.extend(ports)

    if tuple(derived_chat) != EXPECTED_CHAT_PORTS:
        errors.append(
            f"chat topology mismatch: expected {list(EXPECTED_CHAT_PORTS)}, got {derived_chat}"
        )
    if PORT_MAP.get("architect_general") != 8083:
        errors.append("architect_general port is no longer the explicitly excluded 8083")
    if 8083 in derived_chat:
        errors.append("deprecated qwen3.5-122B port 8083 is included in chat smoke")
    if tuple(EMBEDDER_PORTS) != EXPECTED_EMBEDDER_PORTS:
        errors.append(
            f"embedder topology mismatch: expected {list(EXPECTED_EMBEDDER_PORTS)}, got {EMBEDDER_PORTS}"
        )
    return errors


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _chat_row(port: int, timeout_s: float) -> dict[str, Any]:
    started_at = _utc_now()
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    row: dict[str, Any] = {"kind": "chat", "port": port, "url": url, "started_at": started_at}
    try:
        response = httpx.post(
            url,
            json={
                "messages": [{"role": "user", "content": "Return exactly: ok"}],
                "max_tokens": 8,
                "temperature": 0,
                "stream": False,
            },
            timeout=timeout_s,
        )
        row["status_code"] = response.status_code
        response.raise_for_status()
        body = response.json()
        message = body["choices"][0]["message"]
        content = message.get("content")
        reasoning = message.get("reasoning_content")
        finish_reason = body["choices"][0].get("finish_reason")
        row.update(
            content=content,
            finish_reason=finish_reason,
            reasoning_content=reasoning,
            ok=(content == "ok" and finish_reason == "stop" and reasoning in (None, "")),
        )
        if not row["ok"]:
            row["error"] = "expected content='ok', finish_reason='stop', and empty reasoning_content"
    except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
        row.update(ok=False, error=f"{type(exc).__name__}: {exc}")
    return row


def _embedding_vector(body: Any) -> Any:
    if isinstance(body, dict) and "embedding" in body:
        vector = body["embedding"]
        return vector[0] if isinstance(vector, list) and vector and isinstance(vector[0], list) else vector
    if isinstance(body, dict) and isinstance(body.get("data"), list) and body["data"]:
        return body["data"][0].get("embedding")
    return None


def _embedding_row(port: int, timeout_s: float) -> dict[str, Any]:
    started_at = _utc_now()
    url = f"http://127.0.0.1:{port}/embedding"
    row: dict[str, Any] = {"kind": "embedding", "port": port, "url": url, "started_at": started_at}
    try:
        response = httpx.post(url, json={"content": "quarter stack smoke"}, timeout=timeout_s)
        row["status_code"] = response.status_code
        response.raise_for_status()
        vector = _embedding_vector(response.json())
        dimension = len(vector) if isinstance(vector, list) else None
        row.update(dimension=dimension, ok=(dimension == EMBEDDING_DIMENSION))
        if not row["ok"]:
            row["error"] = f"expected one {EMBEDDING_DIMENSION}-dimension embedding"
    except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
        row.update(ok=False, error=f"{type(exc).__name__}: {exc}")
    return row


def write_jsonl_atomic(output: Path, rows: list[dict[str, Any]]) -> None:
    """Publish a complete JSONL artifact, never a partially written one."""
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def run_smoke(output: Path, *, timeout_s: float = TIMEOUT_SECONDS) -> int:
    """Run exactly one sequential request per endpoint and publish all results."""
    errors = topology_errors()
    if errors:
        write_jsonl_atomic(output, [])
        for error in errors:
            print(f"topology error: {error}", file=sys.stderr)
        return 2

    rows: list[dict[str, Any]] = []
    for port in EXPECTED_CHAT_PORTS:
        rows.append(_chat_row(port, timeout_s))
    for port in EXPECTED_EMBEDDER_PORTS:
        rows.append(_embedding_row(port, timeout_s))
    write_jsonl_atomic(output, rows)
    return 0 if all(row["ok"] for row in rows) else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="JSONL result artifact")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    return run_smoke(parse_args(argv).output)


if __name__ == "__main__":
    raise SystemExit(main())
