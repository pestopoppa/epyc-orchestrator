"""Deterministic post-promotion smoke for the both-mode production stack.

The probe targets every chat endpoint the stack manifest says it launches in the
HOT tier, plus the six production BGE embedding endpoints.

2026-08-01: the chat port list used to be a hand-maintained literal. It had gone
closed-world-stale — it still named 8280/8380 and 8282/8382 (frontdoor and
worker_general quarters 4-5, retired when both roles dropped to three instances),
8385/8485 (ingest_long_context quarters 4-5), and 8087 (the retired standalone
vision_escalation 7B, now an alias on worker_vision's :8086 process) — while
MISSING 8074, the architect_critic lane promoted to HOT on 2026-08-01. It also
looked up `vision_escalation` in NUMA_CONFIG, where it has no entry precisely
because it is an alias rather than its own server. The result was a smoke that
probed seven dead ports, skipped a live one, and refused to run at all.

The ports are now DERIVED from `stack_manifest.HOT_SERVERS`, the launcher's own
computed server list (built from ROLE_LAUNCH_META + NUMA_CONFIG) — the same
structure `orchestrator_stack.py` starts from. `topology_errors()` no longer
compares a literal against a derivation; it checks that the derived surface is
internally coherent and agrees with PORT_MAP and NUMA_CONFIG.
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

from scripts.server.stack_manifest import (  # noqa: E402
    EMBEDDER_PORTS,
    HOT_SERVERS,
    PORT_MAP,
    WARM_SERVERS,
)
from scripts.server.stack_numa import NUMA_CONFIG  # noqa: E402


def _chat_servers() -> list[dict[str, Any]]:
    """HOT-tier servers that expose /v1/chat/completions.

    Everything the launcher classifies HOT except the embedding servers, which
    speak /embedding instead and are probed separately.
    """
    return [
        server
        for server in HOT_SERVERS
        if not server.get("embedding") and isinstance(server.get("port"), int)
    ]


def expected_chat_ports() -> tuple[int, ...]:
    """Chat ports this stack declares, in launch order. Derived, never hand-listed."""
    return tuple(server["port"] for server in _chat_servers())


EXPECTED_CHAT_PORTS = expected_chat_ports()
EXPECTED_EMBEDDER_PORTS = (8090, 8091, 8092, 8093, 8094, 8095)
EMBEDDING_DIMENSION = 1024
TIMEOUT_SECONDS = 30.0


def topology_errors() -> list[str]:
    """Return manifest/NUMA incoherence that would make the probe meaningless.

    The chat surface is DERIVED, so there is nothing left to compare it against;
    what IS checkable is whether the sources it is derived from agree with each
    other. PORT_MAP must name the same port the launcher computes for every role
    served by a chat server, every NUMA_CONFIG role must actually be launched, and
    the derived surface must be non-empty and duplicate-free.
    """
    errors: list[str] = []

    servers = _chat_servers()
    ports = [server["port"] for server in servers]
    if not ports:
        errors.append("no chat servers in the manifest HOT tier")
    duplicates = sorted({port for port in ports if ports.count(port) > 1})
    if duplicates:
        errors.append(f"duplicate chat ports in the manifest HOT tier: {duplicates}")

    # Every role served by a chat process — including aliases like coder_escalation
    # and vision_escalation, which ride another role's server and therefore have no
    # NUMA_CONFIG entry of their own — must resolve to the port PORT_MAP advertises.
    computed_role_ports: dict[str, int] = {}
    for server in servers:
        for role in server.get("roles", []):
            computed_role_ports.setdefault(str(role), server["port"])
    for role, port in sorted(computed_role_ports.items()):
        declared = PORT_MAP.get(role)
        if declared is None:
            errors.append(f"{role}: served on {port} but absent from PORT_MAP")
        elif declared != port:
            errors.append(f"{role}: PORT_MAP says {declared}, launcher computes {port}")

    # A NUMA_CONFIG entry for a role nobody launches is dead pinning config.
    launched_roles = {
        str(role)
        for server in HOT_SERVERS + WARM_SERVERS
        for role in server.get("roles", [])
    }
    for role in sorted(NUMA_CONFIG):
        if role not in launched_roles:
            errors.append(f"{role}: has NUMA_CONFIG instances but is never launched")

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
    if isinstance(body, list) and body and isinstance(body[0], dict):
        vector = body[0].get("embedding")
        return vector[0] if isinstance(vector, list) and vector and isinstance(vector[0], list) else vector
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
