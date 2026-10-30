#!/usr/bin/env python3
"""Run the UFH14 DEPLOY-EC lease/credit sequence (dry-run by default).

Run from the orchestrator checkout root, or invoke this file by absolute path:
``python scripts/research/deploy_ec2_lease.py --dry-run``. Execution requires
``--execute --inputs PATH`` and sends only prompts read from that JSON file.
The caller must run it only in the authorized test window described by the
runbook. The default dry-run is offline and prints routes/body keys.

Input JSON for execution must provide ``lane_a_chat_template_kwargs``,
``lane_a_first_messages``, ``lane_b_messages``, ``lane_a_credit_messages``, and
``lane_b_model``. Each messages field is the real OpenAI messages array to send;
no prompt text is synthesized by this runner. Relative ``--inputs`` paths are
resolved from the caller's current working directory; dependencies load from
the active Python environment, and the script imports no checkout modules.

This is a deliberately small successor to the lost 2026-10-03 scratch runner. It
keeps the documented lane-A passthrough and lane-B /v1 client-mode routes. Each
invocation creates a fresh valid x_session_id for lane B, as required by the
current v1 client-session guard.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import re
import uuid
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlsplit

import httpx


SESSION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@+/=-]{0,127}$")
DEFAULT_ORCHESTRATOR = "http://127.0.0.1:8000"
DEFAULT_SERVER_PORT = 8083
LANE_A = "architect_critic"
LANE_B = "coder_escalation"


def run_session_id() -> str:
    """Return a per-run, guard-valid session id; never reuse a static default."""
    value = f"ses_ec2_{uuid.uuid4().hex}"
    if not SESSION_ID_RE.fullmatch(value):  # defensive against future format edits
        raise ValueError("generated x_session_id is invalid")
    return value


def lane_b_client_body(*, session_id: str, model: str,
                       messages: list[dict[str, Any]], max_tokens: int = 64) -> dict[str, Any]:
    """Build the default /v1 client-mode body and fail before dispatch if invalid."""
    if not SESSION_ID_RE.fullmatch(session_id):
        raise ValueError("x_session_id must match the orchestrator request-key format")
    if not model or not isinstance(messages, list) or not messages:
        raise ValueError("lane B requires a model and non-empty messages")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer")
    return {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "x_tool_mode": "client",
        "x_session_id": session_id,
    }


def _validate_base_url(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("orchestrator URL must be an absolute http(s) URL")
    return value.rstrip("/")


def dry_run_plan(orchestrator: str, server_port: int) -> dict[str, Any]:
    """Return offline route and schema information, without placeholder prompts."""
    base = _validate_base_url(orchestrator)
    sid = run_session_id()
    lane_b_body_keys = sorted(lane_b_client_body(
        session_id=sid, model="orchestrator", messages=[{"role": "user", "content": "x"}]
    ))
    return {
        "mode": "dry-run; no requests sent",
        "orchestrator": base,
        "server": f"http://127.0.0.1:{server_port}",
        "lane_a": {
            "role": LANE_A,
            "url": f"{base}/v1/passthrough/{LANE_A}/chat/completions",
            "body_keys": ["chat_template_kwargs", "max_tokens", "messages"],
        },
        "lane_b": {
            "role": LANE_B,
            "url": f"{base}/v1/chat/completions",
            "mode": "client",
            "body_keys": lane_b_body_keys,
        },
        "lease_phase": {
            "prompt_tokens_approx": 24000,
            "lane_b_delay_s": 2.0,
            "max_tokens": 64,
            "stream": True,
        },
        "credit_phase": {
            "calls": 2,
            "lane": LANE_A,
            "relation": "second prompt extends the first prompt",
        },
    }


def _messages(value: Any, name: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty messages array")
    for index, message in enumerate(value):
        if not isinstance(message, dict) or message.get("role") not in {"system", "user", "assistant"}:
            raise ValueError(f"{name}[{index}] must have a supported role")
        if not isinstance(message.get("content"), (str, list)):
            raise ValueError(f"{name}[{index}].content must be text or content parts")
    return value


def load_inputs(path: str | Path) -> dict[str, Any]:
    """Load explicit prompt/config inputs; refuse oversized or incomplete files."""
    source = Path(path)
    raw = source.read_bytes()
    if len(raw) > 2 * 1024 * 1024:
        raise ValueError("input JSON exceeds 2 MiB")
    try:
        config = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid input JSON: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError("input JSON must be an object")
    required = {"lane_a_chat_template_kwargs", "lane_a_first_messages",
                "lane_b_messages", "lane_a_credit_messages", "lane_b_model"}
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"input JSON missing required fields: {', '.join(missing)}")
    ctk = config["lane_a_chat_template_kwargs"]
    if not isinstance(ctk, dict):
        raise ValueError("lane_a_chat_template_kwargs must be an object")
    model = config["lane_b_model"]
    if not isinstance(model, str) or not model.strip():
        raise ValueError("lane_b_model must be a non-empty string")
    return {
        "lane_a_chat_template_kwargs": ctk,
        "lane_a_first_messages": _messages(config["lane_a_first_messages"], "lane_a_first_messages"),
        "lane_b_messages": _messages(config["lane_b_messages"], "lane_b_messages"),
        "lane_a_credit_messages": _messages(config["lane_a_credit_messages"], "lane_a_credit_messages"),
        "lane_b_model": model,
    }


async def _send_bounded(client: httpx.AsyncClient, url: str, body: dict[str, Any], *,
                        max_response_bytes: int,
                        accepted: asyncio.Event | None = None) -> dict[str, Any]:
    chunks: list[bytes] = []
    total = 0
    async with client.stream("POST", url, json=body) as response:
        status = response.status_code
        if accepted is not None:
            accepted.set()
        async for chunk in response.aiter_bytes():
            total += len(chunk)
            if total > max_response_bytes:
                raise RuntimeError(f"response exceeded {max_response_bytes} byte limit")
            chunks.append(chunk)
    payload = b"".join(chunks)
    if not 200 <= status < 300:
        detail = payload[:512].decode("utf-8", errors="replace")
        raise RuntimeError(f"POST {url} returned HTTP {status}: {detail}")
    return {
        "status_code": status,
        "response_bytes": len(payload),
        "response_sha256": hashlib.sha256(payload).hexdigest(),
    }


async def dispatch_lane_b(client: httpx.AsyncClient, orchestrator: str, *,
                          model: str, messages: list[dict[str, Any]],
                          max_response_bytes: int = 1_000_000) -> tuple[str, dict[str, Any]]:
    """Send one actual lane-B request and return its generated session ID/result."""
    base = _validate_base_url(orchestrator)
    body = lane_b_client_body(session_id=run_session_id(), model=model, messages=messages,
                              max_tokens=64)
    body["stream"] = True
    result = await _send_bounded(
        client, f"{base}/v1/chat/completions", body,
        max_response_bytes=max_response_bytes,
    )
    return body["x_session_id"], result


async def execute_sequence(orchestrator: str, config: dict[str, Any], *,
                           delay_s: float = 2.0, max_response_bytes: int = 1_000_000,
                           client: httpx.AsyncClient | None = None) -> dict[str, Any]:
    """Dispatch the documented two-lane lease then same-lane credit sequence."""
    base = _validate_base_url(orchestrator)
    owned = client is None
    client = client or httpx.AsyncClient(timeout=httpx.Timeout(180.0))
    lane_a_url = f"{base}/v1/passthrough/{LANE_A}/chat/completions"
    def lane_a(messages: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "messages": messages,
            "max_tokens": 64,
            "chat_template_kwargs": config["lane_a_chat_template_kwargs"],
            "stream": True,
        }
    try:
        lane_a_accepted = asyncio.Event()

        async def send_first_lane_a() -> dict[str, Any]:
            try:
                return await _send_bounded(
                    client, lane_a_url, lane_a(config["lane_a_first_messages"]),
                    max_response_bytes=max_response_bytes, accepted=lane_a_accepted,
                )
            finally:
                # Wake the coordinator on early transport/status errors too; it
                # then awaits this task and propagates the original failure.
                lane_a_accepted.set()

        first_task = asyncio.create_task(send_first_lane_a())
        await lane_a_accepted.wait()
        if first_task.done():
            first = await first_task
            raise RuntimeError("lane A completed before the lane B lease overlap")
        await asyncio.sleep(delay_s)
        if first_task.done():
            await first_task
            raise RuntimeError("lane A completed before the lane B lease overlap")

        second_task = asyncio.create_task(dispatch_lane_b(
            client, base, model=config["lane_b_model"], messages=config["lane_b_messages"],
            max_response_bytes=max_response_bytes,
        ))
        try:
            first, (session_id, second) = await asyncio.gather(first_task, second_task)
        except BaseException:
            for task in (first_task, second_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(first_task, second_task, return_exceptions=True)
            raise
        credit = await _send_bounded(client, lane_a_url, lane_a(config["lane_a_credit_messages"]),
                                      max_response_bytes=max_response_bytes)
    finally:
        if owned:
            await client.aclose()
    return {
        "mode": "execute",
        "lane_b_session_id": session_id,
        "responses": {"lease_lane_a": first, "lease_lane_b": second,
                      "credit_lane_a": credit},
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--orchestrator", default=DEFAULT_ORCHESTRATOR)
    parser.add_argument("--server-port", type=int, default=DEFAULT_SERVER_PORT)
    parser.add_argument("--dry-run", action="store_true", help="print the offline request plan")
    parser.add_argument("--execute", action="store_true", help="send configured prompts to the orchestrator")
    parser.add_argument("--inputs", help="JSON containing the three real message arrays and lane settings")
    parser.add_argument("--delay-s", type=float, default=2.0)
    parser.add_argument("--max-response-bytes", type=int, default=1_000_000)
    args = parser.parse_args(argv)
    try:
        if args.execute:
            if args.dry_run:
                parser.error("choose either --dry-run or --execute")
            if not args.inputs:
                parser.error("--execute requires --inputs PATH")
            if not math.isfinite(args.delay_s) or args.delay_s < 0 or args.max_response_bytes <= 0:
                parser.error("delay and response-size limit must be non-negative/positive")
            config = load_inputs(args.inputs)
            result = asyncio.run(execute_sequence(
                args.orchestrator, config, delay_s=args.delay_s,
                max_response_bytes=args.max_response_bytes,
            ))
        else:
            if args.inputs:
                parser.error("--inputs is valid only with --execute")
            result = dry_run_plan(args.orchestrator, args.server_port)
    except (ValueError, OSError, httpx.HTTPError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI test
    raise SystemExit(main())
