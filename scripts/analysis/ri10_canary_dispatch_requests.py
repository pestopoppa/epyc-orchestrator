#!/usr/bin/env python3
"""Dispatch prepared RI-10 canary payloads during a quiet window.

The companion request-plan script is deliberately dry-run only. This script is
the explicit live half: it posts those payloads to /chat, writes scorer-compatible
successful responses, and keeps transport/API failures in a separate artifact.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time
from typing import Any, Callable
import urllib.error
import urllib.request

DEFAULT_API_URL = "http://127.0.0.1:8000/chat"
DISPATCH_SCHEMA = "ri10_canary_dispatch.v1"

PostJson = Callable[[str, dict[str, Any], float], tuple[int, dict[str, Any], str]]


def _iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _post_json(url: str, payload: dict[str, Any], timeout_s: float) -> tuple[int, dict[str, Any], str]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            raw = response.read().decode("utf-8", errors="replace")
            return response.status, json.loads(raw) if raw else {}, raw
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            parsed = {}
        return exc.code, parsed, raw


def autopilot_pids() -> list[str]:
    result = subprocess.run(
        ["pgrep", "-f", "scripts/autopilot/autopilot.py start|autopilot.py start"],
        capture_output=True,
        text=True,
        check=False,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _request_id(payload: dict[str, Any]) -> str:
    request_id = str(payload.get("request_id") or "").strip()
    if not request_id:
        raise ValueError("payload missing request_id")
    return request_id


def _timeout_for(payload: dict[str, Any], padding_s: float) -> float:
    timeout = payload.get("timeout_s")
    try:
        base = float(timeout)
    except (TypeError, ValueError):
        base = 180.0
    return max(1.0, base + padding_s)


def _success_row(
    payload: dict[str, Any],
    response: dict[str, Any],
    *,
    status_code: int,
    elapsed_s: float,
) -> dict[str, Any]:
    answer = str(response.get("answer") or "").strip()
    if not answer:
        raise ValueError(f"{_request_id(payload)}: successful response missing answer")
    return {
        "request_id": _request_id(payload),
        "role": payload.get("force_role"),
        "response": answer,
        "status_code": status_code,
        "elapsed_s": round(elapsed_s, 3),
        "routed_to": response.get("routed_to"),
        "mode": response.get("mode"),
        "tokens_used": response.get("tokens_used"),
        "tokens_generated": response.get("tokens_generated"),
        "predicted_tps": response.get("predicted_tps"),
        "factual_risk_band": response.get("factual_risk_band"),
        "factual_risk_score": response.get("factual_risk_score"),
        "xmas_meta": response.get("xmas_meta"),
    }


def _failure_row(
    payload: dict[str, Any],
    *,
    status_code: int | None,
    elapsed_s: float,
    error: str,
    response: dict[str, Any] | None = None,
    raw: str = "",
) -> dict[str, Any]:
    return {
        "request_id": _request_id(payload),
        "role": payload.get("force_role"),
        "status_code": status_code,
        "elapsed_s": round(elapsed_s, 3),
        "error": error,
        "response_error_code": (response or {}).get("error_code"),
        "response_error_detail": (response or {}).get("error_detail"),
        "raw_response_prefix": raw[:500],
    }


def dispatch_payloads(
    payloads: list[dict[str, Any]],
    *,
    api_url: str = DEFAULT_API_URL,
    post_json: PostJson = _post_json,
    timeout_padding_s: float = 30.0,
    sleep_s: float = 0.0,
    limit: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    selected = payloads[:limit] if limit is not None else payloads
    successes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    started_at = _iso_now()
    for index, payload in enumerate(selected, start=1):
        _request_id(payload)
        timeout_s = _timeout_for(payload, timeout_padding_s)
        start = time.monotonic()
        try:
            status_code, response, raw = post_json(api_url, payload, timeout_s)
            elapsed_s = time.monotonic() - start
            if status_code >= 400 or response.get("error_code"):
                failures.append(
                    _failure_row(
                        payload,
                        status_code=status_code,
                        elapsed_s=elapsed_s,
                        error=f"http_status_{status_code}",
                        response=response,
                        raw=raw,
                    )
                )
            else:
                successes.append(
                    _success_row(
                        payload,
                        response,
                        status_code=status_code,
                        elapsed_s=elapsed_s,
                    )
                )
        except Exception as exc:
            elapsed_s = time.monotonic() - start
            failures.append(
                _failure_row(
                    payload,
                    status_code=None,
                    elapsed_s=elapsed_s,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
        if sleep_s > 0 and index < len(selected):
            time.sleep(sleep_s)
    summary = {
        "schema_version": DISPATCH_SCHEMA,
        "started_at": started_at,
        "finished_at": _iso_now(),
        "api_url": api_url,
        "requested": len(selected),
        "succeeded": len(successes),
        "failed": len(failures),
        "status": "ready" if not failures and len(successes) == len(selected) else "failures",
    }
    return successes, failures, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payloads-jsonl", type=Path, required=True)
    parser.add_argument("--responses-jsonl", type=Path, required=True)
    parser.add_argument("--failures-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--timeout-padding-s", type=float, default=30.0)
    parser.add_argument("--sleep-s", type=float, default=0.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--host-quiet-confirmed",
        action="store_true",
        help="Required for live dispatch; asserts the operator has reserved a quiet window.",
    )
    parser.add_argument(
        "--allow-autopilot-active",
        action="store_true",
        help="Bypass the AutoPilot process guard. Intended only for explicit operator experiments.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.host_quiet_confirmed:
        raise SystemExit("refusing live dispatch without --host-quiet-confirmed")
    pids = autopilot_pids()
    if pids and not args.allow_autopilot_active:
        raise SystemExit(f"refusing live dispatch while AutoPilot is active: {', '.join(pids)}")
    payloads = load_jsonl(args.payloads_jsonl)
    successes, failures, summary = dispatch_payloads(
        payloads,
        api_url=args.api_url,
        timeout_padding_s=args.timeout_padding_s,
        sleep_s=args.sleep_s,
        limit=args.limit,
    )
    write_jsonl(successes, args.responses_jsonl)
    write_jsonl(failures, args.failures_jsonl)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
