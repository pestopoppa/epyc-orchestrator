#!/usr/bin/env python3
"""Build deterministic RI-10 factual-risk canary request payloads.

This script is intentionally dry-run only: it prepares request IDs whose
hashes land in the requested canary arms, but it never dispatches inference.
Use it to stage a controlled quiet-window campaign after confirming the API is
running code that samples RI-10 canary arms from caller request_id.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
from typing import Any, Iterable

import yaml

from src.classifiers.factual_risk import assess_risk, get_mode

SCRIPT_PATH = Path(__file__).resolve()
ORCH_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_CLASSIFIER_CONFIG = ORCH_ROOT / "orchestration" / "classifier_config.yaml"
DEFAULT_API_URL = "http://127.0.0.1:8000/chat"
DEFAULT_PROMPT = (
    "CEO of OpenAI is correct source. Tesla was founded in 2003. "
    "Fact is true. Fact is correct. Evidence is proof. "
    "Statistic is percentage. Population is number of country. "
    "Language is currency. Location is accurate."
)
DEFAULT_ROLES = ("frontdoor", "worker_general", "worker_vision")
ARMS = ("enforce", "shadow")


def _iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_factual_risk_config(path: Path = DEFAULT_CLASSIFIER_CONFIG) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} did not contain a mapping")
    factual = loaded.get("factual_risk") or {}
    if not isinstance(factual, dict):
        raise ValueError(f"{path} did not contain factual_risk mapping")
    return factual


def configured_canary_roles(config: dict[str, Any]) -> list[str]:
    roles = config.get("canary_roles") or []
    if not isinstance(roles, list):
        raise ValueError("factual_risk.canary_roles must be a list")
    return [str(role) for role in roles if str(role)]


def _clean_role(role: str) -> str:
    return "".join(ch if ch.isalnum() else "-" for ch in role).strip("-")


def _request_id(role: str, arm: str, index: int, candidate: int) -> str:
    return f"ri10-{_clean_role(role)}-{arm}-{index:03d}-{candidate:05d}"


def _payload(
    *,
    role: str,
    request_id: str,
    prompt: str,
    max_tokens: int,
    timeout_s: int,
    max_queue_wait_ms: int,
) -> dict[str, Any]:
    return {
        "prompt": prompt,
        "mock_mode": False,
        "real_mode": True,
        "force_role": role,
        "force_mode": "direct",
        "request_id": request_id,
        "request_priority": "background",
        "workload_class": "campaign",
        "max_turns": 1,
        "max_tokens": max_tokens,
        "timeout_s": timeout_s,
        "max_queue_wait_ms": max_queue_wait_ms,
    }


def _risk_summary(prompt: str, roles: Iterable[str], config: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for role in roles:
        result = assess_risk(prompt, role=role, config=config)
        out[role] = {
            "risk_score": result.risk_score,
            "adjusted_risk_score": result.adjusted_risk_score,
            "risk_band": result.risk_band,
            "role_adjustment": result.role_adjustment,
            "features": result.risk_features,
        }
    return out


def build_plan(
    *,
    config_path: Path = DEFAULT_CLASSIFIER_CONFIG,
    roles: Iterable[str] | None = None,
    per_role_per_arm: int = 10,
    prompt: str = DEFAULT_PROMPT,
    max_candidates: int = 100_000,
    max_tokens: int = 32,
    timeout_s: int = 180,
    max_queue_wait_ms: int = 90_000,
) -> dict[str, Any]:
    config = load_factual_risk_config(config_path)
    mode = str(config.get("mode") or "")
    if mode != "canary":
        raise ValueError(f"factual_risk.mode must be canary, got {mode!r}")

    selected_roles = list(roles or configured_canary_roles(config) or DEFAULT_ROLES)
    canary_roles = set(configured_canary_roles(config))
    if canary_roles:
        nonparticipants = [role for role in selected_roles if role not in canary_roles]
        if nonparticipants:
            raise ValueError(
                "requested role(s) are outside factual_risk.canary_roles: "
                + ", ".join(nonparticipants)
            )

    risk = _risk_summary(prompt, selected_roles, config)
    not_high = [role for role, entry in risk.items() if entry["risk_band"] != "high"]
    if not_high:
        raise ValueError(
            "prompt is not high factual-risk for role(s): " + ", ".join(not_high)
        )

    requests: list[dict[str, Any]] = []
    for role in selected_roles:
        for arm in ARMS:
            found = 0
            candidate = 0
            while found < per_role_per_arm and candidate < max_candidates:
                candidate += 1
                request_id = _request_id(role, arm, found + 1, candidate)
                if get_mode(config, role=role, sample_key=request_id) != arm:
                    continue
                found += 1
                requests.append(
                    {
                        "role": role,
                        "expected_factual_risk_mode": arm,
                        "request_id": request_id,
                        "prompt": prompt,
                        "payload": _payload(
                            role=role,
                            request_id=request_id,
                            prompt=prompt,
                            max_tokens=max_tokens,
                            timeout_s=timeout_s,
                            max_queue_wait_ms=max_queue_wait_ms,
                        ),
                    }
                )
            if found < per_role_per_arm:
                raise RuntimeError(
                    f"only found {found}/{per_role_per_arm} {arm} ids for {role} "
                    f"within {max_candidates} candidates"
                )

    return {
        "generated_at": _iso_now(),
        "config_path": str(config_path),
        "factual_risk_mode": mode,
        "canary_ratio": float(config.get("canary_ratio", 0.25)),
        "canary_roles": configured_canary_roles(config),
        "selected_roles": selected_roles,
        "per_role_per_arm": per_role_per_arm,
        "request_count": len(requests),
        "prompt_risk": risk,
        "requests": requests,
    }


def _render_jsonl(plan: dict[str, Any]) -> str:
    return "\n".join(json.dumps(item["payload"], sort_keys=True) for item in plan["requests"]) + "\n"


def _render_curl(plan: dict[str, Any], api_url: str) -> str:
    lines = []
    for item in plan["requests"]:
        payload = json.dumps(item["payload"], sort_keys=True)
        lines.append(
            "curl -fsS -X POST "
            + shlex.quote(api_url)
            + " -H 'Content-Type: application/json' -d "
            + shlex.quote(payload)
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CLASSIFIER_CONFIG)
    parser.add_argument(
        "--role",
        action="append",
        dest="roles",
        help="Role to include; repeat for multiple. Default: configured canary_roles.",
    )
    parser.add_argument("--per-role-per-arm", type=int, default=10)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-candidates", type=int, default=100_000)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--timeout-s", type=int, default=180)
    parser.add_argument("--max-queue-wait-ms", type=int, default=90_000)
    parser.add_argument("--format", choices=("json", "jsonl", "curl"), default="json")
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.per_role_per_arm < 1:
        raise SystemExit("--per-role-per-arm must be positive")

    plan = build_plan(
        config_path=args.config,
        roles=args.roles,
        per_role_per_arm=args.per_role_per_arm,
        prompt=args.prompt,
        max_candidates=args.max_candidates,
        max_tokens=args.max_tokens,
        timeout_s=args.timeout_s,
        max_queue_wait_ms=args.max_queue_wait_ms,
    )
    if args.format == "jsonl":
        rendered = _render_jsonl(plan)
    elif args.format == "curl":
        rendered = _render_curl(plan, args.api_url)
    else:
        rendered = json.dumps(plan, indent=2, sort_keys=True) + "\n"

    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
