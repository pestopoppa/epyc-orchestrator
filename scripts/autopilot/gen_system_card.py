#!/usr/bin/env python3
"""Generate the AutoPilot controller system card from live repository state."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = SCRIPT_DIR.parents[1]

TRANSLATION = str.maketrans(
    {
        "—": "-",
        "–": "-",
        "→": "->",
        "×": "x",
        "≥": ">=",
        "≤": "<=",
        "≈": "~",
        "σ": "sigma",
        "Δ": "Delta",
    }
)

TRUST_BOUNDARY_FILES = [
    "scripts/benchmark/seed_specialist_routing.py",
    "scripts/benchmark/debug_scorer.py",
    "scripts/benchmark/dataset_adapters.py",
    "scripts/benchmark/question_pool.py",
    "benchmarks/prompts/question_pool.jsonl",
    "scripts/autopilot/safety_gate.py",
    "scripts/autopilot/eval_tower.py",
]


def _clean(value: Any, max_len: int | None = None) -> str:
    text = str(value if value is not None else "")
    text = text.translate(TRANSLATION)
    text = " ".join(text.split())
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.replace("|", "/")
    if max_len is not None and len(text) > max_len:
        return text[: max_len - 3].rstrip() + "..."
    return text


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        return yaml.safe_load(path.read_text()) or {}
    except OSError:
        return {}


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _nested(data: dict[str, Any], *keys: str) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _port_from_url(url: Any) -> str:
    if not url:
        return ""
    match = re.search(r":(\d+)(?:/|$)", str(url))
    return match.group(1) if match else ""


def _format_number(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return _clean(value) or "n/a"
    return f"{number:.3f}".rstrip("0").rstrip(".")


def _format_acceleration(config: Any) -> str:
    if not isinstance(config, dict):
        return "none"
    raw_type = config.get("type") or config.get("spec_type") or "none"
    accel_type = _clean(raw_type, 32)
    details: list[str] = []
    if "experts" in config:
        details.append(f"experts={config['experts']}")
    if config.get("lookup") is not None:
        details.append(f"lookup={str(config.get('lookup')).lower()}")
    if "draft_max" in config:
        details.append(f"draft_max={config['draft_max']}")
    if "spec_type" in config and config.get("spec_type") != raw_type:
        details.append(f"spec={config['spec_type']}")
    return accel_type if not details else f"{accel_type} ({', '.join(details)})"


def _format_launch_requirements(requirements: Any) -> str:
    if not isinstance(requirements, dict):
        return "none"
    parts: list[str] = []
    mmproj_path = requirements.get("mmproj_path")
    if mmproj_path:
        parts.append(f"mmproj={Path(str(mmproj_path)).name}")
    draft_path = requirements.get("draft_model_path")
    if draft_path:
        parts.append(f"draft={Path(str(draft_path)).name}")
    return ", ".join(parts) if parts else "none"


def _stack_prior_role_rows(stack_priors: dict[str, Any]) -> list[str]:
    roles = stack_priors.get("roles")
    if not isinstance(roles, dict):
        return []

    rows: list[str] = []
    for name, record in sorted(roles.items()):
        if not isinstance(record, dict) or record.get("deployment_status") != "live_stack":
            continue
        serving = record.get("serving") if isinstance(record.get("serving"), dict) else {}
        priors = record.get("priors") if isinstance(record.get("priors"), dict) else {}
        ports = serving.get("ports")
        port_values = [str(port) for port in ports if isinstance(port, int)] if isinstance(ports, list) else []
        if not port_values:
            endpoint_port = _port_from_url(serving.get("endpoint"))
            if endpoint_port:
                port_values.append(endpoint_port)
        if not port_values:
            continue
        launch = serving.get("launch") if isinstance(serving.get("launch"), dict) else {}
        description = (
            f"{_clean(record.get('deployment_status'), 32)}; "
            f"binding={_clean(serving.get('binding'), 48)}; "
            f"status={_clean(record.get('status'), 32)}"
        )
        rows.append(
            "| "
            + " | ".join(
                [
                    _clean(name, 28),
                    _clean(", ".join(port_values), 28),
                    _clean(record.get("display_name") or record.get("model_id") or "unknown", 44),
                    _clean(serving.get("tier") or "n/a", 12),
                    _clean(_format_acceleration(record.get("acceleration")), 48),
                    _clean(_format_launch_requirements(launch.get("requirements")), 48),
                    _clean(_format_number(priors.get("throughput_tps")), 18),
                    _clean(description, 96),
                ]
            )
            + " |"
        )
    return rows


def _registry_role_rows(registry: dict[str, Any]) -> list[str]:
    server_mode = registry.get("server_mode") or {}
    roles = registry.get("roles") or {}
    if not isinstance(server_mode, dict) or not isinstance(roles, dict):
        return []

    rows: list[str] = []
    for name in sorted(server_mode):
        if name == "dev":
            continue
        server_cfg = server_mode.get(name) or {}
        role_cfg = roles.get(name) or {}
        if not isinstance(server_cfg, dict) or not isinstance(role_cfg, dict):
            continue
        model_type = server_cfg.get("model_type")
        registry_role = name in roles
        hot_local_alias = server_cfg.get("tier") == "hot" and model_type is None
        if not registry_role and not hot_local_alias:
            continue
        backend_type = _nested(role_cfg, "backend", "type") or "local"
        if backend_type != "local":
            continue
        port = server_cfg.get("port") or _port_from_url(server_cfg.get("url"))
        if not port:
            continue
        model = (
            server_cfg.get("model")
            or _nested(role_cfg, "model", "name")
            or server_cfg.get("model_role")
            or "unknown"
        )
        tier = server_cfg.get("tier") or _nested(role_cfg, "memory", "residency") or "n/a"
        accel = _format_acceleration(
            server_cfg.get("acceleration") or role_cfg.get("acceleration")
        )
        throughput = (
            server_cfg.get("throughput")
            or _nested(role_cfg, "performance", "optimized_tps")
            or _nested(role_cfg, "performance", "baseline_tps")
            or "n/a"
        )
        description = server_cfg.get("description") or role_cfg.get("description") or ""
        rows.append(
            "| "
            + " | ".join(
                [
                    _clean(name, 28),
                    _clean(port, 12),
                    _clean(model, 44),
                    _clean(tier, 12),
                    _clean(accel, 48),
                    "none",
                    _clean(throughput, 18),
                    _clean(description, 96),
                ]
            )
            + " |"
        )
    return rows


def _baseline_payload(
    root: Path,
    state_override: dict[str, Any] | None,
) -> tuple[dict[str, Any], str]:
    baseline_yaml = _load_yaml(root / "orchestration" / "autopilot_baseline.yaml")
    state = state_override if state_override is not None else _load_json(
        root / "orchestration" / "autopilot_state.json"
    )
    state_baseline = state.get("baseline_state") if isinstance(state, dict) else None
    if isinstance(state_baseline, dict) and state_baseline:
        return state_baseline, "orchestration/autopilot_state.json:baseline_state"
    return baseline_yaml, "orchestration/autopilot_baseline.yaml"


def _dict_for_tier(data: dict[str, Any], key: str, tier: int) -> dict[str, Any]:
    by_tier = data.get(key)
    if not isinstance(by_tier, dict):
        return {}
    payload = by_tier.get(str(tier))
    if payload is None:
        payload = by_tier.get(tier)
    return payload if isinstance(payload, dict) else {}


def _tier_sort_key(item: tuple[Any, Any]) -> int:
    try:
        return int(item[0])
    except (TypeError, ValueError):
        return 999


def _tier_baseline_lines(baseline: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    tier_values = baseline.get("baselines_by_tier") or {}
    if isinstance(tier_values, dict):
        for tier, quality in sorted(tier_values.items(), key=_tier_sort_key):
            counts = _dict_for_tier(baseline, "per_suite_counts_by_tier", int(tier))
            suite_count = len(_dict_for_tier(
                baseline,
                "per_suite_quality_by_tier",
                int(tier),
            ))
            count_note = f", {len(counts)} suites with counts" if counts else ""
            lines.append(
                f"- T{tier}: quality baseline {_format_number(quality)} "
                f"({suite_count} suites{count_note})"
            )
    if not lines and baseline.get("quality") is not None:
        lines.append(f"- Legacy flat quality baseline: {_format_number(baseline['quality'])}")
    return lines or ["- No baseline data found."]


def _active_suite_line(baseline: dict[str, Any]) -> str:
    suites = _dict_for_tier(baseline, "per_suite_quality_by_tier", 1)
    if not suites:
        suites = baseline.get("per_suite_quality") or {}
    suite_names = sorted(str(name) for name in suites if name)
    if not suite_names:
        return "- Active suites: not found in baseline state."
    return "- Active T1 suites: " + ", ".join(suite_names)


def _runtime_state_lines(state: dict[str, Any]) -> list[str]:
    if not state:
        return ["- AutoPilot state file not found or unreadable."]
    lines = [
        f"- paused: {str(bool(state.get('paused'))).lower()}",
        f"- trial_counter: {state.get('trial_counter', 'unknown')}",
    ]
    pause_reason = state.get("pause_reason")
    if state.get("paused") and pause_reason:
        lines.append(f"- pause_reason: {_clean(pause_reason, 180)}")
    in_flight = state.get("in_flight_trial")
    if isinstance(in_flight, dict):
        action = in_flight.get("action") or {}
        action_type = action.get("type") if isinstance(action, dict) else None
        lines.append(
            "- in_flight_trial: "
            f"{in_flight.get('trial_id', 'unknown')} "
            f"({action_type or 'unknown action'})"
        )
    last_invalid = state.get("last_invalid_reason")
    if last_invalid:
        lines.append(f"- last_invalid_reason: {_clean(last_invalid, 180)}")
    return lines


def _tier_spec_lines(root: Path) -> list[str]:
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from src.autopilot_core.tier_specs import (  # type: ignore
            DEFAULT_FRONTIER_TIER,
            LEGACY_OBJECTIVE_POLICY,
            MIN_FRONTIER_EVAL_TIER,
            TASK_RATE_OBJECTIVE_POLICY,
            TIER_SPECS,
        )
    except Exception as exc:
        msg = _clean(f"{type(exc).__name__}: {exc}")
        return [f"- Tier spec import failed: {msg}"]

    lines = [
        f"- minimum frontier tier: T{MIN_FRONTIER_EVAL_TIER}",
        f"- default frontier tier: T{DEFAULT_FRONTIER_TIER}",
        f"- legacy objective policy: {LEGACY_OBJECTIVE_POLICY}",
        f"- task-rate shadow policy: {TASK_RATE_OBJECTIVE_POLICY}",
    ]
    for tier, spec in sorted(TIER_SPECS.items()):
        lines.append(f"- T{tier}: {_clean(spec.label)}")
    return lines


def generate_system_card(
    root: Path | str = DEFAULT_ROOT,
    *,
    state_override: dict[str, Any] | None = None,
) -> str:
    """Return the current AutoPilot system card as Markdown."""
    root_path = Path(root)
    registry = _load_yaml(root_path / "orchestration" / "model_registry.yaml")
    stack_priors = _load_yaml(root_path / "orchestration" / "derived" / "stack_priors.yaml")
    state = state_override if state_override is not None else _load_json(
        root_path / "orchestration" / "autopilot_state.json"
    )
    baseline, baseline_source = _baseline_payload(root_path, state_override)
    role_source = "orchestration/derived/stack_priors.yaml"
    role_rows = _stack_prior_role_rows(stack_priors)
    if not role_rows:
        role_source = "orchestration/model_registry.yaml (degraded fallback)"
        role_rows = _registry_role_rows(registry)
    active_role_names = {
        row.split("|")[1].strip()
        for row in role_rows
        if row.startswith("|") and not row.startswith("| Role")
    }

    lines: list[str] = [
        "# AutoPilot Generated System Card",
        "",
        "Generated from repository files at controller-prompt assembly time.",
        "Do not hand-edit this file; edit the source registries or constitution.",
        "",
        "## Runtime State",
        "",
        *_runtime_state_lines(state),
        "",
        "## Active Model-Serving Roles",
        "",
        f"- Source: {role_source}",
        "",
    ]
    if role_rows:
        lines.extend(
            [
                "| Role | Port | Model | Tier | Acceleration | Requirements | Throughput | Description |",
                "|---|---:|---|---|---|---|---:|---|",
                *role_rows,
            ]
        )
    else:
        lines.append("- No active local server roles found in generated stack priors or registry.")
    legacy_architect_role = "architect" "_coding"
    if legacy_architect_role not in active_role_names:
        lines.extend(
            [
                "",
                f"- {legacy_architect_role} is not an active server role in stack priors; "
                "do not target it as a live role or port.",
            ]
        )

    lines.extend(
        [
            "",
            "## Evaluation Instrument",
            "",
            *_tier_spec_lines(root_path),
            _active_suite_line(baseline),
            "",
            "## Baselines",
            "",
            f"- Source: {baseline_source}",
            *_tier_baseline_lines(baseline),
            "",
            "## Eval Trust Boundary",
            "",
            "These files are measurement/trust-boundary surfaces, not autonomous "
            "experiment knobs:",
        ]
    )
    lines.extend(f"- `{path}`" for path in TRUST_BOUNDARY_FILES)
    lines.extend(
        [
            "",
            "## Generated-Card Rules",
            "",
            "- Runtime facts in this card supersede old handoffs, memories, and "
            "program text.",
            "- If this card contradicts an action idea, skip or choose an "
            "observational action.",
            "- If this card is missing a role, port, suite, or flag, do not invent it.",
            "- Regenerate this card after registry, baseline, tier-spec, or "
            "autopilot-state changes.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def write_system_card(root: Path | str = DEFAULT_ROOT, output: Path | None = None) -> Path:
    root_path = Path(root)
    out_path = output or (root_path / "scripts" / "autopilot" / "system_card.md")
    out_path.write_text(generate_system_card(root_path))
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--stdout", action="store_true")
    args = parser.parse_args(argv)

    text = generate_system_card(args.root)
    output = args.output or (args.root / "scripts" / "autopilot" / "system_card.md")
    if args.stdout:
        print(text, end="")
        return 0
    if args.check:
        try:
            current = output.read_text()
        except OSError:
            current = ""
        if current != text:
            print(f"{output} is stale; run gen_system_card.py", file=sys.stderr)
            return 1
        return 0
    output.write_text(text)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
