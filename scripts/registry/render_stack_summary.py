#!/usr/bin/env python3
"""Render operator-facing stack summaries from generated stack priors."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import yaml

DEFAULT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STACK_PRIORS = DEFAULT_ROOT / "orchestration" / "derived" / "stack_priors.yaml"
DEFAULT_REGISTRY = DEFAULT_ROOT / "orchestration" / "model_registry.yaml"
DEFAULT_OUTPUT = DEFAULT_ROOT / "docs" / "generated" / "current_stack_summary.md"

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


def clean_cell(value: Any, max_len: int | None = None) -> str:
    text = str(value if value is not None else "")
    text = text.translate(TRANSLATION)
    text = " ".join(text.split())
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.replace("|", "/")
    if max_len is not None and len(text) > max_len:
        return text[: max_len - 3].rstrip() + "..."
    return text


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except OSError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def nested(data: dict[str, Any], *keys: str) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def port_from_url(url: Any) -> str:
    if not url:
        return ""
    match = re.search(r":(\d+)(?:/|$)", str(url))
    return match.group(1) if match else ""


def format_number(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return clean_cell(value) or "n/a"
    return f"{number:.3f}".rstrip("0").rstrip(".")


def format_acceleration(config: Any) -> str:
    if not isinstance(config, dict):
        return "none"
    raw_type = config.get("type") or config.get("spec_type") or "none"
    accel_type = clean_cell(raw_type, 32)
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


def format_launch_requirements(requirements: Any) -> str:
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


def stack_prior_role_rows(stack_priors: dict[str, Any]) -> list[str]:
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
            endpoint_port = port_from_url(serving.get("endpoint"))
            if endpoint_port:
                port_values.append(endpoint_port)
        if not port_values:
            continue
        launch = serving.get("launch") if isinstance(serving.get("launch"), dict) else {}
        description = (
            f"{clean_cell(record.get('deployment_status'), 32)}; "
            f"binding={clean_cell(serving.get('binding'), 48)}; "
            f"status={clean_cell(record.get('status'), 32)}"
        )
        rows.append(
            "| "
            + " | ".join(
                [
                    clean_cell(name, 28),
                    clean_cell(", ".join(port_values), 28),
                    clean_cell(record.get("display_name") or record.get("model_id") or "unknown", 44),
                    clean_cell(serving.get("tier") or "n/a", 12),
                    clean_cell(format_acceleration(record.get("acceleration")), 48),
                    clean_cell(format_launch_requirements(launch.get("requirements")), 48),
                    clean_cell(format_number(priors.get("throughput_tps")), 18),
                    clean_cell(description, 96),
                ]
            )
            + " |"
        )
    return rows


def registry_role_rows(registry: dict[str, Any]) -> list[str]:
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
        backend_type = nested(role_cfg, "backend", "type") or "local"
        if backend_type != "local":
            continue
        port = server_cfg.get("port") or port_from_url(server_cfg.get("url"))
        if not port:
            continue
        model = (
            server_cfg.get("model")
            or nested(role_cfg, "model", "name")
            or server_cfg.get("model_role")
            or "unknown"
        )
        tier = server_cfg.get("tier") or nested(role_cfg, "memory", "residency") or "n/a"
        accel = format_acceleration(server_cfg.get("acceleration") or role_cfg.get("acceleration"))
        throughput = (
            server_cfg.get("throughput")
            or nested(role_cfg, "performance", "optimized_tps")
            or nested(role_cfg, "performance", "baseline_tps")
            or "n/a"
        )
        description = server_cfg.get("description") or role_cfg.get("description") or ""
        rows.append(
            "| "
            + " | ".join(
                [
                    clean_cell(name, 28),
                    clean_cell(port, 12),
                    clean_cell(model, 44),
                    clean_cell(tier, 12),
                    clean_cell(accel, 48),
                    "none",
                    clean_cell(throughput, 18),
                    clean_cell(description, 96),
                ]
            )
            + " |"
        )
    return rows


def render_role_table(rows: list[str]) -> list[str]:
    if not rows:
        return ["No active local server roles found."]
    return [
        "| Role | Port | Model | Tier | Acceleration | Requirements | Throughput | Description |",
        "|---|---:|---|---|---|---|---:|---|",
        *rows,
    ]


def render_current_stack_summary(
    *,
    stack_priors_path: Path = DEFAULT_STACK_PRIORS,
    registry_path: Path = DEFAULT_REGISTRY,
) -> str:
    stack_priors = load_yaml(stack_priors_path)
    registry = load_yaml(registry_path)
    source = "orchestration/derived/stack_priors.yaml"
    rows = stack_prior_role_rows(stack_priors)
    if not rows:
        source = "orchestration/model_registry.yaml (degraded fallback)"
        rows = registry_role_rows(registry)

    lines = [
        "# Current Stack Summary",
        "",
        "Generated from structured stack truth. Do not hand-edit this file; run:",
        "",
        "```bash",
        "uv run python scripts/registry/stack_change_pipeline.py update",
        "```",
        "",
        f"Source: `{source}`",
        "",
        *render_role_table(rows),
    ]
    return "\n".join(lines).rstrip() + "\n"


def write_current_stack_summary(
    output: Path = DEFAULT_OUTPUT,
    *,
    stack_priors_path: Path = DEFAULT_STACK_PRIORS,
    registry_path: Path = DEFAULT_REGISTRY,
) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        render_current_stack_summary(
            stack_priors_path=stack_priors_path,
            registry_path=registry_path,
        ),
        encoding="utf-8",
    )
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stack-priors", type=Path, default=DEFAULT_STACK_PRIORS)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--stdout", action="store_true")
    args = parser.parse_args(argv)

    text = render_current_stack_summary(
        stack_priors_path=args.stack_priors,
        registry_path=args.registry,
    )
    if args.stdout:
        print(text, end="")
        return 0
    if args.check:
        try:
            current = args.output.read_text(encoding="utf-8")
        except OSError:
            current = ""
        if current != text:
            print(f"{args.output} is stale; run stack_change_pipeline.py update")
            return 1
        return 0
    write_current_stack_summary(
        args.output,
        stack_priors_path=args.stack_priors,
        registry_path=args.registry,
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
