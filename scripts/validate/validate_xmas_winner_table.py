#!/usr/bin/env python3
"""Validate X-MAS winner-table production readiness."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.classifiers.xmas_routing import load_winner_table  # noqa: E402

DEFAULT_CLASSIFIER_CONFIG = REPO_ROOT / "orchestration" / "classifier_config.yaml"


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except Exception as exc:
        raise ValueError(f"failed to parse {path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a mapping")
    return loaded


def _resolve_table_path(config_path: Path, raw_path: object) -> Path | None:
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = (config_path.parent.parent / path).resolve()
    return path


DOMAIN_PROXY_DERIVATION = "domain_winner_reused_for_function"


def validate_table(
    path: Path,
    *,
    require_evidence: bool = True,
    require_function_axis: bool = False,
) -> list[str]:
    """Return validation errors for one winner-table artifact."""
    try:
        table = load_winner_table(
            path,
            require_complete=True,
            require_evidence=require_evidence,
        )
    except Exception as exc:
        return [str(exc)]
    if (
        require_function_axis
        and table.provenance.get("derivation_mode") == DOMAIN_PROXY_DERIVATION
    ):
        return [
            "winner table uses domain-proxy evidence; mode=enforce requires "
            "true function-axis 5x5 sweep evidence"
        ]
    return []


def validate_config(config_path: Path) -> list[str]:
    """Return errors for classifier config X-MAS enforce readiness."""
    try:
        config = _load_yaml_mapping(config_path)
    except Exception as exc:
        return [str(exc)]

    raw = config.get("xmas_routing", {})
    if not isinstance(raw, dict):
        return ["xmas_routing must be a mapping"]

    mode = str(raw.get("mode", "off")).strip().lower()
    if mode != "enforce":
        return []

    table_path = _resolve_table_path(config_path, raw.get("winner_table_path"))
    if table_path is None:
        return ["xmas_routing.mode=enforce requires winner_table_path"]

    errors = validate_table(
        table_path,
        require_evidence=True,
        require_function_axis=True,
    )
    return [f"{table_path}: {error}" for error in errors]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CLASSIFIER_CONFIG,
        help="classifier_config.yaml path to inspect for mode=enforce readiness",
    )
    parser.add_argument(
        "--table",
        type=Path,
        help="validate one table artifact directly instead of reading config",
    )
    parser.add_argument(
        "--allow-unevidenced",
        action="store_true",
        help="only require a complete 5x5 table when validating --table directly",
    )
    parser.add_argument(
        "--require-function-axis",
        action="store_true",
        help=(
            "when validating --table directly, reject domain-proxy artifacts and "
            "require true 5x5 function-axis evidence"
        ),
    )
    args = parser.parse_args()

    if args.table:
        errors = validate_table(
            args.table,
            require_evidence=not args.allow_unevidenced,
            require_function_axis=args.require_function_axis,
        )
    else:
        errors = validate_config(args.config)

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print("X-MAS winner-table validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
