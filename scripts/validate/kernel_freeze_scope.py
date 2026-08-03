#!/usr/bin/env python3
"""Derive the regression scope for freezing a new production kernel.

Freezing a kernel needs one question answered: *which models must show no
regression before this kernel may serve?* The answer is not a curated list — a
curated list goes stale the moment a role is repointed. It is derivable: the
models that matter for backend B are exactly the models whose roles resolve to
backend B in the compiled stack priors.

That is what makes the four kernels (cpu / gpu / stt / tts) independently
upgradable. A whisper.cpp upgrade cannot regress a role that never calls it, so
it should not be gated on one.

Usage:
    python scripts/validate/kernel_freeze_scope.py            # all backends
    python scripts/validate/kernel_freeze_scope.py --backend gpu
    python scripts/validate/kernel_freeze_scope.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import yaml  # noqa: E402

DEFAULT_PRIORS = REPO_ROOT / "orchestration" / "derived" / "stack_priors.yaml"


def _backend_of(binary_path: str) -> str:
    """Classify a resolved binary path back to its backend."""
    p = binary_path or ""
    if "build-hip" in p:
        return "gpu"
    if "whisper" in p:
        return "stt"
    if "qwentts" in p:
        return "tts"
    return "cpu"


def freeze_scope(priors_path: Path = DEFAULT_PRIORS) -> dict[str, list[dict]]:
    """Return {backend: [ {role, model, binary_path, context_tokens, spec} ]}."""
    priors = yaml.safe_load(priors_path.read_text()) or {}
    scope: dict[str, list[dict]] = {}
    for role, record in sorted((priors.get("roles") or {}).items()):
        serving = record.get("serving") if isinstance(record, dict) else None
        if not isinstance(serving, dict):
            continue
        launch = serving.get("launch")
        if not isinstance(launch, dict):
            continue
        runtime = launch.get("runtime") or {}
        requirements = launch.get("requirements") or {}
        binary_path = runtime.get("binary_path")
        if not isinstance(binary_path, str) or not binary_path:
            continue
        flags = runtime.get("flags") or {}
        spec = flags.get("spec") if isinstance(flags, dict) else {}
        cache = runtime.get("cache") or {}
        scope.setdefault(_backend_of(binary_path), []).append(
            {
                "role": role,
                "model_path": requirements.get("model_path"),
                "binary_path": binary_path,
                "context_tokens": cache.get("context_tokens"),
                "spec_type": (spec or {}).get("type"),
            }
        )
    return scope


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", choices=("cpu", "gpu", "stt", "tts"))
    ap.add_argument("--priors", type=Path, default=DEFAULT_PRIORS)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    scope = freeze_scope(args.priors)
    if args.backend:
        scope = {args.backend: scope.get(args.backend, [])}

    if args.json:
        print(json.dumps(scope, indent=2))
        return 0

    for backend, rows in sorted(scope.items()):
        print(f"\n=== backend {backend}: {len(rows)} role(s) must show no regression ===")
        if not rows:
            print("  (none — a kernel serving no role needs no regression gate)")
        seen_models: set[str] = set()
        for row in rows:
            model = (row["model_path"] or "?").split("/")[-1]
            marker = " " if model in seen_models else "*"
            seen_models.add(model)
            print(
                f"  {marker} {row['role']:20s} {model[:52]:52s}"
                f" ctx={row['context_tokens']} spec={row['spec_type']}"
            )
        print(f"  ({len(seen_models)} distinct model(s); '*' marks first appearance)")
    print(
        "\nGate: bench each distinct model at its declared shape and recipe on the "
        "CANDIDATE kernel vs the frozen production one, per MEASUREMENT.md. Pair every "
        "speed number with a correctness check. Any regression outside tolerance blocks "
        "the freeze for THAT backend only."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
