"""RTG-55 MHS-5 — Harness-R1 held-out patches as a labelled CONSTRAIN/REPLACE corpus.

Source: ``github.com/DeepExperience/Harness-R1`` ``examples/heldout_generalization/``
(Apache-2.0). Three editors (Harness-R1, Qwen3.5-397B, DeepSeek-V4-Pro) × 3 evidence
seeds × 3 benchmarks. Each patch was applied to 1,270 held-out tasks against a frozen
Qwen3.5-9B, with per-patch rescued/regressed counts in ``results.json``.

This module DERIVES labels; it never vendors patch code. For each of the 27 cells it
verifies ``patch_sha256``, extracts the effect kinds the hook code can return
(``{'kind': ...}`` literals, read by AST, never executed), and maps them onto the MHS-1
``MutationEffect`` vocabulary so the MHS-4 risk prior can be checked against outcomes:

* ``force_action`` / ``rewrite_action``            → REPLACE (override)
* ``block_and_prompt`` / ``inject_hint``            → CONSTRAIN
* prompt-side hooks only (``on_init``, ``make_pre_hint``, ``on_post_step`` with no kind)
                                                    → CONSTRAIN
* a non-literal ``kind``                            → UNKNOWN
* a patch the validator rejected                    → UNSAFE (scored as no-patch, delta 0)

A patch's effect is its riskiest effect under ``MUTATION_EFFECT_RISK``.

Zero compute: reads JSON, parses Python source, writes JSON.

Usage:
    python3 -m scripts.autopilot.species.heldout_effect_corpus \\
        --corpus-dir <Harness-R1>/examples/heldout_generalization \\
        --source-revision <git sha> \\
        --out orchestration/datasets/harness_r1_heldout_effect_corpus.json
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

try:  # package import (production) or flat ``species.`` import (tests)
    from .prompt_forge import MUTATION_EFFECT_RISK, MutationEffect
except ImportError:  # pragma: no cover - script execution fallback
    from scripts.autopilot.species.prompt_forge import MUTATION_EFFECT_RISK, MutationEffect

SCHEMA = "epyc.rtg55.heldout_effect_corpus.v1"

REPLACE_KINDS = frozenset({"force_action", "rewrite_action"})
CONSTRAIN_KINDS = frozenset({"block_and_prompt", "inject_hint"})


class CorpusIntegrityError(ValueError):
    """The corpus on disk does not match the hashes its own results file records."""


def extract_effect_kinds(code: str) -> tuple[set[str], bool]:
    """Literal ``kind`` values a hook may return, and whether any ``kind`` is dynamic.

    Covers both dict literals (``{'kind': 'x'}``) and ``dict(kind='x')`` calls.
    """
    kinds: set[str] = set()
    dynamic = False

    def _take(value: ast.AST) -> None:
        nonlocal dynamic
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            kinds.add(value.value)
        else:
            dynamic = True

    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values, strict=True):
                if isinstance(key, ast.Constant) and key.value == "kind":
                    _take(value)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "dict"
        ):
            for keyword in node.keywords:
                if keyword.arg == "kind":
                    _take(keyword.value)
    return kinds, dynamic


def classify_patch_effect(kinds: set[str], dynamic: bool, *, valid: bool) -> MutationEffect:
    """Map a patch's effect kinds onto the MHS-1 vocabulary (riskiest effect wins)."""
    if not valid:
        return MutationEffect.UNSAFE
    effects: set[MutationEffect] = set()
    for kind in kinds:
        if kind in REPLACE_KINDS:
            effects.add(MutationEffect.REPLACE)
        elif kind in CONSTRAIN_KINDS:
            effects.add(MutationEffect.CONSTRAIN)
        else:
            effects.add(MutationEffect.UNKNOWN)
    if dynamic:
        effects.add(MutationEffect.UNKNOWN)
    if not effects:
        effects.add(MutationEffect.CONSTRAIN)  # prompt-side hooks only
    return max(effects, key=lambda effect: MUTATION_EFFECT_RISK[effect])


def _patch_row(
    corpus_dir: Path, editor: str, seed: str, bench: str, record: dict[str, Any]
) -> dict[str, Any]:
    held_out = record["held_out"]
    valid = bool(record.get("valid"))
    kinds: set[str] = set()
    dynamic = False
    hooks = sorted(record.get("hooks") or [])
    code_chars = 0
    patch_file = record.get("patch_file")
    if valid:
        if not patch_file:
            raise CorpusIntegrityError(f"{editor}/{seed}/{bench}: valid patch without a file")
        raw = (corpus_dir / patch_file).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != record.get("patch_sha256"):
            raise CorpusIntegrityError(
                f"{patch_file}: sha256 {digest} != recorded {record.get('patch_sha256')}"
            )
        for action in json.loads(raw).get("actions", []):
            code = action.get("code")
            if isinstance(code, str):
                code_chars += len(code)
                found, dyn = extract_effect_kinds(code)
                kinds |= found
                dynamic |= dyn
    effect = classify_patch_effect(kinds, dynamic, valid=valid)
    rescued = int(held_out["rescued_failures"])
    regressed = int(held_out["regressed_successes"])
    return {
        "editor": editor,
        "seed": str(seed),
        "benchmark": bench,
        "valid": valid,
        "validation_error": record.get("validation_error"),
        "patch_sha256": record.get("patch_sha256"),
        "hooks": hooks,
        "effect_kinds": sorted(kinds),
        "dynamic_kind": dynamic,
        "effect": effect.value,
        "effect_risk": MUTATION_EFFECT_RISK[effect],
        "code_chars": code_chars,
        "held_out_tasks": int(held_out["tasks"]),
        "delta_pass": int(held_out["delta_pass"]),
        "rescued_failures": rescued,
        "regressed_successes": regressed,
    }


def summarize_by_effect(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per-effect outcome summary (pass-count deltas over each patch's own held-out split)."""
    out: dict[str, dict[str, Any]] = {}
    for effect in MutationEffect:
        group = [r for r in rows if r["effect"] == effect.value]
        if not group:
            continue
        deltas = [r["delta_pass"] for r in group]
        pp = [100.0 * r["delta_pass"] / r["held_out_tasks"] for r in group]
        rescued = sum(r["rescued_failures"] for r in group)
        regressed = sum(r["regressed_successes"] for r in group)
        out[effect.value] = {
            "n_patches": len(group),
            "delta_pass_sum": sum(deltas),
            "delta_pass_mean": round(statistics.fmean(deltas), 4),
            "delta_pp_mean": round(statistics.fmean(pp), 4),
            "delta_pp_min": round(min(pp), 4),
            "n_negative": sum(1 for d in deltas if d < 0),
            "rescued_failures": rescued,
            "regressed_successes": regressed,
            "rescue_to_regression": round(rescued / regressed, 4) if regressed else None,
        }
    return out


def derive_corpus(corpus_dir: Path, *, source_revision: str = "") -> dict[str, Any]:
    """Build the labelled corpus from a checked-out ``heldout_generalization`` directory."""
    results_path = corpus_dir / "results.json"
    results_raw = results_path.read_bytes()
    results = json.loads(results_raw)
    rows: list[dict[str, Any]] = []
    for editor, editor_record in sorted(results["editors"].items()):
        for seed, benches in sorted(editor_record["per_seed"].items()):
            for bench, record in sorted(benches.items()):
                rows.append(_patch_row(corpus_dir, editor, seed, bench, record))
    valid_rows = [r for r in rows if r["valid"]]
    worst = sorted(valid_rows, key=lambda r: r["delta_pass"] / r["held_out_tasks"])[:5]
    return {
        "schema": SCHEMA,
        "source": {
            "repository": "github.com/DeepExperience/Harness-R1",
            "path": "examples/heldout_generalization",
            "revision": source_revision,
            "license": "Apache-2.0",
            "results_sha256": hashlib.sha256(results_raw).hexdigest(),
            "protocol": results.get("protocol"),
            "target_agent": results.get("target_agent"),
            "held_out_tasks": results.get("held_out_tasks"),
        },
        "mapping": {
            "replace_kinds": sorted(REPLACE_KINDS),
            "constrain_kinds": sorted(CONSTRAIN_KINDS),
            "no_kind": "constrain (prompt-side hooks only)",
            "dynamic_kind": "unknown",
            "invalid_patch": "unsafe (validator-rejected; scored as no-patch, delta 0)",
            "patch_effect": "riskiest effect under MUTATION_EFFECT_RISK",
        },
        "rows": rows,
        "by_effect": summarize_by_effect(rows),
        "by_effect_valid_only": summarize_by_effect(valid_rows),
        "worst_valid_patches": [
            {k: r[k] for k in ("editor", "seed", "benchmark", "effect", "effect_kinds", "delta_pass")}
            for r in worst
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--source-revision", default="")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    corpus = derive_corpus(args.corpus_dir, source_revision=args.source_revision)
    args.out.write_text(json.dumps(corpus, indent=1, sort_keys=True) + "\n")
    print(json.dumps(corpus["by_effect_valid_only"], indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
