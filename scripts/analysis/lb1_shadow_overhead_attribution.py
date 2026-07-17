#!/usr/bin/env python3
"""LB-1 offline shadow-reviewer OFF/ON overhead attribution.

Given two **paired-arm artifacts** — a shadow-OFF and a shadow-ON eval run over the
SAME questions/seed (the RCP-W2 per-question output shape: per-question ``eval_wall_s``
+ ``tokens`` + ``t/s``) — this tool attributes the ON-vs-OFF slowdown to the shadow
reviewer: the median / aggregate decode-t/s delta, the eval-wall delta, and the
generated-token delta, each with Wilson (accuracy) and paired-difference confidence
intervals.

It is **PURE offline analysis**: the attribution reads two artifacts already on disk and
does arithmetic. It never runs inference, never starts a server, and never issues a
``/chat`` request. It is the offline half of the LB-1 regression-attribution loop
(unblocks ``LB-1-regression-attribution``).

Execution model (mirrors ``scripts/analysis/run_paired_ab.py`` +
``bsv_paired_runner.py``):

  * DEFAULT = a pure dry-run **plan**. It validates the two artifact paths, parses them,
    resolves the paired question intersection, guards the paired-comparison profile
    (``paired_stats.require_matched_comparison`` on ``dataset_sha256`` + shadow-independent
    base profile), COMPUTES the offline attribution (pure — no model), and prints the
    plan + result. ``inference_ran`` is always False here. Nothing that needs a model runs.
  * The ONLY thing that ever needs inference is (re)GENERATING the two paired arms from
    scratch. That is gated behind BOTH ``--execute`` AND the env flag
    ``LB1_SHADOW_ATTRIBUTION_INFERENCE=1`` (default OFF). Without the env flag, ``--execute``
    degrades to the dry-run plan so a manifest command authored with the real args is still
    safe to simulate. When it does run, every generation rides the **placement queue**
    (``request_priority=background`` + ``workload_class=eval_batch``), NEVER a foreground
    ``/chat`` request.

Statistics wiring reused (not re-implemented):

  * Wilson 95% score interval — ``src/llm_primitives/stat_tests.wilson_interval``.
  * Exact paired McNemar over the per-question correctness vectors + the
    ``dataset_sha256`` / ``test_profile`` pairing gate —
    ``scripts/autopilot/paired_stats``.

Every number this tool emits is a pre-decision OBSERVATION (MEASUREMENT.md) until the
operator's gate table adjudicates it; results are **model/quant-indexed, never
role-indexed** (``feedback_model_not_role_indexing``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ORCH_ROOT = SCRIPT_DIR.parents[1]
for _p in (str(ORCH_ROOT), str(SCRIPT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

RUNNER_VERSION = "lb1-shadow-attribution-v1"

# Env flag gating real inference. Default OFF => dry-run, no model touched. Mirrors
# run_paired_ab's AUTOPILOT_PAIRED_AB_INFERENCE.
LB1_INFERENCE_ENV = "LB1_SHADOW_ATTRIBUTION_INFERENCE"

# Placement-queue transport constants (RM-3). Arm (re)generation rides the SAME
# background/eval_batch placement path an autopilot eval fan-out uses — never /chat.
PLACEMENT_QUEUE_TRANSPORT = "placement_queue"
PLACEMENT_REQUEST_PRIORITY = "background"
PLACEMENT_WORKLOAD_CLASS = "eval_batch"

# Two-sided 95% normal quantile (matches stat_tests.DEFAULT_WILSON_Z).
DEFAULT_Z = 1.959964

# Field-name aliases accepted in a per-question record (tolerant loader).
_QID_KEYS = ("qid", "question_id", "id", "task_id")
_SUITE_KEYS = ("suite", "domain")
_WALL_KEYS = ("eval_wall_s", "wall_s", "eval_wall", "elapsed_s", "latency_s")
_TOKENS_KEYS = ("tokens", "tokens_generated", "n_tokens", "completion_tokens", "output_tokens")
_TPS_KEYS = ("tps", "t_per_s", "tokens_per_s", "tokens_per_second", "speed", "decode_tps")
_CORRECT_KEYS = ("correct", "is_correct", "pass", "passed")

# Artifact-level keys carrying the per-question list.
_RECORDS_KEYS = ("records", "question_results", "rows", "questions", "items")


# ══════════════════════════════════════════════════════════════════════════════
# Deferred cross-module imports (pure stdlib-only modules; never serving path)
# ══════════════════════════════════════════════════════════════════════════════


def _load_paired_stats():
    """Import scripts/autopilot/paired_stats.py (McNemar + profile gate)."""
    try:
        from scripts.autopilot import paired_stats as ps
    except Exception:  # noqa: BLE001 - namespace-package fallback
        ap = str(ORCH_ROOT / "scripts" / "autopilot")
        if ap not in sys.path:
            sys.path.insert(0, ap)
        import paired_stats as ps  # type: ignore[no-redef]
    return ps


def _load_wilson():
    """Import wilson_interval from src/llm_primitives/stat_tests.py."""
    from src.llm_primitives.stat_tests import wilson_interval

    return wilson_interval


# ══════════════════════════════════════════════════════════════════════════════
# Coercion helpers (pure)
# ══════════════════════════════════════════════════════════════════════════════


def _first(item: dict[str, Any], keys: tuple[str, ...], default: Any = None) -> Any:
    for k in keys:
        if k in item and item[k] not in (None, ""):
            return item[k]
    return default


def _as_float(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _as_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on", "correct", "pass", "passed"}:
        return True
    if text in {"0", "false", "no", "off", "wrong", "fail", "failed"}:
        return False
    return None


def _shadow_flag(value: Any) -> bool | None:
    """Interpret a ``shadow`` / ``shadow_enabled`` value as ON (True) / OFF (False)."""
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"on", "1", "true", "enabled", "yes"}:
        return True
    if text in {"off", "0", "false", "disabled", "no"}:
        return False
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Per-question record + arm artifact (pure — file reads only, no model)
# ══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class PerQuestionRecord:
    qid: str
    suite: str
    eval_wall_s: float
    tokens: int
    tps: float
    correct: bool | None


def normalize_record(raw: dict[str, Any], *, index: int) -> PerQuestionRecord:
    """Normalize one raw per-question row into a :class:`PerQuestionRecord`.

    ``tps`` is read directly when present, else derived as ``tokens / eval_wall_s``
    (the RCP-W2 identity). A row missing both a wall and a usable t/s is invalid.
    """
    qid = str(_first(raw, _QID_KEYS, default=f"q{index}"))
    suite = str(_first(raw, _SUITE_KEYS, default="") or "")
    wall = _as_float(_first(raw, _WALL_KEYS))
    tokens_f = _as_float(_first(raw, _TOKENS_KEYS))
    tps = _as_float(_first(raw, _TPS_KEYS))

    tokens = int(round(tokens_f)) if tokens_f is not None else 0
    if tps is None:
        if wall is None or wall <= 0:
            raise ValueError(
                f"record qid={qid!r} has no t/s and no positive eval_wall_s to derive it"
            )
        tps = tokens / wall
    if wall is None:
        if tps <= 0:
            raise ValueError(f"record qid={qid!r} has no eval_wall_s and non-positive t/s")
        wall = tokens / tps if tps else 0.0

    return PerQuestionRecord(
        qid=qid,
        suite=suite,
        eval_wall_s=float(wall),
        tokens=tokens,
        tps=float(tps),
        correct=_as_bool(_first(raw, _CORRECT_KEYS)),
    )


@dataclass
class ArmArtifact:
    shadow_enabled: bool
    model: str
    quant: str
    seed: int
    dataset_sha256: str
    records: dict[str, PerQuestionRecord] = field(default_factory=dict)
    path: str | None = None

    @property
    def label(self) -> str:
        return "shadow_on" if self.shadow_enabled else "shadow_off"

    def base_profile(self) -> str:
        """Shadow-INDEPENDENT scoring profile shared by both arms.

        Deliberately EXCLUDES the shadow flag: the two arms differ ONLY in shadow
        state but must otherwise share model/quant/seed/dataset for the pairing to be
        valid (mirrors run_paired_ab.build_test_profile excluding arm-specific flags).
        """
        return f"model={self.model};quant={self.quant};seed={self.seed};dataset={self.dataset_sha256}"


def _parse_artifact_payload(payload: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Split a parsed artifact into (metadata, per-question rows)."""
    if isinstance(payload, list):
        return {}, [r for r in payload if isinstance(r, dict)]
    if isinstance(payload, dict):
        for key in _RECORDS_KEYS:
            value = payload.get(key)
            if isinstance(value, list):
                return payload, [r for r in value if isinstance(r, dict)]
        raise ValueError(
            "artifact object has no per-question list under any of "
            f"{_RECORDS_KEYS!r}"
        )
    raise ValueError(f"artifact payload must be a list or object, got {type(payload).__name__}")


def _read_artifact_payload(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith(("{", "[")):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    # JSONL fallback: one per-question object per line.
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _compute_dataset_sha256(records: dict[str, PerQuestionRecord], seed: int) -> str:
    """Stable content hash over the sorted qids + seed — the paired dataset identity."""
    h = hashlib.sha256()
    h.update(str(seed).encode("utf-8"))
    h.update(b"\x00")
    for qid in sorted(records):
        h.update(qid.encode("utf-8"))
        h.update(b"\x01")
    return "sha256:" + h.hexdigest()


def load_arm_artifact(
    source: str | Path | dict[str, Any] | list[Any],
    *,
    shadow_override: bool | None = None,
) -> ArmArtifact:
    """Load one paired-arm artifact (RCP-W2 shape) into an :class:`ArmArtifact`.

    ``source`` may be a path, an already-parsed dict/list, or a JSONL file. Pure — file
    reads and arithmetic only, no model. ``shadow_override`` forces the arm's shadow
    state when the artifact does not carry one (e.g. ``--off-artifact`` / ``--on-artifact``
    naming the arm on the CLI).
    """
    path_str: str | None = None
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"arm artifact not found: {path}")
        payload = _read_artifact_payload(path)
        path_str = str(path)
    else:
        payload = source

    meta, raw_rows = _parse_artifact_payload(payload)
    if not raw_rows:
        raise ValueError(f"arm artifact has zero per-question records: {path_str or '<inline>'}")

    records: dict[str, PerQuestionRecord] = {}
    for i, raw in enumerate(raw_rows):
        rec = normalize_record(raw, index=i)
        if rec.qid in records:
            raise ValueError(f"duplicate qid in arm artifact: {rec.qid!r}")
        records[rec.qid] = rec

    shadow = shadow_override
    if shadow is None:
        shadow = _shadow_flag(meta.get("shadow_enabled"))
    if shadow is None:
        shadow = _shadow_flag(meta.get("shadow"))
    if shadow is None:
        raise ValueError(
            f"cannot determine shadow ON/OFF for arm {path_str or '<inline>'}: "
            "supply shadow/shadow_enabled in the artifact or pass it on the CLI"
        )

    seed_val = meta.get("seed")
    try:
        seed = int(seed_val) if seed_val is not None else 0
    except (TypeError, ValueError):
        seed = 0

    dataset_sha256 = str(meta.get("dataset_sha256") or "").strip()
    if not dataset_sha256:
        dataset_sha256 = _compute_dataset_sha256(records, seed)

    return ArmArtifact(
        shadow_enabled=bool(shadow),
        model=str(meta.get("model") or "unknown"),
        quant=str(meta.get("quant") or "unknown"),
        seed=seed,
        dataset_sha256=dataset_sha256,
        records=records,
        path=path_str,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Pairing + profile gate
# ══════════════════════════════════════════════════════════════════════════════


def resolve_arms(arm_a: ArmArtifact, arm_b: ArmArtifact) -> tuple[ArmArtifact, ArmArtifact]:
    """Order the two arms as (off, on); require exactly one of each and a matched profile.

    Uses ``paired_stats.require_matched_comparison`` on ``dataset_sha256`` + the
    shadow-INDEPENDENT base profile, so a model/quant/seed/dataset drift between arms is
    a hard refusal (a paired attribution across mismatched arms is invalid).
    """
    ps = _load_paired_stats()
    if arm_a.shadow_enabled == arm_b.shadow_enabled:
        raise ValueError(
            "the two arms must be one shadow-OFF and one shadow-ON; got both "
            f"shadow_enabled={arm_a.shadow_enabled}"
        )
    off = arm_a if not arm_a.shadow_enabled else arm_b
    on = arm_b if not arm_a.shadow_enabled else arm_a

    ps.require_matched_comparison(
        {"dataset_sha256": off.dataset_sha256, "test_profile": off.base_profile()},
        {"dataset_sha256": on.dataset_sha256, "test_profile": on.base_profile()},
    )
    return off, on


def paired_qids(off: ArmArtifact, on: ArmArtifact) -> list[str]:
    shared = sorted(set(off.records) & set(on.records))
    if not shared:
        raise ValueError("shadow-OFF and shadow-ON arms share no question ids (cannot pair)")
    return shared


# ══════════════════════════════════════════════════════════════════════════════
# Attribution (PURE core — the offline analysis)
# ══════════════════════════════════════════════════════════════════════════════


def _paired_delta_ci(deltas: list[float], z: float) -> dict[str, Any]:
    """Normal-approximation paired-difference CI over per-question deltas."""
    n = len(deltas)
    mean = statistics.fmean(deltas) if n else 0.0
    if n < 2:
        return {
            "n": n,
            "mean": round(mean, 6),
            "sd": None,
            "se": None,
            "ci95": [round(mean, 6), round(mean, 6)],
        }
    sd = statistics.stdev(deltas)  # sample sd (ddof=1)
    se = sd / math.sqrt(n)
    half = z * se
    return {
        "n": n,
        "mean": round(mean, 6),
        "sd": round(sd, 6),
        "se": round(se, 6),
        "ci95": [round(mean - half, 6), round(mean + half, 6)],
    }


@dataclass
class ArmAggregate:
    label: str
    n: int
    total_tokens: int
    total_wall_s: float
    aggregate_tps: float
    median_tps: float
    correct: int | None
    accuracy: float | None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["total_wall_s"] = round(self.total_wall_s, 6)
        d["aggregate_tps"] = round(self.aggregate_tps, 6)
        d["median_tps"] = round(self.median_tps, 6)
        if self.accuracy is not None:
            d["accuracy"] = round(self.accuracy, 6)
        return d


def _aggregate_arm(arm: ArmArtifact, qids: list[str]) -> ArmAggregate:
    recs = [arm.records[q] for q in qids]
    total_tokens = sum(r.tokens for r in recs)
    total_wall = sum(r.eval_wall_s for r in recs)
    agg_tps = (total_tokens / total_wall) if total_wall > 0 else 0.0
    median_tps = statistics.median([r.tps for r in recs])
    corrects = [r.correct for r in recs]
    has_correct = all(c is not None for c in corrects)
    correct = sum(1 for c in corrects if c) if has_correct else None
    accuracy = (correct / len(recs)) if (has_correct and recs) else None
    return ArmAggregate(
        label=arm.label,
        n=len(recs),
        total_tokens=total_tokens,
        total_wall_s=total_wall,
        aggregate_tps=agg_tps,
        median_tps=median_tps,
        correct=correct,
        accuracy=accuracy,
    )


def compute_attribution(
    off: ArmArtifact,
    on: ArmArtifact,
    *,
    z: float = DEFAULT_Z,
) -> dict[str, Any]:
    """Attribute the ON-vs-OFF shadow overhead. PURE — no model, no network.

    Deltas are ``on - off`` (a positive eval-wall delta / negative t/s delta = the
    shadow reviewer costing time). Returns a model/quant-indexed observation dict.
    """
    ps = _load_paired_stats()
    wilson_interval = _load_wilson()

    if off.model != on.model or off.quant != on.quant:
        raise ValueError(
            "paired arms disagree on model/quant: "
            f"{off.model}/{off.quant} != {on.model}/{on.quant}"
        )

    qids = paired_qids(off, on)
    n = len(qids)

    off_agg = _aggregate_arm(off, qids)
    on_agg = _aggregate_arm(on, qids)

    wall_deltas = [on.records[q].eval_wall_s - off.records[q].eval_wall_s for q in qids]
    tps_deltas = [on.records[q].tps - off.records[q].tps for q in qids]

    aggregate_tps_delta = on_agg.aggregate_tps - off_agg.aggregate_tps
    median_tps_delta = on_agg.median_tps - off_agg.median_tps
    eval_wall_total_delta = on_agg.total_wall_s - off_agg.total_wall_s
    token_total_delta = on_agg.total_tokens - off_agg.total_tokens
    overhead_fraction = (
        (off_agg.aggregate_tps - on_agg.aggregate_tps) / off_agg.aggregate_tps
        if off_agg.aggregate_tps > 0
        else None
    )

    # Accuracy Wilson CIs + McNemar over the per-question correctness vectors (present
    # iff BOTH arms carried correctness). Shadow overhead is a SPEED cost; this checks
    # the shadow reviewer did not perturb accuracy.
    accuracy_block: dict[str, Any] = {"available": False}
    have_correct = off_agg.correct is not None and on_agg.correct is not None
    if have_correct:
        off_wilson = wilson_interval(off_agg.correct, n, z)
        on_wilson = wilson_interval(on_agg.correct, n, z)
        off_vec = {
            q: ps.QuestionOutcome(
                qid=q, suite=off.records[q].suite, correct=bool(off.records[q].correct), trial_id=0
            )
            for q in qids
        }
        on_vec = {
            q: ps.QuestionOutcome(
                qid=q, suite=on.records[q].suite, correct=bool(on.records[q].correct), trial_id=1
            )
            for q in qids
        }
        mcn = ps.mcnemar_from_vectors(off_vec, on_vec, label_a="shadow_off", label_b="shadow_on")
        accuracy_block = {
            "available": True,
            "off_correct": off_agg.correct,
            "on_correct": on_agg.correct,
            "off_accuracy": round(off_agg.accuracy, 6),
            "on_accuracy": round(on_agg.accuracy, 6),
            "off_wilson95": [round(off_wilson[0], 6), round(off_wilson[1], 6)],
            "on_wilson95": [round(on_wilson[0], 6), round(on_wilson[1], 6)],
            "accuracy_delta_on_minus_off": round(on_agg.accuracy - off_agg.accuracy, 6),
            "mcnemar_p_value_two_sided": mcn.p_value_two_sided,
            "mcnemar": asdict(mcn),
        }

    return {
        "kind": "lb1_shadow_overhead_attribution",
        "runner_version": RUNNER_VERSION,
        "indexed_by": "model_quant",  # NEVER role (feedback_model_not_role_indexing)
        "model": off.model,
        "quant": off.quant,
        "model_quant_key": f"{off.model}/{off.quant}",
        "dataset_sha256": off.dataset_sha256,
        "base_profile": off.base_profile(),
        "z": z,
        "paired_qids": n,
        "arms": {
            "shadow_off": off_agg.to_dict(),
            "shadow_on": on_agg.to_dict(),
        },
        "delta_on_minus_off": {
            "aggregate_tps": round(aggregate_tps_delta, 6),
            "median_tps": round(median_tps_delta, 6),
            "eval_wall_total_s": round(eval_wall_total_delta, 6),
            "eval_wall_mean_s": round(statistics.fmean(wall_deltas), 6),
            "tokens_total": token_total_delta,
        },
        "overhead_fraction_tps": (
            round(overhead_fraction, 6) if overhead_fraction is not None else None
        ),
        "paired_ci": {
            "eval_wall_delta_s": _paired_delta_ci(wall_deltas, z),
            "tps_delta": _paired_delta_ci(tps_deltas, z),
        },
        "accuracy": accuracy_block,
        "observation_only": True,  # pre-decision (MEASUREMENT.md); never gates alone
    }


# ══════════════════════════════════════════════════════════════════════════════
# Plan (dry-run) — pure, model-free
# ══════════════════════════════════════════════════════════════════════════════


def build_plan(
    off: ArmArtifact,
    on: ArmArtifact,
    attribution: dict[str, Any],
    *,
    output_path: Path | None,
    execute_requested: bool,
    env_enabled: bool,
) -> dict[str, Any]:
    notes = [
        "DRY-RUN: pure offline attribution — no model, no /chat, no placement-queue dispatch.",
        f"arm (re)generation would require --execute AND {LB1_INFERENCE_ENV}=1.",
        "all numbers are pre-decision observations (MEASUREMENT.md); never gate alone.",
    ]
    if execute_requested and not env_enabled:
        notes.append(
            f"--execute requested but {LB1_INFERENCE_ENV} not set; falling back to "
            "dry-run (no inference)."
        )
    return {
        "kind": "lb1_shadow_overhead_plan",
        "runner_version": RUNNER_VERSION,
        "mode": "dry_run",
        "inference_ran": False,
        "indexed_by": "model_quant",
        "model": off.model,
        "quant": off.quant,
        "model_quant_key": f"{off.model}/{off.quant}",
        "arms": {
            "shadow_off": {
                "path": off.path,
                "records": len(off.records),
                "dataset_sha256": off.dataset_sha256,
            },
            "shadow_on": {
                "path": on.path,
                "records": len(on.records),
                "dataset_sha256": on.dataset_sha256,
            },
        },
        "paired_qids": attribution["paired_qids"],
        "transport": {
            "transport": PLACEMENT_QUEUE_TRANSPORT,
            "request_priority": PLACEMENT_REQUEST_PRIORITY,
            "workload_class": PLACEMENT_WORKLOAD_CLASS,
            "uses_chat_endpoint": False,
        },
        "output_path": str(output_path) if output_path else None,
        "attribution": attribution,
        "notes": notes,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Execution seam (env-gated; placement queue; NEVER run in tests)
# ══════════════════════════════════════════════════════════════════════════════


def _env_flag_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _generate_arms_via_placement_queue(  # pragma: no cover - inference path
    *,
    manifest: Path,
    model: str,
    quant: str,
    seed: int,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Regenerate the shadow-OFF and shadow-ON arms over the PLACEMENT QUEUE.

    Real inference seam — reached ONLY under ``--execute`` AND
    ``LB1_SHADOW_ATTRIBUTION_INFERENCE=1``. Each generation rides the same
    ``request_priority=background`` + ``workload_class=eval_batch`` placement path an
    autopilot eval fan-out uses (never a foreground ``/chat`` request). Never exercised
    by the unit tests. The operator owns the no-concurrent-inference quiet window.
    """
    _research = Path("/mnt/raid0/llm/epyc-inference-research")
    _bench = str(_research / "scripts" / "benchmark")
    if _bench not in sys.path:
        sys.path.insert(0, _bench)
    from seeding_orchestrator import call_orchestrator_forced  # type: ignore

    raise NotImplementedError(
        "LB-1 arm (re)generation is an operator-quiet-window action: it drives the "
        f"shadow OFF/ON eval fan-out via call_orchestrator_forced (request_priority="
        f"{PLACEMENT_REQUEST_PRIORITY!r}, workload_class={PLACEMENT_WORKLOAD_CLASS!r}). "
        "Provide the two artifacts to the offline attribution path instead."
    )


# ══════════════════════════════════════════════════════════════════════════════
# Top-level orchestration + CLI
# ══════════════════════════════════════════════════════════════════════════════


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Load both arms, resolve pairing + profile gate, compute the offline attribution.

    DEFAULT path is pure (dry-run plan carrying the attribution). The ``--execute`` +
    env-gated arm-(re)generation seam is never entered in tests.
    """
    off_shadow = None if args.off_shadow_auto else False
    on_shadow = None if args.on_shadow_auto else True
    arm_a = load_arm_artifact(args.off_artifact, shadow_override=off_shadow)
    arm_b = load_arm_artifact(args.on_artifact, shadow_override=on_shadow)

    off, on = resolve_arms(arm_a, arm_b)
    attribution = compute_attribution(off, on, z=args.z)

    output_path = Path(args.output) if args.output else None
    execute_requested = bool(args.execute)
    env_enabled = _env_flag_enabled(LB1_INFERENCE_ENV)

    if execute_requested and env_enabled:  # pragma: no cover - inference path
        if not args.manifest:
            raise SystemExit("--execute needs --manifest to (re)generate the paired arms")
        _generate_arms_via_placement_queue(
            manifest=Path(args.manifest),
            model=off.model,
            quant=off.quant,
            seed=off.seed,
            output_dir=output_path or (ORCH_ROOT / "data" / "lb1_shadow_attribution"),
        )

    plan = build_plan(
        off,
        on,
        attribution,
        output_path=output_path,
        execute_requested=execute_requested,
        env_enabled=env_enabled,
    )
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(attribution, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
    return plan


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "LB-1 offline shadow-reviewer OFF/ON overhead attribution. DEFAULT is a pure "
            "dry-run PLAN that parses two paired-arm artifacts, resolves the paired "
            "question set, and COMPUTES the offline attribution (median/aggregate t/s, "
            "eval-wall + token deltas, Wilson/paired CIs) with NO inference. Arm "
            f"regeneration needs --execute AND {LB1_INFERENCE_ENV}=1."
        )
    )
    p.add_argument(
        "--off-artifact",
        required=True,
        help="path to the shadow-OFF paired-arm artifact (RCP-W2 per-question shape).",
    )
    p.add_argument(
        "--on-artifact",
        required=True,
        help="path to the shadow-ON paired-arm artifact (RCP-W2 per-question shape).",
    )
    p.add_argument(
        "--off-shadow-auto",
        action="store_true",
        help="read the OFF arm's shadow state from the artifact instead of forcing OFF.",
    )
    p.add_argument(
        "--on-shadow-auto",
        action="store_true",
        help="read the ON arm's shadow state from the artifact instead of forcing ON.",
    )
    p.add_argument("--z", type=float, default=DEFAULT_Z, help="normal quantile for CIs (default 95%%).")
    p.add_argument("--output", default=None, help="optional path to write the attribution JSON.")
    p.add_argument(
        "--manifest",
        default=None,
        help="corpus manifest for arm (re)generation (only used under --execute + env gate).",
    )
    p.add_argument(
        "--execute",
        action="store_true",
        help="attempt arm (re)generation via the placement queue (STILL env-gated by "
        f"{LB1_INFERENCE_ENV}=1; otherwise falls back to the dry-run offline attribution).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
