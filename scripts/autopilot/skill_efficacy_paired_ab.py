#!/usr/bin/env python3
"""EV-10a — paired skill-ON vs skill-OFF efficacy A/B runner.

This is the standalone runner referenced (as a fabricated command) by the
inference-batch manifest entry ``EV-10a-skill-efficacy-ab``. It DRIVES the landed
decision-logic library ``scripts/autopilot/skill_efficacy.py`` (EV-10a) over TWO
eval arms scored on the SAME questions — a with-artifact (skill-ON) arm and a
no-artifact (skill-OFF) arm — honoring the dev/test_normal split discipline and
the per-suite negative-delta guard (SkillsBench 16/84 protection).

Two responsibilities, cleanly split (exactly the ``screening_tier_runner.py``
pattern):

  1. **Plan resolution + paired-stats wiring** (pure, inference-free — this is the
     entire surface the unit tests exercise): resolve the paired core corpus,
     partition it deterministically into the dev / test_normal splits, pair a
     skill-ON and skill-OFF arm on the *identical* question set per split, and —
     given the two arms' per-question outcomes — compute the paired verdict:
       * the EV-10a efficacy verdict via
         ``skill_efficacy.evaluate_skill_efficacy`` /
         ``evaluate_skill_efficacy_split`` (per-suite delta + negative-delta guard
         + dev/test AND discipline),
       * the exact paired-McNemar test via
         ``paired_stats.mcnemar_from_vectors`` (b − c discordant counts, exact
         two-sided p),
       * per-arm Wilson score CIs via ``src/llm_primitives/stat_tests.wilson_interval``,
       * a hard ``require_matched_comparison`` gate so two arms are paired ONLY
         when their dataset hash + eval profile match (the treatment is the skill
         presence, never a data/profile drift).
     Results are indexed by (model, quant) — never by role (measurement policy).

  2. **Execution bridge** (env-flag-gated ``AUTOPILOT_SKILL_EFFICACY_AB_INFERENCE=1``
     AND ``--run``, DEFAULT OFF): with the flag OFF the resolved paired plan is
     returned as a dry-run and NO inference happens — the default path validates
     config, resolves the corpus, and prints the planned paired run without
     touching a model. With the flag ON, drive both arms over the eval tower via
     the PLACEMENT QUEUE (``request_priority=background`` + ``workload_class=eval_batch``,
     NEVER a foreground ``/chat`` call — the eval fan-out discipline), emit per-arm
     JSONL + the paired result, and revert the skill artifact between arms. The
     execution bridge is modeled on ``bsv_paired_runner.py`` /
     ``screening_tier_runner.py`` (deferred ``EvalTower`` import, autopilot-stopped
     assumption) and is intentionally NEVER reached by the tests.

Relationship to the DEPLOYED gate (``AUTOPILOT_SKILL_EFFICACY_GATE``, wired in
``scripts/autopilot/actions.py``): that flag turns the SAME
``evaluate_skill_efficacy`` decision into a live accept-path revert for an
autopilot mutation. This runner is the offline paired-A/B *validator* of that
decision — it has its OWN inference gate
(``AUTOPILOT_SKILL_EFFICACY_AB_INFERENCE``) so a dry-run in any session never
runs inference, and it keeps ``AUTOPILOT_SKILL_EFFICACY_GATE`` isolated from
``AUTOPILOT_BSV2_ACCEPT_GATE`` for attribution (flag-isolation contract,
skill_efficacy.py L35-45).

Constraints honored (CLAUDE.md + eval-tower-verification.md EV-10):
  * NEVER starts/stops autopilot, NEVER writes autopilot_state.json / journals /
    runtime_flags.json, and NEVER edits actions.py / skill_efficacy.py /
    eval_tower.py (all imported READ-ONLY).
  * Every number produced here is a pre-decision OBSERVATION (MEASUREMENT.md) —
    the operator loop / the deployed gate consumes it, it never self-gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

SCRIPT_DIR = Path(__file__).resolve().parent
ORCH_ROOT = SCRIPT_DIR.parents[1]
for _p in (str(SCRIPT_DIR), str(ORCH_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── decision-lib + paired-stats imports (dual path: package or bare script) ──
try:
    from scripts.autopilot.skill_efficacy import (  # type: ignore[import-not-found]
        EfficacyVerdict,
        evaluate_skill_efficacy,
        evaluate_skill_efficacy_split,
    )
except Exception:  # noqa: BLE001 - direct-script execution fallback
    from skill_efficacy import (  # type: ignore[no-redef]
        EfficacyVerdict,
        evaluate_skill_efficacy,
        evaluate_skill_efficacy_split,
    )

try:
    from scripts.autopilot.paired_stats import (  # type: ignore[import-not-found]
        ComparisonProfile,
        McNemarResult,
        QuestionOutcome,
        mcnemar_from_vectors,
        require_matched_comparison,
    )
except Exception:  # noqa: BLE001
    from paired_stats import (  # type: ignore[no-redef]
        ComparisonProfile,
        McNemarResult,
        QuestionOutcome,
        mcnemar_from_vectors,
        require_matched_comparison,
    )

from src.llm_primitives.stat_tests import DEFAULT_WILSON_Z, wilson_interval

# ── constants ────────────────────────────────────────────────────────────────
RUNNER_VERSION = "skill-efficacy-paired-ab-v1"
PROTOCOL_ID = "eval-tower.skill-efficacy-ab.v1"

# The runner's OWN inference gate — distinct from the deployed accept-path gate
# (AUTOPILOT_SKILL_EFFICACY_GATE) so an offline dry-run never runs a model.
SKILL_EFFICACY_AB_INFERENCE_ENV = "AUTOPILOT_SKILL_EFFICACY_AB_INFERENCE"
# The deployed decision-lib gate (referenced for provenance only; NOT set here).
SKILL_EFFICACY_GATE_ENV = "AUTOPILOT_SKILL_EFFICACY_GATE"

# Placement-queue transport (eval fan-out discipline). A paired arm rides the
# SAME background/eval_batch path a normal autopilot eval fan-out uses — never a
# foreground /chat request.
PLACEMENT_QUEUE_TRANSPORT = "placement_queue"
PLACEMENT_REQUEST_PRIORITY = "background"
PLACEMENT_WORKLOAD_CLASS = "eval_batch"

# Canonical arm labels (match the manifest command: --arms with_artifact,no_artifact).
ARM_SKILL_ON = "with_artifact"
ARM_SKILL_OFF = "no_artifact"
_ON_ALIASES = {"with_artifact", "with", "with_skill", "skill_on", "on"}
_OFF_ALIASES = {"no_artifact", "without", "no_skill", "skill_off", "off", "baseline"}

# Default dev/test split discipline (AppWorld dev/test_normal convention).
DEFAULT_SPLITS = ("dev", "test_normal")

DEFAULT_CORE_DIR = ORCH_ROOT / "benchmarks" / "prompts"
_CORE_METADATA_KEY = "__core_metadata__"
DEFAULT_EVAL_ROLE = "frontdoor"
DEFAULT_REGISTRY_PATH = ORCH_ROOT / "orchestration" / "model_registry.yaml"


def _env_flag_enabled(name: str) -> bool:
    """True iff env var ``name`` is a truthy flag (matches actions._env_flag_enabled)."""
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


# ══════════════════════════════════════════════════════════════════════════════
# Arm parsing + config
# ══════════════════════════════════════════════════════════════════════════════


def parse_arms(arms_csv: str) -> dict[str, bool]:
    """Parse ``--arms`` into ``{label: skill_enabled}`` (exactly one ON + one OFF).

    Accepts the canonical ``with_artifact,no_artifact`` plus a small alias set.
    Raises ``ValueError`` on anything that is not exactly one skill-ON and one
    skill-OFF arm — a paired A/B is meaningless otherwise.
    """
    labels = [a.strip() for a in str(arms_csv).split(",") if a.strip()]
    if len(labels) != 2:
        raise ValueError(f"--arms must name exactly 2 arms (ON,OFF); got {labels!r}")
    mapping: dict[str, bool] = {}
    for label in labels:
        key = label.lower()
        if key in _ON_ALIASES:
            mapping[label] = True
        elif key in _OFF_ALIASES:
            mapping[label] = False
        else:
            raise ValueError(
                f"unrecognized arm label {label!r}; expected one of "
                f"{sorted(_ON_ALIASES)} (skill-ON) or {sorted(_OFF_ALIASES)} (skill-OFF)"
            )
    if sum(1 for v in mapping.values() if v) != 1:
        raise ValueError(
            f"--arms must pair exactly one skill-ON with one skill-OFF arm; got {mapping!r}"
        )
    return mapping


# ══════════════════════════════════════════════════════════════════════════════
# Corpus resolution (pure — deterministic, tolerant of a missing core)
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class CorpusResolution:
    corpus_id: str
    path: str | None
    exists: bool
    questions: list[dict[str, str]]  # [{qid, suite}] in file order
    n_rows: int
    dataset_sha256: str
    suites: list[str]
    error: str | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "corpus_id": self.corpus_id,
            "path": self.path,
            "exists": self.exists,
            "n_rows": self.n_rows,
            "dataset_sha256": self.dataset_sha256,
            "suites": list(self.suites),
            "error": self.error,
            "notes": list(self.notes),
        }


def dataset_sha256(qids: list[str]) -> str:
    """Stable content hash over the SORTED unique qids of a question set.

    This is the ``ComparisonProfile.dataset_sha256`` that pins the exact scored
    question set — two paired arms may be compared ONLY when this matches.
    """
    payload = "\x00".join(sorted(set(qids)))
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _row_qid(row: dict[str, Any]) -> str:
    explicit = str(row.get("qid") or row.get("id") or row.get("question_id") or "").strip()
    if explicit:
        return explicit
    payload = f"{row.get('suite', 'unknown')}\x00{row.get('prompt', '')}".encode(
        "utf-8", errors="replace"
    )
    return hashlib.sha1(payload).hexdigest()[:16]


def resolve_corpus(
    *,
    core_id: str | None = None,
    questions_path: str | None = None,
    core_dir: Path = DEFAULT_CORE_DIR,
    suites: set[str] | None = None,
) -> CorpusResolution:
    """Resolve the paired-core corpus to a concrete [{qid, suite}] question set.

    Path precedence: explicit ``questions_path`` -> ``core_dir/<core_id>.jsonl``
    -> autodetect a single ``*.jsonl`` under ``core_dir``. Pure file read; skips
    the optional ``__core_metadata__`` row. Tolerant: a missing corpus yields
    ``exists=False`` with an error note (so a dry-run can still print a plan) —
    it never raises. Optionally filters to a subset of ``suites``.
    """
    core_dir = Path(core_dir)
    resolved_core_id = core_id or ""
    error: str | None = None
    notes: list[str] = []

    if questions_path:
        path: Path | None = Path(questions_path)
        if not resolved_core_id:
            resolved_core_id = path.stem
    elif core_id:
        path = core_dir / f"{core_id}.jsonl"
    else:
        candidates = sorted(core_dir.glob("*.jsonl")) if core_dir.exists() else []
        if len(candidates) == 1:
            path = candidates[0]
            resolved_core_id = path.stem
        elif len(candidates) > 1:
            path = None
            error = (
                f"multiple cores under {core_dir}; pass --core-id "
                f"({', '.join(c.stem for c in candidates)})"
            )
        else:
            path = None
            error = f"no core JSONL found under {core_dir}"

    questions: list[dict[str, str]] = []
    present_suites: set[str] = set()
    if path is not None and path.exists():
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict) or row.get(_CORE_METADATA_KEY):
                    if isinstance(row, dict) and row.get("core_id") and not resolved_core_id:
                        resolved_core_id = str(row["core_id"])
                    continue
                suite = str(row.get("suite", "")).strip()
                if suites and suite not in suites:
                    continue
                qid = _row_qid(row)
                questions.append({"qid": qid, "suite": suite})
                if suite:
                    present_suites.add(suite)
        exists = True
        if not questions:
            error = error or f"{path}: no scoreable questions after filtering"
    else:
        exists = False
        if error is None:
            error = f"corpus path not found: {path}"

    if suites:
        notes.append(f"filtered to suites={sorted(suites)}")

    return CorpusResolution(
        corpus_id=resolved_core_id or "unknown",
        path=str(path) if path is not None else None,
        exists=exists,
        questions=questions,
        n_rows=len(questions),
        dataset_sha256=dataset_sha256([q["qid"] for q in questions]),
        suites=sorted(present_suites),
        error=error,
        notes=notes,
    )


def split_questions(
    questions: list[dict[str, str]],
    split_names: list[str],
    *,
    seed: int,
) -> dict[str, list[dict[str, str]]]:
    """Deterministically partition questions across splits (same seed -> same split).

    The paired core carries no native dev/test partition, so a stable per-qid hash
    assigns each question to a split round-robin over the hash order. Reproducible
    without persisting the assignment, and balanced to within one question per
    split. Every arm of a given split then scores the IDENTICAL question set.
    """
    if not split_names:
        raise ValueError("at least one split is required")
    buckets: dict[str, list[dict[str, str]]] = {s: [] for s in split_names}
    ordered = sorted(
        questions,
        key=lambda q: hashlib.sha1(f"{seed}\x00{q['qid']}".encode("utf-8")).hexdigest(),
    )
    for i, q in enumerate(ordered):
        buckets[split_names[i % len(split_names)]].append(q)
    return buckets


def build_test_profile(
    *, split: str, seed: int, eval_role: str, n: int
) -> str:
    """Eval-profile identity shared by BOTH arms of a split.

    Encodes the scoring/sampling conditions that must match for a valid paired
    comparison — deliberately EXCLUDING the skill flag, because the skill presence
    is the TREATMENT, not a profile difference.
    """
    return (
        f"{PROTOCOL_ID};role={eval_role};split={split};seed={seed};n={n};"
        "scoring=paired-core;sampling=production_seed"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Model / quant resolution (best-effort registry read; NEVER role-indexed output)
# ══════════════════════════════════════════════════════════════════════════════


def resolve_model_quant(
    role: str,
    *,
    model: str | None = None,
    quant: str | None = None,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
) -> tuple[str, str]:
    """Resolve (model, quant) for ``role`` from the lean registry (CLI overrides win).

    Benches are indexed by MODEL + QUANT, never by role (measurement policy). This
    reader is best-effort and pure: on any failure it returns whatever the caller
    supplied, else ``("unknown", "unknown")``. Tests pass model/quant explicitly
    and never hit the registry.
    """
    if model and quant:
        return model, quant
    r_model, r_quant = model or "unknown", quant or "unknown"
    try:
        import yaml  # type: ignore[import-not-found]

        data = yaml.safe_load(Path(registry_path).read_text(encoding="utf-8")) or {}
        entry = ((data.get("roles") or {}).get(role) or {}).get("model") or {}
        if isinstance(entry, dict):
            r_model = model or str(entry.get("name") or r_model)
            r_quant = quant or str(entry.get("quant") or r_quant)
        if r_quant == "unknown":
            r_quant = str((data.get("runtime_defaults") or {}).get("quantization") or "unknown")
    except Exception:  # noqa: BLE001 - registry read is advisory only
        pass
    return r_model, r_quant


# ══════════════════════════════════════════════════════════════════════════════
# Arm outcome + paired-stats wiring (pure — the tested surface)
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class ArmOutcome:
    """One scored arm of a paired split (skill-ON or skill-OFF).

    ``per_suite_quality`` feeds the EV-10a efficacy verdict; ``question_results``
    (compact ``{qid, suite, correct}`` rows) feed the paired-McNemar test and the
    Wilson CI. ``dataset_sha256`` + ``test_profile`` pin the arm identity for the
    ``require_matched_comparison`` gate.
    """

    label: str
    split: str
    skill_enabled: bool
    model: str
    quant: str
    dataset_sha256: str
    test_profile: str
    per_suite_quality: dict[str, float]
    question_results: list[dict[str, Any]]

    def profile(self) -> ComparisonProfile:
        return ComparisonProfile(
            dataset_sha256=self.dataset_sha256, test_profile=self.test_profile
        )

    def vector(self, trial_id: int = 0) -> dict[str, QuestionOutcome]:
        vec: dict[str, QuestionOutcome] = {}
        for item in self.question_results:
            qid = str(item.get("qid") or item.get("question_id") or "").strip()
            if not qid:
                continue
            vec[qid] = QuestionOutcome(
                qid=qid,
                suite=str(item.get("suite") or ""),
                correct=bool(item.get("correct")),
                trial_id=trial_id,
            )
        return vec

    def n_questions(self) -> int:
        return len(self.vector())

    def n_correct(self) -> int:
        return sum(1 for o in self.vector().values() if o.correct)

    def wilson_ci(self, z: float = DEFAULT_WILSON_Z) -> tuple[float, float]:
        return wilson_interval(self.n_correct(), self.n_questions(), z=z)

    def summary(self, z: float = DEFAULT_WILSON_Z) -> dict[str, Any]:
        n = self.n_questions()
        c = self.n_correct()
        lo, hi = wilson_interval(c, n, z=z)
        return {
            "label": self.label,
            "split": self.split,
            "skill_enabled": self.skill_enabled,
            "model": self.model,
            "quant": self.quant,
            "n_questions": n,
            "n_correct": c,
            "accuracy": (c / n) if n else 0.0,
            "wilson_lower": lo,
            "wilson_upper": hi,
            "per_suite_quality": dict(self.per_suite_quality),
            "dataset_sha256": self.dataset_sha256,
        }


def _per_suite_pass_rate(question_results: list[dict[str, Any]]) -> dict[str, float]:
    """Derive a per-suite pass-rate (0-1) from compact question outcomes."""
    correct: dict[str, int] = {}
    total: dict[str, int] = {}
    for item in question_results:
        suite = str(item.get("suite") or "")
        total[suite] = total.get(suite, 0) + 1
        if item.get("correct"):
            correct[suite] = correct.get(suite, 0) + 1
    return {s: correct.get(s, 0) / n for s, n in total.items() if n}


def arm_outcome_from_question_results(
    *,
    label: str,
    split: str,
    skill_enabled: bool,
    model: str,
    quant: str,
    dataset_sha256: str,
    test_profile: str,
    question_results: list[dict[str, Any]],
    per_suite_quality: dict[str, float] | None = None,
) -> ArmOutcome:
    """Build an :class:`ArmOutcome` from compact per-question outcomes.

    When ``per_suite_quality`` is omitted it is derived as the per-suite pass-rate
    — so a caller with only ``{qid, suite, correct}`` vectors (e.g. a synthetic
    fixture, or an eval harness that did not journal per-suite quality) still gets
    a valid efficacy input.
    """
    return ArmOutcome(
        label=label,
        split=split,
        skill_enabled=skill_enabled,
        model=model,
        quant=quant,
        dataset_sha256=dataset_sha256,
        test_profile=test_profile,
        per_suite_quality=(
            dict(per_suite_quality)
            if per_suite_quality is not None
            else _per_suite_pass_rate(question_results)
        ),
        question_results=list(question_results),
    )


def _transport_summary() -> dict[str, Any]:
    return {
        "transport": PLACEMENT_QUEUE_TRANSPORT,
        "request_priority": PLACEMENT_REQUEST_PRIORITY,
        "workload_class": PLACEMENT_WORKLOAD_CLASS,
        "uses_chat_endpoint": False,
    }


def compute_paired_efficacy(
    off_arm: ArmOutcome,
    on_arm: ArmOutcome,
    *,
    skill: str | None = None,
    regress_threshold: float = 0.10,
    require_aggregate_gain: bool = True,
    require_matched: bool = True,
    wilson_z: float = DEFAULT_WILSON_Z,
) -> dict[str, Any]:
    """Compute the paired verdict for ONE split from its two arms.

    Wires the three primitives on the SAME question set:
      * ``require_matched_comparison`` — refuse to pair arms whose dataset hash or
        eval profile disagree (the skill flag is NOT part of the profile).
      * ``evaluate_skill_efficacy`` — per-suite delta + negative-delta guard.
      * ``mcnemar_from_vectors`` — exact paired-McNemar (b − c discordant counts).
      * ``wilson_interval`` — per-arm Wilson score CI.

    ``off_arm`` is the no-artifact (skill-OFF) baseline; ``on_arm`` is the
    with-artifact (skill-ON) arm — so McNemar's ``delta_b_minus_a`` reads as
    (skill-ON − skill-OFF) accuracy. Returns a pre-decision OBSERVATION dict.
    """
    if off_arm.split != on_arm.split:
        raise ValueError(
            f"paired arms disagree on split: {off_arm.split!r} != {on_arm.split!r}"
        )
    if require_matched:
        # Raises PairedComparisonMismatchError on dataset/profile drift — refuse to
        # pair arms that were not scored on the same questions under the same profile.
        require_matched_comparison(off_arm.profile(), on_arm.profile())
    stats: McNemarResult = mcnemar_from_vectors(
        off_arm.vector(trial_id=0),
        on_arm.vector(trial_id=1),
        label_a=off_arm.label,
        label_b=on_arm.label,
    )
    verdict: EfficacyVerdict = evaluate_skill_efficacy(
        off_arm.per_suite_quality,
        on_arm.per_suite_quality,
        regress_threshold=regress_threshold,
        require_aggregate_gain=require_aggregate_gain,
    )
    off_lo, off_hi = off_arm.wilson_ci(z=wilson_z)
    on_lo, on_hi = on_arm.wilson_ci(z=wilson_z)
    return {
        "kind": "skill_efficacy_ab_split_result",
        "runner_version": RUNNER_VERSION,
        "protocol_id": PROTOCOL_ID,
        "split": off_arm.split,
        "skill": skill,
        "model": on_arm.model,
        "quant": on_arm.quant,
        "dataset_sha256": off_arm.dataset_sha256,
        "test_profile": off_arm.test_profile,
        "regress_threshold": regress_threshold,
        "efficacy": {
            "accept": verdict.accept,
            "aggregate_delta": verdict.aggregate_delta,
            "per_suite_delta": verdict.per_suite_delta,
            "regressed_suites": verdict.regressed_suites,
            "reason": verdict.reason,
        },
        "mcnemar": {
            "shared_qids": stats.shared_qids,
            "on_correct_off_wrong": stats.a_wrong_b_correct,
            "off_correct_on_wrong": stats.a_correct_b_wrong,
            "same_correct": stats.same_correct,
            "same_wrong": stats.same_wrong,
            "p_value_two_sided": stats.p_value_two_sided,
            "accuracy_off": stats.accuracy_a,
            "accuracy_on": stats.accuracy_b,
            "delta_on_minus_off": stats.delta_b_minus_a,
        },
        "wilson_ci": {
            ARM_SKILL_OFF: [off_lo, off_hi],
            ARM_SKILL_ON: [on_lo, on_hi],
        },
        "arms": {
            ARM_SKILL_OFF: off_arm.summary(z=wilson_z),
            ARM_SKILL_ON: on_arm.summary(z=wilson_z),
        },
        "transport": _transport_summary(),
        "observation_only": True,  # pre-decision (MEASUREMENT.md)
    }


def compute_split_paired_efficacy(
    arms_by_split: dict[str, tuple[ArmOutcome, ArmOutcome]],
    *,
    skill: str | None = None,
    regress_threshold: float = 0.10,
    require_aggregate_gain: bool = True,
    require_matched: bool = True,
    wilson_z: float = DEFAULT_WILSON_Z,
) -> dict[str, Any]:
    """Combine per-split paired results into the dev/test_normal split verdict.

    ``arms_by_split`` maps each split -> (off_arm, on_arm). The overall verdict
    uses ``evaluate_skill_efficacy_split`` on the dev/test arms (accept requires
    ACCEPT on BOTH splits — the overfit-to-dev guard). Also carries the per-split
    paired-McNemar + Wilson observations. Requires both a ``dev`` and a
    ``test_normal`` arm pair to render the split verdict; otherwise reports the
    per-split results with ``split_verdict=None``.
    """
    per_split: dict[str, dict[str, Any]] = {}
    for split, (off_arm, on_arm) in arms_by_split.items():
        per_split[split] = compute_paired_efficacy(
            off_arm,
            on_arm,
            skill=skill,
            regress_threshold=regress_threshold,
            require_aggregate_gain=require_aggregate_gain,
            require_matched=require_matched,
            wilson_z=wilson_z,
        )

    split_verdict: dict[str, Any] | None = None
    if "dev" in arms_by_split and "test_normal" in arms_by_split:
        dev_off, dev_on = arms_by_split["dev"]
        test_off, test_on = arms_by_split["test_normal"]
        v = evaluate_skill_efficacy_split(
            dev_off.per_suite_quality,
            dev_on.per_suite_quality,
            test_off.per_suite_quality,
            test_on.per_suite_quality,
            regress_threshold=regress_threshold,
            require_aggregate_gain=require_aggregate_gain,
        )
        split_verdict = {
            "accept": v.accept,
            "aggregate_delta": v.aggregate_delta,
            "per_suite_delta": v.per_suite_delta,
            "regressed_suites": v.regressed_suites,
            "reason": v.reason,
        }

    # Model/quant are the index key (never role) — carry them at the top level.
    a_sample = next(iter(arms_by_split.values()))[1]
    return {
        "kind": "skill_efficacy_ab_paired_result",
        "runner_version": RUNNER_VERSION,
        "protocol_id": PROTOCOL_ID,
        "skill": skill,
        "model": a_sample.model,
        "quant": a_sample.quant,
        "splits": sorted(arms_by_split),
        "split_verdict": split_verdict,
        "per_split": per_split,
        "transport": _transport_summary(),
        "observation_only": True,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Plan resolution (pure, inference-free)
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class PairedPlan:
    corpus: CorpusResolution
    arm_map: dict[str, bool]
    splits: list[str]
    skill: str | None
    eval_role: str
    model: str
    quant: str
    n_per_arm: int
    seed: int
    regress_threshold: float
    require_aggregate_gain: bool
    per_suite_negative_delta_guard: bool
    jobs: list[dict[str, Any]]
    split_qids: dict[str, list[str]]
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "skill_efficacy_ab_plan",
            "runner_version": RUNNER_VERSION,
            "protocol_id": PROTOCOL_ID,
            "mode": "dry_run",
            "inference_ran": False,
            "skill": self.skill,
            "arms": [
                {"label": label, "skill_enabled": enabled}
                for label, enabled in self.arm_map.items()
            ],
            "splits": list(self.splits),
            "eval_role": self.eval_role,
            "model": self.model,
            "quant": self.quant,
            "n_per_arm": self.n_per_arm,
            "seed": self.seed,
            "regress_threshold": self.regress_threshold,
            "require_aggregate_gain": self.require_aggregate_gain,
            "per_suite_negative_delta_guard": self.per_suite_negative_delta_guard,
            "corpus": self.corpus.to_dict(),
            "split_sizes": {s: len(q) for s, q in self.split_qids.items()},
            "n_jobs": len(self.jobs),
            "jobs": list(self.jobs),
            "transport": _transport_summary(),
            "gate_env": SKILL_EFFICACY_GATE_ENV,
            "inference_env": SKILL_EFFICACY_AB_INFERENCE_ENV,
            "notes": list(self.notes),
            "observation_only": True,
        }


def resolve_paired_plan(
    *,
    corpus: CorpusResolution,
    arm_map: dict[str, bool],
    splits: list[str],
    skill: str | None,
    eval_role: str,
    model: str,
    quant: str,
    n_per_arm: int,
    seed: int,
    regress_threshold: float,
    require_aggregate_gain: bool,
    per_suite_negative_delta_guard: bool,
) -> PairedPlan:
    """Expand config + resolved corpus into a concrete paired A/B dry-run plan.

    Partitions the corpus into the requested splits and materializes one job per
    (split, arm) pinned to the placement-queue transport (NEVER /chat). Pure: no
    inference, no I/O beyond the in-memory corpus.
    """
    split_questions_map = (
        split_questions(corpus.questions, splits, seed=seed)
        if corpus.questions
        else {s: [] for s in splits}
    )
    split_qids = {s: [q["qid"] for q in qs] for s, qs in split_questions_map.items()}

    jobs: list[dict[str, Any]] = []
    for split in splits:
        qids = split_qids[split]
        profile = build_test_profile(
            split=split, seed=seed, eval_role=eval_role, n=n_per_arm
        )
        ds_sha = dataset_sha256(qids)
        for label, skill_enabled in arm_map.items():
            jobs.append(
                {
                    "kind": "skill_efficacy_ab_job",
                    "split": split,
                    "arm": label,
                    "skill": skill if skill_enabled else None,
                    "skill_enabled": skill_enabled,
                    "eval_role": eval_role,
                    "force_role": eval_role,
                    "model": model,
                    "quant": quant,
                    "n": min(n_per_arm, len(qids)) if qids else n_per_arm,
                    "n_available": len(qids),
                    "seed": seed,
                    "dataset_sha256": ds_sha,
                    "test_profile": profile,
                    "transport": PLACEMENT_QUEUE_TRANSPORT,
                    "request_priority": PLACEMENT_REQUEST_PRIORITY,
                    "workload_class": PLACEMENT_WORKLOAD_CLASS,
                }
            )

    notes: list[str] = list(corpus.notes)
    notes.append(
        "paired arms scored on IDENTICAL per-split question sets (dataset_sha256 "
        "matched); the skill presence is the only treatment."
    )
    notes.append(
        "placement-queue transport (background/eval_batch); NEVER a foreground /chat call."
    )
    notes.append(
        "all deltas/verdicts are pre-decision observations (MEASUREMENT.md); the "
        f"deployed gate is {SKILL_EFFICACY_GATE_ENV}."
    )
    if skill is None:
        notes.append(
            "--skill unset: this is an A/A neutrality probe (both arms identical); "
            "require_aggregate_gain is relaxed so a ~0 delta does not auto-reject."
        )
    if not corpus.exists:
        notes.append("CORPUS NOT RESOLVED: plan describes intended arms only.")

    return PairedPlan(
        corpus=corpus,
        arm_map=arm_map,
        splits=splits,
        skill=skill,
        eval_role=eval_role,
        model=model,
        quant=quant,
        n_per_arm=n_per_arm,
        seed=seed,
        regress_threshold=regress_threshold,
        require_aggregate_gain=require_aggregate_gain,
        per_suite_negative_delta_guard=per_suite_negative_delta_guard,
        jobs=jobs,
        split_qids=split_qids,
        notes=notes,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Execution bridge (env-gated; deferred EvalTower import; NEVER run in tests)
# ══════════════════════════════════════════════════════════════════════════════


def _default_tower() -> Any:  # pragma: no cover - inference path
    """Import + construct EvalTower (deferred, exactly like bsv_paired_runner)."""
    try:
        from scripts.autopilot.eval_tower import EvalTower
    except Exception:  # noqa: BLE001
        from eval_tower import EvalTower  # type: ignore[no-redef]
    return EvalTower()


def _default_arm_eval(
    *,
    label: str,
    split: str,
    skill_enabled: bool,
    model: str,
    quant: str,
    dataset_sha256: str,
    test_profile: str,
    n: int,
    seed: int,
    tower: Any,
) -> ArmOutcome:  # pragma: no cover - inference path
    """Score ONE arm over the eval tower and map EvalResult -> ArmOutcome.

    This is the real inference seam. It calls ``tower.eval_t1`` (which fans out
    over the placement queue with ``request_priority=background`` /
    ``workload_class=eval_batch`` internally — never a foreground /chat request)
    on the split's question set and reads ``per_suite_quality`` +
    ``question_results`` off the returned EvalResult. The skill-ON/OFF state is
    toggled by the caller (:func:`execute_paired_ab`) BEFORE this runs, so both
    arms share the identical questions/seed and differ only by the artifact.

    Never exercised by the unit tests (the whole execution bridge is env-gated and
    unreached under the zero-inference constraint).
    """
    result = tower.eval_t1(n=n, seed=seed)
    per_suite = dict(getattr(result, "per_suite_quality", {}) or {})
    question_results = list(getattr(result, "question_results", []) or [])
    return arm_outcome_from_question_results(
        label=label,
        split=split,
        skill_enabled=skill_enabled,
        model=model,
        quant=quant,
        dataset_sha256=dataset_sha256,
        test_profile=test_profile,
        question_results=question_results,
        per_suite_quality=per_suite or None,
    )


def execute_paired_ab(
    plan: PairedPlan,
    *,
    tower: Any | None = None,
    tower_factory: Callable[[], Any] | None = None,
    arm_eval_fn: Callable[..., ArmOutcome] | None = None,
    skill_apply_fn: Callable[[str], Any] | None = None,
    skill_revert_fn: Callable[[str], Any] | None = None,
    output_path: Path | None = None,
    require_matched: bool = True,
) -> dict[str, Any]:  # pragma: no cover - inference path
    """Drive both arms per split over the placement queue and compute paired stats.

    Reached ONLY when ``AUTOPILOT_SKILL_EFFICACY_AB_INFERENCE=1`` AND ``--run``.
    Autopilot-stopped assumption (bsv/screening pattern): the caller owns the
    no-concurrent-inference window; this function never touches autopilot lifecycle
    or state. For each split it scores the skill-OFF arm, applies the skill
    artifact (via ``skill_apply_fn``), scores the skill-ON arm on the SAME
    questions/seed, then reverts the artifact. Emits per-arm + per-split paired
    JSONL to ``output_path``; returns the combined dev/test split result.

    If ``plan.skill`` is set but no ``skill_apply_fn`` is injected, this raises —
    the runner does NOT re-implement the skill-install machinery (that lives in the
    action/species layer); an A/A neutrality probe (``--skill`` unset) needs no
    apply fn.
    """
    tower = tower or (tower_factory or _default_tower)()
    arm_eval_fn = arm_eval_fn or _default_arm_eval
    if plan.skill is not None and skill_apply_fn is None:
        raise RuntimeError(
            "skill A/B requires a skill_apply_fn (skill-install seam lives in the "
            "action/species layer); pass one, or drop --skill for an A/A probe"
        )

    arms_by_split: dict[str, tuple[ArmOutcome, ArmOutcome]] = {}
    for split in plan.splits:
        qids = plan.split_qids.get(split, [])
        n = min(plan.n_per_arm, len(qids)) if qids else plan.n_per_arm
        ds_sha = dataset_sha256(qids)
        profile = build_test_profile(
            split=split, seed=plan.seed, eval_role=plan.eval_role, n=plan.n_per_arm
        )

        off_arm = arm_eval_fn(
            label=ARM_SKILL_OFF, split=split, skill_enabled=False,
            model=plan.model, quant=plan.quant, dataset_sha256=ds_sha,
            test_profile=profile, n=n, seed=plan.seed, tower=tower,
        )
        applied = False
        if plan.skill is not None and skill_apply_fn is not None:
            skill_apply_fn(plan.skill)
            applied = True
        try:
            on_arm = arm_eval_fn(
                label=ARM_SKILL_ON, split=split, skill_enabled=True,
                model=plan.model, quant=plan.quant, dataset_sha256=ds_sha,
                test_profile=profile, n=n, seed=plan.seed, tower=tower,
            )
        finally:
            if applied and skill_revert_fn is not None:
                skill_revert_fn(plan.skill)
        arms_by_split[split] = (off_arm, on_arm)

    result = compute_split_paired_efficacy(
        arms_by_split,
        skill=plan.skill,
        regress_threshold=plan.regress_threshold,
        require_aggregate_gain=plan.require_aggregate_gain,
        require_matched=require_matched,
    )
    if output_path is not None:
        for split, res in result["per_split"].items():
            for arm_label, arm_summary in res["arms"].items():
                _append_jsonl(Path(output_path), {"kind": "skill_efficacy_ab_arm", **arm_summary})
            _append_jsonl(Path(output_path), res)
        _append_jsonl(Path(output_path), result)
    return result


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:  # pragma: no cover
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True, default=str) + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# Top-level orchestration (env-gated dry-run vs execute)
# ══════════════════════════════════════════════════════════════════════════════


def run_skill_efficacy_ab(
    plan: PairedPlan,
    *,
    output_path: Path | None = None,
    tower: Any | None = None,
    tower_factory: Callable[[], Any] | None = None,
    arm_eval_fn: Callable[..., ArmOutcome] | None = None,
    skill_apply_fn: Callable[[str], Any] | None = None,
    skill_revert_fn: Callable[[str], Any] | None = None,
    attempt_run: bool = False,
    require_matched: bool = True,
) -> dict[str, Any]:
    """Return the dry-run plan, OR (env-gated) execute the paired A/B.

    DEFAULT (``AUTOPILOT_SKILL_EFFICACY_AB_INFERENCE`` unset/false, or
    ``attempt_run=False``): returns the resolved plan and runs NO inference —
    the entire surface the unit tests exercise. Only when BOTH ``attempt_run`` and
    the env flag are set does it drive :func:`execute_paired_ab`.
    """
    if not (attempt_run and _env_flag_enabled(SKILL_EFFICACY_AB_INFERENCE_ENV)):
        payload = plan.to_dict()
        reason = "dry-run (no inference, placement-queue transport)"
        if attempt_run and not _env_flag_enabled(SKILL_EFFICACY_AB_INFERENCE_ENV):
            reason = f"--run given but {SKILL_EFFICACY_AB_INFERENCE_ENV} not set; falling back to dry-run"
        payload["reason"] = reason
        return payload

    result = execute_paired_ab(
        plan,
        tower=tower,
        tower_factory=tower_factory,
        arm_eval_fn=arm_eval_fn,
        skill_apply_fn=skill_apply_fn,
        skill_revert_fn=skill_revert_fn,
        output_path=output_path,
        require_matched=require_matched,
    )
    return {
        "mode": "execute",
        "runner_version": RUNNER_VERSION,
        "inference_ran": True,
        "output_path": str(output_path) if output_path else None,
        "result": result,
    }


# ══════════════════════════════════════════════════════════════════════════════
# CLI (__main__)
# ══════════════════════════════════════════════════════════════════════════════


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "EV-10a paired skill-ON vs skill-OFF efficacy A/B runner. Default is a "
            "pure dry-run that resolves the corpus, plans the paired arms, and runs "
            "NO inference. Real inference is gated behind --run AND "
            f"{SKILL_EFFICACY_AB_INFERENCE_ENV}=1."
        )
    )
    p.add_argument(
        "--arms",
        default=f"{ARM_SKILL_ON},{ARM_SKILL_OFF}",
        help="paired arm labels: one skill-ON, one skill-OFF (default with_artifact,no_artifact)",
    )
    p.add_argument(
        "--splits",
        default=",".join(DEFAULT_SPLITS),
        help="comma-separated splits scored on the same corpus (default dev,test_normal)",
    )
    p.add_argument(
        "--skill",
        default=None,
        help="the skill/artifact under test (id or path). Unset => A/A neutrality probe.",
    )
    p.add_argument("--core-id", default=None, help="paired-core id under benchmarks/prompts/")
    p.add_argument(
        "--questions",
        default=None,
        help="explicit corpus JSONL path (overrides --core-id)",
    )
    p.add_argument(
        "--suites",
        default=None,
        help="comma-separated suite filter (restrict the corpus to these suites)",
    )
    p.add_argument("--n", type=int, default=50, help="per-arm question budget")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-role", default=DEFAULT_EVAL_ROLE, help="role the arms evaluate (model/quant index)")
    p.add_argument("--model", default=None, help="override model index (else resolved from registry)")
    p.add_argument("--quant", default=None, help="override quant index (else resolved from registry)")
    p.add_argument("--regress-threshold", type=float, default=0.10)
    p.add_argument(
        "--per-suite-negative-delta-guard",
        action="store_true",
        help="affirm the SkillsBench per-suite negative-delta guard (always active in evaluate_skill_efficacy)",
    )
    p.add_argument(
        "--no-require-aggregate-gain",
        action="store_true",
        help="allow a neutral (~0) aggregate delta to accept (neutrality probe)",
    )
    p.add_argument("--output", default=None, help="JSONL path for per-arm + paired results (execute path only)")
    p.add_argument(
        "--run",
        action="store_true",
        help=f"attempt execution (STILL env-gated by {SKILL_EFFICACY_AB_INFERENCE_ENV}=1; else dry-run)",
    )
    return p


def _config_from_args(args: argparse.Namespace) -> PairedPlan:
    arm_map = parse_arms(args.arms)
    splits = [s.strip() for s in str(args.splits).split(",") if s.strip()]
    suites = (
        {s.strip() for s in str(args.suites).split(",") if s.strip()}
        if args.suites
        else None
    )
    corpus = resolve_corpus(
        core_id=args.core_id,
        questions_path=args.questions,
        suites=suites,
    )
    model, quant = resolve_model_quant(args.eval_role, model=args.model, quant=args.quant)
    # A/A neutrality probe (no skill) relaxes the strict-gain requirement so a ~0
    # delta does not auto-reject; a real skill A/B keeps strict gain unless opted out.
    require_aggregate_gain = not (args.no_require_aggregate_gain or args.skill is None)
    return resolve_paired_plan(
        corpus=corpus,
        arm_map=arm_map,
        splits=splits,
        skill=args.skill,
        eval_role=args.eval_role,
        model=model,
        quant=quant,
        n_per_arm=args.n,
        seed=args.seed,
        regress_threshold=args.regress_threshold,
        require_aggregate_gain=require_aggregate_gain,
        per_suite_negative_delta_guard=args.per_suite_negative_delta_guard,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        plan = _config_from_args(args)
    except ValueError as exc:
        print(json.dumps({"error": str(exc)}, indent=2))
        return 2

    if not args.run:
        print(json.dumps(plan.to_dict(), indent=2, sort_keys=True, default=str))
        return 0

    # --run: still env-gated; falls back to dry-run when the flag is unset.
    result = run_skill_efficacy_ab(
        plan,
        output_path=Path(args.output) if args.output else None,
        attempt_run=True,
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
