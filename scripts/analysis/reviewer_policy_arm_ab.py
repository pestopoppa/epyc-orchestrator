#!/usr/bin/env python3
"""LB-4 per-policy reviewer sampling A/B arm wiring.

A paired A/B driver whose two arms are two reviewer **sampling policies** — a
preset bundle of decode knobs (temperature / top-p / top-k / verbosity / token
budget) applied to the SAME reviewer model on the SAME questions + seed. It
compares, per policy:

  * **throughput** — tokens/sec and mean per-decision latency, and
  * **decision agreement** — how often the two policies land on the same
    :class:`ReviewDecision` (raw agreement + chance-corrected Cohen's kappa),
    plus (when the corpus carries a gold reviewer verdict) each policy's
    accuracy vs. that gold with a McNemar paired test.

This is the sampling-policy sibling of ``scripts/analysis/run_paired_ab.py``
(flag/params arms). It REUSES that harness's seams rather than re-deriving them:
the corpus resolver (``resolve_corpus`` — same-questions/seed pairing +
dataset_sha256), the placement-queue transport constants (RM-3), and the paired
statistics (``scripts/autopilot/paired_stats`` McNemar + the dataset/profile
gate, plus ``src/llm_primitives/stat_tests.wilson_interval``).

Execution is GATED (mirrors ``run_paired_ab.py`` / ``screening_tier_runner.py``):

  * DEFAULT = a pure dry-run **plan**. It validates the two policy specs,
    resolves the corpus, computes the transport + scoring profile, and prints
    the planned paired run. It does NOTHING that needs a model — no server, no
    ``/chat``, no placement-queue dispatch.
  * Real inference happens ONLY under ``--execute`` **and** the env gate
    ``REVIEWER_POLICY_AB_INFERENCE=1`` (default OFF). Without the env flag,
    ``--execute`` degrades to the dry-run plan.

Transport discipline (RM-3): every reviewer generation rides the **placement
queue** (``request_priority=background`` + ``workload_class=eval_batch``), NEVER
a foreground ``/chat`` request.

Indexing: all emitted results are **model/quant-indexed, never role-indexed**
(feedback_model_not_role_indexing) — the policy name and reviewer role are
metadata only. Because the two arms deliberately DIFFER in their sampling knobs,
the shared scoring ``test_profile`` EXCLUDES the per-policy sampling (it pins the
decision scheme + seed + dataset only) so the McNemar pairing stays valid.

All numbers this driver emits are pre-decision OBSERVATIONS (MEASUREMENT.md)
until the operator's gate table adjudicates them; the driver never gates a
keep/revert/promote on its own.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

SCRIPT_DIR = Path(__file__).resolve().parent
ORCH_ROOT = SCRIPT_DIR.parents[1]
for _p in (str(ORCH_ROOT), str(SCRIPT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

RUNNER_VERSION = "reviewer-policy-arm-ab-v1"

# Env flag gating real inference. Default OFF => dry-run, no model touched.
REVIEWER_POLICY_AB_INFERENCE_ENV = "REVIEWER_POLICY_AB_INFERENCE"

# Canonical reviewer decision labels (mirrors ReviewDecision in
# src/proactive_delegation/types.py — kept as local constants so this analysis
# runner never imports the FROZEN serving-path review_service module).
DEC_APPROVE = "approve"
DEC_REQUEST_CHANGES = "request_changes"
DEC_ESCALATE = "escalate"
DEC_REJECT = "reject"
DEC_REQUEST_EVIDENCE = "request_evidence"
DEC_REJECT_TO_EMPTY = "reject_to_empty"
DEC_ABSTAIN = "abstain"
CANONICAL_DECISIONS = (
    DEC_APPROVE,
    DEC_REQUEST_CHANGES,
    DEC_ESCALATE,
    DEC_REJECT,
    DEC_REQUEST_EVIDENCE,
    DEC_REJECT_TO_EMPTY,
    DEC_ABSTAIN,
)
# Reviewer default when a raw response cannot be parsed (mirrors review_service:
# default to request_changes on failure — the conservative, non-approving verdict).
DEFAULT_DECISION = DEC_REQUEST_CHANGES

_DECISION_ALIASES = {
    "approve": DEC_APPROVE,
    "approved": DEC_APPROVE,
    "a": DEC_APPROVE,
    "lgtm": DEC_APPROVE,
    "pass": DEC_APPROVE,
    "changes": DEC_REQUEST_CHANGES,
    "request_changes": DEC_REQUEST_CHANGES,
    "request changes": DEC_REQUEST_CHANGES,
    "c": DEC_REQUEST_CHANGES,
    "escalate": DEC_ESCALATE,
    "e": DEC_ESCALATE,
    "reject": DEC_REJECT,
    "r": DEC_REJECT,
    "request_evidence": DEC_REQUEST_EVIDENCE,
    "evidence": DEC_REQUEST_EVIDENCE,
    "reject_to_empty": DEC_REJECT_TO_EMPTY,
    "abstain": DEC_ABSTAIN,
}

# Sampling-policy knob domains.
VERBOSITY_PRESETS = ("terse", "normal", "verbose")
REASONING_EFFORTS = ("low", "medium", "high")

# Corpus row field holding the (optional) gold reviewer verdict for accuracy scoring.
DEFAULT_GOLD_KEY = "gold_decision"
DEFAULT_DECISION_SCHEME = "canonical_v1"


# ══════════════════════════════════════════════════════════════════════════════
# Seam reuse — load run_paired_ab (corpus resolver + placement constants + stats)
# ══════════════════════════════════════════════════════════════════════════════


def _load_paired_ab():
    """Load scripts/analysis/run_paired_ab.py (the sibling harness we mirror).

    We reuse its corpus resolver, placement-queue transport constants, and the
    paired-stats / wilson loaders rather than re-deriving them here.
    """
    if "run_paired_ab" in sys.modules:
        return sys.modules["run_paired_ab"]
    spec = importlib.util.spec_from_file_location(
        "run_paired_ab", str(SCRIPT_DIR / "run_paired_ab.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_paired_ab"] = mod
    spec.loader.exec_module(mod)
    return mod


_RPAB = _load_paired_ab()

# Placement-queue transport constants (RM-3) — reused from the sibling harness so
# both drivers ride the identical background/eval_batch placement path.
PLACEMENT_QUEUE_TRANSPORT = _RPAB.PLACEMENT_QUEUE_TRANSPORT
PLACEMENT_REQUEST_PRIORITY = _RPAB.PLACEMENT_REQUEST_PRIORITY
PLACEMENT_WORKLOAD_CLASS = _RPAB.PLACEMENT_WORKLOAD_CLASS

resolve_corpus = _RPAB.resolve_corpus
CorpusResolution = _RPAB.CorpusResolution
_env_flag_enabled = _RPAB._env_flag_enabled


# ══════════════════════════════════════════════════════════════════════════════
# Reviewer sampling policies (the two paired arms)
# ══════════════════════════════════════════════════════════════════════════════


def _coerce_knob(key: str, raw: str) -> Any:
    """Coerce + validate ONE sampling knob (raises ValueError on a bad value)."""
    key = key.strip()
    raw = raw.strip()
    if key in ("temperature", "top_p", "min_p"):
        val = float(raw)
        if key == "temperature" and not (0.0 <= val <= 2.0):
            raise ValueError(f"temperature must be in [0.0, 2.0], got {val}")
        if key in ("top_p", "min_p") and not (0.0 < val <= 1.0):
            raise ValueError(f"{key} must be in (0.0, 1.0], got {val}")
        return val
    if key in ("top_k", "max_tokens"):
        val_i = int(raw)
        if key == "max_tokens" and val_i <= 0:
            raise ValueError(f"max_tokens must be > 0, got {val_i}")
        if key == "top_k" and val_i < 0:
            raise ValueError(f"top_k must be >= 0, got {val_i}")
        return val_i
    if key == "verbosity":
        if raw not in VERBOSITY_PRESETS:
            raise ValueError(f"verbosity must be one of {VERBOSITY_PRESETS}, got {raw!r}")
        return raw
    if key == "reasoning_effort":
        if raw not in REASONING_EFFORTS:
            raise ValueError(
                f"reasoning_effort must be one of {REASONING_EFFORTS}, got {raw!r}"
            )
        return raw
    raise ValueError(
        f"unknown sampling knob {key!r}; allowed: temperature, top_p, min_p, top_k, "
        "max_tokens, verbosity, reasoning_effort"
    )


# Knobs that flow into the decode call vs. knobs that shape the reviewer prompt.
_DECODE_KNOBS = ("temperature", "top_p", "top_k", "min_p")


@dataclass
class SamplingPolicy:
    """One reviewer sampling policy = one arm of the paired comparison.

    ``knobs`` holds the validated decode/prompt knobs. ``role`` is the reviewer
    role the policy's inference is routed to over the placement queue (metadata;
    results are NEVER indexed by role). Both arms normally share ONE reviewer
    role — they differ only in their sampling knobs.
    """

    name: str
    knobs: dict[str, Any] = field(default_factory=dict)
    role: str | None = None
    is_baseline: bool = False

    def decode_kwargs(self) -> dict[str, Any]:
        """The subset of knobs passed to the reviewer decode call."""
        return {k: self.knobs[k] for k in _DECODE_KNOBS if k in self.knobs}

    def max_tokens(self) -> int | None:
        return self.knobs.get("max_tokens")

    def verbosity(self) -> str | None:
        return self.knobs.get("verbosity")

    def transport(self) -> dict[str, Any]:
        """Placement-queue transport + role binding for this arm (never /chat)."""
        return {
            "transport": PLACEMENT_QUEUE_TRANSPORT,
            "request_priority": PLACEMENT_REQUEST_PRIORITY,
            "workload_class": PLACEMENT_WORKLOAD_CLASS,
            "force_role": self.role or "",
            "uses_chat_endpoint": False,
        }

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["decode_kwargs"] = self.decode_kwargs()
        d["transport"] = self.transport()
        return d


def parse_policy_spec(spec: str) -> tuple[str, dict[str, Any]]:
    """Parse a ``--policy`` value ``NAME=KEY=VAL[,KEY2=VAL2...]`` -> (name, knobs).

    Example: ``cold=temperature=0.0,top_p=1.0,verbosity=terse,max_tokens=256``.
    An empty body (``NAME=``) is a valid policy with zero explicit knobs (the
    reviewer's registry defaults apply at execute time).
    """
    if "=" not in spec:
        raise ValueError(f"--policy must be NAME=KEY=VAL[,...], got {spec!r}")
    name, body = spec.split("=", 1)
    name = name.strip()
    body = body.strip()
    if not name:
        raise ValueError(f"--policy has empty policy name: {spec!r}")
    knobs: dict[str, Any] = {}
    if not body:
        return name, knobs
    for pair in body.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            raise ValueError(f"knob spec needs KEY=VAL, got {pair!r} in {spec!r}")
        k, v = pair.split("=", 1)
        knobs[k.strip()] = _coerce_knob(k, v)
    return name, knobs


def parse_policy_arms(
    arms: str,
    *,
    policy_specs: list[str] | None = None,
    baseline_arm: str | None = None,
    reviewer_role: str | None = None,
) -> tuple[SamplingPolicy, SamplingPolicy]:
    """Parse ``--arms A,B`` (+ ``--policy`` specs) into (candidate, baseline).

    ``A,B`` must be exactly two DISTINCT policy names. By convention ``A`` (first)
    is the candidate under test and ``B`` (second) is the control baseline, unless
    ``baseline_arm`` names which one is the control. A policy named in ``--arms``
    without a matching ``--policy`` spec is a valid zero-knob (registry-default)
    policy. Both arms are pinned to the SAME ``reviewer_role`` (they differ only
    in sampling).
    """
    names = [n.strip() for n in arms.split(",") if n.strip()]
    if len(names) != 2:
        raise ValueError(f"--arms must be exactly two comma-separated names, got {names!r}")
    if names[0] == names[1]:
        raise ValueError(f"--arms must be two DISTINCT names, got {names!r}")

    spec_map: dict[str, dict[str, Any]] = {}
    for spec in policy_specs or []:
        n, knobs = parse_policy_spec(spec)
        if n not in names:
            raise ValueError(f"--policy {n!r} is not one of --arms {names!r}")
        spec_map[n] = knobs

    baseline_name = baseline_arm.strip() if baseline_arm else names[1]
    if baseline_name not in names:
        raise ValueError(f"--baseline-arm {baseline_name!r} is not one of --arms {names!r}")

    built: dict[str, SamplingPolicy] = {}
    for n in names:
        built[n] = SamplingPolicy(
            name=n,
            knobs=spec_map.get(n, {}),
            role=reviewer_role,
            is_baseline=(n == baseline_name),
        )

    candidate_name = names[0] if names[1] == baseline_name else names[1]
    return built[candidate_name], built[baseline_name]


# ══════════════════════════════════════════════════════════════════════════════
# Reviewer decision extraction (pure) + per-question outcome
# ══════════════════════════════════════════════════════════════════════════════


def normalize_decision(value: Any) -> str:
    """Map a raw decision token to a canonical ReviewDecision label.

    Unknown / empty tokens fall back to :data:`DEFAULT_DECISION`.
    """
    key = str(value or "").strip().lower()
    if key in CANONICAL_DECISIONS:
        return key
    return _DECISION_ALIASES.get(key, DEFAULT_DECISION)


def extract_decision(raw: str) -> str:
    """Extract a canonical reviewer decision from a raw reviewer response.

    Accepts the abbreviated JSON verdict the reviewer emits (``{"d": "approve"}``
    or ``{"decision": "changes"}``) and, failing that, a bare/leading decision
    token. Unparseable input yields :data:`DEFAULT_DECISION` (the conservative
    non-approving verdict, mirroring review_service's failure default).
    """
    text = (raw or "").strip()
    if not text:
        return DEFAULT_DECISION
    # Try a JSON object with an abbreviated 'd' or full 'decision' key.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
            if isinstance(obj, dict):
                token = obj.get("d", obj.get("decision"))
                if token is not None:
                    return normalize_decision(token)
        except json.JSONDecodeError:
            pass
    # Bare token fallback: first non-empty line's leading word.
    first = text.splitlines()[0].strip().lower()
    return normalize_decision(first.split()[0] if first.split() else first)


@dataclass(frozen=True)
class PolicyOutcome:
    """One (policy, question) reviewer outcome — the paired unit of comparison."""

    qid: str
    suite: str
    decision: str
    tokens_out: int
    latency_ms: float
    correct: bool | None = None  # vs. gold reviewer verdict, if the corpus has one


# ══════════════════════════════════════════════════════════════════════════════
# Pure per-policy comparison math (throughput + decision agreement)
# ══════════════════════════════════════════════════════════════════════════════


def decision_distribution(outcomes: list[PolicyOutcome]) -> dict[str, int]:
    """Count of each canonical decision label a policy produced."""
    return dict(Counter(o.decision for o in outcomes))


def policy_throughput(outcomes: list[PolicyOutcome]) -> dict[str, Any]:
    """Aggregate throughput for one policy: tokens/sec + mean per-decision latency."""
    n = len(outcomes)
    total_tokens = sum(int(o.tokens_out) for o in outcomes)
    total_latency_ms = sum(float(o.latency_ms) for o in outcomes)
    tps = (total_tokens / (total_latency_ms / 1000.0)) if total_latency_ms > 0 else 0.0
    mean_latency = (total_latency_ms / n) if n else 0.0
    return {
        "n": n,
        "total_tokens_out": total_tokens,
        "total_latency_ms": round(total_latency_ms, 6),
        "tokens_per_second": round(tps, 6),
        "mean_latency_ms": round(mean_latency, 6),
    }


def agreement_rate(labels_a: list[str], labels_b: list[str]) -> tuple[int, float]:
    """(#positions where the two label vectors agree, agreement fraction)."""
    if len(labels_a) != len(labels_b):
        raise ValueError("agreement_rate needs equal-length paired label vectors")
    n = len(labels_a)
    if n == 0:
        return 0, 0.0
    agree = sum(1 for x, y in zip(labels_a, labels_b) if x == y)
    return agree, round(agree / n, 6)


def cohen_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    """Cohen's kappa — chance-corrected agreement between two paired label vectors.

    ``kappa = (p_o - p_e) / (1 - p_e)`` where ``p_o`` is observed agreement and
    ``p_e`` is the agreement expected from each rater's marginal label
    distribution. Returns 1.0 when both raters are constant AND identical, 0.0
    when perfectly-expected agreement can't be corrected (``p_e == 1``) without
    perfect observed agreement.
    """
    if len(labels_a) != len(labels_b):
        raise ValueError("cohen_kappa needs equal-length paired label vectors")
    n = len(labels_a)
    if n == 0:
        return 0.0
    _, p_o = agreement_rate(labels_a, labels_b)
    count_a = Counter(labels_a)
    count_b = Counter(labels_b)
    labels = set(count_a) | set(count_b)
    p_e = sum((count_a.get(l, 0) / n) * (count_b.get(l, 0) / n) for l in labels)
    if p_e >= 1.0:
        return 1.0 if p_o >= 1.0 else 0.0
    return round((p_o - p_e) / (1.0 - p_e), 6)


def build_scoring_profile(
    *, decision_scheme: str, seed: int, gold_key: str, dataset_sha256: str
) -> str:
    """Scoring profile shared by BOTH policies.

    Deliberately EXCLUDES the per-policy sampling knobs — the two arms differ
    only in their sampling, but the DECISION extraction + gold scoring must be
    identical for a valid McNemar pairing. (Contrast run_paired_ab, whose single
    shared sampling policy DOES appear in its profile.)
    """
    return (
        f"decision_scheme={decision_scheme};seed={seed};"
        f"gold_key={gold_key};dataset={dataset_sha256}"
    )


def compute_policy_comparison(
    baseline_outcomes: list[PolicyOutcome],
    candidate_outcomes: list[PolicyOutcome],
    *,
    baseline_label: str,
    candidate_label: str,
    dataset_sha256: str,
    test_profile: str,
    model: str,
    quant: str,
) -> dict[str, Any]:
    """Pair two reviewer sampling policies -> throughput + decision-agreement report.

    Always emits, over the shared question set:
      * per-policy **throughput** (tokens/sec, mean latency) + a candidate/baseline ratio,
      * inter-policy **decision agreement** (raw rate + Cohen's kappa) and each
        policy's decision distribution.
    When the corpus carries a gold reviewer verdict (``correct`` populated on the
    outcomes), it ALSO emits each policy's accuracy-vs-gold with Wilson 95% CIs and
    the McNemar exact two-sided p over the discordant pairs — reusing the
    ``paired_stats`` seam (guarded by ``require_matched_comparison``).

    Model/quant-indexed; the policy names are metadata, NEVER an index key.
    """
    ps = _RPAB._load_paired_stats()
    wilson_interval = _RPAB._load_wilson()

    # Profile gate: refuse to pair across mismatched dataset/scoring identity.
    ps.require_matched_comparison(
        {"dataset_sha256": dataset_sha256, "test_profile": test_profile},
        {"dataset_sha256": dataset_sha256, "test_profile": test_profile},
    )

    base_by_qid = {o.qid: o for o in baseline_outcomes}
    cand_by_qid = {o.qid: o for o in candidate_outcomes}
    shared = sorted(set(base_by_qid) & set(cand_by_qid))
    n = len(shared)

    base_labels = [base_by_qid[q].decision for q in shared]
    cand_labels = [cand_by_qid[q].decision for q in shared]
    agree_n, agree_rate = agreement_rate(base_labels, cand_labels)
    kappa = cohen_kappa(base_labels, cand_labels)

    base_shared = [base_by_qid[q] for q in shared]
    cand_shared = [cand_by_qid[q] for q in shared]
    base_tp = policy_throughput(base_shared)
    cand_tp = policy_throughput(cand_shared)
    base_tps = base_tp["tokens_per_second"]
    tps_ratio = round(cand_tp["tokens_per_second"] / base_tps, 6) if base_tps > 0 else None

    report: dict[str, Any] = {
        "kind": "reviewer_policy_ab_result",
        "runner_version": RUNNER_VERSION,
        "indexed_by": "model_quant",  # NEVER role (feedback_model_not_role_indexing)
        "model": model,
        "quant": quant,
        "model_quant_key": f"{model}/{quant}",
        "baseline_policy": baseline_label,
        "candidate_policy": candidate_label,
        "dataset_sha256": dataset_sha256,
        "test_profile": test_profile,
        "shared_qids": n,
        "decision_agreement": {
            "agree": agree_n,
            "rate": agree_rate,
            "cohen_kappa": kappa,
            "baseline_distribution": decision_distribution(base_shared),
            "candidate_distribution": decision_distribution(cand_shared),
        },
        "throughput": {
            "baseline": base_tp,
            "candidate": cand_tp,
            "candidate_over_baseline_tps": tps_ratio,
        },
        "has_gold": False,
        "observation_only": True,  # pre-decision (MEASUREMENT.md); never gates alone
    }

    # Accuracy-vs-gold path (McNemar + Wilson) only when the corpus has a gold verdict.
    if n and all(base_by_qid[q].correct is not None for q in shared) and all(
        cand_by_qid[q].correct is not None for q in shared
    ):
        base_vec = {
            q: ps.QuestionOutcome(
                qid=q, suite=base_by_qid[q].suite,
                correct=bool(base_by_qid[q].correct), trial_id=0,
            )
            for q in shared
        }
        cand_vec = {
            q: ps.QuestionOutcome(
                qid=q, suite=cand_by_qid[q].suite,
                correct=bool(cand_by_qid[q].correct), trial_id=1,
            )
            for q in shared
        }
        mcn = ps.mcnemar_from_vectors(
            base_vec, cand_vec, label_a=baseline_label, label_b=candidate_label
        )
        correct_base = sum(1 for q in shared if base_vec[q].correct)
        correct_cand = sum(1 for q in shared if cand_vec[q].correct)
        w_base = wilson_interval(correct_base, n)
        w_cand = wilson_interval(correct_cand, n)
        report["has_gold"] = True
        report["accuracy_vs_gold"] = {
            "baseline_correct": correct_base,
            "candidate_correct": correct_cand,
            "baseline_accuracy": mcn.accuracy_a,
            "candidate_accuracy": mcn.accuracy_b,
            "delta_candidate_minus_baseline": mcn.delta_b_minus_a,
            "baseline_wilson95": [round(w_base[0], 6), round(w_base[1], 6)],
            "candidate_wilson95": [round(w_cand[0], 6), round(w_cand[1], 6)],
            "p_value_two_sided": mcn.p_value_two_sided,
            "a_correct_b_wrong": mcn.a_correct_b_wrong,
            "a_wrong_b_correct": mcn.a_wrong_b_correct,
            "mcnemar": asdict(mcn),
        }
    return report


# ══════════════════════════════════════════════════════════════════════════════
# Per-policy result rows (model/quant-stamped, never role-keyed)
# ══════════════════════════════════════════════════════════════════════════════


def build_policy_rows(
    policy: SamplingPolicy,
    outcomes: list[PolicyOutcome],
    *,
    model: str,
    quant: str,
) -> list[dict[str, Any]]:
    """Serialize one policy's per-question outcomes to model/quant-stamped rows."""
    rows: list[dict[str, Any]] = []
    for o in outcomes:
        rows.append(
            {
                "qid": o.qid,
                "suite": o.suite,
                "policy": policy.name,
                "is_baseline": policy.is_baseline,
                "model": model,
                "quant": quant,
                "decision": o.decision,
                "tokens_out": o.tokens_out,
                "latency_ms": o.latency_ms,
                "correct": o.correct,
                "transport": PLACEMENT_QUEUE_TRANSPORT,
                "observation_only": True,
            }
        )
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# Plan (dry-run) — pure, model-free
# ══════════════════════════════════════════════════════════════════════════════


def build_plan(
    *,
    candidate: SamplingPolicy,
    baseline: SamplingPolicy,
    corpus: CorpusResolution,
    reviewer_role: str,
    decision_scheme: str,
    gold_key: str,
    output_dir: Path,
    seed: int,
    n: int,
    model: str,
    quant: str,
) -> dict[str, Any]:
    profile = build_scoring_profile(
        decision_scheme=decision_scheme,
        seed=seed,
        gold_key=gold_key,
        dataset_sha256=corpus.dataset_sha256,
    )
    return {
        "kind": "reviewer_policy_ab_plan",
        "runner_version": RUNNER_VERSION,
        "mode": "dry_run",
        "inference_required": True,
        "inference_ran": False,
        "indexed_by": "model_quant",
        "model": model,
        "quant": quant,
        "reviewer_role": reviewer_role,
        "decision_scheme": decision_scheme,
        "gold_key": gold_key,
        "seed": seed,
        "n": n,
        "policies": {
            "candidate": candidate.to_dict(),
            "baseline": baseline.to_dict(),
        },
        "corpus": corpus.to_dict(),
        "test_profile": profile,
        "transport": {
            "transport": PLACEMENT_QUEUE_TRANSPORT,
            "request_priority": PLACEMENT_REQUEST_PRIORITY,
            "workload_class": PLACEMENT_WORKLOAD_CLASS,
            "uses_chat_endpoint": False,
        },
        "metrics": ["throughput_tokens_per_second", "decision_agreement", "accuracy_vs_gold"],
        "output_dir": str(output_dir),
        "artifacts": {
            "baseline_jsonl": str(output_dir / f"{baseline.name}.jsonl"),
            "candidate_jsonl": str(output_dir / f"{candidate.name}.jsonl"),
            "comparison_report": str(output_dir / "reviewer_policy_ab_report.json"),
        },
        "notes": [
            "DRY-RUN: no model touched, no /chat, no placement-queue dispatch.",
            f"real inference requires --execute AND {REVIEWER_POLICY_AB_INFERENCE_ENV}=1.",
            "arms differ only in sampling knobs; scoring profile excludes them.",
            "all numbers are pre-decision observations (MEASUREMENT.md).",
        ],
    }


# ══════════════════════════════════════════════════════════════════════════════
# Execution bridge (env-gated; placement queue; NEVER run in tests)
# ══════════════════════════════════════════════════════════════════════════════


def _default_reviewer_probe(
    policy: SamplingPolicy, item: dict[str, Any], reviewer_role: str
) -> tuple[str, int, float]:  # pragma: no cover - inference path
    """Send ONE reviewer generation over the PLACEMENT QUEUE, return (raw, tokens, ms).

    Real inference seam. Reuses the SAME transport eval_tower uses internally —
    ``call_orchestrator_forced`` with ``request_priority=background`` +
    ``workload_class=eval_batch`` (placement queue), applying this policy's decode
    knobs and pinning ``force_role`` to the reviewer role — never a foreground
    ``/chat`` request. Never exercised by the unit tests (the bridge is env-gated).
    """
    import time

    _bench = str(Path("/mnt/raid0/llm/epyc-inference-research") / "scripts" / "benchmark")
    if _bench not in sys.path:
        sys.path.insert(0, _bench)
    from seeding_orchestrator import call_orchestrator_forced  # type: ignore

    start = time.perf_counter()
    resp = call_orchestrator_forced(
        prompt=str(item.get("prompt") or ""),
        force_role=policy.role or reviewer_role,
        force_mode="",
        url="http://localhost:8000",
        timeout=300,
        request_priority=PLACEMENT_REQUEST_PRIORITY,  # placement queue, not /chat
        workload_class=PLACEMENT_WORKLOAD_CLASS,
        **policy.decode_kwargs(),
    )
    latency_ms = (time.perf_counter() - start) * 1000.0
    answer = str(resp.get("answer") or "")
    tokens_out = int(resp.get("tokens_out") or resp.get("completion_tokens") or 0)
    return answer, tokens_out, latency_ms


def execute_paired_policy_ab(
    *,
    candidate: SamplingPolicy,
    baseline: SamplingPolicy,
    corpus: CorpusResolution,
    reviewer_role: str,
    decision_scheme: str,
    gold_key: str,
    output_dir: Path,
    seed: int,
    model: str,
    quant: str,
    reviewer_probe: Callable[..., tuple[str, int, float]] | None = None,
) -> dict[str, Any]:  # pragma: no cover - inference path
    """Drive both policies over the placement queue (same questions/seed) and emit
    the per-policy JSONL + the throughput/agreement/accuracy report. Reached ONLY
    under the env gate; the caller owns the no-concurrent-inference quiet window."""
    probe = reviewer_probe or _default_reviewer_probe
    output_dir.mkdir(parents=True, exist_ok=True)

    outcomes_by_policy: dict[str, list[PolicyOutcome]] = {}
    for policy in (baseline, candidate):
        outs: list[PolicyOutcome] = []
        for item in corpus.items:
            raw, tokens_out, latency_ms = probe(policy, item, reviewer_role)
            decision = extract_decision(raw)
            gold = item.get(gold_key)
            correct = None if gold is None else (decision == normalize_decision(gold))
            outs.append(
                PolicyOutcome(
                    qid=item["qid"],
                    suite=item.get("suite", ""),
                    decision=decision,
                    tokens_out=tokens_out,
                    latency_ms=latency_ms,
                    correct=correct,
                )
            )
        outcomes_by_policy[policy.name] = outs
        _write_jsonl(
            output_dir / f"{policy.name}.jsonl",
            build_policy_rows(policy, outs, model=model, quant=quant),
        )

    profile = build_scoring_profile(
        decision_scheme=decision_scheme, seed=seed, gold_key=gold_key,
        dataset_sha256=corpus.dataset_sha256,
    )
    result = compute_policy_comparison(
        outcomes_by_policy[baseline.name],
        outcomes_by_policy[candidate.name],
        baseline_label=baseline.name,
        candidate_label=candidate.name,
        dataset_sha256=corpus.dataset_sha256,
        test_profile=profile,
        model=model,
        quant=quant,
    )
    (output_dir / "reviewer_policy_ab_report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str) + "\n"
    )
    return {
        "mode": "execute",
        "runner_version": RUNNER_VERSION,
        "inference_ran": True,
        "output_dir": str(output_dir),
        "result": result,
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:  # pragma: no cover
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, default=str) + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# Top-level orchestration + CLI
# ══════════════════════════════════════════════════════════════════════════════


def run_policy_ab(args: argparse.Namespace) -> dict[str, Any]:
    """Resolve policies + corpus, then dry-run OR (env-gated) execute."""
    candidate, baseline = parse_policy_arms(
        args.arms,
        policy_specs=args.policy,
        baseline_arm=args.baseline_arm,
        reviewer_role=args.reviewer_role,
    )

    suites = args.suite.split(",") if args.suite else []
    corpus = resolve_corpus(
        manifest_path=args.manifest, suites=suites, n=args.n, seed=args.seed
    )

    output_dir = Path(args.output) if args.output else (
        ORCH_ROOT / "data" / "reviewer_policy_ab" / f"{candidate.name}__vs__{baseline.name}"
    )

    plan = build_plan(
        candidate=candidate,
        baseline=baseline,
        corpus=corpus,
        reviewer_role=args.reviewer_role,
        decision_scheme=args.decision_scheme,
        gold_key=args.gold_key,
        output_dir=output_dir,
        seed=args.seed,
        n=args.n,
        model=args.model,
        quant=args.quant,
    )

    want_execute = args.execute and _env_flag_enabled(REVIEWER_POLICY_AB_INFERENCE_ENV)
    if not want_execute:
        if args.execute and not _env_flag_enabled(REVIEWER_POLICY_AB_INFERENCE_ENV):
            plan["notes"].append(
                f"--execute requested but {REVIEWER_POLICY_AB_INFERENCE_ENV} not set; "
                "falling back to dry-run (no inference)."
            )
        if not corpus.resolved:
            plan["notes"].append(
                "corpus unresolved (suite-only) — provide --manifest to resolve rows."
            )
        return plan

    if not corpus.resolved:  # pragma: no cover - guarded before inference
        raise RuntimeError(
            "cannot execute: corpus unresolved (suite-only). Provide --manifest with rows."
        )
    return execute_paired_policy_ab(  # pragma: no cover - inference path
        candidate=candidate,
        baseline=baseline,
        corpus=corpus,
        reviewer_role=args.reviewer_role,
        decision_scheme=args.decision_scheme,
        gold_key=args.gold_key,
        output_dir=output_dir,
        seed=args.seed,
        model=args.model,
        quant=args.quant,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "LB-4 reviewer sampling-policy paired A/B. Two arms are two reviewer "
            "sampling policies (temperature/top-p/verbosity/token-budget presets) on "
            "the SAME reviewer model + questions + seed; compares throughput + "
            "decision agreement (+ accuracy vs gold). Default is a pure dry-run PLAN "
            "that runs NO inference; real inference needs --execute AND "
            f"{REVIEWER_POLICY_AB_INFERENCE_ENV}=1."
        )
    )
    p.add_argument(
        "--arms",
        required=True,
        help="exactly two comma-separated policy names, CANDIDATE,BASELINE "
        "(e.g. warm,cold or verbose,terse).",
    )
    p.add_argument(
        "--policy",
        action="append",
        default=[],
        help="repeatable NAME=KEY=VAL[,KEY2=VAL2...] sampling policy; knobs: "
        "temperature, top_p, min_p, top_k, max_tokens, verbosity, reasoning_effort. "
        "A policy named in --arms without a --policy spec uses registry defaults.",
    )
    p.add_argument(
        "--baseline-arm",
        default=None,
        help="which --arms name is the control baseline (default: the second name).",
    )
    p.add_argument(
        "--reviewer-role",
        default="architect",
        help="reviewer role both policies route to over the placement queue "
        "(metadata; results are model/quant-indexed, never role-indexed).",
    )
    p.add_argument(
        "--suite",
        default=None,
        help="comma-separated suite/domain names to select from the corpus.",
    )
    p.add_argument(
        "--manifest",
        default=None,
        help="path to a corpus manifest (JSON list / {items:[...]} / JSONL of task "
        "rows). Required for a concrete corpus; suite-only resolves at execute time.",
    )
    p.add_argument(
        "--gold-key",
        default=DEFAULT_GOLD_KEY,
        help=f"corpus row field holding the gold reviewer verdict (default "
        f"{DEFAULT_GOLD_KEY!r}); when absent, accuracy-vs-gold is skipped.",
    )
    p.add_argument(
        "--decision-scheme",
        default=DEFAULT_DECISION_SCHEME,
        help=f"decision-scheme tag for the shared scoring profile (default "
        f"{DEFAULT_DECISION_SCHEME!r}).",
    )
    p.add_argument("--n", type=int, default=50, help="cap on paired tasks per policy (same for both).")
    p.add_argument("--seed", type=int, default=42, help="shared seed (paired: same questions both arms).")
    p.add_argument("--model", default="unknown", help="model id for model/quant indexing (never role).")
    p.add_argument("--quant", default="unknown", help="quant id for model/quant indexing (never role).")
    p.add_argument("--output", default=None, help="output dir for per-policy JSONL + comparison report.")
    p.add_argument(
        "--execute",
        action="store_true",
        help="attempt real inference (STILL env-gated by "
        f"{REVIEWER_POLICY_AB_INFERENCE_ENV}=1; otherwise falls back to dry-run).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = run_policy_ab(args)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
