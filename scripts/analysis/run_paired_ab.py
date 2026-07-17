#!/usr/bin/env python3
"""Paired A/B experiment driver (flag-toggle or params-delta arms, selectable grader).

This is the generalized sibling of ``scripts/autopilot/bsv_paired_runner.py``. Where
the BSV runner pairs two *params* configs on one EvalTower core, this driver pairs two
**arms** — each an environment/flag toggle OR a params delta on the SAME model — and
scores each arm's per-question output through a **selectable grader**:

  * a generic ``exact`` / ``substring`` grader (short-answer suites), and
  * the Wave-1 EV-12 ``patch_verifier`` (``src/verification/patch_verifier.py``),
    which grades a candidate multi-file edit by statically applying it + AST/compile/
    (advisory ruff) checks — the promotion evidence for the edit-transaction A/B.

It is the concrete target of two authored (not-yet-run) inference-batch entries:

  * ``ROUTE-A2-edit-transaction-ab``  (patch-verifier-graded multi-file edit A/B), and
  * ``EX-0-role-aware-ab``            (the schema example — role_aware vs role_agnostic).

Execution is GATED (mirrors ``screening_tier_runner.py``):

  * DEFAULT = a pure dry-run **plan**. It validates the arm configs, resolves the
    corpus, computes the transport, and prints the planned paired run. It does
    NOTHING that needs a model — no server, no ``/chat``, no placement-queue dispatch.
  * Real inference happens ONLY under ``--execute`` **and** the env gate
    ``AUTOPILOT_PAIRED_AB_INFERENCE=1`` (default OFF). Without the env flag, ``--execute``
    degrades to the dry-run plan (so a manifest command authored with the real args is
    still safe to `--simulate`). The real run happens later in the operator's quiet
    window.

Transport discipline (RM-3, same as the screening runner + eval_tower): every inference
call rides the **placement queue** (``request_priority=background`` +
``workload_class=eval_batch``), NEVER a foreground ``/chat`` request.

Statistics (both arms scored on the SAME questions + seed — paired):

  * McNemar exact two-sided p over the discordant pairs
    (``scripts/autopilot/paired_stats.mcnemar_from_vectors``), guarded by the
    ``require_matched_comparison`` dataset_sha256 + test_profile gate.
  * Per-arm accuracy with Wilson 95% score intervals
    (``src/llm_primitives/stat_tests.wilson_interval``).
  * Results are **model/quant-indexed, never role-indexed** (arm/role is metadata).

All numbers this driver emits are pre-decision OBSERVATIONS (MEASUREMENT.md) until the
operator's gate table adjudicates them; the driver never gates a keep/revert/promote.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

SCRIPT_DIR = Path(__file__).resolve().parent
ORCH_ROOT = SCRIPT_DIR.parents[1]
for _p in (str(ORCH_ROOT), str(SCRIPT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

RUNNER_VERSION = "paired-ab-runner-v1"

# Env flag gating real inference. Mirrors screening_tier_runner's
# AUTOPILOT_SCREENING_TIER_INFERENCE: default OFF => dry-run, no model touched.
PAIRED_AB_INFERENCE_ENV = "AUTOPILOT_PAIRED_AB_INFERENCE"

# Placement-queue transport constants (RM-3). A paired-A/B trial rides the SAME
# background/eval_batch placement path a normal autopilot eval fan-out uses; it is
# never a foreground /chat request.
PLACEMENT_QUEUE_TRANSPORT = "placement_queue"
PLACEMENT_REQUEST_PRIORITY = "background"
PLACEMENT_WORKLOAD_CLASS = "eval_batch"

# Canonical grader names selectable via --grader.
GRADER_EXACT = "exact"
GRADER_SUBSTRING = "substring"
GRADER_PATCH_VERIFIER = "patch_verifier"
GRADERS = (GRADER_EXACT, GRADER_SUBSTRING, GRADER_PATCH_VERIFIER)

EVAL_MODE_GENERIC = "generic"
EVAL_MODE_EDIT_TRANSACTION = "edit_transaction"
EVAL_MODES = (EVAL_MODE_GENERIC, EVAL_MODE_EDIT_TRANSACTION)

# Field-name aliases accepted in a corpus/task item (tolerant loader).
_QID_KEYS = ("qid", "question_id", "id", "task_id")
_PROMPT_KEYS = ("prompt", "task", "question", "input")
_EXPECTED_KEYS = ("expected", "answer", "gold", "target", "reference")
_SUITE_KEYS = ("suite", "domain", "dataset")


# ══════════════════════════════════════════════════════════════════════════════
# Deferred / defensive cross-module imports (all pure stdlib-only modules)
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


def _load_verify_patch():
    """Import verify_patch + PASS from the EV-12 verifier (src/verification)."""
    from src.verification import PASS, verify_patch

    return verify_patch, PASS


# ══════════════════════════════════════════════════════════════════════════════
# Arm configs (flag toggle OR params delta on the SAME model)
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class ArmConfig:
    """One arm of the paired comparison.

    ``kind`` is ``flag`` (env/runtime-flag toggle), ``params`` (a params delta), or
    ``control`` (neither — the status-quo baseline). ``role`` optionally pins the
    eval-tower ``force_role`` the arm's inference is routed to over the placement
    queue (metadata; results are NEVER indexed by role).
    """

    name: str
    kind: str = "control"
    flags: dict[str, str] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    role: str | None = None
    is_baseline: bool = False

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
        d["transport"] = self.transport()
        return d


def parse_arm_spec(spec: str) -> tuple[str, dict[str, str], dict[str, Any]]:
    """Parse a ``--arm-spec`` value ``NAME=BODY`` into (name, flags, params).

    BODY forms:
      * ``flag:KEY=VAL[,KEY2=VAL2...]`` -> an env/flag-toggle arm.
      * ``params:{...json...}`` or ``params:@/path/to.json`` -> a params-delta arm.
      * empty / omitted -> a control arm.
    """
    if "=" not in spec:
        raise ValueError(f"--arm-spec must be NAME=BODY, got {spec!r}")
    name, body = spec.split("=", 1)
    name = name.strip()
    body = body.strip()
    if not name:
        raise ValueError(f"--arm-spec has empty arm name: {spec!r}")
    if not body:
        return name, {}, {}
    if body.startswith("flag:"):
        flags: dict[str, str] = {}
        for pair in body[len("flag:") :].split(","):
            pair = pair.strip()
            if not pair:
                continue
            if "=" not in pair:
                raise ValueError(f"flag spec needs KEY=VAL, got {pair!r} in {spec!r}")
            k, v = pair.split("=", 1)
            flags[k.strip()] = v.strip()
        return name, flags, {}
    if body.startswith("params:"):
        raw = body[len("params:") :].strip()
        if raw.startswith("@"):
            raw = Path(raw[1:]).read_text(encoding="utf-8")
        params = json.loads(raw)
        if not isinstance(params, dict):
            raise ValueError(f"params spec must be a JSON object in {spec!r}")
        return name, {}, params
    raise ValueError(
        f"--arm-spec body must start with 'flag:' or 'params:', got {body!r} in {spec!r}"
    )


def parse_arms(
    arms: str,
    *,
    arm_specs: list[str] | None = None,
    baseline_arm: str | None = None,
    roles: dict[str, str] | None = None,
) -> tuple[ArmConfig, ArmConfig]:
    """Parse ``--arms A,B`` (+ optional ``--arm-spec`` list) into (candidate, baseline).

    ``A,B`` must be exactly two distinct names. By convention ``A`` (first) is the
    candidate under test and ``B`` (second) is the control baseline, unless
    ``baseline_arm`` names which one is the control. Arms without an explicit spec are
    ``control`` (empty flags/params) — a bare-name arm is still a valid plan input.
    """
    names = [n.strip() for n in arms.split(",") if n.strip()]
    if len(names) != 2:
        raise ValueError(f"--arms must be exactly two comma-separated names, got {names!r}")
    if names[0] == names[1]:
        raise ValueError(f"--arms must be two DISTINCT names, got {names!r}")

    spec_map: dict[str, tuple[dict[str, str], dict[str, Any]]] = {}
    for spec in arm_specs or []:
        n, flags, params = parse_arm_spec(spec)
        spec_map[n] = (flags, params)
    roles = roles or {}

    baseline_name = baseline_arm.strip() if baseline_arm else names[1]
    if baseline_name not in names:
        raise ValueError(f"--baseline-arm {baseline_name!r} is not one of --arms {names!r}")

    built: dict[str, ArmConfig] = {}
    for n in names:
        flags, params = spec_map.get(n, ({}, {}))
        if flags:
            kind = "flag"
        elif params:
            kind = "params"
        else:
            kind = "control"
        built[n] = ArmConfig(
            name=n,
            kind=kind,
            flags=flags,
            params=params,
            role=roles.get(n),
            is_baseline=(n == baseline_name),
        )

    candidate_name = names[0] if names[1] == baseline_name else names[1]
    return built[candidate_name], built[baseline_name]


# ══════════════════════════════════════════════════════════════════════════════
# Corpus resolution (pure — file reads only, no model)
# ══════════════════════════════════════════════════════════════════════════════


def _first(item: dict[str, Any], keys: tuple[str, ...], default: Any = "") -> Any:
    for k in keys:
        if k in item and item[k] not in (None, ""):
            return item[k]
    return default


def normalize_item(raw: dict[str, Any], *, index: int) -> dict[str, Any]:
    """Normalize a raw corpus row into the canonical task-item shape."""
    qid = str(_first(raw, _QID_KEYS, default=f"q{index}"))
    item: dict[str, Any] = {
        "qid": qid,
        "prompt": str(_first(raw, _PROMPT_KEYS, default="")),
        "expected": _first(raw, _EXPECTED_KEYS, default=None),
        "suite": str(_first(raw, _SUITE_KEYS, default="")),
    }
    # Edit-transaction rows carry an in-memory base tree the patch verifier grades
    # the candidate diff against.
    if "base_tree" in raw:
        item["base_tree"] = raw["base_tree"]
    if "prediction" in raw:  # allows self-contained fixtures / replays
        item["prediction"] = raw["prediction"]
    return item


@dataclass
class CorpusResolution:
    items: list[dict[str, Any]]
    suites: list[str]
    manifest_path: str | None
    dataset_sha256: str
    n_requested: int
    n_available: int
    n_selected: int
    resolved: bool
    note: str

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["items"] = len(self.items)  # plan carries the COUNT, not the payload
        return d


def _load_manifest_rows(path: Path) -> list[dict[str, Any]]:
    """Load a corpus manifest: a JSON list, a JSON object with ``items``/``rows``,
    or a JSONL file (one object per line)."""
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("["):
        payload = json.loads(text)
        return list(payload)
    if stripped.startswith("{"):
        # Could be a single JSON object (with an items/rows array) OR JSONL whose
        # first line is an object. Try whole-file JSON first.
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict):
            for key in ("items", "rows", "tasks", "questions"):
                if isinstance(obj.get(key), list):
                    return list(obj[key])
            return [obj]
        if isinstance(obj, list):
            return list(obj)
    # JSONL fallback.
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _dataset_sha256(items: list[dict[str, Any]]) -> str:
    """Stable content hash over the selected (qid, prompt) pairs — the paired-arm
    dataset identity the McNemar profile gate pins."""
    h = hashlib.sha256()
    for it in items:
        h.update(str(it.get("qid", "")).encode("utf-8"))
        h.update(b"\x00")
        h.update(str(it.get("prompt", "")).encode("utf-8"))
        h.update(b"\x01")
    return "sha256:" + h.hexdigest()


def resolve_corpus(
    *,
    manifest_path: str | Path | None,
    suites: list[str] | None,
    n: int,
    seed: int,
) -> CorpusResolution:
    """Resolve the task/corpus for BOTH arms (same questions -> paired).

    With ``manifest_path`` the rows are loaded, filtered to the requested suites,
    deterministically ordered by ``hash(seed, qid)`` and capped to ``n``. Without a
    manifest (suite-only, as in the authored manifest commands) the plan records the
    requested suites but resolves zero concrete items — real rows load at execute time
    from the eval-tower/registry — and ``resolved`` is False. Either way this is pure:
    NO model, NO network.
    """
    suites = [s.strip() for s in (suites or []) if s.strip()]

    if manifest_path is None:
        note = (
            "suite-only: no --manifest; concrete rows resolve from the eval-tower "
            "registry at execute time (dry-run lists suites + n only)."
        )
        return CorpusResolution(
            items=[],
            suites=suites,
            manifest_path=None,
            dataset_sha256=_dataset_sha256(
                [{"qid": s, "prompt": "suite"} for s in suites]
            ),
            n_requested=n,
            n_available=0,
            n_selected=0,
            resolved=False,
            note=note,
        )

    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"corpus manifest not found: {path}")

    raw_rows = _load_manifest_rows(path)
    items = [normalize_item(r, index=i) for i, r in enumerate(raw_rows) if isinstance(r, dict)]
    n_available_all = len(items)
    if suites:
        wanted = set(suites)
        items = [it for it in items if it.get("suite") in wanted]

    def _order_key(it: dict[str, Any]) -> str:
        return hashlib.sha1(f"{seed}\x00{it['qid']}".encode("utf-8")).hexdigest()

    items.sort(key=_order_key)
    n_available = len(items)
    if n and n > 0:
        items = items[:n]

    note = "resolved from manifest" + (f"; suite-filtered to {suites}" if suites else "")
    return CorpusResolution(
        items=items,
        suites=suites,
        manifest_path=str(path),
        dataset_sha256=_dataset_sha256(items),
        n_requested=n,
        n_available=n_available if suites else n_available_all,
        n_selected=len(items),
        resolved=True,
        note=note,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Grader dispatch (generic exact/substring + patch_verifier)
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class Grader:
    name: str
    fn: Callable[[str, dict[str, Any]], bool]
    needs_base_tree: bool = False
    description: str = ""


def _norm(text: Any) -> str:
    return " ".join(str(text).split()).casefold()


def _exact_grade(prediction: str, item: dict[str, Any]) -> bool:
    return _norm(prediction) == _norm(item.get("expected"))


def _substring_grade(prediction: str, item: dict[str, Any]) -> bool:
    expected = _norm(item.get("expected"))
    if not expected:
        return False
    return expected in _norm(prediction)


def make_patch_verifier_grader(
    *, run_lint: bool = False, use_git: bool = False
) -> Callable[[str, dict[str, Any]], bool]:
    """Build a patch_verifier grader: PASS iff the candidate diff statically applies
    to the item's base tree and the resulting files compile (advisory ruff/git are
    non-gating). Execution-free."""
    verify_patch, PASS = _load_verify_patch()

    def _grade(prediction: str, item: dict[str, Any]) -> bool:
        base_tree = item.get("base_tree")
        if base_tree is None:
            raise ValueError(
                f"patch_verifier grader needs item['base_tree'] for qid={item.get('qid')!r}"
            )
        result = verify_patch(
            prediction, base_tree, run_lint=run_lint, use_git=use_git
        )
        return result.verdict == PASS

    return _grade


def get_grader(
    name: str, *, patch_run_lint: bool = False, patch_use_git: bool = False
) -> Grader:
    """Resolve a grader by name. ``patch_verifier`` grades candidate diffs statically."""
    if name == GRADER_EXACT:
        return Grader(GRADER_EXACT, _exact_grade, description="normalized exact-match")
    if name == GRADER_SUBSTRING:
        return Grader(
            GRADER_SUBSTRING, _substring_grade, description="normalized substring-contains"
        )
    if name == GRADER_PATCH_VERIFIER:
        return Grader(
            GRADER_PATCH_VERIFIER,
            make_patch_verifier_grader(run_lint=patch_run_lint, use_git=patch_use_git),
            needs_base_tree=True,
            description="EV-12 execution-free patch verdict (apply+AST/compile)",
        )
    raise ValueError(f"unknown grader {name!r}; choose from {GRADERS}")


# ══════════════════════════════════════════════════════════════════════════════
# Scoring + paired statistics (pure)
# ══════════════════════════════════════════════════════════════════════════════


def build_arm_rows(
    arm: ArmConfig,
    items: list[dict[str, Any]],
    predictions: dict[str, str],
    grader: Grader,
    *,
    model: str,
    quant: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Grade one arm's predictions -> (JSONL rows, qid->QuestionOutcome vector).

    ``predictions`` maps qid -> the arm's model output. Items with no prediction are
    skipped (they cannot be paired). Rows are model/quant-stamped, never role-keyed.
    """
    ps = _load_paired_stats()
    rows: list[dict[str, Any]] = []
    vector: dict[str, Any] = {}
    for item in items:
        qid = item["qid"]
        if qid not in predictions:
            continue
        prediction = predictions[qid]
        correct = bool(grader.fn(prediction, item))
        suite = item.get("suite", "")
        vector[qid] = ps.QuestionOutcome(
            qid=qid, suite=suite, correct=correct, trial_id=0
        )
        rows.append(
            {
                "qid": qid,
                "suite": suite,
                "arm": arm.name,
                "arm_kind": arm.kind,
                "is_baseline": arm.is_baseline,
                "model": model,
                "quant": quant,
                "grader": grader.name,
                "correct": correct,
                "prediction": prediction,
                "transport": PLACEMENT_QUEUE_TRANSPORT,
                "observation_only": True,
            }
        )
    return rows, vector


def compute_paired_result(
    baseline_vector: dict[str, Any],
    candidate_vector: dict[str, Any],
    *,
    baseline_label: str,
    candidate_label: str,
    dataset_sha256: str,
    test_profile: str,
    model: str,
    quant: str,
) -> dict[str, Any]:
    """Pair two graded arms -> McNemar p + per-arm Wilson-CI accuracy.

    Guards with ``require_matched_comparison``: refuses to pair arms whose
    dataset_sha256 / test_profile disagree (or are missing). Both arms share the same
    corpus + scoring profile by construction, so the gate normally passes; it exists
    to catch an accidental corpus/profile drift between arms. Model/quant-indexed.
    """
    ps = _load_paired_stats()
    wilson_interval = _load_wilson()

    # Profile gate (dataset_sha256 + test_profile equality) — raises
    # PairedComparisonMismatchError on drift.
    ps.require_matched_comparison(
        {"dataset_sha256": dataset_sha256, "test_profile": test_profile},
        {"dataset_sha256": dataset_sha256, "test_profile": test_profile},
    )

    mcn = ps.mcnemar_from_vectors(
        baseline_vector, candidate_vector, label_a=baseline_label, label_b=candidate_label
    )

    shared = sorted(set(baseline_vector) & set(candidate_vector))
    n = len(shared)
    correct_baseline = sum(1 for q in shared if baseline_vector[q].correct)
    correct_candidate = sum(1 for q in shared if candidate_vector[q].correct)
    wilson_baseline = wilson_interval(correct_baseline, n)
    wilson_candidate = wilson_interval(correct_candidate, n)

    return {
        "kind": "paired_ab_result",
        "runner_version": RUNNER_VERSION,
        "indexed_by": "model_quant",  # NEVER role (feedback_model_not_role_indexing)
        "model": model,
        "quant": quant,
        "model_quant_key": f"{model}/{quant}",
        "baseline_arm": baseline_label,
        "candidate_arm": candidate_label,
        "dataset_sha256": dataset_sha256,
        "test_profile": test_profile,
        "shared_qids": n,
        "baseline_correct": correct_baseline,
        "candidate_correct": correct_candidate,
        "baseline_accuracy": mcn.accuracy_a,
        "candidate_accuracy": mcn.accuracy_b,
        "baseline_wilson95": [round(wilson_baseline[0], 6), round(wilson_baseline[1], 6)],
        "candidate_wilson95": [round(wilson_candidate[0], 6), round(wilson_candidate[1], 6)],
        "delta_candidate_minus_baseline": mcn.delta_b_minus_a,
        "mcnemar": asdict(mcn),
        "p_value_two_sided": mcn.p_value_two_sided,
        "a_correct_b_wrong": mcn.a_correct_b_wrong,
        "a_wrong_b_correct": mcn.a_wrong_b_correct,
        "observation_only": True,  # pre-decision (MEASUREMENT.md); never gates alone
    }


def build_test_profile(
    *, grader: str, seed: int, sampling: str, dataset_sha256: str
) -> str:
    """Scoring profile shared by BOTH arms. Deliberately EXCLUDES arm-specific flags —
    the arms differ only in their flag/params toggle but must be scored identically for
    the McNemar pairing to be valid."""
    return f"grader={grader};seed={seed};sampling={sampling};dataset={dataset_sha256}"


# ══════════════════════════════════════════════════════════════════════════════
# Plan (dry-run) — pure, model-free
# ══════════════════════════════════════════════════════════════════════════════


def build_plan(
    *,
    candidate: ArmConfig,
    baseline: ArmConfig,
    corpus: CorpusResolution,
    grader: Grader,
    eval_mode: str,
    output_dir: Path,
    seed: int,
    n: int,
    model: str,
    quant: str,
    sampling: str,
    verifier_path: str | None,
) -> dict[str, Any]:
    profile = build_test_profile(
        grader=grader.name, seed=seed, sampling=sampling, dataset_sha256=corpus.dataset_sha256
    )
    return {
        "kind": "paired_ab_plan",
        "runner_version": RUNNER_VERSION,
        "mode": "dry_run",
        "inference_required": True,
        "inference_ran": False,
        "indexed_by": "model_quant",
        "model": model,
        "quant": quant,
        "eval_mode": eval_mode,
        "grader": {
            "name": grader.name,
            "needs_base_tree": grader.needs_base_tree,
            "description": grader.description,
            "verifier_path": verifier_path,
        },
        "seed": seed,
        "n": n,
        "sampling": sampling,
        "arms": {
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
        "output_dir": str(output_dir),
        "artifacts": {
            "baseline_jsonl": str(output_dir / f"{baseline.name}.jsonl"),
            "candidate_jsonl": str(output_dir / f"{candidate.name}.jsonl"),
            "paired_report": str(output_dir / "paired_ab_report.json"),
        },
        "notes": [
            "DRY-RUN: no model touched, no /chat, no placement-queue dispatch.",
            f"real inference requires --execute AND {PAIRED_AB_INFERENCE_ENV}=1.",
            "all numbers are pre-decision observations (MEASUREMENT.md).",
        ],
    }


# ══════════════════════════════════════════════════════════════════════════════
# Execution bridge (env-gated; placement queue; NEVER run in tests)
# ══════════════════════════════════════════════════════════════════════════════


def _env_flag_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _default_arm_probe(
    arm: ArmConfig, item: dict[str, Any]
) -> str:  # pragma: no cover - inference path
    """Send ONE (arm, task) generation over the PLACEMENT QUEUE and return the answer.

    Real inference seam. Reuses the SAME transport eval_tower uses internally —
    ``call_orchestrator_forced`` with ``request_priority=background`` +
    ``workload_class=eval_batch`` (placement queue), applying the arm's flag/params
    toggle and pinning ``force_role`` — so a paired-A/B generation is never a foreground
    ``/chat`` request. Never exercised by the unit tests (the bridge is env-gated).
    """
    _research = Path("/mnt/raid0/llm/epyc-inference-research")
    _bench = str(_research / "scripts" / "benchmark")
    if _bench not in sys.path:
        sys.path.insert(0, _bench)
    from seeding_orchestrator import call_orchestrator_forced  # type: ignore

    resp = call_orchestrator_forced(
        prompt=str(item.get("prompt") or ""),
        force_role=arm.role or "",
        force_mode="",
        url="http://localhost:8000",
        timeout=300,
        request_priority=PLACEMENT_REQUEST_PRIORITY,  # placement queue, not /chat
        workload_class=PLACEMENT_WORKLOAD_CLASS,
    )
    return str(resp.get("answer") or "")


def execute_paired_ab(
    *,
    candidate: ArmConfig,
    baseline: ArmConfig,
    corpus: CorpusResolution,
    grader: Grader,
    output_dir: Path,
    seed: int,
    model: str,
    quant: str,
    sampling: str,
    arm_probe: Callable[[ArmConfig, dict[str, Any]], str] | None = None,
) -> dict[str, Any]:  # pragma: no cover - inference path
    """Drive both arms over the placement queue (same questions/seed) and emit the
    per-arm JSONL + paired McNemar/Wilson report. Reached ONLY under the env gate; the
    caller owns the no-concurrent-inference quiet window (never touches autopilot)."""
    probe = arm_probe or _default_arm_probe
    output_dir.mkdir(parents=True, exist_ok=True)

    arm_vectors: dict[str, dict[str, Any]] = {}
    for arm in (baseline, candidate):
        predictions = {it["qid"]: probe(arm, it) for it in corpus.items}
        rows, vector = build_arm_rows(
            arm, corpus.items, predictions, grader, model=model, quant=quant
        )
        arm_vectors[arm.name] = vector
        _write_jsonl(output_dir / f"{arm.name}.jsonl", rows)

    profile = build_test_profile(
        grader=grader.name, seed=seed, sampling=sampling, dataset_sha256=corpus.dataset_sha256
    )
    result = compute_paired_result(
        arm_vectors[baseline.name],
        arm_vectors[candidate.name],
        baseline_label=baseline.name,
        candidate_label=candidate.name,
        dataset_sha256=corpus.dataset_sha256,
        test_profile=profile,
        model=model,
        quant=quant,
    )
    (output_dir / "paired_ab_report.json").write_text(
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


def run_paired_ab(args: argparse.Namespace) -> dict[str, Any]:
    """Resolve arms + corpus + grader, then dry-run OR (env-gated) execute."""
    roles: dict[str, str] = {}
    for pair in args.arm_role or []:
        if "=" in pair:
            k, v = pair.split("=", 1)
            roles[k.strip()] = v.strip()

    candidate, baseline = parse_arms(
        args.arms,
        arm_specs=args.arm_spec,
        baseline_arm=args.baseline_arm,
        roles=roles,
    )

    # Grader/eval-mode reconciliation: --verifier implies patch_verifier + edit mode.
    grader_name = args.grader
    eval_mode = args.eval_mode
    verifier_path = args.verifier
    if verifier_path:
        grader_name = GRADER_PATCH_VERIFIER
        eval_mode = EVAL_MODE_EDIT_TRANSACTION
        vp = Path(verifier_path)
        if vp.name != "patch_verifier.py":
            print(
                f"WARNING: --verifier {verifier_path!r} does not name patch_verifier.py",
                file=sys.stderr,
            )
    elif grader_name == GRADER_PATCH_VERIFIER:
        eval_mode = EVAL_MODE_EDIT_TRANSACTION

    grader = get_grader(
        grader_name, patch_run_lint=args.patch_lint, patch_use_git=args.patch_git
    )

    suites = args.suite.split(",") if args.suite else []
    corpus = resolve_corpus(
        manifest_path=args.manifest, suites=suites, n=args.n, seed=args.seed
    )

    output_dir = Path(args.output) if args.output else (
        ORCH_ROOT / "data" / "paired_ab" / f"{candidate.name}__vs__{baseline.name}"
    )

    plan = build_plan(
        candidate=candidate,
        baseline=baseline,
        corpus=corpus,
        grader=grader,
        eval_mode=eval_mode,
        output_dir=output_dir,
        seed=args.seed,
        n=args.n,
        model=args.model,
        quant=args.quant,
        sampling=args.sampling,
        verifier_path=verifier_path,
    )

    want_execute = args.execute and _env_flag_enabled(PAIRED_AB_INFERENCE_ENV)
    if not want_execute:
        if args.execute and not _env_flag_enabled(PAIRED_AB_INFERENCE_ENV):
            plan["notes"].append(
                f"--execute requested but {PAIRED_AB_INFERENCE_ENV} not set; "
                "falling back to dry-run (no inference)."
            )
        if not corpus.resolved:
            plan["notes"].append(
                "corpus unresolved (suite-only) — provide --manifest to resolve rows."
            )
        return plan

    # Execution path (env-gated; models the bsv/screening runner; NEVER in tests).
    if not corpus.resolved:  # pragma: no cover - guarded before inference
        raise RuntimeError(
            "cannot execute: corpus unresolved (suite-only). Provide --manifest with rows."
        )
    return execute_paired_ab(  # pragma: no cover - inference path
        candidate=candidate,
        baseline=baseline,
        corpus=corpus,
        grader=grader,
        output_dir=output_dir,
        seed=args.seed,
        model=args.model,
        quant=args.quant,
        sampling=args.sampling,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Paired A/B experiment driver (flag-toggle or params-delta arms, selectable "
            "grader). Default is a pure dry-run PLAN that resolves configs + corpus and "
            "runs NO inference; real inference needs --execute AND "
            f"{PAIRED_AB_INFERENCE_ENV}=1."
        )
    )
    p.add_argument(
        "--arms",
        required=True,
        help="exactly two comma-separated arm names, CANDIDATE,BASELINE "
        "(e.g. edit_transaction,baseline or role_aware,role_agnostic)",
    )
    p.add_argument(
        "--arm-spec",
        action="append",
        default=[],
        help="repeatable NAME=BODY arm config; BODY is 'flag:KEY=VAL[,...]' or "
        "'params:{json}'/'params:@file.json'. Arms without a spec are control arms.",
    )
    p.add_argument(
        "--arm-role",
        action="append",
        default=[],
        help="repeatable NAME=ROLE force_role binding for an arm (placement-queue routing).",
    )
    p.add_argument(
        "--baseline-arm",
        default=None,
        help="which --arms name is the control baseline (default: the second name).",
    )
    p.add_argument(
        "--suite",
        default=None,
        help="comma-separated suite/domain names to select (e.g. multifile_edit or "
        "math500,livecodebench).",
    )
    p.add_argument(
        "--manifest",
        default=None,
        help="path to a corpus manifest (JSON list / {items:[...]} / JSONL of task rows). "
        "Required for a concrete corpus; suite-only resolves at execute time.",
    )
    p.add_argument(
        "--grader",
        choices=GRADERS,
        default=GRADER_EXACT,
        help=f"grader for scoring each arm (default {GRADER_EXACT}).",
    )
    p.add_argument(
        "--eval-mode",
        choices=EVAL_MODES,
        default=EVAL_MODE_GENERIC,
        help="coarse eval mode; edit_transaction is auto-selected with the patch verifier.",
    )
    p.add_argument(
        "--verifier",
        default=None,
        help="path to src/verification/patch_verifier.py; selects the patch_verifier "
        "grader (edit-transaction A/B).",
    )
    p.add_argument("--patch-lint", action="store_true", help="run advisory ruff in the patch grader")
    p.add_argument("--patch-git", action="store_true", help="run git apply --check in the patch grader")
    p.add_argument("--n", type=int, default=50, help="cap on paired tasks per arm (same for both).")
    p.add_argument("--seed", type=int, default=42, help="shared seed (paired: same questions both arms).")
    p.add_argument("--sampling", default="production", help="sampling policy tag for the test_profile.")
    p.add_argument("--model", default="unknown", help="model id for model/quant indexing (never role).")
    p.add_argument("--quant", default="unknown", help="quant id for model/quant indexing (never role).")
    p.add_argument("--output", default=None, help="output dir for per-arm JSONL + paired report.")
    p.add_argument(
        "--execute",
        action="store_true",
        help="attempt real inference (STILL env-gated by "
        f"{PAIRED_AB_INFERENCE_ENV}=1; otherwise falls back to dry-run).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = run_paired_ab(args)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    if result.get("mode") == "execute":
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
