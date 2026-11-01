"""Observe the same archived incumbent on the two eval draws at a core rotation.

This module is deliberately outside the optimizer's objective path. It only appends
predeclaration, pair, and readout ledger events; it never writes a trial, archive,
baseline, or live prompt file.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from statistics import fmean, stdev
from typing import Any, Mapping

from src.autopilot_core.infra_fingerprint import (
    COMPARABLE,
    collect_infra_fingerprint,
    compare_infra_fingerprints,
    fingerprint_digest,
)

PLAN_EVENT = "autopilot_core_rotation_observation_plan"
BOUNDARY_PIN_EVENT = "autopilot_core_rotation_observation_boundary_pin"
PAIR_EVENT = "autopilot_core_rotation_observation_pair"
SUMMARY_EVENT = "autopilot_core_rotation_observation_summary"
SCHEMA_VERSION = 1
_PROMPT_ROOT_DEFAULT = Path(__file__).resolve().parents[2] / "tmp" / "gepa_prompt_roots"


def core_rotation_plan(
    *, study_id: str, planned_k: int, first_boundary_trial_id: int, period: int
) -> dict[str, Any]:
    """Build a fixed, positive-K boundary plan without reading any eval result."""
    study_id = str(study_id or "").strip()
    if not study_id:
        raise ValueError("study_id must be predeclared")
    if isinstance(planned_k, bool) or int(planned_k) != planned_k or int(planned_k) < 1:
        raise ValueError("planned_k must be a positive integer")
    if isinstance(period, bool) or int(period) != period or int(period) < 1:
        raise ValueError("rotation period must be a positive integer")
    if (
        isinstance(first_boundary_trial_id, bool)
        or int(first_boundary_trial_id) != first_boundary_trial_id
        or int(first_boundary_trial_id) <= 0
    ):
        raise ValueError("first boundary must be a positive trial id")
    period = int(period)
    first = int(first_boundary_trial_id)
    if first % period:
        raise ValueError("first boundary trial id is not a core-rotation boundary")
    k = int(planned_k)
    boundaries = []
    for offset in range(k):
        boundary_trial_id = first + offset * period
        incoming = boundary_trial_id // period
        boundaries.append(
            {
                "boundary_trial_id": boundary_trial_id,
                "outgoing_rotation_index": incoming - 1,
                "incoming_rotation_index": incoming,
            }
        )
    return {
        "type": PLAN_EVENT,
        "schema_version": SCHEMA_VERSION,
        "study_id": study_id,
        "planned_k": k,
        "rotation_period_trials": period,
        "planned_boundaries": boundaries,
        "metric": "eval_result.quality",
        "metric_direction": "higher_is_better",
        "drop_definition": "q_outgoing_rotation - q_incoming_rotation",
        "authority": "observe_only",
    }


def summarize_pairs(
    study_id: str,
    planned_k: int,
    pairs: list[Mapping[str, Any]],
    planned_boundaries: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Summarize only a complete, finite, exactly planned set of paired boundaries."""
    k = int(planned_k)
    complete = [p for p in pairs if p.get("status") == "complete"]
    planned_ids = [
        int(row["boundary_trial_id"])
        for row in (planned_boundaries or [])
        if row.get("boundary_trial_id") is not None
    ]
    failures = [
        {
            "boundary_trial_id": p.get("boundary_trial_id"),
            "reason": str(p.get("reason") or "incomplete pair"),
        }
        for p in pairs
        if p.get("status") != "complete"
    ]
    drops: list[float] = []
    for pair in complete:
        value = pair.get("drop")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            failures.append(
                {
                    "boundary_trial_id": pair.get("boundary_trial_id"),
                    "reason": "nonfinite_or_invalid_drop",
                }
            )
            continue
        drops.append(float(value))
    complete_ids = [
        int(p["boundary_trial_id"]) for p in complete if p.get("boundary_trial_id") is not None
    ]
    missing_ids = sorted(set(planned_ids) - set(complete_ids))
    if (
        len(complete) != k
        or len(drops) != k
        or failures
        or (planned_ids and (len(set(planned_ids)) != k or set(complete_ids) != set(planned_ids)))
    ):
        return {
            "type": SUMMARY_EVENT,
            "schema_version": SCHEMA_VERSION,
            "study_id": study_id,
            "status": "incomplete",
            "planned_k": k,
            "pairs_recorded": len(pairs),
            "pairs_complete": len(drops),
            "pairs_incomplete": max(k - len(drops), 0),
            "planned_boundary_trial_ids": planned_ids,
            "missing_boundary_trial_ids": missing_ids,
            "failures": failures,
            "mean_drop": None,
            "standard_error": None,
            "mean_drop_plus_minus_se": None,
        }
    average = fmean(drops)
    se = stdev(drops) / math.sqrt(k) if k >= 2 else None
    return {
        "type": SUMMARY_EVENT,
        "schema_version": SCHEMA_VERSION,
        "study_id": study_id,
        "status": "complete",
        "planned_k": k,
        "pairs_recorded": k,
        "pairs_complete": k,
        "pairs_incomplete": 0,
        "planned_boundary_trial_ids": planned_ids,
        "missing_boundary_trial_ids": [],
        "failures": [],
        "mean_drop": average,
        "standard_error": se,
        "mean_drop_plus_minus_se": (
            {"mean": average, "se": se} if se is not None else {"mean": average, "se": None}
        ),
        "positive_means": "worse quality on incoming draw",
    }


def _tree_sha256(root: Path) -> tuple[str, int]:
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("prompt snapshot is missing or is a symlink")
    digest = hashlib.sha256()
    count = 0
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ValueError(f"prompt snapshot contains a symlink: {rel}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"prompt snapshot contains a non-file entry: {rel}")
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
        digest.update(b"\0")
        count += 1
    if not count:
        raise ValueError("prompt snapshot is empty")
    return digest.hexdigest(), count


def _archive_incumbent(
    archive: Any,
    journal: Any,
    checkpoint_dir: Path,
    current_infra: Mapping[str, Any],
    current_flags: Mapping[str, Any],
    current_served_identity: Mapping[str, Any],
) -> tuple[dict[str, Any], Path]:
    member = archive.production_best()
    if member is None:
        raise ValueError("no archived production incumbent")
    entries = journal.entries_with_supersessions()
    row = next((e for e in entries if int(e.trial_id) == int(member.trial_id)), None)
    if row is None:
        raise ValueError("production incumbent trial is absent from the journal")
    baseline_infra = row.infra_fingerprint if isinstance(row.infra_fingerprint, Mapping) else {}
    comparison = compare_infra_fingerprints(
        current_infra, baseline_infra, reference_label="incumbent_trial"
    )
    if comparison.get("status") != COMPARABLE:
        raise ValueError(
            "current serving/source regime does not match incumbent: " + str(comparison)
        )

    checkpoint_root = Path(checkpoint_dir).resolve(strict=True)
    checkpoint_link = checkpoint_root / "production_best"
    try:
        checkpoint = checkpoint_link.resolve(strict=True)
        if checkpoint == checkpoint_root or not checkpoint.is_relative_to(checkpoint_root):
            raise ValueError("production-best checkpoint resolves outside its owner directory")
        meta_path = checkpoint / "checkpoint_meta.json"
        if meta_path.is_symlink():
            raise ValueError("checkpoint metadata is a symlink")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"production-best checkpoint unavailable: {type(exc).__name__}") from exc
    if int(meta.get("trial_id", -1)) != int(member.trial_id):
        raise ValueError("archive production-best trial and checkpoint trial differ")
    if meta.get("config_snapshot") != member.config_snapshot:
        raise ValueError("archive and checkpoint incumbent configuration differ")
    if meta.get("feature_flags") != dict(current_flags):
        raise ValueError("current runtime feature flags do not match the incumbent checkpoint")
    if not meta.get("feature_flags") or not current_flags:
        raise ValueError("runtime/incumbent feature flags are not fully attestable")

    cfg = member.config_snapshot if isinstance(member.config_snapshot, Mapping) else {}
    action_type = str(cfg.get("type") or "")
    if action_type not in {"prompt_mutation", "gepa_optimize"}:
        raise ValueError(
            f"incumbent action is not a prompt-only artifact: {action_type or 'unknown'}"
        )
    prompt_action_fields = {
        "prompt_mutation": {"type", "file", "mutation", "description"},
        "gepa_optimize": {"type", "file", "max_evals", "description"},
    }[action_type]
    if set(cfg) - prompt_action_fields:
        raise ValueError("incumbent snapshot contains nonprompt or unknown mutation knobs")
    target_file = Path(str(cfg.get("file") or ""))
    if (
        not str(cfg.get("file") or "").strip()
        or target_file.is_absolute()
        or ".." in target_file.parts
    ):
        raise ValueError("incumbent snapshot lacks a safe prompt-file target")
    details = row.eval_details if isinstance(row.eval_details, Mapping) else {}
    incumbent_served_identity = details.get("ap70_served_runtime_identity")
    if (
        not isinstance(incumbent_served_identity, Mapping)
        or incumbent_served_identity.get("status") != "observed"
    ):
        raise ValueError("incumbent journal row has no prospective served-runtime identity witness")
    stored_digest = incumbent_served_identity.get("identity_sha256")
    stored_body = {
        key: value for key, value in incumbent_served_identity.items() if key != "identity_sha256"
    }
    recomputed_stored_digest = hashlib.sha256(
        json.dumps(stored_body, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()
    if stored_digest != recomputed_stored_digest:
        raise ValueError("incumbent journal served-runtime identity digest is invalid")
    if dict(incumbent_served_identity.get("feature_flags") or {}) != dict(current_flags):
        raise ValueError("incumbent write-side runtime flags do not match the current API flags")
    if dict(current_served_identity.get("feature_flags") or {}) != dict(current_flags):
        raise ValueError("attested API worker flags do not match the current API flags")
    if incumbent_served_identity.get("identity_sha256") != current_served_identity.get(
        "identity_sha256"
    ):
        raise ValueError(
            "current served runtime does not match the incumbent's write-side identity"
        )
    served = details.get("served_content")
    files = served.get("files") if isinstance(served, Mapping) else None
    if not isinstance(files, Mapping) or not files:
        raise ValueError("incumbent has no complete served-prompt file identity")
    for rel, expected in files.items():
        rel_path = Path(str(rel))
        if (
            rel_path.is_absolute()
            or ".." in rel_path.parts
            or not str(rel).startswith("orchestration/prompts/")
        ):
            raise ValueError("incumbent includes a non-prompt served artifact")
        prompt_file = checkpoint / "prompts" / rel_path.relative_to("orchestration/prompts")
        if prompt_file.is_symlink() or not prompt_file.is_file():
            raise ValueError("incumbent prompt artifact is missing from its checkpoint")
        if _file_sha256(prompt_file) != str(expected):
            raise ValueError("checkpoint prompt bytes do not match incumbent served-content SHA")
    prompt_root = checkpoint / "prompts"
    prompt_sha, prompt_files = _tree_sha256(prompt_root)
    return (
        {
            "trial_id": int(member.trial_id),
            "config_fingerprint": str(member.config_fingerprint or ""),
            "config_snapshot": dict(member.config_snapshot),
            "checkpoint_target": str(checkpoint),
            "checkpoint_metadata_sha256": _file_sha256(checkpoint / "checkpoint_meta.json"),
            "prompt_tree_sha256": prompt_sha,
            "prompt_file_count": prompt_files,
            "feature_flags": dict(current_flags),
            "served_runtime_identity": dict(incumbent_served_identity),
            "infra_fingerprint": dict(current_infra),
            "infra_digest": fingerprint_digest(current_infra),
            "incumbent_trial_infra_digest": fingerprint_digest(baseline_infra),
        },
        prompt_root,
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _make_tree_read_only(root: Path) -> None:
    """Freeze the request-scoped copy before a server can read it."""
    for path in sorted(Path(root).rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if path.is_symlink():
            raise ValueError(f"request prompt mirror contains a symlink: {path}")
        path.chmod(0o555 if path.is_dir() else 0o444)
    Path(root).chmod(0o555)


def _current_flags(url: str) -> dict[str, Any]:
    import httpx

    response = httpx.get(f"{url.rstrip('/')}/config/attest", timeout=10)
    response.raise_for_status()
    payload = response.json()
    flags = payload.get("flags") if isinstance(payload, Mapping) else None
    if not isinstance(flags, Mapping) or not flags:
        raise ValueError("live API feature flags are missing")
    return dict(flags)


def _current_served_identity(url: str) -> dict[str, Any]:
    """Request the opt-in API-loaded and backend-serving identity witness."""
    import httpx

    response = httpx.get(f"{url.rstrip('/')}/config/attest?include_served_identity=1", timeout=8)
    response.raise_for_status()
    payload = response.json()
    identity = payload.get("served_identity") if isinstance(payload, Mapping) else None
    if not isinstance(identity, Mapping) or identity.get("status") != "observed":
        reason = (
            identity.get("reason") if isinstance(identity, Mapping) else "missing identity payload"
        )
        raise ValueError(f"live served identity is unavailable: {reason}")
    digest = identity.get("identity_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("live served identity digest is unavailable")
    digest_body = {key: value for key, value in identity.items() if key != "identity_sha256"}
    observed_digest = hashlib.sha256(
        json.dumps(digest_body, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()
    if observed_digest != digest:
        raise ValueError("live served identity digest does not match its fields")
    return dict(identity)


def _override_base() -> Path:
    configured = os.environ.get("ORCHESTRATOR_PROMPT_ROOT_OVERRIDE_BASE")
    root = Path(configured).expanduser() if configured else _PROMPT_ROOT_DEFAULT
    return root.resolve(strict=False)


def _evaluation_environment_identity() -> dict[str, str]:
    """Bind evaluator/server knobs without disclosing their literal values."""
    names = sorted(
        key
        for key in os.environ
        if key.startswith(("AUTOPILOT_", "ORCHESTRATOR_")) or key == "LLM_JUDGE_ROLE"
    )
    return {key: hashlib.sha256(os.environ[key].encode("utf-8")).hexdigest() for key in names}


def _pair_identity_unchanged(
    *,
    prompt_source: Path,
    mirror: Path,
    expected_prompt_sha: str,
    flags: Mapping[str, Any],
    url: str,
    expected_infra_digest: str,
    expected_environment_identity: Mapping[str, str],
    expected_served_identity_sha256: str,
) -> None:
    source_sha, _ = _tree_sha256(prompt_source)
    mirror_sha, _ = _tree_sha256(mirror)
    if source_sha != expected_prompt_sha or mirror_sha != expected_prompt_sha:
        raise ValueError("incumbent prompt bytes changed during paired observation")
    if _current_flags(url) != dict(flags):
        raise ValueError("runtime feature flags changed during paired observation")
    if _current_served_identity(url).get("identity_sha256") != expected_served_identity_sha256:
        raise ValueError(
            "API/backend process or loaded serving identity changed during paired observation"
        )
    if _evaluation_environment_identity() != dict(expected_environment_identity):
        raise ValueError("evaluator/server environment knobs changed during paired observation")
    current = collect_infra_fingerprint()
    if fingerprint_digest(current) != expected_infra_digest:
        raise ValueError("serving/source/recipe identity changed during paired observation")


def _quality_result(result: Any, expected_rotation: int, expected_core_n: int) -> dict[str, Any]:
    details = getattr(result, "details", None)
    details = details if isinstance(details, Mapping) else {}
    provenance = details.get("tier_mix_provenance")
    provenance = provenance if isinstance(provenance, Mapping) else {}
    rotation = provenance.get("core_rotation_index")
    if int(rotation) != expected_rotation:
        raise ValueError("evaluator did not produce the requested rotation draw")
    if details.get("core_selection") != "tier_stratified":
        raise ValueError("evaluator did not use the rotating tier-stratified core")
    if not bool(getattr(result, "quality_measured", False)):
        raise ValueError("quality was not measured")
    quality = getattr(result, "quality", None)
    if (
        isinstance(quality, bool)
        or not isinstance(quality, (int, float))
        or not math.isfinite(quality)
    ):
        raise ValueError("quality is nonnumeric or nonfinite")
    n_questions = int(getattr(result, "n_questions", 0) or 0)
    n_completed = getattr(result, "n_questions_completed", None)
    n_requested = getattr(result, "n_questions_requested", None)
    denominator = details.get("quality_denominator")
    if int(details.get("base_core_questions", -1)) != int(expected_core_n):
        raise ValueError("evaluator did not draw the full predeclared T1 core size")
    if n_questions <= 0 or n_completed != n_questions or n_requested != n_questions:
        raise ValueError("paired T1 evaluation is partial")
    if (
        isinstance(denominator, bool)
        or not isinstance(denominator, int)
        or denominator != n_questions
    ):
        raise ValueError("paired T1 quality denominator excludes failed or partial rows")
    question_rows = getattr(result, "question_results", None)
    if not isinstance(question_rows, list) or len(question_rows) != n_questions:
        raise ValueError("paired T1 question-result rows are incomplete")
    if any(bool(getattr(row, "partial", False)) for row in question_rows):
        raise ValueError("paired T1 contains partial model responses")
    sidecar = details.get("question_sidecar_persistence")
    if not isinstance(sidecar, Mapping) or sidecar.get("archive_complete") is not True:
        raise ValueError("paired T1 question-result sidecar is incomplete")
    counts = details.get("partition_counts")
    counts = counts if isinstance(counts, Mapping) else {}
    return {
        "rotation_index": expected_rotation,
        "quality": float(quality),
        "core_id": str(getattr(result, "core_id", "") or ""),
        "dataset_sha256": str(details.get("dataset_sha256") or ""),
        "n_questions": n_questions,
        "n_questions_completed": n_completed,
        "quality_denominator": denominator,
        "question_sidecar_persistence": dict(sidecar),
        "quality_denominator_n": counts.get("core"),
        "eval_wall_s": float(getattr(result, "eval_wall_s", 0.0) or 0.0),
        "sum_request_elapsed_s": float(getattr(result, "sum_request_elapsed_s", 0.0) or 0.0),
    }


def _existing_plan(journal: Any, study_id: str) -> dict[str, Any] | None:
    plans = journal.ledger_events(PLAN_EVENT)
    matches = [p for p in plans if p.get("study_id") == study_id]
    if len(matches) > 1:
        raise ValueError("study id has more than one predeclaration")
    return dict(matches[0]) if matches else None


def _pair_events(journal: Any, study_id: str) -> list[dict[str, Any]]:
    return [dict(e) for e in journal.ledger_events(PAIR_EVENT) if e.get("study_id") == study_id]


def _boundary_pins(journal: Any, study_id: str) -> list[dict[str, Any]]:
    return [
        dict(e) for e in journal.ledger_events(BOUNDARY_PIN_EVENT) if e.get("study_id") == study_id
    ]


def _append_incomplete_pair(
    journal: Any,
    *,
    study_id: str,
    boundary: Mapping[str, Any],
    reason: str,
    incumbent: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    event = {
        "type": PAIR_EVENT,
        "schema_version": SCHEMA_VERSION,
        "study_id": study_id,
        "boundary_trial_id": int(boundary["boundary_trial_id"]),
        "outgoing_rotation_index": int(boundary["outgoing_rotation_index"]),
        "incoming_rotation_index": int(boundary["incoming_rotation_index"]),
        "status": "incomplete",
        "reason": str(reason)[:500],
        "incumbent": dict(incumbent or {}),
        "q_outgoing": None,
        "q_incoming": None,
        "drop": None,
        "timing_cost": None,
    }
    return journal.append_ledger_event(event)


def capture_at_rotation_boundary(
    *,
    journal: Any,
    archive: Any,
    tower: Any,
    checkpoint_dir: Path,
    orchestrator_url: str,
    study_id: str,
    planned_k: int,
    boundary_trial_id: int,
    rotation_period_trials: int,
    dispatch_fingerprint: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Run a predeclared, request-scoped paired diagnostic at a planned boundary.

    The caller must gate this behind the explicit default-off AutoPilot opt-in and
    call it before dispatching the next action. No active prompt/config file is edited.
    """
    study_id = str(study_id or "").strip()
    if not study_id:
        raise ValueError("opt-in requires an explicit study id")
    trial_id = int(boundary_trial_id)
    period = int(rotation_period_trials)
    if period < 1 or trial_id <= 0 or trial_id % period:
        return None
    requested_plan = core_rotation_plan(
        study_id=study_id,
        planned_k=planned_k,
        first_boundary_trial_id=trial_id,
        period=period,
    )
    plan = _existing_plan(journal, study_id)
    if plan is None:
        plan = requested_plan
        # Freeze K and the planned draw identities before touching incumbent or
        # checkpoint state. A failed identity check remains an explicit refusal
        # under this same preregistered plan.
        journal.append_ledger_event(plan)
    elif (
        int(plan.get("planned_k", -1)) != int(planned_k)
        or int(plan.get("rotation_period_trials", -1)) != period
    ):
        raise ValueError("environment plan differs from the existing study predeclaration")
    boundary = next(
        (b for b in plan["planned_boundaries"] if int(b["boundary_trial_id"]) == trial_id),
        None,
    )
    if boundary is None:
        return None
    if any(
        int(e.get("boundary_trial_id", -1)) == trial_id for e in _pair_events(journal, study_id)
    ):
        return None
    prior_pin = next(
        (
            e
            for e in _boundary_pins(journal, study_id)
            if int(e.get("boundary_trial_id", -1)) == trial_id
        ),
        None,
    )
    if prior_pin is not None:
        pair = _append_incomplete_pair(
            journal,
            study_id=study_id,
            boundary=boundary,
            reason="previous process ended after pinning identities but before recording the pair; not rerun",
            incumbent=prior_pin.get("identity")
            if isinstance(prior_pin.get("identity"), Mapping)
            else None,
        )
        summary = summarize_pairs(
            study_id,
            int(planned_k),
            _pair_events(journal, study_id),
            list(plan.get("planned_boundaries") or []),
        )
        journal.append_ledger_event(summary)
        return {"pair": pair, "summary": summary}

    current_infra = collect_infra_fingerprint()
    identity_changed_before_pin = False
    if dispatch_fingerprint is not None:
        identity_changed_before_pin = fingerprint_digest(current_infra) != fingerprint_digest(
            dispatch_fingerprint
        )
        current_infra = dict(dispatch_fingerprint)
    try:
        if identity_changed_before_pin:
            raise ValueError("serving/source/recipe identity changed before AP-70 predeclaration")
        flags = _current_flags(orchestrator_url)
        served_identity = _current_served_identity(orchestrator_url)
        incumbent, prompt_source = _archive_incumbent(
            archive, journal, checkpoint_dir, current_infra, flags, served_identity
        )
        prompt_sha = str(incumbent["prompt_tree_sha256"])
        base = _override_base()
        base.mkdir(parents=True, exist_ok=True)
        study_slug = hashlib.sha256(study_id.encode("utf-8")).hexdigest()[:12]
        pair_root = Path(tempfile.mkdtemp(prefix=f"ap70-{study_slug}-{trial_id}-", dir=base))
        mirror = pair_root / "prompts"
        shutil.copytree(prompt_source, mirror, symlinks=True)
        _make_tree_read_only(mirror)
        mirror_sha, _ = _tree_sha256(mirror)
        if mirror_sha != prompt_sha:
            raise ValueError("scratch prompt mirror differs from archived incumbent")
        # The request root is accepted only below the same root configured by the API.
        if not mirror.resolve(strict=True).is_relative_to(base):
            raise ValueError("request-scoped prompt mirror is outside the API's allowed root")
        identity = {
            **incumbent,
            "source_root": str(prompt_source),
            "request_prompt_root": str(mirror),
            "request_prompt_tree_sha256": mirror_sha,
            "evaluation_environment_identity": _evaluation_environment_identity(),
            "served_runtime_identity": dict(served_identity),
        }
    except Exception as exc:  # noqa: BLE001 - observation cannot break an ordinary trial
        journal.append_ledger_event(
            {
                "type": BOUNDARY_PIN_EVENT,
                "schema_version": SCHEMA_VERSION,
                "study_id": study_id,
                "boundary_trial_id": trial_id,
                "outgoing_rotation_index": int(boundary["outgoing_rotation_index"]),
                "incoming_rotation_index": int(boundary["incoming_rotation_index"]),
                "planned_k": int(planned_k),
                "rotation_period_trials": period,
                "predeclared_boundary_trial_id": trial_id,
                "boundary_identity_status": "unavailable",
                "boundary_identity_reason": f"{type(exc).__name__}: {exc}"[:500],
                "identity": {"infra_fingerprint": dict(current_infra)},
            }
        )
        pair = _append_incomplete_pair(
            journal, study_id=study_id, boundary=boundary, reason=str(exc)
        )
        summary = summarize_pairs(
            study_id,
            int(planned_k),
            _pair_events(journal, study_id),
            list(plan.get("planned_boundaries") or []),
        )
        journal.append_ledger_event(summary)
        return {"pair": pair, "summary": summary}

    # The plan and all identities are fsync'd before the first paired request.
    journal.append_ledger_event(
        {
            "type": BOUNDARY_PIN_EVENT,
            "schema_version": SCHEMA_VERSION,
            "study_id": study_id,
            "boundary_trial_id": trial_id,
            "outgoing_rotation_index": int(boundary["outgoing_rotation_index"]),
            "incoming_rotation_index": int(boundary["incoming_rotation_index"]),
            "planned_k": int(planned_k),
            "rotation_period_trials": period,
            "boundary_identity_status": "verified",
            "identity": identity,
        }
    )

    from eval_tower import EVAL_SPEC_SEED, EVAL_T1_SPEC_N

    previous_context = (
        getattr(tower, "_trial_id_context", None),
        getattr(tower, "_trial_path_context", None),
        getattr(tower, "_question_artifact_dir", None),
        getattr(tower, "_core_rotation_observation_context", None),
    )
    pair_result: dict[str, Any]
    try:
        outgoing_index = int(boundary["outgoing_rotation_index"])
        incoming_index = int(boundary["incoming_rotation_index"])
        results: dict[str, dict[str, Any]] = {}
        for name, rotation in (("outgoing", outgoing_index), ("incoming", incoming_index)):
            _pair_identity_unchanged(
                prompt_source=prompt_source,
                mirror=mirror,
                expected_prompt_sha=prompt_sha,
                flags=flags,
                url=orchestrator_url,
                expected_infra_digest=str(incumbent["infra_digest"]),
                expected_environment_identity=identity["evaluation_environment_identity"],
                expected_served_identity_sha256=str(
                    identity["served_runtime_identity"]["identity_sha256"]
                ),
            )
            artifact_dir = pair_root / f"question-results-{name}"
            tower._trial_id_context = None
            tower._trial_path_context = pair_root
            tower._question_artifact_dir = artifact_dir
            result = tower.eval_t1_observation(
                n=EVAL_T1_SPEC_N,
                seed=EVAL_SPEC_SEED,
                trial_id=trial_id,
                core_rotation_index=rotation,
                prompt_root=str(mirror),
            )
            results[name] = _quality_result(result, rotation, EVAL_T1_SPEC_N)
            _pair_identity_unchanged(
                prompt_source=prompt_source,
                mirror=mirror,
                expected_prompt_sha=prompt_sha,
                flags=flags,
                url=orchestrator_url,
                expected_infra_digest=str(incumbent["infra_digest"]),
                expected_environment_identity=identity["evaluation_environment_identity"],
                expected_served_identity_sha256=str(
                    identity["served_runtime_identity"]["identity_sha256"]
                ),
            )
        drop = float(results["outgoing"]["quality"] - results["incoming"]["quality"])
        pair_result = {
            "type": PAIR_EVENT,
            "schema_version": SCHEMA_VERSION,
            "study_id": study_id,
            "boundary_trial_id": trial_id,
            "outgoing_rotation_index": outgoing_index,
            "incoming_rotation_index": incoming_index,
            "status": "complete",
            "metric": "eval_result.quality",
            "metric_direction": "higher_is_better",
            "positive_means": "worse quality on incoming draw",
            "incumbent": identity,
            "scratch_root": str(pair_root),
            "question_result_artifacts": [
                str(pair_root / "question-results-outgoing" / "question_results.T1.jsonl"),
                str(pair_root / "question-results-incoming" / "question_results.T1.jsonl"),
            ],
            "outgoing": results["outgoing"],
            "incoming": results["incoming"],
            "q_outgoing": results["outgoing"]["quality"],
            "q_incoming": results["incoming"]["quality"],
            "drop": drop,
            "timing_cost": {
                "outgoing_eval_wall_s": results["outgoing"]["eval_wall_s"],
                "incoming_eval_wall_s": results["incoming"]["eval_wall_s"],
                "sum_request_elapsed_s": (
                    results["outgoing"]["sum_request_elapsed_s"]
                    + results["incoming"]["sum_request_elapsed_s"]
                ),
                "n_questions": results["outgoing"]["n_questions"]
                + results["incoming"]["n_questions"],
            },
        }
    except Exception as exc:  # noqa: BLE001 - no diagnostic failure may fail a trial
        pair_result = {
            "type": PAIR_EVENT,
            "schema_version": SCHEMA_VERSION,
            "study_id": study_id,
            "boundary_trial_id": trial_id,
            "outgoing_rotation_index": int(boundary["outgoing_rotation_index"]),
            "incoming_rotation_index": int(boundary["incoming_rotation_index"]),
            "status": "incomplete",
            "reason": f"{type(exc).__name__}: {exc}"[:500],
            "incumbent": identity,
            "scratch_root": str(pair_root),
            "question_result_artifacts": [
                str(pair_root / "question-results-outgoing" / "question_results.T1.jsonl"),
                str(pair_root / "question-results-incoming" / "question_results.T1.jsonl"),
            ],
            "outgoing": None,
            "incoming": None,
            "q_outgoing": None,
            "q_incoming": None,
            "drop": None,
            "timing_cost": None,
        }
    finally:
        (
            tower._trial_id_context,
            tower._trial_path_context,
            tower._question_artifact_dir,
            tower._core_rotation_observation_context,
        ) = previous_context

    pair = journal.append_ledger_event(pair_result)
    all_pairs = _pair_events(journal, study_id)
    summary = summarize_pairs(
        study_id,
        int(planned_k),
        all_pairs,
        list(plan.get("planned_boundaries") or []),
    )
    journal.append_ledger_event(summary)
    return {"pair": pair, "summary": summary}
