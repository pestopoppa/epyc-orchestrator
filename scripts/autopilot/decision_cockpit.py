"""AP-50 decision cockpit: "what optimizes the orchestrator", as a decision surface.

The optimization brief (``optimization_brief.py``) answers the question with a
narrative and a lever ledger parsed from the periodically generated digest. This
module answers it as a *decision* surface, sourced live from the evidence itself:

  * every rotated journal shard (``journal_shards``; never the base file alone),
    with append-only supersession events folded in;
  * the NumericSwarm Optuna study DB (read-only ``mode=ro``);
  * ``autopilot_state.json`` (declared producer state, active eras, incumbent
    bundle), the ``production_best`` checkpoint meta, and the instrument-era
    registry.

Contract ``epyc.autopilot.decision_cockpit.v1`` (sections):

  inputs            per-input read status; an unreadable input is ``unknown``,
                    never zero and never a green default
  eras              current measurement era (default view) + every era bucket.
                    A bucket is (quality era, speed era); nothing is pooled across
                    buckets
  current           in-flight / last trial: intervention, hypothesis, falsifier
  objective_deltas  incumbent vs candidate, with n, binomial SE, paired
                    discordance and replication; OP-20 producer fence
  funnel            proposed -> executed -> valid -> kept -> promoted ->
                    currently_live, with rejection reasons at every drop
  lever_scoreboard  per-lever counts and deltas from journal + study evidence
  provenance        drill-down graph, columns hypothesis -> experiment ->
                    evidence -> verdict -> runtime_state, explicit edge semantics
  caveats           OP-20, era fence, attribution limits

Read-only by construction: no journal append, no torn-tail repair (the
``ExperimentJournal`` class is deliberately NOT instantiated because its loader
may quarantine a torn tail), no state write, sqlite opened ``mode=ro``. Observe
only: nothing here feeds fitness, archive admission, promotion or authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    from scripts.autopilot.journal_shards import journal_shards, shard_batch_index
except ModuleNotFoundError:  # pragma: no cover - bare-module import context
    from journal_shards import journal_shards, shard_batch_index  # type: ignore

import yaml

SCHEMA = "epyc.autopilot.decision_cockpit.v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JOURNAL_DIR = REPO_ROOT / "orchestration"
DEFAULT_STATE_PATH = REPO_ROOT / "orchestration" / "autopilot_state.json"
DEFAULT_ERAS_PATH = REPO_ROOT / "orchestration" / "instrument_eras.yaml"
DEFAULT_STUDY_DB = REPO_ROOT / "orchestration" / "optuna_study.db"
DEFAULT_CHECKPOINTS_DIR = REPO_ROOT / "orchestration" / "autopilot_checkpoints"

UNKNOWN = "unknown"
QUALITY_SCOPES = ("eval_quality", "autopilot_quality")
SPEED_SCOPE = "autopilot_speed"
PRE_REGISTRY = "pre-registry"
QUALITY_SCALE = 3.0

STAGES = ("proposed", "executed", "valid", "kept", "promoted", "currently_live")

#: Action types that gather evidence or maintain the loop but do not test a
#: runtime intervention. They are counted, never put through the intervention
#: funnel (a seed batch cannot be "kept").
EVIDENCE_ONLY_ACTIONS = frozenset(
    {
        "seed_batch",
        "deep_eval",
        "train_routing_models",
        "distill_skillbank",
        "distill_knowledge",
        "rollback",
    }
)
#: OP-20: quality from the seeding path and from EvalTower-scored trials is not
#: comparable until the ``task_failed`` ruling lands in BOTH producers (AP-64).
SEEDING_ACTIONS = frozenset({"seed_batch"})

OP20_CAVEAT = (
    "OP-20 unresolved: EvalTower-scored quality and seeding-path (seed_batch) quality "
    "score task_failed differently until the ruling is applied to both producers "
    "(AP-64). They are reported separately and never merged or compared."
)
ERA_CAVEAT = (
    "Numbers from different instrument eras are not comparable "
    "(orchestration/instrument_eras.yaml). Every view is scoped to ONE era bucket "
    "(quality era x speed era); there is no pooled all-era view."
)
LIVE_CAVEAT = (
    "currently_live is provable only for the trial named by the production_best "
    "checkpoint meta. A kept intervention not named there is 'unknown': the "
    "checkpoint does not record per-intervention runtime config."
)
SE_CAVEAT = (
    "Uncertainty is the binomial SE of accuracy scaled to the 0-3 quality axis "
    "(quality = 3 * correct / n_scored, checked per row). Rows where that identity "
    "does not hold carry no SE rather than a guessed one."
)

EDGE_SEMANTICS = {
    "tested_by": "hypothesis -> experiment: this trial was dispatched to test the hypothesis text",
    "produced": "experiment -> evidence: the trial executed and journaled this measurement",
    "not_executed": "experiment -> verdict: the action never ran (invalid/skipped); no evidence exists",
    "judged": "evidence -> verdict: the safety gate's AP-24 keep/revert/excluded decision on this evidence",
    "resulted_in": "verdict -> runtime_state: what the verdict did to the serving runtime",
}
RUNTIME_STATES = {
    "rt:not_executed": "never applied (not executed)",
    "rt:no_verdict": "no verdict recorded",
    "rt:reverted": "reverted / excluded (runtime restored)",
    "rt:kept": "kept (applied at the time)",
    "rt:promoted": "promoted (baseline adopted)",
    "rt:live": "live in production_best",
    "rt:live_unknown": "kept; current liveness unknown",
}


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _parse_ts(value: Any) -> float | None:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _iso(ts: float | None) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _era_short(era_id: Any) -> str:
    m = re.match(r"(E\d+[a-z]?)", str(era_id or ""))
    return m.group(1) if m else str(era_id or "")


def _reason_code(text: Any, *, default: str = "unspecified") -> str:
    code = re.sub(r"[^a-z0-9]+", "_", str(text or "").strip().lower()).strip("_")
    return code[:48] or default


def _clip(text: Any, n: int = 160) -> str:
    s = " ".join(str(text or "").split())
    return s if len(s) <= n else s[: n - 1] + "…"


# --------------------------------------------------------------------------- #
# input readers — each returns an explicit status, never a silent empty
# --------------------------------------------------------------------------- #
def read_eras(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {
        "path": str(path),
        "status": UNKNOWN,
        "eras": [],
        "absence_means": "no era can be assigned: every row is era-unknown and no view is shown",
    }
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        out["status"] = "absent"
        return out
    except (OSError, yaml.YAMLError) as exc:
        out["status"] = "unreadable"
        out["error"] = str(exc)[:200]
        return out
    rows = data.get("eras") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        out["status"] = "unreadable"
        out["error"] = "registry has no eras list"
        return out
    eras = []
    for row in rows:
        if not isinstance(row, dict) or not row.get("id"):
            continue
        eras.append(
            {
                "id": str(row["id"]),
                "scope": str(row.get("scope") or ""),
                "from_ts": _parse_ts(row.get("from")),
                "until_ts": _parse_ts(row.get("until")),
                "from": row.get("from"),
            }
        )
    out["eras"] = eras
    out["status"] = "ok"
    return out


def era_at(eras: list[dict[str, Any]], ts: float | None, scopes: Iterable[str]) -> str | None:
    """Latest-opened era of ``scopes`` active at ``ts`` (None before any boundary)."""
    if ts is None:
        return None
    scopes = set(scopes)
    best: tuple[float, str] | None = None
    for era in eras:
        if era["scope"] not in scopes:
            continue
        start = era["from_ts"]
        end = era["until_ts"]
        if start is None or ts < start:
            continue
        if end is not None and ts >= end:
            continue
        if best is None or start >= best[0]:
            best = (start, era["id"])
    return best[1] if best else None


def bucket_id(quality_era: str | None, speed_era: str | None) -> str:
    return f"{quality_era or PRE_REGISTRY}|{speed_era or PRE_REGISTRY}"


def bucket_label(bid: str) -> str:
    q, _, s = bid.partition("|")
    qs, ss = _era_short(q), _era_short(s)
    return qs if qs == ss else f"{qs}/{ss}"


_SLIM_EVAL_KEYS = ("task_rate_qph", "goodput_qph", "learning_exclusion", "seq_paired_baseline")


def _slim_trial(obj: dict[str, Any]) -> dict[str, Any]:
    """Keep only what the cockpit reads — shards are tens of MB of question text."""
    ed = obj.get("eval_details") if isinstance(obj.get("eval_details"), dict) else {}
    details = ed.get("details") if isinstance(ed.get("details"), dict) else {}
    qres = []
    for q in ed.get("question_results") or []:
        if isinstance(q, dict) and q.get("qid"):
            qres.append((str(q["qid"]), bool(q.get("correct")), bool(q.get("error"))))
    slim = {k: obj.get(k) for k in (
        "trial_id", "timestamp", "species", "action_type", "tier", "quality", "speed",
        "cost", "reliability", "pareto_status", "config_snapshot", "parent_trial",
        "hypothesis", "expected_mechanism", "falsifier", "failure_analysis",
        "deficiency_category", "keep_revert_decision", "bug_corrupted_by",
        "bug_corrupted_reason", "outcome_status", "seq", "baseline_pin",
    )}
    slim["details"] = {
        k: details.get(k) for k in ("correct", "n_scored", "total", "quality_denominator")
    }
    slim["eval"] = {k: ed.get(k) for k in _SLIM_EVAL_KEYS}
    slim["question_results"] = qres
    return slim


def read_journal(journal_dir: Path) -> dict[str, Any]:
    """Read EVERY rotated shard, numerically ordered; fold supersessions."""
    out: dict[str, Any] = {
        "dir": str(journal_dir),
        "status": UNKNOWN,
        "shards": [],
        "shard_count": 0,
        "trial_rows": 0,
        "event_rows": 0,
        "bad_lines": 0,
        "duplicate_trial_ids": 0,
        "supersessions_applied": 0,
        "absence_means": (
            "no journal shard was found: AutoPilot has never journaled here or the "
            "path drifted; every count is unknown, not zero"
        ),
    }
    trials: dict[int, dict[str, Any]] = {}
    events: dict[str, list[dict[str, Any]]] = {}
    shards = journal_shards(Path(journal_dir))
    if not shards:
        out["status"] = "absent"
        out["trials"] = []
        out["events"] = events
        return out
    unreadable = False
    for path in shards:
        info: dict[str, Any] = {
            "name": path.name,
            "batch": shard_batch_index(path),
            "rows": 0,
            "trial_rows": 0,
            "bad_lines": 0,
            "first_trial_id": None,
            "last_trial_id": None,
            "newest_ts": None,
        }
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        info["bad_lines"] += 1
                        continue
                    if not isinstance(obj, dict):
                        info["bad_lines"] += 1
                        continue
                    info["rows"] += 1
                    etype = obj.get("type")
                    if etype is not None:
                        events.setdefault(str(etype), []).append(obj)
                        out["event_rows"] += 1
                        continue
                    tid = _int(obj.get("trial_id"))
                    if tid is None:
                        info["bad_lines"] += 1
                        continue
                    info["trial_rows"] += 1
                    if info["first_trial_id"] is None:
                        info["first_trial_id"] = tid
                    info["last_trial_id"] = tid
                    ts = _parse_ts(obj.get("timestamp"))
                    if ts is not None and (info["newest_ts"] is None or ts > info["newest_ts"]):
                        info["newest_ts"] = ts
                    if tid in trials:
                        out["duplicate_trial_ids"] += 1
                    slim = _slim_trial(obj)
                    slim["_shard"] = path.name
                    trials[tid] = slim  # later shard / later line wins
        except OSError as exc:
            info["error"] = str(exc)[:200]
            unreadable = True
        info["newest_at"] = _iso(info.pop("newest_ts"))
        out["shards"].append(info)
        out["trial_rows"] += info["trial_rows"]
        out["bad_lines"] += info["bad_lines"]
    out["shard_count"] = len(shards)
    # Append-only supersessions: the runtime read view (mirrors
    # ExperimentJournal.entries_with_supersessions, without instantiating it).
    for event in events.get("supersession", []):
        fields = event.get("fields")
        targets = event.get("target_trial_ids")
        if not isinstance(fields, dict) or not isinstance(targets, list):
            continue
        allowed = {k: v for k, v in fields.items() if k in ("bug_corrupted_by", "bug_corrupted_reason",
                                                             "outcome_status", "keep_revert_decision")}
        for target in targets:
            tid = _int(target)
            if tid is not None and tid in trials and allowed:
                trials[tid].update(allowed)
                trials[tid]["_superseded"] = True
                out["supersessions_applied"] += 1
    out["trials"] = [trials[k] for k in sorted(trials)]
    out["events"] = events
    if unreadable:
        out["status"] = "unreadable"
    elif out["bad_lines"]:
        out["status"] = "partial"
    else:
        out["status"] = "ok"
    return out


def read_state(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {
        "path": str(path),
        "status": UNKNOWN,
        "absence_means": (
            "declared producer state, active eras and the incumbent are unknown; "
            "no default era and no incumbent deltas"
        ),
    }
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        out["status"] = "absent"
        return out
    except (OSError, ValueError) as exc:
        out["status"] = "unreadable"
        out["error"] = str(exc)[:200]
        return out
    if not isinstance(data, dict):
        out["status"] = "unreadable"
        out["error"] = "state is not a JSON object"
        return out
    out["status"] = "ok"
    out["data"] = data
    return out


def read_checkpoint(checkpoints_dir: Path) -> dict[str, Any]:
    meta_path = Path(checkpoints_dir) / "production_best" / "checkpoint_meta.json"
    out: dict[str, Any] = {
        "path": str(meta_path),
        "status": UNKNOWN,
        "absence_means": "no journaled intervention can be proven live",
    }
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        out["status"] = "absent"
        return out
    except (OSError, ValueError) as exc:
        out["status"] = "unreadable"
        out["error"] = str(exc)[:200]
        return out
    if not isinstance(meta, dict):
        out["status"] = "unreadable"
        return out
    out["status"] = "ok"
    try:
        out["resolved"] = str(meta_path.parent.resolve().name)
    except OSError:
        out["resolved"] = None
    out["trial_id"] = _int(meta.get("trial_id"))
    out["is_production_best"] = bool(meta.get("is_production_best"))
    out["notes"] = _clip(meta.get("notes"), 200)
    out["schema_version"] = meta.get("schema_version")
    snap = meta.get("config_snapshot")
    out["config_snapshot"] = snap if isinstance(snap, dict) else {}
    return out


def read_studies(db_path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {
        "path": str(db_path),
        "status": UNKNOWN,
        "studies": {},
        "absence_means": "numeric levers carry journal evidence only; study columns are unknown",
    }
    if not Path(db_path).exists():
        out["status"] = "absent"
        return out
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=2.0)
    except sqlite3.Error as exc:
        out["status"] = "unreadable"
        out["error"] = str(exc)[:200]
        return out
    try:
        studies: dict[str, dict[str, Any]] = {}
        for sid, name in conn.execute("SELECT study_id, study_name FROM studies"):
            studies[str(name)] = {"study_id": sid, "complete": 0, "fail": 0, "running": 0,
                                  "other": 0, "best_quality": None, "best_trial_number": None,
                                  "directions": []}
        by_id = {v["study_id"]: v for v in studies.values()}
        for sid, direction in conn.execute(
            "SELECT study_id, direction FROM study_directions ORDER BY study_id, objective"
        ):
            if sid in by_id:
                by_id[sid]["directions"].append(direction)
        for sid, state, cnt in conn.execute(
            "SELECT study_id, state, COUNT(*) FROM trials GROUP BY study_id, state"
        ):
            if sid not in by_id:
                continue
            key = {"COMPLETE": "complete", "FAIL": "fail", "RUNNING": "running"}.get(str(state), "other")
            by_id[sid][key] += int(cnt)
        for sid, number, value in conn.execute(
            "SELECT t.study_id, t.number, v.value FROM trials t JOIN trial_values v "
            "ON v.trial_id = t.trial_id WHERE t.state = 'COMPLETE' AND v.objective = 0"
        ):
            rec = by_id.get(sid)
            val = _float(value)
            if rec is None or val is None:
                continue
            if rec["best_quality"] is None or val > rec["best_quality"]:
                rec["best_quality"] = val
                rec["best_trial_number"] = number
        out["studies"] = studies
        out["status"] = "ok"
    except sqlite3.Error as exc:
        out["status"] = "unreadable"
        out["error"] = str(exc)[:200]
    finally:
        conn.close()
    return out


# --------------------------------------------------------------------------- #
# per-row classification
# --------------------------------------------------------------------------- #
def producer_class(row: dict[str, Any]) -> str:
    if str(row.get("action_type") or "") in SEEDING_ACTIONS:
        return "seeding"
    if _has_evidence(row):
        return "eval_tower"
    return "none"


def _has_evidence(row: dict[str, Any]) -> bool:
    tier = _int(row.get("tier"))
    q = _float(row.get("quality"))
    if tier is None or tier < 1 or q is None:
        return False
    n = _int((row.get("details") or {}).get("n_scored"))
    if n is not None and n <= 0:
        return False
    if q == 0.0 and not _float(row.get("speed")) and not _float(row.get("reliability")):
        return False
    return True


def levers_for(row: dict[str, Any]) -> list[str]:
    atype = str(row.get("action_type") or "")
    snap = row.get("config_snapshot") if isinstance(row.get("config_snapshot"), dict) else {}
    if atype in EVIDENCE_ONLY_ACTIONS:
        return []
    if atype == "numeric_trial":
        return [f"numeric:{snap.get('surface') or UNKNOWN}"]
    if atype == "structural_experiment":
        flags = snap.get("flags") if isinstance(snap.get("flags"), dict) else {}
        return [f"flag:{k}" for k in sorted(flags)] or ["flag:unknown"]
    if atype in ("prompt_mutation", "gepa_optimize", "structural_prune", "few_shot_evolution",
                 "targeted_fix", "compress"):
        return [f"prompt:{snap.get('file') or UNKNOWN}"]
    if atype == "code_mutation":
        return [f"code:{snap.get('file') or UNKNOWN}"]
    return [f"{atype or UNKNOWN}:{snap.get('surface') or snap.get('file') or 'action'}"]


def intervention_summary(row: dict[str, Any]) -> dict[str, Any]:
    snap = row.get("config_snapshot") if isinstance(row.get("config_snapshot"), dict) else {}
    out: dict[str, Any] = {"action_type": row.get("action_type"), "levers": levers_for(row)}
    for key in ("surface", "params", "flags", "file", "mutation", "block", "n_questions", "tier"):
        if key in snap:
            val = snap[key]
            out[key] = _clip(val, 120) if isinstance(val, str) else val
    if snap.get("description"):
        out["description"] = _clip(snap["description"], 200)
    return out


def signature(row: dict[str, Any]) -> str:
    snap = row.get("config_snapshot") if isinstance(row.get("config_snapshot"), dict) else {}
    core = {k: v for k, v in snap.items() if k not in ("description", "max_evals")}
    raw = json.dumps(core, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


_QUALITY_BASELINE_PROSE_RE = re.compile(
    r"Quality regression:\s*[0-9]+(?:\.[0-9]+)?\s+vs\s+baseline\s+([0-9]+(?:\.[0-9]+)?)")
_VIOLATION_RE = re.compile(r"VIOLATIONS:\s*\n\s*-\s*([^\n]+)")


def revert_reason(row: dict[str, Any]) -> tuple[str, str]:
    text = str(row.get("failure_analysis") or "")
    m = _VIOLATION_RE.search(text)
    first = m.group(1) if m else ""
    if not first:
        for line in text.splitlines():
            if line.strip() and not line.strip().endswith(":"):
                first = line.strip().lstrip("- ")
                break
    low = first.lower()
    if "throughput floor" in low:
        code = "throughput_floor"
    elif "quality regression" in low:
        code = "quality_regression"
    elif "regression" in low and "suite" in low:
        code = "suite_regression"
    elif "reliability" in low:
        code = "reliability_floor"
    elif "seq" in low or "sequential" in low:
        code = "sequential_gate"
    elif first:
        code = _reason_code(first.split(":")[0])
    else:
        code = "no_reason_recorded"
    return code, _clip(first, 160)


def classify(row: dict[str, Any], *, promotions: dict[int, str], live_trial_id: int | None,
             checkpoint_known: bool) -> dict[str, Any]:
    """States + the reason the row dropped out of the funnel (if it did)."""
    tid = _int(row.get("trial_id"))
    status = str(row.get("outcome_status") or "ok").strip().lower()
    decision = str(row.get("keep_revert_decision") or "").strip().lower()
    bug = str(row.get("bug_corrupted_by") or "").strip()
    states: dict[str, Any] = {s: False for s in STAGES}
    states["proposed"] = True
    drop: dict[str, str] | None = None
    anomalies: list[str] = []
    runtime = "rt:no_verdict"

    states["executed"] = status == "ok" and str(row.get("species") or "") != "(killed)"
    if not states["executed"]:
        code = _reason_code(row.get("deficiency_category") or status or "not_executed")
        if str(row.get("species") or "") == "(killed)":
            code = "autopilot_killed_mid_trial"
        drop = {"stage": "executed", "code": code,
                "detail": _clip(row.get("failure_analysis") or row.get("bug_corrupted_reason"))}
        runtime = "rt:not_executed"
    else:
        if bug:
            drop = {"stage": "valid", "code": _reason_code(bug),
                    "detail": _clip(row.get("bug_corrupted_reason") or bug)}
        elif decision == "excluded":
            ex = (row.get("eval") or {}).get("learning_exclusion")
            detail = ex.get("reason") if isinstance(ex, dict) else revert_reason(row)[1]
            drop = {"stage": "valid", "code": "learning_excluded", "detail": _clip(detail)}
        elif not _has_evidence(row):
            drop = {"stage": "valid", "code": "no_scored_evidence",
                    "detail": _clip(row.get("failure_analysis") or "no scored eval evidence journaled")}
        else:
            states["valid"] = True
        if decision == "keep" and not states["valid"]:
            anomalies.append("kept_on_invalid_evidence")
        if decision in ("revert", "excluded"):
            runtime = "rt:reverted"
        elif decision == "keep":
            runtime = "rt:kept"
    if states["valid"] and drop is None:
        if decision == "keep":
            states["kept"] = True
        elif decision in ("revert", "unchanged"):
            code, detail = revert_reason(row) if decision == "revert" else ("unchanged_config", "")
            drop = {"stage": "kept", "code": code, "detail": detail}
        else:
            drop = {"stage": "kept", "code": "no_verdict_recorded", "detail": ""}
    promo_src = promotions.get(tid) if tid is not None else None
    if states["kept"]:
        if promo_src:
            states["promoted"] = True
            runtime = "rt:promoted"
        else:
            seq = row.get("seq") if isinstance(row.get("seq"), dict) else {}
            seq_state = str(seq.get("state") or "")
            code = f"seq_{_reason_code(seq_state)}" if seq_state else "no_promotion_recorded"
            drop = {"stage": "promoted", "code": code,
                    "detail": "no baseline promotion names this trial"}
    elif promo_src:
        anomalies.append("promotion_without_kept_verdict")
    if states["kept"]:
        if live_trial_id is not None and tid == live_trial_id:
            states["currently_live"] = True
            runtime = "rt:live"
        else:
            states["currently_live"] = UNKNOWN
            if runtime != "rt:promoted":
                runtime = "rt:live_unknown"
            if drop is None:
                drop = {"stage": "currently_live", "code": (
                    "not_attributable_to_production_best" if checkpoint_known
                    else "production_best_unreadable"),
                    "detail": LIVE_CAVEAT}
    return {"states": states, "drop": drop, "anomalies": anomalies,
            "runtime_state": runtime, "promotion_source": promo_src}


# --------------------------------------------------------------------------- #
# objective deltas
# --------------------------------------------------------------------------- #
def _accuracy_se(correct: int | None, n: int | None) -> float | None:
    if correct is None or n is None or n <= 0:
        return None
    p = correct / n
    return QUALITY_SCALE * math.sqrt(max(p * (1 - p), 0.0) / n)


def incumbent_from_state(state: dict[str, Any], eras: list[dict[str, Any]]) -> dict[str, Any]:
    bundle = state.get("multitier_baseline_bundle")
    out: dict[str, Any] = {"available": False, "source": "autopilot_state.multitier_baseline_bundle"}
    if not isinstance(bundle, dict) or not isinstance(bundle.get("tiers"), dict):
        out["reason"] = "no multitier baseline bundle in state"
        return out
    boundary = _parse_ts(bundle.get("boundary"))
    tiers: dict[str, Any] = {}
    for key, tier in bundle["tiers"].items():
        if not isinstance(tier, dict):
            continue
        outcomes = tier.get("outcomes") if isinstance(tier.get("outcomes"), dict) else {}
        n = len(outcomes) or _int(tier.get("n_questions"))
        correct = sum(1 for v in outcomes.values() if v is True) if outcomes else None
        q = _float(tier.get("quality"))
        identity_ok = (correct is not None and n and q is not None
                       and abs(QUALITY_SCALE * correct / n - q) < 1e-6)
        tiers[str(key)] = {
            "quality": q,
            "n": n,
            "correct": correct,
            "se": _accuracy_se(correct, n) if identity_ok else None,
            "reliability": _float(tier.get("reliability")),
            "core_id": tier.get("core_id"),
            "_outcomes": {str(k): bool(v) for k, v in outcomes.items()},
        }
    out.update({
        "available": bool(tiers),
        "status": bundle.get("status"),
        "policy_version": bundle.get("policy_version"),
        "boundary": bundle.get("boundary"),
        "era_bucket": bucket_id(
            era_at(eras, boundary, QUALITY_SCOPES), era_at(eras, boundary, (SPEED_SCOPE,))
        ) if boundary is not None and eras else None,
        "producer_class": "eval_tower",
        "tiers": tiers,
    })
    return out


def delta_row(row: dict[str, Any], *, incumbent: dict[str, Any], era: str,
              replication: dict[str, list[int]]) -> dict[str, Any]:
    details = row.get("details") or {}
    q = _float(row.get("quality"))
    n = _int(details.get("n_scored"))
    correct = _int(details.get("correct"))
    identity = (q is not None and n and correct is not None
                and abs(QUALITY_SCALE * correct / n - q) < 1e-6)
    se = _accuracy_se(correct, n) if identity else None
    tier = _int(row.get("tier"))
    pclass = producer_class(row)
    vs_inc: dict[str, Any] = {"comparable": False}
    inc_tier = (incumbent.get("tiers") or {}).get(str(tier)) if incumbent.get("available") else None
    if not incumbent.get("available"):
        vs_inc["reason"] = "incumbent_unknown"
    elif pclass != "eval_tower":
        vs_inc["reason"] = "op20_producer_not_comparable"
    elif incumbent.get("era_bucket") != era:
        vs_inc["reason"] = f"era_mismatch:incumbent={bucket_label(str(incumbent.get('era_bucket')))}"
    elif inc_tier is None:
        vs_inc["reason"] = f"no_incumbent_for_tier_{tier}"
    elif q is None or inc_tier.get("quality") is None:
        vs_inc["reason"] = "quality_missing"
    else:
        delta = q - inc_tier["quality"]
        dse = (math.sqrt(se ** 2 + inc_tier["se"] ** 2)
               if se is not None and inc_tier.get("se") is not None else None)
        shared = cand_only = inc_only = 0
        outcomes = inc_tier.get("_outcomes") or {}
        for qid, ok, err in row.get("question_results") or []:
            if err or qid not in outcomes:
                continue
            shared += 1
            if ok and not outcomes[qid]:
                cand_only += 1
            elif outcomes[qid] and not ok:
                inc_only += 1
        vs_inc.update({
            "comparable": True, "delta": delta, "delta_se": dse,
            "z": (delta / dse) if dse else None,
            "incumbent_quality": inc_tier["quality"], "incumbent_n": inc_tier.get("n"),
            "paired": {"shared_qids": shared, "candidate_only_correct": cand_only,
                       "incumbent_only_correct": inc_only},
        })
    pin = row.get("baseline_pin") if isinstance(row.get("baseline_pin"), dict) else {}
    if pin:
        pinned = {"source": pin.get("source") or "structured",
                  "baseline_quality": _float(pin.get("baseline_quality")),
                  "era": pin.get("eval_quality_era") or None}
    else:
        # Only the quality-regression sentence names a QUALITY baseline; the generic
        # "baseline N" regex also matches the throughput-floor sentence (a t/s value).
        m = _QUALITY_BASELINE_PROSE_RE.search(str(row.get("failure_analysis") or ""))
        val = _float(m.group(1)) if m else None
        if val is not None and val <= QUALITY_SCALE:
            pinned = {"source": "legacy_quality_regression_prose", "baseline_quality": val, "era": None}
        else:
            pinned = {"source": "absent" if m is None else "legacy_scale_suspect",
                      "baseline_quality": None, "era": None}
    if pinned["baseline_quality"] is not None and q is not None:
        pinned["delta"] = q - pinned["baseline_quality"]
    seq = row.get("seq") if isinstance(row.get("seq"), dict) else {}
    sig = signature(row)
    same = replication.get(sig, [])
    return {
        "trial_id": row.get("trial_id"),
        "timestamp": row.get("timestamp"),
        "tier": tier,
        "levers": levers_for(row),
        "producer_class": pclass,
        "candidate": {"quality": q, "n_scored": n, "correct": correct, "se": se,
                      "reliability": _float(row.get("reliability")),
                      "task_rate_qph": _float((row.get("eval") or {}).get("task_rate_qph"))},
        "vs_incumbent": vs_inc,
        "vs_pinned_baseline": pinned,
        "replication": {"signature": sig, "valid_runs_same_signature": len(same),
                        "trial_ids": same[-8:], "seq_k": _int(seq.get("k")),
                        "seq_state": seq.get("state") or None,
                        "seq_confirmed": seq.get("confirmed") if seq else None},
        "verdict": row.get("keep_revert_decision") or "",
    }


# --------------------------------------------------------------------------- #
# builder
# --------------------------------------------------------------------------- #
def _row_summary(row: dict[str, Any], cls: dict[str, Any], era: str) -> dict[str, Any]:
    return {
        "trial_id": row.get("trial_id"),
        "timestamp": row.get("timestamp"),
        "era_bucket": era,
        "era_label": bucket_label(era),
        "species": row.get("species"),
        "intervention": intervention_summary(row),
        "hypothesis": _clip(row.get("hypothesis"), 300),
        "expected_mechanism": _clip(row.get("expected_mechanism"), 160),
        "falsifier": _clip(row.get("falsifier"), 300) or None,
        "verdict": row.get("keep_revert_decision") or "",
        "outcome_status": row.get("outcome_status") or "ok",
        "states": cls["states"],
        "drop": cls["drop"],
        "runtime_state": cls["runtime_state"],
    }


def _promotions(journal: dict[str, Any], state: dict[str, Any] | None) -> dict[int, str]:
    out: dict[int, str] = {}
    for ev in (journal.get("events") or {}).get("baseline_promotion", []):
        tid = _int(ev.get("source_trial_id"))
        if tid is not None:
            out[tid] = f"baseline_promotion_event:{_reason_code(ev.get('reason'))}"
    for row in journal.get("trials") or []:
        seq = row.get("seq") if isinstance(row.get("seq"), dict) else {}
        tid = _int(row.get("trial_id"))
        if tid is not None and seq.get("baseline_promotion_finalized") and tid not in out:
            out[tid] = "seq_baseline_promotion_finalized"
    if isinstance(state, dict):
        acc = state.get("multitier_last_accepted")
        if isinstance(acc, dict):
            tid = _int(acc.get("terminal_trial_id"))
            if tid is not None:
                out.setdefault(tid, "multitier_candidate_accepted")
    return out


def build_decision_cockpit(
    *,
    journal_dir: Path = DEFAULT_JOURNAL_DIR,
    state_path: Path = DEFAULT_STATE_PATH,
    eras_path: Path = DEFAULT_ERAS_PATH,
    study_db: Path = DEFAULT_STUDY_DB,
    checkpoints_dir: Path = DEFAULT_CHECKPOINTS_DIR,
    era: str | None = None,
    status_filter: str | None = None,
    now: float | None = None,
    graph_limit: int = 60,
    candidate_limit: int = 25,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc).timestamp() if now is None else float(now)
    eras_in = read_eras(Path(eras_path))
    journal = read_journal(Path(journal_dir))
    state_in = read_state(Path(state_path))
    ckpt = read_checkpoint(Path(checkpoints_dir))
    studies = read_studies(Path(study_db))
    state = state_in.get("data") if state_in["status"] == "ok" else None
    eras = eras_in["eras"] if eras_in["status"] == "ok" else []

    caveats = [OP20_CAVEAT, ERA_CAVEAT, LIVE_CAVEAT, SE_CAVEAT]

    # ---- eras ---------------------------------------------------------------
    active = (state or {}).get("active_instrument_eras") if state else None
    active = active if isinstance(active, dict) else {}
    registry_q = era_at(eras, now, QUALITY_SCOPES) if eras else None
    registry_s = era_at(eras, now, (SPEED_SCOPE,)) if eras else None
    current_bucket: str | None
    current_source: str
    if active.get("eval_quality") and active.get("autopilot_speed"):
        current_bucket = bucket_id(str(active["eval_quality"]), str(active["autopilot_speed"]))
        current_source = "autopilot_state.active_instrument_eras"
        if eras and current_bucket != bucket_id(registry_q, registry_s):
            caveats.append(
                "state active_instrument_eras disagrees with the registry's latest era at now "
                f"({bucket_label(current_bucket)} vs {bucket_label(bucket_id(registry_q, registry_s))}); "
                "the state's declared era is shown")
    elif eras:
        current_bucket = bucket_id(registry_q, registry_s)
        current_source = "instrument_eras_registry_fallback"
    else:
        current_bucket = None
        current_source = UNKNOWN

    rows = journal.get("trials") or []
    row_bucket: dict[int, str] = {}
    bucket_counts: dict[str, int] = {}
    for row in rows:
        ts = _parse_ts(row.get("timestamp"))
        b = bucket_id(era_at(eras, ts, QUALITY_SCOPES), era_at(eras, ts, (SPEED_SCOPE,))) if eras else UNKNOWN
        row_bucket[int(row["trial_id"])] = b
        bucket_counts[b] = bucket_counts.get(b, 0) + 1
    available = sorted(bucket_counts, key=lambda b: min(
        (int(r["trial_id"]) for r in rows if row_bucket[int(r["trial_id"])] == b), default=0))
    if current_bucket and current_bucket not in bucket_counts:
        available.append(current_bucket)
    selected = (era if era and era != "current" else current_bucket)
    selection_note = None
    if era and era != "current" and era not in bucket_counts and era != current_bucket:
        selection_note = f"requested era {era!r} has no journaled trials"

    live_tid = ckpt.get("trial_id") if ckpt["status"] == "ok" else None
    promotions = _promotions(journal, state)
    classified: dict[int, dict[str, Any]] = {}
    for row in rows:
        classified[int(row["trial_id"])] = classify(
            row, promotions=promotions, live_trial_id=live_tid, checkpoint_known=ckpt["status"] == "ok")

    journal_known = journal["status"] in ("ok", "partial")
    era_known = selected is not None and bool(eras)
    in_era = [r for r in rows if era_known and row_bucket[int(r["trial_id"])] == selected]
    interventions = [r for r in in_era if str(r.get("action_type") or "") not in EVIDENCE_ONLY_ACTIONS]
    evidence_only = [r for r in in_era if str(r.get("action_type") or "") in EVIDENCE_ONLY_ACTIONS]

    # ---- funnel -------------------------------------------------------------
    funnel: dict[str, Any] = {"era_bucket": selected, "known": journal_known and era_known}
    if funnel["known"]:
        counts = {s: 0 for s in STAGES}
        unknown_live = 0
        reasons: dict[str, dict[str, dict[str, Any]]] = {s: {} for s in STAGES[1:]}
        anomalies: dict[str, int] = {}
        for r in interventions:
            c = classified[int(r["trial_id"])]
            for s in STAGES:
                v = c["states"][s]
                if v is True:
                    counts[s] += 1
            if c["states"]["currently_live"] == UNKNOWN:
                unknown_live += 1
            d = c["drop"]
            if d:
                slot = reasons[d["stage"]].setdefault(d["code"], {"count": 0, "example": d["detail"],
                                                                   "trial_ids": []})
                slot["count"] += 1
                slot["trial_ids"] = (slot["trial_ids"] + [r["trial_id"]])[-6:]
            for a in c["anomalies"]:
                anomalies[a] = anomalies.get(a, 0) + 1
        stages = []
        prev = None
        for s in STAGES:
            entry: dict[str, Any] = {"stage": s, "count": counts[s]}
            if s == "currently_live":
                entry["unknown"] = unknown_live
            if s == "currently_live":
                # Not a subset of "promoted": a kept config is applied at keep time,
                # with or without a later baseline promotion.
                entry["dropped"] = None
                entry["of_stage"] = "kept"
                entry["reasons"] = sorted(
                    ({"code": k, **v} for k, v in reasons[s].items()),
                    key=lambda x: -x["count"])
            elif prev is not None:
                entry["dropped"] = prev - counts[s]
                entry["reasons"] = sorted(
                    ({"code": k, **v} for k, v in reasons[s].items()),
                    key=lambda x: -x["count"])
            stages.append(entry)
            prev = counts[s]
        by_producer: dict[str, dict[str, int]] = {}
        for r in in_era:
            p = producer_class(r)
            c = classified[int(r["trial_id"])]["states"]
            slot = by_producer.setdefault(p, {"rows": 0, "executed": 0, "valid": 0})
            slot["rows"] += 1
            slot["executed"] += int(c["executed"] is True)
            slot["valid"] += int(c["valid"] is True)
        funnel.update({
            "stages": stages,
            "anomalies": anomalies,
            "evidence_only_actions": {
                "count": len(evidence_only),
                "by_type": _count_by(evidence_only, "action_type"),
                "note": "evidence-gathering / maintenance actions; not interventions, not in the funnel",
            },
            "by_producer_class": by_producer,
        })
        crs = (state or {}).get("critic_rejected_signatures") if state else None
        if isinstance(crs, dict):
            lo, hi = _bucket_window(eras, selected)
            n_fence = 0
            for rec in crs.values():
                ts = _parse_ts(rec.get("recorded_at")) if isinstance(rec, dict) else None
                if ts is not None and (lo is None or ts >= lo) and (hi is None or ts < hi):
                    n_fence += int(rec.get("count") or 1)
            funnel["critic_rejections_before_journal"] = {
                "count": n_fence,
                "note": "planner drafts the critic rejected (state ledger); they never became journal "
                        "rows, so they sit outside the proposed count",
            }
        else:
            funnel["critic_rejections_before_journal"] = {"count": None, "note": "state unknown"}
    else:
        funnel["reason"] = ("journal unknown" if not journal_known else "era unknown")

    # ---- replication index ----------------------------------------------------
    replication: dict[str, list[int]] = {}
    for r in interventions:
        if classified[int(r["trial_id"])]["states"]["valid"] is True:
            replication.setdefault(signature(r), []).append(int(r["trial_id"]))

    # ---- objective deltas -----------------------------------------------------
    incumbent = incumbent_from_state(state, eras) if state else {"available": False, "reason": "state unknown"}
    valid_iv = [r for r in interventions if classified[int(r["trial_id"])]["states"]["valid"] is True]
    cand_rows = [delta_row(r, incumbent=incumbent, era=str(selected), replication=replication)
                 for r in valid_iv]
    seeding_rows = [r for r in in_era if producer_class(r) == "seeding"
                    and classified[int(r["trial_id"])]["states"]["valid"] is True]
    comparable = [c for c in cand_rows if c["vs_incumbent"]["comparable"]]
    best_by_tier: dict[str, Any] = {}
    for c in comparable:
        t = str(c["tier"])
        if t not in best_by_tier or c["vs_incumbent"]["delta"] > best_by_tier[t]["delta"]:
            best_by_tier[t] = {"trial_id": c["trial_id"], "delta": c["vs_incumbent"]["delta"],
                               "delta_se": c["vs_incumbent"]["delta_se"]}
    inc_public = {k: v for k, v in incumbent.items() if k != "tiers"}
    if incumbent.get("tiers"):
        inc_public["tiers"] = {t: {k: v for k, v in d.items() if not k.startswith("_")}
                               for t, d in incumbent["tiers"].items()}
    objective_deltas = {
        "era_bucket": selected,
        "incumbent": inc_public,
        "candidates": list(reversed(cand_rows))[:candidate_limit],
        "candidate_count": len(cand_rows),
        "comparable_count": len(comparable),
        "best_comparable_by_tier": best_by_tier,
        "seeding_producer": {
            "valid_rows": len(seeding_rows),
            "mean_quality": (sum(_float(r.get("quality")) or 0 for r in seeding_rows) / len(seeding_rows))
            if seeding_rows else None,
            "note": OP20_CAVEAT,
        },
        "empty_reason": (None if cand_rows else
                         "no valid intervention trial has been journaled in this era"),
    }

    # ---- lever scoreboard -----------------------------------------------------
    levers: dict[str, dict[str, Any]] = {}
    cand_by_tid = {x["trial_id"]: x for x in cand_rows}
    for r in interventions:
        c = classified[int(r["trial_id"])]
        dr = cand_by_tid.get(r["trial_id"])
        for lever in levers_for(r):
            slot = levers.setdefault(lever, {
                "lever": lever, **{s: 0 for s in STAGES}, "currently_live_unknown": 0,
                "reverted": 0, "deltas": [], "delta_sources": {}, "last_trial_id": None,
                "last_timestamp": None, "multi_factor_trials": 0,
            })
            for s in STAGES:
                slot[s] += int(c["states"][s] is True)
            slot["currently_live_unknown"] += int(c["states"]["currently_live"] == UNKNOWN)
            slot["reverted"] += int(str(r.get("keep_revert_decision") or "") in ("revert", "excluded"))
            slot["multi_factor_trials"] += int(len(levers_for(r)) > 1)
            slot["last_trial_id"] = r["trial_id"]
            slot["last_timestamp"] = r.get("timestamp")
            if dr is not None:
                if dr["vs_incumbent"].get("comparable"):
                    slot["deltas"].append(dr["vs_incumbent"]["delta"])
                    src = "incumbent"
                elif dr["vs_pinned_baseline"].get("delta") is not None:
                    slot["deltas"].append(dr["vs_pinned_baseline"]["delta"])
                    src = str(dr["vs_pinned_baseline"]["source"])
                else:
                    src = None
                if src:
                    slot["delta_sources"][src] = slot["delta_sources"].get(src, 0) + 1
    speed_era = str(selected).partition("|")[2] if selected else ""
    study_prefix = f"_era_{speed_era.replace('-', '_')}" if speed_era and speed_era != PRE_REGISTRY else None
    study_status = studies["status"]
    if study_prefix and study_status == "ok":
        pat = re.compile(r"^autopilot_(.+?)" + re.escape(study_prefix) + r"(?:_epoch\d+)?$")
        for name, rec in studies["studies"].items():
            m = pat.match(name)
            if not m:
                continue
            lever = f"numeric:{m.group(1)}"
            slot = levers.setdefault(lever, {
                "lever": lever, **{s: 0 for s in STAGES}, "currently_live_unknown": 0,
                "reverted": 0, "deltas": [], "delta_sources": {}, "last_trial_id": None,
                "last_timestamp": None, "multi_factor_trials": 0,
            })
            st = slot.setdefault("study", {"names": [], "complete": 0, "fail": 0, "running": 0,
                                           "best_quality": None})
            st["names"].append(name)
            for k in ("complete", "fail", "running"):
                st[k] += rec[k]
            if rec["best_quality"] is not None and (st["best_quality"] is None
                                                    or rec["best_quality"] > st["best_quality"]):
                st["best_quality"] = rec["best_quality"]
    scoreboard = []
    for slot in levers.values():
        ds = slot.pop("deltas")
        slot["n_deltas"] = len(ds)
        slot["best_delta"] = max(ds) if ds else None
        slot["mean_delta"] = (sum(ds) / len(ds)) if ds else None
        slot["evidence_strength"] = ("none" if not ds else "anecdotal" if len(ds) < 3 else "repeated")
        if slot["lever"].startswith("numeric:") and "study" not in slot:
            slot["study"] = ({"status": study_status} if study_status != "ok"
                             else {"names": [], "complete": 0, "fail": 0, "running": 0,
                                   "best_quality": None, "note": "no study for this surface in this era"})
        scoreboard.append(slot)
    scoreboard.sort(key=lambda s: (-s["kept"], -(s["best_delta"] if s["best_delta"] is not None else -9),
                                   -s["executed"], s["lever"]))

    # ---- provenance graph -----------------------------------------------------
    graph_rows = interventions
    if status_filter and status_filter in STAGES:
        graph_rows = [r for r in graph_rows if classified[int(r["trial_id"])]["states"][status_filter] is True]
    truncated = len(graph_rows) > graph_limit
    graph_rows = graph_rows[-graph_limit:]
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []
    for r in graph_rows:
        tid = int(r["trial_id"])
        c = classified[tid]
        hyp = _clip(r.get("hypothesis") or "(no hypothesis recorded)", 140)
        hid = "h:" + hashlib.sha256(hyp.encode()).hexdigest()[:10]
        nodes.setdefault(hid, {"id": hid, "column": "hypothesis", "label": hyp, "trial_ids": []})
        nodes[hid]["trial_ids"].append(tid)
        xid = f"x:{tid}"
        nodes[xid] = {"id": xid, "column": "experiment", "trial_id": tid,
                      "label": f"T{tid} {r.get('action_type')} {', '.join(levers_for(r))}"[:120],
                      "falsifier": _clip(r.get("falsifier"), 200) or None, "states": c["states"]}
        edges.append({"from": hid, "to": xid, "relation": "tested_by"})
        vid = f"v:{tid}"
        verdict = r.get("keep_revert_decision") or ("not executed" if not c["states"]["executed"]
                                                      else "no verdict")
        nodes[vid] = {"id": vid, "column": "verdict", "trial_id": tid, "label": verdict,
                      "drop": c["drop"], "states": c["states"]}
        if c["states"]["executed"]:
            eid = f"e:{tid}"
            details = r.get("details") or {}
            q = _float(r.get("quality"))
            nodes[eid] = {"id": eid, "column": "evidence", "trial_id": tid,
                          "label": (f"q={q:.3f} n={details.get('n_scored')} T{r.get('tier')}"
                                    if q is not None else "no quality"),
                          "valid": c["states"]["valid"] is True,
                          "producer_class": producer_class(r), "states": c["states"]}
            edges.append({"from": xid, "to": eid, "relation": "produced"})
            edges.append({"from": eid, "to": vid, "relation": "judged"})
        else:
            edges.append({"from": xid, "to": vid, "relation": "not_executed"})
        rt = c["runtime_state"]
        nodes.setdefault(rt, {"id": rt, "column": "runtime_state", "label": RUNTIME_STATES[rt]})
        edges.append({"from": vid, "to": rt, "relation": "resulted_in"})
    provenance = {
        "era_bucket": selected,
        "columns": ["hypothesis", "experiment", "evidence", "verdict", "runtime_state"],
        "nodes": list(nodes.values()),
        "edges": edges,
        "edge_semantics": EDGE_SEMANTICS,
        "filters": {"eras": [{"id": b, "label": bucket_label(b), "trials": bucket_counts.get(b, 0)}
                             for b in available],
                    "statuses": list(STAGES), "status_applied": status_filter if status_filter in STAGES else None},
        "truncated": truncated,
        "limit": graph_limit,
    }

    # ---- current trial --------------------------------------------------------
    in_flight = (state or {}).get("in_flight_trial") if state else None
    paused = (state or {}).get("paused") if state else None
    if state is None:
        cur_status = UNKNOWN
    elif isinstance(in_flight, dict) and in_flight:
        cur_status = "in_flight"
    elif paused is True:
        cur_status = "idle_declared_paused"
    else:
        cur_status = "none_in_flight"
    last = rows[-1] if rows else None
    last_in_era = in_era[-1] if in_era else None
    current = {
        "status": cur_status,
        "producer_declared": {"paused": paused if state else UNKNOWN,
                              "trial_counter": (state or {}).get("trial_counter") if state else UNKNOWN},
        "in_flight": ({"trial_id": in_flight.get("trial_id"),
                       "action": in_flight.get("action") if isinstance(in_flight.get("action"), dict) else None,
                       "started_at": in_flight.get("started_at")}
                      if isinstance(in_flight, dict) and in_flight else None),
        "last_journaled": (_row_summary(last, classified[int(last["trial_id"])],
                                        row_bucket[int(last["trial_id"])]) if last else None),
        "last_in_selected_era": (_row_summary(last_in_era, classified[int(last_in_era["trial_id"])], str(selected))
                                 if last_in_era else None),
        "last_in_selected_era_reason": (None if last_in_era else
                                        ("journal unknown" if not journal_known else
                                         "no trial journaled in the selected era")),
    }

    # ---- freshness (producer-written timestamps, never mtime) ----------------
    newest = max((_parse_ts(r.get("timestamp")) or 0.0 for r in rows), default=0.0) or None
    newest_era = max((_parse_ts(r.get("timestamp")) or 0.0 for r in in_era), default=0.0) or None
    evidence_freshness = {
        "newest_trial_at": _iso(newest),
        "newest_trial_age_s": (now - newest) if newest else None,
        "newest_in_selected_era_at": _iso(newest_era),
        "producer_declared_state": ("paused" if paused is True else "running_or_undeclared"
                                    if state else UNKNOWN),
        "note": ("AutoPilot declares itself paused: silence is the declared state, not a dead producer"
                 if paused is True else
                 "no declared pause: an old newest_trial_at means the producer stopped reporting"),
    }

    inputs = {
        "journal": {k: v for k, v in journal.items() if k not in ("trials", "events")},
        "state": {k: v for k, v in state_in.items() if k != "data"},
        "eras": {k: v for k, v in eras_in.items() if k != "eras"},
        "study": {k: v for k, v in studies.items() if k != "studies"},
        "production_best": ckpt,
    }
    if studies["status"] == "ok":
        inputs["study"]["study_count"] = len(studies["studies"])

    payload = {
        "schema": SCHEMA,
        "generated_at": now,
        "generated_at_iso": _iso(now),
        "read_only": True,
        "authority": "observe_only: no fitness, archive, promotion or authority effect",
        "inputs": inputs,
        "eras": {
            "current": {"bucket": current_bucket, "label": bucket_label(current_bucket) if current_bucket else UNKNOWN,
                        "source": current_source,
                        "active_instrument_eras": active or (UNKNOWN if state is None else {})},
            "selected": selected,
            "selected_label": bucket_label(selected) if selected else UNKNOWN,
            "selection_note": selection_note,
            "available": provenance["filters"]["eras"],
            "selected_trial_count": len(in_era) if era_known else None,
        },
        "current": current,
        "objective_deltas": objective_deltas,
        "funnel": funnel,
        "lever_scoreboard": {"era_bucket": selected, "source": "journal_shards+optuna_study",
                             "digest_used": False, "levers": scoreboard},
        "provenance": provenance,
        "evidence_freshness": evidence_freshness,
        "caveats": caveats,
    }
    payload["health"] = cockpit_health(payload)
    return payload


def _count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        k = str(r.get(key) or UNKNOWN)
        out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def _bucket_window(eras: list[dict[str, Any]], bucket: str | None) -> tuple[float | None, float | None]:
    """[start, end) of a bucket: the later of its two era starts, the next boundary after."""
    if not bucket or not eras:
        return None, None
    q, _, s = bucket.partition("|")
    starts = [e["from_ts"] for e in eras if e["id"] in (q, s) and e["from_ts"] is not None]
    lo = max(starts) if starts else None
    later = [e["from_ts"] for e in eras
             if e["scope"] in (*QUALITY_SCOPES, SPEED_SCOPE) and e["from_ts"] is not None
             and lo is not None and e["from_ts"] > lo]
    return lo, (min(later) if later else None)


def cockpit_health(payload: dict[str, Any]) -> dict[str, Any]:
    """Three-valued data health: ok / absent / degraded (never green by default)."""
    inputs = payload.get("inputs") or {}
    reasons: list[str] = []
    status = "ok"
    j = (inputs.get("journal") or {}).get("status")
    if j == "absent":
        return {"status": "absent", "reasons": ["journal: no shard found"]}
    for name in ("journal", "state", "eras"):
        st = (inputs.get(name) or {}).get("status")
        if st != "ok":
            status = "degraded"
            reasons.append(f"{name}: {st}")
    for name in ("study", "production_best"):
        st = (inputs.get(name) or {}).get("status")
        if st not in ("ok", "absent"):
            status = "degraded"
            reasons.append(f"{name}: {st}")
    if not (payload.get("eras") or {}).get("selected"):
        status = "degraded"
        reasons.append("era: current measurement era unknown")
    return {"status": status, "reasons": reasons}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--journal-dir", type=Path, default=DEFAULT_JOURNAL_DIR)
    ap.add_argument("--state", type=Path, default=DEFAULT_STATE_PATH)
    ap.add_argument("--eras", type=Path, default=DEFAULT_ERAS_PATH)
    ap.add_argument("--study-db", type=Path, default=DEFAULT_STUDY_DB)
    ap.add_argument("--checkpoints", type=Path, default=DEFAULT_CHECKPOINTS_DIR)
    ap.add_argument("--era", default=None, help="era bucket id ('Q|S') or 'current'")
    ap.add_argument("--status", default=None, choices=STAGES)
    ap.add_argument("--summary", action="store_true", help="print a short parse summary, not the contract")
    args = ap.parse_args(argv)
    payload = build_decision_cockpit(
        journal_dir=args.journal_dir, state_path=args.state, eras_path=args.eras,
        study_db=args.study_db, checkpoints_dir=args.checkpoints, era=args.era,
        status_filter=args.status,
    )
    if args.summary:
        j = payload["inputs"]["journal"]
        print(json.dumps({
            "health": payload["health"],
            "journal": {k: j.get(k) for k in ("status", "shard_count", "trial_rows", "event_rows",
                                              "bad_lines", "duplicate_trial_ids", "supersessions_applied")},
            "shards": j.get("shards"),
            "eras_found": payload["eras"]["available"],
            "current_era": payload["eras"]["current"],
            "selected_trial_count": payload["eras"]["selected_trial_count"],
            "funnel": [(s["stage"], s["count"]) for s in payload["funnel"].get("stages", [])],
            "levers": len(payload["lever_scoreboard"]["levers"]),
            "study": payload["inputs"]["study"],
        }, indent=1, default=str))
    else:
        print(json.dumps(payload, indent=1, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
