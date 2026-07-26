"""Read-only synthesis of AutoPilot planner assessments into an operator brief.

The node-link insight graph answers *provenance* ("which hint led to which
trial"). This module answers the operator's actual question — *what makes the
orchestrator work best, and how sure are we* — by aggregating the planner's
assessments into four legible sections:

  1. narrative      — a deterministic, templated paragraph (never free-written,
                      so it cannot assert a finding the numbers don't support)
  2. best_config    — the current production-best config, decomposed
  3. levers         — knobs ranked by fANOVA importance + cluster-best value
  4. ruled_out /    — the explored boundary: falsified guardrails (conventions)
     exploring        and queued hypotheses (patterns)

Every numeric claim carries `decision_grade` (per MEASUREMENT.md): it is only
True when planner authority is actually enabled. The honesty banner surfaces
that authority state up front, so an operator never mistakes an observation for
a ratified result.

Design constraints:
  * Pure, read-only functions over file paths -> JSON-ready dict.
  * No live Optuna/NumericSwarm instantiation — the lever ledger is parsed from
    the most recently generated digest markdown, which avoids contending with
    the running autopilot's study DB.
  * Reuses the proven W6 audit reporter for the trust banner.
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any

from scripts.autopilot.audit_block_report import build_report as _w6_build_report
from scripts.autopilot.journal_shards import journal_shards
from src.autopilot_core.baseline_ledger import baseline_ledger_authority_enabled

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_PATH = REPO_ROOT / "orchestration" / "autopilot_state.json"
# JRN-5/7: discover every rotated shard (was a hardcoded base + _1 pair that
# silently ignored _2 and beyond once the journal rotated a second time).
DEFAULT_JOURNAL_PATHS = journal_shards(REPO_ROOT / "orchestration")
DEFAULT_STRATEGY_DB = (
    REPO_ROOT / "orchestration" / "repl_memory" / "strategies" / "strategies.db"
)
# Autopilot digests are written into the epyc-root governance tree, not the
# orchestrator repo (matches the epyc-root paths dashboard.py already uses).
DEFAULT_DIGEST_DIR = Path("/mnt/raid0/llm/epyc-root/progress")

# --------------------------------------------------------------------------- #
# GEPA optimizer-provenance no-op window (display-only honesty label).
# Incident (progress/2026-07-25 GEPA no-op finding): the GEPA reflective-mutation
# path was a GUARANTEED NO-OP — it silently produced no prompt mutation — from
# 2026-06-04 (trial-521 evidence) until fix commit ed6288ea ("fix(autopilot):
# GEPA reflective mutation was a guaranteed no-op; thread --numa-mode into stack
# gate", author date 2026-07-25T11:27:13Z). Every gepa_optimize trial/fence/churn
# row minted inside this window carries broken optimizer provenance. Label only —
# never delete rows or rescale values across this window.
# --------------------------------------------------------------------------- #
GEPA_NOOP_WINDOW_FROM_TS = 1780531200.0  # 2026-06-04T00:00:00Z (trial-521 evidence)
GEPA_NOOP_WINDOW_UNTIL_TS = 1784978833.0  # 2026-07-25T11:27:13Z (ed6288ea author date)
GEPA_NOOP_WINDOW_LABEL = "reflective-mutation no-op — optimizer provenance broken"
GEPA_PROVENANCE_WINDOWS: list[dict[str, Any]] = [
    {
        "from_ts": GEPA_NOOP_WINDOW_FROM_TS,
        "until_ts": GEPA_NOOP_WINDOW_UNTIL_TS,
        "label": GEPA_NOOP_WINDOW_LABEL,
    }
]
GEPA_NOOP_CAVEAT = (
    "GEPA reflective-mutation no-op window 2026-06-04 → 2026-07-25 — "
    "optimizer provenance broken for rows minted in this window"
)

_UNTRUSTED_STATUSES = {"invalid", "skipped"}
# AP-24 verdict records that disqualify a trial from "best config": a reverted
# config is not running, and a learning-excluded measurement (mad_noise /
# exogenous reload) must not crown an incumbent. Legacy rows without the field
# ("") and kept/unchanged rows stay eligible. NOTE: a journal row's
# outcome_status stays "ok" when the safety verdict fails (the *experiment* ran
# fine), so _is_trusted alone cannot catch these — trial 1061 (2026-07-02) was
# displayed as best config after failing its verdict and being rolled back.
_INELIGIBLE_KEEP_REVERT = {"revert", "excluded"}


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def load_journal_rows(paths: list[Path]) -> list[dict[str, Any]]:
    """Load trial rows from one or more journal JSONL files (snapshots skipped)."""
    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path or not Path(path).exists():
            continue
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and obj.get("type") is None and "trial_id" in obj:
                rows.append(obj)
    return rows


def load_state(path: Path) -> dict[str, Any]:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def latest_digest_text(digest_dir: Path) -> str:
    """Return the text of the most recent `*-autopilot.md` digest, or ''."""
    try:
        candidates = sorted(Path(digest_dir).glob("*/*-autopilot.md"))
    except OSError:
        return ""
    if not candidates:
        return ""
    try:
        return candidates[-1].read_text()
    except OSError:
        return ""


# --------------------------------------------------------------------------- #
# Section builders
# --------------------------------------------------------------------------- #
def _autopilot_seq_verdict_live() -> bool:
    """True iff a live ``autopilot.py`` process has ``AUTOPILOT_SEQ_VERDICT``
    truthy. Sequential-verdict authority is env-gated on the running autopilot
    (not a state flag), so we read it from the live process. Fail-safe: if no
    autopilot is running or the env is unreadable, returns False (off)."""
    import glob

    for cmdpath in glob.glob("/proc/[0-9]*/cmdline"):
        try:
            cmd = open(cmdpath, "rb").read()
        except OSError:
            continue
        if b"autopilot.py" not in cmd:
            continue
        pid = cmdpath.rsplit("/", 2)[1]
        try:
            raw = open(f"/proc/{pid}/environ", "rb").read()
        except OSError:
            continue
        for entry in raw.split(b"\0"):
            if entry.startswith(b"AUTOPILOT_SEQ_VERDICT="):
                return entry.split(b"=", 1)[1].strip().lower() in (b"1", b"true", b"yes", b"on")
    return False


def _era_short(era_id: Any) -> str:
    """'E8-autopilot-speed' → 'E8' (mirrors the dashboard's short-label rule)."""
    m = re.match(r"(E\d+[a-z]?)", str(era_id or ""))
    return m.group(1) if m else str(era_id or "")


def _live_era_holds(state: dict[str, Any]) -> dict[str, Any]:
    """Read the SAME fail-closed era holds the safety gate enforces (E1, 2026-07-26).

    Quality: ``SafetyGate.quality_rebaseline_required`` — the resident baseline's
    ``eval_quality_era`` differs from ``active_instrument_eras.eval_quality``
    (e.g. E7 baseline under an active E8 era). Speed: the state's open
    ``frontier_rerun_required`` marker (``required`` not False), with its
    completed/min numeric-trial counters. No new state is invented here — these
    are exactly the keys the gate reads, so the banner can never contradict the
    gate's "archive.update SKIPPED (safety verdict failed)" behaviour again.
    """
    active_eras = state.get("active_instrument_eras")
    active_eras = active_eras if isinstance(active_eras, dict) else {}
    active_quality_era = str(active_eras.get("eval_quality") or "").strip()
    baseline_state = state.get("baseline_state")
    baseline_state = baseline_state if isinstance(baseline_state, dict) else {}
    baseline_quality_era = str(baseline_state.get("eval_quality_era") or "").strip()
    quality_hold = bool(active_quality_era) and baseline_quality_era != active_quality_era

    frr = state.get("frontier_rerun_required")
    frr = frr if isinstance(frr, dict) else None
    speed_hold = bool(frr) and frr.get("required") is not False
    try:
        completed = int((frr or {}).get("completed_numeric_trials") or 0)
    except (TypeError, ValueError):
        completed = 0
    try:
        minimum = int((frr or {}).get("min_numeric_trials") or 0)
    except (TypeError, ValueError):
        minimum = 0

    q_short = _era_short(active_quality_era) or "current-era"
    s_short = _era_short(active_eras.get("autopilot_speed")) or q_short
    return {
        "quality_rebaseline_required": quality_hold,
        "quality_authority": (
            f"HELD pending {q_short} baseline (fail-closed)"
            if quality_hold
            else "OK — baseline era matches active eval_quality era"
            + (f" ({q_short})" if active_quality_era else "")
        ),
        "quality_hold_detail": {
            "active_eval_quality_era": active_quality_era or None,
            "baseline_eval_quality_era": baseline_quality_era or None,
        },
        "frontier_rerun_required": speed_hold,
        "speed_authority": (
            f"pending {s_short} numeric rerun ({completed}/{minimum})"
            if speed_hold
            else "OK — no frontier rerun marker open"
        ),
        "speed_hold_detail": {
            "completed_numeric_trials": completed if frr else None,
            "min_numeric_trials": minimum if frr else None,
            "opened_at": (frr or {}).get("opened_at"),
        },
        "any_hold_active": quality_hold or speed_hold,
    }


def authority_banner(
    state: dict[str, Any],
    w6: dict[str, Any],
    *,
    seq_verdict_live: bool | None = None,
) -> dict[str, Any]:
    """Surface whether any planner finding can currently be ratified.

    Baseline authority = the **consent-gated** state flag (state flag AND the
    operator-owned consent file). Sequential authority = the live autopilot's
    ``AUTOPILOT_SEQ_VERDICT`` env (env-gated, not a state flag). Both must hold
    and the W6 gaming alarm must be clear — AND no fail-closed era hold may be
    open (E8 quality rebaseline / frontier rerun) — for findings to be
    decision-grade. The holds come from the live state via ``_live_era_holds``:
    trial 1446 showed the old banner printing "kept configs are decision-grade"
    while the gate was skipping ``archive.update`` on
    ``quality_rebaseline_required`` — the banner must never contradict the gate.
    """
    baseline = baseline_ledger_authority_enabled(state)
    if seq_verdict_live is None:
        seq_verdict_live = _autopilot_seq_verdict_live()
    sequential = bool(seq_verdict_live)
    gaming_alarm = bool(w6.get("gaming_alarm"))
    holds = _live_era_holds(state)
    authority_mechanism_enabled = baseline and sequential and not gaming_alarm
    decision_grade_possible = authority_mechanism_enabled and not holds["any_hold_active"]

    if decision_grade_possible:
        trust_note = (
            "Authority ENABLED (baseline + sequential, W6 clear) — kept configs are decision-grade."
        )
    elif holds["any_hold_active"]:
        hold_bits = []
        if holds["quality_rebaseline_required"]:
            hold_bits.append(f"quality: {holds['quality_authority']}")
        if holds["frontier_rerun_required"]:
            hold_bits.append(f"speed: {holds['speed_authority']}")
        mech = "" if authority_mechanism_enabled else " Planner authority is also OFF."
        trust_note = (
            "Era holds ACTIVE — "
            + "; ".join(hold_bits)
            + ". Findings below are OBSERVATIONS (fail-closed); kept configs are "
            "NOT decision-grade until the holds clear." + mech
        )
    else:
        trust_note = (
            "Authority OFF — every finding below is an OBSERVATION, not "
            "decision-grade; no config can be promoted yet."
        )

    return {
        "baseline_authority_enabled": baseline,
        "sequential_authority_enabled": sequential,
        "w6_gaming_alarm": gaming_alarm,
        "w6_clearance_clean_trials_required": w6.get(
            "gaming_alarm_clearance_clean_trials_required"
        ),
        "current_era_audited_trials": w6.get("trusted_audited_trial_count"),
        "authority_mechanism_enabled": authority_mechanism_enabled,
        "holds": holds,
        "decision_grade_possible": decision_grade_possible,
        "trust_note": trust_note,
    }


_SURFACE_HDR = re.compile(r"^####\s+`([^`]+)`\s+—\s+(\d+)\s+completed trials")
_BEST_Q = re.compile(r"best quality \(obj 0\):\s+\*\*([\d.]+)\*\*")
_KV = re.compile(r"`([^`]+)`\s*=\s*([\d.eE+-]+)")
_CLUSTER_KV = re.compile(r"`([^`=]+)=([^`]+)`")


def levers_from_digest(digest_text: str) -> list[dict[str, Any]]:
    """Parse the NumericSwarm surface sections of a generated digest into a
    ranked lever ledger. Each lever = the dominant fANOVA knob of its surface
    with its cluster-selected recommended value."""
    levers: list[dict[str, Any]] = []
    surface: str | None = None
    n_trials = 0
    best_q: float | None = None
    cluster: dict[str, str] = {}
    importance: dict[str, float] = {}

    def _flush() -> None:
        if not surface or not importance:
            return
        top_key, top_imp = max(importance.items(), key=lambda kv: kv[1])
        levers.append(
            {
                "surface": surface,
                "lever": top_key,
                "importance": round(top_imp, 4),
                "recommended": cluster.get(top_key),
                "best_quality": best_q,
                "n_trials": n_trials,
                "all_importances": dict(
                    sorted(importance.items(), key=lambda kv: kv[1], reverse=True)
                ),
            }
        )

    for line in digest_text.splitlines():
        m = _SURFACE_HDR.match(line.strip())
        if m:
            _flush()
            surface, n_trials = m.group(1), int(m.group(2))
            best_q, cluster, importance = None, {}, {}
            continue
        if surface is None:
            continue
        mq = _BEST_Q.search(line)
        if mq:
            best_q = float(mq.group(1))
            continue
        if "cluster-selected best" in line:
            cluster = {k: v for k, v in _CLUSTER_KV.findall(line)}
            continue
        if "fANOVA importance" in line:
            importance = {k: float(v) for k, v in _KV.findall(line)}
            continue
    _flush()
    levers.sort(key=lambda x: x["importance"], reverse=True)
    return levers


def _is_trusted(row: dict[str, Any]) -> bool:
    if row.get("bug_corrupted_by"):
        return False
    try:
        if int(row.get("tier", 1)) < 1:
            return False
    except (TypeError, ValueError):
        return False
    return str(row.get("outcome_status") or "ok") not in _UNTRUSTED_STATUSES


def best_config(
    rows: list[dict[str, Any]],
    *,
    exclude_before_ts: float | None,
) -> dict[str, Any]:
    """Current best (highest-quality, speed tie-break) trusted current-era trial
    whose config survived its safety verdict (AP-24 keep), decomposed into its
    config settings. `status` is "promoted" only when the trial's own sequential
    promotion record finalized (seq.baseline_promotion_finalized) — the global
    authority banner enables promotion but never constitutes one."""
    eligible = [
        r
        for r in rows
        if _is_trusted(r)
        and (exclude_before_ts is None or _row_ts(r) >= exclude_before_ts)
        and r.get("quality") is not None
        and str(r.get("keep_revert_decision") or "") not in _INELIGIBLE_KEEP_REVERT
    ]
    if not eligible:
        return {"available": False}
    best = max(
        eligible,
        key=lambda r: (float(r.get("quality") or 0), float(r.get("speed") or 0)),
    )
    snap = best.get("config_snapshot") or {}
    settings = (
        [{"key": k, "value": v} for k, v in sorted(snap.items())][:12]
        if isinstance(snap, dict)
        else []
    )
    seq = best.get("seq") if isinstance(best.get("seq"), dict) else {}
    promoted = bool(seq.get("baseline_promotion_finalized"))
    return {
        "available": True,
        "trial_id": best.get("trial_id"),
        "promoted": promoted,
        "status": "promoted" if promoted else "incumbent",
        "keep_revert_decision": str(best.get("keep_revert_decision") or ""),
        "pareto_status": str(best.get("pareto_status") or ""),
        "objective": {
            "quality": best.get("quality"),
            "speed": best.get("speed"),
            "cost": best.get("cost"),
            "reliability": best.get("reliability"),
        },
        "settings": settings,
    }


def _row_ts(row: dict[str, Any]) -> float:
    raw = row.get("timestamp")
    if isinstance(raw, (int, float)):
        return float(raw)
    text = str(raw or "").strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        import datetime as _dt

        return _dt.datetime.fromisoformat(text).timestamp()
    except (ValueError, TypeError):
        return 0.0


def ruled_out_and_exploring(
    strategy_db: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Conventions (entry_type='convention') = ruled-out boundary;
    patterns (entry_type='pattern') = queued hypotheses."""
    ruled_out: list[dict[str, Any]] = []
    exploring: list[dict[str, Any]] = []
    if not Path(strategy_db).exists():
        return ruled_out, exploring
    try:
        conn = sqlite3.connect(f"file:{strategy_db}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
    except sqlite3.Error:
        return ruled_out, exploring
    try:
        cur = conn.execute(
            "SELECT id, description, insight, species, entry_type, metadata_json "
            "FROM strategies WHERE entry_type IN ('convention','pattern') "
            "ORDER BY created_at DESC LIMIT 200"
        )
        for r in cur:
            meta = {}
            try:
                meta = json.loads(r["metadata_json"] or "{}")
            except json.JSONDecodeError:
                pass
            item = {
                "id": r["id"],
                "statement": (r["insight"] or r["description"] or "").strip()[:300],
                "species": r["species"],
                "source_handoff": meta.get("source_handoff"),
                "bind_status": meta.get("bind_status"),
            }
            if r["entry_type"] == "convention":
                ruled_out.append(item)
            else:
                exploring.append(item)
    except sqlite3.Error:
        pass
    finally:
        conn.close()
    return ruled_out, exploring


_REJECT_TYPE_LABEL = {
    "structural_experiment": "structural flag flip",
    "prompt_mutation": "prompt edit",
    "code_mutation": "code edit",
    "numeric_trial": "numeric sweep",
    "seed_batch": "eval seeding",
    "deep_eval": "deep eval",
    "gepa_optimize": "GEPA prompt evolution",
    "structural_prune": "prompt prune",
    "few_shot_evolution": "few-shot edit",
    "targeted_fix": "targeted fix",
    "compress": "prompt compress",
}


def _short_reason(reason: Any) -> str:
    """First clause of a critic reason, banner-stripped, capped for display."""
    text = str(reason or "").strip()
    if text.lower().startswith("critic rejected:"):
        text = text.split(":", 1)[1].strip()
    for sep in ("; ", ". "):
        if sep in text:
            text = text.split(sep, 1)[0]
            break
    return text[:160]


def ruled_out_experiments(
    state: dict[str, Any], *, limit: int = 6, journal_rows: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Surface *what was tried and rejected, and why* — the intuition the bare
    "N dead-ends fenced" count omits. The two state ledgers carry different
    signal, so they stay separate rather than being pooled into one count race:

      * ``fenced`` (from ``critic_rejected_signatures``) — the critic vetoed a
        specific proposal; each row carries a human ``reason`` and a repeat
        ``count`` (how many times the planner re-proposed it = how firmly the
        boundary is fenced). This is the reason-bearing "and why" list.
      * ``invalid_by_surface`` (from journal rows, with state fallback) —
        proposals that failed structural validation before execution, grouped
        by experiment *type*: a churn summary of which surfaces the planner
        keeps drafting malformed actions against, not specific falsified ideas.
      * ``corrupted_by_surface`` (from journal rows) — executed trials that were
        exogenously corrupted by reloads / host events / other operator-side
        teardowns. These are surfaced separately so they do not get mistaken
        for planner malformed-proposal churn.
      * ``stale_fenced_by_surface`` (from journal rows) — critic fences whose
        associated trial later ended up bug-corrupted. These are hidden from
        the main "ruled out" list so poisoned context does not look like a live
        boundary.

    Observations only — a rejection means a lever did not clear the gate, never
    a ratified law.
    """

    def _label(atype: str, detail: str) -> str:
        base = _REJECT_TYPE_LABEL.get(atype, atype)
        return f"{base} · {detail}"[:80] if detail else base[:80]

    # E2 (2026-07-26): era-scope the fence list. Fences are never deleted at an
    # era boundary — but each row now carries the provenance it was minted at
    # (trial id + timestamp) and a "pre-<era>" tag when minted before the
    # current pareto_epoch_ts, so pre-E8 operational fences read as historical
    # context rather than current-era findings. The data model does not
    # distinguish durable engineering guardrails from era-scoped operational
    # fences, so this is a per-row label, not a partition.
    try:
        epoch_ts = float(state.get("pareto_epoch_ts") or 0.0) or None
    except (TypeError, ValueError):
        epoch_ts = None
    active_eras = state.get("active_instrument_eras")
    active_eras = active_eras if isinstance(active_eras, dict) else {}
    epoch_era_short = _era_short(active_eras.get("autopilot_speed")) or "epoch"
    pre_epoch_tag = f"pre-{epoch_era_short}"

    def _minted_ts(rec: dict[str, Any], trial_row: dict[str, Any] | None) -> float | None:
        ts = _row_ts({"timestamp": rec.get("recorded_at")})
        if ts:
            return ts
        if trial_row is not None:
            ts = _row_ts(trial_row)
            if ts:
                return ts
        return None

    def _era_tags(item: dict[str, Any], minted_ts: float | None) -> None:
        item["minted_ts"] = minted_ts
        pre = bool(minted_ts is not None and epoch_ts is not None and minted_ts < epoch_ts)
        item["pre_epoch"] = pre
        item["era_tag"] = pre_epoch_tag if pre else None

    def _gepa_caveat(item: dict[str, Any]) -> None:
        # E3: GEPA rows share task C's no-op-window caveat — a gepa_optimize
        # fence/churn count accumulated while the mutation path was a no-op
        # says nothing about GEPA's real search behaviour.
        if str(item.get("kind") or "") == "gepa_optimize":
            item["provenance_caveat"] = GEPA_NOOP_CAVEAT

    fenced: list[dict[str, Any]] = []
    stale_fenced_by_surface: list[dict[str, Any]] = []
    trial_rows: dict[int, dict[str, Any]] = {}
    if isinstance(journal_rows, list):
        for row in journal_rows:
            if isinstance(row, dict):
                try:
                    trial_rows[int(row.get("trial_id"))] = row
                except (TypeError, ValueError):
                    continue
    crs = state.get("critic_rejected_signatures")
    if isinstance(crs, dict):
        for rec in crs.values():
            if not isinstance(rec, dict):
                continue
            action = rec.get("action") if isinstance(rec.get("action"), dict) else {}
            atype = str(action.get("type") or "?")
            flags = action.get("flags")
            if isinstance(flags, dict) and flags:
                detail = ", ".join(f"{k}={v}" for k, v in list(flags.items())[:2])
            else:
                detail = str(action.get("file") or action.get("surface") or "")
            trial_id = rec.get("trial_id")
            try:
                trial_row = trial_rows.get(int(trial_id))
            except (TypeError, ValueError):
                trial_row = None
            minted_ts = _minted_ts(rec, trial_row)
            if trial_row and str(trial_row.get("bug_corrupted_by") or "").strip():
                stale_item = {
                    "label": _label(atype, detail),
                    "kind": atype,
                    "count": int(rec.get("count") or 1),
                }
                _gepa_caveat(stale_item)
                stale_fenced_by_surface.append(stale_item)
                continue
            item = {
                "label": _label(atype, detail),
                "kind": atype,
                "count": int(rec.get("count") or 1),
                "why": _short_reason(rec.get("reason")),
                "last_trial": trial_id,
                "minted_trial": trial_id,
                "minted_at": rec.get("recorded_at"),
            }
            _era_tags(item, minted_ts)
            _gepa_caveat(item)
            fenced.append(item)
    fenced.sort(key=lambda x: x["count"], reverse=True)
    stale_fenced_by_surface.sort(key=lambda x: (-x["count"], x["kind"]))

    invalid_by_surface: list[dict[str, Any]] = []
    corrupted_by_surface: list[dict[str, Any]] = []

    def _surface_name(row: dict[str, Any]) -> str:
        return str(
            row.get("action_type")
            or (row.get("action") or {}).get("type")
            or row.get("surface")
            or row.get("species")
            or "?"
        )

    if isinstance(journal_rows, list):
        malformed: dict[str, int] = {}
        corrupted: dict[str, int] = {}
        for row in journal_rows:
            if not isinstance(row, dict):
                continue
            atype = _surface_name(row)
            bug = str(row.get("bug_corrupted_by") or "").strip()
            outcome = str(row.get("outcome_status") or "").strip().lower()
            if bug:
                corrupted[atype] = corrupted.get(atype, 0) + 1
            elif outcome in {"invalid", "skipped"}:
                malformed[atype] = malformed.get(atype, 0) + 1
        invalid_by_surface = [
            {"label": _REJECT_TYPE_LABEL.get(a, a), "kind": a, "count": c}
            for a, c in sorted(
                malformed.items(), key=lambda kv: (-kv[1], kv[0])
            )
        ]
        corrupted_by_surface = [
            {"label": _REJECT_TYPE_LABEL.get(a, a), "kind": a, "count": c}
            for a, c in sorted(
                corrupted.items(), key=lambda kv: (-kv[1], kv[0])
            )
        ]
        for churn_item in (*invalid_by_surface, *corrupted_by_surface):
            _gepa_caveat(churn_item)
    else:
        inv = state.get("invalid_signature_counts")
        if isinstance(inv, dict):
            agg: dict[str, int] = {}
            for sig, cnt in inv.items():
                try:
                    meta = json.loads(sig)
                except (json.JSONDecodeError, TypeError):
                    meta = {}
                atype = str(meta.get("type") or meta.get("mutation") or "?")
                agg[atype] = agg.get(atype, 0) + int(cnt or 0)
            invalid_by_surface = [
                {"label": _REJECT_TYPE_LABEL.get(a, a), "kind": a, "count": c}
                for a, c in sorted(agg.items(), key=lambda kv: (-kv[1], kv[0]))
            ]
            for churn_item in invalid_by_surface:
                _gepa_caveat(churn_item)

    return {
        "fenced": fenced[:limit],
        "invalid_by_surface": invalid_by_surface[:limit],
        "corrupted_by_surface": corrupted_by_surface[:limit],
        "stale_fenced_by_surface": stale_fenced_by_surface[:limit],
    }


def _narrative(
    *,
    trial_counter: Any,
    levers: list[dict[str, Any]],
    best: dict[str, Any],
    ruled_out: list[dict[str, Any]],
    exploring: list[dict[str, Any]],
    ruled_out_exp: dict[str, Any],
    banner: dict[str, Any],
) -> str:
    """Deterministic prose assembled from the structured fields above."""
    parts: list[str] = []
    if levers:
        top = levers[0]
        rec = f" (→ {top['recommended']})" if top.get("recommended") else ""
        parts.append(
            f"The dominant quality lever is {top['lever']}{rec}, importance "
            f"{top['importance']:.2f}"
        )
        if len(levers) > 1:
            parts.append(f"{levers[1]['lever']} is second")
    else:
        parts.append("No NumericSwarm importance data is available yet")
    if best.get("available"):
        q = best["objective"].get("quality")
        label = best.get("status") or "incumbent"
        parts.append(
            f"the {label} best config is trial {best.get('trial_id')} "
            f"(quality {q})" if q is not None else f"an {label} best config exists"
        )
    parts.append(f"{len(ruled_out)} dead-ends remain fenced")
    fenced = (ruled_out_exp or {}).get("fenced") or []
    if fenced:
        top_ro = fenced[0]
        parts.append(
            f"the most-fenced dead-end is {top_ro['label']} "
            f"({top_ro['count']}× rejected)"
        )
    parts.append(f"{len(exploring)} hypotheses are queued")
    lead = f"At trial {trial_counter}: " if trial_counter is not None else ""
    holds = banner.get("holds") if isinstance(banner.get("holds"), dict) else {}
    if banner.get("decision_grade_possible"):
        tail = " Authority is on; kept configs are decision-grade."
    elif holds.get("any_hold_active"):
        # Never claim decision-grade while a fail-closed era hold is open.
        tail = (
            " Nothing is decision-grade — fail-closed era holds are active "
            "(quality baseline reseed / speed numeric rerun pending)."
        )
    else:
        tail = " Nothing is promoted — authority is off, so these are observations."
    return lead + "; ".join(parts) + "." + tail


# --------------------------------------------------------------------------- #
# Top-level
# --------------------------------------------------------------------------- #
def build_optimization_brief(
    *,
    state_path: Path | None = None,
    journal_paths: list[Path] | None = None,
    strategy_db: Path | None = None,
    digest_dir: Path | None = None,
    digest_text: str | None = None,
    alarm_window: int = 30,
) -> dict[str, Any]:
    """Assemble the full operator optimization brief (read-only)."""
    state_path = state_path or DEFAULT_STATE_PATH
    journal_paths = journal_paths or DEFAULT_JOURNAL_PATHS
    strategy_db = strategy_db or DEFAULT_STRATEGY_DB
    digest_dir = digest_dir or DEFAULT_DIGEST_DIR

    state = load_state(state_path)
    rows = load_journal_rows([Path(p) for p in journal_paths])
    exclude_before_ts = state.get("pareto_exclude_before_ts")
    try:
        exclude_before_ts = float(exclude_before_ts) if exclude_before_ts else None
    except (TypeError, ValueError):
        exclude_before_ts = None

    try:
        w6 = _w6_build_report(
            rows, alarm_window=alarm_window, exclude_before_ts=exclude_before_ts
        )
    except Exception:  # noqa: BLE001 — report stays read-only even if audit fails
        w6 = {}

    banner = authority_banner(state, w6)
    if digest_text is None:
        digest_text = latest_digest_text(digest_dir)
    levers = levers_from_digest(digest_text)
    best = best_config(rows, exclude_before_ts=exclude_before_ts)
    ruled_out, exploring = ruled_out_and_exploring(strategy_db)
    ruled_out_exp = ruled_out_experiments(state, journal_rows=rows)

    # decision_grade is a property of the whole brief right now: nothing can be
    # ratified unless authority is enabled. Tag every lever accordingly.
    for lever in levers:
        lever["decision_grade"] = banner["decision_grade_possible"]

    narrative = _narrative(
        trial_counter=state.get("trial_counter"),
        levers=levers,
        best=best,
        ruled_out=ruled_out,
        exploring=exploring,
        ruled_out_exp=ruled_out_exp,
        banner=banner,
    )

    return {
        "read_only": True,
        "checkpoint": {
            "trial_counter": state.get("trial_counter"),
            "era_exclude_before_ts": exclude_before_ts,
        },
        "authority": banner,
        "narrative": narrative,
        "best_config": best,
        "levers": levers,
        "ruled_out_experiments": ruled_out_exp,
        "ruled_out": ruled_out,
        "exploring": exploring,
        # E4 (2026-07-26): constitutionally, old-evidence hypotheses stay queued
        # (priors are valid); this note keeps the section honest about lineage.
        "exploring_note": "hypotheses may derive from prior-era evidence (valid as priors)",
        # Task C twin: the GEPA no-op provenance window, so brief consumers see
        # the same caveat window the gepa panel renders.
        "gepa_provenance_windows": GEPA_PROVENANCE_WINDOWS,
    }


if __name__ == "__main__":  # pragma: no cover — manual smoke
    import sys

    json.dump(build_optimization_brief(), sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")
