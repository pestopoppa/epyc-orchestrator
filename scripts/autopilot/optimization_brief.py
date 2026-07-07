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
from src.autopilot_core.baseline_ledger import baseline_ledger_authority_enabled

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_PATH = REPO_ROOT / "orchestration" / "autopilot_state.json"
DEFAULT_JOURNAL_PATHS = [
    REPO_ROOT / "orchestration" / "autopilot_journal.jsonl",
    REPO_ROOT / "orchestration" / "autopilot_journal_1.jsonl",
]
DEFAULT_STRATEGY_DB = (
    REPO_ROOT / "orchestration" / "repl_memory" / "strategies" / "strategies.db"
)
# Autopilot digests are written into the epyc-root governance tree, not the
# orchestrator repo (matches the epyc-root paths dashboard.py already uses).
DEFAULT_DIGEST_DIR = Path("/mnt/raid0/llm/epyc-root/progress")

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
    and the W6 gaming alarm must be clear for findings to be decision-grade.
    """
    baseline = baseline_ledger_authority_enabled(state)
    if seq_verdict_live is None:
        seq_verdict_live = _autopilot_seq_verdict_live()
    sequential = bool(seq_verdict_live)
    gaming_alarm = bool(w6.get("gaming_alarm"))
    decision_grade_possible = baseline and sequential and not gaming_alarm
    return {
        "baseline_authority_enabled": baseline,
        "sequential_authority_enabled": sequential,
        "w6_gaming_alarm": gaming_alarm,
        "w6_clearance_clean_trials_required": w6.get(
            "gaming_alarm_clearance_clean_trials_required"
        ),
        "current_era_audited_trials": w6.get("trusted_audited_trial_count"),
        "decision_grade_possible": decision_grade_possible,
        "trust_note": (
            "Authority ENABLED (baseline + sequential, W6 clear) — kept configs are decision-grade."
            if decision_grade_possible
            else "Authority OFF — every finding below is an OBSERVATION, not "
            "decision-grade; no config can be promoted yet."
        ),
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
    state: dict[str, Any], *, limit: int = 6
) -> dict[str, Any]:
    """Surface *what was tried and rejected, and why* — the intuition the bare
    "N dead-ends fenced" count omits. The two state ledgers carry different
    signal, so they stay separate rather than being pooled into one count race:

      * ``fenced`` (from ``critic_rejected_signatures``) — the critic vetoed a
        specific proposal; each row carries a human ``reason`` and a repeat
        ``count`` (how many times the planner re-proposed it = how firmly the
        boundary is fenced). This is the reason-bearing "and why" list.
      * ``invalid_by_surface`` (from ``invalid_signature_counts``) — proposals
        that failed structural validation before execution, aggregated by
        experiment *type*: a churn summary of which surfaces the planner keeps
        drafting malformed actions against, not specific falsified ideas.

    Observations only — a rejection means a lever did not clear the gate, never
    a ratified law.
    """

    def _label(atype: str, detail: str) -> str:
        base = _REJECT_TYPE_LABEL.get(atype, atype)
        return f"{base} · {detail}"[:80] if detail else base[:80]

    fenced: list[dict[str, Any]] = []
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
            fenced.append(
                {
                    "label": _label(atype, detail),
                    "kind": atype,
                    "count": int(rec.get("count") or 1),
                    "why": _short_reason(rec.get("reason")),
                    "last_trial": rec.get("trial_id"),
                }
            )
    fenced.sort(key=lambda x: x["count"], reverse=True)

    invalid_by_surface: list[dict[str, Any]] = []
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
            for a, c in sorted(agg.items(), key=lambda kv: kv[1], reverse=True)
        ]

    return {
        "fenced": fenced[:limit],
        "invalid_by_surface": invalid_by_surface[:limit],
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
    tail = (
        " Nothing is promoted — authority is off, so these are observations."
        if not banner.get("decision_grade_possible")
        else " Authority is on; kept configs are decision-grade."
    )
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
    ruled_out_exp = ruled_out_experiments(state)

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
    }


if __name__ == "__main__":  # pragma: no cover — manual smoke
    import sys

    json.dump(build_optimization_brief(), sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")
