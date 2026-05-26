"""URE-1: routing decision-uncertainty shadow logging + offline ingest (intake-607 §5.2.5).

Shadow-only, flag-gated by `features().ure_uncertainty_shadow_log` (default off → zero behavior
change). Computes a first-pass, **UNCALIBRATED** uncertainty estimate from the routing
decision-meta and appends a JSONL record. Calibration (ECE/abstention) is the J10 inference task;
enforcement comes only after that gate passes (URE-1 audit #1: calibration precedes enforcement).

`emit_uncertainty_shadow()` is try/except-safe — it MUST NOT break the routing hot path.

The offline `ingest_uncertainty_shadow()` converts the JSONL into `approval_record` rows in the
shared trace schema (`src/trace/harness_schema`), so J10 can join uncertainty to outcomes.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SHADOW_PATH = _REPO_ROOT / "data" / "trace" / "uncertainty_shadow.jsonl"

# classifier confidence string → numeric (when the classifier fast-path set a label)
_CONF_MAP = {"high": 0.9, "medium": 0.6, "med": 0.6, "low": 0.3}

#: spread/margin below this is treated as "near-flat" → maximally uncertain on that signal
_MARGIN_SCALE = 0.1


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _conf_to_float(v) -> float | None:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    return _CONF_MAP.get(str(v).lower())


def compute_routing_uncertainty(meta: dict) -> dict:
    """First-pass uncertainty in [0,1] (higher = more uncertain) + per-signal components.

    UNCALIBRATED — the J10 calibration task learns weights / a threshold from logged components
    vs. realized "would-escalation-help" outcomes. Tolerates any subset of signals being absent.
    """
    q = [x for x in (meta.get("q_topk") or []) if x is not None]
    sel = [x for x in (meta.get("selection_score_topk") or []) if x is not None]
    n_alt = len(q)
    q_spread = (max(q) - min(q)) if n_alt >= 2 else 0.0
    top2_margin = (q[0] - q[1]) if n_alt >= 2 else None
    sel_margin = (sel[0] - sel[1]) if len(sel) >= 2 else None
    q_conf = meta.get("q_robust_confidence")
    clf = _conf_to_float(meta.get("classifier_confidence"))
    source = meta.get("decision_source") or ""

    components: dict[str, float] = {}
    signals: list[float] = []

    # Flat-Q pathology (DAR-1: 96% uniform Q): multiple alternatives but ~no spread → uncertain.
    if n_alt >= 2:
        flat = 1.0 - _clip(q_spread / _MARGIN_SCALE)
        components["flat_q"] = round(flat, 4)
        signals.append(flat)
    # Small top-2 Q margin → uncertain.
    if top2_margin is not None:
        m = 1.0 - _clip(top2_margin / _MARGIN_SCALE)
        components["q_top2_margin"] = round(m, 4)
        signals.append(m)
    # Low robust confidence → uncertain.
    if q_conf is not None:
        lc = 1.0 - _clip(float(q_conf))
        components["low_q_confidence"] = round(lc, 4)
        signals.append(lc)
    # Small selection-score margin → uncertain.
    if sel_margin is not None:
        sm = 1.0 - _clip(sel_margin / _MARGIN_SCALE)
        components["selection_margin"] = round(sm, 4)
        signals.append(sm)
    # Low classifier confidence (when the classifier fast-path produced one) → uncertain.
    if clf is not None:
        cc = 1.0 - _clip(clf)
        components["low_classifier_confidence"] = round(cc, 4)
        signals.append(cc)
    # Source prior: rules/abstain carry more inherent uncertainty than a confident learned pick.
    src_prior = {"rules": 0.6, "risk_abstain_escalate": 0.7, "learned_explore": 0.5}.get(source, 0.3)
    components["source_prior"] = src_prior
    signals.append(src_prior)

    score = round(sum(signals) / len(signals), 4) if signals else src_prior
    return {"score": score, "components": components, "n_alternatives": n_alt}


def emit_uncertainty_shadow(meta: dict, *, request_id: str | None = None, path=None) -> bool:
    """Append one shadow record. Never raises (returns False on any failure)."""
    try:
        u = compute_routing_uncertainty(meta)
        rec = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "request_id": request_id,
            "decision_source": meta.get("decision_source"),
            "chosen_action": meta.get("chosen_action"),
            "q_topk": meta.get("q_topk"),
            "selection_score_topk": meta.get("selection_score_topk"),
            "q_robust_confidence": meta.get("q_robust_confidence"),
            "classifier_confidence": meta.get("classifier_confidence"),
            "uncertainty_score": u["score"],
            "uncertainty_components": u["components"],
            "n_alternatives": u["n_alternatives"],
        }
        p = Path(path) if path else DEFAULT_SHADOW_PATH
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        return True
    except Exception:
        logger.debug("URE-1 uncertainty shadow log failed", exc_info=True)
        return False


def ingest_uncertainty_shadow(jsonl_path, conn) -> int:
    """Offline: convert shadow JSONL → `approval_record` rows. Returns count ingested."""
    from src.trace.harness_schema import ApprovalRecord, insert_approval_record

    p = Path(jsonl_path)
    if not p.exists():
        return 0
    n = 0
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        insert_approval_record(conn, ApprovalRecord(
            request_id=rec.get("request_id"),
            selected_role=rec.get("chosen_action"),
            quality_score=rec.get("q_robust_confidence"),
            uncertainty_score=rec.get("uncertainty_score"),
            uncertainty_components=rec.get("uncertainty_components"),
            trigger_reason="uncertainty_shadow",
            approval_boundary="shadow_log_only",
            actor="ure1_shadow",
            created_ts_utc=rec.get("ts") or datetime.now(timezone.utc).isoformat(),
        ))
        n += 1
    return n
