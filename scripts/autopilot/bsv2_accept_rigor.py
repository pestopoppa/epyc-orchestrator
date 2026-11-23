"""BSV-2 accept-rigor rules R1-R5 (AP-BSV2-RIGOR-GATE-CODE).

Pure, dependency-free helpers that extend the BSV-2 paired accept gate
(``actions._bsv2_accepts``, env ``AUTOPILOT_BSV2_ACCEPT_GATE``). They run only when the
sub-flag ``autopilot_bsv2_accept_rigor`` (env ``AUTOPILOT_BSV2_ACCEPT_RIGOR``, default off)
is also on. Rules (design: handoffs/active/autopilot-program.md, AP-BSV2-ACCEPT-RIGOR):

* R1 disjoint accept tasks: accept qids share nothing with the proposal batch's qids.
  Fail closed: an unknown proposal-qid set is a blocker, not a pass.
* R2 fresh parent re-run: the parent arm is evaluated in the same window; a cached or
  journal-replayed parent is a blocker.
* R3 sealed test split: sealed qids never appear in an accept eval or in any earlier
  round's exposed qids; the headline checkpoint is chosen by a training-side key only.
* R4 per-round decision record: status split (completed / solver_limit / infra_error /
  candidate_error, retry for infra_error only) and tokens per accepted gain (None, not
  zero, when nothing was accepted).
* R5 budget unit: rollouts or tokens, never wall-clock / trials / iterations.

State lives in ``ctx.state["bsv2_rigor"]`` (JSON-safe lists/dicts).
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

STATUSES = ("completed", "solver_limit", "infra_error", "candidate_error")
BUDGET_UNITS = ("rollouts", "tokens")
STATE_KEY = "bsv2_rigor"
_MAX_ROUNDS_KEPT = 200


def _qids(rows: Iterable[Mapping[str, Any]] | None) -> list[str]:
    return [str(r.get("qid")) for r in (rows or []) if r.get("qid")]


def row_status(row: Mapping[str, Any]) -> str:
    """Status of one question row; absent = completed, unknown value = candidate_error."""
    raw = row.get("status")
    if raw in (None, ""):
        return "completed"
    return str(raw) if str(raw) in STATUSES else "candidate_error"


def status_split(rows: Iterable[Mapping[str, Any]] | None) -> dict[str, int]:
    split = {s: 0 for s in STATUSES}
    for r in rows or []:
        split[row_status(r)] += 1
    return split


def retry_allowed(status: str) -> bool:
    """R4: only infra errors may be retried."""
    return status == "infra_error"


def _tokens(rows: Iterable[Mapping[str, Any]] | None) -> int:
    return sum(int(r.get("tokens_generated") or 0) for r in (rows or []))


def tokens_per_accepted_gain(tokens: int, gain_correct: float, accepted: bool) -> float | None:
    """Tokens spent per additional correct answer; None (undefined) unless accepted with gain > 0."""
    if not accepted or gain_correct <= 0:
        return None
    return tokens / gain_correct


def new_state() -> dict[str, Any]:
    return {
        "proposal_qids": None,  # None = unknown -> R1 fails closed
        "sealed_qids": [],
        "exposed_qids": [],
        "budget": {"unit": None, "limit": None, "spent": 0},
        "rounds": [],
    }


def ensure_state(state: dict[str, Any]) -> dict[str, Any]:
    cur = state.get(STATE_KEY)
    if not isinstance(cur, dict):
        cur = new_state()
        state[STATE_KEY] = cur
    for k, v in new_state().items():
        cur.setdefault(k, v)
    return cur


def evaluate_rigor(
    *,
    rigor_state: dict[str, Any],
    baseline_rows: list[Mapping[str, Any]] | None,
    candidate_rows: list[Mapping[str, Any]] | None,
    baseline_fresh: bool,
    paired_accept: bool,
    accuracy_delta: float | None,
) -> dict[str, Any]:
    """Apply R1-R5 to one round. Returns the decision record (never mutates state)."""
    blockers: list[str] = []
    b_ids, c_ids = _qids(baseline_rows), _qids(candidate_rows)
    accept_ids = sorted(set(b_ids) & set(c_ids))

    # R1
    proposal = rigor_state.get("proposal_qids")
    if proposal is None:
        r1 = {"ok": False, "reason": "proposal qid set unknown (fail closed)"}
    else:
        overlap = sorted(set(accept_ids) & {str(q) for q in proposal})
        r1 = {"ok": not overlap, "overlap": overlap[:20], "overlap_n": len(overlap)}
    if not r1["ok"]:
        blockers.append("R1 accept tasks not disjoint from proposal batch: " + str(r1.get("reason") or r1["overlap_n"]))

    # R2
    r2 = {"ok": bool(baseline_fresh)}
    if not baseline_fresh:
        blockers.append("R2 parent arm not a fresh same-window re-run")

    # R3
    sealed = {str(q) for q in rigor_state.get("sealed_qids") or []}
    exposed = {str(q) for q in rigor_state.get("exposed_qids") or []}
    leak = sorted(sealed & (set(accept_ids) | set(b_ids) | set(c_ids)))
    burned = sorted(sealed & exposed)
    r3 = {"ok": not leak and not burned, "sealed_in_accept_eval": leak[:20], "sealed_already_exposed": burned[:20]}
    if not r3["ok"]:
        blockers.append("R3 sealed qids exposed to the search")

    # R4 statuses (infra_error rows make the round non-decisive; retry those only)
    split = {"baseline": status_split(baseline_rows), "candidate": status_split(candidate_rows)}
    infra = split["baseline"]["infra_error"] + split["candidate"]["infra_error"]
    cand_err = split["candidate"]["candidate_error"] + split["baseline"]["candidate_error"]
    if infra:
        blockers.append(f"R4 {infra} infra_error rows: retry required (infra_error only)")
    if cand_err:
        blockers.append(f"R4 {cand_err} candidate_error rows")
    tokens = _tokens(baseline_rows) + _tokens(candidate_rows)

    # R5
    budget = dict(rigor_state.get("budget") or {})
    unit = budget.get("unit")
    if unit not in BUDGET_UNITS:
        r5 = {"ok": False, "unit": unit, "reason": f"budget unit must be one of {BUDGET_UNITS}"}
        blockers.append("R5 budget unit not rollouts/tokens")
        round_cost = None
    else:
        round_cost = (len(baseline_rows or []) + len(candidate_rows or [])) if unit == "rollouts" else tokens
        limit = budget.get("limit")
        spent = float(budget.get("spent") or 0)
        over = limit is not None and spent + round_cost > float(limit)
        r5 = {"ok": not over, "unit": unit, "limit": limit, "spent_before": spent, "round_cost": round_cost}
        if over:
            blockers.append("R5 budget exhausted")

    accept = bool(paired_accept) and not blockers
    n = len(accept_ids)
    gain_correct = (accuracy_delta or 0.0) * n
    return {
        "rules": {"R1": r1, "R2": r2, "R3": r3, "R4": {"ok": not infra and not cand_err, "status_split": split}, "R5": r5},
        "blockers": blockers,
        "accept": accept,
        "paired_accept": bool(paired_accept),
        "accept_qids": accept_ids,
        "tokens": tokens,
        "round_cost": round_cost,
        "budget_unit": unit,
        "gain_correct": gain_correct,
        "tokens_per_accepted_gain": tokens_per_accepted_gain(tokens, gain_correct, accept),
    }


def commit_round(rigor_state: dict[str, Any], record: Mapping[str, Any]) -> None:
    """Record the round: spend budget, mark accept qids exposed (R3), keep the decision record."""
    exposed = {str(q) for q in rigor_state.get("exposed_qids") or []}
    exposed.update(record.get("accept_qids") or [])
    rigor_state["exposed_qids"] = sorted(exposed)
    cost = record.get("round_cost")
    if cost is not None:
        rigor_state.setdefault("budget", {})["spent"] = float(rigor_state["budget"].get("spent") or 0) + cost
    rounds = list(rigor_state.get("rounds") or [])
    rounds.append({k: record[k] for k in ("accept", "blockers", "tokens", "budget_unit", "tokens_per_accepted_gain") if k in record})
    rigor_state["rounds"] = rounds[-_MAX_ROUNDS_KEPT:]


def select_headline_checkpoint(
    checkpoints: list[Mapping[str, Any]], key: str = "train_score"
) -> Mapping[str, Any] | None:
    """R3: pick the headline checkpoint by a training-side key; any test-side key is refused."""
    if "test" in key.lower() or "sealed" in key.lower():
        raise ValueError(f"headline key {key!r} is test-side; pick by a training-side rule")
    scored = [c for c in checkpoints if c.get(key) is not None]
    return max(scored, key=lambda c: c[key]) if scored else None
