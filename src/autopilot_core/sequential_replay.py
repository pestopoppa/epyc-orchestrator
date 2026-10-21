"""Read-only journal replays for the sequential gate (SEQ-5, SEQ-6, SEQ-7, MHS-16).

Pure functions over already-loaded journal row dicts. Nothing here writes, changes a
gate, or re-defines what counts as a promotion; every function is a report. Callers
own the frozen-input digest (see ``scripts/autopilot/gate_replay_reports.py``).
"""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .journal_reconstruction import parse_journal_ts
from .measurement_guards import is_quality_admissible
from .planner_evidence import DEFAULT_EVIDENCE_CORE_ID
from .sequential_verdict import (
    DEFAULT_POLICY,
    SequentialPolicy,
    baseline_profile_from_trials,
    quality_trial_statistic,
    rebuild_candidate_view,
)

# Live constants mirrored from scripts/autopilot/autopilot.py (read there, not imported:
# importing autopilot.py runs its module-level environment reads).
BASELINE_PROFILE_LIMIT = 120
BASELINE_PROFILE_MIN_TRIALS = 3
PROMOTION_FINAL_CONFIRM_E = 100.0
PAIRED_Z_CRITICAL = 1.96
STOP_RULE_NONSIGNIFICANT_KEEP_STREAK = 3
BUCKET_RUN_WARN_LENGTH = 3
SEED_PROFILE_ROWS = 3


# --------------------------------------------------------------------------- helpers
def outcome_map(question_results: Any) -> dict[str, bool]:
    """Same normalization as autopilot._question_outcome_map (CJ-8 disposition filter)."""
    if not isinstance(question_results, list):
        return {}
    out: dict[str, bool] = {}
    for item in question_results:
        if not isinstance(item, Mapping):
            continue
        qid = str(item.get("qid") or item.get("question_id") or "").strip()
        if not qid or not is_quality_admissible(item.get("disposition")):
            continue
        out[qid] = bool(item.get("correct"))
    return out


def _details(row: Mapping[str, Any]) -> Mapping[str, Any]:
    ed = row.get("eval_details")
    return ed if isinstance(ed, Mapping) else {}


def _seq(row: Mapping[str, Any]) -> Mapping[str, Any]:
    s = row.get("seq")
    return s if isinstance(s, Mapping) else {}


def _tier(row: Mapping[str, Any]) -> int | None:
    try:
        return int(row.get("tier"))
    except (TypeError, ValueError):
        return None


def incumbent_representative(row: Mapping[str, Any]) -> bool:
    return (
        str(row.get("pareto_status") or "") == "frontier"
        or str(row.get("keep_revert_decision") or "") == "keep"
    )


def _usable(row: Mapping[str, Any], era_before_ts: float | None) -> bool:
    if row.get("bug_corrupted_by"):
        return False
    if str(row.get("outcome_status") or "ok") in {"invalid", "skipped"}:
        return False
    if era_before_ts is not None:
        ts = parse_journal_ts(row.get("timestamp"))
        if ts is None or ts < era_before_ts:
            return False
    return True


def _rolling_profile(
    rows: Sequence[Mapping[str, Any]],
    end: int,
    tier: int | None,
    era_before_ts: float | None,
    limit: int,
    incumbent_only: bool = True,
) -> tuple[dict[str, float], int]:
    """The live null: up to ``limit`` most recent eligible rows strictly before ``end``."""
    trials: list[dict[str, bool]] = []
    for idx in range(end - 1, -1, -1):
        row = rows[idx]
        if not _usable(row, era_before_ts) or _tier(row) != tier:
            continue
        if incumbent_only and not incumbent_representative(row):
            continue
        omap = outcome_map(_details(row).get("question_results"))
        if omap:
            trials.append(omap)
            if len(trials) >= limit:
                break
    if len(trials) < BASELINE_PROFILE_MIN_TRIALS:
        return {}, len(trials)
    return baseline_profile_from_trials(reversed(trials)), len(trials)


# --------------------------------------------------------------------------- SEQ-5
def promotion_attempt_ledger(
    rows: Iterable[Mapping[str, Any]],
    *,
    policy: SequentialPolicy = DEFAULT_POLICY,
    final_confirm_e: float = PROMOTION_FINAL_CONFIRM_E,
    era_before_ts: float | None = None,
) -> dict[str, Any]:
    """Lifetime promotion-attempt ledger folded from the journal (derived, not measured).

    ``promotion_attempts_opened`` is a LOWER BOUND: an attempt is visible only through a
    journaled ``seq`` row (a confirmed candidate row, or a fresh-eval row naming its
    source). An attempt interrupted before either row exists is invisible.
    """
    rows = [r for r in rows if isinstance(r, Mapping)]
    opened: set[tuple[str, Any]] = set()
    finalized: set[tuple[str, Any]] = set()
    confirmed_fps: set[str] = set()
    tested_fps: set[str] = set()
    for row in rows:
        seq = _seq(row)
        if not seq or str(seq.get("core_id") or DEFAULT_EVIDENCE_CORE_ID) != DEFAULT_EVIDENCE_CORE_ID:
            continue
        cand = str(seq.get("candidate") or "")
        if cand:
            tested_fps.add(cand)
            if seq.get("confirmed") is True or str(seq.get("state") or "") == "confirmed":
                confirmed_fps.add(cand)
                if not seq.get("baseline_promotion_fresh_eval"):
                    opened.add((cand, row.get("trial_id")))
        if seq.get("baseline_promotion_fresh_eval"):
            ref = seq.get("baseline_promotion_fresh_eval_for")
            ref = ref if isinstance(ref, Mapping) else {}
            key = (str(ref.get("candidate") or cand), ref.get("source_trial_id"))
            opened.add(key)
            if seq.get("baseline_promotion_finalized") is True:
                finalized.add(key)
    k_opened = len(opened)
    k_final = len(finalized)
    per_attempt_alpha = 1.0 / final_confirm_e if final_confirm_e > 0 else float("nan")
    r = len(confirmed_fps)
    expected_false_confirms = len(confirmed_fps) * policy.alpha
    # the live era fence: charged set after the boundary (rows before it are dropped)
    post_fence = None
    if era_before_ts is not None:
        post = [
            row for row in rows
            if (ts := parse_journal_ts(row.get("timestamp"))) is not None and ts >= era_before_ts
        ]
        post_fence = {
            "quality_exclude_before_ts": era_before_ts,
            "rows_at_or_after_boundary": len(post),
            "fingerprints_with_seq_after_boundary": len(
                {str(_seq(x).get("candidate")) for x in post if _seq(x).get("candidate")}
            ),
        }
    return {
        "promotion_attempts_opened": k_opened,
        "promotion_attempts_finalized": k_final,
        "promotion_attempts_failed_or_interrupted": k_opened - k_final,
        "promotion_attempts_opened_is_lower_bound": True,
        "promotion_alpha_per_attempt": round(per_attempt_alpha, 6),
        "lifetime_false_promotion_bound": round(per_attempt_alpha * k_opened, 6),
        "promotion_alpha_per_attempt_for_declared_delta_life": (
            "delta_life / K_opened; undefined until the operator declares delta_life "
            f"(K_opened={k_opened})"
        ),
        "fingerprints_confirmed": r,
        "est_false_confirm_fraction": round(expected_false_confirms / max(r, 1), 6),
        "est_false_confirm_fraction_note": (
            "expected_false_confirms / max(R, 1): conservative all-null heuristic; "
            "0.0 here because R == 0 or no confirmation was ever journaled"
            if r == 0
            else "expected_false_confirms / max(R, 1): conservative all-null heuristic"
        ),
        "cumulative_rate_drift_bound": round(k_final * policy.rate_noninferiority_margin, 6),
        "cumulative_rate_drift_bound_basis": (
            "finalized promotions x rate_noninferiority_margin "
            f"({policy.rate_noninferiority_margin})"
        ),
        "rate_reference_is_latest_only": True,
        "rate_reference_note": (
            "rate and quality nulls are rolling medians/profiles over the "
            f"{BASELINE_PROFILE_LIMIT} most recent incumbent-representative trials "
            "(autopilot.py _seq_inputs_for_trial); there is no fixed reference"
        ),
        "instrument_era_boundary_effect": (
            "the era fence (quality_exclude_before_ts) drops every row before the boundary "
            "from the charged fingerprint set, so charged alpha RESETS to zero at an "
            "instrument-era boundary; this ledger counts across eras"
        ),
        "instrument_era_fence": post_fence,
        "tested_fingerprints_all_eras": len(tested_fps),
        "reopen_triggers": {
            "DP-A_option_b": "K_opened above about 5 per quarter",
            "DP-C_option_b": (
                "fingerprint confirmations still binding after the era and rate-axis "
                "faults are fixed"
            ),
        },
        "claim_status": "descriptive census of an ingested journal; needs a ClaimTuple before it gates a promotion",
    }


# --------------------------------------------------------------------------- SEQ-6
def _match_rate(rows, groups, era_before_ts, limit, incumbent_only) -> tuple[int, int]:
    ok = tot = 0
    for (cand, tier), idxs in groups.items():
        for idx in idxs:
            omap = outcome_map(_details(rows[idx]).get("question_results"))
            if not omap:
                continue
            prof, _ = _rolling_profile(rows, idx, tier, era_before_ts, limit, incumbent_only)
            if not prof:
                continue
            tot += 1
            if abs(quality_trial_statistic(omap, prof).z - float(_seq(rows[idx])["z"])) <= 1e-4:
                ok += 1
    return ok, tot


def frozen_null_replay(
    rows: Iterable[Mapping[str, Any]],
    *,
    policy: SequentialPolicy = DEFAULT_POLICY,
    era_before_ts: float | None = None,
    profile_limit: int = BASELINE_PROFILE_LIMIT,
    incumbent_only: bool | None = None,
) -> dict[str, Any]:
    """Recompute each candidate's quality z/E under ONE frozen null (as of its first trial).

    Group = (candidate, tier), the live prior-observation scope. The LIVE rolling null is
    first rebuilt at every trial and compared with the journaled z (harness validation).
    ``incumbent_only=None`` picks, by journaled-z match count, between the current-code
    rule (only incumbent-representative rows feed the null) and the as-recorded rule (all
    usable same-tier rows): old rows were written before the B4 filter existed. Both match
    rates are reported. Counterfactual comparison is frozen vs the REBUILT rolling null
    (same harness, so the only difference is freezing) and vs the journaled E.
    """
    rows = [r for r in rows if isinstance(r, Mapping)]
    groups: dict[tuple[str, int | None], list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        seq = _seq(row)
        if not seq or str(seq.get("core_id") or DEFAULT_EVIDENCE_CORE_ID) != DEFAULT_EVIDENCE_CORE_ID:
            continue
        if not seq.get("candidate") or "z" not in seq:
            continue
        if not _usable(row, era_before_ts):
            continue
        groups[(str(seq["candidate"]), _tier(row))].append(idx)

    rates = {
        flag: _match_rate(rows, groups, era_before_ts, profile_limit, flag) for flag in (True, False)
    }
    if incumbent_only is None:
        incumbent_only = rates[True][0] >= rates[False][0]

    not_replayable: Counter[str] = Counter()
    out_groups: list[dict[str, Any]] = []
    for (cand, tier), idxs in sorted(groups.items(), key=lambda kv: kv[1][0]):
        frozen, n_base = _rolling_profile(rows, idxs[0], tier, era_before_ts, profile_limit, incumbent_only)
        if not frozen:
            not_replayable["fewer_than_3_baseline_rows_at_first_trial"] += 1
            continue
        live_obs: list[tuple[int | None, float]] = []
        roll_obs: list[tuple[int | None, float]] = []
        frozen_obs: list[tuple[int | None, float]] = []
        for idx in idxs:
            row = rows[idx]
            omap = outcome_map(_details(row).get("question_results"))
            if not omap:
                not_replayable["row_without_question_results"] += 1
                continue
            try:
                live_z = float(_seq(row)["z"])
            except (TypeError, ValueError):
                not_replayable["row_with_nonnumeric_z"] += 1
                continue
            rolling, _ = _rolling_profile(rows, idx, tier, era_before_ts, profile_limit, incumbent_only)
            if not rolling:
                not_replayable["row_with_unrebuildable_rolling_null"] += 1
                continue
            tid = row.get("trial_id")
            live_obs.append((tid, live_z))
            roll_obs.append((tid, quality_trial_statistic(omap, rolling).z))
            frozen_obs.append((tid, quality_trial_statistic(omap, frozen).z))
        if not live_obs:
            not_replayable["no_row_with_vector_and_z"] += 1
            continue

        def fold(obs):
            v = rebuild_candidate_view(candidate=cand, core_id=DEFAULT_EVIDENCE_CORE_ID,
                                       observations=obs, policy=policy)
            return v, [u.state for u in v.quality_updates], v.quality_state.wealth

        lv, l_states, l_e = fold(live_obs)
        rv, r_states, r_e = fold(roll_obs)
        fv, f_states, f_e = fold(frozen_obs)
        n = min(len(l_states), len(f_states))
        out_groups.append({
            "candidate": cand, "tier": tier, "k": len(live_obs),
            "baseline_rows_in_frozen_null": n_base,
            "journal_final_state": lv.state, "rolling_rebuilt_final_state": rv.state,
            "frozen_final_state": fv.state,
            "final_verdict_changed_vs_rolling": rv.state != fv.state,
            "final_verdict_changed_vs_journal": lv.state != fv.state,
            "steps_changed_vs_rolling": sum(1 for a, b in zip(r_states[:n], f_states[:n]) if a != b),
            "steps_changed_vs_journal": sum(1 for a, b in zip(l_states[:n], f_states[:n]) if a != b),
            "journal_E": round(l_e, 6), "rolling_rebuilt_E": round(r_e, 6), "frozen_E": round(f_e, 6),
            "log10_E_frozen_over_rolling": (
                round(math.log10(f_e / r_e), 4) if r_e > 0 and f_e > 0 else None),
            "frozen_crosses_confirm_e": f_e >= policy.confirm_e,
            "rolling_crosses_confirm_e": r_e >= policy.confirm_e,
            "journal_crosses_confirm_e": l_e >= policy.confirm_e,
        })
    ratios = [g["log10_E_frozen_over_rolling"] for g in out_groups
              if g["log10_E_frozen_over_rolling"] is not None]
    return {
        "denominator_groups_with_seq_vector_rows": len(groups),
        "replayed_groups": len(out_groups),
        "not_replayable": dict(not_replayable),
        "harness_validation": {
            "rule_used": "incumbent_only" if incumbent_only else "all_usable_same_tier_rows",
            "z_match_incumbent_only": {"matches": rates[True][0], "of": rates[True][1]},
            "z_match_all_usable_same_tier": {"matches": rates[False][0], "of": rates[False][1]},
            "note": "journaled z is reproduced only to the extent matches/of; the rest is harness infidelity, not a gate fact",
        },
        "groups_final_verdict_changed_vs_rolling": sum(g["final_verdict_changed_vs_rolling"] for g in out_groups),
        "groups_final_verdict_changed_vs_journal": sum(g["final_verdict_changed_vs_journal"] for g in out_groups),
        "groups_any_step_verdict_changed_vs_rolling": sum(
            1 for g in out_groups if g["steps_changed_vs_rolling"] > 0),
        "total_steps_changed_vs_rolling": sum(g["steps_changed_vs_rolling"] for g in out_groups),
        "total_steps_changed_vs_journal": sum(g["steps_changed_vs_journal"] for g in out_groups),
        "total_steps": sum(g["k"] for g in out_groups),
        "groups_E_changed_over_1pct_vs_rolling": sum(
            1 for g in out_groups
            if abs(g["frozen_E"] - g["rolling_rebuilt_E"]) > 0.01 * max(g["rolling_rebuilt_E"], 1e-12)),
        "groups_crossing_confirm_e": {
            "journal": sum(g["journal_crosses_confirm_e"] for g in out_groups),
            "rolling_rebuilt": sum(g["rolling_crosses_confirm_e"] for g in out_groups),
            "frozen": sum(g["frozen_crosses_confirm_e"] for g in out_groups),
        },
        "median_abs_log10_E_ratio_vs_rolling": (
            round(statistics.median(abs(x) for x in ratios), 4) if ratios else None),
        "max_abs_log10_E_ratio_vs_rolling": round(max((abs(x) for x in ratios), default=0.0), 4),
        "groups": out_groups,
        "decision_rule": "report only; changing the live null is the operator's (human-amendment boundary)",
    }


# --------------------------------------------------------------------------- SEQ-7
def paired_keep_replay(
    rows: Iterable[Mapping[str, Any]],
    *,
    z_critical: float = PAIRED_Z_CRITICAL,
    streak: int = STOP_RULE_NONSIGNIFICANT_KEEP_STREAK,
) -> dict[str, Any]:
    """Paired discordant-pair (McNemar form) statistic of each keep row vs its predecessor.

    Predecessor incumbent = the most recent earlier keep row of the same tier that carries a
    question vector. z = (c - b) / sqrt(b + c), c = questions the candidate gained, b = lost,
    over shared qids; z < z_critical means the accept was not paired-significant (false elite).
    Unit caveat: an AutoPilot trial is ONE pass over about 37 scoreable questions, not three
    attempts per question, so the McNemar form is the available paired unit.
    """
    rows = [r for r in rows if isinstance(r, Mapping)]
    keeps = [(i, r) for i, r in enumerate(rows) if r.get("keep_revert_decision") == "keep"]
    no_vector: list[Any] = []
    no_pred = 0
    results: list[dict[str, Any]] = []
    last_by_tier: dict[int | None, tuple[Any, dict[str, bool]]] = {}
    for idx, row in keeps:
        omap = outcome_map(_details(row).get("question_results"))
        if not omap:
            no_vector.append(row.get("trial_id"))
            continue
        tier = _tier(row)
        pred = last_by_tier.get(tier)
        last_by_tier[tier] = (row.get("trial_id"), omap)
        if pred is None:
            no_pred += 1
            continue
        shared = set(omap) & set(pred[1])
        c = sum(1 for q in shared if omap[q] and not pred[1][q])
        b = sum(1 for q in shared if not omap[q] and pred[1][q])
        n = b + c
        z = (c - b) / math.sqrt(n) if n else 0.0
        results.append({
            "trial_id": row.get("trial_id"), "pred_trial_id": pred[0], "tier": tier,
            "action_type": row.get("action_type"), "shared_qids": len(shared),
            "gained": c, "lost": b, "paired_z": round(z, 4), "significant": z >= z_critical,
        })
    false_elite = [r for r in results if not r["significant"]]
    # stop rule: first `streak` consecutive non-significant keeps within a tier ends the stream
    saved: dict[str, Any] = {}
    for tier in sorted({r["tier"] for r in results}, key=lambda t: (t is None, t)):
        run = 0
        trigger = None
        for r in results:
            if r["tier"] != tier:
                continue
            run = run + 1 if not r["significant"] else 0
            if run >= streak:
                trigger = r["trial_id"]
                break
        if trigger is not None:
            later = [x for x in rows if isinstance(x.get("trial_id"), int)
                     and _tier(x) == tier and x["trial_id"] > trigger]
            saved[str(tier)] = {
                "trigger_trial_id": trigger,
                "later_trials_same_tier": len(later),
                "later_keep_rows_same_tier": sum(
                    1 for x in later if x.get("keep_revert_decision") == "keep"),
            }
    return {
        "keep_rows_total": len(keeps),
        "keep_rows_without_question_results": len(no_vector),
        "keep_rows_without_predecessor_vector": no_pred,
        "keep_rows_with_paired_statistic": len(results),
        "false_elite_count": len(false_elite),
        "false_elite_share": round(len(false_elite) / len(results), 4) if results else None,
        "z_critical": z_critical,
        "by_action_type": {
            at: {
                "n": sum(1 for r in results if r["action_type"] == at),
                "false_elite": sum(1 for r in false_elite if r["action_type"] == at),
            }
            for at in sorted({str(r["action_type"]) for r in results})
        },
        "stop_rule": f"stop a tier's stream after {streak} consecutive non-significant keeps",
        "rounds_saved_by_stop_rule": saved,
        "rows": results,
        "unit_caveat": (
            "an AutoPilot trial is one pass over about 37 scoreable questions (P-QUAL-T1), not "
            "three attempts per question; the published screen cannot be copied verbatim and "
            "the available paired unit is the McNemar form"
        ),
        "decision_rule": "report only; feeds HIL-2 item 11; changes no gate",
    }


# --------------------------------------------------------------------------- MHS-16
def _bucket(row: Mapping[str, Any]) -> tuple[str, str]:
    cfg = row.get("config_snapshot")
    cfg = cfg if isinstance(cfg, Mapping) else {}
    mtype = str(cfg.get("mutation") or cfg.get("mutation_type") or cfg.get("type") or row.get("action_type") or "")
    target = str(cfg.get("file") or cfg.get("target") or cfg.get("target_file") or "")
    return mtype, target


def cumulative_drift_report(
    rows: Iterable[Mapping[str, Any]],
    *,
    seed_rows: int = SEED_PROFILE_ROWS,
    run_warn: int = BUCKET_RUN_WARN_LENGTH,
) -> dict[str, Any]:
    """Drift of accepted ships against the SEED baseline (first ``seed_rows`` keep vectors/tier)."""
    rows = [r for r in rows if isinstance(r, Mapping)]
    keeps = [r for r in rows if r.get("keep_revert_decision") == "keep"]
    by_tier: dict[int | None, list[Mapping[str, Any]]] = defaultdict(list)
    for r in keeps:
        by_tier[_tier(r)].append(r)
    tiers_out: dict[str, Any] = {}
    no_vec_total = 0
    for tier, krs in sorted(by_tier.items(), key=lambda kv: (kv[0] is None, kv[0])):
        vec = [(r, outcome_map(_details(r).get("question_results"))) for r in krs]
        no_vec = [r.get("trial_id") for r, m in vec if not m]
        no_vec_total += len(no_vec)
        with_vec = [(r, m) for r, m in vec if m]
        seed = with_vec[:seed_rows]
        seed_profile = baseline_profile_from_trials([m for _, m in seed]) if seed else {}
        seed_ids = {r.get("trial_id") for r, _ in seed}
        seed_suites: dict[str, list[float]] = defaultdict(list)
        for r, _ in seed:
            for s, v in (_details(r).get("per_suite_quality") or {}).items():
                if isinstance(v, (int, float)):
                    seed_suites[s].append(float(v))
        seed_suite_mean = {s: sum(v) / len(v) for s, v in seed_suites.items()}
        flips_per_row: list[dict[str, Any]] = []
        cum_flipped: set[str] = set()
        for r, m in with_vec:
            if r.get("trial_id") in seed_ids:
                continue
            flipped = sorted(q for q, ok in m.items()
                             if not ok and seed_profile.get(q, 0.0) >= 0.5)
            cum_flipped.update(flipped)
            flips_per_row.append({"trial_id": r.get("trial_id"), "passed_to_failed": len(flipped),
                                  "shared_with_seed": sum(1 for q in m if q in seed_profile)})
        last_suites = {}
        for r, _ in reversed(with_vec):
            ps = _details(r).get("per_suite_quality")
            if isinstance(ps, Mapping) and ps:
                last_suites = {s: float(v) for s, v in ps.items() if isinstance(v, (int, float))}
                break
        cum_delta = {s: round(last_suites[s] - seed_suite_mean[s], 4)
                     for s in sorted(last_suites) if s in seed_suite_mean}
        runs: list[dict[str, Any]] = []
        cur: list[Mapping[str, Any]] = []
        cur_b: tuple[str, str] | None = None
        for r in krs:
            b = _bucket(r)
            if b == cur_b:
                cur.append(r)
            else:
                if cur:
                    runs.append({"bucket": list(cur_b), "length": len(cur),
                                 "first_trial": cur[0].get("trial_id"), "last_trial": cur[-1].get("trial_id")})
                cur, cur_b = [r], b
        if cur:
            runs.append({"bucket": list(cur_b), "length": len(cur),
                         "first_trial": cur[0].get("trial_id"), "last_trial": cur[-1].get("trial_id")})
        tiers_out[str(tier)] = {
            "keep_rows": len(krs), "keep_rows_without_question_results": len(no_vec),
            "seed_trial_ids": sorted(i for i in seed_ids if i is not None),
            "seed_questions": len(seed_profile),
            "rows_compared": len(flips_per_row),
            "rows_with_any_flip": sum(1 for f in flips_per_row if f["passed_to_failed"] > 0),
            "max_flips_in_one_row": max((f["passed_to_failed"] for f in flips_per_row), default=0),
            "distinct_questions_ever_flipped": len(cum_flipped),
            "cumulative_per_suite_delta_last_vs_seed": cum_delta,
            "longest_same_bucket_run": max((x["length"] for x in runs), default=0),
            "same_bucket_runs_at_or_above_warn": [x for x in runs if x["length"] >= run_warn],
            "same_bucket_concentration_warn": any(x["length"] >= run_warn for x in runs),
            "flips_per_row": flips_per_row,
        }
    return {
        "keep_rows_total": len(keeps),
        "keep_rows_without_question_results": no_vec_total,
        "baseline_basis": f"seed = first {seed_rows} vector-bearing keep rows per tier (no per-question seed profile is journaled)",
        "bucket_run_warn_length": run_warn,
        "tiers": tiers_out,
        "decision_rule": "report only; warn-only; no gate change (what counts as a ship is human-amendment territory)",
    }
