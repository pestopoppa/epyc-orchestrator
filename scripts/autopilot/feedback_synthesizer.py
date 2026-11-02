#!/usr/bin/env python3
"""P17.BT-5a retrospective feedback synthesizer and its offline backtest.

Consumes logged ``(config, score)`` pairs from the AutoPilot journal (every rotated
shard, supersessions folded, the shared STM trust filter applied) and emits a
feedback hint per numeric parameter for the proposal step:

* **directional**: the hint determines the SIGN of the change ("increase X" /
  "decrease X"), per the intake-1144 admission test;
* **non-directional**: the parameter is named as consequential but the sign is
  withheld because it is not stable under resampling (narrowing path);
* **none**: neither, so the existing scalar path applies.

Nothing here is wired into the live loop. ``backtest`` is the offline bench
against the no-feedback baseline that the box requires before any wiring: a
chronological walk-forward in which every hint is fitted on strictly earlier
trials only, so no future information leaks into a hint.

Score. Quality is not comparable across eras (eval-set and stack changes move
the level by more than any parameter does), so each trial's score is a local
residual: quality minus the median quality of the ``baseline_window`` preceding
learning-eligible trials at the same tier (any action type). Speed is reported
the same way as a secondary objective.

Bench. For each hinted trial, the proposer's move on a hinted parameter is the
sign of (trial value - median of that parameter's history values). A move is
concordant when it has the hint's sign. Success is a strictly positive residual.

* no-feedback baseline = success rate of all hinted trials (the historical
  proposer did not see the hints);
* follow arm = trials whose concordant moves outnumber discordant ones;
* uplift = follow rate - baseline rate;
* scrambled-feedback control = the same uplift with every hint sign replaced by
  a seeded coin flip (intake-1139 found random feedback can score below none),
  giving a one-sided permutation p-value.

Predeclared verdict (constants below, fixed before the run):
INCONCLUSIVE_INSUFFICIENT_COVERAGE when hinted < MIN_HINTED_TRIALS or either arm
< MIN_ARM_TRIALS; SIGNAL when uplift >= MIN_UPLIFT and p <= MAX_PERM_P; else
NO_SIGNAL. Stop rule: one planned look over the whole fixed retrospective corpus;
no interim looks and no re-run with altered thresholds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
ORCH_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ORCH_ROOT))

from experiment_journal import ExperimentJournal, is_learning_eligible  # noqa: E402
from journal_shards import journal_shards  # noqa: E402

SCHEMA = "epyc.autopilot.feedback_synthesizer_backtest.v1"

# Synthesizer parameters.
MIN_HISTORY = 6  # minimum (value, residual) points on a parameter before any hint
MIN_ABS_RHO = 0.30  # minimum |Spearman rho| for a parameter to be named at all
MIN_SIGN_STABILITY = 0.80  # bootstrap sign agreement for the hint to carry a sign
BOOTSTRAP_REPS = 200
BASELINE_WINDOW = 10
MIN_BASELINE_POINTS = 3

# Predeclared backtest verdict thresholds.
MIN_HINTED_TRIALS = 20
MIN_ARM_TRIALS = 8
MIN_UPLIFT = 0.10
MAX_PERM_P = 0.05
PERMUTATION_REPS = 2000
SEED = 20261102


@dataclass(frozen=True)
class ConfigScore:
    trial_id: int
    timestamp: str
    tier: int
    surface: str
    params: dict[str, float]
    quality: float
    speed: float
    residual: dict[str, float | None] = field(default_factory=dict)


@dataclass(frozen=True)
class Hint:
    param: str
    kind: str  # "directional" | "non_directional"
    sign: int  # +1 increase, -1 decrease, 0 withheld
    rho: float
    n: int
    sign_stability: float
    center: float

    def text(self) -> str:
        stats = f"(rho={self.rho:+.2f}, n={self.n}, sign stability {self.sign_stability:.2f})"
        if self.kind == "directional":
            verb = "increase" if self.sign > 0 else "decrease"
            return f"{verb} {self.param} relative to {self.center:g} {stats}"
        return f"{self.param} matters; direction unresolved {stats}"


def param_key(surface: str | None, key: str) -> str:
    """Normalise ``semantic_k`` under surface ``memrl_retrieval`` to its dotted key."""
    return key if "." in key else f"{surface or 'unknown'}.{key}"


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _local_residuals(entries: Sequence[Any], window: int) -> dict[int, dict[str, float | None]]:
    """Residual of quality/speed vs the median of the preceding same-tier eligible trials."""
    out: dict[int, dict[str, float | None]] = {}
    history: dict[int, list[Any]] = {}
    for entry in entries:
        if not is_learning_eligible(entry) or not (_numeric(entry.quality) or 0.0) > 0.0:
            continue
        prior = history.setdefault(int(entry.tier), [])
        recent = prior[-window:]
        res: dict[str, float | None] = {}
        for objective in ("quality", "speed"):
            values = [float(getattr(e, objective)) for e in recent
                      if _numeric(getattr(e, objective)) is not None]
            current = _numeric(getattr(entry, objective))
            res[objective] = (current - statistics.median(values)
                              if current is not None and len(values) >= MIN_BASELINE_POINTS else None)
        out[int(entry.trial_id)] = res
        prior.append(entry)
    return out


def extract_pairs(entries: Sequence[Any], *, window: int = BASELINE_WINDOW) -> list[ConfigScore]:
    """Learning-eligible numeric trials with non-empty numeric params, in trial order."""
    residuals = _local_residuals(entries, window)
    pairs: list[ConfigScore] = []
    for entry in sorted(entries, key=lambda e: int(e.trial_id)):
        if entry.action_type != "numeric_trial" or not is_learning_eligible(entry):
            continue
        snapshot = entry.config_snapshot if isinstance(entry.config_snapshot, dict) else {}
        raw = snapshot.get("params")
        if not isinstance(raw, dict):
            continue
        params = {param_key(snapshot.get("surface"), str(k)): v for k, v in raw.items()}
        params = {k: float(v) for k, v in params.items() if _numeric(v) is not None}
        if not params or int(entry.trial_id) not in residuals:
            continue
        surface = snapshot.get("surface") or next(iter(params)).split(".", 1)[0]
        pairs.append(ConfigScore(
            trial_id=int(entry.trial_id), timestamp=str(entry.timestamp), tier=int(entry.tier),
            surface=str(surface), params=params, quality=float(entry.quality),
            speed=float(entry.speed), residual=residuals[int(entry.trial_id)],
        ))
    return pairs


def _ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        mean_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = mean_rank
        i = j + 1
    return ranks


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Spearman rho with average ranks; 0.0 when either side is constant."""
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    if sxx == 0 or syy == 0:
        return 0.0
    return sxy / math.sqrt(sxx * syy)


def synthesize_hints(history: Sequence[ConfigScore], *, objective: str = "quality",
                     seed: int = SEED, reps: int = BOOTSTRAP_REPS) -> dict[str, Hint]:
    """Fit hints on ``history`` only. Deterministic for a given seed."""
    points: dict[str, list[tuple[float, float]]] = {}
    for pair in history:
        residual = pair.residual.get(objective)
        if residual is None:
            continue
        for key, value in pair.params.items():
            points.setdefault(key, []).append((value, residual))
    hints: dict[str, Hint] = {}
    for key in sorted(points):
        pts = points[key]
        if len(pts) < MIN_HISTORY:
            continue
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        rho = spearman(xs, ys)
        if abs(rho) < MIN_ABS_RHO:
            continue
        rng = random.Random(f"{seed}:{objective}:{key}:{len(pts)}")
        agree = 0
        for _ in range(reps):
            sample = [pts[rng.randrange(len(pts))] for _ in pts]
            r = spearman([p[0] for p in sample], [p[1] for p in sample])
            agree += (r > 0) == (rho > 0) and r != 0
        stability = agree / reps
        directional = stability >= MIN_SIGN_STABILITY
        hints[key] = Hint(
            param=key, kind="directional" if directional else "non_directional",
            sign=(1 if rho > 0 else -1) if directional else 0, rho=round(rho, 6), n=len(pts),
            sign_stability=round(stability, 4), center=statistics.median(xs),
        )
    return hints


def render_hints(hints: dict[str, Hint]) -> str:
    """Proposal-step text block. Empty hints mean the existing scalar path applies."""
    if not hints:
        return "(no feedback hint: scalar path)"
    return "\n".join(f"- {hints[k].text()}" for k in sorted(hints))


def _trial_units(pair: ConfigScore, hints: dict[str, Hint]) -> list[tuple[str, int, int]]:
    """(param, move sign, hint sign) for each directional hint the trial's config touches."""
    units = []
    for key, value in sorted(pair.params.items()):
        hint = hints.get(key)
        if hint is None or hint.kind != "directional":
            continue
        move = (value > hint.center) - (value < hint.center)
        if move:
            units.append((key, move, hint.sign))
    return units


def _arm_stats(trials: list[dict[str, Any]], signs: list[list[int]] | None = None) -> dict[str, Any]:
    follow = oppose = 0
    follow_success = oppose_success = 0
    total_success = sum(t["success"] for t in trials)
    for idx, trial in enumerate(trials):
        hint_signs = signs[idx] if signs is not None else [u[2] for u in trial["units"]]
        score = sum(1 if u[1] == s else -1 for u, s in zip(trial["units"], hint_signs))
        if score > 0:
            follow += 1
            follow_success += trial["success"]
        elif score < 0:
            oppose += 1
            oppose_success += trial["success"]
    n = len(trials)
    base = total_success / n if n else None
    f_rate = follow_success / follow if follow else None
    o_rate = oppose_success / oppose if oppose else None
    return {
        "hinted_trials": n, "baseline_success_rate": base,
        "follow_trials": follow, "follow_success_rate": f_rate,
        "oppose_trials": oppose, "oppose_success_rate": o_rate,
        "uplift_follow_vs_baseline": (f_rate - base) if f_rate is not None and base is not None else None,
    }


def backtest(pairs: Sequence[ConfigScore], *, objective: str = "quality", seed: int = SEED,
             perm_reps: int = PERMUTATION_REPS) -> dict[str, Any]:
    """Walk-forward bench of synthesized hints against the no-feedback baseline."""
    evaluable = [p for p in pairs if p.residual.get(objective) is not None]
    trials: list[dict[str, Any]] = []
    hint_kinds = {"directional": 0, "non_directional": 0}
    per_trial: list[dict[str, Any]] = []
    for idx, pair in enumerate(evaluable):
        history = evaluable[:idx]
        hints = synthesize_hints(history, objective=objective, seed=seed)
        for hint in hints.values():
            hint_kinds[hint.kind] += 1
        units = _trial_units(pair, hints)
        residual = float(pair.residual[objective])
        record = {"trial_id": pair.trial_id, "surface": pair.surface, "residual": residual,
                  "hints_available": len(hints), "units": units, "success": int(residual > 0)}
        per_trial.append({k: v for k, v in record.items()})
        if units:
            trials.append(record)
    observed = _arm_stats(trials)
    rng = random.Random(f"{seed}:{objective}:scramble")
    null: list[float] = []
    for _ in range(perm_reps):
        scrambled = [[rng.choice((-1, 1)) for _ in t["units"]] for t in trials]
        stats = _arm_stats(trials, scrambled)
        if stats["uplift_follow_vs_baseline"] is not None:
            null.append(stats["uplift_follow_vs_baseline"])
    uplift = observed["uplift_follow_vs_baseline"]
    p_value = ((1 + sum(1 for u in null if u >= uplift)) / (1 + len(null))
               if uplift is not None and null else None)
    unit_pairs = [(u[1] * u[2], t["residual"]) for t in trials for u in t["units"]]
    concord = [r for s, r in unit_pairs if s > 0]
    discord = [r for s, r in unit_pairs if s < 0]
    if (observed["hinted_trials"] < MIN_HINTED_TRIALS or observed["follow_trials"] < MIN_ARM_TRIALS
            or observed["oppose_trials"] < MIN_ARM_TRIALS):
        verdict = "INCONCLUSIVE_INSUFFICIENT_COVERAGE"
    elif uplift is not None and uplift >= MIN_UPLIFT and p_value is not None and p_value <= MAX_PERM_P:
        verdict = "SIGNAL"
    else:
        verdict = "NO_SIGNAL"
    return {
        "objective": objective,
        "evaluable_trials": len(evaluable),
        "trials_with_any_hint": sum(1 for t in per_trial if t["hints_available"]),
        "hint_emissions": hint_kinds,
        **observed,
        "scrambled_control": {"reps": perm_reps, "null_uplift_mean":
                              statistics.fmean(null) if null else None,
                              "one_sided_p": p_value},
        "unit_level": {
            "units": len(unit_pairs), "concordant": len(concord), "discordant": len(discord),
            "mean_residual_concordant": statistics.fmean(concord) if concord else None,
            "mean_residual_discordant": statistics.fmean(discord) if discord else None,
        },
        "verdict": verdict,
        "per_trial": per_trial,
    }


def coverage_ceiling(pairs: Sequence[ConfigScore], *, objective: str = "quality") -> dict[str, Any]:
    """Threshold-free upper bound on hinted trials.

    Counts the trials that touch at least one parameter with >= MIN_HISTORY earlier
    evaluable points and whose value differs from that history's median, i.e. the
    hinted-trial count if EVERY eligible parameter carried a directional hint. When
    this is below MIN_HINTED_TRIALS, no choice of rho or stability threshold can
    reach coverage: the corpus itself is short, not the synthesizer's filters.
    """
    evaluable = [p for p in pairs if p.residual.get(objective) is not None]
    seen: dict[str, list[float]] = {}
    ceiling = 0
    for pair in evaluable:
        touched = False
        for key, value in pair.params.items():
            prior = seen.get(key, [])
            if len(prior) >= MIN_HISTORY and value != statistics.median(prior):
                touched = True
        ceiling += touched
        for key, value in pair.params.items():
            seen.setdefault(key, []).append(value)
    return {"objective": objective, "max_hinted_trials_any_threshold": ceiling,
            "min_hinted_trials_required": MIN_HINTED_TRIALS,
            "reachable": ceiling >= MIN_HINTED_TRIALS}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(journal_dir: Path) -> dict[str, Any]:
    shards = journal_shards(journal_dir)
    journal = ExperimentJournal(journal_dir=journal_dir)
    entries = journal.entries_with_supersessions()
    pairs = extract_pairs(entries)
    final_hints = synthesize_hints(pairs, objective="quality")
    return {
        "schema": SCHEMA,
        "inputs": {"journal_dir": str(journal_dir),
                   "shards": [{"name": p.name, "sha256": _sha256(p)} for p in shards],
                   "trial_entries": len(entries),
                   "supersession_events": len(journal.supersession_events())},
        "parameters": {
            "min_history": MIN_HISTORY, "min_abs_rho": MIN_ABS_RHO,
            "min_sign_stability": MIN_SIGN_STABILITY, "bootstrap_reps": BOOTSTRAP_REPS,
            "baseline_window": BASELINE_WINDOW, "min_baseline_points": MIN_BASELINE_POINTS,
            "min_hinted_trials": MIN_HINTED_TRIALS, "min_arm_trials": MIN_ARM_TRIALS,
            "min_uplift": MIN_UPLIFT, "max_perm_p": MAX_PERM_P,
            "permutation_reps": PERMUTATION_REPS, "seed": SEED,
        },
        "corpus": {
            "config_score_pairs": len(pairs),
            "trial_id_range": [pairs[0].trial_id, pairs[-1].trial_id] if pairs else None,
            "surfaces": {s: sum(1 for p in pairs if p.surface == s)
                         for s in sorted({p.surface for p in pairs})},
            "params": {k: sum(1 for p in pairs if k in p.params)
                       for k in sorted({k for p in pairs for k in p.params})},
        },
        "backtest": {obj: backtest(pairs, objective=obj) for obj in ("quality", "speed")},
        "coverage_ceiling": {obj: coverage_ceiling(pairs, objective=obj)
                             for obj in ("quality", "speed")},
        "unlogged_params": {
            "numeric_trial_entries_with_empty_or_missing_params": sum(
                1 for e in entries if e.action_type == "numeric_trial"
                and not (isinstance(e.config_snapshot, dict)
                         and isinstance(e.config_snapshot.get("params"), dict)
                         and e.config_snapshot["params"])),
            "note": "rows without logged params cannot be (config, score) pairs; never back-filled",
        },
        "final_hints_quality": {k: asdict(v) for k, v in final_hints.items()},
        "final_hints_quality_text": render_hints(final_hints),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--journal-dir", type=Path, required=True,
                        help="directory holding every autopilot_journal*.jsonl shard (read only)")
    parser.add_argument("--out", type=Path, help="write the JSON record here")
    args = parser.parse_args(argv)
    if not args.journal_dir.is_dir():
        print(f"journal directory does not exist: {args.journal_dir}", file=sys.stderr)
        return 2
    record = run(args.journal_dir.resolve())
    text = json.dumps(record, indent=1, sort_keys=True)
    if args.out:
        args.out.write_text(text + "\n")
    q = record["backtest"]["quality"]
    print(json.dumps({k: q[k] for k in ("verdict", "evaluable_trials", "hinted_trials",
                                         "baseline_success_rate", "follow_trials",
                                         "follow_success_rate", "oppose_trials",
                                         "oppose_success_rate", "uplift_follow_vs_baseline")}
                     | {"perm_p": q["scrambled_control"]["one_sided_p"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
