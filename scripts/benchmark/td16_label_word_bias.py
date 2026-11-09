#!/usr/bin/env python3
"""TD-16 driver: label-word bias control on the native noul path (routing-program.md TD-16).

Question: on the worker model at its current pin, does the native ``noul`` readout (one constrained yes/no token per
question, TD-1c/1d) carry a label-word bias that a neutral two-option choice (A/B keys, the yes/no CONDITIONS as option
text, both orders) does not? The item set is hunch's 240 look-alike pairs frozen at
``research/fixtures/td16-hunch-20261102/td16-lookalikes.jsonl`` (sha256 pinned below; 80 positive / 160 negative).

Arms (each over all 240 items; the SAME catalogue batches in every arm so the arms are paired):
  N      native noul, yes_if / no_if as criteria            (the readout as run today)
  Nq     native noul, question + context only               (question-only variant, no conditions)
  C_AB   native 2-option choice, keys A,B, A = yes condition, B = no condition
  C_BA   same, order swapped (A = no condition, B = yes condition)
  Cq_AB  native choice, keys A,B, option text is just "Yes" / "No" (question-only variant of C), A = Yes
  Cq_BA  same, swapped
  J      JSON-mode noul with the same criteria as N (value only, no probability)
``--batch`` questions share one catalogue call (default 8, the TD-1c catalogue size). Native arms need token
probabilities, so they run against the CHAMPION sidecar (operator ruling 6); J runs on the same server.

Metrics per arm (and for C = the mean of the two orders mapped back to P(yes)): AUROC, argmax accuracy, Brier, ECE (10 equal
bins), order disagreement (C and Cq), both constant baselines (always-no accuracy, base-rate Brier), and item-level bootstrap
95% CIs (B=2000, seed 16; paired where two arms are compared: C - N and Cq - Nq). No gate follows (wiki/routing-intelligence.md L40).

Validity: an arm whose unanswered share (a question with no decision) exceeds 20% voids the run (valid=false, rc 3) and the run
stops at once; a void run is a driver/prompt failure, not a model result. rc 0 = complete and valid, 75 = partial (stop file or
deadline; rerun resumes from --results), 3 = voided, 2 = refusal. Live mode refuses without WINDOW_CONFIRMED=1.

  WINDOW_CONFIRMED=1 python scripts/benchmark/td16_label_word_bias.py --server-url worker_general=http://127.0.0.1:8199 \
      --results out/results.jsonl --out out/RESULT.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

FIXTURE_SHA256 = "937c4ff28332e7541c92aa62d5ab7b7100c4b4c322cb2524141e807889e0f315"
DEFAULT_FIXTURE = "/mnt/raid0/llm/epyc-root/research/fixtures/td16-hunch-20261102/td16-lookalikes.jsonl"
ARMS = ("N", "Nq", "C_AB", "C_BA", "Cq_AB", "Cq_BA", "J")
STATE = "Judge each item independently, using only the item's own text."
MAX_UNANSWERED_SHARE = 0.20
BOOT_B, BOOT_SEED = 2000, 16


# ---------------------------------------------------------------- fixture and questions
def load_items(path: str | Path = DEFAULT_FIXTURE, check_sha: bool = True) -> list[dict]:
    raw = Path(path).read_bytes()
    if check_sha and hashlib.sha256(raw).hexdigest() != FIXTURE_SHA256:
        raise SystemExit(f"fixture sha256 mismatch for {path}: not the frozen TD-16 set")
    items = [json.loads(x) for x in raw.decode().splitlines() if x.strip()]
    if check_sha and (len(items) != 240 or sum(i["label"] for i in items) != 80):
        raise SystemExit("fixture is not 240 items / 80 positive")
    return items


def batches(items: list[dict], size: int) -> list[list[dict]]:
    """Deterministic shuffled catalogue batches (same in every arm: pairing). Sorted by id, seeded shuffle."""
    order = sorted(items, key=lambda i: i["id"])
    random.Random(16).shuffle(order)
    return [order[k:k + size] for k in range(0, len(order), size)]


def _body(item: dict) -> str:
    ctx = "\n".join(f"{k}: {v}" for k, v in sorted(item["context"].items()))
    return f"{item['check']['question']}\n{ctx}"


def make_question(item: dict, arm: str):
    from src.typed_decisions.types import Question, QuestionKind
    chk = item["check"]
    if arm in ("N", "J"):
        return Question(id=item["id"], kind=QuestionKind.NOUL, text=_body(item),
                        criteria=(f"YES if: {chk['yes_if']}", f"NO if: {chk['no_if']}"))
    if arm == "Nq":
        return Question(id=item["id"], kind=QuestionKind.NOUL, text=_body(item))
    swapped = arm.endswith("_BA")
    if arm.startswith("Cq"):
        yes_t, no_t = "Yes", "No"
    else:
        yes_t, no_t = chk["yes_if"], chk["no_if"]
    descs = (no_t, yes_t) if swapped else (yes_t, no_t)
    return Question(id=item["id"], kind=QuestionKind.CHOICE, text=_body(item), options=("A", "B"), option_descriptions=descs)


def yes_key(arm: str) -> str | None:
    """The option key that means YES for a choice arm (None for noul arms)."""
    if arm.startswith("C"):
        return "A" if arm.endswith("_AB") else "B"
    return None


def p_yes(arm: str, decision) -> float | None:
    """P(yes) from one native Decision, or the 0/1 value of a JSON decision. None = no usable readout."""
    probs = {str(k).lower(): float(v) for k, v in dict(decision.probabilities).items()}
    k = yes_key(arm)
    if arm == "J":
        return None if not isinstance(decision.value, bool) else (1.0 if decision.value else 0.0)
    if k is None:
        return probs.get("true")
    return probs.get(k.lower())


# ---------------------------------------------------------------- metrics (pure python)
def auroc(scores: list[float], labels: list[int]) -> float | None:
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return None
    wins = 0.0
    for p in pos:
        for q in neg:
            wins += 1.0 if p > q else 0.5 if p == q else 0.0
    return wins / (len(pos) * len(neg))


def accuracy(scores, labels) -> float:
    return sum((s >= 0.5) == (y == 1) for s, y in zip(scores, labels)) / len(labels)


def brier(scores, labels) -> float:
    return sum((s - y) ** 2 for s, y in zip(scores, labels)) / len(labels)


def ece(scores, labels, bins: int = 10) -> float:
    n, tot = len(labels), 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, s in enumerate(scores) if (lo <= s < hi) or (b == bins - 1 and s == 1.0)]
        if idx:
            conf = sum(scores[i] for i in idx) / len(idx)
            freq = sum(labels[i] for i in idx) / len(idx)
            tot += len(idx) / n * abs(conf - freq)
    return tot


def metrics(scores, labels) -> dict:
    return {"n": len(labels), "auroc": auroc(scores, labels), "accuracy": accuracy(scores, labels),
            "brier": brier(scores, labels), "ece": ece(scores, labels)}


def _pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def boot_ci(fn, n_items: int, b: int = BOOT_B, seed: int = BOOT_SEED) -> dict:
    """Percentile 95% CI of fn(index_list) over item-level resamples (None-valued resamples are skipped)."""
    rng = random.Random(seed)
    vals = []
    for _ in range(b):
        v = fn([rng.randrange(n_items) for _ in range(n_items)])
        if v is not None:
            vals.append(v)
    return {"point": fn(list(range(n_items))), "lo": _pct(vals, 0.025) if vals else None,
            "hi": _pct(vals, 0.975) if vals else None, "reps": len(vals)}


def analyze(per_arm: dict[str, dict[str, float]], labels: dict[str, int], b: int = BOOT_B) -> dict:
    """per_arm[arm][item_id] = P(yes). Returns per-arm metrics with CIs, C/Cq order-averages, order disagreement and the
    paired C-N / Cq-Nq differences. Items are restricted to those every arm in a comparison answered."""
    out: dict = {"arms": {}, "paired": {}, "baselines": {}}
    ids_all = sorted(labels)
    y_all = [labels[i] for i in ids_all]
    out["baselines"] = {"always_no_accuracy": sum(1 for y in y_all if y == 0) / len(y_all),
                        "base_rate_brier": brier([sum(y_all) / len(y_all)] * len(y_all), y_all)}
    views = dict(per_arm)
    for fam in ("C", "Cq"):
        ab, ba = per_arm.get(f"{fam}_AB"), per_arm.get(f"{fam}_BA")
        if ab is not None and ba is not None:
            both = [i for i in ids_all if i in ab and i in ba]
            views[f"{fam}_avg"] = {i: (ab[i] + ba[i]) / 2 for i in both}
            dis = sum((ab[i] >= 0.5) != (ba[i] >= 0.5) for i in both)
            out["arms"].setdefault(f"{fam}_avg", {})["order_disagreement"] = {"n": len(both), "disagree": dis,
                                                                              "rate": dis / len(both) if both else None}
    for arm, d in views.items():
        ids = [i for i in ids_all if i in d]
        if not ids:
            continue
        s, y = [d[i] for i in ids], [labels[i] for i in ids]
        row = out["arms"].setdefault(arm, {})
        row.update(metrics(s, y))
        row["ci95"] = {m: boot_ci(lambda ix, m=m: metrics([s[j] for j in ix], [y[j] for j in ix])[m], len(ids), b)
                       for m in ("auroc", "accuracy", "brier", "ece")}
    for c, n in (("C_avg", "N"), ("Cq_avg", "Nq")):
        if c in views and n in views:
            ids = [i for i in ids_all if i in views[c] and i in views[n]]
            if not ids:
                continue
            sc, sn, y = [views[c][i] for i in ids], [views[n][i] for i in ids], [labels[i] for i in ids]
            res = {"n": len(ids)}
            for m in ("auroc", "accuracy", "brier", "ece"):
                def diff(ix, m=m):
                    a = metrics([sc[j] for j in ix], [y[j] for j in ix])[m]
                    bb = metrics([sn[j] for j in ix], [y[j] for j in ix])[m]
                    return None if a is None or bb is None else a - bb
                res[m + "_diff_" + c + "_minus_" + n] = boot_ci(diff, len(ids), b)
            d_auc = res[f"auroc_diff_{c}_minus_{n}"]
            res["c_beats_n_beyond_ci_on_auroc"] = bool(d_auc["lo"] is not None and d_auc["lo"] > 0)
            res["n_beats_c_beyond_ci_on_auroc"] = bool(d_auc["hi"] is not None and d_auc["hi"] < 0)
            out["paired"][f"{c}_vs_{n}"] = res
    return out


# ---------------------------------------------------------------- run
def server_props_model(url: str) -> str | None:
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/props", timeout=15) as r:
            mp = json.loads(r.read()).get("model_path")
        return mp if isinstance(mp, str) and mp else None
    except Exception:
        return None


def read_units(path: Path) -> dict[str, dict]:
    units: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                u = json.loads(line)
                units[u["unit_id"]] = u
    return units


def run_batch(primitives, arm: str, batch: list[dict], role: str, run_fn):
    qs = [make_question(i, arm) for i in batch]
    res = run_fn(primitives, state=STATE, questions=qs, role=role, mode="json" if arm == "J" else "native")
    by_id = {d.question_id: d for d in res.decisions}
    rows = []
    for it in batch:
        d = by_id.get(it["id"])
        p = p_yes(arm, d) if d is not None else None
        rows.append({"item": it["id"], "label": it["label"], "p_yes": p,
                     "failure": None if p is not None else ("no_decision" if d is None else "no_readout")})
    return rows, {"elapsed_ms": getattr(res, "elapsed_ms", None), "prompt_sha256": getattr(res, "prompt_sha256", None),
                  "failures": [f"{f.reason}:{f.detail}" for f in getattr(res, "failures", ())][:5]}


def main(argv=None, primitives=None, run_fn=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fixture", default=DEFAULT_FIXTURE)
    ap.add_argument("--no-sha-check", action="store_true", help="tests only")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--role", default="worker_general")
    ap.add_argument("--server-url", action="append", default=[], help="ROLE=http://127.0.0.1:PORT (the champion sidecar)")
    ap.add_argument("--results", required=True, help="jsonl, one unit per (arm, batch); resumed from")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stop-file")
    ap.add_argument("--deadline-epoch", type=float)
    ap.add_argument("--boot-b", type=int, default=BOOT_B)
    a = ap.parse_args(argv)

    arms = a.arms.split(",")
    bad = [x for x in arms if x not in ARMS]
    if bad:
        sys.exit(f"unknown arm(s) {bad}; known {ARMS}")
    items = load_items(a.fixture, check_sha=not a.no_sha_check)
    labels = {i["id"]: i["label"] for i in items}
    plan = batches(items, a.batch)
    overrides = {}
    for spec in a.server_url:
        r, _, u = spec.partition("=")
        if not (r and u.startswith("http://127.0.0.1:")):
            sys.exit(f"--server-url must be ROLE=http://127.0.0.1:PORT, got {spec!r}")
        overrides[r] = u.rstrip("/")
    if primitives is None:
        if os.environ.get("WINDOW_CONFIRMED") != "1":
            print("refusing: live mode runs inference. Set WINDOW_CONFIRMED=1 inside a granted window.", file=sys.stderr)
            return 2
        if a.role not in overrides:
            print(f"refusing: --server-url {a.role}=URL is required (never the production stack's configured URL)", file=sys.stderr)
            return 2
        from src.typed_decisions.measure import _live_primitives
        primitives = _live_primitives(overrides)
    if run_fn is None:
        from src.typed_decisions.runner import run_typed_decisions as run_fn

    results = Path(a.results)
    results.parent.mkdir(parents=True, exist_ok=True)
    units = read_units(results)
    stopped, t0 = None, time.time()

    def should_stop():
        if a.stop_file and Path(a.stop_file).exists():
            return "stop-file"
        if a.deadline_epoch and time.time() >= a.deadline_epoch:
            return "deadline"
        return None

    def unanswered(arm):
        q = sum(1 for k, u in units.items() if u.get("arm") == arm for r in u["rows"] if r["p_yes"] is None)
        return q, len(items)

    voided: list[str] = []
    for arm in arms:
        for bi, batch in enumerate(plan):
            uid = f"{arm}-b{bi:03d}"
            if uid in units:
                continue
            stopped = should_stop()
            if stopped:
                break
            rows, meta = run_batch(primitives, arm, batch, a.role, run_fn)
            u = {"unit_id": uid, "arm": arm, "batch": bi, "ok": all(r["p_yes"] is not None for r in rows), "rows": rows, **meta}
            units[uid] = u
            with results.open("a") as f:
                f.write(json.dumps(u) + "\n")
            q, n = unanswered(arm)
            if q * 1000 > int(MAX_UNANSWERED_SHARE * 1000) * n:   # cannot recover: stop spending samples
                voided.append(arm)
                stopped = "invalid-arm"
                break
        if stopped:
            break

    per_arm: dict[str, dict[str, float]] = {x: {} for x in arms}
    unans = {}
    for u in units.values():
        if u["arm"] in per_arm:
            for r in u["rows"]:
                if r["p_yes"] is not None:
                    per_arm[u["arm"]][r["item"]] = r["p_yes"]
    for x in arms:
        unans[x] = {"answered": len(per_arm[x]), "unanswered": sum(1 for u in units.values() if u["arm"] == x for r in u["rows"]
                                                                  if r["p_yes"] is None)}
        if unans[x]["unanswered"] * 1000 > int(MAX_UNANSWERED_SHARE * 1000) * len(items) and x not in voided:
            voided.append(x)
    complete = all(len(read_arm_units(units, x)) == len(plan) for x in arms)
    valid = bool(complete and not voided)
    res = {"schema": "epyc.td16_label_word_bias.v1", "complete": complete, "valid": valid, "void_arms": sorted(voided), "stopped": stopped,
           "fixture_sha256": FIXTURE_SHA256, "n_items": len(items), "batch": a.batch, "n_batches": len(plan), "arms_run": arms,
           "role": a.role, "server_model_path": server_props_model(overrides[a.role]) if a.role in overrides else None,
           "unanswered": unans, "wall_s": round(time.time() - t0, 1),
           "analysis": analyze({k: v for k, v in per_arm.items() if v}, labels, a.boot_b) if any(per_arm.values()) else None,
           "note": "observation per MEASUREMENT.md; no gate follows (wiki/routing-intelligence.md L40); a void run is not a model result"}
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: res[k] for k in ("complete", "valid", "void_arms", "stopped", "unanswered")}, indent=1))
    if voided:
        return 3
    return 0 if complete else 75


def read_arm_units(units: dict, arm: str) -> list[dict]:
    return [u for u in units.values() if u["arm"] == arm]


if __name__ == "__main__":
    sys.exit(main())
