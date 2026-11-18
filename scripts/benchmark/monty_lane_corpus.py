#!/usr/bin/env python3
"""D-RI-MONTY-LANE: replay the T1 corpus through CPython and through the Monty lane, side by side.

Reuses the corpus loaders of ``monty_corpus_share.py`` (tap sessions in turn order, role-labelled eval cells). Unlike
the T1 run, the lane now BRIDGES every REPL callable, so replaying the corpus would really write files, run shells
and call models. It must not: every REPL tool other than the pure context helpers (peek, grep, context_len,
chunk_context, FINAL, FINAL_VAR) is replaced in BOTH arms by an inert stub that records its call and returns a fixed
string. With identical stubs both arms see identical tool results, so the comparison is byte parity:
same output, error, FINAL and user state per cell. Per cell it records the lane's route, screen reason and
fallback reason. Offline, CPU only; the canonical run uses the private Monty venv.

    python scripts/benchmark/monty_lane_corpus.py --out docs/experiments/monty-lane-corpus.json
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.features import Features, set_features  # noqa: E402
from src.repl_environment import REPLConfig, REPLEnvironment  # noqa: E402

REAL = frozenset({"peek", "grep", "context_len", "chunk_context", "FINAL", "FINAL_VAR"})


def _load_t1():
    spec = importlib.util.spec_from_file_location("monty_corpus_share", os.path.join(ROOT, "scripts/benchmark/monty_corpus_share.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["monty_corpus_share"] = mod
    spec.loader.exec_module(mod)
    return mod


def _stubbed_env(context: str, timeout_s: int) -> REPLEnvironment:
    from src.repl_environment.monty_adapter import tool_names

    env = REPLEnvironment(context=context, config=REPLConfig(timeout_seconds=timeout_s))
    env._stub_calls = []
    for n in tool_names(env):
        v = env._globals.get(n)
        if n in REAL or not callable(v) or type(v).__name__ == "module" or n.startswith("_"):
            continue

        def stub(*a, _n=n, **k):
            env._charge_tool_call(_n)
            env._stub_calls.append(_n)
            return f"[stub {_n}]"

        env._globals[n] = stub
    return env


def _state(env) -> dict:
    from src.repl_environment.monty_adapter import tool_names

    names = tool_names(env)
    out = {}
    for k, v in env._globals.items():
        if k in names or k.startswith("_"):
            continue
        out[k] = repr(v) if not callable(v) else f"<callable {getattr(v, '__name__', type(v).__name__)}>"
    return {"globals": out, "artifacts": repr(env.artifacts)}


def run_group(cells: list[dict], context: str, timeout_s: int) -> None:
    from src.repl_environment.monty_adapter import MontyREPLAdapter

    cp = _stubbed_env(context, timeout_s)
    me = _stubbed_env(context, timeout_s)
    ad = MontyREPLAdapter(me, telemetry=False)
    try:
        for c in cells:
            t0 = time.perf_counter()
            a = cp.execute(c["code"])
            t1 = time.perf_counter()
            b = ad.execute(c["code"])
            t2 = time.perf_counter()
            row = ad.route_log[-1]
            same = (a.output, a.error, a.is_final, a.final_answer) == (b.output, b.error, b.is_final, b.final_answer)
            c["lane"] = {
                "route": row["route"], "screen": row["screen"], "fallback": row.get("fallback"),
                "parity": same, "state_parity": _state(cp) == _state(me),
                "cpython_error": (a.error or "")[:120] or None, "lane_error": (b.error or "")[:120] or None,
                "cpython_ms": round((t1 - t0) * 1000, 3), "lane_ms": round((t2 - t1) * 1000, 3),
                "worker_reused": row.get("worker_reused"),
            }
            if cp.timed_out or me.timed_out:
                break
    finally:
        ad.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/experiments/monty-lane-corpus.json")
    ap.add_argument("--timeout", type=int, default=3)
    args = ap.parse_args()
    set_features(Features(monty_repl=True))
    t1 = _load_t1()
    cells = t1.load_tap() + t1.load_eval()
    fix = t1.fixture_corpus(ROOT)
    for c in cells:
        c["code"] = c["code"].strip("\n")
        c["sha"] = "c" + hashlib.sha256(c["code"].encode()).hexdigest()[:12]
        c["fixture"] = (c["source"] == "A_tap") and (c["code"].strip() in fix)
    seen: dict[tuple, dict] = {}
    for c in cells:
        k = (c["source"], c["role"], c["sha"])
        if k in seen:
            seen[k]["occurrences"] += 1
        else:
            c["occurrences"] = 1
            seen[k] = c
    uniq = list(seen.values())
    groups: dict = collections.defaultdict(list)
    for c in uniq:
        groups[("tap", c["session"]) if c["source"] == "A_tap" else ("one", c["sha"], c["role"])].append(c)
    for g in groups.values():
        run_group(g, t1.SYNTH_CONTEXT, args.timeout)
    from src.repl_environment.monty_adapter import shutdown_pools

    shutdown_pools()
    done = [c for c in uniq if "lane" in c]

    def share(rows):
        n = len(rows)
        if not n:
            return {"cells": 0}
        insub = [r for r in rows if r["lane"]["screen"] == "in_subset"]
        fb = [r for r in insub if r["lane"]["fallback"]]
        return {
            "cells": n,
            "monty_route": sum(r["lane"]["route"] == "monty" for r in rows),
            "screened_out": n - len(insub),
            "screened_in": len(insub),
            "fell_back": len(fb),
            "parity_output_error_final": sum(r["lane"]["parity"] for r in rows),
            "parity_state": sum(r["lane"]["state_parity"] for r in rows),
            "cpython_ok_lane_failed": sum(1 for r in rows if not r["lane"]["cpython_error"] and r["lane"]["lane_error"]),
            "screen_reasons": dict(collections.Counter(r["lane"]["screen"].split(":")[0] + ":" + (r["lane"]["screen"].split(":") + [""])[1]
                                                       for r in rows if r["lane"]["screen"] != "in_subset").most_common()),
            "fallback_reasons": dict(collections.Counter(r["lane"]["fallback"].split(":")[0] + ":" + (r["lane"]["fallback"].split(":") + [""])[1]
                                                         for r in fb).most_common()),
        }

    strata = {
        "all": done,
        "A_tap": [c for c in done if c["source"] == "A_tap"],
        "A_tap_nonfixture": [c for c in done if c["source"] == "A_tap" and not c["fixture"]],
        "B_eval": [c for c in done if c["source"] != "A_tap"],
    }
    result = {
        "date": time.strftime("%Y-%m-%d"),
        "strata": {k: share(v) for k, v in strata.items()},
        "cells": [{"sha": c["sha"], "source": c["source"], "role": c["role"], "fixture": c["fixture"],
                   "occurrences": c["occurrences"], **c["lane"]} for c in done],
    }
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=1, default=str)
    print(json.dumps(result["strata"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
