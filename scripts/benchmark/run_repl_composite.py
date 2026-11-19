#!/usr/bin/env python3
"""D-RI-PREVIEW-1.2 driver: composite multi-turn history-rendering comparator (fixture: repl_composite_fixture.py).

Modes
  --pilot            ZERO model calls: run scripted reference policies (reader / retype / requery) through every rendering
                     (plus the `full` control) and write the discrimination table. Proves the fixture CAN separate renderings.
  --mock             live-mode plumbing with the `reader` policy standing in for the model (no server, no window).
  --smoke-out FILE   live plumbing smoke: ONE answered chat request (episode 0, turn 1) -> FILE (SMOKE_OK). Not a measurement.
  (default, live)    paired episodes i=0..n-1; for each episode every rendering cell plays the same episode; resumable.

Live mode refuses unless WINDOW_CONFIRMED=1 and the --role NAME=URL@PID passes the R5 port-ownership proof. It never starts
or stops a server. Spec: PREDECLARED.md next to the row (frozen before any run).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load(name: str):
    sp = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    m = importlib.util.module_from_spec(sp)
    sys.modules[name] = m
    sp.loader.exec_module(m)
    return m


fx = _load("repl_composite_fixture")
rp = None  # run_repl_preview (port-ownership proof, role parsing); loaded lazily so --pilot needs nothing else

DEFAULT_N = 30
DEFAULT_LOOKS = (10, 20)
DEFAULT_TEMP = 0.7
DEFAULT_MAX_TOKENS = 700
MAX_BAD_FINAL_SHARE = 0.20      # harness/prompt failure share that voids a cell (FINAL assigned or non-dict)
GAP_LOOK20 = 0.10               # look-20 futility: every pairwise correct-rate gap <= this
ALPHA_PAIR = 0.05 / 3           # Bonferroni over the 3 pairwise exact McNemar tests
MIN_GAP = 0.30
REPLY_KEEP = 1500


def chat(url: str, prompt: str, temperature: float, max_tokens: int, seed: int) -> str:
    body = json.dumps({"messages": [{"role": "user", "content": prompt}], "temperature": temperature, "max_tokens": max_tokens,
                       "seed": seed, "chat_template_kwargs": {"enable_thinking": False}, "cache_prompt": True}).encode()
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


def mcnemar_exact(a: list[bool], b: list[bool]) -> dict:
    """Two-sided exact McNemar on paired correct flags: n10 = a-only, n01 = b-only."""
    n10 = sum(1 for x, y in zip(a, b) if x and not y)
    n01 = sum(1 for x, y in zip(a, b) if y and not x)
    n = n10 + n01
    if n == 0:
        return {"n10": 0, "n01": 0, "p": 1.0}
    k = min(n10, n01)
    p = min(1.0, 2 * sum(math.comb(n, j) for j in range(k + 1)) / 2 ** n)
    return {"n10": n10, "n01": n01, "p": round(p, 6)}


def row_of(ep: dict, played: dict, i: int) -> dict:
    c = fx.classify(played["final"], ep)
    cls = c["cls"] if played["how"] == "final" else ("no_answer")
    return {"i": i, "how": played["how"], "cls": cls, "final": played["final"], "n_ok": c["n_ok"], "early_ok": c["early_ok"],
            "fields": c["fields"], "distractor_picks": c["distractor_picks"], "n_turns": played["n_turns"],
            "turns": [{"code": t["code"][:REPLY_KEEP], "obs": t["obs"][:REPLY_KEEP], "reply": t["reply"][:REPLY_KEEP]}
                      for t in played["turns"]]}


def tally(cells: dict[str, list[dict]]) -> dict:
    out = {}
    for key, rows in cells.items():
        n = len(rows)
        c = {"correct": 0, "partial": 0, "wrong": 0, "no_answer": 0}
        for r in rows:
            c[r["cls"]] += 1
        out[key] = {"n": n, **c, "bad_final": sum(1 for r in rows if r["how"] == "bad_final"),
                    "no_final": sum(1 for r in rows if r["how"] == "no_final"),
                    "correct_rate": round(c["correct"] / n, 4) if n else None,
                    "early_ok_rate": round(sum(r["early_ok"] for r in rows) / n, 4) if n else None,
                    "field_ok_rate": round(sum(r["n_ok"] for r in rows) / (n * len(fx.KEYS)), 4) if n else None,
                    "mean_turns": round(sum(r["n_turns"] for r in rows) / n, 3) if n else None}
    return out


def void_cells(cells: dict, n_planned: int) -> list[str]:
    """A cell whose harness-failure (bad_final) share exceeds 20% of the planned n is a driver/prompt failure, not a result."""
    return sorted(k for k, rows in cells.items() if sum(1 for r in rows if r["how"] == "bad_final") * 5 > n_planned)


def look_decision(cells: dict, look: int) -> str | None:
    """Futility-only interim look on the first `look` paired episodes. Returns a futility name or None (continue)."""
    c = {k: sum(1 for r in rows[:look] if r["cls"] == "correct") for k, rows in cells.items()}
    if any(len(rows) < look for rows in cells.values()):
        return None
    if look == 10 and all(v >= 9 for v in c.values()):
        return "FUTILITY_CEILING"
    if look == 10 and all(v == 0 for v in c.values()):
        return "FUTILITY_FLOOR"
    if look == 20:
        vals = list(c.values())
        if max(vals) - min(vals) <= GAP_LOOK20 * look:
            return "FUTILITY_NO_GAP"
    return None


def pair_analysis(cells: dict, renderings: list[str], role: str) -> dict:
    out = {}
    flags = {r: [x["cls"] == "correct" for x in cells[f"{role}|{r}"]] for r in renderings}
    for a_i in range(len(renderings)):
        for b_i in range(a_i + 1, len(renderings)):
            a, b = renderings[a_i], renderings[b_i]
            m = min(len(flags[a]), len(flags[b]))
            st = mcnemar_exact(flags[a][:m], flags[b][:m])
            gap = (sum(flags[a][:m]) - sum(flags[b][:m])) / m if m else None
            st.update({"n_pairs": m, "rate_diff_a_minus_b": round(gap, 4) if gap is not None else None,
                       "separates": bool(m and abs(gap) >= MIN_GAP and st["p"] < ALPHA_PAIR)})
            out[f"{a} vs {b}"] = st
    return out


def run_pilot(n: int, out: Path) -> int:
    table = {}
    for mode in ("reader", "retype", "requery"):
        for rend in fx.ALL_RENDERINGS:
            rows = []
            for i in range(n):
                ep = fx.build_episode(i)
                pol = fx.reference_policy(mode, ep["project"])
                played = fx.play_episode(ep, rend, pol)
                row = row_of(ep, played, i)
                row["prompt_chars"] = [len(fx.build_prompt(ep, played["turns"][:t], rend, t + 1)) for t in range(played["n_turns"])]
                rows.append(row)
            t = tally({f"{mode}|{rend}": rows})[f"{mode}|{rend}"]
            t["mean_prompt_chars"] = round(sum(sum(r["prompt_chars"]) for r in rows) / sum(len(r["prompt_chars"]) for r in rows), 1)
            t["early_visible_all3"] = None
            table[f"{mode}|{rend}"] = t
            table[f"{mode}|{rend}"]["flags"] = [r["cls"] == "correct" for r in rows]
    res = {"schema": "epyc.repl_composite_pilot.v1", "model_calls": 0, "n_episodes": n, "fixture_sha256": fx.fixture_sha256(n),
           "table": table,
           "pairs_reader": pair_analysis({k: [{"cls": "correct" if f else "wrong"} for f in v["flags"]]
                                          for k, v in table.items()}, list(fx.RENDERINGS), "reader")}
    for v in table.values():
        del v["flags"]
    out.write_text(json.dumps(res, indent=1) + "\n")
    for k, v in table.items():
        print(f"{k:24s} correct {v['correct']:>2}/{v['n']}  early_ok {v['early_ok_rate']}  field_ok {v['field_ok_rate']}  turns {v['mean_turns']}  prompt_chars {v['mean_prompt_chars']}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--role", action="append", default=[], help="NAME=http://127.0.0.1:PORT@PID")
    ap.add_argument("--renderings", default=",".join(fx.RENDERINGS))
    ap.add_argument("--n", type=int, default=DEFAULT_N)
    ap.add_argument("--looks", default=",".join(map(str, DEFAULT_LOOKS)))
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMP)
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--max-turns", type=int, default=fx.MAX_TURNS)
    ap.add_argument("--deny-ports", default="8070,8071,8072,8080,8081,8082,8083,8090,8180,8181,8182,8000,8100")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--smoke-out")
    ap.add_argument("--cell-dir")
    ap.add_argument("--stop-file")
    ap.add_argument("--deadline-epoch", type=float)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    if a.pilot:
        if not a.out:
            ap.error("--out required")
        return run_pilot(a.n, Path(a.out))

    renderings = a.renderings.split(",")
    for r in renderings:
        if r not in fx.RENDERINGS:
            sys.exit(f"unknown rendering {r!r}; measurement cells are {fx.RENDERINGS}")
    looks = [int(x) for x in a.looks.split(",") if x]
    roles: list[tuple[str, str | None, int | None]]
    if a.mock:
        roles = [("mock_role", None, None)]
    else:
        global rp
        rp = _load("run_repl_preview")
        if os.environ.get("WINDOW_CONFIRMED") != "1":
            sys.exit("refusing: live mode runs inference. Set WINDOW_CONFIRMED=1 inside an operator-confirmed window.")
        if len(a.role) != 1:
            sys.exit("live mode needs exactly one --role NAME=URL@PID")
        name, url, pid = rp.parse_role(a.role[0])
        rp.prove_port_ownership(url, pid, {int(x) for x in a.deny_ports.split(",") if x})
        roles = [(name, url, pid)]
    role, url, _ = roles[0]

    if a.smoke_out:
        ep = fx.build_episode(0)
        prompt = fx.build_prompt(ep, [], renderings[0], 1, a.max_turns)
        text = fx.reference_policy("reader", ep["project"])(prompt, 1) if a.mock else chat(url, prompt, a.temperature, a.max_tokens, 1000)
        ok = bool(fx._CODE.search(text))
        Path(a.smoke_out).write_text(json.dumps({"verdict": "SMOKE_OK" if ok else "SMOKE_NO_CODE_BLOCK", "mock": a.mock,
                                                 "reply_head": text[:300], "note": "plumbing only, not a measurement"}, indent=1) + "\n")
        return 0 if ok else 1
    if not a.out:
        ap.error("--out is required")

    def reply_fn_for(i: int, ep: dict):
        if a.mock:
            return fx.reference_policy("reader", ep["project"])
        return lambda prompt, t: chat(url, prompt, a.temperature, a.max_tokens, seed=1000 + 10 * i + t)

    cells: dict[str, list[dict]] = {f"{role}|{r}": [] for r in renderings}
    cf = {r: (Path(a.cell_dir) / f"{role}__{r}.jsonl" if a.cell_dir else None) for r in renderings}
    if a.cell_dir:
        Path(a.cell_dir).mkdir(parents=True, exist_ok=True)
        for r in renderings:
            if cf[r].exists():
                cells[f"{role}|{r}"] = [json.loads(x) for x in cf[r].read_text().splitlines() if x.strip()]
    stopped = None
    futility = None
    t0 = time.time()
    for i in range(a.n):
        for r in renderings:
            rows = cells[f"{role}|{r}"]
            if len(rows) > i:
                continue
            if a.stop_file and Path(a.stop_file).exists():
                stopped = "stop-file"
            elif a.deadline_epoch and time.time() >= a.deadline_epoch:
                stopped = "deadline"
            if stopped:
                break
            ep = fx.build_episode(i)
            played = fx.play_episode(ep, r, reply_fn_for(i, ep), a.max_turns)
            rows.append(row_of(ep, played, i))
            if cf[r] is not None:
                with cf[r].open("a") as f:
                    f.write(json.dumps(rows[-1]) + "\n")
            if i + 1 >= min(looks or [10]) and void_cells(cells, a.n):
                stopped = "invalid-cell"
                break
        if stopped:
            break
        if (i + 1) in looks:
            futility = look_decision(cells, i + 1)
            if futility:
                stopped = futility
                break
    n_done = min(len(v) for v in cells.values())
    complete = n_done == a.n
    voided = void_cells(cells, a.n)
    valid = bool(not voided)
    done = bool(valid and (complete or futility))
    res = {"schema": "epyc.repl_composite_result.v1", "mock": a.mock, "n_per_cell": a.n, "looks": looks, "temperature": a.temperature,
           "max_turns": a.max_turns, "roles": [role], "renderings": renderings, "fixture_sha256": fx.fixture_sha256(a.n),
           "predeclared": "PREDECLARED.md in the row dir (frozen before any run)", "complete": complete, "valid": valid,
           "done": done, "void_cells": voided, "stopped": stopped, "futility": futility, "paired_episodes_done": n_done,
           "counts": tally(cells), "pairs": pair_analysis(cells, renderings, role) if n_done else {}, "rows": cells,
           "wall_s": round(time.time() - t0, 1),
           "note": "a mock result is plumbing only and must never be cited as a finding" if a.mock else ""}
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: res[k] for k in ("complete", "valid", "done", "stopped", "futility", "paired_episodes_done", "counts", "pairs")}, indent=1))
    if voided:
        print("INVALID: bad_final share above 20% in", voided, "- a driver/prompt failure, not a model result")
        return 3
    return 0 if done else 75


if __name__ == "__main__":
    sys.exit(main())
