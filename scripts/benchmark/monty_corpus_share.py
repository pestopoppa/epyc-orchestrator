#!/usr/bin/env python3
"""D-RI-MONTY-T1: share of REAL REPL cells that run unchanged under the restricted Monty adapter.

CPU only, offline, no inference servers. Run with the PRIVATE venv that has ``pydantic-monty``:

    PYTHONPATH=.:/mnt/raid0/llm/epyc-orchestrator/.venv/lib/python3.11/site-packages:/usr/lib/python3/dist-packages \
        /mnt/raid0/llm/tmp/monty-slice-venv/bin/python scripts/benchmark/monty_corpus_share.py \
        --out docs/experiments/monty-corpus-share.json

Cell sources (what exists on this host; see the report):
  A  ``/mnt/raid0/llm/tmp/repl_tap.log``  graph REPL tap: full code of every cell the graph REPL executed (no role).
  B  eval results with a role label: ``3way_*.jsonl`` answers that contain a REPL ``FINAL(`` call, and the
     ``tool_compliance`` responses (REPL-tool prompt) in ``runs/**/*.json``. Model-emitted code, not executed cells.

Stages per cell
  (a) the REPL AST security gate (``REPLEnvironment._validate_code``)
  (b) a static Monty-subset screen (own AST scan: generators, class bases, unsupported imports, absent names)
  (c) real Monty execution through ``MontyREPLAdapter`` (worker confined under the SEC-EXEC shim; ``peek``/``grep``/
      ``context_len``/``FINAL`` bridged to a synthetic context; every other REPL callable is DENIED by the adapter, so no
      cell can reach a live service).
Cells of one tap session (turn counter 1,2,3..) run in order in one adapter so cross-cell state resolves as it did.
"""

from __future__ import annotations

import argparse
import ast
import collections
import glob
import hashlib
import json
import os
import random
import re
import sys
import time

os.environ.setdefault("ORCHESTRATOR_MONTY_REPL", "1")

from src.repl_environment import REPLConfig, REPLEnvironment  # noqa: E402
from src.repl_environment.monty_adapter import MontyREPLAdapter, MontyUnavailable  # noqa: E402

TAP = "/mnt/raid0/llm/tmp/repl_tap.log"
EVAL_DIR = "/mnt/raid0/llm/epyc-inference-research/benchmarks/results"
SYNTH_CONTEXT = "alpha line one\nbeta line two 123\ngamma line three\n" * 200
CELL_TIMEOUT_S = 3


# ------------------------------------------------------------------ corpus
def load_tap() -> list[dict]:
    text = open(TAP, errors="replace").read()
    cells: list[dict] = []
    pat = re.compile(
        r"={72}\n\[(?P<ts>[^\]]+)\] REPL turn (?P<turn>\d+)(?: \| (?P<meta>[^\n]*))?\n-{72}\n\[turn \d+\] \$ python3 <<'CODE'\n(?P<code>.*?)\nCODE\n"
        r"(?P<res>.*?)(?=\n?={72}\n|\Z)", re.S)
    sess = 0
    for m in pat.finditer(text):
        turn = int(m["turn"])
        if turn == 1:
            sess += 1
        res = m["res"]
        meta = dict(kv.split("=", 1) for kv in (m["meta"] or "").split() if "=" in kv)
        cells.append({
            "source": "A_tap", "role": meta.get("role", "unlabelled"), "request_id": meta.get("request_id"),
            "task_id": meta.get("task_id"), "session": sess, "turn": turn, "ts": m["ts"],
            "code": m["code"], "orig_error": ("ERROR:" in res), "orig_final": ("] FINAL" in res),
            "orig_err_text": res.split("ERROR:", 1)[1][:300] if "ERROR:" in res else "",
        })
    return cells


def _py_blocks(txt: str) -> list[str]:
    bl = re.findall(r"```(?:python|py)?\n(.*?)(?:```|\Z)", txt, re.S)
    if bl:
        return bl
    try:  # no fence: keep the text only if it is itself Python
        ast.parse(txt)
        return [txt]
    except SyntaxError:
        return []


def load_eval() -> list[dict]:
    cells: list[dict] = []
    for f in sorted(glob.glob(EVAL_DIR + "/eval/3way_2026*.jsonl")):
        for ln in open(f, errors="replace"):
            try:
                d = json.loads(ln)
            except Exception:
                continue
            for k, r in (d.get("role_results") or {}).items():
                a = (r or {}).get("answer") or ""
                if "FINAL(" not in a:
                    continue
                for b in _py_blocks(a):
                    if "FINAL(" in b:
                        cells.append({"source": "B_3way_final", "role": k.split(":")[0], "session": None, "turn": 1,
                                      "code": b, "orig_error": False, "orig_final": False, "orig_err_text": ""})
    for f in glob.glob(EVAL_DIR + "/runs/**/*.json", recursive=True):
        try:
            d = json.load(open(f))
            tc = d["results"]["tool_compliance"]
        except Exception:
            continue
        for q, v in tc.items():
            for b in _py_blocks(v.get("response") or ""):
                cells.append({"source": "B_tool_compliance", "role": str(d.get("model_role")), "session": None,
                              "turn": 1, "code": b, "orig_error": False, "orig_final": False, "orig_err_text": ""})
    return cells


def fixture_corpus(root: str) -> str:
    parts = []
    for pat in ("tests/**/*.py", "scripts/**/*.py"):
        for f in glob.glob(os.path.join(root, pat), recursive=True):
            try:
                parts.append(open(f, errors="replace").read())
            except Exception:
                pass
    return "\n".join(parts)


# ------------------------------------------------------------------ (b) static screen
MONTY_MODULES = {"json", "re", "math", "collections", "itertools", "functools", "typing", "datetime", "string",
                 "random", "sys", "os", "asyncio", "pathlib", "dataclasses"}  # refined by observation, see report
ABSENT_NAMES = {"statistics", "numpy", "np", "scipy", "pandas", "pd", "fractions", "decimal", "heapq", "bisect",
                "textwrap", "operator", "copy", "hashlib", "base64", "csv", "io", "urllib", "difflib"}
# REPL tool names the model uses that the parent only registers when tools are enabled (NameError here, not a Monty gap
# in itself, but outside the bridged peek/grep/context_len/FINAL subset all the same).
REPL_TOOL_NAMES = {"TOOL", "llm_call", "llm_batch", "list_tools", "web_search", "web_fetch", "run_shell", "list_dir",
                   "file_info", "file_write_safe", "ocr_document", "analyze_figure", "CALL", "ALLOWED_TOOLS"}
BRIDGED = {"peek", "grep", "context_len", "FINAL", "FINAL_VAR", "print"}


def static_screen(code: str, repl_callables: set[str]) -> list[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ["cpython_syntax_error"]
    out: set[str] = set()
    defined = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    defined |= {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    for n in ast.walk(tree):
        if isinstance(n, (ast.Yield, ast.YieldFrom)):
            out.add("generator")
        elif isinstance(n, ast.GeneratorExp):
            out.add("generator_expression")
        elif isinstance(n, ast.ClassDef) and n.bases:
            out.add("class_inheritance_user_exception" if any(
                isinstance(b, ast.Name) and "Exception" in b.id or "Error" in getattr(b, "id", "") for b in n.bases)
                else "class_inheritance")
        elif isinstance(n, ast.ClassDef):
            out.add("class_def")
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            out.add("import_statement")
        elif isinstance(n, ast.Name) and n.id in ABSENT_NAMES and n.id not in defined:
            out.add(f"absent_name:{n.id}")
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
            nm = n.func.id
            if nm in (repl_callables | REPL_TOOL_NAMES) and nm not in BRIDGED and nm not in defined:
                out.add(f"repl_callable:{nm}")
            elif nm in ("peek", "grep") and any(k.arg == "file_path" for k in n.keywords):
                out.add(f"repl_callable:{nm}_file_path")
        elif isinstance(n, ast.Match):
            out.add("match_statement")
        elif isinstance(n, (ast.With, ast.AsyncWith)):
            out.add("with_statement")
    return sorted(out)


# ------------------------------------------------------------------ (c) classification of runtime errors
def classify_runtime(err: str, denied: list[dict]) -> str:
    if denied:
        return "repl_callable_denied:" + sorted({d.get("name", "?") for d in denied})[0]
    e = err or ""
    if "NotImplementedError" in e:
        if "generator" in e.lower() or "yield" in e.lower():
            return "generator"
        if "class" in e.lower() or "inherit" in e.lower() or "base" in e.lower():
            return "class_or_exception_subclass"
        return "NotImplementedError:" + e[:60].replace("\n", " ")
    if e.startswith("SyntaxError"):
        return "monty_syntax_unsupported:" + e[:70].replace("\n", " ")
    m = re.match(r"ModuleNotFoundError:?\s*(.*)", e)
    if m:
        return "module_missing:" + (re.findall(r"['\"]?([\w.]+)['\"]?", m[1]) or ["?"])[-1] if m[1] else "module_missing"
    m = re.match(r"NameError:?.*?'(\w+)'", e)
    if m:
        n = m[1]
        if n in REPL_TOOL_NAMES:
            return "repl_callable_unbridged:" + n
        return "name_absent_stdlib_or_pkg:" + n if n in ABSENT_NAMES else "NameError:" + n
    m = re.match(r"(\w+Error|\w+Exception|\w+):", e)
    return "runtime:" + (m[1] if m else e[:40].replace("\n", " "))


def run_session(cells: list[dict], timeout_s: int, ast_gate_probe: REPLEnvironment) -> None:
    """Run cells (one tap session or one cell) in order through one adapter; fill cell['res']."""
    cfg = REPLConfig(timeout_seconds=timeout_s)
    env = REPLEnvironment(context=SYNTH_CONTEXT, config=cfg)
    ad = MontyREPLAdapter(env)
    try:
        for c in cells:
            code = c["code"]
            n_den = len(ad.denied_log)
            n_unav = len(ad.unavailable)
            t0 = time.perf_counter()
            try:
                r = ad.execute(code)
                err = r.error
                final = r.is_final
            except MontyUnavailable as e:
                err, final = f"MontyUnavailable: {e}", False
            c["res"] = {
                "error": err, "final": final,
                "denied": ad.denied_log[n_den:],
                "nonjson_state": sorted(set(ad.unavailable) - set(list(ad.unavailable)[:n_unav])) if len(ad.unavailable) > n_unav else [],
                "ms": round((time.perf_counter() - t0) * 1000, 1),
                "terminal_timeout": bool(env.timed_out),
            }
            if env.timed_out:
                # timeout is terminal for this env; later cells of the session cannot be judged on it
                idx = cells.index(c)
                for c2 in cells[idx + 1:]:
                    c2["res"] = {"error": "SKIPPED_AFTER_TIMEOUT", "final": False, "denied": [], "nonjson_state": [],
                                 "ms": 0, "terminal_timeout": False}
                break
    finally:
        ad.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/experiments/monty-corpus-share.json")
    ap.add_argument("--max-cells", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261118)
    ap.add_argument("--root", default=os.getcwd())
    args = ap.parse_args()

    tap = load_tap()
    ev = load_eval()
    fix = fixture_corpus(args.root)
    for c in tap + ev:
        c["code"] = c["code"].strip("\n")
        c["sha"] = "c" + hashlib.sha256(c["code"].encode()).hexdigest()[:12]  # "c" prefix: an all-digit hex run trips the PII hook
        c["fixture"] = (c["source"] == "A_tap") and (c["code"].strip() in fix)

    raw_counts = collections.Counter((c["source"], c["role"]) for c in tap + ev)
    # dedupe identical code within (source, role); keep occurrence count
    seen: dict[tuple, dict] = {}
    for c in tap + ev:
        k = (c["source"], c["role"], c["sha"])
        if k in seen:
            seen[k]["occurrences"] += 1
        else:
            c["occurrences"] = 1
            seen[k] = c
    uniq = list(seen.values())

    # Sample: all B cells (small), A cells stratified fixture / non-fixture up to the cap.
    rng = random.Random(args.seed)
    B = [c for c in uniq if c["source"] != "A_tap"]
    A = [c for c in uniq if c["source"] == "A_tap"]
    room = max(0, args.max_cells - len(B))
    if len(A) > room:
        # keep whole tap sessions in order: sample sessions
        by_sess = collections.defaultdict(list)
        for c in A:
            by_sess[c["session"]].append(c)
        keys = list(by_sess)
        rng.shuffle(keys)
        A, n = [], 0
        for k in keys:
            if n + len(by_sess[k]) > room:
                continue
            A += by_sess[k]
            n += len(by_sess[k])
    sample = A + B

    probe = REPLEnvironment(context=SYNTH_CONTEXT, config=REPLConfig(timeout_seconds=CELL_TIMEOUT_S))
    repl_callables = {k for k in probe._builtin_global_keys if callable(probe._globals.get(k))}

    # (a) gate and (b) static screen per cell
    from src.repl_environment.environment import REPLSecurityError  # noqa: E402
    for c in sample:
        try:
            probe._validate_code(c["code"])
            c["gate"] = None
        except REPLSecurityError as e:
            c["gate"] = str(e)[:120]
        except Exception as e:  # noqa: BLE001
            c["gate"] = f"gate_exc:{type(e).__name__}"
        c["static"] = static_screen(c["code"], repl_callables)

    # (c) execution: tap cells grouped by session in order, eval cells one per adapter
    groups: list[list[dict]] = []
    cur: dict[int, list[dict]] = collections.defaultdict(list)
    for c in sample:
        if c["source"] == "A_tap":
            cur[c["session"]].append(c)
        else:
            groups.append([c])
    groups = list(cur.values()) + groups
    t0 = time.time()
    for i, g in enumerate(groups):
        run_session(g, CELL_TIMEOUT_S, probe)
        if i % 100 == 0:
            print(f"[{i}/{len(groups)}] {time.time() - t0:.0f}s", file=sys.stderr, flush=True)

    # classify
    for c in sample:
        r = c.get("res") or {"error": "NOT_RUN", "denied": [], "final": False, "nonjson_state": []}
        if c["gate"]:
            c["verdict"], c["klass"] = "fail", "gate_refused:" + re.sub(r"\d+", "N", c["gate"])[:70]
        elif "cpython_syntax_error" in c["static"]:
            # not Python even for CPython (truncated or prose-wrapped model output): a corpus defect, not a Monty gap
            c["verdict"], c["klass"] = "orig_also_errored", "invalid_python_in_cpython"
        elif r["error"] == "SKIPPED_AFTER_TIMEOUT":
            c["verdict"], c["klass"] = "skipped", "after_session_timeout"
        elif r.get("terminal_timeout"):
            c["verdict"], c["klass"] = "fail", "timeout_%ds" % CELL_TIMEOUT_S
        elif r["error"]:
            k = classify_runtime(r["error"], r["denied"])
            # an error the ORIGINAL run also produced is a cell defect, not a Monty gap
            c["verdict"] = "orig_also_errored" if c.get("orig_error") else "fail"
            c["klass"] = k
        elif r["denied"]:
            c["verdict"], c["klass"] = "fail", "repl_callable_denied:" + sorted({d.get("name", "?") for d in r["denied"]})[0]
        elif r["nonjson_state"]:
            c["verdict"], c["klass"] = "pass_state_loss", "nonjson_state"
        else:
            c["verdict"], c["klass"] = "pass", ""


    HARD = ("generator", "class_inheritance", "absent_name:", "repl_callable:", "cpython_syntax_error", "match_statement")

    def hard(c):
        return any(f.startswith(HARD) for f in c["static"])

    def share(cs):
        n = len(cs)
        cnt = collections.Counter(c["verdict"] for c in cs)
        det = n - cnt["orig_also_errored"] - cnt["skipped"]
        w = lambda f: sum(c["occurrences"] for c in cs if f(c))
        wdet = sum(c["occurrences"] for c in cs if c["verdict"] not in ("orig_also_errored", "skipped"))
        return {"n_unique": n, "n_occurrences": sum(c["occurrences"] for c in cs), "pass": cnt["pass"],
                "pass_state_loss": cnt["pass_state_loss"], "fail": cnt["fail"],
                "indeterminate_orig_errored_or_skipped": cnt["orig_also_errored"] + cnt["skipped"],
                "share_unchanged_strict": round(cnt["pass"] / det, 3) if det else None,
                "share_unchanged_incl_state_loss": round((cnt["pass"] + cnt["pass_state_loss"]) / det, 3) if det else None,
                "share_unchanged_strict_occurrence_weighted": round(w(lambda c: c["verdict"] == "pass") / wdet, 3) if wdet else None}

    strata = {}
    for name, f in [("A_tap_all", lambda c: c["source"] == "A_tap"),
                    ("A_tap_nonfixture", lambda c: c["source"] == "A_tap" and not c["fixture"]),
                    ("A_tap_fixture", lambda c: c["source"] == "A_tap" and c["fixture"]),
                    ("B_all_labelled", lambda c: c["source"] != "A_tap")]:
        strata[name] = share([c for c in sample if f(c)])
    for role in sorted({c["role"] for c in sample if c["source"] != "A_tap"}):
        strata["B_role:" + role] = share([c for c in sample if c["source"] != "A_tap" and c["role"] == role])
    strata["ALL_nonfixture"] = share([c for c in sample if not c["fixture"]])
    classes = collections.Counter()
    for c in sample:
        if c["verdict"] in ("fail", "pass_state_loss"):
            classes[(c["source"] == "A_tap" and not c["fixture"], c["klass"].split(":")[0] if not c["klass"].startswith(("repl_callable", "module_missing", "name_absent")) else c["klass"])] += 1
    flagstats = collections.defaultdict(lambda: [0, 0])
    for c in sample:
        if c["verdict"] in ("orig_also_errored", "skipped"):
            continue
        for f in c["static"]:
            flagstats[f][0] += 1
            flagstats[f][1] += c["verdict"] == "pass"
    conf = collections.Counter()
    for c in sample:
        if c["verdict"] in ("orig_also_errored", "skipped"):
            continue
        conf[("static_in_subset" if not hard(c) else "static_out", "monty_runs" if c["verdict"] in ("pass", "pass_state_loss") else "monty_fails")] += 1
    summary = {"strata": strata,
               "failure_classes_unique_cells": {("nonfixture_tap|" if k[0] else "other|") + k[1]: v for k, v in classes.most_common()},
               "static_flag_vs_pass": {f: {"cells": a, "pass": b} for f, (a, b) in sorted(flagstats.items())},
               "static_screen_confusion": {f"{a}|{b}": n for (a, b), n in conf.items()}}

    # emit (no code bodies in the json: sha, flags, class, 120-char head only)
    def row(c):
        return {k: c.get(k) for k in ("source", "role", "session", "turn", "sha", "fixture", "occurrences", "gate",
                                      "static", "verdict", "klass", "orig_error")} | {
            "head": c["code"][:100].replace("\n", " \\n "), "err": ((c.get("res") or {}).get("error") or "")[:160],
            "ms": (c.get("res") or {}).get("ms")}

    out = {
        "run": {"date": time.strftime("%Y-%m-%d"), "seed": args.seed, "cell_timeout_s": CELL_TIMEOUT_S,
                "max_cells": args.max_cells, "raw_counts": {f"{a}|{b}": n for (a, b), n in sorted(raw_counts.items())},
                "unique_cells": len(uniq), "sampled": len(sample)},
        "summary": summary,
        "cells": [row(c) for c in sample],
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    probe.close() if hasattr(probe, "close") else None
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
