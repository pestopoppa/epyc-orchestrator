#!/usr/bin/env python3
"""HIL-LB1: score Locals' loop detectors (token-period, paraphrase, in-grammar with grace) against the
Who&When Pro labelled loop set, beside the existing L2 loop detector and intake-2395's repeated-call
metric; then the constrained-output evidence (fixtures + recorded inference-tap replies).

Read-only, zero inference, deterministic. Outputs only aggregate counts (and public Who&When ids);
recorded tap replies are private and contribute counts only. Paths are arguments; an absent set is
reported as absent, never faked.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.trace.l2 import loop_detectors as LD  # noqa: E402
from src.trace.l2.detectors import DetectorConfig, detect  # noqa: E402
from src.trace.l2.loaders import _pro_steps, load_claude_session  # noqa: E402
from src.trace.l2.model import L2Trace  # noqa: E402

DEFAULT_TOKENIZER = "/mnt/raid0/llm/hf-models/Qwen3-4B-Instruct-2507/tokenizer.json"
LOOP_MODE = "A.4"


# ------------------------------------------------------------------------------ scoring helpers
def score(pred: list[bool], gold: list[bool]) -> dict:
    tp = sum(p and g for p, g in zip(pred, gold))
    fp = sum(p and not g for p, g in zip(pred, gold))
    fn = sum((not p) and g for p, g in zip(pred, gold))
    tn = len(gold) - tp - fp - fn
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    return {"n": len(gold), "n_pos": tp + fn, "n_neg": fp + tn, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": None if prec is None else round(prec, 4),
            "recall": None if rec is None else round(rec, 4), "f1": round(f1, 4),
            "fp_rate_on_negatives": round(fp / (fp + tn), 4) if fp + tn else None}


# ------------------------------------------------------------------------------ Who&When Pro
def _units_from_steps(steps) -> list[LD.CallUnit]:
    return [LD.CallUnit(s.tool, s.args if s.args is not None else s.args_raw)
            for s in steps if s.kind == "tool_call"]


def _reply_texts(traj: list[dict]) -> list[str]:
    """One text per agent turn: reasoning + emitted call (action/code) or agent output + tool calls."""
    out = []
    for x in traj:
        k = x.get("kind")
        if k == "action":
            body = x.get("code") if "code" in x else x.get("action")
            out.append(f"{x.get('reasoning') or ''}\n{body if isinstance(body, str) else json.dumps(body)}")
        elif k == "agent":
            o = x.get("output") if isinstance(x.get("output"), str) else ""
            out.append(f"{o}\n{x.get('tool_actions') or ''}")
    return out


def _drop_exempt(t: L2Trace, exempt: list[bool]) -> L2Trace:
    it = iter(exempt)
    keep = [s for s in t.steps if s.kind != "tool_call" or not next(it)]
    return L2Trace(trace_id=t.trace_id, task_key=t.task_key, outcome=t.outcome, role=t.role,
                   source=t.source, steps=keep)


DETECTORS = (
    "l2_loop_existing", "l2_loop_existing_exempt",
    "repeated_call_2395_ge1", "repeated_call_2395_ge1_exempt",
    "token_period_stream", "token_period_stream_exempt",
    "paraphrase_stream_exempt", "paraphrase_norm_stream_exempt",
    "locals_tools_guard_stream_exempt",
    "in_grammar_stream_exempt", "no_tools_stream_exempt",
    "token_period_reply", "paraphrase_reply", "in_grammar_reply",
)


def whowhen_rows(path: Path, tok) -> list[dict]:
    cfg = DetectorConfig()
    rows = []
    for line in path.read_text().splitlines():
        r = json.loads(line)
        gt = json.loads(r["ground_truth"])
        traj = json.loads(r["trajectory"])
        t = L2Trace(trace_id=r["id"], task_key=r["id"], outcome="fail", source="whowhen_pro",
                    role=r.get("framework"), steps=_pro_steps(traj))
        units = _units_from_steps(t.steps)
        ex = LD.test_after_edit_exempt(units)
        fires: dict[str, int | None] = {}   # detector -> 0-based call index of the first fire (or -1 if trace-level)

        def l2_first(tr):
            ds = [d for d in detect(tr, cfg) if d.category == "loop"]
            if not ds:
                return None
            tool_steps = [s.idx for s in tr.steps if s.kind == "tool_call"]
            return tool_steps.index(ds[0].step_idx) if ds[0].step_idx in tool_steps else -1

        fires["l2_loop_existing"] = l2_first(t)
        fires["l2_loop_existing_exempt"] = l2_first(_drop_exempt(t, ex))
        rep = LD.repeated_calls(units)
        rep_ex = LD.repeated_calls(units, ex)
        fires["repeated_call_2395_ge1"] = rep[0] if rep else None
        fires["repeated_call_2395_ge1_exempt"] = rep_ex[0] if rep_ex else None

        def stream_fire(exempt, profile=None, para=None):
            text, kept = LD.action_stream(units, exempt)
            if para is not None:
                k = LD.first_line_fire(text, normalize=para)
                if k is None:
                    return None
                # map the k-th qualifying line back to its call
                q = [i for i, ln in zip(kept, text.splitlines()) if len(ln.strip()) >= 12]
                return q[k - 1]
            toks = tok(text)
            n = LD.first_period_fire(toks, profile)
            if n is None:
                return None
            # call index whose line contains token n-1: count newlines in the decoded prefix
            prefix_lines = _prefix_line_count(text, toks, n, tok)
            return kept[min(prefix_lines, len(kept) - 1)] if kept else None

        fires["token_period_stream"] = stream_fire(None, LD.TOOLS)
        fires["token_period_stream_exempt"] = stream_fire(ex, LD.TOOLS)
        fires["paraphrase_stream_exempt"] = stream_fire(ex, para=False)
        fires["paraphrase_norm_stream_exempt"] = stream_fire(ex, para=True)
        a, b = fires["token_period_stream_exempt"], fires["paraphrase_stream_exempt"]
        fires["locals_tools_guard_stream_exempt"] = min(x for x in (a, b) if x is not None) if (a is not None or b is not None) else None
        fires["in_grammar_stream_exempt"] = stream_fire(ex, LD.IN_GRAMMAR)
        fires["no_tools_stream_exempt"] = stream_fire(ex, LD.NO_TOOLS)

        replies = _reply_texts(traj)

        def reply_fire(fn):
            for i, txt in enumerate(replies):
                if fn(txt):
                    return i
            return None

        fires["token_period_reply"] = reply_fire(lambda s: LD.first_period_fire(tok(s), LD.TOOLS) is not None)
        fires["paraphrase_reply"] = reply_fire(lambda s: LD.first_line_fire(s) is not None)
        fires["in_grammar_reply"] = reply_fire(lambda s: LD.first_period_fire(tok(s), LD.IN_GRAMMAR) is not None)
        rows.append({"id": r["id"], "framework": r.get("framework"), "mode": gt.get("mode"),
                     "gold_step": gt.get("step"), "n_calls": len(units), "n_exempt": sum(ex),
                     "n_repeats": len(rep), "n_repeats_exempt": len(rep_ex), "fires": fires})
    return rows


def _prefix_line_count(text: str, toks, n: int, tok) -> int:
    """Number of complete lines before the token at position n-1 (tokenizer-agnostic: binary search
    on the shortest text prefix whose token count reaches n)."""
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi) // 2
        if len(tok(text[:mid])) >= n:
            hi = mid
        else:
            lo = mid + 1
    return text[:max(lo - 1, 0)].count("\n")


def whowhen_table(rows: list[dict]) -> dict:
    pops = {"all": rows, "alfagent": [r for r in rows if r["framework"] == "alfagent"]}
    out: dict = {"scope": "injected behaviour, not natural frequency; positives = mode A.4 (all in alfagent)",
                 "per_framework_N": dict(Counter(r["framework"] for r in rows)),
                 "loop_N": sum(r["mode"] == LOOP_MODE for r in rows), "populations": {}}
    for name, rs in pops.items():
        gold = [r["mode"] == LOOP_MODE for r in rs]
        tab = {"prevalence_baseline (flag all)": score([True] * len(rs), gold)}
        for d in DETECTORS:
            tab[d] = score([r["fires"][d] is not None for r in rs], gold)
        for k in (2, 3, 5):
            tab[f"repeated_call_2395_ge{k}_exempt (descriptive sweep)"] = score([r["n_repeats_exempt"] >= k for r in rs], gold)
        out["populations"][name] = tab
    # timing vs the injected step on positives (alfagent: call index i = step_number i+1)
    timing = {}
    pos = [r for r in rows if r["mode"] == LOOP_MODE]
    for d in DETECTORS:
        c = Counter()
        for r in pos:
            f = r["fires"][d]
            if f is None:
                c["no_fire"] += 1
            elif f < 0 or r["gold_step"] is None:
                c["trace_level"] += 1
            elif f + 1 < int(r["gold_step"]):
                c["before_injected_step"] += 1
            else:
                c["at_or_after_injected_step"] += 1
        timing[d] = dict(c)
    out["positives_fire_timing"] = timing
    out["exemption"] = {"traces_with_any_exempt_call": sum(r["n_exempt"] > 0 for r in rows),
                        "exempt_calls_total": sum(r["n_exempt"] for r in rows),
                        "alfagent_traces_with_any_exempt_call": sum(r["n_exempt"] > 0 for r in pops["alfagent"])}
    out["missed_positive_ids_by_token_period_stream_exempt"] = sorted(
        r["id"] for r in pos if r["fires"]["token_period_stream_exempt"] is None)[:60]
    return out


# ------------------------------------------------------------------------------ constrained output
def _td1_obj(i: int, vary: bool) -> dict:
    h = hashlib.sha256(f"td1-{i}".encode()).digest()
    pa = round(h[0] / 255, 2) if vary else 0.5
    return {"noul": False, "probabilities": {"A": pa, "B": round(1 - pa, 2)}, "confidence": 1.0}


def constrained_fixtures() -> dict[str, tuple[str, bool]]:
    """name -> (text, is_degenerate). Deterministic. Legitimate shapes end; degenerate ones run on."""
    fx: dict[str, tuple[str, bool]] = {}
    fx["td1_24q_varying"] = (json.dumps({"answers": [_td1_obj(i, True) for i in range(24)]}, indent=2), False)
    fx["td1_24q_identical"] = (json.dumps({"answers": [_td1_obj(i, False) for i in range(24)]}, indent=2), False)
    fx["grid_64_identical_rows"] = ("[\n" + ",\n".join(["  [0, 0, 0, 0, 0, 0, 0, 0]"] * 64) + "\n]", False)
    fx["identical_obj_x40_then_close"] = (json.dumps([_td1_obj(0, False)] * 40, indent=2), False)
    one = json.dumps(_td1_obj(0, False), indent=2)
    fx["runaway_identical_obj_to_limit"] = ("[\n" + ",\n".join([one] * 400), True)
    legit = json.dumps({"answers": [_td1_obj(i, True) for i in range(12)]}, indent=2)[:-3]
    fx["legit_prefix_then_runaway"] = (legit + ",\n" + ",\n".join(["    " + json.dumps(_td1_obj(3, True))] * 300), True)
    return fx


def _current_guard_fire(chunks: list[str]) -> int | None:
    """The live free-form guard as wired in inference.py: every 50 chunks on the accumulated text."""
    from src.llm_primitives.inference import _detect_streaming_repetition
    acc = []
    for i, c in enumerate(chunks, 1):
        acc.append(c)
        if i % 50 == 0 and _detect_streaming_repetition("".join(acc)):
            return i
    return None


def constrained_table(tok, max_tokens: int = 4096) -> dict:
    out = {}
    for name, (text, degenerate) in constrained_fixtures().items():
        rt = LD.regex_tokenize(text)
        # chunk = one regex token (the tap streams roughly one token per chunk)
        toks = tok(text)[:max_tokens]
        out[name] = {
            "degenerate": degenerate, "n_tokens": len(toks),
            "current_guard_fire_chunk": _current_guard_fire(rt),
            "tools_fire_token": LD.first_period_fire(toks, LD.TOOLS),
            "in_grammar_fire_token": LD.first_period_fire(toks, LD.IN_GRAMMAR),
            "no_tools_fire_token": LD.first_period_fire(toks, LD.NO_TOOLS),
        }
    sweep = {}
    for rows in (16, 32, 48, 64, 96, 128):
        for shape, row in (("grid_row", "  [0, 0, 0, 0, 0, 0, 0, 0]"),
                           ("compact_obj", "  " + json.dumps(_td1_obj(0, False)))):
            toks = tok("[\n" + ",\n".join([row] * rows) + "\n]")
            sweep[f"{shape}_x{rows}"] = {"n_tokens": len(toks),
                                         "in_grammar_fire_token": LD.first_period_fire(toks, LD.IN_GRAMMAR),
                                         "no_tools_fire_token": LD.first_period_fire(toks, LD.NO_TOOLS)}
    out["legit_identical_rows_sweep"] = sweep
    return out


def tap_replies(paths: list[Path]):
    """Yield (request_id, chunks) for each recorded reply, deduplicated by request id across files.
    Streamed replies are their ordered chunk texts; non-streamed ones are one 'response' text."""
    seen: set[str] = set()
    for p in paths:
        chunks: dict[str, list[str]] = defaultdict(list)
        resp: dict[str, str] = {}
        with open(p, errors="replace") as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                e, rid = d.get("event"), d.get("request_id")
                if not rid:
                    continue
                if e == "chunk" and isinstance(d.get("text"), str):
                    chunks[rid].append(d["text"])
                elif e == "response" and isinstance(d.get("text"), str):
                    resp[rid] = d["text"]
        for rid in sorted(set(chunks) | set(resp)):
            if rid in seen:
                continue
            seen.add(rid)
            yield rid, (chunks[rid] if rid in chunks else [resp[rid]])


def _json_shaped(text: str) -> bool:
    s = text.strip()
    if s.startswith("```json"):
        s = s[7:].strip().rstrip("`").strip()
    return s[:1] in ("{", "[")


def tap_table(paths: list[Path], tok) -> dict:
    present = [p for p in paths if p.exists()]
    if not present:
        return {"status": "absent"}
    c: dict[str, Counter] = {"all": Counter(), "json_shaped": Counter()}
    ntoks = []
    for _, chunks in tap_replies(present):
        text = "".join(chunks)
        if not text.strip():
            continue
        toks = tok(text)
        ntoks.append(len(toks))
        groups = ["all"] + (["json_shaped"] if _json_shaped(text) else [])
        res = {
            "current_guard": (_current_guard_fire(chunks) is not None) if len(chunks) > 1 else
            (_current_guard_fire(LD.regex_tokenize(text)) is not None),
            "tools_token_period": LD.first_period_fire(toks, LD.TOOLS) is not None,
            "paraphrase": LD.first_line_fire(text) is not None,
            "in_grammar": LD.first_period_fire(toks, LD.IN_GRAMMAR) is not None,
            "no_tools": LD.first_period_fire(toks, LD.NO_TOOLS) is not None,
            "periodic_tail_at_end_in_grammar": LD.looping(toks, LD.IN_GRAMMAR.reps, 256, LD.IN_GRAMMAR.min_span,
                                                           LD.IN_GRAMMAR.short_span),
        }
        if res["current_guard"] and len(chunks) > 1:
            fire = _current_guard_fire(chunks)
            res["current_guard_fire_is_last_chunk (cut live)"] = fire == len(chunks)
            res["current_guard_fire_reply_continued (guard inactive)"] = fire < len(chunks)
            body = text.strip()
            if body.startswith("```"):
                body = body.strip("`").removeprefix("json").strip()
            try:
                json.loads(body)
                res["current_guard_fire_reply_is_valid_json"] = True
            except ValueError:
                res["current_guard_fire_reply_is_valid_json"] = False
        res["reply_tokens_ge_4096"] = len(toks) >= 4096
        for g in groups:
            c[g]["n"] += 1
            for k, v in res.items():
                c[g][k] += bool(v)
    ntoks.sort()
    q = lambda f: ntoks[min(int(f * len(ntoks)), len(ntoks) - 1)] if ntoks else None  # noqa: E731
    return {"files": [p.name for p in present], "label": "none (natural recorded replies; firing counts only)",
            "json_shaped_is_a_proxy": "reply text starts with { or [ (request grammar flag is not recorded in the tap)",
            "counts": {g: dict(v) for g, v in c.items()},
            "reply_tokens_p50_p90_p99_max": [q(0.5), q(0.9), q(0.99), ntoks[-1] if ntoks else None]}


# ------------------------------------------------------------------------------ coding sessions
def claude_session_census(root: Path, sample: int, tok) -> dict:
    """Unlabelled natural coding sessions (Claude Code format): how often each tool-level detector
    fires with and without the test-after-edit exemption. Deterministic sample: files ordered by
    sha256 of their path relative to ``root``; counts only, no content leaves this function."""
    files = sorted((p for p in root.rglob("*.jsonl") if p.stat().st_size < 30_000_000),
                   key=lambda p: hashlib.sha256(str(p.relative_to(root)).encode()).hexdigest())[:sample]
    cfg = DetectorConfig()
    c = Counter()
    for f in files:
        t = load_claude_session(f)
        units = _units_from_steps(t.steps)
        if not units:
            c["sessions_without_calls"] += 1
            continue
        c["sessions_with_calls"] += 1
        c["calls"] += len(units)
        ex = LD.test_after_edit_exempt(units)
        c["exempt_calls"] += sum(ex)
        c["sessions_with_exempt_call"] += any(ex)
        kinds = Counter(LD.classify_call(u) for u in units)
        c["mutating_calls"] += kinds["mutating"]
        c["verifying_calls"] += kinds["verifying"]
        res = {
            "repeated_call_ge1": (bool(LD.repeated_calls(units)), bool(LD.repeated_calls(units, ex))),
            "l2_loop_existing": (any(d.category == "loop" for d in detect(t, cfg)),
                                 any(d.category == "loop" for d in detect(_drop_exempt(t, ex), cfg))),
            "token_period_stream": (LD.first_period_fire(tok(LD.action_stream(units)[0]), LD.TOOLS) is not None,
                                    LD.first_period_fire(tok(LD.action_stream(units, ex)[0]), LD.TOOLS) is not None),
        }
        for k, (raw, exm) in res.items():
            c[f"{k}|fires_raw"] += raw
            c[f"{k}|fires_exempt"] += exm
            c[f"{k}|flipped_by_exemption"] += raw and not exm
    return {"n_files": len(files), "label": "none (natural sessions; firing counts only)", "counts": dict(sorted(c.items()))}


# ------------------------------------------------------------------------------ markdown
def to_markdown(res: dict) -> str:
    L = ["# HIL-LB1 loop-detector scoring", "",
         f"Tokenizer: `{res['tokenizer']['kind']}` sha256 `{res['tokenizer'].get('sha256', '-')[:16]}`. "
         "Zero inference, deterministic.", ""]
    ww = res.get("whowhen_pro")
    if ww:
        L += [f"## Who&When Pro text split — {ww['scope']}", "",
              f"Loop N={ww['loop_N']}; frameworks: {ww['per_framework_N']}.", ""]
        for pop, tab in ww["populations"].items():
            L += [f"### population: {pop}", "",
                  "| detector | N | pos | TP | FP | FN | TN | precision | recall | F1 | FP rate on neg |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
            for d, s in tab.items():
                L.append(f"| {d} | {s['n']} | {s['n_pos']} | {s['tp']} | {s['fp']} | {s['fn']} | {s['tn']} | "
                         f"{s['precision']} | {s['recall']} | {s['f1']} | {s['fp_rate_on_negatives']} |")
            L.append("")
        L += ["### fire timing on positives (vs injected step)", "", "| detector | timing |", "|---|---|"]
        for d, t in ww["positives_fire_timing"].items():
            L.append(f"| {d} | {t} |")
        L += ["", f"Exemption: {ww['exemption']}", ""]
    ct = res.get("constrained_fixtures")
    if ct:
        L += ["## Constrained-output fixtures", "",
              "| fixture | degenerate | tokens | current guard (chunk) | tools (tok) | in_grammar (tok) | no_tools (tok) |",
              "|---|---|---|---|---|---|---|"]
        for k, v in ct.items():
            if k == "legit_identical_rows_sweep":
                continue
            L.append(f"| {k} | {v['degenerate']} | {v['n_tokens']} | {v['current_guard_fire_chunk']} | "
                     f"{v['tools_fire_token']} | {v['in_grammar_fire_token']} | {v['no_tools_fire_token']} |")
        L += ["", "Legitimate identical-row sweep (closed arrays; any fire here cuts a legitimate reply):", "",
              "| shape | tokens | in_grammar (tok) | no_tools (tok) |", "|---|---|---|---|"]
        for k, v in ct["legit_identical_rows_sweep"].items():
            L.append(f"| {k} | {v['n_tokens']} | {v['in_grammar_fire_token']} | {v['no_tools_fire_token']} |")
        L.append("")
    tp = res.get("inference_tap")
    if tp:
        L += ["## Recorded EPYC replies (inference tap, unlabelled)", "", f"```\n{json.dumps(tp, indent=1)}\n```", ""]
    cs = res.get("coding_sessions")
    if cs:
        L += ["## Natural coding sessions (unlabelled) — test-after-edit exemption effect", "",
              f"```\n{json.dumps(cs, indent=1)}\n```", ""]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--whowhen-pro", help="Who&When Pro text.jsonl")
    ap.add_argument("--tap", action="append", default=[], help="inference_tap_events.jsonl (repeatable)")
    ap.add_argument("--claude-sessions", help="dir of Claude-Code session *.jsonl (recursive)")
    ap.add_argument("--claude-sample", type=int, default=150)
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER, help="tokenizer.json; 'regex' for the built-in")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--belief-out", help="VB-HILLB1-WRITE: also write hil_lb1_loop_score.beliefs.jsonl "
                    "into this dir via the root belief-kernel writer (opt-in; needs --whowhen-pro and the "
                    "real tokenizer)")
    ap.add_argument("--run-id", default="hil-lb1-run", help="run id for the belief sidecar")
    ap.add_argument("--epyc-root", help="root checkout (default: $EPYC_ROOT, epyc-root-main view, /workspace)")
    a = ap.parse_args(argv)
    if a.tokenizer != "regex" and Path(a.tokenizer).exists():
        tok = LD.hf_tokenizer(a.tokenizer)
        tk = {"kind": a.tokenizer, "sha256": hashlib.sha256(Path(a.tokenizer).read_bytes()).hexdigest()}
    else:
        tok, tk = LD.regex_tokenize, {"kind": "regex", "sha256": "-"}
    res: dict = {"tokenizer": tk, "profiles": {k: vars(v) for k, v in LD.PROFILES.items()},
                 "source": "Generated-Computers/Locals@0d39e16194e2c123ce10695e45619111eb0e45b0 locals/server.py"}
    if a.whowhen_pro and Path(a.whowhen_pro).exists():
        res["whowhen_pro"] = whowhen_table(whowhen_rows(Path(a.whowhen_pro), tok))
    else:
        res["whowhen_pro"] = None
    res["constrained_fixtures"] = constrained_table(tok)
    res["inference_tap"] = tap_table([Path(p) for p in a.tap], tok) if a.tap else None
    res["coding_sessions"] = (claude_session_census(Path(a.claude_sessions), a.claude_sample, tok)
                              if a.claude_sessions and Path(a.claude_sessions).is_dir() else None)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "hil_lb1_loop_score.json").write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    (out / "hil_lb1_loop_score.md").write_text(to_markdown(res))
    if a.belief_out:
        return write_belief_sidecar(a, out / "hil_lb1_loop_score.json")
    return 0


def write_belief_sidecar(a: argparse.Namespace, results: Path) -> int:
    """VB-HILLB1-WRITE: emit the sidecar AFTER the result is on disk; a refusal is loud (exit 3)."""
    import subprocess

    from scripts.lib.epyc_root import load_adapter

    ref = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                         text=True).stdout.strip()
    try:
        Path(a.belief_out).mkdir(parents=True, exist_ok=True)
        side = load_adapter("hil_lb1_loop_score", a.epyc_root).write_belief_measurements(
            a.belief_out, run_id=a.run_id,
            producer="epyc-orchestrator/scripts/trace/hil_lb1_loop_score.py", results_path=results,
            labelled_set_path=a.whowhen_pro if a.whowhen_pro and Path(a.whowhen_pro).is_file() else None,
            code_ref=ref if len(ref) == 40 else None)
    except Exception as exc:  # noqa: BLE001 - result exists; surface the refusal, never hide it
        print(f"!!! BELIEF CAPTURE FAILED [hil-lb1]: REFUSED {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3
    print(f"belief sidecar: {side}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
