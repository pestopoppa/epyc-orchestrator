#!/usr/bin/env python3
"""D-RI-PREVIEW-1.2: composite multi-turn research fixture for the REPL history-rendering comparator.

R5 (single-turn, one 50-item batch) hit the ceiling on all three renderings because the model could always
iterate the full `batch` variable in code. This fixture moves the question to the HISTORY: a 7-question composite
research task over a LOCAL, frozen, seed-generated corpus (68 docs, no web). Sub-question k is revealed at turn k.
Q1-Q3 (turns 1-3) find early facts (lead, site+region, approved budget of one project); Q4-Q6 are dependent lookups
that need those facts; Q7 asks FINAL(dict) combining all of them. The model never has to "answer" Q1-Q3 anywhere
except by what it prints/keeps in the REPL, so whether the early facts are still usable at turns 4-7 depends on how
the earlier OUTPUTS are rendered into the history (the renderings below mirror the R5 cells):

  current       : repr(output)[:80] + "..."                          (no true length; today's get_state())
  bounded_k3    : output(len=N lines, first 3: [...], last 3: [...]) (true length stated)
  half_first25  : first ceil(N/2) lines, length not stated           (R5's "half" cell)
  full          : verbatim. PILOT CONTROL ONLY, never a measurement cell.

The newest output is always shown in full; only older outputs are rendered. Code of every turn is always shown in
full (so a model that RETYPES a fact into its code, or keeps it in a variable, is robust: that is the competence under
test). Scoring is an end-state oracle on the final dict (exact field match, no LLM judge). Everything is a pure
function of the episode index: the same episodes are used in every rendering cell (paired design).
"""
from __future__ import annotations

import itertools
import json
import os
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path

N_PROJECTS = 16
RENDERINGS = ("current", "bounded_k3", "half_first25")
ALL_RENDERINGS = RENDERINGS + ("full",)
MAX_TURNS = 9
OBS_CAP = 3500
CODE_CAP = 1500
HIT_CAP = 24
KEYS = ("lead", "site", "region", "budget_k", "lead_rate", "mult_pct", "loaded_budget_k")
EARLY_KEYS = ("lead", "site", "region", "budget_k")  # facts found in turns 1-3

PROJECTS = ["KESTREL", "HALCYON", "MERIDIAN", "OBSIDIAN", "TAMARIND", "LUMEN", "CORVID", "PELICAN", "ARGENT", "SOLSTICE",
            "BRACKEN", "VESPER", "QUARRY", "NIMBUS", "CINDER", "FALCONET", "MARROW", "GLACIER", "TUNDRA", "ZEPHYR",
            "OSPREY", "LANTERN", "WILLOW", "EMBER"]
FIRST = ["Ines", "Tomas", "Marta", "Oskar", "Priya", "Lucan", "Hana", "Dmitri", "Sade", "Ravi", "Elin", "Bruno", "Yara",
         "Felix", "Noor", "Anselm", "Mei", "Jonas", "Talia", "Viktor"]
LAST = ["Calloway", "Weir", "Okafor", "Lindqvist", "Brandt", "Moreau", "Tanaka", "Haddad", "Novak", "Ferreira", "Keane",
        "Sorensen", "Abara", "Whitlock", "Duarte", "Kovac", "Mbeki", "Ostrow", "Pryce", "Salazar"]
REGIONS = ["WEST", "EAST", "NORTH", "SOUTH"]
# target position (0-15) of the asked project inside each list: (i*a + b) % 16 with odd a is a permutation, so any
# 16 consecutive episodes cover every position exactly once per list.
POS_COEF = {"roster": (7, 2), "site": (5, 9), "budget": (3, 4)}


def _position(i: int, kind: str) -> int:
    a, b = POS_COEF[kind]
    return (i * a + b) % N_PROJECTS


def build_episode(i: int) -> dict:
    rng = random.Random(f"dripreview12:{i}")
    names = rng.sample(PROJECTS, N_PROJECTS)
    people = rng.sample([f"{f} {l}" for f, l in itertools.product(FIRST, LAST)], N_PROJECTS)
    target = rng.randrange(N_PROJECTS)  # index into names/people of the asked project
    mult = {r: rng.randrange(88, 141, 3) for r in REGIONS}
    proj = []
    for k, name in enumerate(names):
        budget = rng.randrange(200, 1000, 10)
        proj.append({
            "name": name, "lead": people[k], "prev_lead": people[(k + 5) % N_PROJECTS],
            "site": f"{rng.choice('KMRTW')}{rng.choice('ABCDEFGH')}-{100 + k}",
            "prev_site": f"{rng.choice('KMRTW')}{rng.choice('ABCDEFGH')}-{200 + rng.randrange(80)}",
            "region": rng.choice(REGIONS), "budget": budget, "requested": budget + rng.choice([-1, 1]) * rng.randrange(40, 200, 10),
            "rate": rng.randrange(40, 121, 2), "grade": rng.choice("ABCD") + str(rng.randrange(1, 4)),
        })

    def ordered(kind: str) -> list[dict]:
        rest = [p for k, p in enumerate(proj) if k != target]
        rng.shuffle(rest)
        rest.insert(_position(i, kind), proj[target])
        return rest

    docs = []
    for rank, p in enumerate(ordered("roster")):
        docs.append({"id": f"roster-{rank:02d}", "kind": "roster", "title": f"Roster: {p['name']}",
                     "text": f"project {p['name']} | lead {p['lead']} | prev lead {p['prev_lead']}"})
    for rank, p in enumerate(ordered("site")):
        docs.append({"id": f"site-{rank:02d}", "kind": "site", "title": f"Site assignment: {p['name']}",
                     "text": f"project {p['name']} | site {p['site']} | region {p['region']} | prev site {p['prev_site']}"})
    for rank, p in enumerate(ordered("budget")):
        docs.append({"id": f"budget-{rank:02d}", "kind": "budget", "title": f"Budget line: {p['name']}",
                     "text": f"project {p['name']} | approved budget_k {p['budget']} | requested_k {p['requested']}"})
    plist = proj[:]
    rng.shuffle(plist)
    for rank, p in enumerate(plist):
        docs.append({"id": f"profile-{rank:02d}", "kind": "profile", "title": f"Profile: {p['lead']}",
                     "text": f"{p['lead']} | hourly_rate {p['rate']} | grade {p['grade']}"})
    for rank, r in enumerate(REGIONS):
        docs.append({"id": f"region-{rank:02d}", "kind": "region", "title": f"Region factor: {r}",
                     "text": f"region {r} | cost multiplier pct {mult[r]}"})
    t = proj[target]
    expected = {"lead": t["lead"], "site": t["site"], "region": t["region"], "budget_k": t["budget"], "lead_rate": t["rate"],
                "mult_pct": mult[t["region"]], "loaded_budget_k": t["budget"] * mult[t["region"]] // 100}
    P = t["name"]
    questions = [
        f"Q1: Who leads project {P}?",
        f"Q2: Which site hosts project {P}, and in which region is that site?",
        f"Q3: What is the approved budget of project {P} (budget_k)?",
        "Q4: What is the hourly rate of the lead of that project?",
        "Q5: What cost multiplier (percent) applies to the region of that project's site?",
        "Q6: What is that project's approved budget after applying that multiplier, in k, rounded down?",
        "Q7: Finish with FINAL(<dict>) with exactly these keys: lead, site, region, budget_k, lead_rate, mult_pct, "
        "loaded_budget_k (strings for lead/site/region, integers for the rest).",
    ]
    return {"schema": "epyc.repl_composite_episode.v1", "i": i, "project": P, "docs": docs, "questions": questions,
            "expected": expected,
            "positions": {k: _position(i, k) for k in POS_COEF},
            "distractors": {"lead": t["prev_lead"], "site": t["prev_site"], "budget_k": t["requested"]}}


def fixture_sha256(n: int = 30) -> str:
    import hashlib
    return hashlib.sha256(json.dumps([build_episode(i) for i in range(n)], sort_keys=True).encode()).hexdigest()


# ---------------------------------------------------------------------------------------------- renderings
def units(text: str) -> list[str]:
    """Rendering units of an output: its lines; a very long line (a printed list) is split between container elements."""
    out = []
    for ln in text.splitlines():
        if len(ln) > 240:
            out.extend(s for s in re.split(r"(?<=[\]\}\)]),\s*(?=[\[\{\(])", ln) if s)
        else:
            out.append(ln)
    return out


def render_obs(text: str, rendering: str, k: int = 3) -> str:
    """Render an OLDER output under `rendering` (the newest output is never passed here)."""
    if rendering == "full":
        return text if len(text) <= OBS_CAP else text[:OBS_CAP] + "\n[...cut]"
    if rendering == "current":
        p = repr(text)
        return p if len(p) <= 80 else p[:80] + "..."
    u = units(text)
    n = len(u)
    if rendering.startswith("bounded"):
        if n <= 2 * k:
            return "\n".join(u)
        return f"output(len={n} lines, first {k}: {u[:k]!r}, last {k}: {u[-k:]!r})"
    if rendering == "half_first25":
        return "\n".join(u[:(n + 1) // 2])
    raise ValueError(rendering)


INTRO = """You are working in a Python REPL over a local document corpus (roster, site, budget, profile and region documents).
Already defined in the REPL: search(term) -> list of hit dicts {{id, title, text}} (case-insensitive substring match over title and text, at most {cap} hits), read(doc_id) -> str, and FINAL(x), which submits the final answer. Variables persist between turns. Outputs of earlier turns are shown below and may be abbreviated.

Questions so far (work them in the REPL; only the final dict is graded):
{qs}

History:
{hist}
This is turn {t} of at most {tmax}. Reply with exactly one ```python code block that will be run in the REPL.
FINAL is a function already defined in the REPL: call it, for example FINAL({{...}}). Never assign to it (not `FINAL = ...`)."""


def build_prompt(ep: dict, turns: list[dict], rendering: str, t: int, tmax: int = MAX_TURNS) -> str:
    qs = "\n".join(ep["questions"][: min(t, len(ep["questions"]))])
    parts = []
    for j, tr in enumerate(turns, 1):
        obs = tr["obs"] if tr["obs"].strip() else "(no output)"
        shown = (obs if len(obs) <= OBS_CAP else obs[:OBS_CAP] + "\n[...cut]") if j == len(turns) else render_obs(obs, rendering)
        code = tr["code"] if len(tr["code"]) <= CODE_CAP else tr["code"][:CODE_CAP] + "\n# [...cut]"
        parts.append(f"--- turn {j} code ---\n{code}\n--- turn {j} output ---\n{shown}")
    return INTRO.format(cap=HIT_CAP, qs=qs, hist=("\n".join(parts) + "\n") if parts else "(none yet)\n", t=t, tmax=tmax)


# ---------------------------------------------------------------------------------------------- REPL execution
_RUNNER = r'''
import json, sys, io, contextlib
spec = json.load(sys.stdin)
docs = spec["docs"]
def search(term):
    t = str(term).lower()
    return [{"id": d["id"], "title": d["title"], "text": d["text"]} for d in docs if t in (d["title"] + " " + d["text"]).lower()][:spec["hit_cap"]]
def read(doc_id):
    for d in docs:
        if d["id"] == str(doc_id):
            return d["title"] + "\n" + d["text"]
    raise KeyError("no such doc: %s" % (doc_id,))
finals = []
def FINAL(x):
    finals.append(x)
g = {"__name__": "__repl__", "search": search, "read": read, "FINAL": FINAL}
res = []
for code in spec["blocks"]:
    buf = io.StringIO(); err = None; n0 = len(finals)
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(code, "<cell>", "exec"), g)
    except BaseException as e:
        err = type(e).__name__ + ": " + str(e)[:200]
    assigned = g.get("FINAL") is not FINAL
    g["FINAL"] = FINAL
    res.append({"out": buf.getvalue()[:spec["obs_cap"] + 500], "err": err, "final": finals[n0:], "assigned": assigned})
print(json.dumps(res, default=repr))
'''


def exec_blocks(ep: dict, blocks: list[str], timeout_s: float = 15.0):
    """Re-execute every cell so far in one isolated child (deterministic, namespace persists); return per-cell results,
    or None on timeout/crash of the child."""
    payload = json.dumps({"docs": ep["docs"], "blocks": blocks, "hit_cap": HIT_CAP, "obs_cap": OBS_CAP})
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "r.py").write_text(_RUNNER)
        try:
            p = subprocess.run([sys.executable, "-I", str(Path(d) / "r.py")], input=payload, capture_output=True, text=True,
                               timeout=timeout_s, cwd=d, env={"PATH": os.environ.get("PATH", "")})
            return json.loads(p.stdout.strip().splitlines()[-1])
        except Exception:
            return None


_CODE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.S)


def play_episode(ep: dict, rendering: str, reply_fn, max_turns: int = MAX_TURNS) -> dict:
    """Run one episode. reply_fn(prompt, turn) -> reply text. Returns {final, how, turns:[{code,obs,reply}], n_turns}."""
    blocks: list[str] = []
    turns: list[dict] = []
    final, how = None, "no_final"
    for t in range(1, max_turns + 1):
        prompt = build_prompt(ep, turns, rendering, t, max_turns)
        reply = reply_fn(prompt, t)
        m = _CODE.search(reply)
        if not m:
            turns.append({"code": "", "obs": "Error: reply had no ```python code block.", "reply": reply})
            continue
        code = m.group(1)
        res = exec_blocks(ep, blocks + [code])
        if res is None:
            turns.append({"code": code, "obs": "Error: the cell timed out or crashed the REPL (cell discarded).", "reply": reply})
            continue
        blocks.append(code)
        r = res[-1]
        obs = r["out"] + (f"\nError: {r['err']}" if r["err"] else "")
        if r["assigned"]:
            obs += "\nError: FINAL is a function; call FINAL(x). Assigning to it records nothing."
        turns.append({"code": code, "obs": obs, "reply": reply})
        if r["final"]:
            final = r["final"][-1]
            how = "final" if isinstance(final, dict) else "bad_final"
            break
    return {"final": final if how == "final" else None, "how": how, "n_turns": len(turns), "turns": turns}


# ---------------------------------------------------------------------------------------------- scoring
def _norm(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return int(v) if float(v) == int(v) else v
    if isinstance(v, str):
        s = v.strip()
        return int(s) if re.fullmatch(r"-?\d+", s) else s.lower()
    return None


def classify(final, ep: dict) -> dict:
    """End-state oracle. cls: correct (all 7 fields) | partial (>=1) | wrong (0) | no_answer (final None)."""
    if not isinstance(final, dict):
        return {"cls": "no_answer", "fields": {}, "n_ok": 0, "early_ok": False, "distractor_picks": []}
    exp = ep["expected"]
    fields = {k: (k in final and _norm(final[k]) == _norm(exp[k])) for k in KEYS}
    n_ok = sum(fields.values())
    picks = [k for k, dv in ep["distractors"].items() if k in final and _norm(final[k]) == _norm(dv)]
    return {"cls": "correct" if n_ok == len(KEYS) else ("partial" if n_ok else "wrong"), "fields": fields, "n_ok": n_ok,
            "early_ok": all(fields[k] for k in EARLY_KEYS), "distractor_picks": picks}


# ---------------------------------------------------------------------------------------------- scripted reference policies
def _facts(prompt: str, P: str) -> dict:
    """Facts a scripted policy can read from the RENDERED history (plus any value retyped into earlier code)."""
    f = {}
    for key, val in re.findall(r"^(lead|site|region) = '(.*)'$", prompt, re.M):
        f[key] = val  # retyped into code (read first: later lookups depend on them)
    for key, val in re.findall(r"^(budget_k|lead_rate|mult_pct|loaded_budget_k) = (\d+)$", prompt, re.M):
        f[key] = int(val)
    pats = {
        "lead": rf"project {P} \| lead ([A-Za-z' -]+?) \| prev lead",
        "site+region": rf"project {P} \| site (\S+) \| region (\w+) \| prev site",
        "budget_k": rf"project {P} \| approved budget_k (\d+) \| requested_k",
        "loaded": r"LOADED_K=(\d+)",
    }
    m = re.search(pats["lead"], prompt)
    if m:
        f["lead"] = m.group(1)
    m = re.search(pats["site+region"], prompt)
    if m:
        f["site"], f["region"] = m.group(1), m.group(2)
    m = re.search(pats["budget_k"], prompt)
    if m:
        f["budget_k"] = int(m.group(1))
    if f.get("lead"):
        m = re.search(rf"{re.escape(f['lead'])} \| hourly_rate (\d+)", prompt)
        if m:
            f["lead_rate"] = int(m.group(1))
    if f.get("region"):
        m = re.search(rf"region {f['region']} \| cost multiplier pct (\d+)", prompt)
        if m:
            f["mult_pct"] = int(m.group(1))
    m = re.findall(pats["loaded"], prompt)
    if m:
        f["loaded_budget_k"] = int(m[-1])
    return f


def reference_policy(mode: str, P: str):
    """mode: 'reader' (relies on the rendered history only), 'retype' (copies each fact into its own code the turn after it
    appears), 'requery' (reader that re-runs search(P) when a needed early fact is no longer visible). Pure function of the
    prompt it is shown, so it proves what the RENDERING preserves; it never reads the episode."""
    show = "for h in search({q!r}): print(h)"

    def retype_lines(f):
        out = []
        for k in ("lead", "site", "region"):
            if k in f:
                out.append(f"{k} = {f[k]!r}")
        for k in ("budget_k", "lead_rate", "mult_pct", "loaded_budget_k"):
            if k in f:
                out.append(f"{k} = {f[k]}")
        return "\n".join(out) + ("\n" if out else "")

    def act(prompt: str, t: int) -> str:
        f = _facts(prompt, P)
        pre = retype_lines(f) if mode == "retype" else ""
        if t == 1:
            return show.format(q="roster")
        if t == 2:
            return pre + show.format(q="site")
        if t == 3:
            return pre + show.format(q="budget")
        if mode == "requery":
            if not all(k in f for k in ("lead", "site", "region", "budget_k")):
                return show.format(q=P)
            if "lead_rate" not in f:
                return show.format(q=f["lead"])
            if "mult_pct" not in f:
                return show.format(q="Region factor: " + f["region"])
            if "loaded_budget_k" not in f:
                return f"print('LOADED_K=%d' % ({f['budget_k']} * {f['mult_pct']} // 100))"
            t = 7
        if t == 4:
            return pre + (show.format(q=f["lead"]) if "lead" in f else "print('lead unknown')")
        if t == 5:
            return pre + (show.format(q="Region factor: " + f["region"]) if "region" in f else "print('region unknown')")
        if t == 6:
            ok = "budget_k" in f and "mult_pct" in f
            return pre + (f"print('LOADED_K=%d' % ({f['budget_k']} * {f['mult_pct']} // 100))" if ok else "print('LOADED_K=0')")
        d = {"lead": f.get("lead", "unknown"), "site": f.get("site", "unknown"), "region": f.get("region", "unknown"),
             "budget_k": f.get("budget_k", 0), "lead_rate": f.get("lead_rate", 0), "mult_pct": f.get("mult_pct", 0),
             "loaded_budget_k": f.get("loaded_budget_k", 0)}
        return f"FINAL({d!r})"

    return lambda prompt, t: "```python\n" + act(prompt, t) + "\n```"


if __name__ == "__main__":
    e = build_episode(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
    print(json.dumps({k: e[k] for k in ("i", "project", "questions", "expected", "positions", "distractors")}, indent=1))
    print(len(e["docs"]), "docs")
