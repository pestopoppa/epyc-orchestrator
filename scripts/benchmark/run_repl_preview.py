#!/usr/bin/env python3
"""D-RI-PREVIEW-1 model driver (queue row ``s5-dripreview1``).

Runs the bounded-preview comparator on served worker role(s): for every (role, rendering) cell it sends the
fixture task with that rendering of the REPL variable ``batch`` to the role's PRIVATE llama-server, executes
the reply the way the REPL would (a code block run with ``batch`` bound, ``FINAL(x)`` sets the answer; a reply
with no code block falls back to its last integer), and tallies ``correct / blind_rule_fit / retyping_loss /
other`` with ``repl_preview_fixture.classify``. The scorer and fixture are NOT changed here.

The spec (roles, renderings, n, temperature, decision rule) is frozen BEFORE any run in
``artifacts/queue-builders-3-20261021/dripreview1/PREDECLARED.md`` (root repo); this driver reads the numbers
from flags whose defaults equal that file, and records the flags it was run with in the result.

No new runtime flag: only script arguments. Never starts a server. Live mode refuses unless
``WINDOW_CONFIRMED=1`` and every ``--role NAME=URL@PID`` passes the port-ownership proof (``ss`` pid == PID,
loopback only, port not in ``--deny-ports``). ``--mock`` needs no server and no window (plumbing only).

  python scripts/benchmark/run_repl_preview.py --role worker_general=http://127.0.0.1:9461@12345 --out r.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("repl_preview_fixture", ROOT / "scripts" / "benchmark" / "repl_preview_fixture.py")
fx_mod = importlib.util.module_from_spec(_spec)
sys.modules["repl_preview_fixture"] = fx_mod
_spec.loader.exec_module(fx_mod)

RENDERINGS = ("current", "bounded_k3", "half_first25")
DEFAULT_N = 30
DEFAULT_TEMP = 0.7
DEFAULT_MAX_TOKENS = 700
# production stack ports (frontdoor/architect/ingest...): a private server must never be one of these
DEFAULT_DENY_PORTS = "8070,8071,8072,8080,8081,8082,8083,8090,8180,8181,8182,8000,8100"

PROMPT = """You are working in a Python REPL. The variable below is already defined in the REPL namespace.

REPL state:
{rendering}

Task: {task}

Reply with exactly one ```python code block that will be run in the REPL, ending with FINAL(<integer>)."""

_CODE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.S)
_RUNNER = """
import json, sys
batch = json.load(open(sys.argv[1]))
_out = []
def FINAL(x): _out.append(x)
try:
    exec(compile(open(sys.argv[2]).read(), "<reply>", "exec"), {"batch": batch, "FINAL": FINAL, "__name__": "__repl__"})
except BaseException as e:
    print(json.dumps({"error": type(e).__name__ + ": " + str(e)[:200]})); sys.exit(0)
print(json.dumps({"final": _out[-1] if _out else None}))
"""


def run_reply(text: str, batch: list[dict], timeout_s: float = 10.0):
    """(answer|None, how). Code block -> executed in an isolated child; else last integer in the text."""
    m = _CODE.search(text)
    if not m:
        ints = re.findall(r"-?\d+", text)
        return (int(ints[-1]) if ints else None), "text_int"
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "b.json").write_text(json.dumps(batch))
        (Path(d) / "c.py").write_text(m.group(1))
        (Path(d) / "r.py").write_text(_RUNNER)
        try:
            p = subprocess.run([sys.executable, "-I", str(Path(d) / "r.py"), str(Path(d) / "b.json"), str(Path(d) / "c.py")],
                               capture_output=True, text=True, timeout=timeout_s, cwd=d, env={"PATH": os.environ.get("PATH", "")})
            res = json.loads(p.stdout.strip().splitlines()[-1])
        except Exception as e:
            return None, "exec_failed:" + type(e).__name__
    if "error" in res:
        return None, "code_error"
    f = res["final"]
    if isinstance(f, bool) or not isinstance(f, (int, float, str)):
        return None, "bad_final"
    try:
        return int(float(f)), "code"
    except ValueError:
        return None, "bad_final"


def build_prompt(fx: dict, rendering: str) -> str:
    return PROMPT.format(rendering=fx["renderings"][rendering], task=fx["task"])


def chat(url: str, prompt: str, temperature: float, max_tokens: int, seed: int) -> str:
    body = json.dumps({"messages": [{"role": "user", "content": prompt}], "temperature": temperature, "max_tokens": max_tokens,
                       "seed": seed, "chat_template_kwargs": {"enable_thinking": False}, "cache_prompt": False}).encode()
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


def listening_pid(port: int):
    out = subprocess.run(["ss", "-H", "-ltnp", f"sport = :{port}"], capture_output=True, text=True).stdout
    m = re.search(r"pid=(\d+)", out)
    return int(m.group(1)) if m else None


def prove_port_ownership(url: str, pid: int, deny: set[int]) -> int:
    m = re.match(r"^https?://(?:127\.0\.0\.1|localhost):(\d+)$", url.rstrip("/"))
    if not m:
        sys.exit(f"role URL must be loopback host:port, got {url!r}")
    port = int(m.group(1))
    if port in deny:
        sys.exit(f"refusing: port {port} is a production stack port")
    owner = listening_pid(port)
    if owner != pid:
        sys.exit(f"refusing: port {port} is owned by pid {owner}, not {pid} (a foreign server may answer)")
    return port


def parse_role(spec: str):
    name, _, rest = spec.partition("=")
    url, _, pid = rest.rpartition("@")
    if not (name and url and pid.isdigit()):
        sys.exit(f"--role must be NAME=http://127.0.0.1:PORT@PID, got {spec!r}")
    return name, url, int(pid)


def mock_reply(rendering: str, i: int) -> str:
    """Scripted: sees the whole batch -> iterates `batch`; otherwise retypes what it saw (plumbing only)."""
    if rendering == "bounded_k3":
        return "```python\nFINAL(sum(-x['qty'] if x['tag']=='R' else x['qty'] for x in batch))\n```"
    if rendering == "half_first25":
        return "```python\nFINAL(sum(x['qty'] for x in batch[:25]))\n```" if i % 2 else "```python\nFINAL(sum(x['qty'] for x in batch))\n```"
    return "```python\nFINAL(sum(x['qty'] for x in batch[:3]))\n```"


def tally(fx: dict, cells: dict) -> dict:
    out = {}
    for key, rows in cells.items():
        rendering = key.split("|", 1)[1]
        c = {"correct": 0, "blind_rule_fit": 0, "retyping_loss": 0, "other": 0, "no_answer": 0}
        for r in rows:
            c["no_answer" if r["answer"] is None else fx_mod.classify(r["answer"], fx, rendering)] += 1
        n = len(rows)
        out[key] = {"n": n, **c, "correct_rate": round(c["correct"] / n, 4) if n else None}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--role", action="append", default=[], help="NAME=http://127.0.0.1:PORT@PID (repeatable; one private server per distinct GGUF)")
    ap.add_argument("--renderings", default=",".join(RENDERINGS))
    ap.add_argument("--n", type=int, default=DEFAULT_N)
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMP)
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--deny-ports", default=DEFAULT_DENY_PORTS)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    fx = fx_mod.build_fixture()
    renderings = a.renderings.split(",")
    for r in renderings:
        if r not in fx["renderings"]:
            sys.exit(f"unknown rendering {r!r}; fixture has {sorted(fx['renderings'])}")
    roles: list[tuple[str, str | None, int | None]]
    if a.mock:
        roles = [("mock_role", None, None)]
    else:
        if os.environ.get("WINDOW_CONFIRMED") != "1":
            sys.exit("refusing: live mode runs inference. Set WINDOW_CONFIRMED=1 inside an operator-confirmed window.")
        if not a.role:
            sys.exit("live mode needs at least one --role NAME=URL@PID")
        deny = {int(x) for x in a.deny_ports.split(",") if x}
        roles = []
        for spec in a.role:
            name, url, pid = parse_role(spec)
            prove_port_ownership(url, pid, deny)
            roles.append((name, url, pid))

    cells: dict[str, list[dict]] = {}
    t0 = time.time()
    for name, url, _pid in roles:
        for rendering in renderings:
            rows = []
            prompt = build_prompt(fx, rendering)
            for i in range(a.n):
                text = mock_reply(rendering, i) if a.mock else chat(url, prompt, a.temperature, a.max_tokens, seed=1000 + i)
                ans, how = run_reply(text, fx["batch"])
                rows.append({"i": i, "answer": ans, "how": how})
            cells[f"{name}|{rendering}"] = rows
    res = {"schema": "epyc.repl_preview_result.v1", "mock": a.mock, "n_per_cell": a.n, "temperature": a.temperature,
           "roles": [r[0] for r in roles], "renderings": renderings, "fixture_expected": fx["expected"],
           "predeclared": "artifacts/queue-builders-3-20261021/dripreview1/PREDECLARED.md (root repo)",
           "counts": tally(fx, cells), "rows": cells, "wall_s": round(time.time() - t0, 1),
           "note": "a mock result is plumbing only and must never be cited as a finding" if a.mock else ""}
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res["counts"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
