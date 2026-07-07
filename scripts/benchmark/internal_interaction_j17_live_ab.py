#!/usr/bin/env python3
"""J17 live A/B for review_before_commit consult at the edit-transaction seam.

This harness keeps the production API route inert: it calls
``run_edit_transaction`` in-process and uses the live orchestrator API only for
the coder draft and optional architect consult calls. The checked-in BEP slice
has five code-edit tasks, so the default 50-turn run repeats that fixed slice
ten times and records the repetition explicitly in the artifact.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

ORCH = Path("/mnt/raid0/llm/epyc-orchestrator")
TASKS_PATH = ORCH / "data" / "bep_sandbox" / "tasks.jsonl"
DEFAULT_API_URL = "http://localhost:8000/chat"
DEFAULT_OUTPUT_ROOT = ORCH / "orchestration" / "reports"
DEFAULT_SCRATCH = Path("/mnt/raid0/llm/tmp/j17_review_before_commit")
TARGETED_TASKS: list[dict[str, Any]] = [
    {
        "id": "r1_compat_import_shim",
        "kind": "compatibility_refactor",
        "prompt": (
            "Rename the public helper implementation from `legacy_math.py` to `math_core.py`, "
            "but preserve backward compatibility for existing callers that still import "
            "`legacy_math.add`. Update `service.py` to use the new module. Hidden callers must "
            "still be able to `from legacy_math import add`."
        ),
        "files": {
            "legacy_math.py": "def add(a, b):\n    return a + b\n",
            "service.py": "from legacy_math import add\n\ndef total(values):\n    acc = 0\n    for v in values:\n        acc = add(acc, v)\n    return acc\n",
            "client.py": "from legacy_math import add\n\nprint(add(2, 5))\n",
        },
        "verifier_cmd": (
            "python3 - <<'PY'\n"
            "import client, legacy_math, math_core, service\n"
            "assert service.total([1,2,3]) == 6\n"
            "assert legacy_math.add(4,5) == 9\n"
            "assert math_core.add(4,5) == 9\n"
            "print('PASS')\n"
            "PY"
        ),
    },
    {
        "id": "r2_config_migration_defaults",
        "kind": "api_migration",
        "prompt": (
            "`settings.load()` currently understands only `timeout`. Migrate it to accept both "
            "old configs with `timeout` and new configs with `request_timeout_ms`. The returned "
            "dict must always contain `request_timeout_ms`. Preserve the default timeout of 1000 "
            "when neither key is present. Update `app.py` to use the new key."
        ),
        "files": {
            "settings.py": "def load(config):\n    return {'timeout': config.get('timeout', 1000)}\n",
            "app.py": "from settings import load\n\ndef timeout(config):\n    return load(config)['timeout']\n",
        },
        "verifier_cmd": (
            "python3 - <<'PY'\n"
            "from settings import load\n"
            "from app import timeout\n"
            "assert load({'timeout': 7})['request_timeout_ms'] == 7\n"
            "assert load({'request_timeout_ms': 8})['request_timeout_ms'] == 8\n"
            "assert load({})['request_timeout_ms'] == 1000\n"
            "assert timeout({'timeout': 11}) == 11\n"
            "assert timeout({'request_timeout_ms': 12}) == 12\n"
            "assert 'timeout' not in load({'timeout': 7})\n"
            "print('PASS')\n"
            "PY"
        ),
    },
    {
        "id": "r3_transaction_rollback",
        "kind": "semantic_bugfix",
        "prompt": (
            "`transfer()` should be atomic: if the destination deposit fails, the source balance "
            "must be restored and the function should return False. On success it should return "
            "True. Do not change the public Account API."
        ),
        "files": {
            "bank.py": "class Account:\n    def __init__(self, balance=0, fail_deposit=False):\n        self.balance = balance\n        self.fail_deposit = fail_deposit\n\n    def withdraw(self, amount):\n        if self.balance < amount:\n            return False\n        self.balance -= amount\n        return True\n\n    def deposit(self, amount):\n        if self.fail_deposit:\n            raise RuntimeError('deposit failed')\n        self.balance += amount\n        return True\n\n\ndef transfer(src, dst, amount):\n    if not src.withdraw(amount):\n        return False\n    dst.deposit(amount)\n    return True\n",
        },
        "verifier_cmd": (
            "python3 - <<'PY'\n"
            "from bank import Account, transfer\n"
            "a, b = Account(100), Account(0)\n"
            "assert transfer(a, b, 30) is True\n"
            "assert (a.balance, b.balance) == (70, 30)\n"
            "c, d = Account(100), Account(0, fail_deposit=True)\n"
            "assert transfer(c, d, 40) is False\n"
            "assert (c.balance, d.balance) == (100, 0)\n"
            "assert transfer(Account(5), Account(0), 10) is False\n"
            "print('PASS')\n"
            "PY"
        ),
    },
    {
        "id": "r4_parser_comments_quotes",
        "kind": "edge_case_parser",
        "prompt": (
            "`parse_env()` should parse KEY=VALUE lines, ignore blank lines and lines whose first "
            "non-space character is '#', strip surrounding whitespace, and preserve '#' characters "
            "inside values. Values may contain '=' after the first separator."
        ),
        "files": {
            "envparse.py": "def parse_env(text):\n    out = {}\n    for line in text.splitlines():\n        if not line or line.startswith('#'):\n            continue\n        key, value = line.split('=')\n        out[key] = value\n    return out\n",
        },
        "verifier_cmd": (
            "python3 - <<'PY'\n"
            "from envparse import parse_env\n"
            "text = ' A = one # keep\\n# skip\\nB=x=y\\n   # skip too\\nEMPTY=\\n'\n"
            "assert parse_env(text) == {'A': 'one # keep', 'B': 'x=y', 'EMPTY': ''}\n"
            "print('PASS')\n"
            "PY"
        ),
    },
    {
        "id": "r5_async_ordering",
        "kind": "async_semantics",
        "prompt": (
            "`fetch_all()` should run the provided async fetcher concurrently for every URL but "
            "return results in the same order as the input URLs. Do not sort by completion order."
        ),
        "files": {
            "loader.py": "import asyncio\n\nasync def fetch_all(urls, fetcher):\n    results = []\n    for url in urls:\n        results.append(await fetcher(url))\n    return results\n",
        },
        "verifier_cmd": (
            "python3 - <<'PY'\n"
            "import asyncio, time\n"
            "from loader import fetch_all\n"
            "async def fetcher(url):\n"
            "    await asyncio.sleep({'slow': 0.05, 'fast': 0.0, 'mid': 0.02}[url])\n"
            "    return url.upper()\n"
            "async def main():\n"
            "    start = time.monotonic()\n"
            "    got = await fetch_all(['slow','fast','mid'], fetcher)\n"
            "    assert got == ['SLOW','FAST','MID']\n"
            "    assert time.monotonic() - start < 0.09\n"
            "asyncio.run(main())\n"
            "print('PASS')\n"
            "PY"
        ),
    },
    {
        "id": "r6_cycle_safe_graph",
        "kind": "algorithmic_bugfix",
        "prompt": (
            "`reachable()` should return every node reachable from `start`, including `start`, "
            "without infinite recursion on cycles. Preserve deterministic sorted output."
        ),
        "files": {
            "graph.py": "def reachable(graph, start):\n    out = []\n    def visit(node):\n        out.append(node)\n        for nxt in graph.get(node, []):\n            visit(nxt)\n    visit(start)\n    return sorted(out)\n",
        },
        "verifier_cmd": (
            "python3 - <<'PY'\n"
            "from graph import reachable\n"
            "g = {'a': ['b', 'c'], 'b': ['c'], 'c': ['a'], 'd': ['a']}\n"
            "assert reachable(g, 'a') == ['a','b','c']\n"
            "assert reachable(g, 'd') == ['a','b','c','d']\n"
            "print('PASS')\n"
            "PY"
        ),
    },
    {
        "id": "r7_casefold_dedupe",
        "kind": "data_contract",
        "prompt": (
            "`normalize_users()` should deduplicate by case-insensitive email while preserving "
            "the first occurrence order. Returned records should contain lower-case email and a "
            "stripped display name."
        ),
        "files": {
            "users.py": "def normalize_users(rows):\n    return [{'email': r['email'], 'name': r['name']} for r in rows]\n",
        },
        "verifier_cmd": (
            "python3 - <<'PY'\n"
            "from users import normalize_users\n"
            "rows = [\n"
            " {'email':' A@X.COM ', 'name':' Ann '},\n"
            " {'email':'a@x.com', 'name':'Other'},\n"
            " {'email':'b@x.com', 'name':' Bob'},\n"
            "]\n"
            "assert normalize_users(rows) == [\n"
            " {'email':'a@x.com', 'name':'Ann'},\n"
            " {'email':'b@x.com', 'name':'Bob'},\n"
            "]\n"
            "print('PASS')\n"
            "PY"
        ),
    },
    {
        "id": "r8_optional_dependency_fallback",
        "kind": "dependency_boundary",
        "prompt": (
            "`render_report()` should use `rich` if it is importable, but must still work without "
            "rich installed. In the fallback path return a plain string with one `key: value` line "
            "per sorted key."
        ),
        "files": {
            "reporting.py": "def render_report(data):\n    from rich.table import Table\n    table = Table()\n    for key, value in data.items():\n        table.add_row(key, str(value))\n    return table\n",
        },
        "verifier_cmd": (
            "python3 - <<'PY'\n"
            "import builtins\n"
            "from reporting import render_report\n"
            "real_import = builtins.__import__\n"
            "def fake_import(name, *args, **kwargs):\n"
            "    if name.startswith('rich'):\n"
            "        raise ImportError('no rich')\n"
            "    return real_import(name, *args, **kwargs)\n"
            "builtins.__import__ = fake_import\n"
            "try:\n"
            "    assert render_report({'b':2, 'a':1}) == 'a: 1\\nb: 2'\n"
            "finally:\n"
            "    builtins.__import__ = real_import\n"
            "print('PASS')\n"
            "PY"
        ),
    },
    {
        "id": "r9_plugin_registry",
        "kind": "multi_file_api_contract",
        "prompt": (
            "Add a tiny plugin registry. `registry.register(name, func)` stores a callable, "
            "`registry.run(name, value)` invokes it, and duplicate names should raise ValueError. "
            "Update `main.py` to register the existing `plugins.double` function and print running "
            "it on 4."
        ),
        "files": {
            "registry.py": "PLUGINS = {}\n",
            "plugins.py": "def double(x):\n    return x * 2\n",
            "main.py": "print('not wired')\n",
        },
        "verifier_cmd": (
            "python3 - <<'PY'\n"
            "import subprocess, sys\n"
            "from registry import register, run\n"
            "def inc(x): return x + 1\n"
            "register('inc', inc)\n"
            "assert run('inc', 2) == 3\n"
            "try:\n"
            "    register('inc', inc)\n"
            "except ValueError:\n"
            "    pass\n"
            "else:\n"
            "    raise AssertionError('duplicate did not fail')\n"
            "assert subprocess.check_output([sys.executable, 'main.py'], text=True).strip() == '8'\n"
            "print('PASS')\n"
            "PY"
        ),
    },
    {
        "id": "r10_path_security",
        "kind": "security_boundary",
        "prompt": (
            "`safe_read(root, relpath)` should read only files under root. Reject absolute paths "
            "and `..` escapes by raising ValueError. Normal nested relative paths should still work."
        ),
        "files": {
            "files.py": "from pathlib import Path\n\ndef safe_read(root, relpath):\n    return (Path(root) / relpath).read_text()\n",
            "data/info.txt": "ok\n",
        },
        "verifier_cmd": (
            "python3 - <<'PY'\n"
            "from pathlib import Path\n"
            "from files import safe_read\n"
            "root = Path('.').resolve()\n"
            "assert safe_read(root, 'data/info.txt') == 'ok\\n'\n"
            "for bad in ['../secret.txt', '/etc/passwd']:\n"
            "    try:\n"
            "        safe_read(root, bad)\n"
            "    except ValueError:\n"
            "        pass\n"
            "    else:\n"
            "        raise AssertionError(f'accepted {bad}')\n"
            "print('PASS')\n"
            "PY"
        ),
    },
]

sys.path.insert(0, str(ORCH))
from src.edit_transaction import EditResult, run_edit_transaction  # noqa: E402
from src.orchestration.consultation import consult  # noqa: E402
from src.orchestration.review_consult_gate import review_before_commit_targeted_gate  # noqa: E402


@dataclass
class HTTPPrimitives:
    api_url: str
    timeout_s: float
    calls: list[dict[str, Any]] = field(default_factory=list)

    def request_context(self, **_kwargs: Any):
        return nullcontext()

    def llm_call(
        self,
        prompt: str,
        *,
        role: str,
        n_tokens: int,
        json_schema: dict[str, Any] | None = None,
    ) -> str:
        started = time.monotonic()
        payload = {
            "prompt": prompt,
            "force_role": role,
            "force_mode": "direct",
            "max_turns": 1,
            "max_tokens": n_tokens,
            "cache_prompt": False,
            "mock_mode": False,
            "real_mode": True,
            "request_priority": "background",
            "workload_class": "campaign",
            "session_id": f"j17-consult-{int(started * 1000)}",
        }
        if json_schema:
            # /chat does not expose constrained decoding; keep the schema in
            # the prompt via consult() and record that this was advisory JSON.
            payload["context"] = "json_schema_supplied_by_consult_helper"
        resp = httpx.post(self.api_url, json=payload, timeout=self.timeout_s)
        elapsed = time.monotonic() - started
        resp.raise_for_status()
        data = resp.json()
        text = data.get("answer") or data.get("response") or data.get("content") or ""
        self.calls.append(
            {
                "role": role,
                "wall_s": round(elapsed, 3),
                "answer_chars": len(text),
                "status_code": resp.status_code,
                "mode": data.get("mode"),
            }
        )
        return str(text)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _load_tasks(suite: str) -> tuple[list[dict[str, Any]], list[str]]:
    if suite == "bep":
        return _load_jsonl(TASKS_PATH), [
            "BEP slice has 5 unique tasks repeated to reach the requested turns.",
            "This slice is useful for seam mechanics but is too simple for consult-value claims.",
        ]
    if suite == "targeted":
        return list(TARGETED_TASKS), [
            "Targeted consult-value slice has 10 unique higher-risk edit tasks repeated to reach the requested turns.",
            "Tasks stress compatibility shims, migration defaults, rollback semantics, parsing edge cases, concurrency, graph cycles, optional dependencies, plugin contracts, and path safety.",
            "The slice is synthetic but designed so a pre-commit reviewer can plausibly catch hidden verifier failures.",
        ]
    raise ValueError(f"unknown task suite: {suite}")


def _git_head() -> str:
    r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ORCH, text=True, capture_output=True, timeout=10)
    return r.stdout.strip()


def _autopilot_active() -> bool:
    r = subprocess.run(["pgrep", "-af", "scripts/autopilot/autopilot.py"], text=True, capture_output=True)
    return r.returncode == 0 and bool(r.stdout.strip())


def _reset_root(root: Path, files: dict[str, str] | None) -> None:
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    for rel, content in (files or {}).items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)


def _run_verifier(root: Path, cmd: str) -> bool:
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    try:
        r = subprocess.run(cmd, shell=True, cwd=root, text=True, capture_output=True, timeout=60, env=env)
    except Exception:
        return False
    return r.returncode == 0 and "PASS" in (r.stdout + r.stderr)


def _chat_llm(api_url: str, timeout_s: float, role: str, prompt: str, *, session_id: str) -> tuple[str, dict[str, Any]]:
    started = time.monotonic()
    payload = {
        "prompt": prompt,
        "force_role": role,
        "force_mode": "direct",
        "max_turns": 1,
        "cache_prompt": False,
        "mock_mode": False,
        "real_mode": True,
        "request_priority": "background",
        "workload_class": "campaign",
        "session_id": session_id,
    }
    resp = httpx.post(api_url, json=payload, timeout=timeout_s)
    elapsed = time.monotonic() - started
    resp.raise_for_status()
    data = resp.json()
    text = data.get("answer") or data.get("response") or data.get("content") or ""
    return str(text), {
        "wall_s": round(elapsed, 3),
        "answer_chars": len(str(text)),
        "status_code": resp.status_code,
        "mode": data.get("mode"),
    }


def _classify(result: EditResult | None, raw: str, verifier_ok: bool, error: str) -> str:
    if error:
        return "chat/http error"
    if result is None:
        return "chat/http error"
    if not result.ok:
        err = (result.error or "").lower()
        if "scope too large" in err:
            return "scope-cap reject"
        if "no valid file blocks" in err or not raw.strip():
            return "parse/no blocks"
        if "functional verifier failed" in err:
            return "verifier fail"
        return "rollback/self-check"
    if not verifier_ok:
        return "verifier fail"
    return "pass"


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * p)))
    return round(ordered[idx], 3)


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_arm: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_arm.setdefault(row["arm"], []).append(row)
    out: dict[str, Any] = {}
    for arm, arm_rows in sorted(by_arm.items()):
        coder_wall = [float(r["coder_call"]["wall_s"]) for r in arm_rows if r.get("coder_call")]
        consult_wall = [
            float(c["wall_s"])
            for r in arm_rows
            for c in (r.get("consult_calls") or [])
        ]
        consult_events = [e for r in arm_rows for e in (r.get("consult_events") or [])]
        out[arm] = {
            "turns": len(arm_rows),
            "passes": sum(1 for r in arm_rows if r["bucket"] == "pass"),
            "quality": round(sum(1 for r in arm_rows if r["bucket"] == "pass") / max(1, len(arm_rows)), 4),
            "coder_wall_p50_s": _percentile(coder_wall, 0.5),
            "coder_wall_p95_s": _percentile(coder_wall, 0.95),
            "consult_calls": len(consult_wall),
            "consult_wall_p50_s": _percentile(consult_wall, 0.5),
            "consult_wall_p95_s": _percentile(consult_wall, 0.95),
            "consult_successes": sum(1 for e in consult_events if e.get("success")),
            "consult_failures": sum(1 for e in consult_events if not e.get("success")),
            "consult_skips": sum(1 for e in consult_events if e.get("skipped")),
            "rerun_requests": sum(1 for e in consult_events if e.get("rerun_requested")),
            "cache_hits": sum(int(e.get("cache_hit", 0) or 0) for e in consult_events),
        }
    if "baseline" in out and "consult" in out:
        b = out["baseline"]
        c = out["consult"]
        out["comparison"] = {
            "quality_delta_pp": round(100 * (c["quality"] - b["quality"]), 3),
            "coder_wall_p50_delta_pct": (
                round(100 * (c["coder_wall_p50_s"] - b["coder_wall_p50_s"]) / b["coder_wall_p50_s"], 3)
                if b.get("coder_wall_p50_s") else None
            ),
            "cache_hit_rate": (
                round(c["cache_hits"] / c["consult_calls"], 4)
                if c.get("consult_calls") else None
            ),
            "gate_notes": [
                "BEP slice has 5 unique tasks repeated to reach 50 turns.",
                "Consult helper currently records cache_ttl_seconds but no cache_hit events.",
                "Wall-clock p50 is used as the live proxy for coder decode p50 in this harness.",
            ],
        }
    if "baseline" in out and "gated" in out:
        b = out["baseline"]
        g = out["gated"]
        out["gated_comparison"] = {
            "quality_delta_pp": round(100 * (g["quality"] - b["quality"]), 3),
            "coder_wall_p50_delta_pct": (
                round(100 * (g["coder_wall_p50_s"] - b["coder_wall_p50_s"]) / b["coder_wall_p50_s"], 3)
                if b.get("coder_wall_p50_s") else None
            ),
            "consult_calls": g.get("consult_calls", 0),
            "consult_skips": g.get("consult_skips", 0),
            "rerun_requests": g.get("rerun_requests", 0),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--confirm-clean-window", action="store_true")
    ap.add_argument("--allow-autopilot-active", action="store_true")
    ap.add_argument("--api-url", default=DEFAULT_API_URL)
    ap.add_argument("--task-suite", choices=["bep", "targeted"], default="bep")
    ap.add_argument(
        "--arms",
        default="baseline,consult",
        help="Comma-separated arms to run: baseline, consult, gated.",
    )
    ap.add_argument("--turns", type=int, default=50)
    ap.add_argument("--timeout-s", type=float, default=360)
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--scratch-root", type=Path, default=DEFAULT_SCRATCH)
    args = ap.parse_args(argv)

    if not args.apply:
        print("Plan only. Pass --apply --confirm-clean-window to run live J17 A/B.")
        return 0
    if not args.confirm_clean_window:
        print("REFUSING live inference: pass --confirm-clean-window.", file=sys.stderr)
        return 2
    if _autopilot_active() and not args.allow_autopilot_active:
        print("REFUSING live inference while AutoPilot is active.", file=sys.stderr)
        return 75

    tasks, task_notes = _load_tasks(args.task_suite)
    arms = [arm.strip() for arm in args.arms.split(",") if arm.strip()]
    allowed_arms = {"baseline", "consult", "gated"}
    if not arms or any(arm not in allowed_arms for arm in arms):
        print(f"Invalid --arms {args.arms!r}; choose from {sorted(allowed_arms)}", file=sys.stderr)
        return 2
    run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.output_root / f"internal_interaction_j17_ab_{run_ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "turns.jsonl"
    rows: list[dict[str, Any]] = []
    task_plan = [tasks[i % len(tasks)] for i in range(args.turns)]

    with rows_path.open("w", encoding="utf-8") as fh:
        for arm in arms:
            for idx, task in enumerate(task_plan, start=1):
                root = args.scratch_root / arm / f"{idx:03d}_{task['id']}"
                _reset_root(root, task.get("files"))
                coder_calls: list[dict[str, Any]] = []
                primitives = HTTPPrimitives(args.api_url, args.timeout_s)
                error = ""

                def llm(prompt: str, *, _arm: str = arm, _idx: int = idx, _task: dict[str, Any] = task) -> str:
                    text, meta = _chat_llm(
                        args.api_url,
                        args.timeout_s,
                        "coder_escalation",
                        prompt,
                        session_id=f"j17-{_arm}-{_idx:03d}-{_task['id']}",
                    )
                    coder_calls.append(meta)
                    return text

                review = None
                review_gate = None
                if arm in {"consult", "gated"}:
                    def review(review_context: str) -> tuple[dict[str, Any], dict[str, Any]]:
                        return consult(
                            consultant_role="architect_general",
                            requester_role="coder_escalation",
                            skill="review_before_commit",
                            context=review_context,
                            primitives=primitives,
                            override_priority="background",
                        )
                if arm == "gated":
                    def review_gate(context: dict[str, Any]) -> dict[str, Any]:
                        decision = review_before_commit_targeted_gate(
                            task_prompt=str(context.get("task_prompt") or ""),
                            current_paths=list(context.get("current_paths") or []),
                            draft_paths=list(context.get("draft_paths") or []),
                            delete_paths=list(context.get("delete_paths") or []),
                            raw_model_output=str(context.get("raw_model_output") or ""),
                        )
                        return {"enabled": decision.enabled, "reasons": list(decision.reasons)}

                try:
                    result, raw = run_edit_transaction(
                        llm,
                        task["prompt"],
                        root,
                        target_files=list((task.get("files") or {}).keys()) or None,
                        verify_fn=lambda tx_root, cmd=task["verifier_cmd"]: _run_verifier(tx_root, cmd),
                        review_before_commit=review,
                        enable_review_before_commit=(arm in {"consult", "gated"}),
                        review_before_commit_gate=review_gate,
                    )
                except Exception as exc:
                    result, raw, error = None, "", f"{type(exc).__name__}: {exc}"
                verifier_ok = bool(result and result.ok and _run_verifier(root, task["verifier_cmd"]))
                bucket = _classify(result, raw, verifier_ok, error)
                row = {
                    "run_ts": run_ts,
                    "arm": arm,
                    "turn_index": idx,
                    "task_id": task["id"],
                    "task_kind": task.get("kind"),
                    "bucket": bucket,
                    "txn_ok": bool(result and result.ok),
                    "verifier_ok": verifier_ok,
                    "written": result.written if result else [],
                    "deleted": result.deleted if result else [],
                    "error": error or (result.error if result else ""),
                    "coder_call": coder_calls[0] if coder_calls else {},
                    "coder_calls_total": len(coder_calls),
                    "consult_calls": primitives.calls,
                    "consult_events": result.consult_events if result else [],
                }
                rows.append(row)
                fh.write(json.dumps(row, sort_keys=True) + "\n")
                fh.flush()
                print(
                    f"[{arm}] {idx}/{args.turns} {task['id']} {bucket} "
                    f"coder_calls={len(coder_calls)} consult_calls={len(primitives.calls)}",
                    flush=True,
                )

    summary = {
        "kind": "internal_interaction_j17_live_ab",
        "run_ts": run_ts,
        "orch_head": _git_head(),
        "api_url": args.api_url,
        "task_suite": args.task_suite,
        "arms": arms,
        "turns_requested_per_arm": args.turns,
        "unique_task_count": len(tasks),
        "fixed_slice_repetitions": args.turns / max(1, len(tasks)),
        "rows_path": str(rows_path),
        "summary": _summarize(rows),
    }
    if "comparison" in summary["summary"]:
        summary["summary"]["comparison"]["gate_notes"] = [
            *task_notes,
            "Consult helper currently records cache_ttl_seconds but no cache_hit events.",
            "Wall-clock p50 is used as the live proxy for coder decode p50 in this harness.",
        ]
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (out_dir / "summary.md").write_text(
        "# J17 Internal Interaction Live A/B\n\n"
        f"- run_ts: `{run_ts}`\n"
        f"- orch_head: `{summary['orch_head']}`\n"
        f"- task suite: `{args.task_suite}`\n"
        f"- rows: `{rows_path}`\n"
        f"- unique tasks: `{len(tasks)}` repeated to `{args.turns}` turns per arm\n\n"
        "```json\n" + json.dumps(summary["summary"], indent=2, sort_keys=True) + "\n```\n"
    )
    print(json.dumps(summary["summary"], indent=2, sort_keys=True))
    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
