"""Scripted mock LLM / embedder / primitives for dry and smoke runs. NO model, NO server.

The mock solver reads the sandbox data directly (privileged) and emits valid call sequences, with
deterministic injected faults so every metric path is exercised: a failing first REPL snippet on
seed % 3 == 0, a duplicate write on seed % 3 == 1 (arm A), and a wrong-argument call on seed % 5 == 0.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from scripts.benchmark.codemode_ab.llm import Reply
from scripts.benchmark.codemode_ab.tasks import LEDGER_MIN_CENTS, Task
from scripts.benchmark.codemode_ab.world import World, sku_price_cents


def _tok(s: str) -> int:
    return max(1, len(s) // 4)


def _solve_calls(task: Task, world: World) -> list[tuple[str, dict[str, Any]]]:
    """The canonical write calls for the task, computed from sandbox data."""
    rd = lambda rel: world.path(rel).read_text()  # noqa: E731
    f = task.family
    if f in ("S-AGG", "S-FILT"):
        items = [json.loads(p.read_text()) for p in sorted(world.path("data/inbox").glob("*.json"))]
        if f == "S-AGG":
            body = {"total_qty": sum(i["qty"] for i in items), "ship_count": sum(i["flag"] == "ship" for i in items),
                    "max_id": max(items, key=lambda i: i["qty"])["id"]}
            return [("file_write_safe", {"path": "out/summary.json", "content": json.dumps(body)})]
        return [("file_write_safe", {"path": f"out/ship/{i['id']}.txt", "content": f"{i['id']}:{i['qty']}"})
                for i in items if i["flag"] == "ship"]
    if f in ("N-CSV", "N-LEDGER"):
        lines = rd("data/orders.csv").strip().splitlines()
        head = lines[0].split(",")
        rows = [dict(zip(head, ln.split(","))) for ln in lines[1:]]
        if f == "N-CSV":
            tot = sum(int(r["qty"]) * sku_price_cents(r["sku"]) for r in rows)
            return [("write_file", {"path": "out/total.json", "content": json.dumps({"rows": len(rows), "total_cents": tot})})]
        return [("syn_ledger_post", {"account": r["account"], "amount_cents": int(r["amount_cents"]), "memo": r["order_id"]})
                for r in rows if int(r["amount_cents"]) >= LEDGER_MIN_CENTS]
    if f == "N-STAT":
        nums = json.loads(rd("data/series.json"))
        return [("write_file", {"path": "out/stats.json", "content": json.dumps({"mean": round(sum(nums) / len(nums), 3), "max": max(nums)})})]
    sku, qty = task.expected["sku"], task.expected["qty"]
    return [("write_file", {"path": "out/quote.json", "content": json.dumps({"sku": sku, "qty": qty, "total_cents": qty * sku_price_cents(sku)})})]


class MockLLM:
    """chat(): function-calling turns when ``tools`` is given (arm A), code turns otherwise (REPL arms)."""

    def __init__(self, task: Task, world: World, seed: int):
        self.task, self.world, self.seed, self.turn = task, world, seed, 0
        self.usage_total = {"prompt_tokens": 0, "completion_tokens": 0}

    def chat(self, messages, *, tools=None, max_tokens=0, seed=0, json_schema=None) -> Reply:
        self.turn += 1
        plen = sum(len(str(m.get("content") or "")) for m in messages)
        if tools:
            r = self._a_turn(messages)
        else:
            r = self._repl_turn()
        r.in_tokens, r.out_tokens, r.in_cached, r.wall_s = _tok("x" * plen), _tok(r.content + json.dumps(r.tool_calls)), 0, 0.001
        return r

    def _a_turn(self, messages) -> Reply:
        t, w = self.task, self.world
        calls: list[tuple[str, dict[str, Any]]] = []
        if self.turn == 1:  # discovery + reads
            if t.tier == "novel":
                calls.append(("list_tools", {}))
            if t.family in ("S-AGG", "S-FILT"):
                calls.append(("list_dir", {"path": "data/inbox"}))
                calls += [("peek", {"n": 100000, "file_path": str(p.relative_to(w.root))}) for p in sorted(w.path("data/inbox").glob("*.json"))]
            elif t.tier == "static":
                calls.append(("peek", {"n": 100000, "file_path": "data/orders.csv"}))
            else:
                calls.append(("call_tool", {"name": "read_file", "arguments": {"path": "data/series.json" if t.family == "N-STAT" else "data/orders.csv"}}))
        elif self.turn == 2:  # lookups
            if t.family == "N-CSV":
                for sku in sorted({ln.split(",")[1] for ln in w.path("data/orders.csv").read_text().splitlines()[1:]}):
                    calls.append(("call_tool", {"name": "syn_sku_price", "arguments": {"sku": sku}}))
            elif t.family == "N-QUOTE":
                calls.append(("call_tool", {"name": "syn_sku_price", "arguments": {"sku": t.expected["sku"]}}))
            elif t.family == "N-STAT":
                calls.append(("call_tool", {"name": "statistics", "arguments": {"data": json.loads(w.path("data/series.json").read_text()), "metrics": ["mean", "max"]}}))
        elif self.turn == 3:  # writes
            writes = _solve_calls(t, w)
            for name, args in writes:
                calls.append((name, args) if name == "file_write_safe" else ("call_tool", {"name": name, "arguments": args}))
            if self.seed % 3 == 1 and calls:  # injected duplicate write
                calls.append(calls[0])
            if self.seed % 5 == 0:  # injected wrong-argument call
                calls.append(("file_info", {"path": "does/not/exist"}))
        if not calls:
            return Reply(content="done") if self.turn >= 3 else Reply(content="", tool_calls=[
                {"id": f"c{self.turn}_x", "name": "list_tools", "arguments": {}, "bad_json": False}])
        return Reply(content="", tool_calls=[{"id": f"c{self.turn}_{i}", "name": nm, "arguments": a, "bad_json": False} for i, (nm, a) in enumerate(calls)])

    def _repl_turn(self) -> Reply:
        t = self.task
        if self.turn == 1 and self.seed % 3 == 0:
            return Reply(content='```python\nresult = CALL("no_such_tool", x=1)\n```')
        lines = ["import json"]
        for name, args in _solve_calls(t, self.world):
            lines.append(f"CALL({name!r}, **{args!r})" if name in ("write_file", "syn_ledger_post") else f"{name}(**{args!r})")
        lines.append('FINAL("done")')
        return Reply(content="```python\n" + "\n".join(lines) + "\n```")


class MockEmbedder:
    """Hashed bag-of-words embeddings; implements the DenseEmbedder protocol."""

    model_id = "mock-hash-embedder"

    def embed(self, texts, *, timeout_s=0.0, admission_wait_s=0.0):
        import numpy as np
        from src.repl_environment.context_search import DenseResult, tokenize

        out = np.zeros((len(texts), 256), dtype="float32")
        for i, t in enumerate(texts):
            for w in tokenize(t):
                out[i, int(hashlib.md5(w.encode()).hexdigest()[:4], 16) % 256] += 1.0
        return DenseResult(out, self.model_id)


class MockPrimitives:
    """typed_decisions seam: flags a group when any of its names/keywords appears in the REQUEST."""

    usage_total = {"prompt_tokens": 0, "completion_tokens": 0}

    def __init__(self, keywords: dict[str, list[str]]):
        self.keywords = keywords

    def llm_call(self, prompt: str, *, role="worker", n_tokens=0, json_schema=None, temperature=0.0, seed=0, **_) -> str:
        req = prompt.split("STATE:\n", 1)[1].split("QUESTION CATALOG", 1)[0].lower()
        import re as _re

        answers = {}
        for g in _re.findall(r"id=needs_(\w+) kind=noul", prompt):
            answers[f"needs_{g}"] = {"noul": any(k in req for k in self.keywords.get(g, []))}
        self.usage_total["prompt_tokens"] += _tok(prompt)
        self.usage_total["completion_tokens"] += _tok(json.dumps(answers))
        return json.dumps({"answers": answers})
