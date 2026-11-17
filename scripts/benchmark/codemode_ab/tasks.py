"""Deterministic task set with end-state oracles (no LLM judge).

Per seed: 4 sweep families x N in {5,20,50,100,200} + 2 fixed single-step novel-tool tasks = 22 tasks.

tier "static": solvable with tools named in today's static prompt (list_dir, peek, json_parse, file_write_safe).
tier "novel":  REQUIRES a tool that the static prompt does not name (syn_sku_price / syn_ledger_post / statistics).

``needed`` is a list of any-of groups: the minimal tool requirement. Selector recall (arm C) is the fraction
of groups with at least one member in the top-k; ``needed_all_in_topk`` is all groups covered.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from scripts.benchmark.codemode_ab.world import World, sku_price_cents

SWEEP_N = (5, 20, 50, 100, 200)
STAGES = {1: (0, 5, 20), 2: (50,), 3: (100, 200)}  # n == 0 marks the fixed single-step tasks
FAMILIES = ("S-AGG", "S-FILT", "N-CSV", "N-LEDGER")
SINGLES = ("N-STAT", "N-QUOTE")
SKU_POOL = [f"SKU-{i:04d}" for i in range(1, 31)]
ACCOUNTS = ["ACC-ALPHA", "ACC-BRAVO", "ACC-CHARLIE", "ACC-DELTA", "ACC-ECHO"]
LEDGER_MIN_CENTS = 5000

READ_ANY = ["read_file", "peek"]
WRITE_ANY = ["write_file", "file_write_safe"]


@dataclass
class Task:
    task_id: str
    family: str
    n: int
    seed: int
    tier: str
    prompt: str
    needed: list[list[str]]
    setup: Callable[[World], None]
    oracle: Callable[[World], tuple[bool, dict[str, Any]]]
    expected: dict[str, Any] = field(default_factory=dict)

    @property
    def stage(self) -> int:
        return next(s for s, ns in STAGES.items() if self.n in ns)


def _rng(family: str, n: int, seed: int) -> random.Random:
    return random.Random(int(hashlib.sha256(f"{family}|{n}|{seed}".encode()).hexdigest()[:12], 16))


def _items(rng: random.Random, n: int) -> list[dict[str, Any]]:
    qtys = rng.sample(range(1, 10_000), n)  # unique, so "largest" is unambiguous
    flags = [rng.choice(["ship", "hold", "drop"]) for _ in range(n)]
    return [{"id": f"it-{i + 1:03d}", "qty": q, "flag": f} for i, (q, f) in enumerate(zip(qtys, flags))]


def _read_json(world: World, rel: str) -> Any:
    p = world.path(rel)
    return json.loads(p.read_text()) if p.is_file() else None


def _task_s_agg(n: int, seed: int) -> Task:
    rng = _rng("S-AGG", n, seed)
    items = _items(rng, n)
    exp = {"total_qty": sum(i["qty"] for i in items), "ship_count": sum(i["flag"] == "ship" for i in items),
           "max_id": max(items, key=lambda i: i["qty"])["id"]}

    def setup(w: World) -> None:
        d = w.path("data/inbox")
        d.mkdir(parents=True, exist_ok=True)
        for k, it in enumerate(items):
            (d / f"item_{k + 1:03d}.json").write_text(json.dumps(it))

    def oracle(w: World) -> tuple[bool, dict[str, Any]]:
        try:
            got = _read_json(w, "out/summary.json")
        except ValueError:
            return False, {"reason": "unparsable out/summary.json"}
        return got == exp, {"got": got, "expected": exp}

    prompt = (f"The directory data/inbox holds {n} JSON files (item_001.json ...). Each is an object with keys "
              '"id", "qty" (integer) and "flag" ("ship", "hold" or "drop"). Read every file and write '
              'out/summary.json containing exactly the JSON object {"total_qty": <sum of qty over ALL items>, '
              '"ship_count": <number of items whose flag is "ship">, "max_id": <id of the item with the largest qty>}.')
    return Task(f"S-AGG-n{n}-s{seed}", "S-AGG", n, seed, "static", prompt,
                [["list_dir", "list_directory"], READ_ANY, WRITE_ANY], setup, oracle, exp)


def _task_s_filt(n: int, seed: int) -> Task:
    rng = _rng("S-FILT", n, seed)
    items = _items(rng, n)
    exp = {i["id"]: f'{i["id"]}:{i["qty"]}' for i in items if i["flag"] == "ship"}

    def setup(w: World) -> None:
        d = w.path("data/inbox")
        d.mkdir(parents=True, exist_ok=True)
        for k, it in enumerate(items):
            (d / f"item_{k + 1:03d}.json").write_text(json.dumps(it))

    def oracle(w: World) -> tuple[bool, dict[str, Any]]:
        out = w.path("out/ship")
        got = {p.stem: p.read_text() for p in out.glob("*.txt")} if out.is_dir() else {}
        missing = sorted(set(exp) - set(got))
        extra = sorted(set(got) - set(exp))
        wrong = sorted(k for k in set(exp) & set(got) if got[k] != exp[k])
        return not (missing or extra or wrong), {"missing": missing[:5], "extra": extra[:5], "wrong": wrong[:5],
                                                  "n_expected": len(exp), "n_got": len(got)}

    prompt = (f"The directory data/inbox holds {n} JSON files (item_001.json ...), each an object with keys "
              '"id", "qty" and "flag". For EVERY item whose flag is "ship", create the file out/ship/<id>.txt whose '
              'content is exactly "<id>:<qty>" (for example it-007:1234), with no trailing newline. '
              "Create no other files.")
    return Task(f"S-FILT-n{n}-s{seed}", "S-FILT", n, seed, "static", prompt,
                [["list_dir", "list_directory"], READ_ANY, WRITE_ANY], setup, oracle, {"n_expected": len(exp)})


def _orders(rng: random.Random, n: int) -> list[dict[str, Any]]:
    return [{"order_id": f"ORD-{i + 1:04d}", "account": rng.choice(ACCOUNTS), "sku": rng.choice(SKU_POOL),
             "qty": rng.randint(1, 9), "amount_cents": rng.choice([1200, 2500, 4999, 5000, 7500, 12000, 30000])}
            for i in range(n)]


def _write_orders_csv(w: World, orders: list[dict[str, Any]], cols: list[str]) -> None:
    p = w.path("data/orders.csv")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join([",".join(cols)] + [",".join(str(o[c]) for c in cols) for o in orders]) + "\n")


def _task_n_csv(n: int, seed: int) -> Task:
    rng = _rng("N-CSV", n, seed)
    orders = _orders(rng, n)
    exp = {"rows": n, "total_cents": sum(o["qty"] * sku_price_cents(o["sku"]) for o in orders)}

    def setup(w: World) -> None:
        _write_orders_csv(w, orders, ["order_id", "sku", "qty"])

    def oracle(w: World) -> tuple[bool, dict[str, Any]]:
        try:
            got = _read_json(w, "out/total.json")
        except ValueError:
            return False, {"reason": "unparsable out/total.json"}
        return got == exp, {"got": got, "expected": exp}

    prompt = (f"data/orders.csv has {n} order rows with columns order_id,sku,qty. Use the product price lookup tool "
              "to get the unit price (in cents) of each SKU, then write out/total.json containing exactly "
              '{"rows": <number of order rows>, "total_cents": <sum over all rows of qty * unit price in cents>}. '
              "Prices are only available from the tool; do not guess them.")
    return Task(f"N-CSV-n{n}-s{seed}", "N-CSV", n, seed, "novel", prompt,
                [READ_ANY, ["syn_sku_price"], WRITE_ANY], setup, oracle, exp)


def _task_n_ledger(n: int, seed: int) -> Task:
    rng = _rng("N-LEDGER", n, seed)
    orders = _orders(rng, n)
    exp = sorted((o["account"], o["amount_cents"], o["order_id"]) for o in orders if o["amount_cents"] >= LEDGER_MIN_CENTS)

    def setup(w: World) -> None:
        _write_orders_csv(w, orders, ["order_id", "account", "amount_cents"])

    def oracle(w: World) -> tuple[bool, dict[str, Any]]:
        got = sorted((e["account"], e["amount_cents"], e["memo"]) for e in w.ledger())
        return got == exp, {"n_expected": len(exp), "n_got": len(got),
                            "missing": [list(x) for x in sorted(set(exp) - set(got))[:3]],
                            "extra_or_dup": len(got) - len(set(got))}

    prompt = (f"data/orders.csv has {n} order rows with columns order_id,account,amount_cents. Post to the accounting "
              f"ledger, using the ledger posting tool, one entry for EVERY order whose amount_cents is at least "
              f"{LEDGER_MIN_CENTS}: account = the row's account, amount_cents = the row's amount_cents, memo = the "
              "row's order_id. Post each qualifying order exactly once (the ledger is append-only; a repeat creates a "
              "duplicate entry). Post nothing for the other orders.")
    return Task(f"N-LEDGER-n{n}-s{seed}", "N-LEDGER", n, seed, "novel", prompt,
                [READ_ANY, ["syn_ledger_post"]], setup, oracle, {"n_expected": len(exp)})


def _task_n_stat(seed: int) -> Task:
    rng = _rng("N-STAT", 0, seed)
    nums = [rng.randint(1, 500) for _ in range(40)]
    exp = {"mean": round(sum(nums) / len(nums), 3), "max": max(nums)}

    def setup(w: World) -> None:
        p = w.path("data/series.json")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(nums))

    def oracle(w: World) -> tuple[bool, dict[str, Any]]:
        try:
            got = _read_json(w, "out/stats.json")
        except ValueError:
            return False, {"reason": "unparsable out/stats.json"}
        ok = isinstance(got, dict) and set(got) == {"mean", "max"} and abs(float(got["mean"]) - exp["mean"]) <= 0.001 \
            and float(got["max"]) == exp["max"]
        return ok, {"got": got, "expected": exp}

    prompt = ('data/series.json is a JSON list of 40 numbers. Use the statistics tool to compute their mean and maximum '
              'and write out/stats.json containing exactly {"mean": <mean rounded to 3 decimals>, "max": <maximum>}.')
    return Task(f"N-STAT-s{seed}", "N-STAT", 0, seed, "novel", prompt,
                [READ_ANY, ["statistics", "calculate"], WRITE_ANY], setup, oracle, exp)


def _task_n_quote(seed: int) -> Task:
    rng = _rng("N-QUOTE", 0, seed)
    sku, qty = rng.choice(SKU_POOL), rng.randint(2, 40)
    exp = {"sku": sku, "qty": qty, "total_cents": qty * sku_price_cents(sku)}

    def setup(w: World) -> None:
        w.path("out").mkdir(parents=True, exist_ok=True)

    def oracle(w: World) -> tuple[bool, dict[str, Any]]:
        try:
            got = _read_json(w, "out/quote.json")
        except ValueError:
            return False, {"reason": "unparsable out/quote.json"}
        return got == exp, {"got": got, "expected": exp}

    prompt = (f"Quote an order of {qty} units of {sku}: look up its unit price with the product price lookup tool and "
              f'write out/quote.json containing exactly {{"sku": "{sku}", "qty": {qty}, "total_cents": <qty * unit price in cents>}}.')
    return Task(f"N-QUOTE-s{seed}", "N-QUOTE", 0, seed, "novel", prompt,
                [["syn_sku_price"], WRITE_ANY], setup, oracle, exp)


_BUILDERS = {"S-AGG": _task_s_agg, "S-FILT": _task_s_filt, "N-CSV": _task_n_csv, "N-LEDGER": _task_n_ledger}


def build_tasks(seeds: list[int], ns: tuple[int, ...] = SWEEP_N) -> list[Task]:
    out: list[Task] = []
    for s in seeds:
        out += [_task_n_stat(s), _task_n_quote(s)]
        for n in ns:
            out += [_BUILDERS[f](n, s) for f in FAMILIES]
    return out
