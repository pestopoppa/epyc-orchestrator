"""Deterministic task set with end-state oracles (no LLM judge).

Per seed: 4 sweep families x N in {5,20,50,100,200} + 2 fixed single-step novel-tool tasks = 22 tasks.

tier "static": solvable with tools named in today's LIVE static prompt (COMPACT list: peek, file_write_safe).
tier "novel":  REQUIRES a tool that the static prompt does not name (syn_sku_price / syn_ledger_post / statistics).

``needed`` is a list of any-of groups: the minimal tool requirement. Selector recall (arm C) is the fraction
of groups with at least one member in the top-k; ``needed_all_in_topk`` is all groups covered.
"""

from __future__ import annotations

import functools
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
    # CM probe (cm_probe.py): a task with ``phases`` is a two-phase session. The REPL is checkpointed after phase 1 and restored
    # into a fresh REPL for phase 2 (a branched session); ``blocked_in_phase2`` tools raise in phase 2. Empty for every base task.
    phases: tuple[str, ...] = ()
    blocked_in_phase2: tuple[str, ...] = ()

    @property
    def stage(self) -> int:
        return next((s for s, ns in STAGES.items() if self.n in ns), 1)  # CM tasks (n=60 ...) are not in a sweep stage


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

    prompt = (f"The directory data/inbox holds {n} JSON files (item_001.json ... item_{n:03d}.json). Each is an object with keys "
              '"id", "qty" (integer) and "flag" ("ship", "hold" or "drop"). Read every file and write '
              'out/summary.json containing exactly the JSON object {"total_qty": <sum of qty over ALL items>, '
              '"ship_count": <number of items whose flag is "ship">, "max_id": <id of the item with the largest qty>}.')
    return Task(f"S-AGG-n{n}-s{seed}", "S-AGG", n, seed, "static", prompt,
                [READ_ANY, WRITE_ANY], setup, oracle, exp)


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

    prompt = (f"The directory data/inbox holds {n} JSON files (item_001.json ... item_{n:03d}.json), each an object with keys "
              '"id", "qty" and "flag". For EVERY item whose flag is "ship", create the file out/ship/<id>.txt whose '
              'content is exactly "<id>:<qty>" (for example it-007:1234), with no trailing newline. '
              "Create no other files.")
    return Task(f"S-FILT-n{n}-s{seed}", "S-FILT", n, seed, "static", prompt,
                [READ_ANY, WRITE_ANY], setup, oracle, {"n_expected": len(exp)})


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


# ======================================================================================================================
# CM probe task additions (CM-4 classifier functions, CM-5 result handles). Deterministic end-state oracles, no LLM judge.
# ``build_tasks`` is NOT changed: the sealed base plan of the earlier rows is untouched; these are built by ``build_cm_tasks``.
#
#   C-LABEL-n{20,100}  semantic labelling of N support messages into 4 categories; gold = the generator's label; success iff
#                      accuracy >= 0.90. A script can label the items without them entering the prompt (CM-4 ``classify``).
#   C-RISK-n20         label 20 real calibration prompts with the factual-risk band the system's classifier assigns; gold =
#                      the production ``risk_bands`` output (deterministic regex classifier). Capability task: the control arm
#                      has no way to reproduce the classifier's rule, so a control success means a lucky guess (expected ~0).
#   C-ROUTE-n20        same with the role classifier's ``trinity_role`` (worker | thinker); gold = production ``route_classify``.
#   H-PRICE-n{20,60}   two-phase session (CM-5): phase 1 prices every distinct SKU with the lookup tool; the REPL is then
#                      checkpointed/restored into a fresh REPL (a branched session) and the lookup tool is withdrawn; phase 2 must
#                      write the order total from what phase 1 kept. Handle store vs plain restored variables.
# ======================================================================================================================
CM_SEEDS = (0, 1, 2, 3, 4)
CM_LABELS = ("billing", "shipping", "technical", "account")
LABEL_PASS_FRACTION = 0.90
CM_FAMILIES = ("C-LABEL", "C-RISK", "C-ROUTE", "H-PRICE")
_POOL_FILE = Path(__file__).resolve().parents[3] / "orchestration" / "factual_risk_calibration_v2.jsonl"

_LABEL_OPENERS = ["", "", "Hi team, ", "Hello, ", "Quick question: ", "Urgent: ", "Hi, ", "Good morning. ", "Dear support, "]
_LABEL_CLOSERS = ["", "", " Thanks.", " Please advise.", " Thank you in advance.", " This is the second time I am asking.", " Regards."]
_LABEL_TEMPLATES: dict[str, list[str]] = {
    "billing": [
        "I was charged twice for invoice {ref} this month and need the duplicate payment reversed.",
        "My card was billed {amt} but the plan I picked costs less than that.",
        "Can you send a receipt for the payment on {date}? The amount on my bank statement looks wrong.",
        "Why did my renewal cost more than last year when I never changed anything?",
        "Please refund the {amt} I paid for the annual plan, I cancelled inside the trial.",
        "There is a charge on my statement for order {ref} that I do not recognise.",
        "The invoice for {date} shows the wrong tax amount and the wrong company name.",
        "I upgraded the plan yesterday but the prorated difference was never credited back.",
        "My payment of {amt} failed twice and now the invoice is marked overdue.",
        "Can I switch from monthly to yearly payments and keep the discount I was promised?",
        "We were charged for five seats although we only use three of them.",
        "The coupon code was not applied to invoice {ref}, so we overpaid.",
    ],
    "shipping": [
        "My parcel {ref} has been stuck at the depot since {date} and nobody can tell me when it will arrive.",
        "The courier left my package at the wrong address and the neighbours say they never saw it.",
        "Tracking says delivered but there is nothing at my door.",
        "Can I change the delivery address for order {ref} before it leaves the warehouse?",
        "Only two of the three boxes in order {ref} arrived, where is the third one?",
        "The package arrived with the outer carton crushed and the contents scratched.",
        "I need the delivery moved to {date} because nobody will be home earlier.",
        "How long does standard delivery take to the islands, and is there an express option?",
        "The carrier keeps attempting delivery at times when the office is closed.",
        "My order {ref} was sent to the old address even though I updated it last week.",
        "The tracking page has shown the same status for nine days now.",
        "Can the parcel be held at a pickup point instead of being delivered to the house?",
    ],
    "technical": [
        "The app crashes with error code {err} every time I open the export screen.",
        "Sync stopped working after the latest update and the spinner never finishes.",
        "I get a timeout when uploading files larger than {mb} MB.",
        "The login page shows a blank white screen in the browser since this morning.",
        "Dark mode makes the chart labels unreadable on the dashboard.",
        "The mobile client freezes whenever I rotate the phone while a report is loading.",
        "Exported spreadsheets are missing the last column and the header row is duplicated.",
        "The API returns status {err} for requests that worked fine last week.",
        "Notifications arrive twice on the desktop client but never on the phone.",
        "After the payment screen update the checkout page keeps reloading in a loop.",
        "Search returns no results for terms that I can see in the documents.",
        "The integration with our calendar disconnects after a few minutes and has to be re-authorised.",
    ],
    "account": [
        "I need to change the email address on my profile because I lost access to the old mailbox.",
        "Please close my account and delete the data you hold about me.",
        "I forgot my password and the reset link has expired.",
        "How do I add a second administrator to our team workspace?",
        "Two-factor codes stopped arriving after I changed my phone number.",
        "Can the account owner be transferred to my colleague, who is taking over the project?",
        "My username is spelled wrong on my profile and I cannot edit it.",
        "I want to merge my two accounts into one before {date}.",
        "Someone else may have signed in to my account, please lock it and help me recover it.",
        "How do I remove a former employee from our organisation's member list?",
        "Please change the display language and the time zone of my profile.",
        "I cannot find where to download the data export of my account settings.",
    ],
}


def _label_text(rng: random.Random, label: str) -> str:
    body = rng.choice(_LABEL_TEMPLATES[label]).format(
        ref=f"{rng.choice('ABCDEFGH')}{rng.randint(1000, 9999)}", amt=f"${rng.randint(5, 400)}.{rng.randint(0, 99):02d}",
        date=f"2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}", err=rng.choice([403, 404, 500, 502, 503, "E-17", "E-204"]),
        mb=rng.choice([5, 10, 25, 50, 100]))
    return rng.choice(_LABEL_OPENERS) + body + rng.choice(_LABEL_CLOSERS)


def _task_c_label(n: int, seed: int) -> Task:
    rng = _rng("C-LABEL", n, seed)
    labels = [CM_LABELS[i % len(CM_LABELS)] for i in range(n)]
    rng.shuffle(labels)
    items = [{"id": f"t-{i + 1:03d}", "text": _label_text(rng, lab)} for i, lab in enumerate(labels)]
    gold = {it["id"]: lab for it, lab in zip(items, labels)}
    need = -(-int(n * 100 * LABEL_PASS_FRACTION) // 100)  # ceil(0.9 n)

    def setup(w: World) -> None:
        p = w.path("data/items.json")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(items))

    def oracle(w: World) -> tuple[bool, dict[str, Any]]:
        return _label_oracle(w, "out/labels.json", gold, need)

    prompt = (f"data/items.json is a JSON list of {n} customer messages, each an object with keys \"id\" and \"text\". Assign EVERY message "
              "exactly one category from: billing, shipping, technical, account (billing = charges, invoices, refunds, payment "
              "problems; shipping = delivery, parcels, tracking; technical = errors, crashes, sync, uploads, performance; account = "
              "login, profile, users, closing or recovering the account). Write out/labels.json containing exactly one JSON object "
              "that maps each id to its category string.")
    return Task(f"C-LABEL-n{n}-s{seed}", "C-LABEL", n, seed, "novel", prompt, [READ_ANY, WRITE_ANY], setup, oracle,
                {"need_correct": need, "n": n, "gold": gold})


def _label_oracle(w: World, rel: str, gold: dict[str, str], need: int) -> tuple[bool, dict[str, Any]]:
    try:
        got = _read_json(w, rel)
    except ValueError:
        return False, {"reason": f"unparsable {rel}"}
    if not isinstance(got, dict):
        return False, {"reason": f"{rel} is not a JSON object"}
    correct = sum(1 for k, v in gold.items() if got.get(k) == v)
    return correct >= need, {"correct": correct, "n": len(gold), "need": need, "n_got": len(got)}


@functools.lru_cache(maxsize=1)
def _classifier_pool() -> list[dict[str, Any]]:
    """Calibration prompts (<= 400 chars, no code fences) with the production classifiers' outputs, computed once.
    Deterministic: the file is checked in and both classifiers are rule-based."""
    from src.repl_environment.codemode_tools import ClassifierBudget, risk_bands, route_classify

    texts = []
    for ln in _POOL_FILE.read_text().splitlines():
        if not ln.strip():
            continue
        t = json.loads(ln).get("prompt", "")
        if 40 <= len(t) <= 400 and "```" not in t:
            texts.append(t)
    rows: list[dict[str, Any]] = []
    for i in range(0, len(texts), 200):
        chunk = texts[i:i + 200]
        rb, rc = risk_bands(chunk, budget=ClassifierBudget(10 ** 9)), route_classify(chunk, budget=ClassifierBudget(10 ** 9))
        rows += [{"text": t, "risk_band": a["risk_band"], "trinity_role": b["trinity_role"]} for t, a, b in zip(chunk, rb, rc)]
    return rows


def _task_classifier(kind: str, n: int, seed: int) -> Task:
    fam, key, bands, rel = (("C-RISK", "risk_band", ("low", "medium"), "out/risk.json") if kind == "risk"
                            else ("C-ROUTE", "trinity_role", ("worker", "thinker"), "out/roles.json"))
    rng = _rng(fam, n, seed)
    pool = _classifier_pool()
    picked: list[dict[str, Any]] = []
    for b in bands:
        cand = sorted((r for r in pool if r[key] == b), key=lambda r: r["text"])
        picked += rng.sample(cand, n // len(bands))
    rng.shuffle(picked)
    items = [{"id": f"q-{i + 1:03d}", "text": r["text"]} for i, r in enumerate(picked)]
    gold = {it["id"]: r[key] for it, r in zip(items, picked)}

    def setup(w: World) -> None:
        p = w.path("data/queries.json")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(items))

    def oracle(w: World) -> tuple[bool, dict[str, Any]]:
        return _label_oracle(w, rel, gold, n)  # exact: every id

    if kind == "risk":
        what = ("the factual-risk band (\"low\" or \"medium\") that the system's own factual-risk classifier assigns to that text "
                "(a rule-based classifier: you cannot infer its rule from the wording alone)")
    else:
        what = ("the routing role (\"worker\" or \"thinker\") that the system's own role classifier assigns to that text "
                "(a rule-based classifier: you cannot infer its rule from the wording alone)")
    prompt = (f"data/queries.json is a JSON list of {n} objects with keys \"id\" and \"text\". For each one determine {what}. "
              f"Write {rel} containing exactly one JSON object that maps each id to its label string.")
    return Task(f"{fam}-n{n}-s{seed}", fam, n, seed, "novel", prompt, [READ_ANY, WRITE_ANY], setup, oracle, {"n": n, "classifier": key, "gold": gold})


def _task_h_price(n: int, seed: int) -> Task:
    rng = _rng("H-PRICE", n, seed)
    orders = _orders(rng, n)
    skus = sorted({o["sku"] for o in orders})
    exp = {"rows": n, "total_cents": sum(o["qty"] * sku_price_cents(o["sku"]) for o in orders)}

    def setup(w: World) -> None:
        _write_orders_csv(w, orders, ["order_id", "sku", "qty"])

    def oracle(w: World) -> tuple[bool, dict[str, Any]]:
        try:
            got = _read_json(w, "out/total.json")
        except ValueError:
            return False, {"reason": "unparsable out/total.json"}
        return got == exp, {"got": got, "expected": exp, "n_distinct_skus": len(skus)}

    p1 = (f"data/orders.csv has {n} order rows with columns order_id,sku,qty. PHASE 1 of 2: use the product price lookup tool to get "
          "the unit price (in cents) of every distinct SKU in the file. Write no files in this phase. Phase 2 will ask for the order total; the "
          "price lookup tool will be withdrawn then, but your REPL session carries over into phase 2, so keep whatever you will need. "
          "Finish this phase with FINAL() once the prices are collected.")
    p2 = (f"PHASE 2 of 2 (the session was restored from a checkpoint; the product price lookup tool is no longer available). data/orders.csv "
          f"still has the {n} order rows. Using the unit prices collected in phase 1, write out/total.json containing exactly "
          '{"rows": <number of order rows>, "total_cents": <sum over all rows of qty * unit price in cents>}. Do not guess prices.')
    return Task(f"H-PRICE-n{n}-s{seed}", "H-PRICE", n, seed, "novel", p1, [READ_ANY, ["syn_sku_price"], WRITE_ANY], setup, oracle,
                {"n": n, "n_distinct_skus": len(skus), "answer": exp}, phases=(p1, p2), blocked_in_phase2=("syn_sku_price",))


def build_cm_tasks(seeds: list[int] | tuple[int, ...] = CM_SEEDS) -> list[Task]:
    """The CM additions, ordered (seed, family, n). ``C-LABEL`` n in {20, 100}; the others n in {20} (C-RISK, C-ROUTE) and {20, 60} (H-PRICE)."""
    out: list[Task] = []
    for s in seeds:
        out += [_task_c_label(20, s), _task_c_label(100, s), _task_classifier("risk", 20, s), _task_classifier("route", 20, s),
                _task_h_price(20, s), _task_h_price(60, s)]
    return out
