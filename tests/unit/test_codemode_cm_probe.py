"""CM probe (scripts/benchmark/codemode_ab/cm_probe.py, cm_plan.py, run_cm.py, the CM task additions): plan integrity, oracles, arm plumbing.
No model, no server: a scripted LLM answers the REPL turns and the classify sub-calls. Scratch worlds live under /mnt/raid0/llm/tmp/codemode-cm-test (never deleted)."""

from __future__ import annotations

import json
import os
import uuid
from argparse import Namespace
from pathlib import Path

import pytest

from scripts.benchmark.codemode_ab import arms as A
from scripts.benchmark.codemode_ab import cm_plan, run_cm, run_probe
from scripts.benchmark.codemode_ab.cm_probe import ADDENDUM, ARMS_CM, CM_FLAGS, run_episode_cm
from scripts.benchmark.codemode_ab.llm import Reply
from scripts.benchmark.codemode_ab.tasks import build_cm_tasks
from scripts.benchmark.codemode_ab.world import World, sku_price_cents

ORCH = Path(__file__).resolve().parents[2]
TMP = Path("/mnt/raid0/llm/tmp/codemode-cm-test")


@pytest.fixture
def work():
    return TMP / uuid.uuid4().hex


@pytest.fixture
def selectors(work):
    from src.prompt_builders import tool_selector as ts

    template = World(work / "template", ORCH)
    sel = run_probe.build_selectors(Namespace(mock=True), template, None)
    run_probe.register_native_picker(sel, template)
    os.environ["ORCHESTRATOR_TOOL_SELECTOR_LOG"] = ""
    yield sel
    ts.set_native_selector_provider(None)
    os.environ.pop("ORCHESTRATOR_TOOL_SELECTOR_LOG", None)


class ScriptedLLM:
    """REPL turns come from ``script`` in order; a call with ``json_schema`` is a typed-decision (classify) sub-call answered with ``label``."""

    def __init__(self, script: list[str], label: str = "billing"):
        self.script, self.label, self.prompts, self.sub_prompts = list(script), label, [], []
        self.usage_total = {"prompt_tokens": 0, "completion_tokens": 0}

    def chat(self, messages, *, tools=None, max_tokens=0, seed=0, json_schema=None):
        text = messages[-1]["content"]
        if json_schema is not None:
            self.sub_prompts.append(text)
            return Reply(content=json.dumps({"answers": {"label": {"choice": self.label}}}), in_tokens=50, out_tokens=5)
        self.prompts.append(text)
        code = self.script.pop(0) if self.script else 'FINAL("done")'
        return Reply(content="```python\n" + code + "\n```", in_tokens=1000, out_tokens=100)


def task(tid):
    return next(t for t in build_cm_tasks(cm_plan.CM_SEEDS) if t.task_id == tid)


def run(work, selectors, arm, tid, script, **kw):
    t = task(tid)
    w = World(work / f"{arm}_{tid}", ORCH)
    llm = ScriptedLLM(script, **kw)
    rec = run_episode_cm(t, arm, w, llm, 1000, selectors=selectors)
    return rec, w, llm


# --- plan integrity ---------------------------------------------------------------------------------------
def test_plan_sizes_and_every_task_is_built():
    idx = run_cm.task_index()
    sizes = {f: len(cm_plan.pairs(f)) for f in cm_plan.FEATURES}
    assert sizes == {"cm4": 50, "cm5": 40, "cm2": 30}
    for f in cm_plan.FEATURES:
        ps = cm_plan.pairs(f)
        assert len({t for t, _ in ps}) == len(ps) and all(t in idx for t, _ in ps)
    k4 = [k for _, k in cm_plan.pairs("cm4")]
    assert (k4.count("H"), k4.count("L"), k4.count("K")) == (30, 10, 10)
    assert {k for _, k in cm_plan.pairs("cm4")[:10]} >= {"H", "L", "K"}  # the first look already holds every kind
    assert [k for _, k in cm_plan.pairs("cm5")].count("P") == 10
    assert cm_plan.pairs("cm2")[0][0] == "N-CSV-n20-s0" and cm_plan.pairs("cm2")[9][0] == "S-FILT-n5-s0"


def test_new_tasks_are_deterministic_and_balanced():
    a, b = build_cm_tasks([0, 1]), build_cm_tasks([0, 1])
    assert [t.prompt for t in a] == [t.prompt for t in b]
    assert [t.expected.get("gold") for t in a] == [t.expected.get("gold") for t in b]
    for t in a:
        g = t.expected.get("gold")
        if t.family == "C-LABEL":
            assert set(g.values()) == {"billing", "shipping", "technical", "account"} and len(g) == t.n
        if t.family == "C-RISK":
            assert sorted(set(g.values())) == ["low", "medium"] and list(g.values()).count("low") == 10
        if t.family == "C-ROUTE":
            assert sorted(set(g.values())) == ["thinker", "worker"] and list(g.values()).count("worker") == 10
        if t.family == "H-PRICE":
            assert t.phases and t.phases[0] == t.prompt and t.blocked_in_phase2 == ("syn_sku_price",)


@pytest.mark.parametrize("tid", ["C-LABEL-n20-s0", "C-LABEL-n100-s1", "C-RISK-n20-s2", "C-ROUTE-n20-s3"])
def test_label_oracles(work, tid):
    t = task(tid)
    w = World(work / tid, ORCH)
    t.setup(w)
    rel = {"C-LABEL": "out/labels.json", "C-RISK": "out/risk.json", "C-ROUTE": "out/roles.json"}[t.family]
    assert t.oracle(w)[0] is False  # nothing written
    gold = t.expected["gold"]
    w.path("out").mkdir(parents=True, exist_ok=True)
    w.path(rel).write_text(json.dumps(gold))
    assert t.oracle(w)[0] is True
    tol = t.n // 10 if t.family == "C-LABEL" else 0  # C-LABEL tolerates 10 percent wrong, the classifier tasks none
    for n_wrong, want in ((tol, True), (tol + 1, False)):
        wrong = dict(gold)
        for k in list(wrong)[:n_wrong]:
            wrong[k] = "zzz"
        w.path(rel).write_text(json.dumps(wrong))
        assert t.oracle(w)[0] is want


def test_h_price_oracle_and_block(work):
    t = task("H-PRICE-n20-s0")
    w = World(work / "h", ORCH)
    t.setup(w)
    assert t.oracle(w)[0] is False
    w.path("out").mkdir(parents=True, exist_ok=True)
    w.path("out/total.json").write_text(json.dumps(t.expected["answer"]))
    assert t.oracle(w)[0] is True
    w.blocked.add("syn_sku_price")
    with pytest.raises(Exception, match="no longer available"):
        w.call("syn_sku_price", sku="SKU-0001")
    assert sku_price_cents("SKU-0001") > 0 and w.audit[-1]["ok"] is False


# --- arm plumbing ------------------------------------------------------------------------------------------
LABEL_CODE = ('items = json.loads(read_file("data/items.json"))\n'
              'labs = classify([i["text"] for i in items], ["billing", "shipping", "technical", "account"])\n'
              'file_write_safe("out/labels.json", json.dumps({i["id"]: l for i, l in zip(items, labs)}))\n'
              'FINAL("ok")')


def test_cm4_exposes_classifiers_counts_subcalls_and_c3_does_not(work, selectors):
    rec, _w, llm = run(work, selectors, "C3_cm4", "C-LABEL-n20-s0", [LABEL_CODE])
    assert rec["arm"] == "C3_cm4" and rec["turn_memory_n"] == 4
    assert {"classify", "route_classify", "risk_bands"} <= set(rec["cm"]["globals_present"])
    assert rec["sub_calls"] == 20 and rec["sub_in_tokens"] == 1000 and rec["sub_out_tokens"] == 100 and len(llm.sub_prompts) == 20
    assert rec["total_tokens_all"] == rec["in_tokens"] + rec["out_tokens"] + 1100
    assert rec["cm"]["uses"].get("classify") == 1 and rec["cm"]["flags"] == {"ORCHESTRATOR_REPL_CLASSIFY": "1"}
    assert ADDENDUM["C3_cm4"].split("`")[1] in llm.prompts[0] and "route_classify" in llm.prompts[0]
    assert rec["success"] is False  # every label "billing": a quarter correct, below the 0.90 bar
    ctl, _w, cl = run(work, selectors, "C3", "C-LABEL-n20-s0", [LABEL_CODE])
    assert not ({"classify", "route_classify", "risk_bands", "store", "load", "handles"} & set(ctl["cm"]["globals_present"]))
    assert ctl["sub_calls"] == 0 and "route_classify" not in cl.prompts[0] and ctl["success"] is False
    assert ctl["turn_failure_reasons"].get("name_error", 0) >= 1 or ctl["failed_snippets"] >= 1  # classify is undefined in the control


def test_cm4_typed_classifier_task_is_solvable_with_the_flag(work, selectors):
    t = task("C-RISK-n20-s0")
    code = ('qs = json.loads(read_file("data/queries.json"))\nrows = risk_bands([q["text"] for q in qs])\n'
            'file_write_safe("out/risk.json", json.dumps({q["id"]: r["risk_band"] for q, r in zip(qs, rows)}))\nFINAL("ok")')
    rec, _w, _l = run(work, selectors, "C3_cm4", t.task_id, [code])
    assert rec["success"] is True and rec["cm"]["uses"].get("risk_bands") == 1 and rec["sub_calls"] == 0


def test_cm5_two_phase_restore_keeps_handles_and_sandbox(work, selectors):
    p1 = ('rows = read_file("data/orders.csv").strip().splitlines()[1:]\n'
          'table = {s: syn_sku_price(s)["price_cents"] for s in sorted({r.split(",")[1] for r in rows})}\n'
          'h = store("prices", table)\nprint(h)\nFINAL("phase 1 done")')
    p2 = ('table = load("prices")\nrows = read_file("data/orders.csv").strip().splitlines()[1:]\n'
          'tot = sum(int(r.split(",")[2]) * table[r.split(",")[1]] for r in rows)\n'
          'file_write_safe("out/total.json", json.dumps({"rows": len(rows), "total_cents": tot}))\nFINAL("done")')
    rec, w, llm = run(work, selectors, "C3_cm5", "H-PRICE-n20-s0", [p1, p2])
    assert rec["success"] is True, rec["oracle"]
    assert rec["cm"]["phases_run"] == 2 and rec["cm"]["restored_into_fresh_repl"] is True and {"store", "load", "handles"} <= set(rec["cm"]["globals_present"])
    assert rec["turns"] == 2 and [t["phase"] for t in json.loads(json.dumps(rec["_transcript"]))] == [1, 2]
    assert "## Previous Turns" not in llm.prompts[1] and "PHASE 2 of 2" in llm.prompts[1] and "PHASE 1 of 2" in llm.prompts[0]
    assert w.blocked == {"syn_sku_price"} and rec["cm"]["uses"] == {"store": 1, "load": 1}


def test_c3_two_phase_restored_variable_survives_and_blocked_tool_fails(work, selectors):
    p1 = ('rows = read_file("data/orders.csv").strip().splitlines()[1:]\n'
          'table = {s: syn_sku_price(s)["price_cents"] for s in sorted({r.split(",")[1] for r in rows})}\nFINAL("phase 1 done")')
    p2 = ('rows = read_file("data/orders.csv").strip().splitlines()[1:]\n'
          'tot = sum(int(r.split(",")[2]) * table[r.split(",")[1]] for r in rows)\n'
          'file_write_safe("out/total.json", json.dumps({"rows": len(rows), "total_cents": tot}))\nFINAL("done")')
    rec, _w, _l = run(work, selectors, "C3", "H-PRICE-n20-s1", [p1, p2])
    assert rec["success"] is True and rec["cm"]["restored_into_fresh_repl"] is True  # plain restored variables are enough: the CM-5 question
    bad = ('rows = read_file("data/orders.csv").strip().splitlines()[1:]\nx = syn_sku_price(rows[0].split(",")[1])\nFINAL("x")')
    rec2, _w2, _l2 = run(work, selectors, "C3", "H-PRICE-n20-s1", [p1, bad, 'FINAL("give up")'])
    assert rec2["success"] is False and rec2["turn_failure_reasons"].get("tool_error", 0) + rec2["turn_failure_reasons"].get("other_error", 0) >= 1


def test_cm2_forces_core_search_and_search_tools_finds_the_price_tool(work, selectors):
    from scripts.benchmark.codemode_ab.tasks import build_tasks

    q = next(x for x in build_tasks([0]) if x.task_id == "N-QUOTE-s0")
    w = World(work / "cm2", ORCH)
    sku = q.expected["sku"]
    code = (f'names = search_tools("look up the unit price of a product SKU")\nprint(names)\nr = TOOL("syn_sku_price", sku="{sku}")\n'
            f'file_write_safe("out/quote.json", json.dumps({{"sku": "{sku}", "qty": {q.expected["qty"]}, "total_cents": {q.expected["qty"]} * r["price_cents"]}}))\nFINAL("ok")')
    llm = ScriptedLLM([code])
    rec = run_episode_cm(q, "C3_cm2", w, llm, 0, selectors=selectors)
    assert rec["success"] is True, rec["oracle"]
    assert rec["selector"]["mode"] == "core_search_forced" and rec["cm"]["forced_core_search"] is True
    assert "syn_sku_price" in rec["selector"]["chosen_unforced"] and rec["selector"]["k"] == 0  # the unforced pick would have listed it; cm2 withholds it
    assert "def syn_sku_price" not in llm.prompts[0] and "search_tools(query)" in llm.prompts[0] and ADDENDUM["C3_cm2"] in llm.prompts[0]
    assert rec["cm"]["uses"].get("search_tools") == 1 and "search_tools" in rec["cm"]["globals_present"]


def test_flags_and_harness_are_restored_after_every_episode(work, selectors):
    before = {k: os.environ.get(k) for k in CM_FLAGS + ("ORCHESTRATOR_REPL_TOOL_NAME_REPAIR",)}
    orig = (A.make_repl, A.pick_for_repl, A.run_arm_repl)
    os.environ["ORCHESTRATOR_REPL_CLASSIFY"] = "1"  # a stray flag must be CLEARED for the control, then restored
    try:
        rec, _w, _l = run(work, selectors, "C3", "C-LABEL-n20-s0", ['FINAL("x")'])
        assert "classify" not in rec["cm"]["globals_present"] and os.environ["ORCHESTRATOR_REPL_CLASSIFY"] == "1"
    finally:
        os.environ.pop("ORCHESTRATOR_REPL_CLASSIFY", None)
    assert (A.make_repl, A.pick_for_repl, A.run_arm_repl) == orig
    assert {k: os.environ.get(k) for k in CM_FLAGS + ("ORCHESTRATOR_REPL_TOOL_NAME_REPAIR",)} == before
    assert A.ARM_ENV["C3"]["ORCHESTRATOR_REPL_TURN_MEMORY"] == "4" and set(ARMS_CM) == {"C3", "C3_cm2", "C3_cm4", "C3_cm5"}
    with pytest.raises(ValueError):
        run(work, selectors, "C3_nomem", "C-LABEL-n20-s0", [])
