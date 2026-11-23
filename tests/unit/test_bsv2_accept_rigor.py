"""AP-BSV2-RIGOR-GATE-CODE: R1-R5 under AUTOPILOT_BSV2_ACCEPT_GATE (default off, fakes only)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "autopilot"))
sys.path.insert(0, str(ROOT))

import bsv2_accept_rigor as rig  # noqa: E402
from scripts.autopilot import actions  # noqa: E402


def _rows(spec: dict[str, bool], **extra):
    return [{"qid": q, "suite": "math", "correct": ok, "tokens_generated": 10, **extra} for q, ok in spec.items()]


def _state(**over):
    st = rig.new_state()
    st["proposal_qids"] = ["p1"]
    st["budget"] = {"unit": "rollouts", "limit": 100, "spent": 0}
    st.update(over)
    return st


def _eval(rows, details=None):
    return SimpleNamespace(question_results=rows, details=details if details is not None else {})


def _run(state, base, cand, fresh=True, paired=True, delta=0.5):
    return rig.evaluate_rigor(
        rigor_state=state, baseline_rows=base, candidate_rows=cand,
        baseline_fresh=fresh, paired_accept=paired, accuracy_delta=delta,
    )


BASE = _rows({"a": False, "b": False})
CAND = _rows({"a": True, "b": True})


def test_all_rules_pass_and_tokens_per_gain():
    rec = _run(_state(), BASE, CAND)
    assert rec["accept"] and not rec["blockers"]
    # delta 0.5 x 2 shared qids = 1 extra correct; 40 tokens
    assert rec["tokens"] == 40 and rec["tokens_per_accepted_gain"] == 40.0
    assert rec["budget_unit"] == "rollouts" and rec["round_cost"] == 4


def test_r1_overlap_and_unknown_fail_closed():
    assert not _run(_state(proposal_qids=["a"]), BASE, CAND)["accept"]
    unknown = _run(_state(proposal_qids=None), BASE, CAND)
    assert not unknown["accept"] and "unknown" in unknown["rules"]["R1"]["reason"]


def test_r2_stale_parent_blocks():
    rec = _run(_state(), BASE, CAND, fresh=False)
    assert not rec["accept"] and any(b.startswith("R2") for b in rec["blockers"])


def test_r3_sealed_in_eval_or_already_exposed_blocks():
    assert not _run(_state(sealed_qids=["a"]), BASE, CAND)["accept"]
    assert not _run(_state(sealed_qids=["z"], exposed_qids=["z"]), BASE, CAND)["accept"]


def test_r3_headline_checkpoint_training_side_only():
    cps = [{"id": 1, "train_score": 0.4, "test_score": 0.9}, {"id": 2, "train_score": 0.6, "test_score": 0.1}]
    assert rig.select_headline_checkpoint(cps)["id"] == 2
    with pytest.raises(ValueError):
        rig.select_headline_checkpoint(cps, key="test_score")
    assert rig.select_headline_checkpoint([]) is None


def test_r4_status_split_retry_and_undefined_gain():
    assert not rig.retry_allowed("solver_limit") and rig.retry_allowed("infra_error")
    assert rig.row_status({"status": "weird"}) == "candidate_error"
    cand = _rows({"a": True, "b": True})
    cand[0]["status"] = "infra_error"
    rec = _run(_state(), BASE, cand)
    assert not rec["accept"] and rec["rules"]["R4"]["status_split"]["candidate"]["infra_error"] == 1
    assert rec["tokens_per_accepted_gain"] is None  # undefined, not zero
    assert _run(_state(), BASE, CAND, paired=False)["tokens_per_accepted_gain"] is None


def test_r5_unit_and_budget():
    assert not _run(_state(budget={"unit": "iterations", "limit": 5, "spent": 0}), BASE, CAND)["accept"]
    assert not _run(_state(budget={"unit": "rollouts", "limit": 3, "spent": 0}), BASE, CAND)["accept"]
    tok = _run(_state(budget={"unit": "tokens", "limit": 1000, "spent": 0}), BASE, CAND)
    assert tok["accept"] and tok["round_cost"] == 40


def test_commit_round_spends_budget_and_exposes_qids():
    st = _state()
    rig.commit_round(st, _run(st, BASE, CAND))
    assert st["budget"]["spent"] == 4 and st["exposed_qids"] == ["a", "b"] and len(st["rounds"]) == 1
    # the same qids are now burned for a later round's sealed set
    assert not _run(_state(sealed_qids=["a"], exposed_qids=st["exposed_qids"]), BASE, CAND)["accept"]


# --- wiring through actions._bsv2_accepts ---------------------------------------------------------------


def _ctx(state=None):
    return SimpleNamespace(state=state if state is not None else {})


def _gate(ctx, base, cand, monkeypatch, rigor_on):
    monkeypatch.setenv("AUTOPILOT_BSV2_ACCEPT_GATE", "1")
    monkeypatch.setenv("AUTOPILOT_BSV2_MIN_SHARED_QIDS", "2")
    if rigor_on:
        monkeypatch.setenv("AUTOPILOT_BSV2_ACCEPT_RIGOR", "1")
    else:
        monkeypatch.delenv("AUTOPILOT_BSV2_ACCEPT_RIGOR", raising=False)
    b = actions.EvalResult(tier=1, quality=1, speed=1, cost=1, reliability=1, question_results=base)
    c = actions.EvalResult(tier=1, quality=1, speed=1, cost=1, reliability=1, question_results=cand)
    ok = actions._bsv2_accepts(
        baseline_result=b, candidate_result=c, artifact_kind="prompt", target="t", mutation_type="m", ctx=ctx
    )
    return ok, c.details["bsv2_accept_gate"]


def test_wiring_rigor_off_is_legacy_behaviour(monkeypatch):
    ctx = _ctx()
    ok, detail = _gate(ctx, BASE, CAND, monkeypatch, rigor_on=False)
    assert ok and "rigor" not in detail and "bsv2_rigor" not in ctx.state


def test_wiring_rigor_on_fails_closed_without_proposal_qids(monkeypatch):
    ctx = _ctx()
    ok, detail = _gate(ctx, BASE, CAND, monkeypatch, rigor_on=True)
    assert not ok and detail["accept"] is False
    assert detail["rigor"]["rules"]["R1"]["ok"] is False


def test_wiring_rigor_on_accepts_when_configured_and_records_round(monkeypatch):
    st = {"bsv2_rigor": _state()}
    ctx = _ctx(st)
    ok, detail = _gate(ctx, BASE, CAND, monkeypatch, rigor_on=True)
    assert ok and detail["rigor"]["tokens_per_accepted_gain"] is not None
    assert st["bsv2_rigor"]["budget"]["spent"] == 4 and len(st["bsv2_rigor"]["rounds"]) == 1


def test_flags_registered_default_off_and_pinned_observe_only():
    from scripts.autopilot.config_applicator import HOT_SWAP_FEATURES
    from src.features import _REGISTRY_BY_NAME, Features

    for name, env in (("autopilot_bsv2_accept_gate", "AUTOPILOT_BSV2_ACCEPT_GATE"),
                      ("autopilot_bsv2_accept_rigor", "AUTOPILOT_BSV2_ACCEPT_RIGOR")):
        spec = _REGISTRY_BY_NAME[name]
        assert spec.default_test is False and spec.default_prod is False and spec.env_name == env
        assert getattr(Features(), name) is False
        assert name not in HOT_SWAP_FEATURES  # AP-BSV2-RIGOR-PIN: the proposer cannot flip its own gate


# --- AP-BSV2-RIGOR-STATE: loop-populated state + proposal-disjoint accept partition -------------------------


class _FakeTower:
    """Pool of 40 qids; ``hybrid_eval`` draws 6 deterministically, honouring ``bsv2_exclude_qids`` like eval_t1."""

    def __init__(self):
        self.bsv2_exclude_qids = None
        self.pool = {"math": [{"qid": f"q{i:02d}", "prompt": f"p{i}", "suite": "math"} for i in range(40)]}

    def _load_pool(self):
        return self.pool

    def draw(self):
        ex = self.bsv2_exclude_qids or set()
        return [q["qid"] for q in self.pool["math"] if q["qid"] not in ex][:6]


def _loop_ctx(monkeypatch, traces=""):
    monkeypatch.setenv("AUTOPILOT_BSV2_ACCEPT_GATE", "1")
    monkeypatch.setenv("AUTOPILOT_BSV2_ACCEPT_RIGOR", "1")
    monkeypatch.setenv("AUTOPILOT_BSV2_MIN_SHARED_QIDS", "2")
    return SimpleNamespace(tower=_FakeTower(), state={"last_traces": traces} if traces else {})


def _eval_pair(qids, cand_ok=True):
    b = [{"qid": q, "suite": "math", "correct": False, "tokens_generated": 10} for q in qids]
    c = [{"qid": q, "suite": "math", "correct": cand_ok, "tokens_generated": 10} for q in qids]
    return b, c


def test_state_loop_populates_and_a_disjoint_accept_passes(monkeypatch):
    ctx = _loop_ctx(monkeypatch)
    assert actions._bsv2_rigor_prepare(ctx) is True
    st = ctx.state["bsv2_rigor"]
    assert st["proposal_qids"] == [] and st["budget"]["unit"] in rig.BUDGET_UNITS and st["budget"]["limit"]
    assert st["sealed_qids"] and set(st["sealed_qids"]) < {f"q{i:02d}" for i in range(40)}
    drawn = ctx.tower.draw()
    assert drawn and not set(drawn) & set(st["sealed_qids"])
    b, c = _eval_pair(drawn)
    ok, detail = _gate(ctx, b, c, monkeypatch, rigor_on=True)
    assert ok and detail["rigor"]["rules"]["R1"]["ok"] and detail["rigor"]["rules"]["R3"]["ok"]
    # the eval just run becomes the next proposal batch, and the next draw is disjoint from it
    actions._bsv2_rigor_note_eval(ctx, SimpleNamespace(question_results=c))
    ctx.tower.bsv2_exclude_qids = None
    assert actions._bsv2_rigor_prepare(ctx) is True
    assert set(ctx.state["bsv2_rigor"]["proposal_qids"]) == set(drawn)
    assert not set(ctx.tower.draw()) & set(drawn)


def test_state_overlapping_accept_is_refused(monkeypatch):
    ctx = _loop_ctx(monkeypatch)
    actions._bsv2_rigor_prepare(ctx)
    st = ctx.state["bsv2_rigor"]
    overlap = ctx.tower.draw()[:3]
    st["proposal_qids"] = sorted(overlap)  # the proposer saw these
    b, c = _eval_pair(overlap)
    ok, detail = _gate(ctx, b, c, monkeypatch, rigor_on=True)
    assert not ok and detail["rigor"]["rules"]["R1"]["ok"] is False
    # sealed qids in the accept eval are refused too (R3)
    sealed = st["sealed_qids"][:3]
    st["proposal_qids"] = []
    b, c = _eval_pair(sealed)
    ok, detail = _gate(ctx, b, c, monkeypatch, rigor_on=True)
    assert not ok and detail["rigor"]["rules"]["R3"]["ok"] is False


def test_state_unknown_proposal_batch_with_traces_fails_closed(monkeypatch):
    ctx = _loop_ctx(monkeypatch, traces="some earlier traces")
    assert actions._bsv2_rigor_prepare(ctx) is True
    assert ctx.state["bsv2_rigor"]["proposal_qids"] is None
    b, c = _eval_pair(ctx.tower.draw())
    ok, _ = _gate(ctx, b, c, monkeypatch, rigor_on=True)
    assert not ok


def test_state_prepare_is_inert_when_flags_off(monkeypatch):
    monkeypatch.delenv("AUTOPILOT_BSV2_ACCEPT_GATE", raising=False)
    monkeypatch.delenv("AUTOPILOT_BSV2_ACCEPT_RIGOR", raising=False)
    ctx = SimpleNamespace(tower=_FakeTower(), state={})
    assert actions._bsv2_rigor_prepare(ctx) is False
    assert "bsv2_rigor" not in ctx.state and ctx.tower.bsv2_exclude_qids is None
