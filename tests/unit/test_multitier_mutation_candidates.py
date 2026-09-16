"""Operator decision 2026-09-16: mutation candidates in the live multitier launcher.

Prompt / code / GEPA mutation candidates with a served-file sha identity can be staged,
validated at T2/T3, re-served at final_t1 and promoted. Every stage re-serves the SAME
content: the sha is verified before the forced eval and re-hashed after it. A mismatch or
a missing sha refuses; a rejection restores the recorded preimage and attests its sha.
Numeric / structural staging is unchanged.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT, REPO_ROOT / "scripts" / "autopilot"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import actions  # noqa: E402
import autopilot  # noqa: E402
from src.autopilot_core.multitier_decision import (  # noqa: E402
    MULTITIER_POLICY_VERSION,
    build_tier_baseline_evidence,
)

REL = "orchestration/prompts/frontdoor.md"
PROMPT = {"type": "prompt_mutation", "file": "frontdoor.md", "mutation": "targeted_fix"}


class _Verdict:
    def __init__(self, passed: bool = True):
        self.passed = passed
        self.seq = None
        self.violations = []

    def __bool__(self):
        return self.passed


def _result(*, tier: int, outcomes: dict[str, bool], quality: float | None = None):
    rows = [{"qid": qid, "correct": value} for qid, value in outcomes.items()]
    return SimpleNamespace(
        tier=tier,
        quality=quality if quality is not None else 3.0 * sum(outcomes.values()) / len(outcomes),
        reliability=1.0,
        core_id=f"core-t{tier}",
        dataset_content_sha256=f"dataset-t{tier}",
        test_profile=f"profile-t{tier}",
        question_results=rows,
        details={},
    )


T2 = {f"q{i}": i % 2 == 0 for i in range(100)}
T3 = {f"h{i}": i % 3 == 0 for i in range(100)}


def _state():
    return {
        autopilot.MULTITIER_BASELINE_STATE_KEY: {
            "policy_version": MULTITIER_POLICY_VERSION,
            "tiers": {
                "2": build_tier_baseline_evidence(_result(tier=2, outcomes=T2)),
                "3": build_tier_baseline_evidence(_result(tier=3, outcomes=T3)),
            },
        }
    }


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


@pytest.fixture
def served(tmp_path, monkeypatch):
    """A served prompt file under a temp orchestrator root, and its handler record."""
    monkeypatch.setattr(autopilot, "MULTITIER_PROMOTION_ENABLED", True)
    monkeypatch.setattr(autopilot, "MULTITIER_MAX_ATTEMPTS_PER_TIER", 3)
    monkeypatch.setattr(autopilot, "ORCH_ROOT", tmp_path)
    monkeypatch.setattr(actions, "ORCH_ROOT", tmp_path)
    target = tmp_path / REL
    target.parent.mkdir(parents=True)
    target.write_text("mutated v2")
    record = actions.served_content_record([target], root=tmp_path)
    record["restore"] = {
        "kind": "prompt",
        "file": "frontdoor.md",
        "new_file": False,
        "preimage": "original v1",
        "preimage_sha256": _sha("original v1"),
    }
    return SimpleNamespace(root=tmp_path, target=target, record=record)


def _gate(quality=1.0):
    return SimpleNamespace(
        use_sequential=False,
        baseline=SimpleNamespace(quality_for_tier=lambda tier, strict=False: quality),
    )


def _eligible(state, served_content, action=PROMPT):
    return autopilot._multitier_candidate_is_eligible(
        state=state,
        gate=_gate(),
        action=dict(action),
        eval_result=SimpleNamespace(tier=1, quality=1.5),
        verdict=_Verdict(),
        pareto_status="frontier",
        served_content=served_content,
    )


def _stage(state, record):
    return autopilot._start_multitier_validation(
        state,
        action=dict(PROMPT),
        eval_result=_result(tier=1, outcomes={"c": True}),
        verdict=_Verdict(),
        trial_counter=10,
        served_content=record,
    )


def _run_to_final(state, trial=11):
    for tier, outcomes in ((2, T2), (3, T3)):
        forced, _, context = autopilot._maybe_force_multitier_due_action(
            state=state, blacklist=[], trial_counter=trial
        )
        assert forced == {"type": "deep_eval", "tier": tier}, forced
        assert autopilot._multitier_validation_served_content(state, context, trial)
        autopilot._record_multitier_validation_result(
            state, context=context, eval_result=_result(tier=tier, outcomes=outcomes),
            verdict=_Verdict(), trial_counter=trial,
        )
        trial += 1
    return trial


# ── staging ───────────────────────────────────────────────────────────────


def test_mutation_with_sha_is_eligible_and_staged(served):
    state = _state()
    assert _eligible(state, served.record) == (True, "ready")
    pending = _stage(state, served.record)
    assert pending["candidate_served_content"] == {"files": {REL: _sha("mutated v2")}}
    assert pending["candidate_restore"]["preimage"] == "original v1"
    # Nothing new on the action: fingerprints / journal rows unchanged.
    assert pending["candidate_action"] == PROMPT
    assert pending["candidate"] == autopilot._config_fingerprint(PROMPT)


@pytest.mark.parametrize("action", [
    PROMPT,
    {"type": "code_mutation", "file": "src/tool_policy.py", "mutation": "targeted_fix"},
    {"type": "gepa_optimize", "file": "frontdoor.md", "max_evals": 50},
])
def test_missing_sha_refuses_staging(served, action):
    eligible, reason = _eligible(_state(), None, action)
    assert eligible is False and "no served-file identity" in reason


def test_missing_preimage_refuses_staging(served):
    record = dict(served.record)
    record.pop("restore")
    eligible, reason = _eligible(_state(), record)
    assert eligible is False and "lacks an exact restore preimage" in reason


def test_structural_prune_stays_out_of_scope(served):
    eligible, reason = _eligible(
        _state(), served.record, {"type": "structural_prune", "file": "frontdoor.md"}
    )
    assert eligible is False and "not replayable" in reason


# ── validation re-serves the same content ────────────────────────────────


def test_matching_content_runs_to_final_t1_with_identity(served):
    state = _state()
    _stage(state, served.record)
    trial = _run_to_final(state)
    forced, _, context = autopilot._maybe_force_multitier_due_action(
        state=state, blacklist=[], trial_counter=trial
    )
    assert forced == {"type": "deep_eval", "tier": 1} and context["stage"] == "final_t1"
    served_now = autopilot._multitier_validation_served_content(state, context, trial)
    assert served_now["files"] == {REL: _sha("mutated v2")}
    # The final_t1 row is identified by what it served, like the candidate row.
    effective = autopilot._multitier_effective_action({"type": "deep_eval", "tier": 1}, context)
    row_form = autopilot._served_content_for_row(served_now)
    from src.autopilot_core.action_identity import action_config_identity

    assert action_config_identity(effective, "r", row_form) == action_config_identity(
        PROMPT, "r", autopilot._served_content_for_row(served.record)
    )


def test_sha_mismatch_before_replay_refuses_and_rolls_back(served):
    state = _state()
    _stage(state, served.record)
    served.target.write_text("re-mutated v3")
    forced, rationale, context = autopilot._maybe_force_multitier_due_action(
        state=state, blacklist=[], trial_counter=11
    )
    assert forced == {"type": "rollback", "to_checkpoint": "production_best"}
    assert "file sha changed since staging" in rationale["reason"]
    assert context["stage"] == "rollback"


def test_reverted_file_cannot_be_reserved(served):
    state = _state()
    _stage(state, served.record)
    served.target.unlink()
    forced, rationale, _ = autopilot._maybe_force_multitier_due_action(
        state=state, blacklist=[], trial_counter=11
    )
    assert forced["type"] == "rollback"
    assert "cannot be re-served" in rationale["reason"]


def test_content_change_during_eval_rejects_and_withholds_identity(served):
    state = _state()
    _stage(state, served.record)
    trial = _run_to_final(state)
    _, _, context = autopilot._maybe_force_multitier_due_action(
        state=state, blacklist=[], trial_counter=trial
    )
    served.target.write_text("changed during eval")
    assert autopilot._multitier_validation_served_content(state, context, trial) is None
    pending = state[autopilot.MULTITIER_PENDING_STATE_KEY]
    assert pending["status"] == "rejected" and state["multitier_rollback_pending"] is True
    assert "during validation" in pending["blocked_reason"]
    # A rejected candidate records no further stage results.
    assert autopilot._record_multitier_validation_result(
        state, context=context, eval_result=_result(tier=1, outcomes={"c": True}),
        verdict=_Verdict(), trial_counter=trial,
    ) is None


def test_promotion_accepted_after_matching_final_t1(served):
    state = _state()
    _stage(state, served.record)
    trial = _run_to_final(state)
    _, _, context = autopilot._maybe_force_multitier_due_action(
        state=state, blacklist=[], trial_counter=trial
    )
    assert autopilot._multitier_validation_served_content(state, context, trial)
    autopilot._record_multitier_validation_result(
        state, context=context, eval_result=_result(tier=1, outcomes={"c": True}),
        verdict=_Verdict(), trial_counter=trial,
    )
    autopilot._finish_multitier_promotion(
        state, baseline_update=SimpleNamespace(updated=True, reason="promoted"),
        trial_counter=trial,
    )
    assert state["multitier_last_accepted"]["candidate"] == autopilot._config_fingerprint(PROMPT)
    assert autopilot.MULTITIER_PENDING_STATE_KEY not in state


@pytest.fixture
def guard_provider_reset():
    import safety_gate

    safety_gate.configure_promotion_guard_archive(None)
    yield
    safety_gate.configure_promotion_guard_archive(None)


def test_final_t1_update_baseline_promotes_on_matching_reproductions(
    tmp_path, guard_provider_reset
):
    """The final_t1 decision itself: re-served content clusters with the candidate row."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_gate_frontier_c_and_b import Loop  # the real provider + journal harness

    loop = Loop(tmp_path)
    served = {"files": {REL: _sha("mutated v2")}}
    decisions = [
        loop.trial("p", 1.8, action=dict(PROMPT), served=served, infra_digest="r")[1]
        for _ in range(3)  # candidate trial, final_t1 attempt 1, final_t1 attempt 2
    ]
    assert [u.updated for u in decisions] == [False, False, True], [u.reason for u in decisions]
    mismatched = loop.trial(
        "p", 2.2, action=dict(PROMPT), served={"files": {REL: _sha("other")}}, infra_digest="r"
    )[1]
    assert not mismatched.updated


# ── rollback restores the preimage ────────────────────────────────────────


class _Forge:
    def __init__(self, root: Path):
        self.root = root

    def _resolve_prompt_path(self, filename):
        return self.root / "orchestration" / "prompts" / filename

    def revert_mutation(self, mutation):
        self._resolve_prompt_path(mutation.file).write_text(mutation.original_content)

    def revert_code_mutation(self, mutation):
        path = self.root / mutation.file
        if mutation.mutation_type == "new_file" and not mutation.original_content:
            path.unlink(missing_ok=True)
        else:
            path.write_text(mutation.original_content)


def _ctx(root, state):
    return SimpleNamespace(
        forge=_Forge(root),
        state=state,
        lab=None,
        gate=SimpleNamespace(reset_failures=lambda: None),
        tower=SimpleNamespace(hybrid_eval=lambda: "evaluated"),
    )


def test_restore_writes_and_attests_the_preimage(served):
    result = actions._restore_mutation_preimage(
        _ctx(served.root, {}), served.record["restore"]
    )
    assert result["status"] == "ok" and served.target.read_text() == "original v1"


def test_restore_attestation_fails_on_wrong_sha(served):
    bad = dict(served.record["restore"], preimage_sha256=_sha("something else"))
    result = actions._restore_mutation_preimage(_ctx(served.root, {}), bad)
    assert result["status"] == "error" and "attestation failed" in result["error"]


def test_restore_removes_a_rejected_new_file(served):
    new = served.root / "src" / "new_module.py"
    new.parent.mkdir(parents=True)
    new.write_text("X = 1\n")
    restore = {"kind": "code", "file": "src/new_module.py", "new_file": True,
               "preimage": "", "preimage_sha256": _sha("")}
    result = actions._restore_mutation_preimage(_ctx(served.root, {}), restore)
    assert result == {"status": "ok", "file": "src/new_module.py", "removed": True}
    assert not new.exists()


def test_rollback_action_restores_a_mutation_candidate(served, monkeypatch):
    state = _state()
    _stage(state, served.record)
    state["multitier_rollback_pending"] = True
    checkpoints = []
    ctx = _ctx(served.root, state)
    ctx.lab = SimpleNamespace(restore_checkpoint=lambda *a: checkpoints.append(a) or {"status": "ok"})
    outcome, _species = actions._action_rollback(
        {"type": "rollback", "to_checkpoint": "production_best"}, ctx
    )
    assert outcome == "evaluated"
    assert served.target.read_text() == "original v1"
    assert checkpoints and "multitier_rollback_pending" not in state
    assert state["multitier_last_event"]["runtime_restore"]["status"] == "ok"


def test_rollback_action_fails_closed_without_preimage(served):
    state = _state()
    _stage(state, served.record)
    state[autopilot.MULTITIER_PENDING_STATE_KEY].pop("candidate_restore")
    state["multitier_rollback_pending"] = True
    ctx = _ctx(served.root, state)
    ctx.lab = SimpleNamespace(restore_checkpoint=lambda *a: {"status": "ok"})
    outcome, _species = actions._action_rollback(
        {"type": "rollback", "to_checkpoint": "production_best"}, ctx
    )
    assert outcome.status == "skipped" and "restore preimage" in outcome.reason


# ── existing behaviour unchanged ─────────────────────────────────────────


@pytest.mark.parametrize("action, expected", [
    ({"type": "numeric_trial", "params": {"routing.x": 1}}, ""),
    ({"type": "numeric_trial", "params": {}}, "candidate numeric_trial lacks replayable applied params"),
    ({"type": "structural_experiment", "flags": {"a": True}}, ""),
    ({"type": "seed_batch"}, "candidate action type is not replayable: seed_batch"),
])
def test_numeric_and_structural_blocker_unchanged(action, expected):
    assert autopilot._seq_promotion_replay_blocker(action) == expected
    assert autopilot._seq_promotion_replay_blocker(action, {"files": {REL: _sha("x")}}) == expected


def test_seq_paths_still_block_mutations():
    """Only multitier staging passes an identity; seq fresh-eval / replay selection do not."""
    assert "no served-file identity" in autopilot._seq_promotion_replay_blocker(dict(PROMPT))
    source = Path(autopilot.__file__).read_text()
    assert "replay_blocker = _seq_promotion_replay_blocker(candidate_action)\n" in source
    assert "if _seq_promotion_replay_blocker(action):\n" in source


def test_no_new_promotion_call_site_for_the_ap55_hold_pattern():
    """Mutation candidates promote through the existing final_t1 call, which the AP-55
    merge train guards; the loop still has exactly three update_baseline call sites."""
    source = Path(autopilot.__file__).read_text()
    body = source[source.index("def _run_loop_inner(") :]
    assert body.count("gate.update_baseline(\n") == 3
