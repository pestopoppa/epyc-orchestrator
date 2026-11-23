"""AP-LH-DISTILL: Life-Harness four-layer taxonomy + zero-model trace-replay pre-check (fakes only, default off)."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "autopilot"))

hl = importlib.import_module("harness_layers")
actions = importlib.import_module("actions")
controller_io = importlib.import_module("controller_io")

TAX_ENV = hl.LAYER_TAXONOMY_ENV
REPLAY_ENV = hl.TRACE_REPLAY_PRECHECK_ENV

SYNTAX_TRACE = "ROLE frontdoor\nPROMPT:\nx\nRESPONSE:\nSyntaxError: invalid syntax\nSyntaxError: invalid syntax again\n"
CALM_TRACE = "ROLE frontdoor\nPROMPT:\nsolve\nRESPONSE:\nFINAL(42)\n"
LOOP_TRACE = "turn\nclick the red door\nclick the red door\nclick the red door\n"
CONTRACT_TRACE = "RESPONSE:\nNameError: name 'peek_all' is not defined\n"


def _fail(trace: str, trial_id: int = 1, outcome: str = "failure") -> dict:
    return {"outcome": outcome, "trial_id": trial_id, "trace": trace, "trace_hash": f"h{trial_id}"}


def _structural(flag: str, value: bool = True, **extra) -> dict:
    return {"type": "structural_experiment", "flags": {flag: value}, **extra}


# ----- the taxonomy maps onto EXISTING surfaces only -----


def test_every_mapped_surface_already_exists() -> None:
    from src.autopilot_core.mutation_guards import CODE_MUTATION_ALLOWLIST
    from src.features import _FEATURE_REGISTRY

    registry = {spec.name for spec in _FEATURE_REGISTRY}
    prompts = ROOT / "orchestration" / "prompts"
    for layer, surf in hl.SURFACES.items():
        assert layer in hl.LAYERS
        for flag in surf["flags"]:
            assert flag in registry, f"{layer}: {flag} is not a registered feature flag"
        for name in surf["prompt_files"]:
            assert (prompts / name).is_file(), f"{layer}: {name} is not a prompt file"
        for name in surf["code_files"]:
            assert name in CODE_MUTATION_ALLOWLIST, f"{layer}: {name} is not an allowlisted code file"
        for action in surf["actions"]:
            assert action in controller_io._ACTION_SCHEMAS
    for name in hl._UNLAYERED_CODE_FILES:
        assert name in CODE_MUTATION_ALLOWLIST
    for name in hl._UNLAYERED_PROMPT_FILES:
        assert (prompts / name).is_file()


def test_no_surface_belongs_to_two_layers() -> None:
    for field in ("flags", "prompt_files", "code_files", "actions"):
        seen: list[str] = []
        for surf in hl.SURFACES.values():
            seen.extend(surf[field])
        assert len(seen) == len(set(seen)), field


def test_every_replay_predicate_flag_is_a_mapped_surface() -> None:
    mapped = {flag for surf in hl.SURFACES.values() for flag in surf["flags"]}
    assert set(hl.FLAG_FIRING_PREDICATES) <= mapped


def test_planner_schema_vocabulary_matches_module_and_validates() -> None:
    assert controller_io._HARNESS_LAYERS == set(hl.LAYER_VOCABULARY)
    ok = {"type": "prompt_mutation", "file": "tools.md", "layer": "environment_contract"}
    assert controller_io.validate_single_variable(ok) is None
    bad = dict(ok, layer="vibes")
    assert "layer must be one of" in (controller_io.validate_single_variable(bad) or "")
    flag = {"type": "structural_experiment", "flags": {"repl_syntax_loop_breaker": True}, "layer": "action_realization"}
    assert controller_io.validate_single_variable(flag) is None


def test_layer_annotation_does_not_change_config_identity() -> None:
    from src.autopilot_core.action_identity import config_fingerprint

    base = _structural("repl_syntax_loop_breaker")
    assert config_fingerprint(base) == config_fingerprint(dict(base, layer="action_realization"))


# ----- surface -> layer -----


@pytest.mark.parametrize(
    ("action", "expected"),
    [
        (_structural("repl_syntax_loop_breaker"), ("action_realization", "flag")),
        (_structural("codemode_prompt_v2"), ("environment_contract", "flag")),
        (_structural("harness_failure_ratio_guard"), ("trajectory_regulation", "flag")),
        (_structural("skillbank"), ("procedural_skill", "flag")),
        (_structural("memrl"), ("flag", "flag")),
        ({"type": "prompt_mutation", "file": "tools.md"}, ("environment_contract", "prompt_file")),
        ({"type": "gepa_optimize", "file": "orchestration/prompts/rules.md"}, ("procedural_skill", "prompt_file")),
        ({"type": "prompt_mutation", "file": "frontdoor.md"}, ("routing", "prompt_file")),
        ({"type": "prompt_mutation", "file": "roles/worker_math.md"}, ("prompt", "prompt_file")),
        ({"type": "code_mutation", "file": "src/escalation.py"}, ("trajectory_regulation", "code_file")),
        ({"type": "code_mutation", "file": "src/api/routes/chat.py"}, ("routing", "code_file")),
        ({"type": "distill_skillbank"}, ("procedural_skill", "action")),
    ],
)
def test_surface_layer_lookup(action, expected) -> None:
    layer, kind, _name = hl.surface_layer(action)
    assert (layer, kind) == expected


# ----- diagnosis -----


def test_classify_trace_signatures_and_residual() -> None:
    assert hl.classify_trace(SYNTAX_TRACE)["layer"] == "action_realization"
    assert hl.classify_trace(CONTRACT_TRACE)["layer"] == "environment_contract"
    loop = hl.classify_trace(LOOP_TRACE)
    assert loop["layer"] == "trajectory_regulation" and "repeated_line" in loop["signatures"]
    residual = hl.classify_trace(CALM_TRACE)
    assert residual == {"layer": "procedural_skill", "basis": "residual", "signatures": []}


def test_diagnosis_picks_majority_layer_with_priority_tiebreak() -> None:
    traces = [
        {"trace": SYNTAX_TRACE, "trial_id": 1, "trace_hash": "a"},
        {"trace": SYNTAX_TRACE, "trial_id": 2, "trace_hash": "b"},
        {"trace": LOOP_TRACE, "trial_id": 3, "trace_hash": "c"},
    ]
    diag = hl.diagnose_failure_traces(traces)
    assert diag["implicated"] == "action_realization"
    assert diag["counts"]["action_realization"] == 2 and diag["n_failures"] == 3
    tie = hl.diagnose_failure_traces(traces[1:])  # one action_realization, one trajectory_regulation
    assert tie["implicated"] == "action_realization"  # LAYERS order breaks the tie
    assert hl.diagnose_failure_traces([])["implicated"] is None


def test_motivating_traces_are_the_newest_failures_only() -> None:
    state = {
        "contrastive_trace_bank": [
            _fail("old", 1), _fail("won", 2, outcome="success"), _fail("mid", 3), _fail("new", 4),
        ]
    }
    got = hl.motivating_failure_traces(state)
    assert [t["trial_id"] for t in got] == [3, 4]
    assert hl.motivating_failure_traces({}) == []


def test_layer_block_names_implicated_layer_and_only_existing_surfaces() -> None:
    diag = hl.diagnose_failure_traces([{"trace": SYNTAX_TRACE, "trial_id": 1, "trace_hash": "a"}])
    block = hl.render_layer_block(diag)
    assert "implicate layer **action_realization**" in block
    assert "- action_realization (implicated):" in block
    assert "repl_syntax_loop_breaker" in block and "tools.md" in block
    assert hl.render_layer_block(hl.diagnose_failure_traces([])) == ""
    assert hl.render_layer_block(None) == ""


def test_layer_record_declared_surface_and_alignment() -> None:
    diag = hl.diagnose_failure_traces([{"trace": SYNTAX_TRACE, "trial_id": 1, "trace_hash": "a"}])
    aligned = hl.build_layer_record(_structural("repl_syntax_loop_breaker"), diag)
    assert (aligned["layer"], aligned["source"], aligned["aligned"]) == ("action_realization", "surface", True)
    off = hl.build_layer_record(_structural("harness_failure_ratio_guard"), diag)
    assert off["layer"] == "trajectory_regulation" and off["aligned"] is False
    declared = hl.build_layer_record(_structural("memrl", layer="procedural_skill"), diag)
    assert (declared["layer"], declared["source"], declared["declared"]) == ("procedural_skill", "declared", "procedural_skill")
    junk = hl.build_layer_record(_structural("memrl", layer="vibes"), diag)
    assert junk["declared_invalid"] is True and junk["source"] == "surface"
    legacy = hl.build_layer_record({"type": "prompt_mutation", "file": "frontdoor.md"}, diag)
    assert legacy["layer"] == "routing" and legacy["aligned"] is None
    assert hl.build_layer_record(_structural("memrl"), None)["implicated"] is None


# ----- trace replay -----


def _traces(*texts: str) -> list[dict]:
    return [{"trace": t, "trial_id": i, "trace_hash": f"h{i}"} for i, t in enumerate(texts)]


def test_replay_would_fire_when_the_guard_matches_a_motivating_trace() -> None:
    rec = hl.replay_precheck(_structural("repl_syntax_loop_breaker"), _traces(CALM_TRACE, SYNTAX_TRACE))
    assert rec["status"] == "would_fire" and rec["fired"] == 1 and rec["motivating"] == 2
    assert not hl.replay_rejects(rec)


def test_replay_rejects_only_a_proved_noop_on_complete_traces() -> None:
    rec = hl.replay_precheck(_structural("repl_syntax_loop_breaker"), _traces(CALM_TRACE, CONTRACT_TRACE))
    assert rec["status"] == "would_not_fire" and rec["fired"] == 0
    assert hl.replay_rejects(rec)


def test_replay_truncated_trace_is_inconclusive_not_a_null() -> None:
    rec = hl.replay_precheck(_structural("repl_syntax_loop_breaker"), _traces("[trace truncated]\n" + CALM_TRACE))
    assert rec["status"] == "not_replayable" and rec["truncated"] == 1
    assert not hl.replay_rejects(rec)


@pytest.mark.parametrize(
    ("action", "reason_part"),
    [
        ({"type": "prompt_mutation", "file": "tools.md"}, "no deterministic replay model"),
        ({"type": "code_mutation", "file": "src/escalation.py"}, "no deterministic replay model"),
        ({"type": "gepa_optimize", "file": "tools.md"}, "no deterministic replay model"),
        (_structural("repl_syntax_loop_breaker", value=False), "off has no replay model"),
        (_structural("memrl"), "no firing predicate"),
        ({"type": "structural_experiment", "flags": {"a": True, "b": True}}, "exactly one flag"),
    ],
)
def test_replay_not_replayable_cases_never_reject(action, reason_part) -> None:
    rec = hl.replay_precheck(action, _traces(SYNTAX_TRACE))
    assert rec["status"] == "not_replayable" and reason_part in rec["reason"]
    assert not hl.replay_rejects(rec)


def test_replay_without_motivating_traces_does_not_reject() -> None:
    rec = hl.replay_precheck(_structural("repl_syntax_loop_breaker"), [])
    assert rec["status"] == "not_replayable" and rec["motivating"] == 0


@pytest.mark.parametrize(
    ("flag", "firing", "silent"),
    [
        ("repl_syntax_loop_breaker", SYNTAX_TRACE, CALM_TRACE),
        ("repl_plan_narration_nudge", "RESPONSE:\nLet me think about this plan first.\n", CALM_TRACE),
        ("harness_failure_ratio_guard", "Traceback (most recent call last)\nValueError: bad\nTypeError: worse\n", CALM_TRACE),
        ("tool_call_parse_error_retry", "error: failed to parse tool call arguments", CALM_TRACE),
        ("tool_call_arg_leak_guard", 'args={"q": "<tool_call>x</tool_call>"}', CALM_TRACE),
        ("stream_degeneration_guard", LOOP_TRACE, CALM_TRACE),
    ],
)
def test_each_firing_predicate_fires_and_stays_quiet(flag, firing, silent) -> None:
    pred = hl.FLAG_FIRING_PREDICATES[flag]
    assert pred(firing) is True
    assert pred(silent) is False


# ----- flags are fully wired and default off -----


@pytest.mark.parametrize("name", ["autopilot_lh_layer_taxonomy", "autopilot_lh_trace_replay_precheck"])
def test_flag_is_fully_wired_and_default_off(name) -> None:
    from src.features import _FEATURE_REGISTRY, Features

    spec = next(s for s in _FEATURE_REGISTRY if s.name == name)
    assert spec.default_test is False and spec.default_prod is False
    assert spec.env_on == "1" and spec.env_off == "0" and spec.env_name.startswith("AUTOPILOT_LH_")
    assert getattr(Features(), name) is False
    config_applicator = importlib.import_module("config_applicator")
    assert name in config_applicator.HOT_SWAP_FEATURES
    assert f"`{name}`" in (ROOT / "scripts" / "autopilot" / "program.md").read_text()
    assert f"{name}: baseline" in (ROOT / "orchestration" / "runtime_flags.spec.yaml").read_text()
    structural_lab = importlib.import_module("species.structural_lab")
    assert name not in structural_lab.PINNED_OBSERVE_ONLY_FLAGS


def test_flag_on_default_off_and_env_on(monkeypatch) -> None:
    monkeypatch.delenv(TAX_ENV, raising=False)
    assert hl.flag_on(TAX_ENV) is False
    monkeypatch.setenv(TAX_ENV, "1")
    assert hl.flag_on(TAX_ENV) is True
    monkeypatch.setenv(TAX_ENV, "0")
    assert hl.flag_on(TAX_ENV) is False


def test_flag_on_follows_a_live_runtime_override(monkeypatch) -> None:
    import src.flag_env as flag_env_mod

    monkeypatch.delenv(REPLAY_ENV, raising=False)
    monkeypatch.setattr(flag_env_mod, "runtime_overrides", lambda: {"autopilot_lh_trace_replay_precheck": True})
    assert hl.flag_on(REPLAY_ENV) is True


# ----- dispatcher wiring (fakes) -----


class _FakeLab:
    def __init__(self):
        self.applied: list[dict] = []

    def current_flags(self):
        return {"repl_syntax_loop_breaker": False, "repl_plan_narration_nudge": False, "memrl": True}

    def propose_flag_experiment(self, flags):
        return {"status": "valid", "proposed_flags": flags}

    def apply_flag_experiment(self, flags):
        self.applied.append(dict(flags))
        return {"status": "ok", "attestation": {"status": "ok"}}


class _FakeTower:
    def __init__(self):
        self.calls = 0

    def hybrid_eval(self):
        self.calls += 1
        return actions.EvalResult(tier=1, quality=1.0, speed=10.0, cost=0.0, reliability=1.0)


class _PassGate:
    def check(self, result):
        return True


def _dispatch(action, state, lab=None, tower=None):
    lab = lab or _FakeLab()
    tower = tower or _FakeTower()
    result, species = actions.dispatch_action(
        action, seeder=None, swarm=SimpleNamespace(mark_epoch=lambda *_: None), forge=None, lab=lab, tower=tower,
        gate=_PassGate(), archive=None, journal=None, state=state,
    )
    return result, species, lab, tower


def test_dispatch_flags_off_is_unannotated_and_unchanged(monkeypatch) -> None:
    monkeypatch.delenv(TAX_ENV, raising=False)
    monkeypatch.delenv(REPLAY_ENV, raising=False)
    state = {"contrastive_trace_bank": [_fail(CALM_TRACE)]}
    result, _sp, lab, tower = _dispatch(_structural("repl_syntax_loop_breaker"), state)
    assert tower.calls == 1 and lab.applied
    assert "harness_layer" not in result.details and "trace_replay_precheck" not in result.details


def test_dispatch_replay_on_skips_a_proved_noop_before_apply_or_eval(monkeypatch) -> None:
    monkeypatch.setenv(REPLAY_ENV, "1")
    monkeypatch.delenv(TAX_ENV, raising=False)
    state = {"contrastive_trace_bank": [_fail(CALM_TRACE, 7)]}
    result, species, lab, tower = _dispatch(_structural("repl_syntax_loop_breaker"), state)
    assert isinstance(result, actions.SkipOutcome) and species == "structural_lab"
    assert result.status == "skipped"  # never "invalid": a proved no-op is not blacklist-eligible
    assert result.reason.startswith("trace_replay_precheck:")
    assert lab.applied == [] and tower.calls == 0
    ann = result.lh_annotations["trace_replay_precheck"]
    assert ann["status"] == "would_not_fire" and ann["trace_hashes"] == ["h7"]
    assert "harness_layer" not in result.lh_annotations  # taxonomy flag is separate and off


def test_dispatch_replay_on_lets_a_firing_flag_through_and_records_it(monkeypatch) -> None:
    monkeypatch.setenv(REPLAY_ENV, "1")
    monkeypatch.delenv(TAX_ENV, raising=False)
    state = {"contrastive_trace_bank": [_fail(SYNTAX_TRACE)]}
    result, _sp, lab, tower = _dispatch(_structural("repl_syntax_loop_breaker"), state)
    assert tower.calls == 1 and lab.applied
    assert result.details["trace_replay_precheck"]["status"] == "would_fire"


def test_dispatch_replay_on_without_traces_proceeds(monkeypatch) -> None:
    monkeypatch.setenv(REPLAY_ENV, "1")
    result, _sp, _lab, tower = _dispatch(_structural("repl_syntax_loop_breaker"), {})
    assert tower.calls == 1
    assert result.details["trace_replay_precheck"]["status"] == "not_replayable"


def test_dispatch_taxonomy_on_records_layer_in_details(monkeypatch) -> None:
    monkeypatch.setenv(TAX_ENV, "1")
    monkeypatch.delenv(REPLAY_ENV, raising=False)
    state = {"contrastive_trace_bank": [_fail(SYNTAX_TRACE)]}
    result, _sp, _lab, tower = _dispatch(_structural("repl_plan_narration_nudge"), state)
    assert tower.calls == 1  # annotation never gates
    rec = result.details["harness_layer"]
    assert rec["layer"] == "trajectory_regulation" and rec["implicated"] == "action_realization"
    assert rec["aligned"] is False
    assert "trace_replay_precheck" not in result.details


def test_scope_excludes_unrelated_action_types(monkeypatch) -> None:
    monkeypatch.setenv(TAX_ENV, "1")
    monkeypatch.setenv(REPLAY_ENV, "1")
    ctx = actions._ActionContext(
        seeder=None, swarm=None, forge=None, lab=None, tower=None, gate=None, archive=None, journal=None, state={}
    )
    assert actions._lh_prepare({"type": "seed_batch"}, ctx) is None
    assert actions._lh_prepare({"type": "numeric_trial"}, ctx) is None


def test_mutation_context_carries_the_layer_block_only_when_taxonomy_on(monkeypatch) -> None:
    class Journal:
        def recent_failures(self, *, species, n):
            return []

        def insights_text(self, n):
            return "(no insights yet)"

        def recent(self, n):
            return []

    state = {"contrastive_trace_bank": [_fail(SYNTAX_TRACE)]}
    action = {"type": "prompt_mutation", "file": "tools.md", "mutation": "targeted_fix"}

    def ctx():
        return actions._ActionContext(
            seeder=None, swarm=None, forge=None, lab=None, tower=None, gate=None, archive=None,
            journal=Journal(), state=dict(state),
        )

    monkeypatch.delenv(TAX_ENV, raising=False)
    off = ctx()
    off.lh = actions._lh_prepare(action, off)
    assert off.lh is None
    assert "Harness Layer Diagnosis" not in actions._build_mutation_context(action, off)[0]

    monkeypatch.setenv(TAX_ENV, "1")
    on = ctx()
    on.lh = actions._lh_prepare(action, on)
    text = actions._build_mutation_context(action, on)[0]
    assert text.startswith("## Harness Layer Diagnosis")
    assert "implicate layer **action_realization**" in text
    assert on.lh["annotations"]["harness_layer"]["layer"] == "environment_contract"  # tools.md


def test_planner_schema_hint_only_when_taxonomy_on(monkeypatch) -> None:
    autopilot = importlib.import_module("autopilot")
    types = ["prompt_mutation", "structural_experiment"]
    monkeypatch.delenv(TAX_ENV, raising=False)
    assert '"layer"' not in autopilot._format_available_action_schemas(types)
    monkeypatch.setenv(TAX_ENV, "1")
    hint = autopilot._format_available_action_schemas(types)
    assert '"layer": one of' in hint and "environment_contract" in hint


def test_skip_row_journals_the_annotations() -> None:
    autopilot = importlib.import_module("autopilot")
    recorded = []

    class J:
        def record(self, entry):
            recorded.append(entry)

    ann = {"trace_replay_precheck": {"status": "would_not_fire"}}
    autopilot._record_skip_trial(
        J(), 3, _structural("repl_syntax_loop_breaker"), "structural_lab", "skipped", "trace_replay_precheck: x", 0,
        lh_annotations=ann,
    )
    # _record_reproposal_rate_windows may touch the journal; the entry is what matters
    assert recorded and recorded[0].eval_details == ann
    assert recorded[0].outcome_status == "skipped"
