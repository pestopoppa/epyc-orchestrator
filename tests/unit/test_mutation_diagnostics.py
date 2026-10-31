"""Synthetic complete-input and native donor controls; no author/model invocation."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import logging
import os
import sys
import fcntl
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/autopilot"))
import mutation_author_capture as author
import crossover_diagnostics as crossover
from bsv_observe import _conflict_severity
from experiment_journal import ExperimentJournal, JournalEntry


def row(trial=1, **changes):
    # Match the current JournalEntry producer shape: error_scope is already
    # recorded from eval core, the trial baseline pin, regime digest and comparison.
    values = dict(trial_id=trial, timestamp="2026-10-06T00:00:00Z", species="prompt_forge",
                  action_type="prompt_mutation", tier=1, quality=2.0, speed=1.0,
                  cost=0.1, reliability=1.0, pareto_status="frontier",
                  baseline_pin={"eval_quality_era": "q1", "autopilot_speed_era": "s1",
                                "baseline_revision": 1},
                  comparability={"status": "COMPARABLE"},
                  error_scope={"schema_version": 1, "comparability": "COMPARABLE",
                               "core_id": "core1", "infra_regime_digest": "a" * 64,
                               "eval_quality_era": "q1", "autopilot_speed_era": "s1"},
                  eval_details={"infra_regime_digest": "a" * 64})
    values.update(changes)
    return JournalEntry(**values)


def capture(row_value, outcomes, history, *, proxy_outcomes=None):
    crossover.capture_trial_features(
        row_value, verdict=SimpleNamespace(passed=True),
        action={"type": "prompt_mutation", "file": "prompts/a.md", "sections": ["rules"]},
        eval_result=SimpleNamespace(core_id="core1", question_results=[
            {"qid": key, "correct": value} for key, value in (outcomes or {}).items()],
            per_suite_quality=proxy_outcomes or {}), history=history)


def method(name, bindings=None, script="species/prompt_forge.py"):
    """Execute the exact source method without importing model-owning PromptForge dependencies."""
    path = ROOT / "scripts/autopilot" / script
    tree = ast.parse(path.read_text())
    body = tree.body if script == "actions.py" else next(node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "PromptForge").body
    node = copy.deepcopy(next(node for node in body if isinstance(node, ast.FunctionDef)
                              and node.name == name))
    node.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[
        ast.alias(name="annotations")], level=0), node], type_ignores=[])
    namespace = {"log": logging.getLogger(__name__), "os": os, "json": json}
    for assignment in tree.body:
        if isinstance(assignment, ast.Assign) and any(isinstance(target, ast.Name)
                and target.id == "MUTATION_TYPES" for target in assignment.targets):
            namespace["MUTATION_TYPES"] = ast.literal_eval(assignment.value)
    namespace.update(bindings or {})
    exec(compile(ast.fix_missing_locations(module), str(path),
                 "exec"), namespace)
    return namespace[name]


def test_default_off_no_capture_or_history_io(monkeypatch, tmp_path):
    monkeypatch.delenv(author.FLAG, raising=False)
    monkeypatch.delenv(crossover.FLAG, raising=False)
    assert author.author_capture_kwargs({}, object()) == {}
    assert author.capture_author_context("private", operator="debug", target="a", inputs={},
                                         context=None, parameters={}) is None
    assert crossover.crossover_context(object(), "a") == ""
    assert list(tmp_path.iterdir()) == []


def test_complete_unicode_metadata_no_raw_prompt_and_unknown_pins(monkeypatch, tmp_path):
    monkeypatch.setenv(author.FLAG, "1")
    prompt = "private Ω🙂\n"
    record = author.capture_author_context(prompt, operator="targeted_fix", target="prompts/a.md",
        inputs={"original_content": "Ω", "failure_context": "bad", "description": "fix"},
        context={"journal_dir": str(tmp_path)}, parameters={"timeout_s": 90})
    assert record["input_sha256"] == hashlib.sha256(prompt.encode()).hexdigest()
    assert record["input_bytes"] == len(prompt.encode()) > record["input_chars"] == len(prompt)
    assert record["approximate_tokens"] == len(prompt) / 4
    assert record["assembly_input_chars"] == {"original_content": 1, "failure_context": 3, "description": 3}
    assert all(record[key] is None for key in ("source_pins", "run_manifest_sha256", "human_identity", "model_pin"))
    text = (tmp_path / "mutation_author_context.v1.jsonl").read_text()
    assert "private" not in text and "Ω" not in text
    assert json.loads(text) == record


def test_append_failure_reports_error_without_private_text(monkeypatch, tmp_path, caplog):
    monkeypatch.setenv(author.FLAG, "1")
    invalid = tmp_path / "not-a-directory"
    invalid.write_text("fixture")
    result = author.capture_author_context("private author input", operator="debug", target="a",
        inputs={}, context={"journal_dir": str(invalid)}, parameters={})
    assert result["capture_error"] == "FileExistsError"
    assert "private author input" not in caplog.text


def test_dispatch_pins_require_matching_native_trial(monkeypatch, tmp_path):
    monkeypatch.setenv(author.FLAG, "1")
    ctx = SimpleNamespace(journal=SimpleNamespace(journal_dir=tmp_path), state={
        "trial_counter": 7, "in_flight_trial": {"trial_id": 6, "run_manifest": {
            "sources": {"app": "native"}, "manifest_sha256": "native-run-manifest"}}})
    assert author.author_capture_kwargs({"type": "prompt_mutation"}, ctx)["author_capture_context"]["source_pins"] is None
    ctx.state["in_flight_trial"]["trial_id"] = 7
    value = author.author_capture_kwargs({"type": "prompt_mutation"}, ctx)["author_capture_context"]
    assert value["source_pins"] == {"app": "native"}
    assert value["run_manifest_sha256"] == "native-run-manifest"
    assert value["trial_id"] == 7 and "human_identity" not in value


@pytest.mark.parametrize("kind", ["prompt", "code"])
def test_real_prompt_boundary_unchanged_and_capture_failure_cannot_mask_author(monkeypatch, tmp_path, kind):
    source_file = tmp_path / "fixture.py"
    source_file.write_text("Ω original")
    bindings = {"_resolve_code_mutation_target": lambda _: source_file,
                "CODE_MUTATION_ALLOWLIST": ["prompts/a.md"],
                "new_file_mutation_root_labels": lambda: ["synthetic-root"]}
    build = method("_build_mutation_prompt" if kind == "prompt" else "_build_code_mutation_prompt", bindings)
    propose = method("propose_mutation" if kind == "prompt" else "propose_code_mutation", bindings)
    class BoundaryReached(Exception):
        pass
    seen = []
    fake = SimpleNamespace(timeout=90, read_prompt=lambda _: "Ω original",
                           _mutation_exclusion_reason=lambda _: None,
                           _negative_transfer_safety_block=lambda: "native safety")
    fake._build_mutation_prompt = lambda **kw: build(fake, **kw)
    fake._build_code_mutation_prompt = fake._build_mutation_prompt
    def invoke(prompt):
        seen.append(prompt)
        if author.capture_enabled():
            assert (tmp_path / "mutation_author_context.v1.jsonl").exists()
        raise BoundaryReached
    fake._invoke_claude = invoke
    kwargs = dict(target_file="prompts/a.md", mutation_type="crossover" if kind == "prompt" else "targeted_fix", failure_context="negative evidence",
                  per_suite_quality={"qa": 2.0}, description="goal")
    monkeypatch.delenv(author.FLAG, raising=False)
    with pytest.raises(BoundaryReached):
        propose(fake, **kwargs)
    monkeypatch.setenv(author.FLAG, "1")
    with pytest.raises(BoundaryReached):
        propose(fake, **kwargs, author_capture_context={"journal_dir": str(tmp_path)})
    assert seen[0] == seen[1]
    assert json.loads((tmp_path / "mutation_author_context.v1.jsonl").read_text())["input_sha256"] == hashlib.sha256(seen[1].encode()).hexdigest()
    monkeypatch.setattr(author, "capture_author_context", lambda *a, **kw: (_ for _ in ()).throw(OSError("private")))
    with pytest.raises(BoundaryReached):
        propose(fake, **kwargs)
    assert seen[2] == seen[0]


def test_busy_author_capture_lock_is_typed_and_fake_author_still_runs(monkeypatch, tmp_path):
    monkeypatch.setenv(author.FLAG, "1")
    journal_file = tmp_path / "mutation_author_context.v1.jsonl"
    locked = threading.Event()
    release = threading.Event()

    def hold_lock():
        with journal_file.open("a") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            locked.set()
            release.wait(3)
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    holder = threading.Thread(target=hold_lock, daemon=True)
    holder.start()
    assert locked.wait(2)
    watchdog = threading.Timer(2, release.set)
    watchdog.start()

    capture_results = []
    original_capture = author.capture_author_context
    def capture_and_record(*args, **kwargs):
        result = original_capture(*args, **kwargs)
        capture_results.append(result)
        return result
    monkeypatch.setattr(author, "capture_author_context", capture_and_record)

    source_file = tmp_path / "fixture.md"
    source_file.write_text("original")
    bindings = {"_resolve_code_mutation_target": lambda _: source_file,
                "CODE_MUTATION_ALLOWLIST": ["prompts/a.md"],
                "new_file_mutation_root_labels": lambda: ["synthetic-root"]}
    build = method("_build_mutation_prompt", bindings)
    propose = method("propose_mutation", bindings)
    class BoundaryReached(Exception):
        pass
    seen = []
    fake = SimpleNamespace(timeout=90, read_prompt=lambda _: "original",
                           _mutation_exclusion_reason=lambda _: None,
                           _negative_transfer_safety_block=lambda: "native safety")
    fake._build_mutation_prompt = lambda **kw: build(fake, **kw)
    def invoke(prompt):
        seen.append(prompt)
        release.set()
        raise BoundaryReached
    fake._invoke_claude = invoke
    try:
        with pytest.raises(BoundaryReached):
            propose(fake, target_file="prompts/a.md", mutation_type="crossover",
                    failure_context="negative evidence", per_suite_quality={"qa": 2.0},
                    description="goal", author_capture_context={"journal_dir": str(tmp_path)})
    finally:
        release.set()
        watchdog.cancel()
        holder.join(3)
    assert seen
    assert capture_results[0]["capture_error"] == "AuthorCaptureBusyError"


def test_native_anchor_deltas_jsonl_roundtrip_and_separate_bsv_policy(monkeypatch, tmp_path):
    monkeypatch.setenv(crossover.FLAG, "1")
    journal = ExperimentJournal(journal_dir=tmp_path)
    first, left, right = row(1), row(2), row(3)
    capture(first, {"a": False, "b": False}, [])
    assert first.eval_details["crossover_features"]["features"]["behavior_signature_delta"]["severity"] is None
    journal.record(first)
    capture(left, {"a": True, "b": False}, journal.entries_with_supersessions())
    journal.record(left)
    capture(right, {"a": False, "b": True}, journal.entries_with_supersessions())
    journal.record(right)
    history = ExperimentJournal(journal_dir=tmp_path).entries_with_supersessions()
    pairs = crossover.donor_pairs(history, "prompts/a.md")
    assert [p["donor_trial_ids"] for p in pairs] == [[2, 3]]
    assert pairs[0]["features"]["disjoint_improvements"]
    severity, reasons = _conflict_severity(
        history[1].eval_details["crossover_features"]["features"],
        history[2].eval_details["crossover_features"]["features"])
    assert severity == "blocking"
    assert reasons == ["same subsystem prompt", "shared file: prompts/a.md", "shared prompt section: rules",
                       "opposing or disjoint sentinel movement across accepted mutations"]
    rendered = crossover.crossover_context(journal, "prompts/a.md")
    assert "donor trials [2, 3]" in rendered
    assert "native per-question outcomes (partial)" in rendered
    assert "recorded improvements [a] / [b]" in rendered
    assert "shared recorded features: same subsystem prompt; shared file: prompts/a.md; shared prompt section: rules" in rendered
    assert "not ground truth" in rendered
    journal.append_supersession_event(target_trial_ids=[2], fields={"keep_revert_decision": "excluded"},
        reason="synthetic", policy_version="supersession-v1", actor="unit-test")
    assert crossover.donor_pairs(journal.entries_with_supersessions(), "prompts/a.md") == []
    assert journal._entries[1].keep_revert_decision == ""


def test_suite_quality_proxy_pair_renders_proxy_label(monkeypatch):
    monkeypatch.setenv(crossover.FLAG, "1")
    first, left, right = row(1), row(2), row(3)

    def proxy_capture(current, outcomes, history):
        crossover.capture_trial_features(
            current, verdict=SimpleNamespace(passed=True),
            action={"type": "prompt_mutation", "file": "prompts/a.md", "sections": ["rules"]},
            eval_result=SimpleNamespace(core_id="core1", per_suite_quality=outcomes,
                                        question_results=[]), history=history)

    proxy_capture(first, {"a": 1.5, "b": 1.5}, [])
    proxy_capture(left, {"a": 2.5, "b": 1.5}, [first])
    proxy_capture(right, {"a": 1.5, "b": 2.5}, [first, left])
    rendered = crossover.crossover_context(
        SimpleNamespace(entries_with_supersessions=lambda: [first, left, right]), "prompts/a.md")
    assert "native suite-quality proxy outcomes (partial proxy)" in rendered
    assert "recorded improvements [a] / [b]" in rendered
    assert "not ground truth" in rendered


def test_writer_and_reader_keep_question_and_proxy_anchor_classes_separate(monkeypatch):
    monkeypatch.setenv(crossover.FLAG, "1")
    question_anchor, proxy_anchor, proxy_left, proxy_right = (row(i) for i in range(1, 5))
    # The first row has coinciding question and suite labels; BSV records questions
    # as its source. The proxy anchor repeats those labels but is a different class.
    capture(question_anchor, {"shared": False, "other": False}, [],
            proxy_outcomes={"shared": 1.0, "other": 1.0})
    capture(proxy_anchor, None, [], proxy_outcomes={"shared": 1.0, "other": 1.0})
    history = [question_anchor, proxy_anchor]
    capture(proxy_left, None, history,
            proxy_outcomes={"shared": 2.5, "other": 1.0})
    capture(proxy_right, None, history + [proxy_left],
            proxy_outcomes={"shared": 1.0, "other": 2.5})

    assert question_anchor.eval_details["crossover_features"]["sentinel_outcome_source"] == "question_results"
    assert proxy_anchor.eval_details["crossover_features"]["sentinel_outcome_source"] == "suite_quality_proxy"
    assert proxy_left.eval_details["crossover_features"]["delta_reference_trial_id"] == proxy_anchor.trial_id
    assert proxy_right.eval_details["crossover_features"]["delta_reference_trial_id"] == proxy_anchor.trial_id
    pairs = crossover.donor_pairs(history + [proxy_left, proxy_right], "prompts/a.md")
    assert [pair["donor_trial_ids"] for pair in pairs] == [[3, 4]]
    assert pairs[0]["sentinel_outcome_source"] == "suite_quality_proxy"


@pytest.mark.parametrize("change", ["scope", "anchor", "legacy", "corrupt", "excluded", "regression", "version", "scope_version_bool", "anchor_excluded", "bool_trial", "signature_bool_trial", "signature_shape", "stale_revision", "stale_era", "stale_infra", "stale_comparability", "missing_error_scope", "stale_error_scope_core"])
def test_pair_unknown_or_ineligible_never_reconstructs(monkeypatch, change):
    monkeypatch.setenv(crossover.FLAG, "1")
    first, left, right = row(1), row(2), row(3)
    capture(first, {"a": False, "b": False}, [])
    capture(left, {"a": True, "b": False}, [first])
    capture(right, {"a": False, "b": True}, [first, left])
    native = right.eval_details["crossover_features"]
    if change == "scope": native["scope"]["core_id"] = "different"
    if change == "anchor": native["delta_reference_trial_id"] = 99
    if change == "legacy": right.eval_details.pop("crossover_features")
    if change == "corrupt": right.bug_corrupted_by = "repair"
    if change == "excluded": right.keep_revert_decision = "excluded"
    if change == "regression": native["features"]["behavior_signature_delta"]["regressed_sentinels"] = ["a"]
    if change == "version": native["features"]["version"] = "stale"
    if change == "scope_version_bool": native["scope"]["schema_version"] = True
    if change == "anchor_excluded": first.keep_revert_decision = "excluded"
    if change == "bool_trial": right.trial_id = True
    if change == "signature_bool_trial": native["signature"]["trial_id"] = True
    if change == "signature_shape": native["signature"].pop("signature_hash")
    if change == "stale_revision": right.baseline_pin["baseline_revision"] = 2
    if change == "stale_era": right.baseline_pin["eval_quality_era"] = "q2"
    if change == "stale_infra": right.eval_details["infra_regime_digest"] = "b" * 64
    if change == "stale_comparability": right.comparability["status"] = "UNVERIFIED"
    if change == "missing_error_scope": right.error_scope = None
    if change == "stale_error_scope_core": right.error_scope["core_id"] = "core2"
    assert crossover.donor_pairs([first, left, right], "prompts/a.md") == []


def test_native_writer_unknown_scope_failed_verdict_and_disabled_no_mutation(monkeypatch):
    current = row()
    monkeypatch.delenv(crossover.FLAG, raising=False)
    capture(current, {"a": False}, [])
    assert "crossover_features" not in current.eval_details
    monkeypatch.setenv(crossover.FLAG, "1")
    current.comparability = {"status": "UNVERIFIED"}
    capture(current, {"a": False}, [])
    assert "crossover_features" not in current.eval_details
    assert crossover.donor_pairs([current], "prompts/a.md") == []
    other = row(2)
    crossover.capture_trial_features(other, verdict=SimpleNamespace(passed=False), action={},
        eval_result=SimpleNamespace(core_id="core1"), history=[])
    assert "crossover_features" not in other.eval_details


def test_bsv_no_overlap_keeps_original_early_return():
    assert _conflict_severity({"subsystem": "a", "behavior_signature_delta": 42},
                             {"subsystem": "b", "behavior_signature_delta": 42}) == ("none", [])


def test_real_crossover_context_caller_preserves_negative_evidence(monkeypatch):
    assemble = method("_build_mutation_context", {
        "_discard_mutation_diversity_coverage": lambda _: None,
        "_ledger_journal_dir": lambda _: None,
    }, script="actions.py")
    journal = SimpleNamespace(
        recent_failures=lambda **_: [SimpleNamespace(trial_id=9, action_type="prompt_mutation")],
        failure_analysis_for_prompt=lambda _: "native negative evidence",
        insights_text=lambda **_: "(no insights yet)", recent=lambda _: [])
    ctx = SimpleNamespace(journal=journal, strategy_store=None, tower=None, state={})
    action = {"file": "prompts/a.md", "mutation": "crossover"}
    monkeypatch.delenv(crossover.FLAG, raising=False)
    original, suites = assemble(action, ctx)
    assert original == "Trial #9 (prompt_mutation):\nnative negative evidence" and suites is None
    monkeypatch.setenv(crossover.FLAG, "1")
    journal.entries_with_supersessions = lambda: []
    diagnostic, _ = assemble(action, ctx)
    assert diagnostic.startswith(original)
    assert diagnostic.endswith("Unknown: no eligible recorded comparable donor pair.")
    action["mutation"] = "targeted_fix"
    assert assemble(action, ctx)[0] == original


def test_selection_writer_to_actual_prompt_context(monkeypatch, tmp_path):
    from species.mutation_graph import MutationGraph
    import actions
    import bsv_observe
    import experiment_journal
    import species.mutation_graph

    for module in (actions, bsv_observe, crossover, experiment_journal, species.mutation_graph):
        assert Path(module.__file__).resolve().is_relative_to(ROOT)
    monkeypatch.setenv(crossover.FLAG, "1")
    monkeypatch.setenv("AUTOPILOT_CROSSOVER_DONOR_SELECTION", "1")
    source_path = tmp_path / "a.md"
    source_path.write_text("current")
    source = next(iter(actions.served_content_record([source_path])["files"]))
    journal = ExperimentJournal(journal_dir=tmp_path)
    for trial, outcomes in [(1, {"a": False, "b": False}),
                            (2, {"a": True, "b": False}),
                            (3, {"a": True, "b": False}),
                            (4, {"a": False, "b": True})]:
        entry = row(trial)
        result = SimpleNamespace(core_id="core1", question_results=[
            {"qid": k, "correct": v} for k, v in outcomes.items()], per_suite_quality={})
        content = f"donor content {trial}"
        entry.eval_details["served_content"] = {"files": {source: hashlib.sha256(content.encode()).hexdigest()}}
        result._crossover_donor_content = ("prompts/a.md", source, content)
        crossover.capture_trial_features(
            entry, verdict=SimpleNamespace(passed=True),
            action={"type": "prompt_mutation", "file": "prompts/a.md", "sections": ["rules"]},
            eval_result=result, history=journal.entries_with_supersessions())
        journal.record(entry)
    history = journal.entries_with_supersessions()
    pairs = MutationGraph.informed_crossover_candidates("prompts/a.md", history, source, top_n=3)
    assert pairs[0]["trial_ids"] == [2, 3]
    assert pairs[-1]["severity"] == "blocking"
    assert pairs[0]["severity"] == _conflict_severity(
        history[1].eval_details["crossover_features"]["features"],
        history[2].eval_details["crossover_features"]["features"])[0]
    assemble = method("_build_mutation_context", {
        "_discard_mutation_diversity_coverage": lambda _: None,
        "_ledger_journal_dir": lambda _: None,
        "served_content_record": actions.served_content_record,
    }, script="actions.py")
    ctx = SimpleNamespace(journal=journal, strategy_store=None, tower=None, state={},
                          forge=SimpleNamespace(_resolve_prompt_path=lambda _: source_path))
    assembled, _ = assemble({"type": "prompt_mutation", "file": "prompts/a.md", "mutation": "crossover"}, ctx)
    assert 'donor content 2' in assembled and 'donor content 3' in assembled
    assert '"trial_ids": [2, 3]' in assembled and source in assembled
    assert hashlib.sha256(b"donor content 2").hexdigest() in assembled
    prompt = method("_build_mutation_prompt")(SimpleNamespace(
        _negative_transfer_safety_block=lambda: "safety"), "prompts/a.md", "crossover",
        "current", assembled, None, "")
    assert 'donor content 2' in prompt and 'donor content 3' in prompt
    monkeypatch.delenv("AUTOPILOT_CROSSOVER_DONOR_SELECTION")
    assert 'donor content 2' not in assemble({"type": "prompt_mutation", "file": "prompts/a.md", "mutation": "crossover"}, ctx)[0]
    history[1].eval_details["crossover_donor_content"]["content"] = "tampered"
    assert all(2 not in p["trial_ids"] for p in MutationGraph.informed_crossover_candidates(
        "prompts/a.md", history, source, top_n=3))
    history[1].eval_details["crossover_donor_content"]["content"] = "donor content 2"
    other_source = "other-root/a.md"
    history[1].eval_details["crossover_donor_content"]["source"] = other_source
    history[1].eval_details["served_content"]["files"][other_source] = hashlib.sha256(b"donor content 2").hexdigest()
    assert all(2 not in p["trial_ids"] for p in MutationGraph.informed_crossover_candidates(
        "prompts/a.md", history, source, top_n=3))
    history[1].eval_details["crossover_donor_content"]["source"] = source
    history[1].eval_details["crossover_features"]["features"]["prompt_sections_touched"] = []
    assert all(2 not in p["trial_ids"] for p in MutationGraph.informed_crossover_candidates(
        "prompts/a.md", history, source, top_n=3))


def test_real_action_captures_only_kept_served_bytes(monkeypatch, tmp_path):
    import actions

    monkeypatch.setenv(crossover.FLAG, "1")
    monkeypatch.setenv("AUTOPILOT_CROSSOVER_DONOR_SELECTION", "1")
    path = tmp_path / "a.md"
    path.write_text("before")
    mutation = SimpleNamespace(file="a.md", original_content="before", mutated_content="after",
                               safety_valid=True, safety_warnings=[], mutation_type="crossover")
    result = SimpleNamespace()
    forge = SimpleNamespace(propose_mutation=lambda **_: mutation,
                            apply_mutation=lambda m: path.write_text(m.mutated_content),
                            revert_mutation=lambda m: path.write_text(m.original_content),
                            _resolve_prompt_path=lambda _: path)
    ctx = SimpleNamespace(forge=forge, tower=SimpleNamespace(hybrid_eval=lambda: result),
                          state={}, swarm=SimpleNamespace(mark_epoch=lambda _: None))
    bindings = {name: getattr(actions, name) for name in (
        "_served_prompt_path", "_record_served_content", "_mutation_restore",
        "served_content_record", "SERVED_CONTENT_STATE_KEY")}
    bindings.update({"_build_mutation_context": lambda *_: ("", None),
                     "author_capture_kwargs": lambda *_: {},
                     "_stash_prompt_interference": lambda *_: None,
                     "_skill_efficacy_without_result": lambda *_: None,
                     "_bsv2_baseline_result": lambda *_: None,
                     "_action_gate_check": lambda *_: True,
                     "_simplicity_check": lambda *_, **__: (True, None),
                     "_skill_efficacy_accepts": lambda **_: True,
                     "_bsv2_accepts": lambda **_: True,
                     "_record_mutation_diversity_coverage": lambda *_, **__: None,
                     "_record_accepted_warrant_telemetry": lambda *_: None})
    action = method("_action_prompt_mutation", bindings, script="actions.py")
    action({"file": "a.md", "mutation": "crossover"}, ctx)
    target, source, content = result._crossover_donor_content
    assert (target, content) == ("a.md", "after")
    assert ctx.state[actions.SERVED_CONTENT_STATE_KEY]["files"][source] == hashlib.sha256(b"after").hexdigest()


@pytest.mark.parametrize("diagnostics,selection,action_type,passed", [
    ("0", "1", "prompt_mutation", True),
    ("1", "0", "prompt_mutation", True),
    ("1", "1", "code_mutation", True),
    ("1", "1", "prompt_mutation", False),
])
def test_writer_consumes_retained_result_on_refusal(monkeypatch, diagnostics, selection, action_type, passed):
    monkeypatch.setenv(crossover.FLAG, diagnostics)
    monkeypatch.setenv("AUTOPILOT_CROSSOVER_DONOR_SELECTION", selection)
    entry = row(10)
    source = "orchestration/prompts/a.md"
    data = "stale"
    entry.eval_details["served_content"] = {"files": {source: hashlib.sha256(data.encode()).hexdigest()}}
    result = SimpleNamespace(core_id="core1", question_results=[{"qid": "a", "correct": True}],
                             per_suite_quality={}, _crossover_donor_content=("prompts/a.md", source, data))
    crossover.capture_trial_features(
        entry, verdict=SimpleNamespace(passed=passed),
        action={"type": action_type, "file": "prompts/a.md", "sections": ["rules"]},
        eval_result=result, history=[])
    assert not hasattr(result, "_crossover_donor_content")
    assert "crossover_donor_content" not in entry.eval_details
