"""AP-ME-1 decision-time context census on synthetic journals; no model or endpoint."""

import json

from scripts.autopilot.experiment_journal import JournalEntry
from scripts.autopilot.operator_context_baseline import (
    ancestors,
    census,
    checkpoint_self_check,
    openmle_class,
    operator_label,
    section_tags,
    summarize,
)
from scripts.autopilot.stm_generated_view import render_generated_stm


def entry(trial_id, parent=None, action_type="numeric_trial", mutation=None, **changes):
    snapshot = {"type": action_type}
    if mutation:
        snapshot["mutation"] = mutation
        snapshot["description"] = "d" * 40
    args = dict(trial_id=trial_id, timestamp=f"2026-07-01T00:{trial_id:02d}:00Z",
                species="numeric_swarm", action_type=action_type, tier=1, quality=1.5,
                speed=10.0, cost=0.1, reliability=1.0, pareto_status="dominated",
                config_snapshot=snapshot, parent_trial=parent, keep_revert_decision="revert",
                hypothesis=f"hypothesis {trial_id}", optimization_directions=f"dir {trial_id}")
    args.update(changes)
    return JournalEntry(**args)


def test_labels_and_section_tags():
    assert operator_label(entry(1, action_type="prompt_mutation", mutation="crossover")) == \
        "prompt_mutation:crossover"
    assert openmle_class("prompt_mutation:crossover") == "crossover"
    assert openmle_class("prompt_mutation:style_transfer") == "crossover"
    assert openmle_class("numeric_trial") == "improve"
    tags = section_tags("## Running Hypotheses\n- [t3] a\n- [t4] b\n## Failure Patterns\n- [t4] x\n")
    assert tags["Running Hypotheses"] == {3, 4} and tags["Failure Patterns"] == {4}
    assert ancestors(4, {4: 3, 3: 1, 1: None}) == {3, 1}
    assert ancestors(4, {4: 3, 3: 4}) == {3, 4}  # cycle-safe


def test_census_renders_strict_prefix_and_counts_lineage():
    rows = [entry(1), entry(2, parent=1), entry(3, parent=1), entry(4, parent=2),
            entry(5, parent=1, action_type="prompt_mutation", mutation="targeted_fix")]
    out = {r["trial_id"]: r for r in census(rows)}
    assert out[1]["cards"] == 0  # first decision sees no memory
    assert out[4]["cards"] == 3  # trials 1-3 only: never itself or later
    assert out[4]["ancestor_cards"] == 2 and out[4]["ancestors_available"] == 2
    assert out[5]["sibling_cards"] == 2 and out[5]["siblings_available"] == 2
    assert out[5]["operator"] == "prompt_mutation:targeted_fix"
    assert out[5]["same_operator_cards"] == 0 and out[4]["same_operator_cards"] == 3
    assert out[5]["description_chars"] == 40 and out[4]["description_chars"] is None
    summary = summarize(list(out.values()), "operator")
    assert summary["numeric_trial"]["decisions"] == 4


def test_untrusted_rows_are_not_cards_but_are_decisions():
    rows = [entry(1), entry(2, bug_corrupted_by="contention"), entry(3)]
    out = {r["trial_id"]: r for r in census(rows)}
    assert len(out) == 3 and out[3]["cards"] == 1


def test_checkpoint_self_check_identity(tmp_path):
    rows = [entry(i, parent=i - 1 if i > 1 else None) for i in range(1, 8)]
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "checkpoint_meta.json").write_text(json.dumps({"trial_id": 5}))
    (ckpt / "autopilot_short_term_memory.md").write_text(
        render_generated_stm([r for r in rows if r.trial_id <= 5]))
    check = checkpoint_self_check(rows, [ckpt, tmp_path / "absent"])
    assert check[0]["byte_identical"] and check[0]["pre_failure_sections_byte_identical"]
    assert check[1]["status"] == "missing"
    (ckpt / "autopilot_short_term_memory.md").write_text(render_generated_stm(rows))
    assert checkpoint_self_check(rows, [ckpt])[0]["byte_identical"] is False
