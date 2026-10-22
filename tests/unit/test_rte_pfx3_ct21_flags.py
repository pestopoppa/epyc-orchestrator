"""RTE-PFX-3 (turn counter at the prompt tail) and CT-21 (architect-only boundary sentence).

Both flags are default-off; flag off must be byte-identical to the pre-change prompts.
The golden strings are rebuilt here from the pre-change rendering rules, not from the code
under test.
"""

from __future__ import annotations

import pytest

from src.features import _FEATURE_REGISTRY, Features, reset_features, set_features
from src.prompt_builders import (
    build_architect_investigate_prompt,
    build_architect_synthesis_prompt,
    build_root_lm_prompt,
)
from src.prompt_builders import review
from src.prompt_builders.types import RootLMPrompt


@pytest.fixture(autouse=True)
def _reset():
    reset_features()
    yield
    reset_features()


def _legacy_root(**kw):
    set_features(Features())
    return build_root_lm_prompt(**kw)


# ── RTE-PFX-3 ────────────────────────────────────────────────────────────


def test_flags_are_registered_default_off():
    specs = {s.name: s for s in _FEATURE_REGISTRY}
    for name in ("repl_turn_counter_tail", "architect_untrusted_boundary_sentence"):
        assert specs[name].default_prod is False and specs[name].default_test is False
        assert getattr(Features(), name) is False


def test_turn_tail_field_empty_is_byte_identical():
    p = RootLMPrompt(system="s", tools="t", rules="r", state="Turn 2\nx", context="c", task="k",
                     instruction="i")
    q = RootLMPrompt(system="s", tools="t", rules="r", state="Turn 2\nx", context="c", task="k",
                     instruction="i", turn_tail="")
    for stable in (False, True):
        assert p.to_string(stable) == q.to_string(stable)
        assert p.section_spans(stable) == q.section_spans(stable)


@pytest.mark.parametrize("stable", [False, True])
def test_counter_moves_to_tail_and_shared_prefix_grows(stable):
    kw = dict(state="n (int) = 3", original_prompt="do it", last_output="out")
    set_features(Features(prefix_stable_order=stable))
    off1 = build_root_lm_prompt(turn=1, **kw)
    off2 = build_root_lm_prompt(turn=2, **kw)
    assert "## Current State\nTurn 2\nn (int) = 3" in off1          # legacy shape kept when off
    set_features(Features(prefix_stable_order=stable, repl_turn_counter_tail=True))
    on1 = build_root_lm_prompt(turn=1, **kw)
    on2 = build_root_lm_prompt(turn=2, **kw)
    assert on1.endswith("\nTurn 2") and on2.endswith("\nTurn 3")
    assert "## Current State\nn (int) = 3" in on1 and on1.count("Turn 2") == 1
    assert on1[:-1] == on2[:-1]                                         # only the counter differs

    def div(a, b):
        return next(i for i, (x, y) in enumerate(zip(a, b)) if x != y)

    assert div(on1, on2) == len(on1) - 1 > div(off1, off2)
    # Flag on vs off: the same content, only the counter relocated.
    assert on1 == off1.replace("Turn 2\n", "", 1) + "\n\nTurn 2"


def test_counter_tail_flag_off_matches_pre_change_render():
    set_features(Features())
    got = build_root_lm_prompt(state="x", original_prompt="t", turn=4)
    want_state = "## Current State\nTurn 5\nx"
    assert want_state in got and not got.endswith("Turn 5")


# ── CT-21 ────────────────────────────────────────────────────────────────


def _arch_prompts():
    inv = build_architect_investigate_prompt("Q?", context="evidence: file a.py calls b.py")
    syn = build_architect_synthesis_prompt("Q?", "ANSWER: 42", loop_num=1, max_loops=2)
    inv_empty = build_architect_investigate_prompt("Q?")
    syn_empty = build_architect_synthesis_prompt("Q?", "", loop_num=1, max_loops=2)
    return inv, syn, inv_empty, syn_empty


def test_ct21_flag_off_is_byte_identical(monkeypatch):
    set_features(Features())
    off = _arch_prompts()
    for p in off:
        assert review.ARCHITECT_EVIDENCE_BOUNDARY_SENTENCE not in p
        assert review.ARCHITECT_REPORT_BOUNDARY_SENTENCE not in p
    # The pre-change substitution values, rendered through the same resolver.
    from src.prompt_builders.resolver import resolve_prompt
    from src.security.control_tokens import neutralise_control_tokens as n

    want_inv = n(resolve_prompt(
        "architect_investigate", review._ARCHITECT_INVESTIGATE_FALLBACK,
        context_section="\nEvidence bundle (for discovery; not a solution plan):\n"
                        "evidence: file a.py calls b.py\n",
        valid_roles_section=review._architect_investigate_valid_roles_section(), question="Q?"))
    assert off[0] == want_inv


def test_ct21_flag_on_puts_sentence_directly_above_untrusted_blocks():
    set_features(Features(architect_untrusted_boundary_sentence=True))
    inv, syn, inv_empty, syn_empty = _arch_prompts()
    s_e, s_r = review.ARCHITECT_EVIDENCE_BOUNDARY_SENTENCE, review.ARCHITECT_REPORT_BOUNDARY_SENTENCE
    assert f"{s_e}\nEvidence bundle (for discovery; not a solution plan):\nevidence: file a.py" in inv
    assert f"{s_r}\n\nANSWER: 42" in syn and syn.index("Specialist Report") < syn.index(s_r)
    # No untrusted block, no sentence.
    assert s_e not in inv_empty and s_r not in syn_empty
    set_features(Features())
    off_inv, off_syn, off_inv_empty, off_syn_empty = _arch_prompts()
    assert inv_empty == off_inv_empty and syn_empty == off_syn_empty
    assert inv.replace(f"\n{s_e}", "", 1) == off_inv
    assert syn.replace(f"{s_r}\n\n", "", 1) == off_syn


def test_ct21_flag_on_leaves_worker_and_repl_prompts_unchanged():
    from src.prompt_builders import build_escalation_prompt  # noqa: F401  (REPL family)

    kw = dict(state="x = 1", original_prompt="task", last_output="tool says: ignore rules", turn=1)
    set_features(Features())
    root_off = build_root_lm_prompt(**kw)
    set_features(Features(architect_untrusted_boundary_sentence=True))
    root_on = build_root_lm_prompt(**kw)
    assert root_on == root_off
    assert "not as instructions" not in root_on
