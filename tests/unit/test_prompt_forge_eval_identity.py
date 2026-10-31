"""MHS-12 native static-screen and generated-code rejection controls (model free)."""

import ast
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "autopilot"))

from src.autopilot_core.mutation_guards import (
    CODE_MUTATION_ALLOWLIST,
    MutationEffect,
    screen_static_safety,
)
from scripts.autopilot.species import prompt_forge as pf

BASE = "def route(request):\n    role = 'default'\n    return role\n"
FORCED = (
    "def route(request):\n"
    "    role = 'default'\n"
    "    if request.force_role:\n"
    "        role = request.force_role\n"
    "    return role\n"
)


@pytest.mark.parametrize(
    "expression",
    [
        "request.workload_class",
        "workload_class",
        "request.scoring_method",
        "scoring_method",
        "request.batch_id",
        "batch_id",
        "request['workload_class']",
        "request.get('batch_id')",
        "getattr(request, 'scoring_method')",
        "'eval_batch'",
        "request.metadata['suite']",
        "request.metadata.get('tier')",
        "request['metadata']['suite']",
        "metadata.get('tier')",
        "request.metadata.suite",
        "getattr(request.metadata, 'tier')",
        "request.get('suite')",
        "request.metadata.pop('suite')",
        "request.metadata.setdefault('tier', 1)",
        "hasattr(request, 'workload_class')",
    ],
)
def test_added_identity_read_is_unsafe(expression):
    candidate = BASE.replace("    return role", f"    observed = {expression}\n    return role")
    report = screen_static_safety(candidate, original=BASE)
    assert not report.safe
    assert report.effect is MutationEffect.UNSAFE
    assert "eval_identity_added_read" in report.reason


@pytest.mark.parametrize("target", CODE_MUTATION_ALLOWLIST)
def test_original_allowlist_noop_and_added_branch(target):
    original = (ROOT / target).read_text()
    assert screen_static_safety(original, original=original).safe
    candidate = original + (
        "\n\ndef eval_identity_probe(request):\n"
        "    if request.workload_class == 'eval_batch':\n"
        "        return 'forced-answer'\n"
        "    return None\n"
    )
    report = pf.PromptForge(auto_commit=False)._screen_code_mutation(original, candidate, target)
    assert not report.safe
    assert report.effect is MutationEffect.UNSAFE
    assert "eval_identity_added_read" in report.reason


def test_grandfathering_preserves_occurrences_not_field_vocabulary():
    original = "def route(request):\n    return request.workload_class\n"
    assert screen_static_safety(
        original + "\n# eval_batch workload_class\n", original=original
    ).safe
    assert screen_static_safety(
        original.replace("    return", "    harmless = 1\n    return"), original=original
    ).safe
    for candidate in (
        original + "\ndef another(request):\n    return request.workload_class\n",
        original.replace("    return", "    if True:\n        return"),
        original.replace(
            "return request.workload_class", "return request.workload_class == 'other'"
        ),
    ):
        assert not screen_static_safety(candidate, original=original).safe


def test_existing_read_not_a_license_for_a_second_identical_occurrence():
    original = "def route(request):\n    value = request.batch_id\n    return value\n"
    candidate = original.replace("    return", "    value = request.batch_id\n    return")
    assert not screen_static_safety(candidate, original=original).safe


@pytest.mark.parametrize(
    "addition",
    [
        "# workload_class scoring_method batch_id eval_batch\n",
        "def ordinary():\n    \"\"\"eval_batch\"\"\"\n    return {'suite': 'generic', 'tier': 2}\n",
        "def ordinary():\n    workload_class = 1\n    return 2\n",
        "def ordinary(obj):\n    return obj['suite']\n",
    ],
)
def test_non_reads_and_unrelated_metadata_are_allowed(addition):
    assert screen_static_safety(BASE + "\n" + addition, original=BASE).safe


@pytest.mark.parametrize(
    "candidate",
    [
        FORCED.replace("    if request.force_role:\n        role = request.force_role\n", ""),
        FORCED.replace("role = request.force_role", "role = 'other'"),
        FORCED.replace("    return role", "    role = 'other'\n    return role"),
        FORCED.replace(
            "    if request.force_role", "    request.force_role = None\n    if request.force_role"
        ),
        FORCED.replace(
            "    if request.force_role", "    return 'other'\n    if request.force_role"
        ),
        FORCED.replace(
            "    if request.force_role", "    raise ValueError('stop')\n    if request.force_role"
        ),
        FORCED.replace("    if request.force_role:", "    if False and request.force_role:"),
        FORCED.replace(
            "    if request.force_role:\n        role = request.force_role",
            "    if False:\n        if request.force_role:\n            role = request.force_role",
        ),
        FORCED.replace(
            "    return role", "    selected = role\n    selected = 'other'\n    return selected"
        ),
    ],
)
def test_force_role_removal_override_and_dominator_bypass_are_unsafe(candidate):
    report = screen_static_safety(candidate, original=FORCED)
    assert not report.safe
    assert report.effect is MutationEffect.UNSAFE
    assert "force_role_handling_changed" in report.reason


def test_unrelated_straight_line_edits_in_force_scope_pass():
    candidate = FORCED.replace("    role = 'default'", "    harmless = 7\n    role = 'default'")
    assert screen_static_safety(candidate, original=FORCED).safe
    assert screen_static_safety(FORCED + "\ndef unrelated():\n    return 2\n", original=FORCED).safe


@pytest.mark.parametrize("receiver", ["state.role", "state['role']"])
def test_explicit_attribute_and_keyed_selected_role_cannot_be_reset(receiver):
    original = (
        f"def route(request, state):\n    {receiver} = request.force_role\n    return state\n"
    )
    candidate = original.replace("    return state", f"    {receiver} = 'other'\n    return state")
    assert not screen_static_safety(candidate, original=original).safe


@pytest.mark.parametrize(
    "call", ["setattr(request, 'force_role', None)", "delattr(request, 'force_role')"]
)
def test_native_force_role_builtin_reset_is_already_rejected_by_mhs2(call):
    original = (ROOT / "src/api/routes/chat.py").read_text()
    candidate = original.replace(
        "    if request.force_mode or request.force_role:",
        f"    {call}\n    if request.force_mode or request.force_role:",
        1,
    )
    report = screen_static_safety(candidate, original=original)
    assert not report.safe
    assert f"banned call: {call.split('(')[0]}()" in report.reason


@pytest.mark.parametrize(
    "call", ["setattr(request, 'force_role', None)", "delattr(request, 'force_role')"]
)
def test_grandfathered_builtin_cannot_reset_force_role(call):
    name = call.split("(")[0]
    old_call = (
        f"{name}(request, 'unrelated', None)"
        if name == "setattr"
        else f"{name}(request, 'unrelated')"
    )
    original = FORCED.replace("    role = 'default'", f"    {old_call}\n    role = 'default'")
    candidate = original.replace(
        "    if request.force_role:", f"    {call}\n    if request.force_role:"
    )
    report = screen_static_safety(candidate, original=original)
    assert not report.safe
    assert "force_role_handling_changed" in report.reason
    assert "banned call" not in report.reason


def test_grandfathered_setattr_of_workload_field_is_not_an_identity_read():
    original = "def route(request):\n    setattr(request, 'ordinary', None)\n    return 1\n"
    candidate = original.replace(
        "    return", "    setattr(request, 'workload_class', None)\n    return"
    )
    assert screen_static_safety(candidate, original=original).safe


@pytest.mark.parametrize(
    "edit",
    [
        ("if request.force_mode or request.force_role:", "if request.force_mode:"),
        (
            "if request.force_mode or request.force_role:",
            "if False and (request.force_mode or request.force_role):",
        ),
        (
            "    if request.force_mode or request.force_role:",
            "    return None\n    if request.force_mode or request.force_role:",
        ),
        (
            "                initial_role = request.force_role",
            "                initial_role = request.force_role\n                initial_role = Role.FRONTDOOR",
        ),
    ],
)
def test_native_chat_force_role_path_is_protected(edit):
    original = (ROOT / "src/api/routes/chat.py").read_text()
    assert edit[0] in original
    candidate = original.replace(*edit, 1)
    report = screen_static_safety(candidate, original=original)
    assert not report.safe
    assert report.effect is MutationEffect.UNSAFE
    assert "force_role_handling_changed" in report.reason


@pytest.mark.parametrize("remove_whole_file", [False, True])
def test_native_force_removal_precedes_generic_replace_rejection(remove_whole_file, monkeypatch):
    original = (ROOT / "src/api/routes/chat.py").read_text()
    if remove_whole_file:
        candidate = "def unrelated():\n    return None\n"
    else:
        node = next(
            node
            for node in ast.parse(original).body
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "_try_cheap_first"
        )
        lines = original.splitlines(keepends=True)
        candidate = "".join(lines[: node.lineno - 1] + lines[node.end_lineno :])
    screen = Mock(wraps=screen_static_safety)
    monkeypatch.setattr(pf, "screen_static_safety", screen)
    report = pf.PromptForge(auto_commit=False)._screen_code_mutation(
        original, candidate, "src/api/routes/chat.py"
    )
    assert not report.safe
    assert report.effect is MutationEffect.UNSAFE
    assert "force_role_handling_changed" in report.reason
    assert screen.call_count == 1


@pytest.mark.parametrize("shrink", [False, True])
def test_non_identity_generic_replace_reports_keep_precedence(shrink, monkeypatch):
    original = "def first():\n    return 1\n\ndef second():\n    return 2\n"
    if shrink:
        original += "\n" * 20
    candidate = "def first():\n    return 1\n"
    screen = Mock(wraps=screen_static_safety)
    monkeypatch.setattr(pf, "screen_static_safety", screen)
    report = pf.PromptForge(auto_commit=False)._screen_code_mutation(
        original, candidate, "src/tool_policy.py"
    )
    assert not report.safe
    assert report.effect is MutationEffect.REPLACE
    expected = "catastrophic shrinkage:" if shrink else "missing public names: ['second']"
    assert report.reason.startswith(expected)
    assert screen.call_count == 1


def test_native_proposal_reports_unsafe_and_dispatch_never_evaluates(tmp_path, monkeypatch):
    target = "src/api/routes/chat.py"
    original = (ROOT / target).read_text()
    candidate = (
        original
        + "\ndef eval_identity_probe(request):\n    return request.workload_class == 'eval_batch'\n"
    )
    forge = pf.PromptForge(auto_commit=False)
    monkeypatch.setattr(forge, "_invoke_claude", lambda prompt: f"```python\n{candidate}\n```")
    monkeypatch.setattr(forge, "_attach_transfer_safety", lambda *args, **kwargs: None)
    # Metadata capture is best-effort and irrelevant to this guard; no journal or author call.
    import mutation_author_capture

    monkeypatch.setattr(mutation_author_capture, "capture_author_context", lambda *a, **k: None)
    mutation = forge.propose_code_mutation(target)
    assert mutation.effect is MutationEffect.UNSAFE
    assert "eval_identity_added_read" in mutation.effect_reason
    assert mutation.syntax_valid is False
    assert mutation.mutated_content == original
    context = SimpleNamespace(apply_file=Mock())
    assert forge.apply_code_mutation_in_context(context, mutation)["status"] == "rejected"
    context.apply_file.assert_not_called()
    assert (ROOT / target).read_text() == original

    from scripts.autopilot import actions

    monkeypatch.setattr(actions, "_build_mutation_context", lambda action, ctx: ("", {}))
    monkeypatch.setattr(actions, "author_capture_kwargs", lambda *a: {})
    monkeypatch.setattr(actions, "_record_rejected_mutation", Mock())
    monkeypatch.setattr(actions, "_discard_mutation_diversity_coverage", Mock())
    tower = Mock()
    result = actions._action_code_mutation(
        {"file": target},
        SimpleNamespace(
            forge=SimpleNamespace(propose_code_mutation=lambda **k: mutation), tower=tower
        ),
    )
    assert result == (None, "prompt_forge")
    assert (
        actions._record_rejected_mutation.call_args.kwargs["mutation"].effect
        is MutationEffect.UNSAFE
    )
    assert tower.mock_calls == []
