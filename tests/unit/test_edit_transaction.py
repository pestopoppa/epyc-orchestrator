"""Unit tests for the one-shot edit transaction (src/edit_transaction.py).

Covers parsing, the TRANSACTIONAL apply (snapshot -> write/delete -> self-check -> promote/rollback),
path-safety, and the end-to-end run() with a stub LLM. No inference.
"""
from __future__ import annotations

import pytest

from src.edit_transaction import (
    parse_edit_response, apply_edit_transaction, assemble_context, build_edit_prompt,
    run_edit_transaction, edit_transaction_enabled, _safe_join,
    EditScopeError, DEFAULT_MAX_BYTES,
)


# ── parsing ────────────────────────────────────────────────────────────
def test_parse_full_file_blocks():
    text = "<<<FILE: calc.py>>>\ndef add(a, b):\n    return a + b\n<<<END>>>\n<<<DELETE: old.py>>>"
    files, deletes = parse_edit_response(text)
    assert files == {"calc.py": "def add(a, b):\n    return a + b"}
    assert deletes == ["old.py"]


def test_parse_fence_fallback():
    files, _ = parse_edit_response("### utils.py\n```python\nX = 1\n```")
    assert files.get("utils.py") == "X = 1\n"


def test_parse_empty_or_prose():
    assert parse_edit_response("") == ({}, [])
    assert parse_edit_response("just prose, no blocks") == ({}, [])


# ── transactional apply ────────────────────────────────────────────────
def test_apply_success_keeps(tmp_path):
    (tmp_path / "calc.py").write_text("def double(x):\n    return x * 2\n")
    res = apply_edit_transaction(tmp_path, {"calc.py": "def double(x):\n    return x*2\ndef square(x):\n    return x*x\n"}, [])
    assert res.ok and res.written == ["calc.py"]
    assert "square" in (tmp_path / "calc.py").read_text()


def test_apply_rolls_back_on_syntax_error(tmp_path):
    (tmp_path / "calc.py").write_text("ORIGINAL = 1\n")
    res = apply_edit_transaction(tmp_path, {"calc.py": "def broken(:\n    pass\n"}, [])
    assert not res.ok and "Error" in res.error
    assert (tmp_path / "calc.py").read_text() == "ORIGINAL = 1\n"  # rolled back


def test_apply_rollback_undoes_partial_transaction(tmp_path):
    # one good edit + one syntactically-broken new file -> the WHOLE transaction rolls back.
    (tmp_path / "a.py").write_text("A = 1\n")
    res = apply_edit_transaction(tmp_path, {"a.py": "A = 2\n", "b.py": "def x(:\n"}, [])
    assert not res.ok
    assert (tmp_path / "a.py").read_text() == "A = 1\n"   # restored
    assert not (tmp_path / "b.py").exists()              # creation undone


def test_apply_runs_functional_verifier_after_self_check(tmp_path):
    (tmp_path / "calc.py").write_text("def square(x):\n    return 0\n")

    def verifier(root):
        ns = {}
        exec((root / "calc.py").read_text(), ns)
        return ns["square"](5) == 25

    res = apply_edit_transaction(
        tmp_path,
        {"calc.py": "def square(x):\n    return x * x\n"},
        [],
        verify_fn=verifier,
    )
    assert res.ok
    assert "return x * x" in (tmp_path / "calc.py").read_text()


def test_apply_rolls_back_on_functional_verifier_failure(tmp_path):
    (tmp_path / "calc.py").write_text("def square(x):\n    return 0\n")

    def verifier(root):
        ns = {}
        exec((root / "calc.py").read_text(), ns)
        return False, f"square(5)={ns['square'](5)}"

    res = apply_edit_transaction(
        tmp_path,
        {"calc.py": "def square(x):\n    return x + x\n"},
        [],
        verify_fn=verifier,
    )
    assert not res.ok
    assert "functional verifier failed" in res.error
    assert "square(5)=10" in res.error
    assert (tmp_path / "calc.py").read_text() == "def square(x):\n    return 0\n"


def test_apply_rolls_back_on_functional_verifier_exception(tmp_path):
    (tmp_path / "a.py").write_text("A = 1\n")
    (tmp_path / "gone.py").write_text("GONE = 1\n")

    def verifier(_root):
        raise AssertionError("task verifier rejected output")

    res = apply_edit_transaction(
        tmp_path,
        {"a.py": "A = 2\n", "new.py": "NEW = 1\n"},
        ["gone.py"],
        verify_fn=verifier,
    )
    assert not res.ok
    assert "task verifier rejected output" in res.error
    assert (tmp_path / "a.py").read_text() == "A = 1\n"
    assert (tmp_path / "gone.py").read_text() == "GONE = 1\n"
    assert not (tmp_path / "new.py").exists()


def test_apply_delete(tmp_path):
    (tmp_path / "helpers.py").write_text("def greet():\n    return 'hi'\n")
    res = apply_edit_transaction(tmp_path, {"utils.py": "def greet():\n    return 'hi'\n"}, ["helpers.py"])
    assert res.ok and not (tmp_path / "helpers.py").exists() and res.deleted == ["helpers.py"]
    assert (tmp_path / "utils.py").exists()


def test_apply_nothing_parsed(tmp_path):
    res = apply_edit_transaction(tmp_path, {}, [])
    assert not res.ok and "no valid file blocks" in res.error


# ── path safety ────────────────────────────────────────────────────────
def test_path_escape_aborts_whole_transaction(tmp_path):
    # FAIL-CLOSED (review #4): any unsafe path aborts the ENTIRE transaction — even the safe edit is
    # not applied, preserving the all-or-nothing safety claim for an agent-facing edit surface.
    res = apply_edit_transaction(tmp_path, {"../escape.py": "X=1\n", "/abs_escape.py": "Y=1\n",
                                            "ok.py": "Z=1\n"}, [])
    assert not res.ok
    assert "../escape.py" in res.rejected and "/abs_escape.py" in res.rejected
    assert res.written == []                              # nothing applied
    assert not (tmp_path / "ok.py").exists()              # the safe one was NOT written either
    assert not (tmp_path.parent / "escape.py").exists()   # escape did not write outside


def test_safe_join(tmp_path):
    assert _safe_join(tmp_path, "a/b.py") is not None       # nested preserved
    assert _safe_join(tmp_path, "../x.py") is None
    assert _safe_join(tmp_path, "/etc/passwd") is None


def test_apply_preserves_nested_paths(tmp_path):
    res = apply_edit_transaction(tmp_path, {"pkg/sub/mod.py": "VALUE = 7\n"}, [])
    assert res.ok and (tmp_path / "pkg" / "sub" / "mod.py").read_text() == "VALUE = 7\n"


# ── assemble + end-to-end with a stub LLM ──────────────────────────────
def test_assemble_and_prompt(tmp_path):
    (tmp_path / "calc.py").write_text("X = 1\n")
    ctx = assemble_context(tmp_path, ["calc.py"])
    assert ctx == {"calc.py": "X = 1\n"}
    prompt = build_edit_prompt("Add square", ctx)
    assert "Add square" in prompt and "calc.py" in prompt and "<<<FILE:" in prompt


def test_assemble_explicit_targets_are_deterministic_and_bounded(tmp_path):
    (tmp_path / "b.py").write_text("B = 2\n")
    (tmp_path / "a.py").write_text("A = 1\n")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "c.py").write_text("C = 3\n")

    ctx = assemble_context(tmp_path, ["b.py", "a.py", "b.py", "nested/../nested/c.py"])
    assert list(ctx) == ["a.py", "b.py", "nested/c.py"]
    assert ctx["a.py"] == "A = 1\n"
    assert "C = 3" in ctx["nested/c.py"]


def test_assemble_explicit_targets_reject_unsafe_paths(tmp_path):
    (tmp_path / "ok.py").write_text("OK = 1\n")
    with pytest.raises(EditScopeError, match="unsafe target file rejected"):
        assemble_context(tmp_path, ["ok.py", "../escape.py"])


def test_run_edit_transaction_with_stub_llm(tmp_path):
    (tmp_path / "calc.py").write_text("def double(x):\n    return x * 2\n")

    def stub(prompt):
        assert "Current file contents:" in prompt  # context was assembled into the prompt
        return "<<<FILE: calc.py>>>\ndef double(x):\n    return x*2\ndef square(x):\n    return x*x\n<<<END>>>"

    res, raw = run_edit_transaction(stub, "Add square(x) to calc.py", tmp_path, ["calc.py"])
    assert res.ok and "square" in (tmp_path / "calc.py").read_text()


def test_run_edit_transaction_review_is_default_inert(tmp_path):
    (tmp_path / "calc.py").write_text("VALUE = 1\n")

    calls = {"review": 0}

    def stub(_prompt):
        return "<<<FILE: calc.py>>>\nVALUE = 2\n<<<END>>>"

    def review(_context):
        calls["review"] += 1
        return {
            "risks": [],
            "blocking_issues": ["should not run"],
            "confidence": 1.0,
            "recommended_delta": "rerun",
        }, {}

    res, raw = run_edit_transaction(
        stub,
        "Edit value",
        tmp_path,
        ["calc.py"],
        review_before_commit=review,
    )

    assert raw
    assert res.ok
    assert calls["review"] == 0
    assert res.consult_events == []
    assert (tmp_path / "calc.py").read_text() == "VALUE = 2"


def test_run_edit_transaction_review_reruns_on_blocking_advisory(tmp_path):
    (tmp_path / "calc.py").write_text("VALUE = 1\n")
    prompts: list[str] = []

    def stub(prompt):
        prompts.append(prompt)
        if len(prompts) == 1:
            return "<<<FILE: calc.py>>>\nVALUE = 2\n<<<END>>>"
        assert "Architect review before commit found blocking issues" in prompt
        return "<<<FILE: calc.py>>>\nVALUE = 3\n<<<END>>>"

    def review(context):
        assert "VALUE = 2" in context
        return {
            "risks": ["missed requested value"],
            "blocking_issues": ["final value must be 3"],
            "confidence": 0.9,
            "recommended_delta": "write VALUE = 3",
        }, {"schema_hash": "abc123"}

    res, raw = run_edit_transaction(
        stub,
        "Set value",
        tmp_path,
        ["calc.py"],
        review_before_commit=review,
        enable_review_before_commit=True,
    )

    assert raw
    assert res.ok
    assert len(prompts) == 2
    assert res.consult_events[0]["success"] is True
    assert res.consult_events[0]["rerun_requested"] is True
    assert res.consult_events[0]["schema_hash"] == "abc123"
    assert (tmp_path / "calc.py").read_text() == "VALUE = 3"


def test_run_edit_transaction_review_denied_proceeds_with_original_draft(tmp_path):
    (tmp_path / "calc.py").write_text("VALUE = 1\n")

    def stub(_prompt):
        return "<<<FILE: calc.py>>>\nVALUE = 2\n<<<END>>>"

    def review(_context):
        raise RuntimeError("consult unavailable")

    res, raw = run_edit_transaction(
        stub,
        "Set value",
        tmp_path,
        ["calc.py"],
        review_before_commit=review,
        enable_review_before_commit=True,
    )

    assert raw
    assert res.ok
    assert res.consult_events[0]["success"] is False
    assert res.consult_events[0]["reason"] == "RuntimeError"
    assert (tmp_path / "calc.py").read_text() == "VALUE = 2"


def test_run_edit_transaction_review_gate_skips_consult(tmp_path):
    (tmp_path / "calc.py").write_text("VALUE = 1\n")
    calls = {"review": 0}

    def stub(_prompt):
        return "<<<FILE: calc.py>>>\nVALUE = 2\n<<<END>>>"

    def review(_context):
        calls["review"] += 1
        return {
            "risks": ["should not run"],
            "blocking_issues": ["should not run"],
            "confidence": 1.0,
        }, {}

    def gate(context):
        assert context["task_prompt"] == "Set value"
        assert context["draft_paths"] == ["calc.py"]
        return {"enabled": False, "reasons": ["plain_single_file_edit"]}

    res, raw = run_edit_transaction(
        stub,
        "Set value",
        tmp_path,
        ["calc.py"],
        review_before_commit=review,
        enable_review_before_commit=True,
        review_before_commit_gate=gate,
    )

    assert raw
    assert res.ok
    assert calls["review"] == 0
    assert res.consult_events == [
        {
            "interaction_type": "consult",
            "skill": "review_before_commit",
            "success": True,
            "skipped": True,
            "reason": "targeted_gate_skip",
            "gate_reasons": ["plain_single_file_edit"],
        }
    ]
    assert (tmp_path / "calc.py").read_text() == "VALUE = 2"


def test_run_edit_transaction_uses_only_explicit_targets(tmp_path):
    (tmp_path / "a.py").write_text("A = 1\n")
    (tmp_path / "b.py").write_text("B = 2\n")
    (tmp_path / "c.py").write_text("C = 3\n")

    seen = {"called": False}

    def stub(prompt):
        seen["called"] = True
        assert "--- a.py ---" in prompt
        assert "--- b.py ---" in prompt
        assert "--- c.py ---" not in prompt
        assert prompt.index("--- a.py ---") < prompt.index("--- b.py ---")
        return "<<<FILE: a.py>>>\nA = 10\n<<<END>>>\n<<<FILE: b.py>>>\nB = 20\n<<<END>>>"

    res, raw = run_edit_transaction(stub, "Edit two files", tmp_path, ["b.py", "a.py", "b.py"])
    assert seen["called"]
    assert raw
    assert res.ok
    assert (tmp_path / "a.py").read_text() == "A = 10"
    assert (tmp_path / "b.py").read_text() == "B = 20"
    assert (tmp_path / "c.py").read_text() == "C = 3\n"


def test_run_edit_transaction_passes_functional_verifier(tmp_path):
    (tmp_path / "calc.py").write_text("def square(x):\n    return 0\n")

    def stub(_prompt):
        return "<<<FILE: calc.py>>>\ndef square(x):\n    return x * x\n<<<END>>>"

    def verifier(root):
        ns = {}
        exec((root / "calc.py").read_text(), ns)
        return ns["square"](6) == 36

    res, raw = run_edit_transaction(
        stub,
        "Fix square",
        tmp_path,
        ["calc.py"],
        verify_fn=verifier,
    )
    assert raw
    assert res.ok
    assert "return x * x" in (tmp_path / "calc.py").read_text()


def test_flag_default_off(monkeypatch):
    monkeypatch.delenv("ORCHESTRATOR_EDIT_TRANSACTION", raising=False)
    assert edit_transaction_enabled() is False
    monkeypatch.setenv("ORCHESTRATOR_EDIT_TRANSACTION", "1")
    assert edit_transaction_enabled() is True


# ── review-hardening 2026-05-27: scope caps (#1) + no-__pycache__ self-check (#3) ──────
def test_assemble_caps_filecount(tmp_path):
    for i in range(5):
        (tmp_path / f"f{i}.py").write_text("X=1\n")
    assert len(assemble_context(tmp_path, max_files=10)) == 5     # within cap
    with pytest.raises(EditScopeError):
        assemble_context(tmp_path, max_files=3)                   # exceeds file cap -> fail-closed


def test_assemble_caps_bytes(tmp_path):
    (tmp_path / "big.py").write_text("X" * 1000)
    with pytest.raises(EditScopeError):
        assemble_context(tmp_path, max_bytes=100)                 # exceeds byte cap -> fail-closed


def test_caps_bound_via_stat_without_reading(tmp_path, monkeypatch):
    # #2: oversized scope is rejected via stat().st_size, WITHOUT loading file content into memory.
    import pathlib
    (tmp_path / "big.py").write_text("X" * 1000)
    monkeypatch.setattr(pathlib.Path, "read_text",
                        lambda self, *a, **k: pytest.fail("read_text called despite oversized scope"))
    with pytest.raises(EditScopeError):
        assemble_context(tmp_path, max_bytes=100)


def test_run_edit_transaction_failclosed_on_oversized_scope(tmp_path):
    # Unscoped whole-root assembly over caps must fail-closed BEFORE calling the model or writing.
    (tmp_path / "big.py").write_text("X" * (DEFAULT_MAX_BYTES + 1))
    called = {"n": 0}

    def stub(prompt):
        called["n"] += 1
        return "<<<FILE: big.py>>>\nY = 2\n<<<END>>>"

    res, _raw = run_edit_transaction(stub, "edit", tmp_path, target_files=None)
    assert not res.ok and "scope too large" in res.error
    assert called["n"] == 0                                       # model NOT called
    assert (tmp_path / "big.py").read_text().startswith("XXXX")   # original untouched


def test_self_check_no_pycache_side_effect(tmp_path):
    # compile(source, path, "exec") validates syntax WITHOUT writing __pycache__/*.pyc that the
    # snapshot/rollback wouldn't track.
    (tmp_path / "m.py").write_text("OLD = 1\n")
    res = apply_edit_transaction(tmp_path, {"m.py": "NEW = 2\n"}, [])
    assert res.ok
    assert not (tmp_path / "__pycache__").exists()
    assert list(tmp_path.rglob("*.pyc")) == []
