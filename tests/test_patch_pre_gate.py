"""Tests for the execution-free patch pre-gate SIGNAL (EV-12 / RE-2).

Zero inference, zero server, no execution of patched code. The signal is a pure
policy layer over ``verify_patch``; these tests drive it with real small unified
diffs against in-memory base trees (and one real git work tree) and assert the
verdict + escalation decision.

Run: ``.venv/bin/python -m pytest tests/test_patch_pre_gate.py`` (single file,
no ``-n auto`` — the checks are fast and hermetic).
"""

import shutil
import subprocess
from pathlib import Path

import pytest

from src.verification.patch_pre_gate import (
    ESCALATE_ON_VERDICT,
    PreGateSignal,
    evaluate_patch_pre_gate,
    should_escalate_patch,
)
from src.verification.patch_verifier import (
    CERT_DIFF,
    CERT_STACK_TRACE,
    FAIL,
    INCONCLUSIVE,
    PASS,
)

# ── real fixtures ──────────────────────────────────────────────────────────

# A tiny, syntactically valid base module.
_BASE = {"pkg/mod.py": "def add(a, b):\n    return a + b\n"}

# 1) Applies cleanly AND the result compiles -> PASS -> proceed (escalate).
_DIFF_OK = (
    "--- a/pkg/mod.py\n"
    "+++ b/pkg/mod.py\n"
    "@@ -1,2 +1,2 @@\n"
    " def add(a, b):\n"
    "-    return a + b\n"
    "+    return a + b + 0\n"
)

# 2) Context does NOT exist in the base -> patch will not apply (git-apply /
#    hunk-context FAIL) -> FAIL -> do NOT escalate.
_DIFF_BAD_CONTEXT = (
    "--- a/pkg/mod.py\n"
    "+++ b/pkg/mod.py\n"
    "@@ -1,2 +1,2 @@\n"
    " def subtract(a, b):\n"
    "-    return a - b\n"
    "+    return a - b - 0\n"
)

# 3) Applies cleanly BUT produces invalid Python -> py_compile / syntax FAIL
#    -> FAIL -> do NOT escalate.
_DIFF_BREAKS_COMPILE = (
    "--- a/pkg/mod.py\n"
    "+++ b/pkg/mod.py\n"
    "@@ -1,2 +1,2 @@\n"
    " def add(a, b):\n"
    "-    return a + b\n"
    "+    return a + b +\n"  # trailing binary operator -> SyntaxError
)


# ── PASS / proceed ─────────────────────────────────────────────────────────


class TestPassProceeds:
    def test_clean_patch_passes_and_escalates(self):
        sig = evaluate_patch_pre_gate(_DIFF_OK, _BASE, run_lint=False)
        assert isinstance(sig, PreGateSignal)
        assert sig.verdict == PASS
        assert sig.should_escalate is True
        assert sig.certificate_type is None
        assert "PASS" in sig.reason

    def test_signal_dict_has_exactly_the_contract_keys(self):
        d = evaluate_patch_pre_gate(_DIFF_OK, _BASE, run_lint=False).to_dict()
        assert set(d) == {"verdict", "certificate_type", "reason", "should_escalate"}
        assert d["verdict"] == PASS
        assert d["should_escalate"] is True

    def test_convenience_helper_matches(self):
        assert should_escalate_patch(_DIFF_OK, _BASE, run_lint=False) is True


# ── FAIL / do-not-escalate ─────────────────────────────────────────────────


class TestFailSuppressesEscalation:
    def test_unapplied_patch_fails_and_does_not_escalate(self):
        sig = evaluate_patch_pre_gate(_DIFF_BAD_CONTEXT, _BASE, run_lint=False)
        assert sig.verdict == FAIL
        assert sig.should_escalate is False
        # apply/context failure carries a diff certificate
        assert sig.certificate_type == CERT_DIFF
        assert "provably non-viable" in sig.reason

    def test_compile_breaking_patch_fails_and_does_not_escalate(self):
        sig = evaluate_patch_pre_gate(_DIFF_BREAKS_COMPILE, _BASE, run_lint=False)
        assert sig.verdict == FAIL
        assert sig.should_escalate is False
        # a py_compile / syntax failure surfaces a stack_trace certificate
        assert sig.certificate_type == CERT_STACK_TRACE

    def test_fail_dict_shape(self):
        d = evaluate_patch_pre_gate(_DIFF_BAD_CONTEXT, _BASE, run_lint=False).to_dict()
        assert d["verdict"] == FAIL
        assert d["should_escalate"] is False
        assert d["certificate_type"] == CERT_DIFF


# ── INCONCLUSIVE / proceed (err toward spending inference) ─────────────────


class TestInconclusiveProceeds:
    def test_unresolvable_base_is_inconclusive_and_escalates(self, tmp_path):
        missing = tmp_path / "does_not_exist"
        sig = evaluate_patch_pre_gate(_DIFF_OK, str(missing), run_lint=False)
        assert sig.verdict == INCONCLUSIVE
        assert sig.should_escalate is True
        assert sig.certificate_type is None

    def test_empty_patch_is_inconclusive_and_escalates(self):
        sig = evaluate_patch_pre_gate("", _BASE, run_lint=False)
        assert sig.verdict == INCONCLUSIVE
        assert sig.should_escalate is True


# ── policy table + git-apply parity ────────────────────────────────────────


class TestPolicy:
    def test_policy_table_only_fail_suppresses(self):
        assert ESCALATE_ON_VERDICT[PASS] is True
        assert ESCALATE_ON_VERDICT[INCONCLUSIVE] is True
        assert ESCALATE_ON_VERDICT[FAIL] is False

    def test_should_escalate_tracks_policy_table(self):
        for diff, base, kwargs in (
            (_DIFF_OK, _BASE, {}),
            (_DIFF_BAD_CONTEXT, _BASE, {}),
            (_DIFF_BREAKS_COMPILE, _BASE, {}),
        ):
            sig = evaluate_patch_pre_gate(diff, base, run_lint=False, **kwargs)
            assert sig.should_escalate is ESCALATE_ON_VERDICT[sig.verdict]

    @pytest.mark.skipif(shutil.which("git") is None, reason="git not available")
    def test_literal_git_apply_check_fail(self, tmp_path):
        # Materialize a real git work tree so verify_patch runs `git apply
        # --check` for real; a bad-context patch must FAIL and suppress escalation.
        subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        (pkg / "mod.py").write_text(_BASE["pkg/mod.py"], encoding="utf-8")

        ok = evaluate_patch_pre_gate(_DIFF_OK, str(tmp_path), run_lint=False, use_git=True)
        assert ok.verdict == PASS
        assert ok.should_escalate is True

        bad = evaluate_patch_pre_gate(
            _DIFF_BAD_CONTEXT, str(tmp_path), run_lint=False, use_git=True
        )
        assert bad.verdict == FAIL
        assert bad.should_escalate is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([str(Path(__file__)), "-q"]))
