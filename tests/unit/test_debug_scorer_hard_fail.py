"""Hard-fail scorer semantics (audit A1 + A2 scorer-identity leg).

A requested scorer that cannot run — missing math_verify, a down llm_judge,
an unparseable GOLD answer, an unknown programmatic verifier — must surface
loudly (ScoringUnavailableError / ValueError), NEVER silently degrade to a
different scorer and score a wrong answer as (in)correct. The catastrophic
prior bug: math_verify's signal.alarm raises ValueError off the main thread,
the bare ``except Exception -> _score_exact_match`` ate it, and every threaded
math eval was scored with the wrong scorer.
"""

from __future__ import annotations

import concurrent.futures
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "benchmark"))

from debug_scorer import ScoringUnavailableError, score_answer  # noqa: E402


# ── math_verify: threaded correctness + no silent fallback ───────────────


def test_math_verify_scores_correctly_in_worker_thread() -> None:
    pytest.importorskip("math_verify")

    def _run(ans: str) -> bool:
        return score_answer(
            answer=ans,
            expected="\\frac{1}{2}",
            scoring_method="math_verify",
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        correct = ex.submit(_run, "The result is \\boxed{\\frac{1}{2}}").result()
        wrong = ex.submit(_run, "The result is \\boxed{\\frac{1}{3}}").result()

    # If the thread/signal ValueError were still being swallowed into
    # exact_match, the correct \frac{1}{2} answer would NOT score True.
    assert correct is True
    assert wrong is False


def test_math_verify_thread_signal_error_raises_not_silent(monkeypatch) -> None:
    # Mocked variant (no real math_verify needed): parse() raises the exact
    # thread/signal ValueError math_verify raises off the main thread. The
    # scorer MUST surface ScoringUnavailableError, not a silent exact_match.
    stub = types.ModuleType("math_verify")

    def _parse(*args, **kwargs):  # noqa: ANN002, ANN003
        raise ValueError(
            "signal only works in main thread of the main interpreter"
        )

    def _verify(gold, pred):  # pragma: no cover - never reached
        return True

    stub.parse = _parse
    stub.verify = _verify
    monkeypatch.setitem(sys.modules, "math_verify", stub)

    with pytest.raises(ScoringUnavailableError):
        score_answer(
            answer="The result is \\boxed{\\frac{1}{2}}",
            expected="\\frac{1}{2}",
            scoring_method="math_verify",
        )


def test_math_verify_missing_dependency_raises(monkeypatch) -> None:
    # sys.modules[name] = None makes `from math_verify import ...` ImportError.
    monkeypatch.setitem(sys.modules, "math_verify", None)
    with pytest.raises(ScoringUnavailableError):
        score_answer(
            answer="\\boxed{2}",
            expected="2",
            scoring_method="math_verify",
        )


def test_math_verify_gold_parse_failure_raises() -> None:
    # A GOLD answer that math_verify extracts nothing from is a dataset/gold
    # defect -> ScoringUnavailableError (an item that cannot be scored), not a
    # False that would mask the defect as a model miss.
    pytest.importorskip("math_verify")
    with pytest.raises(ScoringUnavailableError):
        score_answer(
            answer="\\boxed{2}",
            expected="\\begin{invalid",
            scoring_method="math_verify",
        )


def test_pred_garbage_scores_false() -> None:
    # The MODEL's answer failing to parse is a task failure -> False, NOT
    # scorer-unavailability. GOLD parses fine here.
    pytest.importorskip("math_verify")
    result = score_answer(
        answer="the quick brown fox says hello",
        expected="42",
        scoring_method="math_verify",
    )
    assert result is False


# ── llm_judge: unreachable judge must raise, not substring-fallback ──────


def test_llm_judge_unreachable_raises() -> None:
    # Judge on 127.0.0.1:1 is unreachable; expected string is NOT a substring
    # of the answer, so the top-of-function substring fast-path does not fire.
    # The transport failure must raise, not silently fall back to substring.
    with pytest.raises(ScoringUnavailableError):
        score_answer(
            answer="the model said something entirely different",
            expected="mg/2",
            scoring_method="llm_judge",
            scoring_config={
                "judge_port": 1,
                "judge_host": "127.0.0.1",
                "timeout": 2,
            },
        )


# ── programmatic: unknown verifier is a config defect ────────────────────


def test_unknown_programmatic_verifier_raises() -> None:
    with pytest.raises(ValueError):
        score_answer(
            answer="anything at all",
            expected="whatever",
            scoring_method="programmatic",
            scoring_config={"verifier": "definitely_not_a_verifier"},
        )


# ── A2 leg: seeding_scoring pins the orchestrator debug_scorer copy ──────


def test_seeding_scoring_binds_orchestrator_copy() -> None:
    research_bench = "/mnt/raid0/llm/epyc-inference-research/scripts/benchmark"
    orch_bench = str(REPO_ROOT / "scripts" / "benchmark")

    saved_path = list(sys.path)
    watched_keys = (
        "debug_scorer",
        "epyc_orch_debug_scorer",
        "seeding_scoring",
        "seeding_types",
    )
    saved_modules = {k: sys.modules.get(k) for k in watched_keys}
    try:
        # Force a fresh bare `import debug_scorer` to bind the RESEARCH copy,
        # so we prove score_answer_deterministic ignores that binding and
        # pins the orchestrator copy under its private key.
        sys.modules.pop("debug_scorer", None)
        sys.modules.pop("epyc_orch_debug_scorer", None)
        sys.path.insert(0, research_bench)
        try:
            import debug_scorer  # noqa: F401
        except Exception:
            pass

        sys.path.insert(0, orch_bench)
        import seeding_scoring  # noqa: E402

        # Trivial exact_match: last-line "42" numerically equals expected "42".
        result = seeding_scoring.score_answer_deterministic(
            "42", "42", "exact_match"
        )
        assert result is True

        bound = sys.modules["epyc_orch_debug_scorer"]
        assert bound.__file__ is not None
        assert bound.__file__.startswith(orch_bench)
        assert "epyc-orchestrator" in bound.__file__
    finally:
        sys.path[:] = saved_path
        for key, val in saved_modules.items():
            if val is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = val
