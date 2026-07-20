"""BUILD-evalbatch-verifier-mode contract tests (EV-4 / EV-11 / EV-5/7/8).

Fixture-driven, INFERENCE-FREE. Covers:
  * the six EV-4 calibration metrics computed on synthetic (confidence, label)
    vectors with concrete hand-verified expected values;
  * math_verify per-question scoring + dataset_sha256 recording on a tiny
    synthetic math set (real scorer, no server);
  * the window-runner verifier-mode surface: default tier path unchanged,
    plan-only performs no inference, and the EV-5/7/8 verifier-model pass is
    MODEL-DOWNLOAD gated.

Run: .venv/bin/python -m pytest tests/unit/test_eval_verifier_mode.py -q
(no -n auto — these are cheap and share process-global sys.path state).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (
    REPO_ROOT / "scripts" / "autopilot",
    REPO_ROOT / "scripts" / "benchmark",
    REPO_ROOT,
):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from eval_tower import (  # type: ignore[import-not-found]
    CALIBRATION_METRIC_KEYS,
    EvalTower,
    compute_calibration_metrics,
    dataset_content_sha256,
    score_math_rebaseline_answers,
)

_WINDOW_PATH = REPO_ROOT / "scripts" / "benchmark" / "eval_batch_serving_evaltower_window.py"
_spec = importlib.util.spec_from_file_location("eval_batch_serving_evaltower_window", _WINDOW_PATH)
assert _spec is not None and _spec.loader is not None
window = importlib.util.module_from_spec(_spec)
sys.modules["eval_batch_serving_evaltower_window"] = window
_spec.loader.exec_module(window)


# ── (1) EV-4 calibration metrics — concrete fixtures, no inference ────────────


def test_calibration_metric_keys_are_the_six_ev4_metrics() -> None:
    assert CALIBRATION_METRIC_KEYS == (
        "ece",
        "auroc",
        "top1_accuracy",
        "bottom1_accuracy",
        "spearman_rho",
        "mae",
    )


def test_calibration_metrics_concrete_values() -> None:
    # confidences monotone with a clean per-bin structure; labels [0,0,1,1].
    #   ECE   : each of 4 items lands alone in its 0.1-width bin →
    #           (|0-.2|+|0-.4|+|1-.6|+|1-.8|)/4 = 1.2/4 = 0.3
    #   AUROC : both positives (.6,.8) outrank both negatives (.2,.4) → 1.0
    #   Top-1 : max-conf item (.8) is correct → 1.0
    #   Bot-1 : min-conf item (.2) is incorrect → 0.0
    #   rho   : tie-averaged rank pearson = 4/sqrt(20) = 0.8944271909999159
    #   MAE   : same as ECE here (one item/bin) → 0.3
    m = compute_calibration_metrics([0.2, 0.4, 0.6, 0.8], [0, 0, 1, 1])
    assert m["n"] == 4
    assert m["ece"] == pytest.approx(0.3, abs=1e-12)
    assert m["auroc"] == pytest.approx(1.0, abs=1e-12)
    assert m["top1_accuracy"] == pytest.approx(1.0, abs=1e-12)
    assert m["bottom1_accuracy"] == pytest.approx(0.0, abs=1e-12)
    assert m["spearman_rho"] == pytest.approx(0.8944271909999159, abs=1e-9)
    assert m["mae"] == pytest.approx(0.3, abs=1e-12)


def test_calibration_top_bottom_cohorts_average_ties() -> None:
    # Two items share the max conf (0.9) with labels [1,0] → top1 = 0.5;
    # two share the min conf (0.1) with labels [0,1] → bottom1 = 0.5.
    m = compute_calibration_metrics([0.9, 0.9, 0.1, 0.1], [1, 0, 0, 1])
    assert m["top1_accuracy"] == pytest.approx(0.5, abs=1e-12)
    assert m["bottom1_accuracy"] == pytest.approx(0.5, abs=1e-12)


def test_calibration_metrics_anticorrelated_signal() -> None:
    # Confidence INVERSELY related to correctness: worst possible discrimination.
    m = compute_calibration_metrics([0.8, 0.6, 0.4, 0.2], [0, 0, 1, 1])
    assert m["auroc"] == pytest.approx(0.0, abs=1e-12)
    assert m["top1_accuracy"] == pytest.approx(0.0, abs=1e-12)
    assert m["bottom1_accuracy"] == pytest.approx(1.0, abs=1e-12)
    assert m["spearman_rho"] == pytest.approx(-0.8944271909999159, abs=1e-9)


def test_calibration_metrics_degenerate_confidence_and_class() -> None:
    # All-equal confidence → AUROC undefined (guard), Spearman undefined (const).
    m = compute_calibration_metrics([0.5, 0.5, 0.5, 0.5], [1, 0, 1, 0])
    assert m["auroc"] is None
    assert m["spearman_rho"] is None
    # top/bottom cohorts collapse to the whole set: 2/4 correct → 0.5.
    assert m["top1_accuracy"] == pytest.approx(0.5, abs=1e-12)
    assert m["mae"] == pytest.approx(0.5, abs=1e-12)

    # Single class present → AUROC undefined even with distinct confidences.
    single = compute_calibration_metrics([0.1, 0.4, 0.9], [1, 1, 1])
    assert single["auroc"] is None


def test_calibration_metrics_empty_is_all_none() -> None:
    m = compute_calibration_metrics([], [])
    assert m["n"] == 0
    for key in CALIBRATION_METRIC_KEYS:
        assert m[key] is None


def test_calibration_metrics_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        compute_calibration_metrics([0.1, 0.2], [1])


# ── (2) math_verify scoring + dataset_sha256 — tiny synthetic set, no server ──


def _tiny_math_set() -> list[dict]:
    return [
        {
            "id": "gsm8k_00001",
            "prompt": "1+1? Put your final answer in \\boxed{}.",
            "expected": "2",
            "scoring_method": "math_verify",
            "scoring_config": {"extraction_mode": "expr"},
        },
        {
            "id": "math500_algebra_00001",
            "prompt": "50*2? Put your final answer in \\boxed{}.",
            "expected": "100",
            "scoring_method": "math_verify",
            "scoring_config": {"extraction_mode": "expr"},
        },
    ]


def test_score_math_rebaseline_answers_uses_real_math_verify() -> None:
    questions = _tiny_math_set()
    # First answer correct (boxed 2), second wrong (boxed 99 vs 100).
    scored = score_math_rebaseline_answers(
        questions,
        ["The result is \\boxed{2}", "Final: \\boxed{99}"],
    )
    assert scored == [True, False]

    # Flip the second to correct to prove the scorer actually verifies, not
    # a constant.
    scored2 = score_math_rebaseline_answers(
        questions,
        ["The result is \\boxed{2}", "Final: \\boxed{100}"],
    )
    assert scored2 == [True, True]


def test_score_math_rebaseline_handles_nested_boxed_latex() -> None:
    questions = [
        {
            "id": "math500_frac_00001",
            "prompt": "Return the value. Put your final answer in \\boxed{}.",
            "expected": "\\frac{1}{2}",
            "scoring_method": "math_verify",
            "scoring_config": {"extraction_mode": "expr"},
        }
    ]

    assert score_math_rebaseline_answers(
        questions,
        ["Work omitted. Therefore the answer is \\boxed{\\frac{1}{2}}."],
    ) == [True]


def test_score_math_rebaseline_hard_fails_without_math_verify(monkeypatch) -> None:
    # EV-11: a missing math-verify must RAISE, never silently fall back.
    monkeypatch.setitem(sys.modules, "math_verify", None)
    with pytest.raises(RuntimeError) as exc:
        score_math_rebaseline_answers(_tiny_math_set()[:1], ["\\boxed{2}"])
    assert "math_verify" in str(exc.value).lower() or "math-verify" in str(exc.value).lower()


def test_score_math_rebaseline_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        score_math_rebaseline_answers(_tiny_math_set(), ["only-one-answer"])


def test_dataset_content_sha256_is_deterministic_and_order_sensitive() -> None:
    questions = _tiny_math_set()
    digest = dataset_content_sha256(questions)

    # 64-hex, deterministic across calls.
    assert len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)
    assert dataset_content_sha256(questions) == digest

    # Order is part of identity: reversing changes the digest.
    assert dataset_content_sha256(list(reversed(questions))) != digest
    # Suite and scoring config are part of instrument identity.
    changed_suite = [dict(q) for q in questions]
    changed_suite[0]["suite"] = "math_alt"
    assert dataset_content_sha256(changed_suite) != digest
    changed_config = [dict(q) for q in questions]
    changed_config[0]["scoring_config"] = {"extraction_mode": "expr", "strict": False}
    assert dataset_content_sha256(changed_config) != digest

    # Independently recompute the exact hashing scheme to pin the algorithm.
    h = hashlib.sha256()
    for q in questions:
        for name in ("suite", "id", "prompt", "expected", "scoring_method", "scoring_config"):
            value = q.get(name, "")
            if isinstance(value, (dict, list, tuple)):
                value = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
            h.update(str(value).encode("utf-8", "replace"))
            h.update(b"\x00")
        h.update(b"\x1e")
    assert digest == h.hexdigest()


def test_filter_questions_by_split_is_strict_after_normalization() -> None:
    questions = [
        {"id": "sv_he_r_plus_001", "metadata": {"subset": "HE-R+"}},
        {"id": "sv_he_r_002", "metadata": {"subset": "HE-R"}},
        {"id": "he_r_plus_003"},
        {"id": "he_r_004"},
        {"id": "gsm8k_00001", "metadata": {"subset": "gsm8k"}},
        {"id": "math500_00001", "metadata": {"subset": "math500"}},
        {"id": "math500_00002"},
    ]

    assert [q["id"] for q in EvalTower._filter_questions_by_split(questions, "HE-R+")] == [
        "sv_he_r_plus_001",
        "he_r_plus_003",
    ]
    assert [q["id"] for q in EvalTower._filter_questions_by_split(questions, "HE-R")] == [
        "sv_he_r_002",
        "he_r_004",
    ]
    assert [q["id"] for q in EvalTower._filter_questions_by_split(questions, "gsm8k")] == [
        "gsm8k_00001"
    ]
    assert [q["id"] for q in EvalTower._filter_questions_by_split(questions, "math500")] == [
        "math500_00001",
        "math500_00002",
    ]
    assert EvalTower._filter_questions_by_split(questions, "math") == []


# ── (3) window-runner verifier-mode surface ──────────────────────────────────


def _healthy_preflight(*, autopilot_active: bool = False) -> dict:
    return {"api_health": {"ok": True}, "autopilot_active": autopilot_active}


class _ExplodingTower:
    """Fails on construction — proves plan-only never touches inference."""

    def __init__(self, *_args, **_kwargs) -> None:  # noqa: D401
        raise AssertionError("EvalTower must not be constructed in plan-only mode")


def test_default_mode_is_tier_and_leaves_tier_path() -> None:
    args = window.parse_args(["--tier", "2", "--n", "7"])
    assert args.mode == "tier"


def test_verifier_mode_plan_only_runs_no_inference(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(window.activation_window, "build_preflight", lambda _a: _healthy_preflight())
    monkeypatch.setattr(window, "EvalTower", _ExplodingTower)

    args = window.parse_args(
        [
            "--mode",
            "math_rebaseline",
            "--full",
            "--roles",
            "worker_general,worker_math",
            "--output-dir",
            str(tmp_path),
        ]
    )
    report, rc = window.build_verifier_report(args, output_dir=tmp_path)

    assert rc == 0
    assert report["status"] == "plan_only"
    assert report["applied"] is False
    assert report["result"] is None
    assert report["decision_grade"] is False
    assert report["mode"] == "math_rebaseline"
    assert "--apply --confirm-clean-window" in report["pin_command"]
    # Writing the report is part of the surface and must be JSON-clean.
    json_path, md_path = window.write_verifier_report(report, tmp_path)
    assert json_path.exists() and md_path.exists()


def test_ev4_calibration_plan_only_pin_and_no_inference(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(window.activation_window, "build_preflight", lambda _a: _healthy_preflight())
    monkeypatch.setattr(window, "EvalTower", _ExplodingTower)

    args = window.parse_args(
        [
            "--mode",
            "calibration",
            "--suite",
            "scoring_verifiers",
            "--split",
            "HE-R+",
            "--roles",
            "worker_general",
            "--full",
            "--min-eval-concurrency",
            "3",
            "--output-dir",
            str(tmp_path),
        ]
    )
    report, rc = window.build_verifier_report(args, output_dir=tmp_path)

    assert rc == 0
    assert report["status"] == "plan_only"
    assert report["result"] is None
    pin = report["pin_command"]
    assert "--mode calibration" in pin
    assert "--suite scoring_verifiers" in pin
    assert "--split HE-R+" in pin
    assert "--min-eval-concurrency 3" in pin


def test_verifier_mode_apply_blocks_when_fanout_resolves_serial(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(window.activation_window, "build_preflight", lambda _a: _healthy_preflight())
    observed_roles: list[tuple[str, ...]] = []

    def fake_resolved_eval_concurrency(roles=None):
        observed_roles.append(tuple(roles or ()))
        return 1

    monkeypatch.setattr(window, "_resolved_eval_concurrency", fake_resolved_eval_concurrency)
    monkeypatch.setattr(window, "_optimized_live_stack_status", lambda: {"ok": True, "warnings": []})
    monkeypatch.setattr(window, "EvalTower", _ExplodingTower)

    args = window.parse_args(
        [
            "--mode",
            "calibration",
            "--suite",
            "scoring_verifiers",
            "--split",
            "HE-R+",
            "--roles",
            "worker_general,frontdoor",
            "--full",
            "--min-eval-concurrency",
            "3",
            "--apply",
            "--confirm-clean-window",
            "--output-dir",
            str(tmp_path),
        ]
    )
    report, rc = window.build_verifier_report(args, output_dir=tmp_path)

    assert report["status"] == "blocked"
    assert rc != 0
    assert report["result"] is None
    assert report["resolved_eval_concurrency"] == 1
    assert report["min_eval_concurrency"] == 3
    assert observed_roles == [("worker_general", "frontdoor")]
    assert any("resolved EvalTower concurrency 1" in b for b in report["blockers"])


def test_verifier_mode_apply_requires_explicit_fanout_floor(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(window.activation_window, "build_preflight", lambda _a: _healthy_preflight())
    monkeypatch.setattr(window, "_resolved_eval_concurrency", lambda _roles=None: 1)
    monkeypatch.setattr(window, "_optimized_live_stack_status", lambda: {"ok": True, "warnings": []})
    monkeypatch.setattr(window, "EvalTower", _ExplodingTower)

    args = window.parse_args(
        [
            "--mode",
            "calibration",
            "--suite",
            "scoring_verifiers",
            "--split",
            "HE-R+",
            "--roles",
            "worker_general",
            "--full",
            "--apply",
            "--confirm-clean-window",
            "--output-dir",
            str(tmp_path),
        ]
    )
    report, rc = window.build_verifier_report(args, output_dir=tmp_path)

    assert report["status"] == "blocked"
    assert rc != 0
    assert report["decision_grade"] is False
    assert any("requires explicit --min-eval-concurrency" in b for b in report["blockers"])


def test_verifier_model_pass_is_download_gated(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(window.activation_window, "build_preflight", lambda _a: _healthy_preflight())
    monkeypatch.setattr(window, "_resolved_eval_concurrency", lambda _roles=None: 3)
    monkeypatch.setattr(window, "_optimized_live_stack_status", lambda: {"ok": True, "warnings": []})
    monkeypatch.setattr(window, "EvalTower", _ExplodingTower)

    args = window.parse_args(
        [
            "--mode",
            "math_rebaseline",
            "--full",
            "--verifier",
            "thinkprm",
            "--min-eval-concurrency",
            "3",
            "--apply",
            "--confirm-clean-window",
            "--output-dir",
            str(tmp_path),
        ]
    )
    report, rc = window.build_verifier_report(args, output_dir=tmp_path)

    # Blocked BEFORE any inference: thinkprm is not on disk.
    assert report["status"] == "blocked"
    assert rc != 0
    assert report["result"] is None
    assert report["decision_grade"] is False
    assert report["verifier_gate"]["on_disk"] is False
    assert report["verifier_gate"]["required_download"] == "MODEL-DOWNLOAD-THINKPRM-1.5B"
    assert any("MODEL-DOWNLOAD-THINKPRM-1.5B" in b for b in report["blockers"])


def test_require_verifier_on_disk_raises_model_download() -> None:
    with pytest.raises(RuntimeError) as exc:
        window.require_verifier_on_disk("ouro-2.6b")
    assert "MODEL-DOWNLOAD-OURO-2.6B" in str(exc.value)


def test_apply_without_confirm_clean_window_is_blocked(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(window.activation_window, "build_preflight", lambda _a: _healthy_preflight())
    monkeypatch.setattr(window, "_resolved_eval_concurrency", lambda _roles=None: 3)
    monkeypatch.setattr(window, "_optimized_live_stack_status", lambda: {"ok": True, "warnings": []})
    monkeypatch.setattr(window, "EvalTower", _ExplodingTower)

    args = window.parse_args(
        [
            "--mode",
            "math_rebaseline",
            "--full",
            "--min-eval-concurrency",
            "3",
            "--apply",
            "--output-dir",
            str(tmp_path),
        ]
    )
    report, rc = window.build_verifier_report(args, output_dir=tmp_path)

    assert report["status"] == "blocked"
    assert rc != 0
    assert any("confirm-clean-window" in b for b in report["blockers"])
    assert report["result"] is None


def test_verifier_mode_apply_writes_progress_files(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(window.activation_window, "build_preflight", lambda _a: _healthy_preflight())
    monkeypatch.setattr(window, "_resolved_eval_concurrency", lambda _roles=None: 4)
    monkeypatch.setattr(window, "_optimized_live_stack_status", lambda: {"ok": True, "warnings": []})

    class _ProgressTower:
        def __init__(self, *_args, on_progress=None, **_kwargs) -> None:
            self.on_progress = on_progress

        def eval_calibration(self, *, suite, split, roles, seed, n, full):  # noqa: ARG002
            assert self.on_progress is not None
            self.on_progress(
                {
                    "label": "cal-worker_general",
                    "completed_questions": 1,
                    "total_questions": 2,
                    "correct_questions": 1,
                    "correct_pct": 100.0,
                    "concurrency": 4,
                }
            )
            self.on_progress(
                {
                    "label": "cal-worker_general",
                    "completed_questions": 2,
                    "total_questions": 2,
                    "correct_questions": 1,
                    "correct_pct": 50.0,
                    "concurrency": 4,
                }
            )
            return {
                "dataset_sha256": "abc",
                "n_questions": 2,
                "roles": roles,
                "per_role": {
                    role: {"metrics": {"ece": 0.5}, "n": 2}
                    for role in roles
                },
            }

    monkeypatch.setattr(window, "EvalTower", _ProgressTower)
    args = window.parse_args(
        [
            "--mode",
            "calibration",
            "--suite",
            "scoring_verifiers",
            "--split",
            "HE-R+",
            "--roles",
            "worker_general",
            "--n",
            "2",
            "--min-eval-concurrency",
            "3",
            "--apply",
            "--confirm-clean-window",
            "--output-dir",
            str(tmp_path),
        ]
    )

    report, rc = window.build_verifier_report(args, output_dir=tmp_path)

    assert rc == 0
    assert report["status"] == "complete"
    assert report["decision_grade"] is True
    assert report["live_stack_contract"] == {"ok": True, "warnings": []}
    progress_lines = (tmp_path / "progress.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(progress_lines) == 2
    current = json.loads((tmp_path / "progress.current.json").read_text(encoding="utf-8"))
    assert current["seq"] == 2
    assert current["context"]["resolved_eval_concurrency"] == 4
    assert current["event"]["completed_questions"] == 2


def test_verifier_mode_partial_scored_result_is_not_decision_grade(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(window.activation_window, "build_preflight", lambda _a: _healthy_preflight())
    monkeypatch.setattr(window, "_resolved_eval_concurrency", lambda _roles=None: 4)
    monkeypatch.setattr(window, "_optimized_live_stack_status", lambda: {"ok": True, "warnings": []})

    class _PartialScoredTower:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def eval_calibration(self, *, suite, split, roles, seed, n, full):  # noqa: ARG002
            return {
                "dataset_sha256": "abc",
                "n_questions": 2,
                "roles": roles,
                "per_role": {
                    role: {
                        "n_questions": 2,
                        "n_scored": 1,
                        "accuracy": 1.0,
                        "ece": 0.0,
                    }
                    for role in roles
                },
            }

    monkeypatch.setattr(window, "EvalTower", _PartialScoredTower)
    args = window.parse_args(
        [
            "--mode",
            "calibration",
            "--suite",
            "scoring_verifiers",
            "--split",
            "HE-R+",
            "--roles",
            "worker_general",
            "--n",
            "2",
            "--min-eval-concurrency",
            "3",
            "--apply",
            "--confirm-clean-window",
            "--output-dir",
            str(tmp_path),
        ]
    )

    report, rc = window.build_verifier_report(args, output_dir=tmp_path)

    assert rc == 75
    assert report["status"] == "eval_degenerate"
    assert report["decision_grade"] is False
    assert report["verifier_counts"] == {"n_questions": 2, "n_scored": 1}
    assert any("scored 1/2 non-error questions" in b for b in report["blockers"])


def test_verifier_mode_empty_result_is_not_decision_grade(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(window.activation_window, "build_preflight", lambda _a: _healthy_preflight())
    monkeypatch.setattr(window, "_resolved_eval_concurrency", lambda _roles=None: 4)
    monkeypatch.setattr(window, "_optimized_live_stack_status", lambda: {"ok": True, "warnings": []})

    class _EmptyTower:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def eval_calibration(self, *, suite, split, roles, seed, n, full):  # noqa: ARG002
            return {"dataset_sha256": "abc", "n_questions": 0, "per_role": {}}

    monkeypatch.setattr(window, "EvalTower", _EmptyTower)
    args = window.parse_args(
        [
            "--mode",
            "calibration",
            "--suite",
            "scoring_verifiers",
            "--split",
            "HE-R+",
            "--roles",
            "worker_general",
            "--n",
            "2",
            "--min-eval-concurrency",
            "3",
            "--apply",
            "--confirm-clean-window",
            "--output-dir",
            str(tmp_path),
        ]
    )

    report, rc = window.build_verifier_report(args, output_dir=tmp_path)

    assert rc == 75
    assert report["status"] == "eval_degenerate"
    assert report["decision_grade"] is False
    assert report["verifier_counts"] == {"n_questions": 0, "n_scored": 0}
    assert any("degenerate" in b or "scored 0/2" in b for b in report["blockers"])


def test_verifier_mode_apply_blocks_on_live_stack_contract_drift(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(window.activation_window, "build_preflight", lambda _a: _healthy_preflight())
    monkeypatch.setattr(window, "_resolved_eval_concurrency", lambda _roles=None: 4)
    monkeypatch.setattr(
        window,
        "_optimized_live_stack_status",
        lambda: {"ok": False, "warnings": ["frontdoor missing --spec-type draft-mtp"]},
    )
    monkeypatch.setattr(window, "EvalTower", _ExplodingTower)

    args = window.parse_args(
        [
            "--mode",
            "calibration",
            "--suite",
            "scoring_verifiers",
            "--split",
            "HE-R+",
            "--roles",
            "worker_general,frontdoor",
            "--full",
            "--min-eval-concurrency",
            "3",
            "--apply",
            "--confirm-clean-window",
            "--output-dir",
            str(tmp_path),
        ]
    )
    report, rc = window.build_verifier_report(args, output_dir=tmp_path)

    assert report["status"] == "blocked"
    assert rc != 0
    assert report["result"] is None
    assert report["live_stack_contract"]["warnings"] == ["frontdoor missing --spec-type draft-mtp"]
    assert any("live stack launch contract" in b for b in report["blockers"])
