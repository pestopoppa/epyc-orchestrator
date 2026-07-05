from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = PROJECT_ROOT / "scripts" / "benchmark" / "eval_batch_serving_evaltower_window.py"

spec = importlib.util.spec_from_file_location("eval_batch_serving_evaltower_window", MODULE_PATH)
assert spec is not None and spec.loader is not None
window = importlib.util.module_from_spec(spec)
sys.modules["eval_batch_serving_evaltower_window"] = window
spec.loader.exec_module(window)


class FakeResult:
    def __init__(
        self,
        *,
        tier: int,
        quality: float,
        speed: float,
        reliability: float,
        wall_s: float,
        n_questions: int = 5,
    ) -> None:
        self.tier = tier
        self.quality = quality
        self.speed = speed
        self.cost = 0.25
        self.reliability = reliability
        self.n_questions = n_questions
        self.per_suite_quality = {"general": quality}
        self.per_suite_counts = {"general": n_questions}
        self.routing_distribution = {"frontdoor": 1.0}
        self.question_results = []
        self.core_id = "fake-core"
        self.details = {
            "eval_wall_s": wall_s,
            "speed_metric_mode": "aggregate_batch_tps",
            "aggregate_tps": speed,
            "per_suite_counts": {"general": n_questions},
        }
        self.speed_metric_mode = "aggregate_batch_tps"
        self.median_request_speed = 0.0
        self.aggregate_speed = speed
        self.eval_concurrency = 4
        self.eval_wall_s = wall_s
        self.mean_tools_used = 0.0
        self.tool_use_rate = 0.0
        self.total_tool_calls = 0


class FakeTower:
    results: list[FakeResult] = []

    def __init__(self, *, url: str, timeout: float) -> None:
        self.url = url
        self.timeout = timeout

    def eval_t1(self, *, n: int, seed: int) -> FakeResult:
        return self.results.pop(0)

    def eval_t2(self, *, n: int, seed: int) -> FakeResult:
        return self.results.pop(0)

    def eval_t3(self, *, n: int, seed: int) -> FakeResult:
        return self.results.pop(0)


def _healthy_preflight(*, autopilot_active: bool = False) -> dict:
    return {
        "api_health": {"ok": True},
        "eval_batch_frontdoor_health": {"ok": False},
        "autopilot_active": autopilot_active,
        "config_attest": {"all_sampled_workers_enabled": False},
        "activation_commands": [],
    }


def _step(name: str, ok: bool = True):
    return window.activation_window.StepResult(
        name=name,
        command=name,
        returncode=0 if ok else 2,
        elapsed_s=0.01,
        stdout_tail="",
        stderr_tail="",
    )


def test_plan_only_writes_no_eval_or_activation(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        window.activation_window,
        "build_preflight",
        lambda _args: _healthy_preflight(),
    )
    monkeypatch.setattr(window, "run_eval_arm", lambda *_args, **_kwargs: None)

    args = window.parse_args(["--output-dir", str(tmp_path), "--tier", "3", "--n", "12"])
    report, rc = window.build_report(args, output_dir=tmp_path)

    assert rc == 0
    assert report["status"] == "plan_only"
    assert report["applied"] is False
    assert report["current_arm"] is None
    assert report["planned_current_arm"]["tier"] == 3
    assert report["planned_eval_batch_arm"]["n"] == 12


def test_apply_refuses_active_autopilot_by_default(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        window.activation_window,
        "build_preflight",
        lambda _args: _healthy_preflight(autopilot_active=True),
    )

    args = window.parse_args(["--apply", "--confirm-clean-window", "--output-dir", str(tmp_path)])
    report, rc = window.build_report(args, output_dir=tmp_path)

    assert rc == 75
    assert report["status"] == "blocked"
    assert "AutoPilot appears active" in report["blockers"][0]
    assert report["decision_grade"] is False


def test_successful_apply_runs_both_arms_and_rolls_back(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        window.activation_window,
        "build_preflight",
        lambda _args: _healthy_preflight(),
    )
    monkeypatch.setattr(
        window.activation_window,
        "execute_activation",
        lambda _args, *, output_dir: ([_step("start"), _step("reload"), _step("smoke")], []),
    )
    monkeypatch.setattr(
        window.activation_window,
        "_load_probe_summary",
        lambda _output_dir: {"status": "smoke_passed", "decision_grade": True},
    )
    monkeypatch.setattr(
        window.activation_window,
        "execute_rollback",
        lambda _args: [_step("rollback_reload"), _step("rollback_stop")],
    )

    calls: list[str] = []

    def fake_run_eval_arm(name: str, _args):
        calls.append(name)
        if name == "current":
            return {
                "name": name,
                "ok": True,
                "error": None,
                "metrics": {"quality": 2.0, "speed": 10.0, "reliability": 1.0, "wall_s": 100.0},
            }
        return {
            "name": name,
            "ok": True,
            "error": None,
            "metrics": {"quality": 2.1, "speed": 12.0, "reliability": 1.0, "wall_s": 25.0},
        }

    monkeypatch.setattr(window, "run_eval_arm", fake_run_eval_arm)

    args = window.parse_args(["--apply", "--confirm-clean-window", "--output-dir", str(tmp_path)])
    report, rc = window.build_report(args, output_dir=tmp_path)

    assert rc == 0
    assert report["status"] == "comparison_complete_rolled_back"
    assert calls == ["current", "eval_batch"]
    assert report["decision_grade"] is True
    assert report["comparison"]["wall_speedup_current_over_eval_batch"] == 4.0
    assert [step["name"] for step in report["rollback_steps"]] == [
        "rollback_reload",
        "rollback_stop",
    ]


def test_skip_current_arm_runs_batch_only_and_is_not_decision_grade(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        window.activation_window,
        "build_preflight",
        lambda _args: _healthy_preflight(),
    )
    monkeypatch.setattr(
        window.activation_window,
        "execute_activation",
        lambda _args, *, output_dir: ([_step("start"), _step("reload"), _step("smoke")], []),
    )
    monkeypatch.setattr(
        window.activation_window,
        "_load_probe_summary",
        lambda _output_dir: {"status": "smoke_passed", "decision_grade": True},
    )
    monkeypatch.setattr(
        window.activation_window,
        "execute_rollback",
        lambda _args: [_step("rollback_reload"), _step("rollback_stop")],
    )

    calls: list[str] = []

    def fake_run_eval_arm(name: str, _args):
        calls.append(name)
        return {
            "name": name,
            "ok": True,
            "error": None,
            "metrics": {"quality": 2.1, "speed": 12.0, "reliability": 1.0, "wall_s": 25.0},
        }

    monkeypatch.setattr(window, "run_eval_arm", fake_run_eval_arm)

    args = window.parse_args(
        ["--apply", "--confirm-clean-window", "--skip-current-arm", "--output-dir", str(tmp_path)]
    )
    report, rc = window.build_report(args, output_dir=tmp_path)

    assert rc == 0
    assert report["status"] == "eval_batch_arm_complete_rolled_back"
    assert calls == ["eval_batch"]
    assert report["comparison"] is None
    assert report["decision_grade"] is False


def test_current_arm_failure_blocks_activation(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        window.activation_window,
        "build_preflight",
        lambda _args: _healthy_preflight(),
    )
    activated = False

    def fake_activate(_args, *, output_dir):
        nonlocal activated
        activated = True
        return ([_step("start")], [])

    monkeypatch.setattr(window.activation_window, "execute_activation", fake_activate)
    monkeypatch.setattr(
        window,
        "run_eval_arm",
        lambda _name, _args: {
            "name": "current",
            "ok": False,
            "error": "boom",
            "metrics": {"wall_s": 0.01},
        },
    )

    args = window.parse_args(["--apply", "--confirm-clean-window", "--output-dir", str(tmp_path)])
    report, rc = window.build_report(args, output_dir=tmp_path)

    assert rc == 75
    assert report["status"] == "current_eval_failed"
    assert activated is False
    assert report["decision_grade"] is False


def test_eval_result_metrics_includes_suite_and_batch_fields(monkeypatch) -> None:
    monkeypatch.setattr(window, "EvalTower", FakeTower)
    FakeTower.results = [
        FakeResult(
            tier=1,
            quality=2.2,
            speed=40.0,
            reliability=0.98,
            wall_s=7.0,
        )
    ]
    args = window.parse_args(["--tier", "1", "--n", "5", "--seed", "7"])

    arm = window.run_eval_arm("current", args)

    assert arm["ok"] is True
    assert arm["metrics"]["tier"] == 1
    assert arm["metrics"]["quality"] == 2.2
    assert arm["metrics"]["aggregate_tps"] == 40.0
    assert arm["metrics"]["eval_wall_s"] == 7.0
    assert arm["metrics"]["per_suite_counts"] == {"general": 5}
