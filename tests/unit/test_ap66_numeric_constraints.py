import math

from optuna.samplers.nsgaii._constraints_evaluation import _constrained_dominates
from optuna.study import StudyDirection
import importlib.util
import sys
from pathlib import Path

path = Path(__file__).resolve().parents[2] / "scripts/autopilot/species/numeric_swarm.py"
spec = importlib.util.spec_from_file_location("ap66_numeric_swarm", path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
NumericSwarm = module.NumericSwarm


def qualified_report(swarm, *args, **kwargs):
    swarm.report_result(*args, cost_tier_observed_count=1,
                        reliability_observed_count=1, **kwargs)


def test_cold_start_pending_survives_restart_and_flushes_from_measured_vector(tmp_path):
    db = tmp_path / "study.db"
    swarm = NumericSwarm(db_path=db)
    crash = swarm.suggest_trial("monitor")
    swarm.report_infeasible(
        "monitor", crash["trial_number"], outcome_class="preflight", reason="invalid"
    )
    trial = swarm._get_study("monitor").trials[0]
    assert trial.state.name == "RUNNING" and trial.user_attrs["anchor_queued"]
    restarted = NumericSwarm(db_path=db)
    measured = restarted.suggest_trial("monitor")
    qualified_report(restarted, "monitor", measured["trial_number"], (1.2, 25.0, -0.3, 0.9))
    study = restarted._get_study("monitor")
    crash, measured = study.trials
    assert crash.state.name == "COMPLETE"
    assert crash.values == measured.values
    assert crash.user_attrs["objective_source"] == "synthetic_worst_observed"
    assert all(math.isfinite(x) for x in crash.values)
    assert _constrained_dominates(measured, crash, [StudyDirection.MAXIMIZE] * 4)


def test_crash_loses_to_large_miss_and_nonanswers_do_not_enter_population(tmp_path):
    swarm = NumericSwarm(db_path=tmp_path / "study.db")
    miss = swarm.suggest_trial("monitor")
    qualified_report(swarm,
        "monitor",
        miss["trial_number"],
        (1.0, 2.0, -3.0, 4.0),
        normalized_overages=[1e100],
        outcome_class="slo_miss",
    )
    crash = swarm.suggest_trial("monitor")
    swarm.report_infeasible("monitor", crash["trial_number"], outcome_class="runtime", reason="oom")
    absent = swarm.suggest_trial("monitor")
    swarm.mark_failed("monitor", absent["trial_number"], "infra")
    study = swarm._get_study("monitor")
    miss, crash, absent = study.trials
    assert miss.user_attrs["normalized_overages"] == [1e100]
    assert (
        0 < miss.user_attrs["constraint_violation"] < crash.user_attrs["constraint_violation"] == 1
    )
    assert _constrained_dominates(miss, crash, [StudyDirection.MAXIMIZE] * 4)
    assert absent.state.name == "PRUNED"
    assert swarm._cluster_select(study, "monitor") == {}


def test_infeasible_parent_changes_next_native_suggestion_away(tmp_path):
    import optuna

    def next_point(label, infeasible):
        swarm = NumericSwarm(db_path=tmp_path / f"{label}.db", seed=1)
        study = swarm._get_study("monitor")
        study.sampler = optuna.samplers.NSGAIISampler(
            population_size=2, seed=1, mutation_prob=0, constraints_func=module._constraints
        )
        first = swarm.suggest_trial("monitor")
        qualified_report(swarm, "monitor", first["trial_number"], (1.0, 2.0, -1.0, 1.0))
        second = swarm.suggest_trial("monitor")
        if infeasible:
            swarm.report_infeasible(
                "monitor",
                second["trial_number"],
                outcome_class="runtime",
                reason="controlled failure",
            )
        else:
            qualified_report(swarm, "monitor", second["trial_number"], (1.0, 2.0, -1.0, 1.0))
        third = swarm.suggest_trial("monitor")
        return first["params"], second["params"], third["params"]

    first_a, failed_a, next_a = next_point("control", False)
    first_b, failed_b, next_b = next_point("constrained", True)
    assert first_a == first_b and failed_a == failed_b
    key = "monitor.entropy_threshold"
    assert abs(next_b[key] - failed_b[key]) > abs(next_a[key] - failed_a[key])
