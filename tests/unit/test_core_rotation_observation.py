from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
AUTOPILOT = ROOT / "scripts" / "autopilot"
sys.path.insert(0, str(AUTOPILOT))

import core_rotation_observation as ap70  # noqa: E402
import autopilot  # noqa: E402


def test_plan_freezes_positive_k_and_boundary_indices() -> None:
    plan = ap70.core_rotation_plan(
        study_id="study-1", planned_k=2, first_boundary_trial_id=30, period=10
    )
    assert plan["planned_boundaries"] == [
        {"boundary_trial_id": 30, "outgoing_rotation_index": 2, "incoming_rotation_index": 3},
        {"boundary_trial_id": 40, "outgoing_rotation_index": 3, "incoming_rotation_index": 4},
    ]
    with pytest.raises(ValueError, match="positive integer"):
        ap70.core_rotation_plan(study_id="study-1", planned_k=0, first_boundary_trial_id=30, period=10)
    with pytest.raises(ValueError, match="boundary"):
        ap70.core_rotation_plan(study_id="study-1", planned_k=1, first_boundary_trial_id=31, period=10)


def test_summary_k1_has_no_se_and_incomplete_plan_has_no_mean() -> None:
    one = ap70.summarize_pairs(
        "s", 1, [{"boundary_trial_id": 10, "status": "complete", "drop": 0.25}],
        [{"boundary_trial_id": 10}],
    )
    assert one["status"] == "complete"
    assert one["mean_drop"] == 0.25
    assert one["standard_error"] is None

    missing = ap70.summarize_pairs(
        "s", 2, [{"boundary_trial_id": 10, "status": "complete", "drop": 0.25}],
        [{"boundary_trial_id": 10}, {"boundary_trial_id": 20}],
    )
    assert missing["status"] == "incomplete"
    assert missing["mean_drop"] is None
    assert missing["standard_error"] is None
    assert missing["missing_boundary_trial_ids"] == [20]


def test_quality_readout_refuses_partial_and_nonfinite_results() -> None:
    partial = _result(0, 0.8)
    partial.details["quality_denominator"] = 99
    with pytest.raises(ValueError, match="denominator"):
        ap70._quality_result(partial, 0, 100)
    nonfinite = _result(0, float("nan"))
    with pytest.raises(ValueError, match="nonnumeric or nonfinite"):
        ap70._quality_result(nonfinite, 0, 100)


class _Journal:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def ledger_events(self, event_type: str) -> list[dict]:
        return [event for event in self.events if event.get("type") == event_type]

    def append_ledger_event(self, event: dict) -> dict:
        # This models the real durable journal boundary: events exist before a
        # paired evaluator is allowed to run.
        self.events.append(dict(event))
        return dict(event)


def _result(rotation: int, quality: float) -> SimpleNamespace:
    return SimpleNamespace(
        quality=quality,
        quality_measured=True,
        core_id=f"core-{rotation}",
        n_questions=100,
        n_questions_requested=100,
        n_questions_completed=100,
        question_results=[SimpleNamespace(partial=False) for _ in range(100)],
        eval_wall_s=1.25,
        sum_request_elapsed_s=2.5,
        details={
            "core_selection": "tier_stratified",
            "tier_mix_provenance": {"core_rotation_index": rotation},
            "dataset_sha256": f"dataset-{rotation}",
            "partition_counts": {"core": 100},
            "quality_denominator": 100,
            "base_core_questions": 100,
            "question_sidecar_persistence": {"archive_complete": True},
        },
    )


class _Tower:
    def __init__(self, journal: _Journal) -> None:
        self.journal = journal
        self._trial_id_context = 999
        self._trial_path_context = None
        self._question_artifact_dir = None
        self._core_rotation_observation_context = None
        self.calls: list[tuple[int, str]] = []

    def eval_t1_observation(self, *, n, seed, trial_id, core_rotation_index, prompt_root):
        assert any(event.get("type") == ap70.PLAN_EVENT for event in self.journal.events)
        assert any(event.get("type") == ap70.BOUNDARY_PIN_EVENT for event in self.journal.events)
        assert n == 100 and seed == 42 and trial_id == 10
        self.calls.append((core_rotation_index, prompt_root))
        return _result(core_rotation_index, {0: 0.8, 1: 0.6}[core_rotation_index])


def test_real_capture_path_pins_before_two_same_incumbent_draws(tmp_path, monkeypatch) -> None:
    source = tmp_path / "checkpoint-prompts"
    source.mkdir()
    (source / "router.md").write_text("immutable incumbent", encoding="utf-8")
    prompt_digest, _ = ap70._tree_sha256(source)
    infra = {"status": "verified", "recipe": {"digest": "recipe"}}
    incumbent = {
        "trial_id": 3,
        "config_fingerprint": "cfg-3",
        "config_snapshot": {"type": "prompt_mutation", "file": "router.md"},
        "checkpoint_target": str(source.parent),
        "checkpoint_metadata_sha256": "meta-sha",
        "prompt_tree_sha256": prompt_digest,
        "prompt_file_count": 1,
        "feature_flags": {"router_v2": True},
        "infra_fingerprint": infra,
        "infra_digest": ap70.fingerprint_digest(infra),
        "incumbent_trial_infra_digest": ap70.fingerprint_digest(infra),
    }
    monkeypatch.setattr(ap70, "_archive_incumbent", lambda *_args: (incumbent, source))
    monkeypatch.setattr(ap70, "_current_flags", lambda _url: {"router_v2": True})
    monkeypatch.setattr(ap70, "collect_infra_fingerprint", lambda: infra)
    monkeypatch.setattr(ap70, "_pair_identity_unchanged", lambda **_kwargs: None)
    monkeypatch.setattr(ap70, "_override_base", lambda: tmp_path / "request-roots")

    journal = _Journal()
    tower = _Tower(journal)
    response = ap70.capture_at_rotation_boundary(
        journal=journal,
        archive=object(),
        tower=tower,
        checkpoint_dir=tmp_path,
        orchestrator_url="http://unused",
        study_id="study-paired-1",
        planned_k=1,
        boundary_trial_id=10,
        rotation_period_trials=10,
        dispatch_fingerprint=infra,
    )

    assert response["pair"]["status"] == "complete"
    assert response["pair"]["q_outgoing"] == 0.8
    assert response["pair"]["q_incoming"] == 0.6
    assert response["pair"]["drop"] == pytest.approx(0.2)
    assert tower.calls[0][0] == 0 and tower.calls[1][0] == 1
    assert tower.calls[0][1] == tower.calls[1][1]
    mirror = Path(tower.calls[0][1])
    assert mirror.is_dir()
    assert not (mirror / "router.md").stat().st_mode & 0o222
    assert response["summary"]["standard_error"] is None
    assert tower._trial_id_context == 999
    assert (source / "router.md").read_text(encoding="utf-8") == "immutable incumbent"


def test_capture_refuses_before_evaluation_when_incumbent_identity_is_unavailable(
    tmp_path, monkeypatch
) -> None:
    infra = {"status": "verified", "recipe": {"digest": "recipe"}}
    monkeypatch.setattr(ap70, "collect_infra_fingerprint", lambda: infra)
    monkeypatch.setattr(ap70, "_current_flags", lambda _url: {"router_v2": True})
    monkeypatch.setattr(
        ap70,
        "_archive_incumbent",
        lambda *_args: (_ for _ in ()).throw(ValueError("no exact served snapshot")),
    )
    monkeypatch.setattr(ap70, "_override_base", lambda: tmp_path / "request-roots")
    journal = _Journal()
    tower = _Tower(journal)
    result = ap70.capture_at_rotation_boundary(
        journal=journal,
        archive=object(),
        tower=tower,
        checkpoint_dir=tmp_path,
        orchestrator_url="http://unused",
        study_id="refused-study",
        planned_k=1,
        boundary_trial_id=10,
        rotation_period_trials=10,
        dispatch_fingerprint=infra,
    )
    assert result["pair"]["status"] == "incomplete"
    assert result["pair"]["drop"] is None
    assert tower.calls == []


def test_tree_identity_rejects_symlinks(tmp_path) -> None:
    root = tmp_path / "tree"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("not in snapshot", encoding="utf-8")
    (root / "escape").symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        ap70._tree_sha256(root)


def test_autopilot_hook_is_default_off_and_wired_before_action_dispatch(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(autopilot, "EVAL_CORE_ROTATION_TRIALS", 10)
    monkeypatch.setattr(
        autopilot, "_capture_core_rotation_observation", lambda **kwargs: calls.append(kwargs)
    )
    args = {
        "trial_id": 10,
        "tower": object(),
        "archive": object(),
        "journal": object(),
        "infra_fingerprint": {"x": 1},
        "orchestrator_url": "http://unused",
    }
    monkeypatch.delenv("AUTOPILOT_CORE_ROTATION_OBSERVE", raising=False)
    autopilot._capture_core_rotation_observation_if_enabled(**args)
    assert calls == []

    monkeypatch.setenv("AUTOPILOT_CORE_ROTATION_OBSERVE", "1")
    monkeypatch.setenv("AUTOPILOT_CORE_ROTATION_STUDY_ID", "predeclared-study")
    monkeypatch.setenv("AUTOPILOT_CORE_ROTATION_OBSERVE_K", "1")
    autopilot._capture_core_rotation_observation_if_enabled(**args)
    assert len(calls) == 1
    assert calls[0]["study_id"] == "predeclared-study"
    source = (AUTOPILOT / "autopilot.py").read_text(encoding="utf-8")
    hook_line = next(i for i, line in enumerate(source.splitlines()) if "_capture_core_rotation_observation_if_enabled(" in line and line.startswith("                "))
    dispatch_line = next(
        i for i, line in enumerate(source.splitlines())
        if line.strip().startswith("eval_result, species_name = dispatch_action(")
    )
    assert hook_line < dispatch_line
