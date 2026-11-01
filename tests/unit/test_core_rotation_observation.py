from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime, timezone

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
        ap70.core_rotation_plan(
            study_id="study-1", planned_k=0, first_boundary_trial_id=30, period=10
        )
    with pytest.raises(ValueError, match="boundary"):
        ap70.core_rotation_plan(
            study_id="study-1", planned_k=1, first_boundary_trial_id=31, period=10
        )


def test_summary_k1_has_no_se_and_incomplete_plan_has_no_mean() -> None:
    one = ap70.summarize_pairs(
        "s",
        1,
        [{"boundary_trial_id": 10, "status": "complete", "drop": 0.25}],
        [{"boundary_trial_id": 10}],
    )
    assert one["status"] == "complete"
    assert one["mean_drop"] == 0.25
    assert one["standard_error"] is None

    missing = ap70.summarize_pairs(
        "s",
        2,
        [{"boundary_trial_id": 10, "status": "complete", "drop": 0.25}],
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
        event_types = {event["type"] for event in self.journal.ledger_events()}
        assert ap70.PLAN_EVENT in event_types
        assert ap70.BOUNDARY_PIN_EVENT in event_types
        assert n == 100 and seed == 42 and trial_id == 10
        self.calls.append((core_rotation_index, prompt_root))
        return _result(core_rotation_index, {0: 0.8, 1: 0.6}[core_rotation_index])


def _real_incumbent_fixture(
    tmp_path, monkeypatch, *, action=None, flags=None, served_identity=None
):
    from scripts.autopilot.experiment_journal import ExperimentJournal, JournalEntry
    from scripts.autopilot.pareto_archive import ParetoArchive, ParetoEntry
    from scripts.autopilot.species.structural_lab import StructuralLab

    flags = flags or {"router_v2": True}
    config_snapshot = action or {
        "type": "prompt_mutation",
        "file": "router.md",
        "mutation": "replace",
        "description": "fixture prompt update",
    }
    prompts = tmp_path / "live-prompts"
    prompts.mkdir(parents=True)
    (prompts / "router.md").write_text("immutable incumbent", encoding="utf-8")
    checkpoint_root = tmp_path / "checkpoints"
    monkeypatch.setattr("scripts.autopilot.species.structural_lab.CHECKPOINT_DIR", checkpoint_root)
    monkeypatch.setattr("scripts.autopilot.species.structural_lab.PROMPTS_DIR", prompts)
    monkeypatch.setattr("scripts.autopilot.species.structural_lab.CHECKPOINT_FILES", {})
    monkeypatch.setattr(StructuralLab, "_get_memory_count", lambda _self: 0)
    checkpoint = StructuralLab().checkpoint_state(
        trial_id=3,
        feature_flags=flags,
        config_snapshot=config_snapshot,
        mark_production_best=True,
    )
    source = checkpoint / "prompts"
    infra_components = {
        name: f"digest-{name}"
        for name in ("orchestrator", "evaluator", "kernel", "recipe", "models", "host")
    }
    infra = {
        "status": "verified",
        "digest": "infra-same",
        "component_digests": infra_components,
        "kernel_evidence": "live",
    }
    if served_identity is None:
        served_identity = {
            "status": "observed",
            "identity_basis": ["fixture"],
            "worker": {"pid": 9},
            "feature_flags": dict(flags),
        }
        served_identity["identity_sha256"] = hashlib.sha256(
            json.dumps(served_identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    monkeypatch.setattr(ap70, "_current_served_identity", lambda _url: served_identity)
    monkeypatch.setattr(ap70, "_current_flags", lambda _url: dict(flags))
    monkeypatch.setattr(ap70, "collect_infra_fingerprint", lambda: infra)
    monkeypatch.setattr(ap70, "_override_base", lambda: tmp_path / "request-roots")

    journal = ExperimentJournal(journal_dir=tmp_path / "journal", segment_snapshots=False)
    journal.record(
        JournalEntry(
            trial_id=3,
            timestamp=datetime.now(timezone.utc).isoformat(),
            species=str(config_snapshot.get("type", "unknown")),
            action_type=str(config_snapshot.get("type", "unknown")),
            tier=1,
            quality=0.8,
            speed=1.0,
            cost=0.0,
            reliability=1.0,
            pareto_status="frontier",
            config_snapshot=config_snapshot,
            infra_fingerprint=infra,
            eval_details={
                "served_content": {
                    "files": {
                        "orchestration/prompts/router.md": ap70._file_sha256(source / "router.md")
                    }
                },
                "ap70_served_runtime_identity": served_identity,
            },
        )
    )
    archived = ParetoEntry(
        trial_id=3,
        objectives=(0.8, 1.0, -0.0, 1.0),
        config_snapshot=config_snapshot,
        eval_tier=1,
        is_production_best=True,
        config_fingerprint="cfg-3",
    )
    archive = ParetoArchive.from_archive_payload({"all_entries": [archived.to_dict()]})
    return journal, archive, source, checkpoint_root, infra, flags, served_identity


def test_real_capture_path_pins_before_two_same_incumbent_draws(tmp_path, monkeypatch) -> None:
    from scripts.autopilot.experiment_journal import ExperimentJournal, JournalEntry
    from scripts.autopilot.pareto_archive import ParetoArchive, ParetoEntry
    from scripts.autopilot.species.structural_lab import StructuralLab

    flags = {"router_v2": True}
    config_snapshot = {
        "type": "prompt_mutation",
        "file": "router.md",
        "mutation": "replace",
        "description": "fixture prompt update",
    }
    prompts = tmp_path / "live-prompts"
    prompts.mkdir()
    (prompts / "router.md").write_text("immutable incumbent", encoding="utf-8")
    checkpoint_root = tmp_path / "checkpoints"
    monkeypatch.setattr("scripts.autopilot.species.structural_lab.CHECKPOINT_DIR", checkpoint_root)
    monkeypatch.setattr("scripts.autopilot.species.structural_lab.PROMPTS_DIR", prompts)
    monkeypatch.setattr("scripts.autopilot.species.structural_lab.CHECKPOINT_FILES", {})
    monkeypatch.setattr(StructuralLab, "_get_memory_count", lambda _self: 0)
    checkpoint = StructuralLab().checkpoint_state(
        trial_id=3,
        feature_flags=flags,
        config_snapshot=config_snapshot,
        mark_production_best=True,
    )
    source = checkpoint / "prompts"
    infra_components = {
        name: f"digest-{name}"
        for name in ("orchestrator", "evaluator", "kernel", "recipe", "models", "host")
    }
    infra = {
        "status": "verified",
        "digest": "infra-same",
        "component_digests": infra_components,
        "kernel_evidence": "live",
    }
    served_identity = {
        "status": "observed",
        "identity_basis": ["fixture"],
        "worker": {"pid": 9},
        "feature_flags": dict(flags),
    }
    served_identity["identity_sha256"] = hashlib.sha256(
        json.dumps(served_identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    monkeypatch.setattr(ap70, "_current_served_identity", lambda _url: served_identity)
    monkeypatch.setattr(ap70, "_current_flags", lambda _url: {"router_v2": True})
    monkeypatch.setattr(ap70, "collect_infra_fingerprint", lambda: infra)
    monkeypatch.setattr(ap70, "_override_base", lambda: tmp_path / "request-roots")

    journal = ExperimentJournal(journal_dir=tmp_path / "journal", segment_snapshots=False)
    journal.record(
        JournalEntry(
            trial_id=3,
            timestamp=datetime.now(timezone.utc).isoformat(),
            species="prompt_mutation",
            action_type="prompt_mutation",
            tier=1,
            quality=0.8,
            speed=1.0,
            cost=0.0,
            reliability=1.0,
            pareto_status="frontier",
            config_snapshot=config_snapshot,
            infra_fingerprint=infra,
            eval_details={
                "served_content": {
                    "files": {
                        "orchestration/prompts/router.md": ap70._file_sha256(source / "router.md")
                    }
                },
                "ap70_served_runtime_identity": served_identity,
            },
        )
    )
    archived = ParetoEntry(
        trial_id=3,
        objectives=(0.8, 1.0, -0.0, 1.0),
        config_snapshot=config_snapshot,
        eval_tier=1,
        is_production_best=True,
        config_fingerprint="cfg-3",
    )
    archive = ParetoArchive.from_archive_payload({"all_entries": [archived.to_dict()]})
    tower = _Tower(journal)
    response = ap70.capture_at_rotation_boundary(
        journal=journal,
        archive=archive,
        tower=tower,
        checkpoint_dir=checkpoint_root,
        orchestrator_url="http://unused",
        study_id="study-paired-1",
        planned_k=1,
        boundary_trial_id=10,
        rotation_period_trials=10,
        dispatch_fingerprint=infra,
    )

    assert response["pair"]["status"] == "complete", response["pair"]
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
        "_current_served_identity",
        lambda _url: (_ for _ in ()).throw(ValueError("API identity unavailable")),
    )
    monkeypatch.setattr(
        ap70,
        "_archive_incumbent",
        lambda *_args: (_ for _ in ()).throw(AssertionError("must refuse before archive read")),
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


def test_real_archive_path_refuses_nonprompt_incumbent_and_flag_drift(tmp_path, monkeypatch):
    journal, archive, _source, checkpoint_root, infra, flags, identity = _real_incumbent_fixture(
        tmp_path / "nonprompt",
        monkeypatch,
        action={"type": "structural_experiment", "file": "router.md"},
    )
    tower = _Tower(journal)
    result = ap70.capture_at_rotation_boundary(
        journal=journal,
        archive=archive,
        tower=tower,
        checkpoint_dir=checkpoint_root,
        orchestrator_url="http://unused",
        study_id="nonprompt-study",
        planned_k=1,
        boundary_trial_id=10,
        rotation_period_trials=10,
        dispatch_fingerprint=infra,
    )
    assert result["pair"]["status"] == "incomplete"
    assert "not a prompt-only artifact" in result["pair"]["reason"]
    assert tower.calls == []

    # A separate real archive/checkpoint/journal fixture refuses a runtime flag drift.
    journal2, archive2, _source2, checkpoint_root2, infra2, _flags2, _identity2 = (
        _real_incumbent_fixture(tmp_path / "flag-drift", monkeypatch)
    )
    monkeypatch.setattr(ap70, "_current_flags", lambda _url: {"router_v2": False})
    tower2 = _Tower(journal2)
    result2 = ap70.capture_at_rotation_boundary(
        journal=journal2,
        archive=archive2,
        tower=tower2,
        checkpoint_dir=checkpoint_root2,
        orchestrator_url="http://unused",
        study_id="flag-drift-study",
        planned_k=1,
        boundary_trial_id=10,
        rotation_period_trials=10,
        dispatch_fingerprint=infra2,
    )
    assert result2["pair"]["status"] == "incomplete"
    assert "feature flags" in result2["pair"]["reason"]
    assert tower2.calls == []


def test_real_archive_path_refuses_checkpoint_symlink_and_served_sha_drift(tmp_path, monkeypatch):
    journal, archive, source, checkpoint_root, infra, _flags, _identity = _real_incumbent_fixture(
        tmp_path / "symlink", monkeypatch
    )
    checkpoint_link = checkpoint_root / "production_best"
    checkpoint_link.unlink()
    outside = tmp_path / "outside-checkpoint"
    outside.mkdir()
    (outside / "checkpoint_meta.json").write_text("{}", encoding="utf-8")
    (outside / "prompts").mkdir()
    (outside / "prompts" / "router.md").write_text("immutable incumbent", encoding="utf-8")
    checkpoint_link.symlink_to(outside)
    tower = _Tower(journal)
    result = ap70.capture_at_rotation_boundary(
        journal=journal,
        archive=archive,
        tower=tower,
        checkpoint_dir=checkpoint_root,
        orchestrator_url="http://unused",
        study_id="symlink-study",
        planned_k=1,
        boundary_trial_id=10,
        rotation_period_trials=10,
        dispatch_fingerprint=infra,
    )
    assert result["pair"]["status"] == "incomplete"
    assert "outside" in result["pair"]["reason"]
    assert tower.calls == []

    journal2, archive2, source2, checkpoint_root2, infra2, _flags2, _identity2 = (
        _real_incumbent_fixture(tmp_path / "served-sha", monkeypatch)
    )
    (source2 / "router.md").write_text("changed after journal capture", encoding="utf-8")
    tower2 = _Tower(journal2)
    result2 = ap70.capture_at_rotation_boundary(
        journal=journal2,
        archive=archive2,
        tower=tower2,
        checkpoint_dir=checkpoint_root2,
        orchestrator_url="http://unused",
        study_id="served-sha-study",
        planned_k=1,
        boundary_trial_id=10,
        rotation_period_trials=10,
        dispatch_fingerprint=infra2,
    )
    assert result2["pair"]["status"] == "incomplete"
    assert "served-content SHA" in result2["pair"]["reason"]
    assert tower2.calls == []


def test_real_archive_path_refuses_runtime_and_environment_drift_between_draws(
    tmp_path, monkeypatch
):
    journal, archive, _source, checkpoint_root, infra, _flags, identity = _real_incumbent_fixture(
        tmp_path, monkeypatch
    )

    class MutatingTower(_Tower):
        def eval_t1_observation(self, **kwargs):
            result = super().eval_t1_observation(**kwargs)
            changed = {**identity, "worker": {"pid": 10}}
            changed.pop("identity_sha256")
            changed["identity_sha256"] = hashlib.sha256(
                json.dumps(changed, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            monkeypatch.setattr(ap70, "_current_served_identity", lambda _url: changed)
            return result

    tower = MutatingTower(journal)
    result = ap70.capture_at_rotation_boundary(
        journal=journal,
        archive=archive,
        tower=tower,
        checkpoint_dir=checkpoint_root,
        orchestrator_url="http://unused",
        study_id="runtime-drift-study",
        planned_k=1,
        boundary_trial_id=10,
        rotation_period_trials=10,
        dispatch_fingerprint=infra,
    )
    assert result["pair"]["status"] == "incomplete"
    assert result["pair"]["drop"] is None
    assert len(tower.calls) == 1

    journal2, archive2, _source2, checkpoint_root2, infra2, _flags2, _identity2 = (
        _real_incumbent_fixture(tmp_path / "env-drift", monkeypatch)
    )

    class EnvMutatingTower(_Tower):
        def eval_t1_observation(self, **kwargs):
            result = super().eval_t1_observation(**kwargs)
            monkeypatch.setenv("ORCHESTRATOR_AP70_TEST_DRIFT", "changed")
            return result

    monkeypatch.delenv("ORCHESTRATOR_AP70_TEST_DRIFT", raising=False)
    tower2 = EnvMutatingTower(journal2)
    result2 = ap70.capture_at_rotation_boundary(
        journal=journal2,
        archive=archive2,
        tower=tower2,
        checkpoint_dir=checkpoint_root2,
        orchestrator_url="http://unused",
        study_id="env-drift-study",
        planned_k=1,
        boundary_trial_id=10,
        rotation_period_trials=10,
        dispatch_fingerprint=infra2,
    )
    assert result2["pair"]["status"] == "incomplete"
    assert result2["pair"]["drop"] is None
    assert len(tower2.calls) == 1


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
    hook_line = next(
        i
        for i, line in enumerate(source.splitlines())
        if "_capture_core_rotation_observation_if_enabled(" in line
        and line.startswith("                ")
    )
    dispatch_line = next(
        i
        for i, line in enumerate(source.splitlines())
        if line.strip().startswith("eval_result, species_name = dispatch_action(")
    )
    assert hook_line < dispatch_line


def test_write_side_identity_capture_is_default_off_and_preserves_unavailable(monkeypatch):
    calls = []
    monkeypatch.setattr(
        autopilot,
        "_ap70_current_served_identity",
        lambda url: calls.append(url) or {"status": "observed", "identity_sha256": "a" * 64},
    )
    monkeypatch.delenv("AUTOPILOT_CORE_ROTATION_OBSERVE", raising=False)
    assert autopilot._capture_ap70_write_side_identity_if_enabled("http://unused") is None
    assert calls == []

    monkeypatch.setenv("AUTOPILOT_CORE_ROTATION_OBSERVE", "1")
    assert (
        autopilot._capture_ap70_write_side_identity_if_enabled("http://unused")["status"]
        == "observed"
    )
    assert calls == ["http://unused"]

    monkeypatch.setattr(
        autopilot,
        "_ap70_current_served_identity",
        lambda _url: (_ for _ in ()).throw(TimeoutError("worker did not attest")),
    )
    unavailable = autopilot._capture_ap70_write_side_identity_if_enabled("http://unused")
    assert unavailable["status"] == "unavailable"
    assert "worker did not attest" in unavailable["reason"]
