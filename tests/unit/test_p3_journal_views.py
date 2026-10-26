"""Read-view ownership and the separate legacy quality ceiling."""
from __future__ import annotations

import copy
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/autopilot'))
import autopilot
import safety_gate as sg
from experiment_journal import ExperimentJournal, JournalEntry
from src.autopilot_core.action_identity import row_config_identity
from src.autopilot_core.journal_reconstruction import reconstruct_archive_from_journal_rows
from src.autopilot_core.live_reproductions import live_reproductions
from src.autopilot_core.tier_specs import LEGACY_OBJECTIVE_POLICY


def entry(tid, quality=1.0):
    return JournalEntry(trial_id=tid, timestamp=f'2026-06-13T00:00:{tid:02d}+00:00',
                        species='unit', action_type='structural_experiment', tier=1,
                        quality=quality, speed=10.0, cost=0.5, reliability=1.0,
                        pareto_status='frontier',
                        config_snapshot={'type': 'structural_experiment', 'flags': {'x': 1}},
                        eval_details={'nested': {'value': 1}, 'frontier_admission': 'representative'})


@pytest.fixture
def journal(tmp_path, monkeypatch):
    monkeypatch.setattr(ExperimentJournal, '_emit_trace_event', lambda *args: None)
    j = ExperimentJournal(tmp_path, segment_snapshots=False)
    j.record(entry(1, 2.0))
    yield j
    sg.configure_promotion_guard_archive(None)


def test_ceiling_uses_injected_rows_and_all_eras(journal, monkeypatch):
    import scripts.autopilot.experiment_journal as journal_module
    monkeypatch.setattr(journal_module, 'ExperimentJournal', lambda: pytest.fail('fresh load'))
    # Live provider deliberately has no old-era evidence; the ceiling must retain it.
    sg.configure_promotion_guard_archive(lambda **kw: sg.PromotionGuardView(
        archive=None, objective_policy=LEGACY_OBJECTIVE_POLICY, scope='live'), journal=journal)
    archive = sg._pareto_archive_for_safety_guard()
    expected = reconstruct_archive_from_journal_rows(journal.rows(), None)
    assert archive is not None and expected is not None
    assert max(e.objectives[0] for e in archive.frontier(1)) == 2.0
    assert reconstruct_archive_from_journal_rows(journal.rows(), None,
                                               exclude_before_ts=2_000_000_000) is None
    sg.configure_promotion_guard_archive(None)
    assert sg._QUALITY_CEILING_JOURNAL is None
    assert sg._PROMOTION_GUARD_PROVIDER is None


@pytest.mark.parametrize('fails', [False, True])
def test_loop_releases_journal_on_return_and_error(journal, monkeypatch, fails):
    class Breadcrumb:
        def __getattr__(self, name):
            return lambda *a, **kw: None
    monkeypatch.setattr(autopilot, 'ExitBreadcrumb', Breadcrumb)
    def inner(*args):
        sg.configure_promotion_guard_archive(lambda **kw: None, journal=journal)
        if fails:
            raise RuntimeError('controlled')
    monkeypatch.setattr(autopilot, '_run_loop_inner', inner)
    if fails:
        with pytest.raises(RuntimeError, match='controlled'):
            autopilot.run_loop(use_tui=False)
    else:
        autopilot.run_loop(use_tui=False)
    assert sg._QUALITY_CEILING_JOURNAL is None
    assert sg._PROMOTION_GUARD_PROVIDER is None


def test_consumers_preserve_nested_views_and_invalidate(journal):
    fields = {'eval_details': {'nested': {'value': 2}, 'frontier_admission': 'representative'},
              'config_snapshot': {'type': 'structural_experiment', 'flags': {'x': 2}}}
    original = copy.deepcopy(fields)
    journal.rows(folded=True)  # warm both generations
    journal.append_supersession_event(target_trial_ids=[1], fields=fields,
        reason='unit', policy_version='unit', actor='unit')
    rows = journal.rows() + journal.supersession_events()
    before = copy.deepcopy(rows)
    folded_before = copy.deepcopy([asdict(e) for e in journal.folded_entries()])
    assert journal.rows(folded=True)[0]['eval_details']['nested']['value'] == 2
    assert journal.rows()[0]['eval_details']['nested']['value'] == 1
    for _ in range(2):
        assert reconstruct_archive_from_journal_rows(rows, None) is not None
        assert live_reproductions(rows, tier=1,
                           identity=row_config_identity(journal.rows(folded=True)[0]) or '',
                           objective_policy=LEGACY_OBJECTIVE_POLICY, exclude_before_ts=None)
    assert rows == before
    assert [asdict(e) for e in journal.folded_entries()] == folded_before
    assert fields == original
    journal.record(entry(2))
    assert len(journal.rows()) == 2
    # Reload means a fresh constructor, never calling private _load_existing again.
    reloaded = ExperimentJournal(journal.journal_dir, segment_snapshots=False)
    assert reloaded.rows(folded=True) == journal.rows(folded=True)
