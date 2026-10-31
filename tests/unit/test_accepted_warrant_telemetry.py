"""Accepted warrant telemetry is evidence-bound and cannot decide a gate."""
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'scripts/autopilot'))
actions=importlib.import_module('actions')
journal=importlib.import_module('experiment_journal')


def entry(trial_id, failure='', **extra):
    return journal.JournalEntry(trial_id=trial_id,timestamp='now',species='prompt_forge',
        action_type='prompt_mutation',tier=1,quality=0,speed=0,cost=0,reliability=0,
        pareto_status='dominated',failure_analysis=failure,**extra)


def observe(description, entries):
    result=SimpleNamespace(details={})
    ctx=SimpleNamespace(journal=SimpleNamespace(entries_with_supersessions=lambda:entries))
    actions._record_accepted_warrant_telemetry({'description':description},ctx,result)
    return result.details['accepted_warrant_telemetry']


def test_verified_reference_and_unrelated_unknown():
    verified=observe('Repair Trial #7', [entry(7,'observed failure')])
    assert verified['cites_observed_failure'] is True
    assert verified['verified_failure_trial_ids']==[7]
    assert verified['acceptance_effect']=='none'
    assert verified['acceptance_boundary']=='action_local_retained'
    unknown=observe('change something', [])
    assert unknown['explicit_citation_found'] is False
    assert unknown['cites_observed_failure'] is None


def test_nonfailure_corrupted_and_missing_are_not_positive():
    nonfailure=observe('Trial #8', [entry(8)])
    assert nonfailure['cites_observed_failure'] is False
    corrupt=observe('Trial #9', [entry(9,'failure',bug_corrupted_by='invalid')])
    assert corrupt['cites_observed_failure'] is None and corrupt['status']=='unresolved_ineligible'
    superseded=observe('Trial #11', [entry(11,'failure',keep_revert_decision='excluded')])
    assert superseded['cites_observed_failure'] is None and superseded['status']=='unresolved_ineligible'
    mixed=observe('Trial #9 and Trial #7', [entry(9,'failure',bug_corrupted_by='invalid'), entry(7,'observed')])
    assert mixed['cites_observed_failure'] is True and mixed['verified_failure_trial_ids']==[7]
    missing=observe('Trial #10', [])
    assert missing['cites_observed_failure'] is None and missing['status']=='unresolved'


def test_journal_or_description_failure_is_observe_only():
    result=SimpleNamespace(details={})
    ctx=SimpleNamespace(journal=None)
    actions._record_accepted_warrant_telemetry({'description':'Trial #3'},ctx,result)
    assert result.details['accepted_warrant_telemetry']['status']=='unavailable'
    assert result.details['accepted_warrant_telemetry']['acceptance_effect']=='none'
    assert observe(None,[])['cites_observed_failure'] is None
    truncated=observe('x'*8200,[])
    assert truncated['status']=='scan_truncated' and truncated['scan_limit_chars']==8192
    assert truncated['scanned_chars']==8192 and truncated['scan_truncated'] is True


def test_native_journal_roundtrip_keeps_telemetry_and_final_verdict(tmp_path):
    store=journal.ExperimentJournal(tmp_path,segment_snapshots=False)
    result=SimpleNamespace(details={})
    source=entry(7,'observed failure')
    actions._record_accepted_warrant_telemetry({'description':'Trial #7'},
        SimpleNamespace(journal=SimpleNamespace(entries_with_supersessions=lambda:[source])),result)
    final=entry(12,eval_details={'details':result.details},keep_revert_decision='revert')
    store.record(final)
    restored=journal.ExperimentJournal(tmp_path,segment_snapshots=False).recent(1)[0]
    assert restored.eval_details['details']['accepted_warrant_telemetry']['cites_observed_failure'] is True
    assert restored.keep_revert_decision=='revert' and restored.pareto_status=='dominated'
