"""Synthetic native-fold replay; no live ledger or AutoPilot state is touched."""
import importlib.util
import os
from pathlib import Path
import sys

import pytest

VIDYA = Path(os.environ.get("VIDYA_DIR", "/workspace/scripts/vidya"))
if not (VIDYA / "fold.py").is_file():
    pytest.skip("native Vidya checkout unavailable", allow_module_level=True)
sys.path.insert(0, str(VIDYA))
from fold import fold
from frames import make_frame

spec = importlib.util.spec_from_file_location("kvq_reconsider", Path(__file__).parents[2] / "scripts/autopilot/kvq_reconsider.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
AT = "2026-09-25T00:00:00Z"


def frame(kind, assertion):
    return make_frame(frame_type=f"epyc.vidya/frame/{kind}/v1", assertion=assertion,
                      provenance={"about": "synthetic", "method": "test"}, actor="test",
                      authority_scope="test", created_at=AT)


def setup():
    frames = [frame("claim_proposed", {"claim_id": "c", "display_text": "test", "source_id": "s"}),
              frame("evidence_supports_claim", {"claim_id": "c", "evidence_id": "e", "source_id": "s", "grade": {"Q": "Verified", "T": "Anchored"}})]
    receipt = {"schema": "epyc.vidya.kvq_planner_receipt.v1", "frontier": len(frames), "as_of": AT,
               "state_hash": fold(frames, as_of=AT).state_hash(), "exposed_claim_ids": ["c"],
               "declared_reliance_claim_ids": ["c"]}
    return frames, {"type": "planner_coordinator", "trial_id": 1, "vidya_kvq": receipt}


def test_synthetic_correction_and_retraction():
    frames, row = setup()
    correction = frame("correction_recorded", {"claim_ids": ["c"], "correction_text": "synthetic"})
    r = module.review([row], frames + [correction], as_of=AT, fold=fold)["decisions"][0]
    assert r["status"] == "review_required" and r["events"][0]["correction_ids"] == [correction["frame_id"]]
    retraction = frame("retraction", {"retracts": frames[1]["frame_id"]})
    r = module.review([row], frames + [retraction], as_of=AT, fold=fold)["decisions"][0]
    assert r["status"] == "review_required" and r["events"][0]["removed_support"] == [{"evidence_id": "e", "grade": {"Q": "Verified", "T": "Anchored"}}]


def test_no_invented_reliance_or_stale_frontier():
    frames, row = setup()
    correction = frame("correction_recorded", {"claim_ids": ["c"], "correction_text": "synthetic"})
    row["vidya_kvq"]["exposed_claim_ids"] = []
    assert module.review([row], frames + [correction], as_of=AT, fold=fold)["decisions"][0]["status"] == "no_declared_reliance"
    row["vidya_kvq"]["exposed_claim_ids"] = ["c"]
    row["vidya_kvq"]["state_hash"] = "wrong"
    assert module.review([row, {"type": "planner_coordinator"}], frames + [correction], as_of=AT, fold=fold)["decisions"][0]["status"] == "unknown"
    assert module.review([{"type": "planner_coordinator"}], frames, as_of=AT, fold=fold)["decisions"][0]["status"] == "unknown"


def test_cli_on_native_ledger_copy(tmp_path):
    import json
    import subprocess
    from ledger import Ledger

    frames, row = setup()
    correction = frame("correction_recorded", {"claim_ids": ["c"], "correction_text": "synthetic"})
    ledger_path = tmp_path / "copy.jsonl"
    ledger = Ledger(ledger_path)
    for value in frames + [correction]:
        ledger.append(value)
    archive_path = tmp_path / "archive.jsonl"
    archive_path.write_text(json.dumps(row) + "\n")
    before = ledger_path.read_bytes(), archive_path.read_bytes()
    proc = subprocess.run([sys.executable, str(Path(__file__).parents[2] / "scripts/autopilot/kvq_reconsider.py"),
                           "--vidya-dir", str(VIDYA), "--ledger", str(ledger_path),
                           "--archive", str(archive_path), "--as-of", AT],
                          check=True, text=True, capture_output=True)
    report = json.loads(proc.stdout)
    assert report["decisions"][0]["status"] == "review_required"
    import hashlib
    assert report["provenance"]["ledger_snapshot_sha256"] == hashlib.sha256(before[0]).hexdigest()
    assert report["provenance"]["archive_snapshot_sha256"] == hashlib.sha256(before[1]).hexdigest()
    assert report["decisions"][0]["receipt_state_hash"] == row["vidya_kvq"]["state_hash"]
    assert report["decisions"][0]["receipt_as_of"] == AT
    assert all(report["provenance"][k]["on_disk_sha256"] for k in ("fold", "ledger", "report_script"))
    assert report["decisions"][0]["events"][0]["event_frame_id"] == correction["frame_id"]
    assert (ledger_path.read_bytes(), archive_path.read_bytes()) == before
    archive_path.write_text(json.dumps(row))
    proc = subprocess.run([sys.executable, str(Path(__file__).parents[2] / "scripts/autopilot/kvq_reconsider.py"),
                           "--vidya-dir", str(VIDYA), "--ledger", str(ledger_path),
                           "--archive", str(archive_path), "--as-of", AT], capture_output=True)
    assert proc.returncode != 0


def test_review_persists_after_correction_review_and_unrelated_retraction_is_ignored():
    frames, row = setup()
    correction = frame("correction_recorded", {"claim_ids": ["c"], "correction_text": "synthetic"})
    reviewed = frame("correction_reviewed", {"reviewed": correction["frame_id"]})
    report = module.review([row], frames + [correction, reviewed], as_of=AT, fold=fold)
    assert report["decisions"][0]["status"] == "review_required"
    unrelated = frame("retraction", {"retracts": "unknown_frame"})
    report = module.review([row], frames + [unrelated], as_of=AT, fold=fold)
    assert report["decisions"][0]["status"] == "no_change_detected"


def test_empty_declaration_with_missing_boundary_is_unknown():
    frames, row = setup()
    row["vidya_kvq"]["declared_reliance_claim_ids"] = []
    assert module.review([row], frames, as_of=AT, fold=fold)["decisions"][0]["status"] == "no_declared_reliance"
    del row["vidya_kvq"]["frontier"]
    assert module.review([row], frames, as_of=AT, fold=fold)["decisions"][0]["status"] == "unknown"


def test_retraction_of_one_of_two_same_label_paths_and_claim():
    frames, row = setup()
    other = frame("evidence_supports_claim", {"claim_id": "c", "evidence_id": "e", "source_id": "s2", "grade": {"Q": "Judged", "T": "Located"}})
    frames.append(other)
    row["vidya_kvq"].update(frontier=len(frames), state_hash=fold(frames, as_of=AT).state_hash())
    ret = frame("retraction", {"retracts": other["frame_id"]})
    result = module.review([row], frames + [ret], as_of=AT, fold=fold)["decisions"][0]
    assert result["status"] == "review_required"
    assert result["events"][0]["removed_support"] == [{"evidence_id": "e", "grade": {"Q": "Judged", "T": "Located"}}]
    ret_claim = frame("retraction", {"retracts": frames[0]["frame_id"]})
    result = module.review([row], frames + [ret_claim], as_of=AT, fold=fold)["decisions"][0]
    assert result["status"] == "review_required" and result["events"][0]["claim_retracted"]


def test_retraction_of_original_correction_still_prompts_review():
    frames, row = setup()
    corr = frame("correction_recorded", {"claim_ids": ["c"], "correction_text": "synthetic"})
    frames.append(corr)
    row["vidya_kvq"].update(frontier=len(frames), state_hash=fold(frames, as_of=AT).state_hash())
    ret = frame("retraction", {"retracts": corr["frame_id"]})
    result = module.review([row], frames + [ret], as_of=AT, fold=fold)["decisions"][0]
    assert result["status"] == "review_required"
    assert result["events"][0]["removed_correction_ids"] == [corr["frame_id"]]


def test_time_and_excluded_row_are_unknown():
    frames, row = setup()
    result = module.review([row, {"type": "other"}], frames, as_of="2026-09-24T00:00:00Z", fold=fold)
    assert [x["status"] for x in result["decisions"]] == ["unknown", "unknown"]
    row["vidya_kvq"]["as_of"] = "2026-09-25T00:00:00"
    assert module.review([row], frames, as_of=AT, fold=fold)["decisions"][0]["status"] == "unknown"
    with pytest.raises(ValueError):
        module.review([row], frames, as_of="bad", fold=fold)


def test_cli_rejects_duplicate_and_nonfinite_json(tmp_path):
    import subprocess
    from ledger import Ledger

    frames, _ = setup()
    path = tmp_path / "ledger.jsonl"
    ledger = Ledger(path)
    for value in frames:
        ledger.append(value)
    archive = tmp_path / "archive.jsonl"
    cmd = [sys.executable, str(Path(__file__).parents[2] / "scripts/autopilot/kvq_reconsider.py"),
           "--vidya-dir", str(VIDYA), "--ledger", str(path), "--archive", str(archive), "--as-of", AT]
    for contents in ('{"type":"x","type":"y"}\n', '{"type":"x","value":NaN}\n'):
        archive.write_text(contents)
        assert subprocess.run(cmd, capture_output=True).returncode != 0



def test_dependency_source_correction_and_withdrawal():
    frames, row = setup()
    source_claim = frame("claim_proposed", {"claim_id": "source-claim", "display_text": "source", "source_id": "upstream"})
    support = frame("evidence_supports_claim", {"claim_id": "source-claim", "evidence_id": "upstream-e", "source_id": "upstream", "grade": {"Q": "Verified", "T": "Anchored"}})
    dependency = frame("claim_depends_on", {"claim_id": "c", "depends_on_source": "upstream", "depends_on_entry": "entry-upstream", "rationale": "synthetic"})
    frames.extend([source_claim, support, dependency])
    row["vidya_kvq"].update(frontier=len(frames), state_hash=fold(frames, as_of=AT).state_hash())
    correction = frame("correction_recorded", {"claim_ids": ["source-claim"], "correction_text": "synthetic"})
    result = module.review([row], frames + [correction], as_of=AT, fold=fold)["decisions"][0]
    assert result["status"] == "review_required"
    assert result["events"][0]["dependency_changes"] == [{"source_claim_id": "source-claim", "correction_id": correction["frame_id"]}]
    withdrawal = frame("retraction", {"retracts": support["frame_id"]})
    result = module.review([row], frames + [withdrawal], as_of=AT, fold=fold)["decisions"][0]
    assert result["status"] == "review_required" and result["events"][0]["dependency_alerts"] == ["entry-upstream"]
    unrelated = frame("correction_recorded", {"claim_ids": ["unrelated"], "correction_text": "synthetic"})
    assert module.review([row], frames + [unrelated], as_of=AT, fold=fold)["decisions"][0]["status"] == "no_change_detected"


def test_later_visibility_and_original_receipt_scope():
    frames, row = setup()
    future = make_frame(frame_type="epyc.vidya/frame/correction_recorded/v1",
        assertion={"claim_ids": ["c"], "correction_text": "future"},
        provenance={"about": "synthetic", "method": "test"}, actor="test",
        authority_scope="test", created_at="2026-09-27T00:00:00Z")
    early = module.review([row], frames + [future], as_of="2026-09-26T00:00:00Z", fold=fold)["decisions"][0]
    late = module.review([row], frames + [future], as_of="2026-09-28T00:00:00Z", fold=fold)["decisions"][0]
    assert early["status"] == "no_change_detected" and late["status"] == "review_required"
    # The receipt producer folded all frames in its prefix, even future-stamped ones.
    original_frames = frames + [future]
    row["vidya_kvq"].update(frontier=len(original_frames), state_hash=fold(original_frames, as_of=AT).state_hash())
    assert module.review([row], original_frames, as_of=AT, fold=fold)["decisions"][0]["status"] == "no_change_detected"
    missing = dict(future, pubinfo={})
    assert module.review([row], original_frames + [missing], as_of=AT, fold=fold)["decisions"][0]["status"] == "unknown"


def test_dependency_claim_retraction_uses_pre_event_source_identity():
    frames, row = setup()
    claim = frame("claim_proposed", {"claim_id": "parent", "display_text": "parent", "source_id": "upstream"})
    support = frame("evidence_supports_claim", {"claim_id": "parent", "evidence_id": "e-parent", "source_id": "upstream", "grade": {"Q": "Verified", "T": "Anchored"}})
    dependency = frame("claim_depends_on", {"claim_id": "c", "depends_on_source": "upstream", "depends_on_entry": "upstream", "rationale": "test"})
    frames.extend([claim, support, dependency])
    row["vidya_kvq"].update(frontier=len(frames), state_hash=fold(frames, as_of=AT).state_hash())
    ret = frame("retraction", {"retracts": claim["frame_id"]})
    result = module.review([row], frames + [ret], as_of=AT, fold=fold)["decisions"][0]
    assert result["status"] == "review_required"
    assert {"source_claim_id": "parent", "withdrawn_claim_frame_id": claim["frame_id"]} in result["events"][0]["dependency_changes"]
