"""HIL-6 seam S3: score-free AP-53 projection and journal parity for the sibling loop.

* ``render_for_prompt(..., include_scores=True)`` (the default) is byte-identical to the render
  AutoPilot used before S3: a recorded ledger renders to a pinned SHA-256.
* ``include_scores=False`` (what the sibling's proposer sees, HIL-2 item 7) carries no
  per-suite delta and no gate detail, but keeps the gate name, the repeat count and the diff.
* AutoPilot's own readers (``ExperimentJournal``, ``load_records``, the VB-AP53 re-proposal
  rate fold/backfill) read a loop directory written through the substrate facade.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AUTOPILOT_DIR = ROOT / "scripts" / "autopilot"

from scripts.autopilot import rejected_mutation_ledger as rml  # noqa: E402
from src.harness_loop import substrate  # noqa: E402

#: SHA-256 of the default render below, captured on orchestrator 55465c20 (before S3).
RECORDED_DEFAULT_RENDER_SHA256 = "07f53873bdc11a6383de5d942855825ecb39980e57c0f062a70049745fa1db59"


def _recorded_ledger(journal_dir: Path) -> None:
    """A fixed ledger: gate details with deltas and eval ids, repeats, a long diff."""
    details = ["", "per_suite_regression: gsm8k -0.050 < -0.030",
               "eval_instance_leakage: refs=['gsm8k_00003'] total=1"]
    for i in range(9):
        orig = "".join(f"line {k}\n" for k in range(3 + i % 3))
        mut = orig + f"extra {i % 4}\n" * (1 + (i % 3) * 200)
        rec = rml.build_record(
            target="frontdoor.md", mutation_type="prompt", artifact_kind="prompt",
            rejecting_gate=["safety_gate", "transfer_safety", "per_suite_regression"][i % 3],
            original_content=orig, mutated_content=mut, gate_detail=details[i % 3],
            per_suite_quality={"gsm8k": 0.5 + i / 100, "math": 0.4},
            baseline_per_suite_quality=({"gsm8k": 0.55, "math": 0.42} if i % 4 else None),
            trial_id=(i if i % 5 else None), timestamp=f"2026-10-{1 + i:02d}T12:00:00+00:00",
        )
        assert rml.append_record(journal_dir, rec)
        if i % 4 == 0:
            assert rml.append_record(journal_dir, rec)


def test_default_render_is_byte_identical_to_the_pre_s3_render(tmp_path):
    _recorded_ledger(tmp_path)
    renders = [rml.render_for_prompt(tmp_path, "frontdoor.md", n=n, max_diff_chars=m)
               for n in (1, 3, 10) for m in (rml.DEFAULT_PROMPT_DIFF_CHARS, 40)]
    renders.append(rml.render_for_prompt(tmp_path, "frontdoor.md"))
    assert renders[-1] == rml.render_for_prompt(tmp_path, "frontdoor.md", include_scores=True)
    digest = hashlib.sha256(json.dumps(renders).encode()).hexdigest()
    assert digest == RECORDED_DEFAULT_RENDER_SHA256


def test_score_free_render_drops_deltas_and_gate_detail_only(tmp_path):
    _recorded_ledger(tmp_path)
    scored = rml.render_for_prompt(tmp_path, "frontdoor.md", n=10)
    free = rml.render_for_prompt(tmp_path, "frontdoor.md", n=10, include_scores=False)
    assert "per-suite delta" in scored and "gsm8k" in scored
    for leaked in ("per-suite delta", "gsm8k", "-0.050", "+0.", "eval_instance_leakage"):
        assert leaked not in free, leaked
    # Gate names, repeat counts and diffs survive; only the scored clauses differ.
    assert re.findall(r"rejected by `[^`]+`", free) == re.findall(r"rejected by `[^`]+`", scored)
    assert free.count("this exact diff was rejected") == scored.count("this exact diff was rejected")
    assert re.findall(r"```diff\n.*?```", free, re.S) == re.findall(r"```diff\n.*?```", scored, re.S)
    stripped = re.sub(r"( \([^\n]*?\))?; per-suite delta vs baseline: [^;\n]*", "", scored)
    assert stripped == free


def test_substrate_proposer_render_is_score_free_over_the_loop_dir(tmp_path):
    loop = tmp_path / "loop"
    substrate.open_loop_journal(loop)
    substrate.append_rejection(
        loop, target="orchestration/prompts/compaction.md", mutation_type="prompt",
        artifact_kind="prompt", rejecting_gate="eval_instance_leakage",
        original_content="a\n", mutated_content="a\nb\n",
        gate_detail="eval_instance_leakage: refs=['gsm8k_00003']",
        per_suite_quality={"gsm8k": 0.4}, baseline_per_suite_quality={"gsm8k": 0.6},
    )
    text = substrate.render_rejections_for_proposer(loop, "orchestration/prompts/compaction.md")
    assert "rejected by `eval_instance_leakage`" in text and "+b" in text
    assert "gsm8k" not in text and "delta" not in text
    import pytest

    with pytest.raises(substrate.LiveSubstrateRefused):
        substrate.render_rejections_for_proposer(substrate._autopilot_default_journal_dir(), "x")


def test_autopilot_readers_read_a_loop_directory(tmp_path):
    import reproposal_rate as rr
    from experiment_journal import ExperimentJournal

    loop = tmp_path / "loop"
    journal = substrate.open_loop_journal(loop)
    journal.append_ledger_event({"type": "hil_cycle_commit", "cycle": 1})
    for i in range(3):
        substrate.append_rejection(
            loop, target="t.md", mutation_type="prompt", artifact_kind="prompt",
            rejecting_gate="mutation_risk_gate", original_content="a\n",
            mutated_content="a\nb\n" if i < 2 else "a\nc\n", trial_id=i,
        )
    # The loop wrote typed ledger events and AP-53 rows only, never JournalEntry trial rows.
    reader = ExperimentJournal(journal_dir=loop, segment_snapshots=False)
    assert reader.all_entries() == []
    assert [e["cycle"] for e in reader.ledger_events("hil_cycle_commit")] == [1]
    records = rml.load_records(loop)
    assert len(records) == 3
    counts = rr.ledger_window_counts(records, 0, 100)
    assert counts == {"records": 3, "diff_repeats": 1,
                      "by_rejecting_gate": {"mutation_risk_gate": 3}}
    ident = rr.source_identity(loop)
    assert ident["rejected_mutation_ledger"]["bytes"] > 0
    report = rr.backfill(loop, tmp_path / "retro.jsonl")
    assert report["retrospective_lines"] == 0 and report["belief_rows"] == 0
    # AutoPilot's prompt render reads the same rows (scores included by default).
    assert "this exact diff was rejected 2x" in rml.render_for_prompt(loop, "t.md")
