"""Tests for the constrained-creativity planner upgrade (2026-05-23):

- `_build_exploration_block()` switches between lean and rich fragments
  based on stagnation signals.
- `tail_action_candidates()` semantics (now consumed as seeds, not candidates).
- `unfalsified_hypotheses()` filter logic.
- New `JournalEntry.falsifier` / `rubric_scores` fields round-trip through
  JSONL.
"""

from __future__ import annotations

import importlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AUTOPILOT_DIR = ROOT / "scripts" / "autopilot"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(AUTOPILOT_DIR))

experiment_journal = importlib.import_module("experiment_journal")
ExperimentJournal = experiment_journal.ExperimentJournal
JournalEntry = experiment_journal.JournalEntry

# Match the dotted-path form used by test_autopilot_numeric_apply.py so that
# both tests share the same sys.modules entry — otherwise monkeypatches on
# `scripts.autopilot.autopilot` and reads through bare `autopilot` would
# diverge when the suites run in the same session.
autopilot = importlib.import_module("scripts.autopilot.autopilot")
pareto_archive = importlib.import_module("pareto_archive")
ParetoArchive = pareto_archive.ParetoArchive
ParetoEntry = pareto_archive.ParetoEntry


KNOWN_ACTIONS = [
    "seed_batch", "numeric_trial", "prompt_mutation",
    "gepa_optimize", "code_mutation", "structural_experiment",
    "structural_prune", "slot_compact", "train_routing_models",
    "distill_skillbank", "reset_memories", "deep_eval",
    "rollback", "distill_knowledge",
]


# ── helpers ─────────────────────────────────────────────────────


def _entry(trial_id: int, action_type: str, **kw) -> JournalEntry:
    return JournalEntry(
        trial_id=trial_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        species=kw.pop("species", "auto"),
        action_type=action_type,
        tier=1,
        quality=kw.pop("quality", 0.5),
        speed=10.0,
        cost=0.1,
        reliability=1.0,
        pareto_status=kw.pop("pareto_status", "dominated"),
        **kw,
    )


def _populate(journal: ExperimentJournal, entries: list[JournalEntry]) -> None:
    """Append entries to the journal's in-memory list and persist via record()."""
    for e in entries:
        journal.record(e)


def _fresh_journal(tmp_path: Path) -> ExperimentJournal:
    return ExperimentJournal(journal_dir=tmp_path)


# ── JournalEntry round-trip ─────────────────────────────────────


def test_journal_entry_persists_new_fields(tmp_path: Path) -> None:
    j = _fresh_journal(tmp_path)
    j.record(_entry(
        0, "seed_batch",
        hypothesis="seed math broadly",
        falsifier="no quality gain after 20 questions",
        rubric_scores={"info_gain": 4, "coherence": 5, "usefulness": 3},
        stagnation_signal="hv_slope_10=+0.00000 < eps=0.00100",
    ))
    # Reload from disk and confirm fields survived.
    reloaded = ExperimentJournal(journal_dir=tmp_path)
    e = reloaded.all_entries()[-1]
    assert e.falsifier == "no quality gain after 20 questions"
    assert e.rubric_scores == {"info_gain": 4, "coherence": 5, "usefulness": 3}
    assert e.stagnation_signal == "hv_slope_10=+0.00000 < eps=0.00100"


def test_action_diversity_by_gate_buckets_correctly(tmp_path: Path) -> None:
    j = _fresh_journal(tmp_path)
    # 4 rich (stagnation fired) trials with 3 distinct action_types.
    for i, t in enumerate(["seed_batch", "numeric_trial", "prompt_mutation", "numeric_trial"]):
        j.record(_entry(i, t, stagnation_signal="hv_slope_10 < eps"))
    # 6 lean trials, all seed_batch (the exploit phase).
    for i, t in enumerate(["seed_batch"] * 6):
        j.record(_entry(i + 10, t, stagnation_signal=""))

    stats = j.action_diversity_by_gate(window=20)
    assert stats["window"] == 20
    assert stats["rich"]["count"] == 4
    assert stats["rich"]["distinct_action_types"] == 3
    assert stats["rich"]["entropy_bits"] > 1.0   # 3 types across 4 trials
    assert stats["lean"]["count"] == 6
    assert stats["lean"]["distinct_action_types"] == 1
    assert stats["lean"]["entropy_bits"] == 0.0


# ── hv_slope noise-floor auto-calibration ───────────────────────


def _empty_archive(tmp_path: Path) -> "ParetoArchive":
    """Build a ParetoArchive that doesn't load production state."""
    return ParetoArchive(state_path=tmp_path / "nonexistent_archive.json")


def test_hv_slope_noise_floor_falls_back_to_default_with_short_history(tmp_path: Path) -> None:
    a = _empty_archive(tmp_path)
    # No history at all → must return the supplied default unchanged.
    floor = a.hv_slope_noise_floor(floor_default=1e-3)
    assert floor == 1e-3


def test_hv_slope_noise_floor_clips_below_default_with_calm_history(tmp_path: Path) -> None:
    a = _empty_archive(tmp_path)
    # Push a long, very-low-noise hypervolume trajectory (linear-ish).
    # Slopes should be near-zero variance → calibrated floor below default.
    for tid in range(50):
        a._hv_hist().append((tid, 10.0 + 0.001 * tid))
    floor = a.hv_slope_noise_floor(floor_default=1e-3, floor_min=1e-6)
    assert floor <= 1e-3
    assert floor >= 1e-6


def test_hv_slope_noise_floor_never_exceeds_default_even_with_noisy_history(tmp_path: Path) -> None:
    a = _empty_archive(tmp_path)
    import random
    rng = random.Random(0)
    # Inject a high-noise random walk: slope variance will be large, but the
    # function must clip up to (i.e. never exceed) floor_default.
    h = 10.0
    for tid in range(80):
        h += rng.uniform(-2.0, 2.0)
        a._hv_hist().append((tid, h))
    floor = a.hv_slope_noise_floor(floor_default=1e-3, floor_min=1e-6)
    assert floor <= 1e-3
    assert floor >= 1e-6


def test_journal_entry_defaults_when_legacy_jsonl_missing_fields(tmp_path: Path) -> None:
    # Simulate a legacy file that predates the new fields.
    path = tmp_path / "autopilot_journal.jsonl"
    legacy = {
        "trial_id": 7, "timestamp": "2025-01-01T00:00:00+00:00",
        "species": "auto", "action_type": "seed_batch",
        "tier": 1, "quality": 0.5, "speed": 10.0, "cost": 0.1,
        "reliability": 1.0, "pareto_status": "dominated",
    }
    path.write_text(json.dumps(legacy) + "\n")
    j = ExperimentJournal(journal_dir=tmp_path)
    e = j.all_entries()[-1]
    assert e.falsifier == ""
    assert e.rubric_scores == {}


# ── unfalsified_hypotheses filter ───────────────────────────────


def test_unfalsified_hypotheses_returns_only_entries_with_both_fields(tmp_path: Path) -> None:
    j = _fresh_journal(tmp_path)
    _populate(j, [
        _entry(0, "seed_batch", hypothesis="h0", falsifier=""),                    # no falsifier
        _entry(1, "numeric_trial", hypothesis="", falsifier="some falsifier"),     # no hypothesis
        _entry(2, "prompt_mutation", hypothesis="h2", falsifier="f2"),             # keep
        _entry(3, "code_mutation", hypothesis="h3", falsifier="f3"),               # keep
    ])
    out = j.unfalsified_hypotheses(n=5)
    assert [t[0] for t in out] == [2, 3]
    assert out[0] == (2, "h2", "f2")


def test_unfalsified_hypotheses_excludes_bug_corrupted(tmp_path: Path) -> None:
    j = _fresh_journal(tmp_path)
    _populate(j, [
        _entry(0, "seed_batch", hypothesis="h0", falsifier="f0",
               bug_corrupted_by="deadbeef"),
        _entry(1, "numeric_trial", hypothesis="h1", falsifier="f1"),
    ])
    out = j.unfalsified_hypotheses(n=5)
    assert [t[0] for t in out] == [1]


# ── _build_exploration_block ────────────────────────────────────


class _FakeArchive:
    """Minimal stand-in exposing .geometry() with the keys autopilot reads."""

    def __init__(self, hv_slope_10: float | None) -> None:
        self._slope = hv_slope_10

    def geometry(self) -> dict[str, float]:
        return {"hv_slope_10": self._slope} if self._slope is not None else {}


def _well_populated_journal(tmp_path: Path, varied_types: list[str]) -> ExperimentJournal:
    """Build a journal with ≥5 trustworthy trials and diverse action_types so the
    low-signal flag and the streak signal both stay quiet."""
    j = _fresh_journal(tmp_path)
    for i, t in enumerate(varied_types):
        j.record(_entry(i, t, hypothesis=f"h{i}"))
    return j


def test_lean_fragment_when_no_stagnation_signal(tmp_path: Path) -> None:
    j = _well_populated_journal(
        tmp_path,
        ["seed_batch", "numeric_trial", "prompt_mutation",
         "code_mutation", "structural_experiment", "deep_eval"],
    )
    archive = _FakeArchive(hv_slope_10=0.1)  # healthy growth
    block, signal = autopilot._build_exploration_block(
        journal=j, archive=archive, known_actions=KNOWN_ACTIONS,
    )
    assert "STAGNATING" not in block
    assert "3–5 alternatives" in block
    assert signal == "none (lean prompt)"


def test_rich_fragment_when_hv_slope_below_eps(tmp_path: Path) -> None:
    j = _well_populated_journal(
        tmp_path,
        ["seed_batch", "numeric_trial", "prompt_mutation",
         "code_mutation", "structural_experiment", "deep_eval"],
    )
    archive = _FakeArchive(hv_slope_10=0.0)  # frontier flat
    block, signal = autopilot._build_exploration_block(
        journal=j, archive=archive, known_actions=KNOWN_ACTIONS,
    )
    assert "STAGNATING" in block
    assert "candidate actions" in block
    assert "BT tiebreak" not in block
    assert "hv_slope_10" in signal


def test_rich_fragment_when_action_type_streak(tmp_path: Path) -> None:
    j = _well_populated_journal(
        tmp_path,
        # 5 varied seed entries to clear low_signal, then a 3-trial streak
        ["seed_batch", "numeric_trial", "prompt_mutation",
         "code_mutation", "structural_experiment",
         "deep_eval", "deep_eval", "deep_eval"],
    )
    archive = _FakeArchive(hv_slope_10=0.5)  # healthy
    block, signal = autopilot._build_exploration_block(
        journal=j, archive=archive, known_actions=KNOWN_ACTIONS,
    )
    assert "STAGNATING" in block
    assert "deep_eval" in signal


def test_rich_fragment_when_trustworthy_low_signal(tmp_path: Path) -> None:
    j = _fresh_journal(tmp_path)
    # Only 2 trustworthy entries → triggers low_signal flag.
    j.record(_entry(0, "seed_batch"))
    j.record(_entry(1, "numeric_trial"))
    archive = _FakeArchive(hv_slope_10=0.5)
    block, signal = autopilot._build_exploration_block(
        journal=j, archive=archive, known_actions=KNOWN_ACTIONS,
    )
    assert "STAGNATING" in block
    assert "trustworthy=" in signal


def test_rich_fragment_lists_tail_seeds_and_unfalsified(tmp_path: Path) -> None:
    j = _fresh_journal(tmp_path)
    # Build a journal where seed_batch dominates (so most others are tail) AND
    # at least one entry has both hypothesis+falsifier.
    for i in range(8):
        j.record(_entry(i, "seed_batch"))
    j.record(_entry(
        8, "prompt_mutation",
        hypothesis="trim verbose section", falsifier="quality drops >2pp",
    ))
    # Force stagnation via flat slope.
    archive = _FakeArchive(hv_slope_10=0.0)
    block, signal = autopilot._build_exploration_block(
        journal=j, archive=archive, known_actions=KNOWN_ACTIONS,
    )
    # Tail seeds should include some under-used action_type names.
    assert any(a in block for a in KNOWN_ACTIONS if a != "seed_batch")
    # Unfalsified hypothesis surfaces with its falsifier.
    assert "trim verbose section" in block
    assert "quality drops >2pp" in block


def test_action_availability_filters_non_viable_tail_actions(tmp_path: Path) -> None:
    j = _fresh_journal(tmp_path)
    j.record(_entry(
        185,
        "prompt_mutation",
        failure_analysis="prompt edit regressed quality badly",
        hypothesis="edit frontdoor",
    ))
    availability, viable = autopilot._build_action_availability(
        journal=j,
        known_actions=KNOWN_ACTIONS,
        memory_count=100,
        converged=False,
        slot_memory_text="  healthy queried ports with empty KV cache: frontdoor:8070",
        blacklist=[{"pattern": {"type": "distill_skillbank"}, "reason": "blacklisted"}],
    )
    assert "`slot_compact`" in availability
    assert "not evidence that the eval instrument or host is contaminated" in availability
    assert "`train_routing_models`" in availability
    assert "`reset_memories`" in availability
    assert "`distill_skillbank`" in availability
    assert "`prompt_mutation`" in availability
    assert "slot_compact" not in viable
    assert "train_routing_models" not in viable
    assert "reset_memories" not in viable
    assert "rollback" not in viable
    assert "distill_knowledge" not in viable
    assert "distill_skillbank" not in viable


def test_action_availability_blocks_seed_batch_when_fallbacks_exhausted(
    tmp_path: Path,
) -> None:
    j = _fresh_journal(tmp_path)
    availability, viable = autopilot._build_action_availability(
        journal=j,
        known_actions=KNOWN_ACTIONS,
        memory_count=10_000,
        converged=True,
        slot_memory_text="  healthy queried ports with empty KV cache: frontdoor:8070",
        blacklist=[
            {
                "pattern": {"type": "seed_batch", "n_questions": n_questions},
                "reason": f"blocked {n_questions}",
            }
            for n_questions in autopilot.FALLBACK_SEED_CANDIDATES
        ],
    )

    assert "`seed_batch`" in availability
    assert "all configured measured seed fallback candidates are blacklisted" in availability
    assert "seed_batch" not in viable


def test_slot_query_ports_from_stack_priors_uses_live_primary_llama_entries(tmp_path: Path) -> None:
    priors = tmp_path / "stack_priors.yaml"
    priors.write_text(
        """
roles:
  frontdoor:
    deployment_status: live_stack
    serving:
      binary: llama.cpp
      launch:
        entries:
          - {port: 8070, alias: false}
          - {port: 8080, alias: false}
  coder_escalation:
    deployment_status: live_stack
    serving:
      binary: llama.cpp
      launch:
        entries:
          - {port: 8070, alias: true}
  worker_general:
    deployment_status: live_stack
    serving:
      binary: ik-pr1744
      launch:
        runtime:
          binary_path: /mnt/raid0/llm/ik_llama.cpp/build/bin/llama-server
        entries:
          - {port: 8072, alias: false}
  reap_candidate:
    deployment_status: benchmark_only
    serving:
      binary: llama.cpp
      launch:
        entries:
          - {port: 8099, alias: false}
  embedder:
    deployment_status: live_stack
    serving:
      binary: embedding-server
      launch:
        entries:
          - {port: 8090, alias: false}
""".lstrip(),
        encoding="utf-8",
    )

    assert autopilot._slot_query_ports_from_stack_priors(priors) == {
        "frontdoor": [8070, 8080],
        "worker_general": [8072],
    }


def test_slot_query_ports_falls_back_when_stack_priors_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(autopilot, "_slot_query_ports_from_stack_priors", lambda: {})

    assert autopilot._slot_query_ports() == autopilot._FALLBACK_SLOT_QUERY_PORTS


def test_query_slot_memory_separates_empty_and_unavailable_ports(monkeypatch) -> None:
    monkeypatch.setattr(
        autopilot,
        "_slot_query_ports",
        lambda: {"frontdoor": [8070], "worker_general": [8072]},
    )

    class _Response:
        def __init__(self, status_code: int, payload: list[dict] | None = None) -> None:
            self.status_code = status_code
            self._payload = payload or []

        def json(self) -> list[dict]:
            return self._payload

    def _fake_get(url: str, timeout: float) -> _Response:
        assert timeout == 3.0
        if url.endswith(":8070/slots"):
            return _Response(200, [{"id": 0, "state": "idle", "n_past": 0}])
        raise OSError("connection refused")

    monkeypatch.setattr("httpx.get", _fake_get)

    text = autopilot._query_slot_memory()

    assert "healthy queried ports with empty KV cache: frontdoor:8070" in text
    assert "unavailable configured replica ports: worker_general:8072" in text
    assert "do not infer eval-instrument contamination" in text
    assert "all slots empty or servers offline" not in text


def test_build_exploration_block_resilient_to_archive_errors(tmp_path: Path) -> None:
    j = _well_populated_journal(
        tmp_path,
        ["seed_batch", "numeric_trial", "prompt_mutation",
         "code_mutation", "structural_experiment", "deep_eval"],
    )

    class _BrokenArchive:
        def geometry(self) -> dict:
            raise RuntimeError("boom")

    block, signal = autopilot._build_exploration_block(
        journal=j, archive=_BrokenArchive(), known_actions=KNOWN_ACTIONS,
    )
    # Should still return a usable lean block (no other stagnation signals).
    assert "alternatives" in block
    assert signal == "none (lean prompt)"
