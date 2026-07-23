"""Validate the committed designed T1 core file (benchmarks/prompts/core_v2.jsonl).

The loader ``EvalTower._load_designed_core`` and the fail-closed
``designed_core_activation_guard`` are exercised against the REAL, versioned core
artifact (not a synthetic fixture). Offline: the core embeds full question rows,
so no question-pool load and no network/inference occur.

This is the "loads the real file" acceptance for the core_v2 build. It also
asserts the operator gate is still closed: the live instrument-era registry does
NOT yet authorize core_v2, so activation must fail closed until the operator
appends the human-owned quality/core era row.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "autopilot"))

import eval_tower  # noqa: E402
from eval_tower import EvalTower, dataset_content_sha256  # noqa: E402
from src.autopilot_core.instrument_era_guard import (  # noqa: E402
    designed_core_activation_guard,
)

CORE_ID = "core_v2"
CORE_PATH = REPO_ROOT / "benchmarks" / "prompts" / "core_v2.jsonl"
_NOW = datetime(2026, 7, 23, 12, 0, tzinfo=timezone.utc)

# Suites that MUST NOT appear in the designed core (excluded by design/task).
_FORBIDDEN_SUITES = {"vl", "tulving_episodic", "aa_lcr", "document_extraction", "gaia"}
# Item-level known-dead qids that must be absent.
_KNOWN_DEAD_QIDS = {
    "usaco_silver_1326", "usaco_silver_759",
    "ifeval_2292", "ifeval_3691",
    "bcb_BigCodeBench/228", "bcb_BigCodeBench/51",
    "chart_test_0452", "chart_test_1401",
}


def _load(monkeypatch):
    monkeypatch.setenv("AUTOPILOT_T1_CORE_ID", CORE_ID)
    monkeypatch.delenv("AUTOPILOT_T1_CORE_PATH", raising=False)  # exercise the default path
    return EvalTower()._load_designed_core(CORE_ID)


def test_core_v2_file_exists_at_default_path():
    assert CORE_PATH.exists(), f"missing designed core artifact: {CORE_PATH}"


def test_loader_accepts_real_core_file(monkeypatch):
    questions, metadata, path = _load(monkeypatch)
    assert Path(path) == CORE_PATH
    assert metadata["core_id"] == CORE_ID
    assert metadata["policy_version"] == "core_v2_designed_e7_v1"
    assert metadata["selected_count"] == len(questions)
    assert len(questions) == 50
    # every loaded question is scoreable under the live eval-tower gate
    assert all(eval_tower._is_scoreable_question(q) for q in questions)


def test_core_dataset_sha_is_reproducible(monkeypatch):
    questions, metadata, _ = _load(monkeypatch)
    assert dataset_content_sha256(questions) == metadata["dataset_content_sha256"]
    assert len(metadata["dataset_content_sha256"]) == 64


def test_core_stratification_and_exclusions(monkeypatch):
    questions, metadata, _ = _load(monkeypatch)
    suites = {q["suite"] for q in questions}
    assert len(suites) == 36
    assert suites.isdisjoint(_FORBIDDEN_SUITES)
    ids = {str(q.get("id", "")) for q in questions}
    assert ids.isdisjoint(_KNOWN_DEAD_QIDS)
    # dominant pool suites are capped (decision-value, not pool size)
    from collections import Counter
    per_suite = Counter(q["suite"] for q in questions)
    assert max(per_suite.values()) <= 2
    for dominant in ("general", "mmlu_pro", "thinking", "hotpotqa"):
        assert per_suite[dominant] <= 2
    # metadata records the exclusion rationale
    excl = metadata["exclusions"]
    assert "vl" in excl and "tulving_episodic" in excl
    assert set(excl["zero_source_suites"]["suites"]) == {"aa_lcr", "document_extraction", "gaia"}


def test_core_difficulty_spread_uses_apriori_tier(monkeypatch):
    questions, _, _ = _load(monkeypatch)
    tiers = [int(q["tier"]) for q in questions if q.get("tier") is not None]
    assert len(tiers) == 50
    # spread across all three a-priori difficulty tiers (not saturated on one)
    assert set(tiers) == {1, 2, 3}
    assert min(tiers) == 1 and max(tiers) == 3


def test_activation_guard_fails_closed_on_live_registry(monkeypatch):
    """The operator has NOT authorized core_v2 yet: activation must fail closed."""
    monkeypatch.delenv("AUTOPILOT_INSTRUMENT_ERAS_PATH", raising=False)
    verdict = designed_core_activation_guard(CORE_ID, now=_NOW)
    assert verdict["ok"] is False
    assert verdict["status"] == "missing_core_era"
    assert "human-owned E4/core row" in verdict["reason"]


def test_activation_guard_authorizes_with_operator_era_row(tmp_path, monkeypatch):
    """The ready-to-paste operator era row authorizes activation."""
    eras = tmp_path / "instrument_eras.yaml"
    eras.write_text(
        "eras:\n"
        "  - id: E8-quality-core-v2\n"
        '    from: "2026-07-23T00:00:00Z"\n'
        "    scope: autopilot_quality\n"
        f'    core_id: "{CORE_ID}"\n'
        '    policy_version: "core_v2_designed_e7_v1"\n'
    )
    verdict = designed_core_activation_guard(CORE_ID, path=eras, now=_NOW)
    assert verdict["ok"] is True
    assert verdict["status"] == "authorized"
    assert verdict["era"]["core_id"] == CORE_ID
