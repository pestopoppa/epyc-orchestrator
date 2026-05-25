"""Unit tests for src/context_assembly.py (DCP-1 context bundle data model)."""

from __future__ import annotations

import pytest

from src.context_assembly import (
    InclusionMode,
    SourceKind,
    LineRange,
    merge_line_ranges,
    conservative_char_estimator,
    default_exclusion_reason,
    BundleEntry,
    BudgetBands,
    ContextBundle,
)


# ─── line range merge ────────────────────────────────────────────────────────────

def test_merge_overlapping_and_adjacent() -> None:
    ranges = [LineRange(1, 10), LineRange(11, 20), LineRange(5, 8), LineRange(40, 50)]
    merged = merge_line_ranges(ranges)
    assert [(r.start, r.end) for r in merged] == [(1, 20), (40, 50)]


def test_merge_respects_gap() -> None:
    # 1-10 and 13-15: gap of 2 lines (11,12) — not merged at default gap=0.
    assert len(merge_line_ranges([LineRange(1, 10), LineRange(13, 15)])) == 2
    # With adjacency_gap=2, they merge.
    merged = merge_line_ranges([LineRange(1, 10), LineRange(13, 15)], adjacency_gap=2)
    assert [(r.start, r.end) for r in merged] == [(1, 15)]


def test_merge_empty() -> None:
    assert merge_line_ranges([]) == []


def test_invalid_line_range_rejected() -> None:
    with pytest.raises(ValueError):
        LineRange(0, 5)
    with pytest.raises(ValueError):
        LineRange(10, 5)


# ─── token estimator ─────────────────────────────────────────────────────────────

def test_conservative_estimator_overestimates() -> None:
    assert conservative_char_estimator("") == 0
    # 35 chars / 3.5 = 10; conservative ceil keeps it >= a real tokenizer for code.
    assert conservative_char_estimator("x" * 35) == 10
    assert conservative_char_estimator("a") == 1  # non-empty floor


# ─── exclusion policy ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("path,expected_substr", [
    ("node_modules/foo/bar.js", "vendored"),
    ("src/.venv/lib/x.py", "vendored"),
    ("models/model.gguf", "binary"),
    ("assets/logo.png", "binary"),
    ("package-lock.json", "lockfile"),
    ("Cargo.lock", "lockfile"),
    ("api/service_pb2.py", "generated"),
    (".env.production", "secrets"),
    ("keys/server.pem", "secrets"),
])
def test_exclusion_policy_flags(path: str, expected_substr: str) -> None:
    reason = default_exclusion_reason(path)
    assert reason is not None and expected_substr in reason


def test_exclusion_policy_allows_normal_source() -> None:
    assert default_exclusion_reason("src/orchestration/dispatcher.py") is None
    assert default_exclusion_reason("tests/unit/test_foo.py") is None


# ─── bundle entry ────────────────────────────────────────────────────────────────

def test_entry_normalizes_line_ranges() -> None:
    e = BundleEntry(
        path="a.py", mode=InclusionMode.SLICES,
        line_ranges=[LineRange(10, 20), LineRange(1, 5), LineRange(18, 25)],
        source=SourceKind.COLGREP,
    )
    assert [(r.start, r.end) for r in e.line_ranges] == [(1, 5), (10, 25)]


def test_entry_rejects_bad_mode_and_source() -> None:
    with pytest.raises(ValueError):
        BundleEntry(path="a.py", mode="bogus")
    with pytest.raises(ValueError):
        BundleEntry(path="a.py", source="bogus")


# ─── bundle accounting + policy application ──────────────────────────────────────

def test_add_entry_estimates_and_accounts() -> None:
    b = ContextBundle(budget=100)
    b.add_entry(BundleEntry(path="a.py", mode=InclusionMode.FULL, source=SourceKind.DIRECT_READ),
                body_text="x" * 35)  # → 10 tokens
    b.add_entry(BundleEntry(path="b.py", mode=InclusionMode.FULL, source=SourceKind.DIRECT_READ),
                body_text="y" * 70)  # → 20 tokens
    assert b.total_tokens() == 30
    assert b.remaining() == 70
    assert b.fits()


def test_add_entry_applies_exclusion_policy() -> None:
    b = ContextBundle(budget=100)
    e = b.add_entry(BundleEntry(path="node_modules/x.js", mode=InclusionMode.FULL,
                                source=SourceKind.COLGREP), body_text="z" * 350)
    assert e.mode == InclusionMode.EXCLUDED
    assert "vendored" in e.reason_downgraded_or_excluded
    # Excluded entries do not count toward budget.
    assert b.total_tokens() == 0


def test_manual_seed_bypasses_exclusion_policy() -> None:
    b = ContextBundle(budget=100)
    e = b.add_entry(BundleEntry(path=".env", mode=InclusionMode.FULL,
                                source=SourceKind.MANUAL_SEED), body_text="SECRET=1")
    assert e.mode == InclusionMode.FULL  # explicit manual seed wins
    assert b.total_tokens() > 0


def test_downgrade_and_exclude_with_reasons() -> None:
    b = ContextBundle(budget=10)
    b.add_entry(BundleEntry(path="big.py", mode=InclusionMode.FULL, source=SourceKind.DIRECT_READ),
                body_text="x" * 350)  # 100 tokens > budget
    assert not b.fits()
    assert b.downgrade("big.py", InclusionMode.CODEMAP_ONLY, "over budget; signatures only")
    # downgrade keeps it counting (codemap_only still counts), so set tokens to reflect codemap size
    for e in b.entries:
        if e.path == "big.py":
            e.estimated_tokens = 5
    assert b.fits()
    assert b.exclude("big.py", "still too big")
    assert b.total_tokens() == 0
    assert b.excluded()[0].path == "big.py"
    assert not b.downgrade("missing.py", InclusionMode.FULL, "x")  # miss returns False


def test_manifest_separates_metadata_from_text() -> None:
    b = ContextBundle(budget=100, bundle_id="bnd-1", repo_sha="abc", gitnexus_index_commit="def",
                      bands=BudgetBands(task=10, codemap=20, editable=40, tests=10, output_reserve=20))
    b.add_entry(BundleEntry(path="a.py", mode=InclusionMode.SLICES,
                            line_ranges=[LineRange(1, 5), LineRange(4, 9)],
                            symbol_ids=["a.py::foo"], content_sha256="sha",
                            source=SourceKind.GITNEXUS, reason_included="caller of target"),
                body_text="x" * 35)
    m = b.manifest()
    assert m["bundle_id"] == "bnd-1"
    assert m["repo_sha"] == "abc"
    assert m["bands"]["editable"] == 40
    assert m["total_tokens"] == 10
    entry = m["entries"][0]
    assert entry["line_ranges"] == [[1, 9]]  # merged
    assert entry["symbol_ids"] == ["a.py::foo"]
    assert entry["reason_included"] == "caller of target"
    # manifest carries no rendered body text — only metadata + token estimate
    assert "body_text" not in entry and "content" not in entry


def test_budget_bands_total() -> None:
    bands = BudgetBands(task=10, codemap=20, editable=40, tests=10, output_reserve=20)
    assert bands.total_reserved() == 100
