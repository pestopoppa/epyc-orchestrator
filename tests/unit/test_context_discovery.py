"""Unit tests for src/context_discovery.py (DCP-2 discovery/cost/assemble + DCP-3 codemap)."""

from __future__ import annotations

import hashlib
import textwrap

import pytest

from src.context_assembly import (
    BundleEntry, ContextBundle, InclusionMode, LineRange, SourceKind,
    conservative_char_estimator,
)
from src.context_discovery import (
    DiscoveredHit,
    parse_colgrep_json,
    discover_candidates,
    build_python_codemap,
    cost_candidates,
    assemble_delegation_bundle,
    render_bundle,
    _python_edit_segments,
)


# ─── ColGREP JSON parsing ────────────────────────────────────────────────────────


def test_parse_colgrep_json_variants() -> None:
    payload = [
        {"path": "a.py", "start_line": 10, "end_line": 20, "score": 0.9},
        {"file": "b.py", "start": 5, "end": 5, "relevance": 0.5},  # alt field names
        {"nope": 1},  # no path → skipped
    ]
    hits = parse_colgrep_json(payload)
    assert [h.path for h in hits] == ["a.py", "b.py"]
    assert hits[0].line_ranges[0].start == 10 and hits[0].score == 0.9
    assert hits[1].line_ranges[0].end == 5


def test_parse_colgrep_json_string_and_garbage() -> None:
    assert parse_colgrep_json("not json") == []
    assert (
        parse_colgrep_json('[{"path": "x.py", "start_line": 1, "end_line": 2, "score": 0.3}]')[
            0
        ].path
        == "x.py"
    )
    assert parse_colgrep_json({"results": [{"path": "y.py"}]})[0].path == "y.py"  # nested


@pytest.mark.parametrize(
    "score",
    [float("nan"), float("inf"), float("-inf"), 10**1000],
    ids=("nan", "positive_infinity", "negative_infinity", "large_integer_overflow"),
)
def test_parse_colgrep_json_normalizes_non_finite_or_overflowing_scores(score) -> None:
    hits = parse_colgrep_json(
        [
            {"path": "bad-score.py", "score": score},
            {"path": "ordinary.py", "score": 0.25},
        ]
    )

    assert [hit.score for hit in hits] == [0.0, 0.25]
    ranked = discover_candidates(
        "q", code_search_fn=lambda _query, _limit: hits, max_files=2
    )
    assert [hit.path for hit in ranked] == ["ordinary.py", "bad-score.py"]


@pytest.mark.parametrize(
    "score",
    [float("nan"), float("inf"), float("-inf"), 10**1000],
    ids=("nan", "positive_infinity", "negative_infinity", "large_integer_overflow"),
)
def test_discover_normalizes_non_finite_direct_hits_without_mutating_inputs(score) -> None:
    bad = DiscoveredHit("same.py", [LineRange(1, 2)], score)
    finite_duplicate = DiscoveredHit("same.py", [LineRange(5, 6)], 0.1)
    ordinary = DiscoveredHit("ordinary.py", [LineRange(3, 3)], 0.25)
    original_hits = (bad, finite_duplicate, ordinary)
    original_scores = tuple(hit.score for hit in original_hits)
    original_ranges = tuple(hit.line_ranges for hit in original_hits)
    original_range_values = tuple(tuple(hit.line_ranges) for hit in original_hits)

    ranked = discover_candidates(
        "q", code_search_fn=lambda _query, _limit: list(original_hits), max_files=2
    )

    assert [hit.path for hit in ranked] == ["ordinary.py", "same.py"]
    same = ranked[1]
    assert same.score == 0.1
    assert same.line_ranges == [LineRange(1, 2), LineRange(5, 6)]
    assert all(hit.score is original for hit, original in zip(original_hits, original_scores))
    assert all(hit.line_ranges is original for hit, original in zip(original_hits, original_ranges))
    assert tuple(tuple(hit.line_ranges) for hit in original_hits) == original_range_values


# ─── discovery (pass 1) ──────────────────────────────────────────────────────────


def test_discover_groups_merges_ranks_and_excludes() -> None:
    def fake_search(query, limit):
        return [
            DiscoveredHit("a.py", [LineRange(1, 5)], 0.4),
            DiscoveredHit("a.py", [LineRange(4, 9)], 0.8),  # same file → merge ranges, max score
            DiscoveredHit("node_modules/x.js", [LineRange(1, 2)], 0.99),  # policy-excluded
            DiscoveredHit("b.py", [LineRange(10, 12)], 0.6),
        ]

    hits = discover_candidates("q", code_search_fn=fake_search, max_files=8)
    assert [h.path for h in hits] == ["a.py", "b.py"]  # node_modules dropped; ranked by score
    a = next(h for h in hits if h.path == "a.py")
    assert [(r.start, r.end) for r in a.line_ranges] == [(1, 9)]  # merged
    assert a.score == 0.8


def test_discover_respects_max_files() -> None:
    def fake_search(q, limit):
        return [DiscoveredHit(f"f{i}.py", [LineRange(1, 1)], float(i)) for i in range(10)]

    hits = discover_candidates("q", code_search_fn=fake_search, max_files=3)
    assert len(hits) == 3
    assert [h.path for h in hits] == ["f9.py", "f8.py", "f7.py"]  # top-3 by score


# ─── DCP-3 codemap ───────────────────────────────────────────────────────────────


def test_python_codemap_signatures_only() -> None:
    src = textwrap.dedent('''
        import os

        def top(a: int, b: str = "x") -> bool:
            """Top-level fn docstring.
            second line ignored."""
            return True

        class Foo(Base):
            """Foo does things."""
            def method(self, n) -> None:
                secret = 42
                return None
    ''')
    cm = build_python_codemap(src)
    assert "def top(a: int, b: str='x') -> bool: ..." in cm
    assert "# Top-level fn docstring." in cm
    assert "class Foo(Base):" in cm
    assert "def method(self, n) -> None: ..." in cm
    # bodies are NOT included
    assert "secret = 42" not in cm
    assert "return True" not in cm


def test_python_codemap_syntax_error_returns_none() -> None:
    assert build_python_codemap("def broken(:\n") is None


def test_python_codemap_empty_module_returns_none() -> None:
    assert build_python_codemap("x = 1\n") is None  # no classes/functions


# ─── cost (pass 2) ───────────────────────────────────────────────────────────────


def test_cost_candidates_computes_modes() -> None:
    files = {
        "a.py": "def f():\n    return 1\n" + ("# pad\n" * 50),  # big-ish python
        "b.txt": "x" * 700,  # non-python → no codemap
    }
    hits = [
        DiscoveredHit("a.py", [LineRange(1, 2)], 0.9),
        DiscoveredHit("b.txt", [], 0.5),
    ]
    cands = cost_candidates(hits, file_reader_fn=lambda p: files[p])
    a = next(c for c in cands if c.path == "a.py")
    b = next(c for c in cands if c.path == "b.txt")
    assert a.desired_mode == InclusionMode.SLICES  # had ranges
    assert a.content_sha256 is not None and len(a.content_sha256) == 64
    assert a.cost_codemap < a.cost_full  # codemap cheaper than full body
    assert a.cost_slices < a.cost_full  # 2 lines < whole file
    assert b.desired_mode == InclusionMode.FULL  # no ranges
    assert b.content_sha256 is not None and len(b.content_sha256) == 64
    assert b.priority == 0.5


def test_ast_slice_preserves_decorated_async_method_and_source_context() -> None:
    body = '''import os
from pathlib import Path

class Outer(
    object,
):
    @classmethod
    @unrelated_decorator(
        "earlier",
    )
    def earlier(cls):
        import local_only
        return local_only

    @classmethod
    @decorate(
        "value",
    )
    async def target(
        cls,
        value: str,
    ) -> str:
        return value

def unrelated():
    return os.getcwd()
'''
    files = {"pkg.py": body}
    hit_line = body.splitlines().index("        value: str,") + 1
    candidate = cost_candidates(
        [DiscoveredHit("pkg.py", [LineRange(hit_line, hit_line)], .9)],
        file_reader_fn=lambda path: files[path],
    )[0]
    bundle = assemble_delegation_bundle(
        "edit target",
        budget=candidate.cost_slices,
        code_search_fn=lambda _query, _limit: [
            DiscoveredHit("pkg.py", [LineRange(hit_line, hit_line)], .9)
        ],
        file_reader_fn=lambda path: files[path],
    )
    entry = next(item for item in bundle.entries if item.path == "pkg.py")

    assert entry.mode == InclusionMode.SLICES
    rendered = render_bundle(bundle, file_reader_fn=lambda path: files[path])
    assert "import os" in rendered and "from pathlib import Path" in rendered
    assert "class Outer(" in rendered and "    object," in rendered
    assert "@classmethod" in rendered and "@decorate(" in rendered
    assert "    async def target(" in rendered
    assert "        return value" in rendered
    assert "def unrelated" not in rendered
    assert "def earlier" not in rendered
    assert "unrelated_decorator" not in rendered
    assert "local_only" not in rendered
    assert "# Lines 1-" in rendered
    assert "# ..." in rendered
    planned_text = _python_edit_segments(body, [LineRange(hit_line, hit_line)])[0]
    expected_block = f"### pkg.py (slices)\n{planned_text}\n\n"
    assert candidate.cost_slices == conservative_char_estimator(expected_block)
    assert candidate.line_ranges == entry.line_ranges
    assert entry.render_ranges_verbatim is True
    assert bundle.manifest()["entries"][0]["render_ranges_verbatim"] is True
    assert conservative_char_estimator(rendered) <= bundle.total_tokens()


def test_multi_function_hit_keeps_each_complete_segment() -> None:
    body = "def first():\n    return 1\n\ndef second():\n    return 2\n"
    ranges = [LineRange(2, 2), LineRange(5, 5)]
    result = _python_edit_segments(body, ranges)
    assert result is not None
    rendered, planned_ranges = result
    assert "def first():\n    return 1" in rendered
    assert "def second():\n    return 2" in rendered
    assert planned_ranges == [LineRange(1, 2), LineRange(4, 5)]


def test_colgrep_slices_without_verified_plan_or_pin_fail_closed() -> None:
    body = "def target():\n    return 1\n"
    digest = hashlib.sha256(body.encode()).hexdigest()
    for verified, pinned, ranges in (
        (False, True, [LineRange(1, 2)]),
        (True, False, [LineRange(1, 2)]),
        (True, True, [LineRange(1, 3)]),
        (True, True, [LineRange(2, 2), LineRange(1, 1)]),
    ):
        bundle = ContextBundle(budget=100)
        bundle.add_entry(
            BundleEntry(
                path="target.py",
                mode=InclusionMode.SLICES,
                line_ranges=ranges,
                source=SourceKind.COLGREP,
                content_sha256=digest if pinned else None,
                render_ranges_verbatim=verified,
            ),
            body_text=body,
        )
        assert render_bundle(bundle, file_reader_fn=lambda _path: body) == ""
        entry = bundle.manifest()["entries"][0]
        assert entry["mode"] == InclusionMode.EXCLUDED
        assert "missing evidence" in entry["reason_downgraded_or_excluded"]


def test_ast_planned_slice_preserves_crlf_source_lines() -> None:
    body = "import os\r\n\r\ndef target():\r\n    return os.getcwd()\r\n"
    result = _python_edit_segments(body, [LineRange(4, 4)])
    assert result is not None
    rendered, _ = result
    assert "import os\r\n" in rendered
    assert "def target():\r\n    return os.getcwd()\r\n" in rendered


def test_ast_slice_keeps_module_guarded_import_blocks_only() -> None:
    body = '''from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pkg.contracts import Contract

try:
    import fast_optional
except ImportError:
    fast_optional = None

def target():
    return 1

def unrelated():
    import local_only
    return local_only
    pass
    pass
    pass
    pass
    pass
    pass
    pass
    pass
    pass
    pass
    pass
    pass
    pass
    pass
    pass
    pass
    pass
    pass
    pass
    pass
'''
    files = {"guarded.py": body}
    hit_line = body.splitlines().index("    return 1") + 1
    hit = DiscoveredHit("guarded.py", [LineRange(hit_line, hit_line)], .9)
    candidate = cost_candidates([hit], file_reader_fn=lambda path: files[path])[0]
    bundle = assemble_delegation_bundle(
        "edit target",
        budget=candidate.cost_slices,
        code_search_fn=lambda _query, _limit: [hit],
        file_reader_fn=lambda path: files[path],
    )
    rendered = render_bundle(bundle, file_reader_fn=lambda path: files[path])

    assert "if TYPE_CHECKING:\n    from pkg.contracts import Contract" in rendered
    assert "try:\n    import fast_optional\nexcept ImportError:\n    fast_optional = None" in rendered
    assert "def target():\n    return 1" in rendered
    assert "local_only" not in rendered
    assert candidate.cost_slices == bundle.total_tokens()
    assert conservative_char_estimator(rendered) <= bundle.total_tokens()


def test_unranged_colgrep_and_non_colgrep_slices_keep_existing_rendering() -> None:
    body = "def untouched():\n    return 1\n"
    full_bundle = assemble_delegation_bundle(
        "read file",
        budget=100,
        code_search_fn=lambda _query, _limit: [DiscoveredHit("plain.py", [], .9)],
        file_reader_fn=lambda _path: body,
    )
    assert render_bundle(full_bundle, file_reader_fn=lambda _path: body) == (
        "### plain.py (full)\n" + body
    )

    slices = ContextBundle(budget=100)
    slices.add_entry(
        BundleEntry(
            path="direct.py",
            mode=InclusionMode.SLICES,
            line_ranges=[LineRange(1, 1)],
            source=SourceKind.DIRECT_READ,
        ),
        body_text=body,
    )
    assert "render_ranges_verbatim" not in slices.manifest()["entries"][0]
    assert render_bundle(slices, file_reader_fn=lambda _path: body) == (
        "### direct.py (slices)\ndef untouched():"
    )


def test_unparseable_hit_is_full_or_missing_evidence_never_line_slice() -> None:
    body = "def broken(:\n" + ("# padding\n" * 20)
    hit = DiscoveredHit("broken.py", [LineRange(1, 1)], .9)
    candidate = cost_candidates([hit], file_reader_fn=lambda _path: body)[0]
    assert candidate.desired_mode == InclusionMode.FULL
    assert candidate.cost_slices == candidate.cost_full

    excluded = assemble_delegation_bundle(
        "edit broken",
        budget=1,
        code_search_fn=lambda _query, _limit: [hit],
        file_reader_fn=lambda _path: body,
    )
    entry = next(item for item in excluded.entries if item.path == "broken.py")
    assert entry.mode == InclusionMode.EXCLUDED
    assert "missing evidence" in entry.reason_downgraded_or_excluded


def test_assemble_records_unreadable_hit_as_missing_evidence() -> None:
    bundle = assemble_delegation_bundle(
        "edit gone",
        budget=100,
        code_search_fn=lambda _query, _limit: [DiscoveredHit("gone.py", [LineRange(4, 5)], .9)],
        file_reader_fn=lambda _path: None,
    )
    entry = next(item for item in bundle.manifest()["entries"] if item["path"] == "gone.py")
    assert entry["mode"] == InclusionMode.EXCLUDED
    assert entry["line_ranges"] == [[4, 5]]
    assert "missing evidence" in entry["reason_downgraded_or_excluded"]


def test_cost_candidates_skips_unreadable() -> None:
    def reader(p):
        raise FileNotFoundError(p)

    assert cost_candidates([DiscoveredHit("gone.py", [], 0.5)], file_reader_fn=reader) == []


# ─── end-to-end assemble ─────────────────────────────────────────────────────────


def test_assemble_delegation_bundle_end_to_end() -> None:
    files = {
        "hot.py": "def hot():\n" + ("    x = 1\n" * 40),
        "warm.py": "def warm():\n" + ("    y = 2\n" * 40),
    }

    def fake_search(query, limit):
        return [
            DiscoveredHit("hot.py", [LineRange(1, 2)], 0.9),
            DiscoveredHit("warm.py", [LineRange(1, 2)], 0.3),
        ]

    bundle = assemble_delegation_bundle(
        "fix hot path",
        budget=10_000,
        code_search_fn=fake_search,
        file_reader_fn=lambda p: files[p],
        bundle_id="b1",
    )
    assert bundle.bundle_id == "b1"
    assert bundle.fits()
    paths = {e.path for e in bundle.included()}
    assert paths == {"hot.py", "warm.py"}
    m = bundle.manifest()
    assert m["total_tokens"] <= 10_000
    # hot.py (higher score) is packed; manifest carries per-entry provenance
    hot = next(e for e in m["entries"] if e["path"] == "hot.py")
    assert hot["source"] == "colgrep"
    assert hot["mode"] in ("slices", "full")
    assert hot["content_sha256"] is not None
    assert len(hot["content_sha256"]) == 64


def test_assemble_tight_budget_downgrades_or_excludes() -> None:
    files = {"a.py": "def a():\n" + ("    x=1\n" * 200)}  # large

    def fake_search(q, limit):
        return [DiscoveredHit("a.py", [LineRange(1, 1)], 0.9)]

    bundle = assemble_delegation_bundle(
        "q", budget=5, code_search_fn=fake_search, file_reader_fn=lambda p: files[p]
    )
    # budget of 5 tokens → can't fit full; ends up sliced/codemap or excluded — but never overflows
    assert bundle.total_tokens() <= 5


# ─── DCP-4 render identity ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "mode",
    [InclusionMode.FULL, InclusionMode.SLICES, InclusionMode.CODEMAP_ONLY],
)
def test_render_rejects_changed_bound_body_before_mode_processing(mode) -> None:
    planned_body = "def planned():\n    return 'old'\n"
    changed_body = "def changed():\n    return 'new'\n"
    bundle = ContextBundle(budget=1000)
    bundle.add_entry(
        BundleEntry(
            path="module.py",
            mode=mode,
            line_ranges=[LineRange(1, 2)] if mode == InclusionMode.SLICES else [],
            content_sha256=hashlib.sha256(planned_body.encode("utf-8")).hexdigest(),
        )
    )
    codemap_calls = []

    with pytest.raises(ValueError, match="content changed after context planning"):
        render_bundle(
            bundle,
            file_reader_fn=lambda _path: changed_body,
            codemap_fn=lambda body: codemap_calls.append(body) or "stale codemap",
        )
    assert codemap_calls == []


def test_render_keeps_same_bound_and_unbound_entries_compatible() -> None:
    body = "def current():\n    return 'same'\n"
    bundle = ContextBundle(budget=1000)
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    bundle.add_entry(BundleEntry(path="bound.py", mode=InclusionMode.FULL, content_sha256=digest))
    bundle.add_entry(BundleEntry(path="unbound.py", mode=InclusionMode.FULL))

    rendered = render_bundle(bundle, file_reader_fn=lambda _path: body)

    assert "### bound.py (full)\n" + body in rendered
    assert "### unbound.py (full)\n" + body in rendered


@pytest.mark.parametrize("unreadable", ["none", "exception"])
def test_render_keeps_unreadable_bound_entries_as_skips(unreadable) -> None:
    bundle = ContextBundle(budget=1000)
    bundle.add_entry(
        BundleEntry(path="gone.py", mode=InclusionMode.FULL, content_sha256="0" * 64)
    )

    def reader(_path):
        if unreadable == "exception":
            raise OSError("unreadable")
        return None

    assert render_bundle(bundle, file_reader_fn=reader) == ""
    entry = bundle.manifest()["entries"][0]
    assert entry["mode"] == InclusionMode.EXCLUDED
    assert "missing evidence" in entry["reason_downgraded_or_excluded"]
