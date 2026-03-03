#!/usr/bin/env python3
"""Unit tests for _dedup_pages() in src/tools/web/research.py."""

import pytest

from src.tools.web.research import _dedup_pages, _MIN_PARAGRAPH_LEN


class TestDedupPages:
    """Test paragraph-level SHA256 deduplication."""

    def test_no_overlap_passthrough(self):
        """Pages with no overlapping content pass through unchanged."""
        pages = [
            {"url": "https://a.com", "content": "A" * 100 + "\n\n" + "B" * 100},
            {"url": "https://b.com", "content": "C" * 100 + "\n\n" + "D" * 100},
        ]
        deduped, stats = _dedup_pages(pages)
        assert len(deduped) == 2
        assert deduped[0]["content"] == pages[0]["content"]
        assert deduped[1]["content"] == pages[1]["content"]
        assert stats["paragraphs_removed"] == 0
        assert stats["chars_saved"] == 0
        assert stats["pages_affected"] == 0

    def test_duplicate_paragraph_removed_from_second_page(self):
        """Duplicate paragraph in second page is removed; first page keeps it."""
        shared = "This is a shared paragraph that appears on multiple sites. " * 3
        page_a = {"url": "https://a.com", "content": shared + "\n\nUnique to A " * 10}
        page_b = {"url": "https://b.com", "content": "Unique to B " * 10 + "\n\n" + shared}

        deduped, stats = _dedup_pages([page_a, page_b])

        # First page keeps shared paragraph
        assert shared in deduped[0]["content"]
        # Second page loses it
        assert shared not in deduped[1]["content"]
        assert stats["paragraphs_removed"] == 1
        assert stats["pages_affected"] == 1

    def test_short_paragraphs_always_kept(self):
        """Paragraphs shorter than _MIN_PARAGRAPH_LEN are never deduped."""
        short = "Short."  # well under 80 chars
        assert len(short) < _MIN_PARAGRAPH_LEN

        pages = [
            {"url": "https://a.com", "content": short + "\n\n" + "A" * 100},
            {"url": "https://b.com", "content": short + "\n\n" + "B" * 100},
        ]
        deduped, stats = _dedup_pages(pages)
        # Both pages keep the short paragraph
        assert short in deduped[0]["content"]
        assert short in deduped[1]["content"]
        assert stats["paragraphs_removed"] == 0

    def test_case_whitespace_normalization(self):
        """Case and extra whitespace differences are normalized before hashing."""
        para_v1 = "Hello World this is a test paragraph with enough characters to be long." + " Extra." * 5
        para_v2 = "hello  world  this is a test paragraph with enough characters to be long." + "  extra." * 5
        assert len(para_v1.strip()) >= _MIN_PARAGRAPH_LEN

        pages = [
            {"url": "https://a.com", "content": para_v1},
            {"url": "https://b.com", "content": para_v2},
        ]
        deduped, stats = _dedup_pages(pages)
        assert stats["paragraphs_removed"] == 1
        assert stats["pages_affected"] == 1
        # First page kept, second page's paragraph removed
        assert deduped[0]["content"].strip() == para_v1
        assert deduped[1]["content"].strip() == ""

    def test_empty_and_whitespace_pages(self):
        """Empty and whitespace-only content handled gracefully."""
        pages = [
            {"url": "https://a.com", "content": ""},
            {"url": "https://b.com", "content": "   \n\n   "},
            {"url": "https://c.com", "content": "Real content " * 20},
        ]
        deduped, stats = _dedup_pages(pages)
        assert len(deduped) == 3
        assert stats["paragraphs_removed"] == 0

    def test_stats_consistency(self):
        """Stats counts match actual removals."""
        shared_1 = "Shared paragraph one with enough length to exceed the minimum. " * 2
        shared_2 = "Shared paragraph two with enough length to exceed the minimum. " * 2
        unique = "Unique content that only appears once in a single page context. " * 2

        page_a = {"url": "https://a.com", "content": shared_1 + "\n\n" + shared_2 + "\n\n" + unique}
        page_b = {"url": "https://b.com", "content": shared_1 + "\n\n" + shared_2}
        page_c = {"url": "https://c.com", "content": shared_1 + "\n\n" + "Only on C " * 20}

        deduped, stats = _dedup_pages([page_a, page_b, page_c])
        # page_b loses both shared paragraphs, page_c loses shared_1
        assert stats["paragraphs_removed"] == 3
        assert stats["pages_affected"] == 2
        assert stats["chars_saved"] == len(shared_1.strip()) * 2 + len(shared_2.strip())

    def test_rank_ordering_preserved(self):
        """First page in order retains content; later pages lose duplicates."""
        shared = "This paragraph is duplicated across all three pages in the list. " * 2
        pages = [
            {"url": "https://rank1.com", "content": shared},
            {"url": "https://rank2.com", "content": shared},
            {"url": "https://rank3.com", "content": shared},
        ]
        deduped, stats = _dedup_pages(pages)

        # First page (highest rank) keeps shared content
        assert shared.strip() in deduped[0]["content"]
        # Later pages lose it
        assert deduped[1]["content"].strip() == ""
        assert deduped[2]["content"].strip() == ""
        assert stats["paragraphs_removed"] == 2
        assert stats["pages_affected"] == 2


class TestToolPolicyGroup:
    """Verify web_research is in the group:web tool group."""

    def test_web_research_in_group_web(self):
        from src.tool_policy import TOOL_GROUPS

        assert "web_research" in TOOL_GROUPS["group:web"]
