"""OAB-17 F10: bundle-pull detection in auto_wrap_final is anchored."""
from __future__ import annotations

import pytest

from src.prompt_builders.code_utils import auto_wrap_final


@pytest.mark.parametrize("code", [
    "x = context.get('inbox', '')",
    "context.json('target.recipe')",
    "print(context.index())",
    "hits = context.search('q')",
    "s = context['target']",
    'd = context["plan"]',
    "x = context.item('target')",
    "x = context.page(2)",
    "ks = context.keys()",
    "x = context[name]",
    "x = context[ section ]",
    "x = context[_k]",
])
def test_bundle_pulls_are_left_unwrapped(code):
    assert auto_wrap_final(code) == code


@pytest.mark.parametrize("code, counts", [
    ("context.item('a')", True),
    ("context.page(1)", True),
    ("context.keys()", True),
    ("context[name]", True),
    ("context[:5000]", False),
    ("context[5:10]", False),
    ("context[0]", False),
    ("my_context.item('a')", False),
    ("obj.context.keys()", False),
])
def test_pull_regex_alternation_and_identifier_subscript(code, counts):
    from src.prompt_builders import code_utils
    assert bool(code_utils._CONTEXT_PULL_RE.search(code)) is counts


@pytest.mark.parametrize("code", [
    "context[:5000]",
    "my_context.get(1)",
    "obj.context.get(1)",
    "context[0]",
])
def test_unrelated_context_idioms_no_longer_match(code):
    # They are judged by the ordinary rules; the point is that the OAB-7 anchored
    # pattern alone did not keep them out of FINAL handling.
    from src.prompt_builders import code_utils
    assert code_utils._CONTEXT_PULL_RE.search(code) is None
