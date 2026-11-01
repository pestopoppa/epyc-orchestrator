"""Offline contracts for the OAB-8 SEARCH primitive (no inference or network calls)."""

from types import SimpleNamespace

import pytest
from src.repl_environment.context_bundle import ContextBundle, Section
from src.repl_environment.context_search import ContextSearchIndex
from src.scout_search import ScoutSearch, parse_search_command


def _existing_index():
    return ContextSearchIndex(
        [
            SimpleNamespace(name="profile", text="ggml_vec_dot_q8_0 dominates decode\n"),
            SimpleNamespace(name="notes", text="NUMA interleave controls memory placement\n"),
        ]
    )


def test_parser_preserves_multiword_query_and_ignores_other_verbs():
    assert parse_search_command("  search   ggml_vec_dot_q8_0 hot loop  ").query == (
        "ggml_vec_dot_q8_0 hot loop"
    )
    assert parse_search_command("READ file.py 1 5") is None
    with pytest.raises(ValueError, match="non-empty"):
        parse_search_command("SEARCH   ")


def test_existing_context_search_index_runs_real_lexical_search():
    result = ScoutSearch(_existing_index(), embedder=None).execute("q8_0 decode")
    assert result["status"] == "ok"
    assert result["mode"] == "lexical"
    assert result["hits"]
    assert result["hits"][0]["section"] == "profile"
    assert set(result["hits"][0]) == {
        "section", "start", "end", "line", "score", "via", "mode"
    }


def test_existing_context_bundle_interface_preserves_its_lexical_mode():
    bundle = ContextBundle([Section("scope:one", "needle in the authorized passage")])
    bundle.set_embedder(None)
    result = ScoutSearch(bundle, allowed_sections={"scope:one"}).execute(
        "authorized passage", section="scope:one"
    )
    assert result["status"] == "ok"
    assert result["mode"] == "lexical"
    assert result["hits"][0]["via"] == ["bm25"]


@pytest.mark.parametrize(
    "changes, message, error",
    [
        ({"start": True}, "invalid start", ValueError),
        ({"end": 0}, "invalid end", ValueError),
        ({"start": 5, "end": 4}, "empty or reversed", ValueError),
        ({"line": 1.0}, "invalid line", ValueError),
        ({"score": "0.5"}, "non-numeric", TypeError),
        ({"score": float("nan")}, "non-finite", ValueError),
        ({"via": ["unknown-ranker"]}, "ranking sources", ValueError),
        ({"mode": "unknown-mode"}, "unknown mode", ValueError),
    ],
)
def test_invalid_pointer_records_are_rejected(changes, message, error):
    class Provider:
        def search(self, query, k, *, section, embedder):
            hit = {
                "section": "allowed",
                "start": 0,
                "end": 4,
                "line": 1,
                "score": 0.1,
                "via": ["bm25"],
                "mode": "lexical",
            }
            hit.update(changes)
            return [hit], {"mode": "lexical", "reason": "disabled"}

    with pytest.raises(error, match=message):
        ScoutSearch(Provider()).execute("query")


def test_scope_is_opaque_and_provider_cannot_return_an_out_of_scope_pointer():
    class Provider:
        def search(self, query, k, *, section, embedder):
            assert section == "opaque://bundle/allowed"
            return ([{
                "section": "opaque://bundle/other", "start": 0, "end": 1, "line": 1,
                "score": 1.0, "via": ["bm25"], "mode": "lexical",
            }], {"mode": "lexical", "reason": None})

    search = ScoutSearch(Provider(), allowed_sections={"opaque://bundle/allowed"})
    with pytest.raises(KeyError, match="outside the authorized scope"):
        search.execute("query", section="/etc/passwd")
    with pytest.raises(ValueError, match="outside the requested section"):
        search.execute("query", section="opaque://bundle/allowed")


def test_missing_provider_is_explicitly_unavailable_and_does_not_fake_lexical_hits():
    result = ScoutSearch(None).execute("query")
    assert result["status"] == "unavailable"
    assert result["mode"] == "unavailable"
    assert result["reason"] == "search_backend_unavailable"
    assert result["hits"] == []


@pytest.mark.parametrize(
    "meta, expected_status, expected_mode",
    [
        ({"mode": "lexical", "reason": "disabled"}, "ok", "lexical"),
        ({"mode": "unrecognized", "reason": None}, "unknown", "unknown"),
        ({}, "unknown", "unknown"),
        ({"mode": "unavailable", "reason": "backend_down"}, "unavailable", "unavailable"),
    ],
)
def test_empty_results_require_known_mode_metadata(meta, expected_status, expected_mode):
    class Provider:
        def search(self, query, k, *, section, embedder):
            return [], meta

    result = ScoutSearch(Provider()).execute("query")
    assert result["hits"] == []
    assert result["status"] == expected_status
    assert result["mode"] == expected_mode


def test_explicit_unknown_mode_stays_unknown_even_when_hits_are_present():
    class Provider:
        def search(self, query, k, *, section, embedder):
            return ([{
                "section": "s", "start": 0, "end": 2, "line": 1, "score": 0.5,
                "via": ["bm25"], "mode": "lexical",
            }], {"mode": "mystery", "reason": None})

    result = ScoutSearch(Provider()).execute("query")
    assert result["status"] == "unknown"
    assert result["mode"] == "unknown"


def test_context_bundle_empty_result_without_mode_is_unknown_not_ok():
    bundle = ContextBundle([Section("scope", "unrelated text")])
    bundle.set_embedder(None)
    result = ScoutSearch(bundle).execute("absent-token")
    assert result["hits"] == []
    assert result["status"] == "unknown"
    assert result["mode"] == "unknown"


@pytest.mark.parametrize(
    "allowed, error",
    [
        ("scope", TypeError),
        ({"", "scope"}, ValueError),
        ({"scope", 7}, ValueError),
    ],
)
def test_scope_id_collection_is_validated(allowed, error):
    with pytest.raises(error, match="allowed_sections"):
        ScoutSearch(None, allowed_sections=allowed)


def test_query_validation_and_limits_are_enforced_before_provider_call():
    class Provider:
        def __init__(self):
            self.calls = []

        def search(self, query, k, *, section, embedder):
            self.calls.append((query, k, section))
            return [], {"mode": "lexical", "reason": None}

    provider = Provider()
    search = ScoutSearch(provider)
    with pytest.raises(ValueError, match="non-empty"):
        search.execute(" \n")
    with pytest.raises(TypeError, match="integer"):
        search.execute("query", k=True)
    result = search.execute("x" * 2200, k=500)
    assert len(provider.calls[-1][0]) == 2000
    assert provider.calls[-1][1] == 50
    assert result["k"] == 50
