"""Tests for rubric_review.py — the two-turn rubric reviewer (RD-2).

Zero inference: the author/grader models are stub *completion callables* returning
canned text. We assert (a) rubric authoring produces schema-valid artifacts,
(b) the cache short-circuits the author model on hit + invalidates on refresh,
(c) grading computes S = Σ(w·s)/Σw and maps it to the right decision band, and
(d) majority-of-k engages only near band edges and logs flakiness.
"""

import json

import pytest
from jsonschema import Draft202012Validator

from src.proactive_delegation.rubric_review import (
    DEFAULT_BANDS,
    DecisionBands,
    RubricAuthoringError,
    RubricCache,
    RubricGradingError,
    author_rubric,
    grade_candidate,
    load_review_rubric_schema,
)


# ── stub completion callables (no model, no server) ───────────────────


def _author_stub(items, *, title="stub rubric"):
    payload = json.dumps({"title": title, "items": items})

    def _complete(prompt: str) -> str:
        # authors ignore the prompt and return the canned rubric with prose around it
        return f"Here is the rubric:\n```json\n{payload}\n```"

    _complete.calls = 0  # type: ignore[attr-defined]

    def _counting(prompt: str) -> str:
        _counting.calls += 1  # type: ignore[attr-defined]
        return _complete(prompt)

    _counting.calls = 0  # type: ignore[attr-defined]
    return _counting


def _grader_stub(scores_by_id, decision_hint=None):
    """Grader returns per-item scores; `scores_by_id` maps item id -> 0/1 score."""

    def _complete(prompt: str) -> str:
        grades = [{"item": rid, "score": s} for rid, s in scores_by_id.items()]
        body = {"grades": grades}
        if decision_hint:
            body["decision"] = decision_hint
        return json.dumps(body)

    return _complete


def _sequence_grader(sequence):
    """Grader returning a different canned response on each successive call."""
    state = {"i": 0}

    def _complete(prompt: str) -> str:
        i = min(state["i"], len(sequence) - 1)
        state["i"] += 1
        scores = sequence[i]
        return json.dumps({"grades": [{"item": rid, "score": s} for rid, s in scores.items()]})

    return _complete


THREE_ITEMS = [
    {"text": "Does it satisfy the spec?", "axis": "spec-alignment", "weight": 3},
    {"text": "Are edge cases covered?", "axis": "coverage", "weight": 2},
    {"text": "Is style consistent?", "axis": "integrity", "weight": 1},
]


# ── author_rubric ─────────────────────────────────────────────────────


class TestAuthorRubric:
    def test_authored_rubric_is_schema_valid(self, tmp_path):
        cache = RubricCache(tmp_path / "c.json")
        rubric = author_rubric(
            "code_patch", "code", "some repo context", _author_stub(THREE_ITEMS), cache=cache
        )
        errors = list(Draft202012Validator(load_review_rubric_schema()).iter_errors(rubric))
        assert errors == [], errors
        assert rubric["domain"] == "code"
        assert rubric["version"] == "1.0.0"
        assert [it["id"] for it in rubric["items"]] == ["R1", "R2", "R3"]

    def test_item_ids_renumbered_and_weights_coerced(self, tmp_path):
        cache = RubricCache(tmp_path / "c.json")
        weird = [
            {"text": "a", "axis": "x", "weight": 9},  # clamp to 3
            {"text": "b", "axis": "y", "weight": 0},  # clamp to 1
            {"text": "", "axis": "z", "weight": 2},  # dropped (empty text)
        ]
        rubric = author_rubric("t", "qa", "ctx", _author_stub(weird), cache=cache)
        assert [it["id"] for it in rubric["items"]] == ["R1", "R2"]
        assert [it["weight"] for it in rubric["items"]] == [3, 1]

    def test_cache_hit_does_not_call_author_model(self, tmp_path):
        cache = RubricCache(tmp_path / "c.json")
        author = _author_stub(THREE_ITEMS)
        author_rubric("t", "code", "ctx", author, cache=cache)
        assert author.calls == 1
        # second call: cache hit → author NOT invoked
        again = author_rubric("t", "code", "ctx", author, cache=cache)
        assert author.calls == 1
        assert again["version"] == "1.0.0"

    def test_refresh_bumps_version_and_reauthors(self, tmp_path):
        cache = RubricCache(tmp_path / "c.json")
        author = _author_stub(THREE_ITEMS)
        author_rubric("t", "code", "ctx", author, cache=cache)
        refreshed = author_rubric("t", "code", "ctx", author, cache=cache, refresh=True)
        assert author.calls == 2
        assert refreshed["version"] == "1.1.0"
        # both versions retrievable from cache; latest = refreshed
        assert cache.get("t", "code", version="1.0.0")["version"] == "1.0.0"
        assert cache.get("t", "code")["version"] == "1.1.0"

    def test_cache_persists_across_instances(self, tmp_path):
        path = tmp_path / "c.json"
        author = _author_stub(THREE_ITEMS)
        author_rubric("t", "code", "ctx", author, cache=RubricCache(path))
        # brand-new cache object over the same file → still a hit
        author2 = _author_stub(THREE_ITEMS)
        author_rubric("t", "code", "ctx", author2, cache=RubricCache(path))
        assert author2.calls == 0

    def test_author_no_json_raises(self, tmp_path):
        cache = RubricCache(tmp_path / "c.json")
        with pytest.raises(RubricAuthoringError):
            author_rubric("t", "code", "ctx", lambda p: "no json here", cache=cache)

    def test_author_missing_items_raises(self, tmp_path):
        cache = RubricCache(tmp_path / "c.json")
        with pytest.raises(RubricAuthoringError):
            author_rubric("t", "code", "ctx", lambda p: '{"title": "x"}', cache=cache)


# ── grade_candidate: band mapping + weighted S ────────────────────────


class TestGradeBands:
    def _rubric(self, tmp_path):
        return author_rubric(
            "t", "code", "ctx", _author_stub(THREE_ITEMS), cache=RubricCache(tmp_path / "c.json")
        )

    def test_all_pass_approves(self, tmp_path):
        rubric = self._rubric(tmp_path)
        res = grade_candidate(rubric, "cand", _grader_stub({"R1": 1, "R2": 1, "R3": 1}))
        assert res.S == 1.0
        assert res.decision == "approve"
        assert res.rubric_ref.endswith("@1.0.0")

    def test_all_fail_rejects(self, tmp_path):
        rubric = self._rubric(tmp_path)
        res = grade_candidate(rubric, "cand", _grader_stub({"R1": 0, "R2": 0, "R3": 0}))
        assert res.S == 0.0
        assert res.decision == "reject"

    def test_weighted_S_and_middle_band(self, tmp_path):
        rubric = self._rubric(tmp_path)
        # R1(w3)=1, R2(w2)=0, R3(w1)=1 -> S = (3+0+1)/6 = 0.666...
        res = grade_candidate(rubric, "cand", _grader_stub({"R1": 1, "R2": 0, "R3": 1}))
        assert abs(res.S - 4 / 6) < 1e-6
        # middle band, no failing critical item -> request_changes
        assert res.decision == "request_changes"

    def test_middle_band_critical_fail_requests_evidence(self, tmp_path):
        rubric = self._rubric(tmp_path)
        # R1(w3)=0 (critical fail), R2(w2)=1, R3(w1)=1 -> S = 3/6 = 0.5 -> reject edge
        # nudge into middle band: use bands with reject_at lower so S=0.5 is middle
        bands = DecisionBands(approve_at=0.85, reject_at=0.4)
        res = grade_candidate(
            rubric, "cand", _grader_stub({"R1": 0, "R2": 1, "R3": 1}), bands=bands
        )
        assert abs(res.S - 0.5) < 1e-6
        assert res.decision == "request_evidence"  # critical (w3) item failed

    def test_missing_item_grade_is_conservative_fail(self, tmp_path):
        rubric = self._rubric(tmp_path)
        # grader omits R3 → treated as ungraded binary 0
        res = grade_candidate(rubric, "cand", _grader_stub({"R1": 1, "R2": 1}))
        r3 = next(g for g in res.per_item if g.item == "R3")
        assert r3.graded is False and r3.binary == 0
        assert abs(res.S - 5 / 6) < 1e-6

    def test_grades_for_unknown_items_ignored(self, tmp_path):
        rubric = self._rubric(tmp_path)
        res = grade_candidate(
            rubric, "cand", _grader_stub({"R1": 1, "R2": 1, "R3": 1, "R99": 0})
        )
        assert res.S == 1.0  # R99 not in rubric → ignored
        assert {g.item for g in res.per_item} == {"R1", "R2", "R3"}

    def test_result_carries_rubric_and_per_item_for_h4(self, tmp_path):
        rubric = self._rubric(tmp_path)
        res = grade_candidate(rubric, "cand", _grader_stub({"R1": 1, "R2": 0, "R3": 1}))
        d = res.to_dict()
        # H4 persistence needs the full rubric + per-item grades on the result
        assert d["rubric"]["rubric_id"] == rubric["rubric_id"]
        assert len(d["per_item"]) == 3
        assert all("binary" in g and "weight" in g for g in d["per_item"])


# ── majority-of-k near edges ──────────────────────────────────────────


class TestMajorityK:
    def _rubric(self, tmp_path):
        return author_rubric(
            "t", "code", "ctx", _author_stub(THREE_ITEMS), cache=RubricCache(tmp_path / "c.json")
        )

    def test_single_pass_when_not_near_edge(self, tmp_path):
        rubric = self._rubric(tmp_path)
        calls = {"n": 0}

        def grader(prompt):
            calls["n"] += 1
            return json.dumps({"grades": [{"item": r, "score": 1} for r in ("R1", "R2", "R3")]})

        res = grade_candidate(rubric, "cand", grader, k=5)
        # S = 1.0 is far from both edges → only ONE pass even with k=5
        assert calls["n"] == 1
        assert res.k_used == 1
        assert res.near_edge is False

    def test_majority_of_k_engages_near_edge(self, tmp_path):
        rubric = self._rubric(tmp_path)
        # target S right at the approve edge (0.85 default): make S land in edge margin.
        # Use 2-item-weight config to hit ~0.833 (within 0.05 of 0.85).
        # R1(3)+R2(2)=5 of 6 -> 0.833; within edge_margin(0.05) of 0.85.
        calls = {"n": 0}

        def grader(prompt):
            calls["n"] += 1
            return json.dumps(
                {"grades": [{"item": "R1", "score": 1}, {"item": "R2", "score": 1}, {"item": "R3", "score": 0}]}
            )

        res = grade_candidate(rubric, "cand", grader, k=3)
        assert res.near_edge is True
        assert calls["n"] == 3
        assert res.k_used == 3

    def test_majority_vote_and_flakiness_logged(self, tmp_path):
        rubric = self._rubric(tmp_path)
        # near-edge S (0.833) two passes, one disagreeing pass (S lower).
        # pass1: R1,R2 =1 -> 0.833 (approve-edge, request/approve depending); ensure edge.
        seq = [
            {"R1": 1, "R2": 1, "R3": 0},  # 0.833 near approve edge
            {"R1": 1, "R2": 1, "R3": 0},  # same
            {"R1": 1, "R2": 0, "R3": 0},  # 0.5 -> different band
        ]
        res = grade_candidate(rubric, "cand", _sequence_grader(seq), k=3)
        assert res.k_used == 3
        # majority of the three decisions wins; flakiness = 1/3 (one disagreeing pass)
        assert res.flakiness == pytest.approx(1 / 3, abs=1e-3)
        assert 0.0 <= res.confidence <= 1.0

    def test_all_passes_unparseable_raises(self, tmp_path):
        rubric = self._rubric(tmp_path)
        with pytest.raises(RubricGradingError):
            grade_candidate(rubric, "cand", lambda p: "not json at all")


class TestBandsConfig:
    def test_default_bands_are_the_documented_priors(self):
        assert DEFAULT_BANDS.approve_at == 0.85
        assert DEFAULT_BANDS.reject_at == 0.5

    def test_bands_are_configurable(self, tmp_path):
        rubric = author_rubric(
            "t", "code", "ctx", _author_stub(THREE_ITEMS), cache=RubricCache(tmp_path / "c.json")
        )
        strict = DecisionBands(approve_at=0.99, reject_at=0.1)
        # S=0.833 would approve under defaults but not under a 0.99 approve gate
        res = grade_candidate(
            rubric, "cand", _grader_stub({"R1": 1, "R2": 1, "R3": 0}), bands=strict
        )
        assert res.decision != "approve"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
