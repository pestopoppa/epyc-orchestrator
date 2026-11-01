"""Offline contracts for the pinned TD-17 fixture adapter."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

from src.typed_decisions.jev_ood_calibration import (
    UPSTREAM_COMMIT,
    UPSTREAM_DATA_SHA256,
    _ANGRY_TAILS,
    _LEVEL_NAMES,
    _SYNTH_TEMPLATES,
    _base_urgency,
    _clean_priority_for_tier,
    _possible_priority_values,
    _source_labels,
    build_fixtures,
    generate_fresh_rows,
    load_fixture_inputs,
    load_pinned_rows,
    main,
)


_FIXTURE = (
    Path(__file__).parents[1]
    / "fixtures"
    / "typed_decisions"
    / "jev_ood_calibration"
    / "val.jsonl"
)


def _groups(rows):
    return tuple(tuple(rows[offset : offset + 3]) for offset in range(0, len(rows), 3))


def test_actual_frozen_fixture_is_exactly_pinned_and_has_all_rows():
    raw = _FIXTURE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == UPSTREAM_DATA_SHA256
    rows = load_pinned_rows(_FIXTURE)
    assert len(rows) == 900
    assert [row["type"] for row in rows[:3]] == ["choice", "score", "noul"]
    assert all(
        [row["type"] for row in group] == ["choice", "score", "noul"]
        and group[0]["state"] == group[1]["state"] == group[2]["state"]
        for group in _groups(rows)
    )


def test_seed_zero_generation_reproduces_upstream_frozen_validation_split():
    assert generate_fresh_rows(0) == load_pinned_rows(_FIXTURE)


def test_all_twenty_template_oracles_match_the_pinned_generator_recipe():
    assert sum(len(items) for items in _SYNTH_TEMPLATES.values()) == 20
    fill = {"order": 1234, "amount": 19.99, "days": 4, "tail": _ANGRY_TAILS[0]}
    for queue_templates in _SYNTH_TEMPLATES.values():
        for subject_template, body_template, urgency in queue_templates:
            state = {
                "subject": subject_template.format(**fill),
                "body": body_template.format(**fill).strip(),
            }
            assert _base_urgency(state) == urgency
            assert _clean_priority_for_tier(state, "free") == min(3, urgency + 1)
            assert _clean_priority_for_tier(state, "standard") == min(3, urgency + 1)
            assert _clean_priority_for_tier(state, "gold") == min(3, urgency + 2)
            assert _clean_priority_for_tier(state, "enterprise") == min(3, urgency + 2)
    clamped = {
        "subject": "Charged twice for order #1234",
        "body": f"Two charges. {_ANGRY_TAILS[0]}",
    }
    assert _clean_priority_for_tier(clamped, "enterprise") == 3


def test_fresh_recipe_is_deterministic_and_uses_new_seed_for_population():
    first = generate_fresh_rows(8675309)
    repeated = generate_fresh_rows(8675309)
    other = generate_fresh_rows(8675310)
    assert first == repeated
    assert len(first) == 900
    assert first != other
    assert len(_groups(first)) == 300


def test_paired_fixtures_preserve_noisy_labels_and_oracle_separately():
    frozen = load_pinned_rows(_FIXTURE)
    fresh = list(generate_fresh_rows(741103))

    # Deterministic noise control: a singleton-priority ticket whose observed
    # source label is corrupted must remain a Low structural answer, not become
    # an unknown or have its frozen/source label silently repaired.
    singleton = {
        "channel": "email",
        "customer_tier": "enterprise",
        "subject": "Feature suggestion",
        "body": "It would be nice to sort the dashboard by date. Not urgent. Appreciate any help.",
    }
    for row in fresh[:3]:
        row["state"] = singleton
    fresh[0]["label"] = "general"
    fresh[1]["label"] = 2
    fresh[2]["label"] = False

    fixtures = build_fixtures(frozen, fresh)
    assert {name: len(items) for name, items in fixtures.items()} == {
        "known": 300,
        "fresh_known": 300,
        "tier_ablated": 300,
        "priority_forced_choice": 300,
        "priority_with_unknown": 300,
    }
    frozen_source = _groups(frozen)
    known = fixtures["known"][0]
    first_frozen_choice, first_frozen_score, first_frozen_noul = frozen_source[0]
    frozen_expected = _source_labels(
        "frozen-000", first_frozen_choice, first_frozen_score, first_frozen_noul
    )
    assert known.truth["source_observed_labels"] == frozen_expected
    assert known.truth["source_observed_priority_label"] == first_frozen_score["label"]

    fresh_known = fixtures["fresh_known"][0]
    ablated = fixtures["tier_ablated"][0]
    forced = fixtures["priority_forced_choice"][0]
    with_unknown = fixtures["priority_with_unknown"][0]
    full_state = json.loads(fresh_known.state)
    visible_state = json.loads(ablated.state)
    assert "customer_tier" in full_state
    assert "customer_tier" not in visible_state
    assert {
        key: value for key, value in full_state.items() if key != "customer_tier"
    } == visible_state
    assert fresh_known.pair_id == ablated.pair_id == forced.pair_id == with_unknown.pair_id
    assert forced.state == with_unknown.state == ablated.state
    assert forced.questions[0].text == with_unknown.questions[0].text
    assert forced.questions[0].options == with_unknown.questions[0].options[:-1]
    assert with_unknown.questions[0].options[-1] == "unknown"
    assert replace(
        forced.questions[0], options=with_unknown.questions[0].options
    ) == with_unknown.questions[0]
    assert ablated.truth["source_observed_priority_label"] == 2
    assert ablated.truth["clean_oracle_priority"] == 0
    assert ablated.truth["source_label_noise_disagreement"] is True
    assert ablated.truth["possible_priority_values"] == [0]
    assert forced.truth["benchmark_source_observed_choice"] == _LEVEL_NAMES[2]
    assert forced.truth["structural_singleton_expected_choice"] == _LEVEL_NAMES[0]
    assert with_unknown.truth["structural_expected_choice"] == _LEVEL_NAMES[0]
    assert with_unknown.truth["unknown_expected"] is False
    assert with_unknown.truth["unknown_option_eligible"] is False
    assert with_unknown.questions[0].options[-1] == "unknown"
    assert ablated.truth["source_observed_labels"]["fresh-000-priority"] == 2

    # Every frozen observed label remains byte-domain faithful even when noisy.
    for index, group in enumerate(frozen_source):
        fixture = fixtures["known"][index]
        choice, score, noul = group
        assert fixture.truth["source_observed_labels"] == _source_labels(
            f"frozen-{index:03d}", choice, score, noul
        )
        clean = _clean_priority_for_tier(choice["state"], choice["state"]["customer_tier"])
        assert fixture.truth["clean_oracle_priority"] == clean
        assert fixture.truth["source_label_noise_disagreement"] == (score["label"] != clean)

    fresh_groups = _groups(fresh)
    for index, group in enumerate(fresh_groups):
        choice, score, noul = group
        case_id = f"fresh-{index:03d}"
        pair = fixtures["fresh_known"][index]
        ablated = fixtures["tier_ablated"][index]
        expected_labels = _source_labels(case_id, choice, score, noul)
        assert pair.truth["source_observed_labels"] == expected_labels
        assert ablated.truth["source_observed_labels"] == expected_labels
        assert pair.pair_id == ablated.pair_id
        full_state = json.loads(pair.state)
        visible_state = json.loads(ablated.state)
        expected_visible = {
            key: value
            for key, value in full_state.items()
            if key != "customer_tier"
        }
        assert expected_visible == visible_state
        clean = _clean_priority_for_tier(full_state, full_state["customer_tier"])
        possible = sorted(_possible_priority_values(visible_state))
        assert ablated.truth["clean_oracle_priority"] == clean
        assert ablated.truth["possible_priority_values"] == possible
        assert ablated.truth["source_label_noise_disagreement"] == (score["label"] != clean)
        assert ablated.truth["unknown_eligible"] == (len(possible) > 1)


def test_cli_writes_ten_hashed_outputs_from_actual_pinned_input(tmp_path, capsys):
    assert main(
        [
            "--input",
            str(_FIXTURE),
            "--output-dir",
            str(tmp_path),
            "--fresh-seed",
            "20261101",
        ]
    ) == 0
    emitted = json.loads(capsys.readouterr().out)
    expected_variants = (
        "known",
        "fresh_known",
        "tier_ablated",
        "priority_forced_choice",
        "priority_with_unknown",
    )
    assert emitted["source"]["commit"] == UPSTREAM_COMMIT
    assert emitted["source"]["sha256"] == UPSTREAM_DATA_SHA256
    assert emitted["fresh_population"]["seed"] == 20261101
    assert emitted["fresh_population"]["novelty_or_exposure_claim"] == "none"
    assert emitted["model_calls"] == 0
    assert emitted["metrics_computed"] is False
    assert emitted["required_arms"] == ["json", "native"]
    assert emitted["required_model_pin"] is True
    assert emitted["required_unknown_metrics"] == [
        "forced_choice_error",
        "abstain_precision",
        "abstain_recall",
        "candidate_eligibility",
        "probability_mass_over_eligible_candidates",
        "invalid_responses",
    ]
    assert set(emitted["counts"]) == set(expected_variants)
    assert all(emitted["counts"][name] == 300 for name in expected_variants)
    expected_files = {
        f"{variant}{suffix}"
        for variant in expected_variants
        for suffix in (".jsonl", ".sealed-truth.jsonl")
    }
    assert set(emitted["files"]) == expected_files
    assert all(
        hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == digest
        for name, digest in emitted["files"].items()
    )


def test_fixture_loader_round_trips_question_payloads_through_bench_loader(tmp_path):
    main(
        [
            "--input",
            str(_FIXTURE),
            "--output-dir",
            str(tmp_path),
            "--fresh-seed",
            "4157",
        ]
    )
    loaded = load_fixture_inputs(tmp_path / "known.jsonl")
    assert len(loaded) == 300
    first_input = json.loads((tmp_path / "known.jsonl").read_text().splitlines()[0])
    questions_path = tmp_path / "questions.json"
    questions_path.write_text(json.dumps(first_input["questions"]), encoding="utf-8")
    from src.typed_decisions.bench import _load_questions

    assert loaded[0].questions == tuple(_load_questions(questions_path))
    assert loaded[0].state == first_input["state"]
    assert "source_observed_priority_label" not in first_input
