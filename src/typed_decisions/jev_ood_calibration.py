"""Zero-inference adapter for the pinned Jev OOD calibration dataset (TD-17).

The upstream file contains three typed rows per ticket. This module groups
those rows into the existing typed-decisions ``Question`` catalogue format,
adds a paired customer-tier ablation, and emits separate model inputs and
sealed truth. It does not call a model or compute calibration metrics.

Download ``data/val.jsonl`` from ``UPSTREAM_COMMIT`` before use. The adapter
refuses any byte stream other than the pinned CC0 data file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.typed_decisions.types import Question, QuestionKind

UPSTREAM_REPOSITORY = "https://github.com/scienthoon/jev-ood-calibration"
UPSTREAM_COMMIT = "914d87ab16517f14e833cff63d73b51235b14bd6"
UPSTREAM_DATA_PATH = "data/val.jsonl"
UPSTREAM_DATA_SHA256 = "763949cdf07f422cbb3c4bd040392944034ae4caf23a07de08fcef04cfd1eb33"
UPSTREAM_ORACLE_PATH = "data/generate.py"
UPSTREAM_ORACLE_SHA256 = "96c20efe7b82a4416f9ce86d15117edb539847f5b6da206e2a4f96b40c6622ac"
UPSTREAM_DATA_LICENSE = "CC0 (upstream README declaration)"

_ROW_COUNT = 900
_CASE_COUNT = 300
_FRESH_SOURCE_CASE_COUNT = 3000
_FRESH_VALIDATION_FRACTION = 0.1
_ROW_TYPES = ("choice", "score", "noul")
_LEVEL_NAMES = ("Low", "Normal", "High", "Critical")
_TIERS = ("free", "standard", "gold", "enterprise")
_UNKNOWN_LABEL = "unknown"
_LABEL_NOISE = 0.05

# Data-generation recipe copied from the MIT-licensed upstream oracle at
# UPSTREAM_COMMIT. This is a local deterministic implementation; downloaded
# upstream code is never imported or executed.
_SYNTH_TEMPLATES: dict[str, tuple[tuple[str, str, int], ...]] = {
    "billing": (
        (
            "Charged twice for order #{order}",
            "My card shows two charges of ${amount} for one order. {tail}",
            2,
        ),
        (
            "Refund not received",
            "I returned the item {days} days ago and the refund for ${amount} "
            "is still missing. {tail}",
            1,
        ),
        (
            "Wrong amount on invoice",
            "Invoice #{order} says ${amount} but the quote was lower. {tail}",
            1,
        ),
        (
            "구독 요금이 두 번 결제됐어요",
            "이번 달 카드 명세서에 {amount}달러가 두 번 찍혀 있습니다. {tail}",
            2,
        ),
        (
            "환불이 아직 안 들어왔습니다",
            "{days}일 전에 반품했는데 환불이 안 됐어요. {tail}",
            1,
        ),
    ),
    "shipping": (
        (
            "Package marked delivered but not here",
            "Tracking for order #{order} says delivered {days} days ago. Nothing arrived. {tail}",
            2,
        ),
        (
            "Where is my order?",
            "Order #{order} was placed {days} days ago and tracking has not updated. {tail}",
            1,
        ),
        (
            "Damaged on arrival",
            "The box for order #{order} was crushed and the item inside is broken. {tail}",
            2,
        ),
        (
            "배송이 너무 늦어요",
            "주문 #{order} 한 지 {days}일이 지났는데 아직 배송 준비중입니다. "
            "{tail}",
            1,
        ),
        (
            "배송 완료라는데 못 받았습니다",
            "송장은 배송 완료인데 집에 아무것도 없어요. {tail}",
            2,
        ),
    ),
    "technical": (
        (
            "App crashes on login",
            "Since the last update the app closes immediately after I enter my password. {tail}",
            2,
        ),
        (
            "Cannot reset password",
            "The reset email for my account never arrives, checked spam too. {tail}",
            1,
        ),
        (
            "Export button does nothing",
            "Clicking export on the reports page shows a spinner forever. {tail}",
            1,
        ),
        (
            "로그인이 안 됩니다",
            "비밀번호를 맞게 입력해도 오류 코드 {order}가 뜹니다. {tail}",
            2,
        ),
        (
            "앱이 계속 꺼져요",
            "업데이트 이후 사진을 열면 앱이 종료됩니다. {tail}",
            1,
        ),
    ),
    "general": (
        (
            "Question about your return policy",
            "How many days do I have to return an unopened item? {tail}",
            0,
        ),
        (
            "Do you ship to Canada?",
            "Planning to order as a gift, want to confirm shipping options first. {tail}",
            0,
        ),
        (
            "Feature suggestion",
            "It would be nice to sort the dashboard by date. Not urgent. {tail}",
            0,
        ),
        ("영업시간 문의", "고객센터 전화 상담은 몇 시까지인가요? {tail}", 0),
        (
            "제품 사양 질문",
            "이 모델이 해외 전압에서도 쓸 수 있는지 궁금합니다. {tail}",
            0,
        ),
    ),
}
_QUEUE_OPTIONS = {
    "billing": "Payments, refunds, duplicate charges",
    "shipping": "Delivery status, lost or damaged packages",
    "technical": "App or website bugs, login problems",
    "general": "Questions, feedback, anything else",
}
_TAILS_CALM = (
    "Thanks in advance.",
    "Let me know what you need from me.",
    "Appreciate any help.",
    "확인 부탁드립니다.",
    "답변 기다리겠습니다.",
    "",
)
_ANGRY_TAILS = (
    "This is unacceptable and I want it fixed today.",
    "I have contacted you three times already. Ridiculous.",
    "If this is not resolved I am disputing the charge and leaving a review.",
    "정말 화가 납니다. 당장 처리해 주세요.",
    "이게 몇 번째인지 모르겠네요. 진짜 실망입니다.",
)


def _subject_pattern(template: str) -> re.Pattern[str]:
    escaped = re.escape(template)
    replacements = {
        "{order}": r"\d+",
        "{amount}": r"\d+(?:\.\d+)?",
        "{days}": r"\d+",
    }
    for placeholder, expression in replacements.items():
        escaped = escaped.replace(re.escape(placeholder), expression)
    return re.compile(f"^{escaped}$")


_SUBJECT_URGENCY = tuple(
    (_subject_pattern(subject), urgency)
    for queue_templates in _SYNTH_TEMPLATES.values()
    for subject, _body, urgency in queue_templates
)


class AdapterError(ValueError):
    """The pinned dataset or its source-level contract is invalid."""


@dataclass(frozen=True)
class Fixture:
    """One paired input record; ``truth`` is written to a separate file."""

    fixture_id: str
    pair_id: str
    variant: str
    state: str
    questions: tuple[Question, ...]
    truth: Mapping[str, Any]


def load_pinned_rows(path: str | Path) -> tuple[dict[str, Any], ...]:
    """Read the exact pinned upstream JSONL without executing upstream code."""
    raw = Path(path).read_bytes()
    actual_sha = hashlib.sha256(raw).hexdigest()
    if actual_sha != UPSTREAM_DATA_SHA256:
        raise AdapterError(
            f"{UPSTREAM_DATA_PATH} sha256 mismatch: expected "
            f"{UPSTREAM_DATA_SHA256}, got {actual_sha}"
        )
    try:
        rows = tuple(json.loads(line) for line in raw.decode("utf-8").splitlines())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AdapterError(f"invalid pinned JSONL: {exc}") from exc
    if len(rows) != _ROW_COUNT:
        raise AdapterError(f"expected {_ROW_COUNT} source rows, got {len(rows)}")
    if any(not isinstance(row, dict) for row in rows):
        raise AdapterError("source JSONL rows must be objects")
    return rows


def generate_fresh_rows(seed: int) -> tuple[dict[str, Any], ...]:
    """Reproduce the pinned generator's seeded 300-ticket validation split."""
    rng = random.Random(seed)
    queues = tuple(_SYNTH_TEMPLATES)
    records: list[dict[str, Any]] = []
    for _ in range(_FRESH_SOURCE_CASE_COUNT):
        queue = rng.choice(queues)
        subject_template, body_template, urgency = rng.choice(_SYNTH_TEMPLATES[queue])
        angry = rng.random() < 0.3
        tail = rng.choice(_ANGRY_TAILS if angry else _TAILS_CALM)
        tier = rng.choice(_TIERS)
        channel = rng.choice(("email", "chat", "phone", "app"))
        fill = {
            "order": rng.randint(1000, 9999),
            "amount": rng.choice((19.99, 49.00, 89.99, 120.50, 300.00)),
            "days": rng.randint(1, 21),
            "tail": tail,
        }
        state = {
            "channel": channel,
            "customer_tier": tier,
            "subject": subject_template.format(**fill),
            "body": body_template.format(**fill).strip(),
        }

        priority = urgency + int(angry)
        if tier in {"gold", "enterprise"} and priority >= 1:
            priority += 1
        priority = max(0, min(len(_LEVEL_NAMES) - 1, priority))
        queue_label: str = queue
        angry_label = angry
        if rng.random() < _LABEL_NOISE:
            queue_label = rng.choice(tuple(name for name in queues if name != queue))
        if rng.random() < _LABEL_NOISE:
            priority = rng.randint(0, len(_LEVEL_NAMES) - 1)
        if rng.random() < _LABEL_NOISE:
            angry_label = not angry

        records.extend(
            (
                {
                    "state": state,
                    "type": "choice",
                    "question": "Which support queue should handle this ticket?",
                    "options": dict(_QUEUE_OPTIONS),
                    "label": queue_label,
                },
                {
                    "state": state,
                    "type": "score",
                    "question": "How should this ticket be prioritized?",
                    "levels": list(_LEVEL_NAMES),
                    "label": priority,
                },
                {
                    "state": state,
                    "type": "noul",
                    "question": "The customer sounds angry.",
                    "label": angry_label,
                },
            )
        )
    groups = [records[offset : offset + 3] for offset in range(0, len(records), 3)]
    split_rng = random.Random(seed)
    split_rng.shuffle(groups)
    validation_count = round(len(groups) * _FRESH_VALIDATION_FRACTION)
    return tuple(row for group in groups[:validation_count] for row in group)


def build_fixtures(
    frozen_rows: Sequence[Mapping[str, Any]],
    fresh_rows: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[Fixture, ...]]:
    """Build frozen-known and matched fresh-known/tier-ablated fixture arms."""
    frozen_groups = _group_rows(frozen_rows, source="frozen")
    fresh_groups = _group_rows(fresh_rows, source="fresh")
    if len(frozen_groups) != _CASE_COUNT or len(fresh_groups) != _CASE_COUNT:
        raise AdapterError(f"expected {_CASE_COUNT} tickets in each population")

    result: dict[str, list[Fixture]] = {
        "known": [],
        "fresh_known": [],
        "tier_ablated": [],
        "priority_forced_choice": [],
        "priority_with_unknown": [],
    }

    for index, group in enumerate(frozen_groups):
        choice, score, noul = group
        case_id = f"frozen-{index:03d}"
        state = dict(choice["state"])
        labels = _source_labels(case_id, choice, score, noul)
        result["known"].append(
            Fixture(
                fixture_id=f"td17-{case_id}-known",
                pair_id=f"td17-{case_id}",
                variant="known",
                state=_state_json(state),
                questions=_base_questions(case_id, choice, score, noul),
                truth=_case_truth(labels, score, state),
            )
        )

    for index, group in enumerate(fresh_groups):
        choice, score, noul = group
        case_id = f"fresh-{index:03d}"
        pair_id = f"td17-{case_id}"
        source_state = dict(choice["state"])
        visible_state = dict(source_state)
        hidden_tier = visible_state.pop("customer_tier")
        labels = _source_labels(case_id, choice, score, noul)
        clean_priority = _clean_priority_for_tier(visible_state, hidden_tier)
        possible = _possible_priority_values(visible_state)
        if not possible:
            raise AdapterError(f"no oracle priority values derived for {case_id}")

        result["fresh_known"].append(
            Fixture(
                fixture_id=f"{pair_id}-fresh-known",
                pair_id=pair_id,
                variant="fresh_known",
                state=_state_json(source_state),
                questions=_base_questions(case_id, choice, score, noul),
                truth=_case_truth(labels, score, source_state),
            )
        )

        oracle_truth = {
            **_case_truth(labels, score, source_state),
            "hidden_customer_tier": hidden_tier,
            "clean_oracle_priority": clean_priority,
            "source_observed_priority_label": score["label"],
            "source_label_noise_disagreement": score["label"] != clean_priority,
            "source_observed_priority_is_in_ablated_eligible_set": score["label"] in possible,
            "possible_priority_values": sorted(possible),
            "possible_priority_names": [_LEVEL_NAMES[value] for value in sorted(possible)],
            "unknown_eligible": len(possible) > 1,
            "unknown_option_eligible": len(possible) > 1,
            "unknown_expected": len(possible) > 1,
            "oracle": _oracle_identity(),
        }
        result["tier_ablated"].append(
            Fixture(
                fixture_id=f"{pair_id}-tier-ablated",
                pair_id=pair_id,
                variant="tier_ablated",
                state=_state_json(visible_state),
                questions=_base_questions(case_id, choice, score, noul),
                truth=oracle_truth,
            )
        )
        forced_question = _priority_choice_question(case_id, score, allow_unknown=False)
        unknown_question = _priority_choice_question(case_id, score, allow_unknown=True)
        eligible_names = [_LEVEL_NAMES[value] for value in sorted(possible)]
        singleton_name = eligible_names[0] if len(eligible_names) == 1 else None
        result["priority_forced_choice"].append(
            Fixture(
                fixture_id=f"{pair_id}-forced-choice",
                pair_id=pair_id,
                variant="priority_forced_choice",
                state=_state_json(visible_state),
                questions=(forced_question,),
                truth={
                    **oracle_truth,
                    "benchmark_source_observed_choice": _LEVEL_NAMES[score["label"]],
                    "benchmark_source_observed_label": score["label"],
                    "structural_singleton_expected_choice": singleton_name,
                    "candidate_eligible": eligible_names,
                },
            )
        )
        result["priority_with_unknown"].append(
            Fixture(
                fixture_id=f"{pair_id}-with-unknown",
                pair_id=pair_id,
                variant="priority_with_unknown",
                state=_state_json(visible_state),
                questions=(unknown_question,),
                truth={
                    **oracle_truth,
                    "benchmark_source_observed_choice": _LEVEL_NAMES[score["label"]],
                    "benchmark_source_observed_label": score["label"],
                    "structural_expected_choice": (
                        _UNKNOWN_LABEL if len(possible) > 1 else singleton_name
                    ),
                    "candidate_eligible": eligible_names
                    + ([_UNKNOWN_LABEL] if len(possible) > 1 else []),
                },
            )
        )

    return {name: tuple(fixtures) for name, fixtures in result.items()}


def _group_rows(
    rows: Sequence[Mapping[str, Any]], *, source: str
) -> tuple[tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]], ...]:
    if len(rows) != _ROW_COUNT or len(rows) % 3:
        raise AdapterError(f"{source}: expected {_ROW_COUNT} rows in 300 source triples")
    groups = []
    for offset in range(0, len(rows), 3):
        group = rows[offset : offset + 3]
        if tuple(row.get("type") for row in group) != _ROW_TYPES:
            raise AdapterError(f"{source} rows {offset}:{offset + 3} are not choice/score/noul")
        state = group[0].get("state")
        if not isinstance(state, dict) or any(row.get("state") != state for row in group):
            raise AdapterError(f"{source} rows {offset}:{offset + 3} do not share one state")
        if set(state) != {"channel", "customer_tier", "subject", "body"}:
            raise AdapterError(f"{source} rows {offset}:{offset + 3} have unexpected state fields")
        groups.append((group[0], group[1], group[2]))
    return tuple(groups)


def _source_labels(
    case_id: str,
    choice: Mapping[str, Any],
    score: Mapping[str, Any],
    noul: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        f"{case_id}-queue": choice["label"],
        f"{case_id}-priority": score["label"],
        f"{case_id}-angry": noul["label"],
    }


def _case_truth(
    labels: Mapping[str, Any], score: Mapping[str, Any], state: Mapping[str, Any]
) -> dict[str, Any]:
    clean_priority = _clean_priority_for_tier(state, state["customer_tier"])
    observed_priority = score["label"]
    return {
        "source_observed_labels": dict(labels),
        "source_observed_priority_label": observed_priority,
        "clean_oracle_priority": clean_priority,
        "source_label_noise_disagreement": observed_priority != clean_priority,
        "oracle": _oracle_identity(),
    }


def _oracle_identity() -> dict[str, str]:
    return {
        "repository": UPSTREAM_REPOSITORY,
        "commit": UPSTREAM_COMMIT,
        "path": UPSTREAM_ORACLE_PATH,
        "sha256": UPSTREAM_ORACLE_SHA256,
    }


def write_fixture_bundle(
    input_path: str | Path, *, output_dir: str | Path, fresh_seed: int
) -> dict[str, Any]:
    """Verify frozen inputs and write the frozen and fresh-population fixtures."""
    frozen_rows = load_pinned_rows(input_path)
    fresh_rows = generate_fresh_rows(fresh_seed)
    return _write_fixture_bundle(
        frozen_rows,
        fresh_rows,
        output_dir=output_dir,
        fresh_seed=fresh_seed,
    )


def _write_fixture_bundle(
    frozen_rows: Sequence[Mapping[str, Any]],
    fresh_rows: Sequence[Mapping[str, Any]],
    *,
    output_dir: str | Path,
    fresh_seed: int,
) -> dict[str, Any]:
    """Serialize verified populations; public callers use the pin-checking wrapper."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    fixtures = build_fixtures(frozen_rows, fresh_rows)
    frozen_states = {
        _state_json(group[0]["state"])
        for group in _group_rows(frozen_rows, source="frozen")
    }
    fresh_states = {
        _state_json(group[0]["state"])
        for group in _group_rows(fresh_rows, source="fresh")
    }
    fresh_bytes = _jsonl_bytes(fresh_rows)
    manifest: dict[str, Any] = {
        "schema": "typed_decisions.td17_fixture_bundle.v1",
        "source": {
            "repository": UPSTREAM_REPOSITORY,
            "commit": UPSTREAM_COMMIT,
            "path": UPSTREAM_DATA_PATH,
            "sha256": UPSTREAM_DATA_SHA256,
            "license": UPSTREAM_DATA_LICENSE,
            "oracle_path": UPSTREAM_ORACLE_PATH,
            "oracle_sha256": UPSTREAM_ORACLE_SHA256,
        },
        "fresh_population": {
            "ticket_count": _CASE_COUNT,
            "seed": fresh_seed,
            "recipe_commit": UPSTREAM_COMMIT,
            "recipe_path": UPSTREAM_ORACLE_PATH,
            "recipe_sha256": UPSTREAM_ORACLE_SHA256,
            "source_population_ticket_count": _FRESH_SOURCE_CASE_COUNT,
            "validation_fraction": _FRESH_VALIDATION_FRACTION,
            "split_seed": fresh_seed,
            "serialized_rows_sha256": hashlib.sha256(fresh_bytes).hexdigest(),
            "unique_state_count": len(fresh_states),
            "exact_state_overlap_with_frozen": len(fresh_states & frozen_states),
            "label_noise_rate": _LABEL_NOISE,
            "recipe": "build 3000 tickets, then seeded group split at 10%; take validation groups",
            "novelty_or_exposure_claim": "none",
        },
        "source_rows": _ROW_COUNT,
        "ticket_states": _CASE_COUNT,
        "counts": {name: len(items) for name, items in fixtures.items()},
        "catalogue_rule_disclosed": False,
        "model_calls": 0,
        "metrics_computed": False,
        "required_calibration_metrics": [
            "accuracy",
            "mean_max_probability",
            "ece",
            "brier",
            "nll",
            "perfect_calibration_ece_noise_floor",
            "template_clustered_paired_bootstrap",
            "tie_aware_coverage_and_aurc",
            "refit_temperature_floor_and_clamp_sensitivity",
            "exact_zero_and_one_rates",
            "temperature_search_bound_flag",
            "group_disjoint_out_of_fold_check",
        ],
        "required_model_pin": True,
        "required_arms": ["json", "native"],
        "required_unknown_metrics": [
            "forced_choice_error",
            "abstain_precision",
            "abstain_recall",
            "candidate_eligibility",
            "probability_mass_over_eligible_candidates",
            "invalid_responses",
        ],
        "metric_guard": (
            "derive only from resolved outputs and sealed truth; confidence "
            "separation is not abstention quality"
        ),
        "files": {},
    }
    for name, items in fixtures.items():
        inputs_path = root / f"{name}.jsonl"
        truth_path = root / f"{name}.sealed-truth.jsonl"
        _write_jsonl(
            inputs_path,
            (
                {
                    "fixture_id": fixture.fixture_id,
                    "pair_id": fixture.pair_id,
                    "variant": fixture.variant,
                    "state": fixture.state,
                    "questions": [_question_payload(q) for q in fixture.questions],
                }
                for fixture in items
            ),
        )
        _write_jsonl(
            truth_path,
            (
                {"fixture_id": fixture.fixture_id, **dict(fixture.truth)}
                for fixture in items
            ),
        )
        manifest["files"][inputs_path.name] = _sha256_file(inputs_path)
        manifest["files"][truth_path.name] = _sha256_file(truth_path)
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


@dataclass(frozen=True)
class FixtureInput:
    """Model-facing input compatible with ``run_mode_benchmark``."""

    fixture_id: str
    pair_id: str
    variant: str
    state: str
    questions: tuple[Question, ...]


def load_fixture_inputs(path: str | Path) -> tuple[FixtureInput, ...]:
    """Load one JSONL arm as typed inputs for ``run_mode_benchmark``.

    The bundle JSONL question objects use the same fields accepted by
    ``bench._load_questions``. Sealed truth is deliberately not read here.
    """
    fixtures: list[FixtureInput] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AdapterError(f"{path}:{line_number}: invalid JSONL: {exc}") from exc
        if not isinstance(row, dict) or not isinstance(row.get("questions"), list):
            raise AdapterError(f"{path}:{line_number}: expected a fixture with questions")
        if not isinstance(row.get("state"), str):
            raise AdapterError(f"{path}:{line_number}: state must be a JSON string")
        questions = tuple(
            _question_from_payload(item, path, line_number)
            for item in row["questions"]
        )
        fixtures.append(
            FixtureInput(
                fixture_id=str(row["fixture_id"]),
                pair_id=str(row["pair_id"]),
                variant=str(row["variant"]),
                state=row["state"],
                questions=questions,
            )
        )
    return tuple(fixtures)


def _question_from_payload(item: Any, path: str | Path, line_number: int) -> Question:
    if not isinstance(item, Mapping):
        raise AdapterError(f"{path}:{line_number}: question must be an object")
    try:
        return Question(
            id=str(item["id"]),
            kind=item["kind"],
            text=str(item["text"]),
            options=tuple(item.get("options", ())),
            levels=tuple(item.get("levels", ())),
            criteria=tuple(item.get("criteria", ())),
            option_descriptions=item.get("option_descriptions", ()),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise AdapterError(f"{path}:{line_number}: invalid question: {exc}") from exc


def _base_questions(
    case_id: str,
    choice: Mapping[str, Any],
    score: Mapping[str, Any],
    noul: Mapping[str, Any],
) -> tuple[Question, ...]:
    options = choice.get("options")
    levels = score.get("levels")
    expected_queues = {"billing", "shipping", "technical", "general"}
    if not isinstance(options, dict) or set(options) != expected_queues:
        raise AdapterError(f"{case_id}: unexpected queue-choice options")
    if not isinstance(levels, list) or tuple(levels) != _LEVEL_NAMES:
        raise AdapterError(f"{case_id}: unexpected priority level names")
    if not isinstance(score.get("label"), int) or isinstance(score["label"], bool):
        raise AdapterError(f"{case_id}: priority label must be an integer")
    if score["label"] not in range(4):
        raise AdapterError(f"{case_id}: priority label outside 0..3")
    if not isinstance(noul.get("label"), bool):
        raise AdapterError(f"{case_id}: angry label must be boolean")
    if not isinstance(choice.get("label"), str) or choice["label"] not in options:
        raise AdapterError(f"{case_id}: queue label is not an option")
    return (
        Question(
            id=f"{case_id}-queue",
            kind=QuestionKind.CHOICE,
            text=str(choice["question"]),
            options=tuple(options),
            option_descriptions=tuple(options[key] for key in options),
        ),
        Question(
            id=f"{case_id}-priority",
            kind=QuestionKind.SCORE,
            text=str(score["question"]),
            levels=(0, 1, 2, 3),
            criteria=tuple(
                f"{level}: {name}" for level, name in enumerate(_LEVEL_NAMES)
            ),
        ),
        Question(
            id=f"{case_id}-angry",
            kind=QuestionKind.NOUL,
            text=str(noul["question"]),
        ),
    )


def _priority_choice_question(
    case_id: str, score: Mapping[str, Any], *, allow_unknown: bool
) -> Question:
    options = (*_LEVEL_NAMES, *((_UNKNOWN_LABEL,) if allow_unknown else ()))
    return Question(
        id=f"{case_id}-priority-choice",
        kind=QuestionKind.CHOICE,
        text=str(score["question"]),
        options=options,
    )


def _base_urgency(state: Mapping[str, Any]) -> int:
    subject = state.get("subject")
    if not isinstance(subject, str):
        raise AdapterError("oracle needs a string subject field")
    urgency = next(
        (value for pattern, value in _SUBJECT_URGENCY if pattern.fullmatch(subject)),
        None,
    )
    if urgency is None:
        raise AdapterError("subject does not match a template in the pinned oracle")
    return urgency


def _is_angry(state: Mapping[str, Any]) -> bool:
    body = state.get("body")
    if not isinstance(body, str):
        raise AdapterError("oracle needs a string body field")
    return any(body.endswith(tail) for tail in _ANGRY_TAILS)


def _clean_priority_for_tier(state: Mapping[str, Any], tier: Any) -> int:
    if tier not in _TIERS:
        raise AdapterError(f"unknown customer tier: {tier!r}")
    pre_tier = _base_urgency(state) + int(_is_angry(state))
    return min(
        len(_LEVEL_NAMES) - 1,
        pre_tier + int(tier in {"gold", "enterprise"} and pre_tier >= 1),
    )


def _possible_priority_values(state: Mapping[str, Any]) -> set[int]:
    pre_tier = _base_urgency(state) + int(_is_angry(state))
    return {
        min(
            len(_LEVEL_NAMES) - 1,
            pre_tier + int(tier in {"gold", "enterprise"} and pre_tier >= 1),
        )
        for tier in _TIERS
    }


def _question_payload(question: Question) -> dict[str, Any]:
    return {
        "id": question.id,
        "kind": question.kind.value,
        "text": question.text,
        "options": list(question.options),
        "levels": list(question.levels),
        "criteria": list(question.criteria),
        "option_descriptions": list(question.option_descriptions),
    }


def _state_json(state: Mapping[str, Any]) -> str:
    return json.dumps(state, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.write_bytes(_jsonl_bytes(rows))


def _jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in rows
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="pinned upstream data/val.jsonl")
    parser.add_argument("--output-dir", required=True, help="new fixture bundle directory")
    parser.add_argument(
        "--fresh-seed", required=True, type=int,
        help="seed for a new 300-ticket population from the pinned generator recipe",
    )
    args = parser.parse_args(argv)
    manifest = write_fixture_bundle(
        args.input, output_dir=args.output_dir, fresh_seed=args.fresh_seed
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
