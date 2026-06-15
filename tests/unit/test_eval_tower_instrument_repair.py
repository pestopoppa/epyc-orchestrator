"""Evidence-plane W4 instrument repair coverage."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "autopilot"))

import eval_tower  # noqa: E402
from eval_tower import EvalTower, QuestionResult  # noqa: E402


class _PrefixRng:
    def sample(self, population, k):  # noqa: ANN001
        return list(population[:k])


def test_programmatic_scorer_runs_with_empty_expected(monkeypatch) -> None:
    tower = EvalTower()

    def _fake_call(**_kwargs):  # noqa: ANN001
        return {
            "answer": "This response is intentionally non-empty.",
            "tokens_generated": 7,
            "model": "fake",
        }

    monkeypatch.setattr(eval_tower, "call_orchestrator_forced", _fake_call)

    with eval_tower.httpx.Client(timeout=1) as client:
        result = tower._eval_question(
            {
                "id": "ifeval-empty-expected",
                "suite": "instruction_precision",
                "prompt": "Write any non-empty answer.",
                "expected": "",
                "scoring_method": "programmatic",
                "scoring_config": {"verifier": "non_empty"},
            },
            client,
        )

    assert result.correct is True


def test_empty_expected_still_blocks_plain_exact_match(monkeypatch) -> None:
    tower = EvalTower()

    def _fake_call(**_kwargs):  # noqa: ANN001
        return {
            "answer": "A non-empty answer must not auto-pass.",
            "tokens_generated": 7,
            "model": "fake",
        }

    monkeypatch.setattr(eval_tower, "call_orchestrator_forced", _fake_call)

    with eval_tower.httpx.Client(timeout=1) as client:
        result = tower._eval_question(
            {
                "id": "exact-empty-expected",
                "suite": "general",
                "prompt": "Write any non-empty answer.",
                "expected": "",
                "scoring_method": "exact_match",
            },
            client,
        )

    assert result.correct is False


def test_empty_expected_text_scorer_is_not_scoreable() -> None:
    assert not eval_tower._is_scoreable_question(
        {
            "id": "dead-text",
            "expected": "",
            "scoring_method": "substring",
        }
    )
    assert eval_tower._is_scoreable_question(
        {
            "id": "expected-free-programmatic",
            "expected": "",
            "scoring_method": "programmatic",
        }
    )


def test_code_execution_requires_executable_oracle() -> None:
    assert not eval_tower._is_scoreable_question(
        {
            "id": "code-without-tests",
            "expected": "def solve",
            "scoring_method": "code_execution",
            "scoring_config": {"language": "python"},
        }
    )
    assert not eval_tower._is_scoreable_question(
        {
            "id": "commented-asserts",
            "expected": "",
            "scoring_method": "code_execution",
            "scoring_config": {
                "language": "python",
                "test_code": "# assert add(1, 2) == 3\n",
            },
        }
    )
    assert eval_tower._is_scoreable_question(
        {
            "id": "stdin-tests",
            "expected": "",
            "scoring_method": "code_execution",
            "scoring_config": {
                "language": "python",
                "test_code": "TEST_CASES = [('1\\n', '1\\n')]",
            },
        }
    )


def test_sampling_replaces_unscoreable_items_from_same_suite() -> None:
    suite_qs = [
        {
            "id": "dead-text",
            "expected": "",
            "scoring_method": "substring",
        },
        {
            "id": "valid-text",
            "expected": "answer",
            "scoring_method": "substring",
        },
        {
            "id": "valid-programmatic",
            "expected": "",
            "scoring_method": "programmatic",
        },
    ]

    sample = eval_tower._sample_scoreable_questions(
        "instruction_precision",
        suite_qs,
        per_suite=2,
        rng=_PrefixRng(),
    )

    assert [q["id"] for q in sample] == ["valid-text", "valid-programmatic"]


def test_eval_sample_globally_backfills_when_suite_underfills() -> None:
    pool = {
        "usaco": [
            {
                "id": "dead-usaco-a",
                "suite": "usaco",
                "expected": "",
                "scoring_method": "code_execution",
                "scoring_config": {"language": "python"},
            },
            {
                "id": "dead-usaco-b",
                "suite": "usaco",
                "expected": "",
                "scoring_method": "substring",
            },
        ],
        "math": [
            {
                "id": f"valid-math-{i}",
                "suite": "math",
                "expected": str(i),
                "scoring_method": "exact_match",
            }
            for i in range(6)
        ],
    }

    sample = eval_tower._sample_scoreable_eval_questions(
        pool,
        n=4,
        rng=random.Random(7),
    )

    assert len(sample) == 4
    assert len({id(q) for q in sample}) == 4
    assert all(eval_tower._is_scoreable_question(q) for q in sample)
    assert {q["suite"] for q in sample if "suite" in q}.isdisjoint({"usaco"})
    assert not {q["id"] for q in sample} & {"dead-usaco-a", "dead-usaco-b"}


def test_eval_question_records_chat_response_routed_to(monkeypatch) -> None:
    tower = EvalTower()

    def _fake_call(**_kwargs):  # noqa: ANN001
        return {
            "answer": "FRIEND",
            "routed_to": "worker_vision",
            "tokens_generated": 1,
        }

    monkeypatch.setattr(eval_tower, "call_orchestrator_forced", _fake_call)

    with eval_tower.httpx.Client(timeout=1) as client:
        result = tower._eval_question(
            {
                "id": "vl-route-telemetry",
                "suite": "vl",
                "prompt": "what is written in the image?",
                "expected": "FRIEND",
                "scoring_method": "exact_match",
                "image_path": "/tmp/example.png",
            },
            client,
        )

    assert result.route_used == "worker_vision"


def test_t0_sentinel_suites_are_namespaced_without_mutating_source(monkeypatch) -> None:
    sentinels = [
        {
            "id": "sentinel-a",
            "suite": "general",
            "prompt": "A",
            "expected": "A",
            "scoring_method": "exact_match",
        },
        {
            "id": "sentinel-b",
            "suite": "math",
            "prompt": "B",
            "expected": "B",
            "scoring_method": "exact_match",
        },
    ]
    tower = EvalTower()
    tower._sentinels = sentinels

    def _fake_eval_batch(self, questions, client, **_kwargs):  # noqa: ANN001, ARG001
        return [
            QuestionResult(
                question_id=q["id"],
                suite=q["suite"],
                prompt=q["prompt"],
                expected=q["expected"],
                correct=q["id"] == "sentinel-a",
                tokens_generated=1,
                elapsed_s=1.0,
            )
            for q in questions
        ]

    monkeypatch.setattr(EvalTower, "_eval_batch", _fake_eval_batch)

    result = tower.eval_t0()

    assert result.per_suite_quality == {
        "sentinel_general": 3.0,
        "sentinel_math": 0.0,
    }
    assert sentinels[0]["suite"] == "general"
    assert sentinels[1]["suite"] == "math"


def test_eval_t1_uses_designed_core_when_enabled(tmp_path, monkeypatch) -> None:
    core_path = tmp_path / "core_v2.jsonl"
    rows = [
        {"__core_metadata__": True, "core_id": "core_v2", "source": "unit"},
        {
            "id": "core-a",
            "suite": "math",
            "prompt": "2+2?",
            "expected": "4",
            "scoring_method": "exact_match",
        },
        {
            "id": "core-b",
            "suite": "coder",
            "prompt": "Return x",
            "expected": "x",
            "scoring_method": "exact_match",
        },
    ]
    core_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    monkeypatch.setenv("AUTOPILOT_T1_CORE_ID", "core_v2")
    monkeypatch.setenv("AUTOPILOT_T1_CORE_PATH", str(core_path))

    captured = {}

    def _fake_eval_batch(self, questions, client, **_kwargs):  # noqa: ANN001, ARG001
        captured["ids"] = [q["id"] for q in questions]
        return [
            QuestionResult(
                question_id=q["id"],
                suite=q["suite"],
                prompt=q["prompt"],
                expected=q["expected"],
                correct=True,
                tokens_generated=1,
                elapsed_s=1.0,
            )
            for q in questions
        ]

    monkeypatch.setattr(EvalTower, "_eval_batch", _fake_eval_batch)

    result = EvalTower().eval_t1(n=999, seed=123)

    assert captured["ids"] == ["core-a", "core-b"]
    assert result.core_id == "core_v2"
    assert result.n_questions == 2
    assert result.details["core_selection"] == "designed_core"
    assert result.details["core_metadata"]["source"] == "unit"
    assert result.details["requested_n"] == 999
    assert result.details["base_core_questions"] == 2


def test_eval_t1_missing_designed_core_fails_closed(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AUTOPILOT_T1_CORE_ID", "core_v2")
    monkeypatch.setenv("AUTOPILOT_T1_CORE_PATH", str(tmp_path / "missing.jsonl"))

    result = EvalTower().eval_t1(n=50, seed=42)

    assert result.core_id == "core_v2"
    assert result.quality == 0
    assert result.reliability == 0
    assert result.details["core_selection"] == "designed_core"
    assert result.details["core_path"] == str(tmp_path / "missing.jsonl")
    assert "not found" in result.details["core_error"]


def test_eval_t1_core_path_without_core_id_fails_closed(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("AUTOPILOT_T1_CORE_ID", raising=False)
    monkeypatch.setenv("AUTOPILOT_T1_CORE_PATH", str(tmp_path / "core_v2.jsonl"))

    result = EvalTower().eval_t1(n=50, seed=42)

    assert result.quality == 0
    assert result.reliability == 0
    assert result.details["core_selection"] == "designed_core"
    assert "requires AUTOPILOT_T1_CORE_ID" in result.details["core_error"]


def test_eval_t1_designed_core_rejects_metadata_mismatch(tmp_path, monkeypatch) -> None:
    core_path = tmp_path / "core_v2.jsonl"
    rows = [
        {"__core_metadata__": True, "core_id": "wrong_core"},
        {
            "id": "core-a",
            "suite": "math",
            "prompt": "2+2?",
            "expected": "4",
            "scoring_method": "exact_match",
        },
    ]
    core_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    monkeypatch.setenv("AUTOPILOT_T1_CORE_ID", "core_v2")
    monkeypatch.setenv("AUTOPILOT_T1_CORE_PATH", str(core_path))

    result = EvalTower().eval_t1(n=50, seed=42)

    assert result.core_id == "core_v2"
    assert result.quality == 0
    assert result.reliability == 0
    assert "does not match requested" in result.details["core_error"]


def test_eval_t1_designed_core_rejects_unscoreable_rows(tmp_path, monkeypatch) -> None:
    core_path = tmp_path / "core_v2.jsonl"
    rows = [
        {"__core_metadata__": True, "core_id": "core_v2"},
        {
            "id": "dead-core-row",
            "suite": "general",
            "prompt": "Write anything.",
            "expected": "",
            "scoring_method": "substring",
        },
    ]
    core_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    monkeypatch.setenv("AUTOPILOT_T1_CORE_ID", "core_v2")
    monkeypatch.setenv("AUTOPILOT_T1_CORE_PATH", str(core_path))

    result = EvalTower().eval_t1(n=50, seed=42)

    assert result.core_id == "core_v2"
    assert result.quality == 0
    assert result.reliability == 0
    assert "unscoreable" in result.details["core_error"]


def test_eval_t1_designed_core_can_reference_question_pool_ids(tmp_path, monkeypatch) -> None:
    core_path = tmp_path / "core_v2.jsonl"
    rows = [
        {"__core_metadata__": True, "core_id": "core_v2"},
        {"id": "math/q-from-pool"},
    ]
    core_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    monkeypatch.setenv("AUTOPILOT_T1_CORE_ID", "core_v2")
    monkeypatch.setenv("AUTOPILOT_T1_CORE_PATH", str(core_path))
    tower = EvalTower()
    tower._pool = {
        "math": [
            {
                "id": "q-from-pool",
                "suite": "math",
                "prompt": "3+4?",
                "expected": "7",
                "scoring_method": "exact_match",
            }
        ]
    }

    captured = {}

    def _fake_eval_batch(self, questions, client, **_kwargs):  # noqa: ANN001, ARG001
        captured["ids"] = [q["id"] for q in questions]
        return [
            QuestionResult(
                question_id=q["id"],
                suite=q["suite"],
                prompt=q["prompt"],
                expected=q["expected"],
                correct=True,
                tokens_generated=1,
                elapsed_s=1.0,
            )
            for q in questions
        ]

    monkeypatch.setattr(EvalTower, "_eval_batch", _fake_eval_batch)

    result = tower.eval_t1()

    assert captured["ids"] == ["q-from-pool"]
    assert result.core_id == "core_v2"
    assert result.details["base_core_questions"] == 1


def test_eval_t1_legacy_sampling_records_core_id(monkeypatch) -> None:
    monkeypatch.delenv("AUTOPILOT_T1_CORE_ID", raising=False)
    monkeypatch.delenv("AUTOPILOT_T1_CORE_PATH", raising=False)
    tower = EvalTower()
    tower._pool = {
        "math": [
            {
                "id": "legacy-math",
                "suite": "math",
                "prompt": "1+1?",
                "expected": "2",
                "scoring_method": "exact_match",
            }
        ],
        "coder": [
            {
                "id": "legacy-coder",
                "suite": "coder",
                "prompt": "Return x",
                "expected": "x",
                "scoring_method": "exact_match",
            }
        ],
    }

    def _fake_eval_batch(self, questions, client, **_kwargs):  # noqa: ANN001, ARG001
        return [
            QuestionResult(
                question_id=q["id"],
                suite=q["suite"],
                prompt=q["prompt"],
                expected=q["expected"],
                correct=True,
                tokens_generated=1,
                elapsed_s=1.0,
            )
            for q in questions
        ]

    monkeypatch.setattr(EvalTower, "_eval_batch", _fake_eval_batch)

    result = tower.eval_t1(n=2, seed=42)

    assert result.core_id == "legacy_pool_seed_42_n2"
    assert result.details["core_selection"] == "legacy_pool_seed"
    assert result.details["base_core_questions"] == 2


def test_eval_t1_w6_audit_block_appends_trial_seeded_questions(
    tmp_path,
    monkeypatch,
) -> None:
    core_path = tmp_path / "core_v2.jsonl"
    rows = [
        {"__core_metadata__": True, "core_id": "core_v2"},
        {
            "id": "core-a",
            "suite": "math",
            "prompt": "2+2?",
            "expected": "4",
            "scoring_method": "exact_match",
        },
    ]
    core_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    monkeypatch.setenv("AUTOPILOT_T1_CORE_ID", "core_v2")
    monkeypatch.setenv("AUTOPILOT_T1_CORE_PATH", str(core_path))
    monkeypatch.setenv("AUTOPILOT_W6_AUDIT_BLOCK", "1")
    monkeypatch.setenv("AUTOPILOT_W6_AUDIT_N", "2")
    tower = EvalTower()
    tower._pool = {
        "math": [
            {
                "id": "audit-math",
                "suite": "math",
                "prompt": "3+4?",
                "expected": "7",
                "scoring_method": "exact_match",
            },
        ],
        "coder": [
            {
                "id": "audit-coder",
                "suite": "coder",
                "prompt": "Return x",
                "expected": "x",
                "scoring_method": "exact_match",
            },
        ],
    }

    captured = []

    def _fake_eval_batch(self, questions, client, **_kwargs):  # noqa: ANN001, ARG001
        captured.extend((q["id"], q["eval_partition"]) for q in questions)
        return [
            QuestionResult(
                question_id=q["id"],
                suite=q["suite"],
                prompt=q["prompt"],
                expected=q["expected"],
                correct=q["eval_partition"] == "core",
                tokens_generated=1,
                elapsed_s=1.0,
                eval_partition=q["eval_partition"],
            )
            for q in questions
        ]

    monkeypatch.setattr(EvalTower, "_eval_batch", _fake_eval_batch)

    result = tower.eval_t1(n=999, seed=123, trial_id=17)

    assert captured[0] == ("core-a", "core")
    assert sorted(captured[1:]) == [
        ("audit-coder", "audit"),
        ("audit-math", "audit"),
    ]
    assert result.core_id == "core_v2"
    assert result.n_questions == 1
    assert result.quality == 3.0
    assert result.details["base_core_questions"] == 1
    assert result.details["base_audit_questions"] == 2
    assert result.details["audit_policy"]["active"] is True
    assert result.details["audit_policy"]["shadow_only"] is True
    assert result.details["audit_policy"]["trial_id"] == 17
    assert result.details["audit_policy"]["actual_n"] == 2
    assert result.details["audit_shadow_only"] is True
    assert result.details["audit_shadow_total_n_questions"] == 3
    assert result.details["audit_shadow_decision_n_questions"] == 1
    assert result.details["partition_counts"] == {"core": 1, "audit": 2}
    assert result.details["partition_quality"] == {"core": 3.0, "audit": 0.0}
    assert len(result.question_results) == 3
    audit_rows = [row for row in result.question_results if row["partition"] == "audit"]
    assert {row["suite"] for row in audit_rows} == {"math", "coder"}


def test_eval_t1_w6_audit_block_can_count_audit_in_decision_metrics(
    tmp_path,
    monkeypatch,
) -> None:
    core_path = tmp_path / "core_v2.jsonl"
    rows = [
        {"__core_metadata__": True, "core_id": "core_v2"},
        {
            "id": "core-a",
            "suite": "math",
            "prompt": "2+2?",
            "expected": "4",
            "scoring_method": "exact_match",
        },
    ]
    core_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    monkeypatch.setenv("AUTOPILOT_T1_CORE_ID", "core_v2")
    monkeypatch.setenv("AUTOPILOT_T1_CORE_PATH", str(core_path))
    monkeypatch.setenv("AUTOPILOT_W6_AUDIT_BLOCK", "1")
    monkeypatch.setenv("AUTOPILOT_W6_AUDIT_N", "2")
    monkeypatch.setenv("AUTOPILOT_W6_AUDIT_SHADOW_ONLY", "0")
    tower = EvalTower()
    tower._pool = {
        "math": [
            {
                "id": "audit-math",
                "suite": "math",
                "prompt": "3+4?",
                "expected": "7",
                "scoring_method": "exact_match",
            },
        ],
        "coder": [
            {
                "id": "audit-coder",
                "suite": "coder",
                "prompt": "Return x",
                "expected": "x",
                "scoring_method": "exact_match",
            },
        ],
    }

    def _fake_eval_batch(self, questions, client, **_kwargs):  # noqa: ANN001, ARG001
        return [
            QuestionResult(
                question_id=q["id"],
                suite=q["suite"],
                prompt=q["prompt"],
                expected=q["expected"],
                correct=q["eval_partition"] == "core",
                tokens_generated=1,
                elapsed_s=1.0,
                eval_partition=q["eval_partition"],
            )
            for q in questions
        ]

    monkeypatch.setattr(EvalTower, "_eval_batch", _fake_eval_batch)

    result = tower.eval_t1(n=999, seed=123, trial_id=17)

    assert result.n_questions == 3
    assert result.quality == 1.0
    assert result.details["audit_policy"]["shadow_only"] is False
    assert "audit_shadow_only" not in result.details


def test_eval_t1_w6_audit_block_honors_cadence(tmp_path, monkeypatch) -> None:
    core_path = tmp_path / "core_v2.jsonl"
    rows = [
        {"__core_metadata__": True, "core_id": "core_v2"},
        {
            "id": "core-a",
            "suite": "math",
            "prompt": "2+2?",
            "expected": "4",
            "scoring_method": "exact_match",
        },
    ]
    core_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    monkeypatch.setenv("AUTOPILOT_T1_CORE_ID", "core_v2")
    monkeypatch.setenv("AUTOPILOT_T1_CORE_PATH", str(core_path))
    monkeypatch.setenv("AUTOPILOT_W6_AUDIT_BLOCK", "1")
    monkeypatch.setenv("AUTOPILOT_W6_AUDIT_N", "2")
    monkeypatch.setenv("AUTOPILOT_W6_AUDIT_EVERY_N_TRIALS", "2")

    captured = []

    def _fake_eval_batch(self, questions, client, **_kwargs):  # noqa: ANN001, ARG001
        captured.extend((q["id"], q["eval_partition"]) for q in questions)
        return [
            QuestionResult(
                question_id=q["id"],
                suite=q["suite"],
                prompt=q["prompt"],
                expected=q["expected"],
                correct=True,
                tokens_generated=1,
                elapsed_s=1.0,
                eval_partition=q["eval_partition"],
            )
            for q in questions
        ]

    monkeypatch.setattr(EvalTower, "_eval_batch", _fake_eval_batch)

    result = EvalTower().eval_t1(n=999, seed=123, trial_id=17)

    assert captured == [("core-a", "core")]
    assert result.details["base_audit_questions"] == 0
    assert result.details["audit_policy"]["active"] is False
    assert result.details["audit_policy"]["skip_reason"] == "trial_not_on_audit_cadence"


def test_eval_t1_w6_audit_block_requires_trial_id(tmp_path, monkeypatch) -> None:
    core_path = tmp_path / "core_v2.jsonl"
    rows = [
        {"__core_metadata__": True, "core_id": "core_v2"},
        {
            "id": "core-a",
            "suite": "math",
            "prompt": "2+2?",
            "expected": "4",
            "scoring_method": "exact_match",
        },
    ]
    core_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    monkeypatch.setenv("AUTOPILOT_T1_CORE_ID", "core_v2")
    monkeypatch.setenv("AUTOPILOT_T1_CORE_PATH", str(core_path))
    monkeypatch.setenv("AUTOPILOT_W6_AUDIT_BLOCK", "1")

    result = EvalTower().eval_t1(n=999, seed=123)

    assert result.quality == 0
    assert result.reliability == 0
    assert result.core_id == "core_v2"
    assert "requires a trial_id" in result.details["audit_error"]
