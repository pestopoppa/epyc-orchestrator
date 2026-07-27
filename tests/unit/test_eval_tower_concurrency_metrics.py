"""EvalTower throughput telemetry for concurrent batches."""

from __future__ import annotations

import hashlib
import json
import math
import sys
import threading
import time
from datetime import UTC, datetime
from dataclasses import fields
from pathlib import Path

import httpx
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "autopilot"))

import eval_tower  # noqa: E402
from eval_tower import EvalTower, QuestionResult  # noqa: E402
from safety_gate import Baseline, SafetyGate  # noqa: E402


def test_eval_concurrency_env_override_still_wins(monkeypatch) -> None:
    monkeypatch.setenv("AUTOPILOT_EVAL_CONCURRENCY", "5")
    monkeypatch.setattr(
        eval_tower,
        "_same_role_matrix_allows_eval_fanout",
        lambda _role: False,
    )

    assert eval_tower._eval_concurrency() == 5


def test_eval_batch_progress_callback_reports_each_completed_question(monkeypatch) -> None:
    monkeypatch.setenv("AUTOPILOT_EVAL_CONCURRENCY", "1")
    tower = EvalTower()
    events: list[dict] = []
    tower.on_progress = events.append

    def fake_eval_question(q: dict, client: object) -> QuestionResult:
        return QuestionResult(
            question_id=str(q["id"]),
            suite="unit",
            prompt=str(q["id"]),
            expected="ok",
            correct=bool(q["correct"]),
        )

    monkeypatch.setattr(tower, "_eval_question", fake_eval_question)

    results = tower._eval_batch(
        [
            {"id": "q1", "correct": True},
            {"id": "q2", "correct": False},
            {"id": "q3", "correct": True},
        ],
        client=object(),  # type: ignore[arg-type]
        log_every=2,
        label="T2",
    )

    assert len(results) == 3
    assert events == [
        {
            "label": "T2",
            "completed_questions": 1,
            "total_questions": 3,
            "correct_questions": 1,
            "correct_pct": 100.0,
            "concurrency": 1,
        },
        {
            "label": "T2",
            "completed_questions": 2,
            "total_questions": 3,
            "correct_questions": 1,
            "correct_pct": 50.0,
            "concurrency": 1,
        },
        {
            "label": "T2",
            "completed_questions": 3,
            "total_questions": 3,
            "correct_questions": 2,
            "correct_pct": pytest.approx(100 * 2 / 3),
            "concurrency": 1,
        },
    ]


def test_eval_batch_stamps_shared_eval_batch_id(monkeypatch) -> None:
    monkeypatch.setenv("AUTOPILOT_EVAL_CONCURRENCY", "1")
    tower = EvalTower()
    seen_batch_ids: list[str] = []

    def fake_eval_question(q: dict, client: object) -> QuestionResult:
        seen_batch_ids.append(str(q.get("_eval_batch_id") or ""))
        return QuestionResult(
            question_id=str(q["id"]),
            suite="unit",
            prompt=str(q["id"]),
            expected="ok",
            correct=True,
        )

    monkeypatch.setattr(tower, "_eval_question", fake_eval_question)

    tower._eval_batch(
        [{"id": "q1"}, {"id": "q2"}, {"id": "q3"}],
        client=object(),  # type: ignore[arg-type]
        label="T1",
    )

    assert len(set(seen_batch_ids)) == 1
    assert seen_batch_ids[0].startswith("evaltower-T1-")
    assert seen_batch_ids[0].endswith("-3q")


@pytest.mark.parametrize(
    ("response_kind", "expected_error"),
    [
        ("timeout", "timed out"),
        ("http_503", "backend down"),
        ("payload_error", "backend busy"),
    ],
)
def test_eval_question_fake_transport_error_paths(
    response_kind: str,
    expected_error: str,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if response_kind == "timeout":
            raise httpx.ReadTimeout("timed out", request=request)
        if response_kind == "http_503":
            return httpx.Response(
                503,
                json={"error_code": 503, "error_detail": "backend down"},
                request=request,
            )
        return httpx.Response(200, json={"error": "backend busy"}, request=request)

    # A realistic eval budget (MockTransport responds synchronously, so the
    # value is a test convenience). Kept above the REL-1 deadline-starvation
    # floor (AUTOPILOT_EVAL_MIN_LLAMA_BUDGET_S=30s) so this transport-error
    # surfacing contract is exercised, not pre-empted by the floor.
    tower = EvalTower(timeout=600)
    with httpx.Client(transport=httpx.MockTransport(handler), timeout=600) as client:
        result = tower._eval_question(
            {
                "id": response_kind,
                "suite": "unit",
                "prompt": "Say ok.",
                "expected": "ok",
                "scoring_method": "exact_match",
            },
            client,
        )

    assert result.question_id == response_kind
    assert result.correct is False
    assert result.error
    assert expected_error in result.error
    assert result.tokens_generated == 0


def test_eval_batch_fake_transport_errors_use_non_error_quality_denominator(
    monkeypatch,
) -> None:
    monkeypatch.setenv("AUTOPILOT_EVAL_CONCURRENCY", "1")

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        prompt = str(payload.get("prompt") or "")
        if "payload error" in prompt:
            return httpx.Response(200, json={"error": "backend busy"}, request=request)
        if "http error" in prompt:
            return httpx.Response(
                503,
                json={"error_code": 503, "error_detail": "backend down"},
                request=request,
            )
        return httpx.Response(
            200,
            json={"answer": "ok", "tokens_generated": 4, "model": "mock-frontdoor"},
            request=request,
        )

    # Realistic budget above the REL-1 deadline-starvation floor (30s);
    # MockTransport is synchronous, so this only avoids the floor pre-empting
    # the error-surfacing contract under test.
    tower = EvalTower(timeout=600)
    questions = [
        {"id": "ok", "suite": "unit", "prompt": "Say ok.", "expected": "ok"},
        {
            "id": "payload-error",
            "suite": "unit",
            "prompt": "Return payload error.",
            "expected": "ok",
        },
        {
            "id": "http-error",
            "suite": "unit",
            "prompt": "Return http error.",
            "expected": "ok",
        },
    ]
    with httpx.Client(transport=httpx.MockTransport(handler), timeout=600) as client:
        results = tower._eval_batch(questions, client=client, label="fake-transport")

    assert [r.question_id for r in results] == ["ok", "payload-error", "http-error"]
    assert results[0].correct is True
    assert results[0].error is None
    assert "backend busy" in str(results[1].error)
    assert "backend down" in str(results[2].error)

    agg = tower._aggregate(results, tier=1)
    assert agg.details["n_scored"] == 1
    assert agg.details["quality_denominator"] == 1
    assert agg.details["scoring_errors"] == 2
    assert agg.quality == 3.0
    assert agg.reliability == pytest.approx(1 / 3)


def test_eval_batch_worker_exception_becomes_error_result(monkeypatch) -> None:
    monkeypatch.setenv("AUTOPILOT_EVAL_CONCURRENCY", "1")
    tower = EvalTower(timeout=1)

    def fake_eval_question(q: dict, client: object) -> QuestionResult:
        if q["id"] == "boom":
            raise RuntimeError("worker exploded")
        return QuestionResult(
            question_id=str(q["id"]),
            suite="unit",
            prompt=str(q["id"]),
            expected="ok",
            correct=True,
        )

    monkeypatch.setattr(tower, "_eval_question", fake_eval_question)

    results = tower._eval_batch(
        [{"id": "ok"}, {"id": "boom"}, {"id": "after"}],
        client=object(),  # type: ignore[arg-type]
        label="worker-exception",
    )

    assert [r.question_id for r in results] == ["ok", "boom", "after"]
    assert results[0].error is None
    assert results[1].error == "worker exploded"
    assert results[2].error is None


def test_eval_batch_fails_remaining_questions_after_no_progress_timeout(monkeypatch) -> None:
    monkeypatch.setenv("AUTOPILOT_EVAL_CONCURRENCY", "2")
    monkeypatch.setenv("AUTOPILOT_EVAL_NO_PROGRESS_TIMEOUT_S", "0.05")
    monkeypatch.setenv("AUTOPILOT_EVAL_ORPHAN_DRAIN_TIMEOUT_S", "0.01")
    tower = EvalTower(timeout=1)

    # workers>1 pipelines generation + scoring on separate pools: a hung
    # GENERATION lane is what this test exercises, so the fake replaces
    # _generate_question and hands scoring a ready ``final_result``.
    def fake_generate(q: dict, client: object) -> "eval_tower._GenOutcome":
        if q["id"] != "fast":
            time.sleep(0.2)
        return eval_tower._GenOutcome(
            gen_ended_at_s=time.time(),
            final_result=QuestionResult(
                question_id=str(q["id"]),
                suite="unit",
                prompt=str(q["id"]),
                expected="ok",
                correct=True,
            ),
        )

    monkeypatch.setattr(tower, "_generate_question", fake_generate)

    results = tower._eval_batch(
        [{"id": "fast"}, {"id": "stuck"}, {"id": "queued"}],
        client=object(),  # type: ignore[arg-type]
        label="T1",
    )

    assert [r.question_id for r in results] == ["fast", "stuck", "queued"]
    assert results[0].error is None
    assert results[1].error
    assert results[1].error.startswith("eval_no_progress_timeout:")
    assert "eval_orphan_contamination" in results[1].error
    assert results[1].degraded is True
    assert results[2].error
    assert results[2].error.startswith("eval_no_progress_timeout:")
    assert "eval_orphan_contamination" in results[2].error
    assert results[2].degraded is True


def test_serial_eval_batch_no_progress_marks_orphan_contamination(monkeypatch) -> None:
    monkeypatch.setenv("AUTOPILOT_EVAL_CONCURRENCY", "1")
    monkeypatch.setenv("AUTOPILOT_EVAL_NO_PROGRESS_TIMEOUT_S", "0.02")
    monkeypatch.setenv("AUTOPILOT_EVAL_ORPHAN_DRAIN_TIMEOUT_S", "0.01")
    release = threading.Event()
    tower = EvalTower(timeout=1)

    def fake_eval_question(q: dict, client: object) -> QuestionResult:
        if q["id"] == "stuck":
            release.wait(1.0)
        return QuestionResult(
            question_id=str(q["id"]),
            suite="unit",
            prompt=str(q["id"]),
            expected="ok",
            correct=True,
        )

    monkeypatch.setattr(tower, "_eval_question", fake_eval_question)

    try:
        results = tower._eval_batch(
            [{"id": "stuck"}, {"id": "queued"}],
            client=object(),  # type: ignore[arg-type]
            label="serial-watchdog",
        )
    finally:
        release.set()

    assert [r.question_id for r in results] == ["stuck", "queued"]
    assert results[0].error
    assert results[0].error.startswith("eval_no_progress_timeout:")
    assert "eval_orphan_contamination" in results[0].error
    assert results[0].degraded is True
    assert results[1].error == "eval_cancelled_after_no_progress_timeout"


def test_aggregate_surfaces_abandoned_eval_request_contamination() -> None:
    tower = EvalTower()
    out = tower._aggregate(
        [
            QuestionResult(
                question_id="q1",
                suite="unit",
                prompt="ok",
                expected="ok",
                correct=True,
            ),
            QuestionResult(
                question_id="q2",
                suite="unit",
                prompt="stuck",
                expected="ok",
                error=(
                    "eval_no_progress_timeout: no completed future for 0.1s; "
                    "eval_orphan_contamination: request may still be decoding server-side"
                ),
                degraded=True,
            ),
        ],
        tier=1,
    )

    assert out.degraded_count == 1
    assert out.details["eval_contaminated_by_abandoned_requests"] is True
    assert out.details["eval_orphan_contamination_count"] == 1


def test_serial_eval_batch_fails_remaining_after_wall_budget(monkeypatch) -> None:
    monkeypatch.setenv("AUTOPILOT_EVAL_CONCURRENCY", "1")
    monkeypatch.setenv("AUTOPILOT_EVAL_BATCH_WALL_BUDGET_S", "0.01")
    tower = EvalTower(timeout=1)

    def fake_eval_question(q: dict, client: object) -> QuestionResult:
        time.sleep(0.02)
        return QuestionResult(
            question_id=str(q["id"]),
            suite="unit",
            prompt=str(q["id"]),
            expected="ok",
            correct=True,
        )

    monkeypatch.setattr(tower, "_eval_question", fake_eval_question)

    results = tower._eval_batch(
        [{"id": "first"}, {"id": "second"}, {"id": "third"}],
        client=object(),  # type: ignore[arg-type]
        label="serial-budget",
    )

    assert [r.question_id for r in results] == ["first", "second", "third"]
    assert results[0].error is None
    assert results[1].error
    assert results[1].error.startswith("eval_wall_budget_timeout:")
    assert results[2].error
    assert results[2].error.startswith("eval_wall_budget_timeout:")


def test_eval_concurrency_uses_topology_cap_when_matrix_allows(monkeypatch) -> None:
    from src.runtime import instance_topology

    monkeypatch.delenv("AUTOPILOT_EVAL_CONCURRENCY", raising=False)
    monkeypatch.setenv("AUTOPILOT_EVAL_BOTTLENECK_ROLE", "frontdoor")
    monkeypatch.setattr(instance_topology, "max_safe_concurrency", lambda role: 3)
    monkeypatch.setattr(
        eval_tower,
        "_same_role_matrix_allows_eval_fanout",
        lambda role: role == "frontdoor",
    )
    monkeypatch.setattr(
        eval_tower,
        "_live_safe_concurrency",
        lambda role, cap: cap if role == "frontdoor" else 1,
    )

    assert eval_tower._eval_concurrency() == 3


def test_eval_concurrency_uses_min_cap_across_forced_roles(monkeypatch) -> None:
    from src.runtime import instance_topology

    monkeypatch.delenv("AUTOPILOT_EVAL_CONCURRENCY", raising=False)
    monkeypatch.setenv("AUTOPILOT_EVAL_BOTTLENECK_ROLE", "frontdoor")
    monkeypatch.setattr(
        instance_topology,
        "max_safe_concurrency",
        lambda role: {"frontdoor": 4, "worker_general": 2}.get(role, 1),
    )
    monkeypatch.setattr(
        eval_tower,
        "_same_role_matrix_allows_eval_fanout",
        lambda role: role in {"frontdoor", "worker_general"},
    )
    monkeypatch.setattr(
        eval_tower,
        "_live_safe_concurrency",
        lambda role, cap: {"frontdoor": 4, "worker_general": 2}.get(role, cap),
    )

    assert eval_tower._eval_concurrency(["frontdoor", "worker_general"]) == 2


def test_eval_batch_resolves_concurrency_from_actual_forced_roles(monkeypatch) -> None:
    monkeypatch.delenv("AUTOPILOT_EVAL_CONCURRENCY", raising=False)
    tower = EvalTower()
    observed: list[tuple[str, ...]] = []

    def fake_eval_concurrency(roles=None):
        observed.append(tuple(roles or ()))
        return 1

    def fake_eval_question(q: dict, client: object) -> QuestionResult:
        return QuestionResult(
            question_id=str(q["id"]),
            suite="unit",
            prompt=str(q["id"]),
            expected="ok",
            correct=True,
        )

    monkeypatch.setattr(eval_tower, "_eval_concurrency", fake_eval_concurrency)
    monkeypatch.setattr(tower, "_eval_question", fake_eval_question)

    tower._eval_batch(
        [
            {"id": "q1", "force_role": "worker_general"},
            {"id": "q2", "force_role": "frontdoor"},
            {"id": "q3", "force_role": "worker_general"},
        ],
        client=object(),  # type: ignore[arg-type]
        label="forced",
    )

    assert observed == [("worker_general", "frontdoor")]


def test_eval_concurrency_caps_to_live_fleet_when_static_topology_allows(monkeypatch) -> None:
    from src.runtime import instance_topology

    monkeypatch.delenv("AUTOPILOT_EVAL_CONCURRENCY", raising=False)
    monkeypatch.setenv("AUTOPILOT_EVAL_BOTTLENECK_ROLE", "frontdoor")
    monkeypatch.setattr(instance_topology, "max_safe_concurrency", lambda _role: 3)
    monkeypatch.setattr(
        eval_tower,
        "_same_role_matrix_allows_eval_fanout",
        lambda _role: True,
    )
    monkeypatch.setattr(
        eval_tower,
        "_live_safe_concurrency",
        lambda _role, _cap: 1,
    )

    assert eval_tower._eval_concurrency() == 1


def test_live_safe_concurrency_uses_quarter_mode_live_ports(monkeypatch) -> None:
    from scripts.server import runtime_facts_manifest, stack_numa

    cfg = {
        "worker_general": {
            "instances": [
                ("0-95", 8072, 96),
                ("0-23,96-119", 8082, 48),
                ("24-47,120-143", 8182, 48),
                ("48-71,144-167", 8282, 48),
                ("72-95,168-191", 8382, 48),
            ],
        },
    }
    live_ports = {8082, 8182, 8282, 8382}

    class _Response:
        def __init__(self, status_code: int) -> None:
            self.status_code = status_code

    def fake_get(url: str, timeout: float) -> _Response:
        del timeout
        port = int(url.rsplit(":", 1)[1].split("/", 1)[0])
        return _Response(200 if port in live_ports else 503)

    monkeypatch.setenv("AUTOPILOT_EVAL_REQUIRE_LIVE_FLEET", "1")
    monkeypatch.setattr(stack_numa, "NUMA_CONFIG", cfg)
    monkeypatch.setattr(runtime_facts_manifest, "read_runtime_stack_numa_mode", lambda **_: "quarter")
    # WP-14: the runtime-facts reader seam now also requires a non-empty,
    # consistent selected-server lineup (the URL reader's fail-closed contract)
    # before honoring the manifest mode, so provide a well-formed quarter lineup.
    monkeypatch.setattr(
        runtime_facts_manifest,
        "read_runtime_stack_selected_servers",
        lambda **_: [
            {"port": 8082, "roles": ["worker_general"]},
            {"port": 8182, "roles": ["worker_general"]},
            {"port": 8282, "roles": ["worker_general"]},
            {"port": 8382, "roles": ["worker_general"]},
        ],
    )
    monkeypatch.setattr(eval_tower.httpx, "get", fake_get)

    assert eval_tower._live_safe_concurrency("worker_general", 1) == 4


# ---------------------------------------------------------------------------
# WP-14: runtime-facts reader seam fail-closed contract
# ---------------------------------------------------------------------------

_PHANTOM_RUNTIME_FACTS = {
    "schema": "epyc.orchestrator.runtime_facts",
    "schema_version": 1,
    "runtime_stack": {
        # Real current phantom shape: mode null, empty selected_ports, but a
        # left-behind full-era server lineup.
        "stack_numa_mode": None,
        "selected_servers": [
            {"port": 8070, "roles": ["frontdoor", "coder_escalation", "worker_summarize"], "numa_instance": 0},
            {"port": 8072, "roles": ["worker_general", "worker_math", "toolrunner"], "numa_instance": 0},
        ],
        "selected_ports": [],
    },
}

_WELLFORMED_QUARTER_RUNTIME_FACTS = {
    "schema": "epyc.orchestrator.runtime_facts",
    "schema_version": 1,
    "runtime_stack": {
        "stack_numa_mode": "quarter",
        "selected_servers": [
            {"port": 8082, "roles": ["worker_general", "worker_math"], "numa_instance": 1},
            {"port": 8182, "roles": ["worker_general"], "numa_instance": 2},
        ],
        "selected_ports": [8082, 8182],
    },
}


def _install_runtime_facts(monkeypatch, tmp_path: Path, payload: dict) -> Path:
    from scripts.server import stack_paths
    from scripts.server.runtime_facts_manifest import runtime_facts_manifest_path

    monkeypatch.setitem(stack_paths._PATHS, "tmp_dir", tmp_path)
    manifest_path = runtime_facts_manifest_path()
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    return manifest_path


def test_runtime_facts_stack_numa_mode_rejects_phantom_manifest(
    monkeypatch, tmp_path, caplog
) -> None:
    """WP-14: a phantom full-era manifest (mode null, empty selected_ports) is
    rejected fail-closed and the seam falls back with one loud log line."""
    import logging

    monkeypatch.delenv("ORCHESTRATOR_STACK_NUMA_MODE", raising=False)
    _install_runtime_facts(monkeypatch, tmp_path, _PHANTOM_RUNTIME_FACTS)

    with caplog.at_level(logging.WARNING, logger="autopilot.eval"):
        assert eval_tower._runtime_facts_stack_numa_mode() is None

    rejections = [
        rec for rec in caplog.records if "runtime-facts manifest rejected" in rec.getMessage()
    ]
    assert len(rejections) == 1


def test_runtime_facts_stack_numa_mode_consumes_wellformed_quarter_manifest(
    monkeypatch, tmp_path, caplog
) -> None:
    """WP-14: a well-formed quarter manifest (mode + consistent lineup) is
    consumed and no fail-closed warning is emitted."""
    import logging

    monkeypatch.delenv("ORCHESTRATOR_STACK_NUMA_MODE", raising=False)
    _install_runtime_facts(monkeypatch, tmp_path, _WELLFORMED_QUARTER_RUNTIME_FACTS)

    with caplog.at_level(logging.WARNING, logger="autopilot.eval"):
        assert eval_tower._runtime_facts_stack_numa_mode() == "quarter"

    assert not [
        rec for rec in caplog.records if "runtime-facts manifest rejected" in rec.getMessage()
    ]


def test_live_safe_concurrency_falls_back_when_manifest_is_phantom(
    monkeypatch, tmp_path
) -> None:
    """WP-14: with a phantom manifest present and no env override, the live-fleet
    seam does NOT take the quarter disjoint-concurrency path; it falls back to the
    conservative topology bound instead of the phantom's 4-wide quarter lineup."""
    from scripts.server import stack_numa

    cfg = {
        "worker_general": {
            "instances": [
                ("0-95", 8072, 96),
                ("0-23,96-119", 8082, 48),
                ("24-47,120-143", 8182, 48),
                ("48-71,144-167", 8282, 48),
                ("72-95,168-191", 8382, 48),
            ],
        },
    }
    live_ports = {8082, 8182, 8282, 8382}

    class _Response:
        def __init__(self, status_code: int) -> None:
            self.status_code = status_code

    def fake_get(url: str, timeout: float) -> _Response:
        del timeout
        port = int(url.rsplit(":", 1)[1].split("/", 1)[0])
        return _Response(200 if port in live_ports else 503)

    monkeypatch.delenv("ORCHESTRATOR_STACK_NUMA_MODE", raising=False)
    monkeypatch.setenv("AUTOPILOT_EVAL_REQUIRE_LIVE_FLEET", "1")
    monkeypatch.setattr(stack_numa, "NUMA_CONFIG", cfg)
    monkeypatch.setattr(eval_tower.httpx, "get", fake_get)
    _install_runtime_facts(monkeypatch, tmp_path, _PHANTOM_RUNTIME_FACTS)

    # Phantom rejected -> not treated as quarter -> topology_cap=1 bound wins.
    assert eval_tower._live_safe_concurrency("worker_general", 1) == 1


def test_eval_concurrency_can_exceed_full_first_cap_in_quarter_mode(monkeypatch) -> None:
    from src.runtime import instance_topology

    monkeypatch.delenv("AUTOPILOT_EVAL_CONCURRENCY", raising=False)
    monkeypatch.setenv("AUTOPILOT_EVAL_BOTTLENECK_ROLE", "worker_general")
    monkeypatch.setattr(instance_topology, "max_safe_concurrency", lambda _role: 1)
    monkeypatch.setattr(
        eval_tower,
        "_same_role_matrix_allows_eval_fanout",
        lambda _role: True,
    )
    monkeypatch.setattr(
        eval_tower,
        "_live_safe_concurrency",
        lambda _role, _cap: 4,
    )

    assert eval_tower._eval_concurrency() == 4


def test_live_safe_concurrency_can_be_disabled_for_diagnostics(monkeypatch) -> None:
    monkeypatch.setenv("AUTOPILOT_EVAL_REQUIRE_LIVE_FLEET", "0")

    assert eval_tower._live_safe_concurrency("frontdoor", 3) == 3


def test_eval_concurrency_falls_back_to_serial_when_matrix_blocks(monkeypatch) -> None:
    from src.runtime import instance_topology

    monkeypatch.delenv("AUTOPILOT_EVAL_CONCURRENCY", raising=False)
    monkeypatch.setenv("AUTOPILOT_EVAL_BOTTLENECK_ROLE", "frontdoor")
    monkeypatch.setattr(instance_topology, "max_safe_concurrency", lambda _role: 3)
    monkeypatch.setattr(
        eval_tower,
        "_same_role_matrix_allows_eval_fanout",
        lambda _role: False,
    )

    assert eval_tower._eval_concurrency() == 1


def test_eval_concurrency_falls_back_to_serial_when_matrix_stale(monkeypatch) -> None:
    from scripts.server import stack_numa
    from src.scheduling import contention

    matrix = contention.ContentionMatrix(
        version=1,
        measured_at="",
        host="test",
        topology_hash="old",
        default_floor=0.85,
        same_role={
            "frontdoor": contention.SameRole(role="frontdoor", verdict="allow"),
        },
    )
    monkeypatch.setattr(stack_numa, "NUMA_CONFIG", {"frontdoor": {"instances": []}})
    monkeypatch.setattr(contention, "load_contention_matrix", lambda: matrix)
    monkeypatch.setattr(contention, "topology_fingerprint_for_matrix", lambda _cfg, _matrix: "new")
    monkeypatch.setattr(
        contention,
        "matrix_status",
        lambda current_topology_hash: contention.MatrixStatus.STALE,
    )

    assert not eval_tower._same_role_matrix_allows_eval_fanout("frontdoor")


def test_eval_concurrency_allows_stale_global_matrix_with_fresh_role_cert(monkeypatch) -> None:
    from scripts.server import stack_numa
    from src.scheduling import contention

    cfg = {
        "frontdoor": {
            "instances": [
                ("0-47,96-143", 8070, 96),
                ("0-23,96-119", 8080, 48),
                ("24-47,120-143", 8180, 48),
            ],
        },
    }
    live_ports = {8080, 8180}
    role_hash = contention.role_topology_fingerprint(
        cfg,
        "frontdoor",
        live_ports=live_ports,
    )
    matrix = contention.ContentionMatrix(
        version=1,
        measured_at="",
        host="test",
        topology_hash="stale-whole",
        default_floor=0.85,
        same_role={
            "frontdoor": contention.SameRole(role="frontdoor", verdict="allow"),
        },
        same_role_certifications={
            "frontdoor": contention.SameRoleCertification(
                role="frontdoor",
                topology_hash=role_hash,
                verdict="allow",
                measured_at=datetime.now(UTC).isoformat(),
                live_ports=tuple(sorted(live_ports)),
            ),
        },
    )

    monkeypatch.setattr(stack_numa, "NUMA_CONFIG", cfg)
    monkeypatch.setattr(contention, "load_contention_matrix", lambda: matrix)
    monkeypatch.setattr(contention, "topology_fingerprint_for_matrix", lambda _cfg, _matrix: "new-whole")
    monkeypatch.setattr(
        contention,
        "matrix_status",
        lambda current_topology_hash: contention.MatrixStatus.STALE,
    )
    monkeypatch.setattr(eval_tower, "_live_role_ports", lambda _cfg, _role: live_ports)

    assert eval_tower._same_role_matrix_allows_eval_fanout("frontdoor")


def test_eval_concurrency_rejects_role_cert_when_live_ports_drift(monkeypatch) -> None:
    from scripts.server import stack_numa
    from src.scheduling import contention

    cfg = {
        "frontdoor": {
            "instances": [
                ("0-23,96-119", 8080, 48),
                ("24-47,120-143", 8180, 48),
                ("48-71,144-167", 8280, 48),
            ],
        },
    }
    certified_ports = {8080, 8180}
    matrix = contention.ContentionMatrix(
        version=1,
        measured_at="",
        host="test",
        topology_hash="stale-whole",
        default_floor=0.85,
        same_role={
            "frontdoor": contention.SameRole(role="frontdoor", verdict="allow"),
        },
        same_role_certifications={
            "frontdoor": contention.SameRoleCertification(
                role="frontdoor",
                topology_hash=contention.role_topology_fingerprint(
                    cfg,
                    "frontdoor",
                    live_ports=certified_ports,
                ),
                verdict="allow",
                measured_at=datetime.now(UTC).isoformat(),
                live_ports=tuple(sorted(certified_ports)),
            ),
        },
    )

    monkeypatch.setattr(stack_numa, "NUMA_CONFIG", cfg)
    monkeypatch.setattr(contention, "load_contention_matrix", lambda: matrix)
    monkeypatch.setattr(contention, "topology_fingerprint_for_matrix", lambda _cfg, _matrix: "new-whole")
    monkeypatch.setattr(
        contention,
        "matrix_status",
        lambda current_topology_hash: contention.MatrixStatus.STALE,
    )
    monkeypatch.setattr(eval_tower, "_live_role_ports", lambda _cfg, _role: {8080, 8180, 8280})

    assert not eval_tower._same_role_matrix_allows_eval_fanout("frontdoor")


def test_aggregate_uses_batch_throughput_for_concurrent_objective() -> None:
    tower = EvalTower()
    results = [
        QuestionResult(
            question_id="q1",
            suite="math",
            prompt="a",
            expected="a",
            answer="a",
            correct=True,
            tokens_generated=100,
            elapsed_s=10.0,
            eval_concurrency=3,
            eval_wall_s=12.0,
        ),
        QuestionResult(
            question_id="q2",
            suite="math",
            prompt="b",
            expected="b",
            answer="b",
            correct=True,
            tokens_generated=120,
            elapsed_s=12.0,
            eval_concurrency=3,
            eval_wall_s=12.0,
        ),
        QuestionResult(
            question_id="q3",
            suite="math",
            prompt="c",
            expected="c",
            answer="c",
            correct=False,
            tokens_generated=60,
            elapsed_s=6.0,
            eval_concurrency=3,
            eval_wall_s=12.0,
        ),
    ]

    out = tower._aggregate(results, tier=1)

    assert out.speed == (280 / 12.0)
    assert out.speed_metric_mode == "aggregate_batch_tps"
    assert out.median_request_speed == 10.0
    assert out.aggregate_speed == (280 / 12.0)
    assert out.eval_concurrency == 3
    assert out.eval_wall_s == 12.0
    assert out.sum_request_elapsed_s == 28.0
    assert out.details["objective_speed_tps"] == out.speed
    assert out.details["median_request_tps"] == out.median_request_speed
    assert out.details["aggregate_tps"] == out.aggregate_speed
    assert out.details["speed_metric_mode"] == "aggregate_batch_tps"
    assert out.details["task_rate_qph"] == pytest.approx(900.0)
    assert out.details["goodput_qph"] == pytest.approx(600.0)
    assert out.details["tokens_per_solved_task"] == 140.0


def test_aggregate_quality_denominator_excludes_error_rows() -> None:
    tower = EvalTower()
    results = [
        QuestionResult(
            question_id="q1",
            suite="math",
            prompt="a",
            expected="a",
            answer="a",
            correct=True,
            tokens_generated=100,
            elapsed_s=10.0,
            eval_wall_s=12.0,
        ),
        QuestionResult(
            question_id="q2",
            suite="math",
            prompt="b",
            expected="b",
            answer="wrong",
            correct=False,
            tokens_generated=50,
            elapsed_s=5.0,
            eval_wall_s=12.0,
        ),
        QuestionResult(
            question_id="q3",
            suite="math",
            prompt="c",
            expected="c",
            answer="",
            correct=False,
            error="scorer_unavailable",
            elapsed_s=1.0,
            eval_wall_s=12.0,
        ),
    ]

    out = tower._aggregate(results, tier=1)

    assert out.n_questions == 3
    assert out.quality == pytest.approx(1.5)
    assert out.reliability == pytest.approx(2 / 3)
    assert set(out.per_suite_quality) == {"math"}
    assert out.per_suite_quality["math"] == pytest.approx(1.5)
    assert out.per_suite_counts == {"math": 2}
    assert out.details["n_questions"] == 3
    assert out.details["n_scored"] == 2
    assert out.details["quality_denominator"] == 2
    assert out.details["quality_denominator_semantics"] == "non_error_question_results"
    assert out.details["errors"] == 1
    assert out.details["scoring_errors"] == 1
    assert out.details["per_suite_total_counts"] == {"math": 3}
    assert out.details["goodput_qph"] == pytest.approx(300.0)
    assert out.details["scored_task_rate_qph"] == pytest.approx(600.0)


def test_question_result_has_host_covariates_default_factory() -> None:
    host_covariates_field = next(
        field for field in fields(QuestionResult) if field.name == "host_covariates"
    )

    assert host_covariates_field.default_factory is dict


def test_compact_question_result_includes_bounded_host_covariates() -> None:
    result = QuestionResult(
        question_id="q1",
        suite="math",
        prompt="a",
        expected="a",
        tokens_generated=42,
        elapsed_s=1.0,
        host_covariates={
            "min_core_mhz": 2600.0,
            "host_inflight": 3,
            "numa_balancing": 0,
            "cache_warm_state": "warm",
            "page_cache_mb": 4096.0,
            "mem_available_mb": 8192.0,
            "timestamp": 123.456,
            "loadavg_1min": 7.25,
            "extra_noise": "ignore-me",
        },
    )

    compact = eval_tower._compact_question_result(result)

    assert compact["tokens_generated"] == 42
    assert compact["host_covariates"] == {
        "min_core_mhz": 2600.0,
        "host_inflight": 3,
        "numa_balancing": 0,
        "cache_warm_state": "warm",
        "page_cache_mb": 4096.0,
        "mem_available_mb": 8192.0,
    }
    assert "timestamp" not in compact["host_covariates"]
    assert "loadavg_1min" not in compact["host_covariates"]
    assert "extra_noise" not in compact["host_covariates"]


def test_aggregate_emits_speed_analytics_and_host_covariate_summary() -> None:
    tower = EvalTower()
    results = [
        QuestionResult(
            question_id="q1",
            suite="math",
            prompt="a",
            expected="a",
            tokens_generated=64,
            elapsed_s=1.0,
            eval_concurrency=1,
            eval_wall_s=4.0,
            host_covariates={"min_core_mhz": 2500.0, "host_inflight": 1},
        ),
        QuestionResult(
            question_id="q2",
            suite="math",
            prompt="b",
            expected="b",
            tokens_generated=128,
            elapsed_s=2.0,
            eval_concurrency=1,
            eval_wall_s=4.0,
            host_covariates={
                "min_core_mhz": 2600.0,
                "host_inflight": 2,
                "numa_balancing": 0,
                "cache_warm_state": "warm",
                "page_cache_mb": 4096.0,
                "mem_available_mb": 8192.0,
            },
        ),
        QuestionResult(
            question_id="q3",
            suite="math",
            prompt="c",
            expected="c",
            tokens_generated=256,
            elapsed_s=4.0,
            eval_concurrency=1,
            eval_wall_s=4.0,
            host_covariates={
                "min_core_mhz": 2700.0,
                "host_inflight": 3,
                "numa_balancing": 1,
                "cache_warm_state": "cold",
                "page_cache_mb": 8192.0,
                "mem_available_mb": 16384.0,
            },
        ),
    ]

    out = tower._aggregate(results, tier=1)

    assert out.details["speed_analytics_min_tokens"] == 128
    assert out.details["speed_analytics_n_ge_128"] == 2
    assert out.details["speed_analytics_median_request_tps_ge_128"] == 64.0
    assert "host_timing_covariates" in out.details
    assert isinstance(out.details["host_timing_covariates"], dict)
    assert {
        "min_core_mhz",
        "host_inflight",
        "numa_balancing",
        "cache_warm_state",
        "page_cache_mb",
        "mem_available_mb",
    }.issubset(out.details["host_timing_covariates"])


def test_aggregate_emits_compact_stable_question_results() -> None:
    tower = EvalTower()
    out = tower._aggregate(
        [
            QuestionResult(
                question_id="transient-source-id",
                suite="math",
                prompt="What is two plus two?",
                expected="4",
                answer="4",
                correct=True,
                tokens_generated=5,
                elapsed_s=1.234,
                tools_used=2,
            )
        ],
        tier=1,
    )

    expected_qid = hashlib.sha1(b"math\x00What is two plus two?").hexdigest()[:16]
    assert out.question_results == [
        {
            "qid": expected_qid,
            "question_id": "transient-source-id",
            "suite": "math",
            "partition": "core",
            "correct": True,
            "latency_ms": 1234,
            "tokens_generated": 5,
            "tools_used": 2,
            "answer_hash": eval_tower.normalized_answer_hash("4"),
        }
    ]
    assert "prompt" not in out.question_results[0]
    assert "answer" not in out.question_results[0]


def test_aggregate_emits_truthy_question_provenance_flags() -> None:
    tower = EvalTower()
    out = tower._aggregate(
        [
            QuestionResult(
                question_id="q1",
                qid="stable-q1",
                suite="coder",
                prompt="Write a function",
                expected="ok",
                answer="timeout",
                correct=False,
                error="read_timeout",
                elapsed_s=2.0,
                route_used="frontdoor->worker_general",
                scoring_method="programmatic",
                partial=True,
                degraded=True,
                exogenous_recovered=True,
                external_restart=True,
                retry_count=1,
                tools_used=1,
                tools_called=["read_file"],
                eval_partition="audit",
            )
        ],
        tier=1,
    )

    assert out.question_results == [
        {
            "qid": "stable-q1",
            "question_id": "q1",
            "suite": "coder",
            "partition": "audit",
            "correct": False,
            "latency_ms": 2000,
            "tokens_generated": 0,
            "tools_used": 1,
            "scoring_method": "programmatic",
            "route": "frontdoor->worker_general",
            "tools_called": ["read_file"],
            "error": True,
            "error_detail": "read_timeout",
            "partial": True,
            "degraded": True,
            "exogenous_recovered": True,
            "external_restart": True,
            "retry_count": 1,
        }
    ]
    assert "prompt" not in out.question_results[0]
    assert "answer" not in out.question_results[0]
    assert "answer_hash" not in out.question_results[0]


def test_eval_result_grep_lines_include_concurrency_metrics() -> None:
    tower = EvalTower()
    out = tower._aggregate(
        [
            QuestionResult(
                question_id="q1",
                suite="general",
                prompt="a",
                expected="a",
                tokens_generated=50,
                elapsed_s=5.0,
                eval_concurrency=2,
                eval_wall_s=5.0,
            )
        ],
        tier=0,
    )

    lines = out.to_grep_lines(trial_id=7, species="test")

    assert "METRIC speed: 10.00" in lines
    assert "METRIC speed_metric_mode: aggregate_batch_tps" in lines
    assert "METRIC median_request_speed: 10.00" in lines
    assert "METRIC aggregate_speed: 10.00" in lines
    assert "METRIC eval_concurrency: 2" in lines
    assert "METRIC eval_wall_s: 5.00" in lines


def test_serial_eval_keeps_median_request_speed_as_objective() -> None:
    tower = EvalTower()
    out = tower._aggregate(
        [
            QuestionResult(
                question_id="q1",
                suite="general",
                prompt="a",
                expected="a",
                tokens_generated=100,
                elapsed_s=10.0,
                eval_concurrency=1,
                eval_wall_s=10.0,
            ),
            QuestionResult(
                question_id="q2",
                suite="general",
                prompt="b",
                expected="b",
                tokens_generated=50,
                elapsed_s=10.0,
                eval_concurrency=1,
                eval_wall_s=10.0,
            ),
        ],
        tier=0,
    )

    assert out.speed == 10.0
    assert out.speed_metric_mode == "median_request_tps"
    assert out.median_request_speed == 10.0
    assert out.aggregate_speed == 15.0


def test_safety_gate_uses_effective_speed_not_raw_median_for_concurrent_eval() -> None:
    tower = EvalTower()
    out = tower._aggregate(
        [
            QuestionResult(
                question_id="q1",
                suite="general",
                prompt="a",
                expected="a",
                correct=True,
                tokens_generated=100,
                elapsed_s=10.0,
                eval_concurrency=3,
                eval_wall_s=15.0,
            ),
            QuestionResult(
                question_id="q2",
                suite="general",
                prompt="b",
                expected="b",
                correct=True,
                tokens_generated=100,
                elapsed_s=10.0,
                eval_concurrency=3,
                eval_wall_s=15.0,
            ),
            QuestionResult(
                question_id="q3",
                suite="general",
                prompt="c",
                expected="c",
                correct=True,
                tokens_generated=100,
                elapsed_s=10.0,
                eval_concurrency=3,
                eval_wall_s=15.0,
            ),
        ],
        tier=1,
    )
    assert out.median_request_speed == 10.0
    assert out.speed == 20.0

    gate = SafetyGate()
    gate.baseline = Baseline(quality=2.0, frontdoor_speed=18.0)
    verdict = gate.check(out)

    assert verdict.passed
    assert "throughput" not in verdict.categories


def test_deep_research_expected_contains_items_are_scoreable() -> None:
    assert eval_tower._is_scoreable_question(
        {
            "id": "dr-1",
            "suite": "deep_research_browsecomp",
            "prompt": "Research alpha beta.",
            "expected_contains": ["alpha beta", "gamma delta"],
        }
    )


def test_eval_question_populates_deterministic_rubric_scores(monkeypatch) -> None:
    monkeypatch.delenv("AUTOPILOT_RUBRIC_JUDGE_ROLES", raising=False)
    tower = EvalTower()

    def _fake_call(**_kwargs):  # noqa: ANN001
        return {
            "answer": (
                "# Summary\n"
                "- alpha beta evidence\n"
                "- gamma delta caveat\n"
                "Source: https://example.test/report\n"
                "Therefore the comparison is grounded in the evidence."
            ),
            "tokens_generated": 20,
            "model": "fake",
            "tools_called": ["web_search", "read_file"],
        }

    monkeypatch.setattr(eval_tower, "call_orchestrator_forced", _fake_call)

    with eval_tower.httpx.Client(timeout=1) as client:
        result = tower._eval_question(
            {
                "id": "dr-1",
                "suite": "deep_research_browsecomp",
                "prompt": "Research alpha beta.",
                "expected_contains": ["alpha beta", "gamma delta"],
                "scoring_config": {"rubric_pass_threshold": 0.5},
            },
            client,
        )

    assert result.correct is True
    assert result.scoring_method == "rubric"
    assert result.confidence >= 0.5
    assert result.rubric_scores["factual_accuracy"] == 1.0
    assert result.rubric_scores["tool_calls"] > 0


def test_eval_question_uses_completion_probabilities_for_confidence(monkeypatch) -> None:
    tower = EvalTower()
    seen: dict[str, object] = {}

    def _fake_call(**kwargs):  # noqa: ANN001
        seen.update(kwargs)
        return {
            "answer": "black cat",
            "tokens_generated": 2,
            "model": "fake",
            "completion_probabilities": [
                {"content": "black", "probs": [{"tok_str": "black", "prob": 0.81}]},
                {"content": " cat", "probs": [{"tok_str": " cat", "prob": 0.64}]},
            ],
        }

    monkeypatch.setattr(eval_tower, "call_orchestrator_forced", _fake_call)

    with eval_tower.httpx.Client(timeout=1) as client:
        result = tower._eval_question(
            {
                "id": "mc-1",
                "suite": "unit",
                "prompt": "Choose the phrase.",
                "expected": "black cat",
                "scoring_method": "multiple_choice",
                "scoring_config": {"choices": ["cat", "black cat"]},
            },
            client,
        )

    assert seen["n_probs"] == 5
    assert result.correct is True
    assert result.confidence_source == "completion_probabilities_geomean"
    assert result.confidence == pytest.approx(math.sqrt(0.81 * 0.64))

    aggregate = tower._aggregate([result], tier=1)
    assert aggregate.details["confidence_source_counts"] == {
        "completion_probabilities_geomean": 1
    }
    assert aggregate.details["confidence_is_real"] is True


def test_parse_rubric_judge_scores_accepts_fenced_json() -> None:
    parsed = eval_tower._parse_rubric_judge_scores(
        """```json
        {"scores": {"reasoning_trajectory": 0.25, "tool_calls": 1.7, "bad": "x"}}
        ```"""
    )

    # B7b / SCORE-07 (audit 2026-07-20): out-of-[0,1] judge scores are now
    # REJECTED (scale-drift), not clamped. `tool_calls: 1.7` no longer saturates
    # to a perfect 1.0 — it is dropped so that dimension falls to the
    # deterministic heuristic fallback; `bad: "x"` remains unparseable.
    assert parsed == {"reasoning_trajectory": 0.25}


def test_eval_question_uses_configured_local_rubric_judge(monkeypatch) -> None:
    monkeypatch.setenv("AUTOPILOT_RUBRIC_JUDGE_ROLES", "architect_general")
    monkeypatch.setenv("AUTOPILOT_RUBRIC_JUDGE_TIMEOUT_S", "1")
    tower = EvalTower()
    calls: list[dict] = []

    def _fake_call(**kwargs):  # noqa: ANN001
        calls.append(kwargs)
        if kwargs.get("scoring_method") == "rubric_judge":
            return {
                "answer": (
                    '{"scores":{"reasoning_trajectory":0.2,'
                    '"tool_calls":0.4,"outline":0.6,"content_stage":0.8,'
                    '"factual_accuracy":0.9}}'
                ),
                "model": "judge",
            }
        return {
            "answer": (
                "# Summary\n"
                "- alpha beta evidence\n"
                "- gamma delta caveat\n"
                "Source: https://example.test/report\n"
            ),
            "tokens_generated": 20,
            "model": "Llama-Generator",
            "tools_called": ["web_search"],
        }

    monkeypatch.setattr(eval_tower, "call_orchestrator_forced", _fake_call)

    with eval_tower.httpx.Client(timeout=1) as client:
        result = tower._eval_question(
            {
                "id": "dr-judge",
                "suite": "deep_research_mixed",
                "prompt": "Research alpha beta.",
                "expected_contains": ["alpha beta", "gamma delta"],
                "scoring_config": {"rubric_pass_threshold": 0.5},
            },
            client,
        )

    assert len(calls) == 2
    assert calls[1]["force_role"] == "architect_general"
    assert calls[1]["allow_delegation"] is False
    assert calls[1]["scoring_method"] == "rubric_judge"
    assert result.rubric_scores["reasoning_trajectory"] == 0.2
    assert result.rubric_scores["content_stage"] == 0.8
    assert result.rubric_scores["factual_accuracy"] == 0.9
    # B7b / SCORE-08: a live model judge produced parseable scores → provenance.
    assert result.rubric_source == "judge"


# ── B7b (audit 2026-07-20): SCORE-07/08/09/12 scorer-semantics remainder ──


def test_parse_rubric_judge_scores_rejects_out_of_range_not_clamp() -> None:
    # SCORE-07: a 0-10-scale judge value ("7") must be REJECTED as scale drift,
    # not clamped to a perfect 1.0. In-range values (incl. the 0.0/1.0
    # boundaries) survive unchanged.
    parsed = eval_tower._parse_rubric_judge_scores(
        '{"scores": {"reasoning_trajectory": 7, "tool_calls": -0.5, '
        '"outline": 0.0, "content_stage": 1.0, "factual_accuracy": 0.6}}'
    )
    assert parsed == {
        "outline": 0.0,
        "content_stage": 1.0,
        "factual_accuracy": 0.6,
    }


def test_eval_question_stamps_rubric_source_heuristic_fallback(monkeypatch) -> None:
    # SCORE-08: with no judge roles configured, rubric scores come from the
    # deterministic heuristic fallback and must be stamped as such; the aggregate
    # surfaces the provenance rollup.
    monkeypatch.delenv("AUTOPILOT_RUBRIC_JUDGE_ROLES", raising=False)
    tower = EvalTower()

    def _fake_call(**_kwargs):  # noqa: ANN001
        return {
            "answer": (
                "# Summary\n"
                "- alpha beta evidence\n"
                "Source: https://example.test/report\n"
            ),
            "tokens_generated": 12,
            "model": "fake",
        }

    monkeypatch.setattr(eval_tower, "call_orchestrator_forced", _fake_call)
    with eval_tower.httpx.Client(timeout=1) as client:
        result = tower._eval_question(
            {
                "id": "dr-heur",
                "suite": "deep_research_browsecomp",
                "prompt": "Research alpha beta.",
                "expected_contains": ["alpha beta"],
                "scoring_config": {"rubric_pass_threshold": 0.1},
            },
            client,
        )

    assert result.rubric_source == "heuristic_fallback"
    aggregate = tower._aggregate([result], tier=1)
    assert aggregate.details["rubric_source_counts"] == {"heuristic_fallback": 1}


def test_derive_question_confidence_code_execution_ignores_pass_rate() -> None:
    # SCORE-12: the phantom static pass_rate read is gone; code_execution
    # confidence is a binary correctness proxy with a source that can never be
    # mistaken for real (completion-probability) confidence.
    conf, source = eval_tower._derive_question_confidence(
        scoring_method="code_execution",
        correct=True,
        probability_confidence=None,
        rubric_scores={},
    )
    assert conf == 1.0
    assert source == "code_execution_binary_proxy"

    conf_wrong, source_wrong = eval_tower._derive_question_confidence(
        scoring_method="code_execution",
        correct=False,
        probability_confidence=None,
        rubric_scores={},
    )
    assert conf_wrong == 0.0
    assert source_wrong == "code_execution_binary_proxy"

    # Real completion-probability confidence still wins for non-code paths.
    conf_prob, source_prob = eval_tower._derive_question_confidence(
        scoring_method="exact_match",
        correct=True,
        probability_confidence=0.73,
        rubric_scores={},
    )
    assert conf_prob == 0.73
    assert source_prob == "completion_probabilities_geomean"


def test_derive_question_confidence_code_execution_geomean_when_probs_present() -> None:
    # ESC-7 extension Option A (operator 2026-07-21): a code_execution question
    # WITH probability rows uses the generation-logprob geomean as confidence and
    # stamps completion_probabilities_geomean (real), while the sandbox verdict
    # remains the correctness label — so confidence and correctness can diverge.
    conf, source = eval_tower._derive_question_confidence(
        scoring_method="code_execution",
        correct=False,  # label: test FAILED …
        probability_confidence=0.66,  # … but the model was 0.66-confident in it
        rubric_scores={},
    )
    assert conf == 0.66
    assert source == "completion_probabilities_geomean"

    # WITHOUT probability rows it falls back to the binary proxy (fail-closed).
    conf_fb, source_fb = eval_tower._derive_question_confidence(
        scoring_method="code_execution",
        correct=True,
        probability_confidence=None,
        rubric_scores={},
    )
    assert conf_fb == 1.0
    assert source_fb == "code_execution_binary_proxy"


def test_derive_question_confidence_rubric_unchanged_by_esc7() -> None:
    # ESC-7 leaves rubric alone: its confidence IS the rubric aggregate and
    # overrides any probability row (rubric n_probs stays suppressed anyway).
    scores = {"factual_accuracy": 0.8, "reasoning_trajectory": 0.6}
    expected = eval_tower.aggregate_rubric_score(dict(scores)).score
    conf, source = eval_tower._derive_question_confidence(
        scoring_method="rubric",
        correct=True,
        probability_confidence=0.99,  # ignored for rubric
        rubric_scores=scores,
    )
    assert conf == expected
    assert source == "rubric_score"


def test_aggregate_confidence_is_real_fails_closed_on_mixed_proxy_batch() -> None:
    # ESC-7 Option A: a batch mixing a real-code geomean row + a real-math geomean
    # row is confidence_is_real=True (all sources == completion_probabilities_geomean).
    # The moment ONE code_execution question falls back to the binary proxy, the
    # whole batch flips to confidence_is_real=False (fail-closed accounting).
    tower = EvalTower()

    def _q(qid, source, conf, method):
        return eval_tower.QuestionResult(
            question_id=qid,
            suite="mixed",
            prompt="p",
            expected="e",
            qid=qid,
            answer="a",
            correct=True,
            error=None,
            tokens_generated=1,
            elapsed_s=0.1,
            route_used="fake",
            cost_tier=0,
            scoring_method=method,
            confidence=conf,
            confidence_source=source,
        )

    real_code = _q("c1", "completion_probabilities_geomean", 0.7, "code_execution")
    real_math = _q("m1", "completion_probabilities_geomean", 0.8, "math_verify")
    proxy_code = _q("c2", "code_execution_binary_proxy", 1.0, "code_execution")

    all_real = tower._aggregate([real_code, real_math], tier=1)
    assert all_real.details["confidence_is_real"] is True
    assert all_real.details["confidence_source_counts"] == {
        "completion_probabilities_geomean": 2
    }

    mixed = tower._aggregate([real_code, real_math, proxy_code], tier=1)
    assert mixed.details["confidence_is_real"] is False
    assert mixed.details["confidence_source_counts"] == {
        "completion_probabilities_geomean": 2,
        "code_execution_binary_proxy": 1,
    }


def test_eval_question_code_execution_confidence_is_binary_not_pass_rate(monkeypatch) -> None:
    # SCORE-12 end-to-end: a dataset row carrying pass_rate:0.9 must NOT inject a
    # constant 0.9 confidence into the calibration inputs.
    # ESC-7 extension Option A (operator 2026-07-21): n_probs is now REQUESTED
    # for code_execution, but when the response carries NO completion_probabilities
    # (as here) the confidence falls back to the binary correctness proxy.
    tower = EvalTower()
    seen: dict[str, object] = {}

    def _fake_call(**kwargs):  # noqa: ANN001
        seen.update(kwargs)
        return {"answer": "def f():\n    return 1\n", "tokens_generated": 5, "model": "fake"}

    monkeypatch.setattr(eval_tower, "call_orchestrator_forced", _fake_call)
    monkeypatch.setattr(eval_tower, "_is_scoreable_question", lambda _q: True)
    monkeypatch.setattr(eval_tower, "score_answer_deterministic", lambda **_k: True)

    with eval_tower.httpx.Client(timeout=1) as client:
        result = tower._eval_question(
            {
                "id": "ce-1",
                "suite": "code",
                "prompt": "Write f.",
                "expected": "1",
                "scoring_method": "code_execution",
                "scoring_config": {"pass_rate": 0.9},
            },
            client,
        )

    assert seen["n_probs"] == 5  # ESC-7 Option A: now requested for code_execution
    assert result.correct is True
    # No probability rows returned → binary proxy (float(correct)), NOT phantom 0.9.
    assert result.confidence == 1.0
    assert result.confidence_source == "code_execution_binary_proxy"


def test_eval_question_code_execution_confidence_uses_geomean_when_probs_present(
    monkeypatch,
) -> None:
    # ESC-7 extension Option A (operator 2026-07-21): when the code_execution
    # response carries completion_probabilities, the model's token-level
    # generation-logprob geomean becomes the CONFIDENCE (source
    # completion_probabilities_geomean), while the sandbox verdict stays the
    # correctness LABEL. Such a row is real-confidence and, alone, keeps the
    # aggregate confidence_is_real=True.
    tower = EvalTower()
    seen: dict[str, object] = {}

    def _fake_call(**kwargs):  # noqa: ANN001
        seen.update(kwargs)
        return {
            "answer": "def f():\n    return 1\n",
            "tokens_generated": 2,
            "model": "fake",
            "completion_probabilities": [
                {"content": "def", "probs": [{"tok_str": "def", "prob": 0.9}]},
                {"content": " f", "probs": [{"tok_str": " f", "prob": 0.49}]},
            ],
        }

    monkeypatch.setattr(eval_tower, "call_orchestrator_forced", _fake_call)
    monkeypatch.setattr(eval_tower, "_is_scoreable_question", lambda _q: True)
    monkeypatch.setattr(eval_tower, "score_answer_deterministic", lambda **_k: True)

    with eval_tower.httpx.Client(timeout=1) as client:
        result = tower._eval_question(
            {
                "id": "ce-2",
                "suite": "code",
                "prompt": "Write f.",
                "expected": "1",
                "scoring_method": "code_execution",
            },
            client,
        )

    assert seen["n_probs"] == 5  # requested
    assert result.correct is True  # label = sandbox verdict
    assert result.confidence == pytest.approx(math.sqrt(0.9 * 0.49))  # geomean
    assert result.confidence_source == "completion_probabilities_geomean"

    aggregate = tower._aggregate([result], tier=1)
    assert aggregate.details["confidence_source_counts"] == {
        "completion_probabilities_geomean": 1
    }
    assert aggregate.details["confidence_is_real"] is True


def test_eval_question_null_scoring_config_does_not_error(monkeypatch) -> None:
    # SCORE-12 guard: `scoring_config: null` must not raise AttributeError and
    # turn a scored question into an error result.
    tower = EvalTower()

    def _fake_call(**_kwargs):  # noqa: ANN001
        return {"answer": "Canberra", "tokens_generated": 2, "model": "fake"}

    monkeypatch.setattr(eval_tower, "call_orchestrator_forced", _fake_call)
    with eval_tower.httpx.Client(timeout=1) as client:
        result = tower._eval_question(
            {
                "id": "nc-1",
                "suite": "unit",
                "prompt": "Capital of Australia?",
                "expected": "Canberra",
                "scoring_method": "substring",
                "scoring_config": None,
            },
            client,
        )

    assert result.error is None
    assert result.correct is True


def test_eval_question_forwards_native_tool_schema_when_present(monkeypatch) -> None:
    tower = EvalTower()
    calls: list[dict] = []

    def _fake_call(**kwargs):  # noqa: ANN001
        calls.append(kwargs)
        return {
            "answer": "<answer>ok</answer>",
            "tokens_generated": 7,
            "tools_used": 1,
            "tools_called": ["get_eval_secret"],
        }

    monkeypatch.setattr(eval_tower, "call_orchestrator_forced", _fake_call)
    tool_schema = [
        {
            "type": "function",
            "function": {
                "name": "get_eval_secret",
                "parameters": {"type": "object", "properties": {"name": {"type": "string"}}},
            },
        }
    ]
    tool_choice = {"type": "function", "function": {"name": "get_eval_secret"}}

    with eval_tower.httpx.Client(timeout=1) as client:
        result = tower._eval_question(
            {
                "id": "native-tool",
                "suite": "tool_use_native",
                "prompt": "Use the provided tool.",
                "expected": "ok",
                "scoring_method": "substring",
                "force_mode": "repl",
                "tools": tool_schema,
                "tool_choice": tool_choice,
            },
            client,
        )

    assert result.correct is True
    assert result.tools_called == ["get_eval_secret"]
    assert calls[0]["tools"] == tool_schema
    assert calls[0]["tool_choice"] == tool_choice


def test_eval_question_stamps_eval_batch_request_metadata(monkeypatch) -> None:
    tower = EvalTower()
    calls: list[dict] = []

    def _fake_call(**kwargs):  # noqa: ANN001
        calls.append(kwargs)
        return {"answer": "ok", "tokens_generated": 1}

    monkeypatch.setattr(eval_tower, "call_orchestrator_forced", _fake_call)

    with eval_tower.httpx.Client(timeout=1) as client:
        tower._eval_question(
            {
                "id": "q-meta",
                "suite": "unit",
                "prompt": "Say ok.",
                "expected": "ok",
                "_eval_batch_id": "evaltower-T1-123-1q",
                "allow_delegation": False,
            },
            client,
        )

    assert calls[0]["request_priority"] == "background"
    assert calls[0]["workload_class"] == "eval_batch"
    assert calls[0]["batch_id"] == "evaltower-T1-123-1q"
    assert calls[0]["allow_delegation"] is False


def test_eval_question_forwards_prompt_root_when_present(monkeypatch) -> None:
    tower = EvalTower()
    calls: list[dict] = []

    def _fake_call(**kwargs):  # noqa: ANN001
        calls.append(kwargs)
        return {"answer": "ok", "tokens_generated": 1}

    monkeypatch.setattr(eval_tower, "call_orchestrator_forced", _fake_call)

    with eval_tower.httpx.Client(timeout=1) as client:
        tower._eval_question(
            {
                "id": "gepa",
                "suite": "general",
                "prompt": "Say ok.",
                "expected": "ok",
                "_prompt_root": "/tmp/gepa-prompt-root",
            },
            client,
        )

    assert calls[0]["prompt_root"] == "/tmp/gepa-prompt-root"


def test_eval_question_omits_native_tool_schema_by_default(monkeypatch) -> None:
    tower = EvalTower()
    calls: list[dict] = []

    def _fake_call(**kwargs):  # noqa: ANN001
        calls.append(kwargs)
        return {"answer": "ok", "tokens_generated": 1}

    monkeypatch.setattr(eval_tower, "call_orchestrator_forced", _fake_call)

    with eval_tower.httpx.Client(timeout=1) as client:
        tower._eval_question(
            {
                "id": "legacy",
                "suite": "general",
                "prompt": "Say ok.",
                "expected": "ok",
            },
            client,
        )

    assert "tools" not in calls[0]
    assert "tool_choice" not in calls[0]


def test_aggregate_emits_rubric_process_means() -> None:
    tower = EvalTower()

    out = tower._aggregate(
        [
            QuestionResult(
                question_id="q1",
                suite="deep_research_browsecomp",
                prompt="a",
                expected="",
                correct=True,
                rubric_scores={
                    "reasoning_trajectory": 0.6,
                    "tool_calls": 0.3,
                    "outline": 0.9,
                    "content_stage": 0.5,
                    "factual_accuracy": 0.8,
                },
            ),
            QuestionResult(
                question_id="q2",
                suite="deep_research_browsecomp",
                prompt="b",
                expected="",
                correct=False,
                rubric_scores={
                    "reasoning_trajectory": 1.0,
                    "tool_calls": 0.9,
                    "outline": 0.1,
                    "content_stage": 0.7,
                    "factual_accuracy": 0.2,
                },
            ),
            QuestionResult(
                question_id="q3",
                suite="math",
                prompt="c",
                expected="c",
                correct=True,
            ),
        ],
        tier=1,
    )

    assert out.rubric_reasoning_trajectory == pytest.approx(0.8)
    assert out.rubric_tool_calls == pytest.approx(0.6)
    assert out.rubric_outline == pytest.approx(0.5)
    assert out.rubric_content_stage == pytest.approx(0.6)
    assert out.details["rubric_dimension_means"]["factual_accuracy"] == pytest.approx(0.5)
    assert out.details["rubric_n_questions"] == 2
    lines = out.to_grep_lines(trial_id=9, species="rubric")
    assert "METRIC rubric_reasoning_trajectory: 0.8000" in lines
    assert "METRIC rubric_tool_calls: 0.6000" in lines
    assert not math.isnan(out.rubric_content_stage)
