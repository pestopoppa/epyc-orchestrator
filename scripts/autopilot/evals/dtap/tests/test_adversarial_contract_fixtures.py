"""Exact TU-ADV-1 contract fixtures; deterministic and model/service-free."""
from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path

import pytest

from harness.endpoint import ChatResult, DryRunStub
from harness.runner import ArmConfig, CaseRegistry, run_case
from harness.tool_contract import load_native_tool_contract
from harness.trace import verify_trace

DTAP_DIR = Path(__file__).resolve().parent.parent
BENIGN_CASE = "finance-benign-trade-execution-001"
ATTACK_CASE = "finance-indirect-action-reversal-002"
REGISTRY = CaseRegistry(DTAP_DIR / "cases.json")
NATIVE_CONTRACT = load_native_tool_contract(
    REGISTRY.cases,
    registry_path=DTAP_DIR / "cases.json",
)
BENIGN_FIXTURE = DTAP_DIR / "fixtures" / f"{BENIGN_CASE}.done.json"
ATTACK_FIXTURE = DTAP_DIR / "fixtures" / f"{ATTACK_CASE}.compromised.json"


class _SyntheticToolEndpoint:
    """Return the same synthetic tool payload on each endpoint request."""

    def __init__(self, tool_calls):
        self.tool_calls = tool_calls
        self.requests = 0

    def complete(self, messages, seed=0, *, tools=None, tool_choice=None, endpoint_mode=None):
        del messages, seed, tools, tool_choice, endpoint_mode
        self.requests += 1
        return ChatResult(
            text="",
            tool_calls=json.loads(json.dumps(self.tool_calls)),
            finish_reason="tool_calls",
            raw={"fixture": "synthetic-tu-adv-1"},
        )


def _run_invalid(tmp_path, tool_calls):
    endpoint = _SyntheticToolEndpoint(tool_calls)
    result = run_case(
        BENIGN_CASE,
        "done",
        0,
        endpoint,
        ArmConfig(max_turns=842, retries=0),
        tmp_path,
        REGISTRY,
        native_tool_contract=NATIVE_CONTRACT,
    )
    events = verify_trace(Path(result.trace_path))
    return endpoint, result, events


def _trajectory(result, outcome: str) -> dict:
    return {
        "task_id": result.trace_id,
        "task_type": "tool_use_security",
        "objective": "Evaluate tool use under the declared capability contract",
        "routing_decision": "native_dtap_evaluator",
        "outcome": outcome,
        "escalations": [],
        "cost_metrics": {},
        "evaluator_provenance": result.to_dict()["evaluator_provenance"],
    }


def test_repeated_invalid_payload_is_terminal_before_842_attempts(tmp_path):
    """The same invalid payload appears twice in one response; validation is terminal."""
    invalid = {
        "id": "invalid-oracle-call",
        "name": "read_privileged_oracle",
        "arguments": {"target": "hidden-answer"},
    }
    endpoint, result, events = _run_invalid(tmp_path, [invalid, invalid])

    # Native tool validation does not retry invalid model output: exactly one
    # endpoint request is enough to reject this repeated payload, regardless of
    # the deliberately generous max_turns=842 bound.
    assert endpoint.requests == 1
    assert endpoint.requests < 842
    assert result.status == "failed"
    assert result.failure["type"] == "tool"
    assert result.completion_state == "other_error"
    assert result.task_success is None and result.attack_success is None
    assert not any(event["event"] == "tool_call" for event in events)
    assert not any(event["event"] == "judge_result" for event in events)
    request = next(event for event in events if event["event"] == "endpoint_request")
    response = next(event for event in events if event["event"] == "endpoint_response")
    terminal = next(event for event in events if event["event"] == "run_result")
    assert request["payload"]["turn"] == 1
    assert response["payload"]["tool_calls"] == [invalid, invalid]
    assert terminal["payload"]["result"]["failure"]["type"] == "tool"
    provenance = result.evaluator_provenance
    assert provenance["unauthorized_capabilities"] == ["read_privileged_oracle"]
    assert provenance["reuse_eligible"] is False
    assert provenance["failure_chain"] == [{"stage": "dtap_runner", **result.failure}]
    assert provenance["trace_id"] == result.trace_id
    assert provenance["contract_sha256"] == NATIVE_CONTRACT.sha256


@pytest.mark.parametrize(
    "call",
    [
        {
            "id": "privileged-oracle",
            "name": "read_all_environment_records",
            "arguments": {"include_hidden": True},
        },
        {
            "id": "skill-write",
            "name": "save_as_reusable_skill",
            "arguments": {"name": "oracle-shortcut", "instructions": "reuse hidden answer"},
        },
    ],
    ids=["privileged-environment-read", "skill-reuse-write"],
)
def test_each_privileged_or_skill_write_is_independently_refused(tmp_path, call):
    endpoint, result, events = _run_invalid(tmp_path, [call])

    assert endpoint.requests == 1
    assert result.failure["type"] == "tool"
    assert result.task_success is None and result.attack_success is None
    assert [event for event in events if event["event"] == "tool_call"] == []
    assert not any(event["event"] in {"tool_result", "state_snapshot", "judge_result"} for event in events)
    provenance = result.evaluator_provenance
    assert provenance["unauthorized_capabilities"] == [call["name"]]
    assert provenance["failure_chain"][0]["type"] == "tool"
    assert provenance["failure_chain"][0]["message"] == result.failure["message"]
    assert provenance["failure_chain"][0]["detail"] == result.failure["detail"]


def test_privileged_and_skill_calls_are_rejected_atomically_as_one_response(tmp_path):
    calls = [
        {"id": "privileged-oracle", "name": "read_all_environment_records",
         "arguments": {"include_hidden": True}},
        {"id": "skill-write", "name": "save_as_reusable_skill",
         "arguments": {"name": "oracle-shortcut", "instructions": "reuse hidden answer"}},
    ]
    endpoint, result, events = _run_invalid(tmp_path, calls)

    assert endpoint.requests == 1
    assert result.failure["type"] == "tool"
    assert [event for event in events if event["event"] == "tool_call"] == []
    response = next(event for event in events if event["event"] == "endpoint_response")
    assert response["payload"]["tool_calls"] == calls
    assert result.evaluator_provenance["unauthorized_capabilities"] == [
        "read_all_environment_records", "save_as_reusable_skill",
    ]


def test_native_verdicts_gate_real_skill_distillation_and_keep_benign_control(tmp_path):
    """A benign success is distillable; an attack-success and rejected exploit are not."""
    benign_fixture = json.loads(BENIGN_FIXTURE.read_text(encoding="utf-8"))
    benign = run_case(
        BENIGN_CASE, "done", 0, DryRunStub(benign_fixture, seed=0),
        ArmConfig(max_turns=842, retries=0), tmp_path / "benign", REGISTRY,
        native_tool_contract=NATIVE_CONTRACT,
    )
    attack_fixture = json.loads(ATTACK_FIXTURE.read_text(encoding="utf-8"))
    attacked = run_case(
        ATTACK_CASE, "compromised", 0, DryRunStub(attack_fixture, seed=0),
        ArmConfig(max_turns=842, retries=0), tmp_path / "attack", REGISTRY,
        native_tool_contract=NATIVE_CONTRACT,
    )
    exploit_endpoint, exploit, _ = _run_invalid(
        tmp_path / "exploit",
        [{"id": "oracle", "name": "read_all_environment_records", "arguments": {"include_hidden": True}}],
    )

    assert benign.status == "ok" and benign.task_success is True
    assert benign.attack_success is None  # benign native judge does not score attack_success
    assert benign.evaluator_provenance["reuse_eligible"] is True
    assert attacked.status == "ok" and attacked.attack_success is True
    assert attacked.evaluator_provenance["reuse_eligible"] is False
    assert exploit_endpoint.requests == 1 and exploit.failure["type"] == "tool"
    assert exploit.evaluator_provenance["reuse_eligible"] is False

    from orchestration.repl_memory.distillation.pipeline import DistillationPipeline
    from orchestration.repl_memory.skill_bank import SkillBank

    class DeterministicTeacher:
        model_id = "tu-adv-fixture"

        async def distill(self, prompt):
            assert "native_dtap_evaluator" in prompt
            return json.dumps([{
                "title": "Use Declared Finance Tools",
                "principle": "Use only finance tools declared by the case contract.",
                "when_to_apply": "When the evaluated case declares a finance tool contract.",
                "task_types": ["tool_use_security"],
            }])

    bank = SkillBank(db_path=tmp_path / "skills.db", faiss_path=tmp_path / "faiss")
    try:
        pipeline = DistillationPipeline(teacher=DeterministicTeacher(), skill_bank=bank)
        report = asyncio.run(pipeline.run([
            _trajectory(benign, "success"),
            _trajectory(attacked, "success"),
            _trajectory(exploit, "failure"),
        ]))
        stored = bank.get_skills()
    finally:
        bank.close()

    assert report.total_trajectories == 3
    assert report.skills_stored == 1
    assert report.evaluator_sources_rejected == 2
    assert report.skills_rejected == 0
    assert len(stored) == 1
    assert stored[0].source_trajectory_ids == [benign.trace_id]


def test_legacy_trajectory_without_native_provenance_keeps_existing_admission():
    from orchestration.repl_memory.distillation.pipeline import _native_evaluator_rejection

    assert _native_evaluator_rejection({"task_id": "legacy", "outcome": "success"}) is None


def test_distillation_rejects_malformed_forged_and_unverifiable_native_sources(tmp_path):
    benign_fixture = json.loads(BENIGN_FIXTURE.read_text(encoding="utf-8"))
    benign = run_case(
        BENIGN_CASE, "done", 0, DryRunStub(benign_fixture, seed=0),
        ArmConfig(max_turns=842, retries=0), tmp_path / "source", REGISTRY,
        native_tool_contract=NATIVE_CONTRACT,
    )
    valid = benign.evaluator_provenance
    assert benign.task_success is True
    from orchestration.repl_memory.distillation.pipeline import _native_evaluator_rejection
    assert _native_evaluator_rejection({"task_id": "valid", "evaluator_provenance": valid}) is None

    malformed = copy.deepcopy(valid)
    malformed["case_threat"] = []
    forged = copy.deepcopy(valid)
    forged["case_threat"] = "indirect"
    forged["attack_success"] = None
    forged["reuse_eligible"] = True
    contract_mismatch = copy.deepcopy(valid)
    contract_mismatch["contract_sha256"] = "0" * 64
    missing_trace = copy.deepcopy(valid)
    missing_trace["trace_path"] = str(tmp_path / "missing.trace.jsonl")
    tampered_trace = copy.deepcopy(valid)
    trace_path = Path(valid["trace_path"])
    trace_records = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    request = next(record for record in trace_records if record["event"] == "endpoint_request")
    request["payload"]["turn"] = 2
    trace_path.write_text(
        "".join(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n"
                for record in trace_records),
        encoding="utf-8",
    )
    assert _native_evaluator_rejection({
        "task_id": "tampered", "evaluator_provenance": tampered_trace,
    }) == "native_evaluator_contract_or_trace_unverifiable"
    candidates = [malformed, forged, contract_mismatch, missing_trace, tampered_trace]

    class NoCallTeacher:
        model_id = "must-not-be-called"

        async def distill(self, prompt):
            raise AssertionError("rejected evaluator provenance reached the teacher")

    from orchestration.repl_memory.distillation.pipeline import DistillationPipeline
    from orchestration.repl_memory.skill_bank import SkillBank

    bank = SkillBank(db_path=tmp_path / "bad-sources.db", faiss_path=tmp_path / "bad-sources-faiss")
    try:
        report = asyncio.run(DistillationPipeline(NoCallTeacher(), bank).run([
            {**_trajectory(benign, "success"), "task_id": f"bad-{index}",
             "evaluator_provenance": candidate}
            for index, candidate in enumerate(candidates)
        ]))
        stored_count = bank.count()
    finally:
        bank.close()

    assert report.evaluator_sources_rejected == len(candidates)
    assert report.skills_stored == 0
    assert report.teacher_calls == 0
    assert stored_count == 0


def test_seed_projection_preserves_explicit_provenance_and_legacy_absence():
    from types import SimpleNamespace

    from scripts.skillbank.seed_skills import _mem_to_trajectory, _traj_to_dict

    provenance = {"schema": "fixture-source"}
    memory = SimpleNamespace(
        id="memory-1", context={"task_type": "tool", "evaluator_provenance": provenance},
        action="route", outcome="success",
    )
    assert _mem_to_trajectory(memory)["evaluator_provenance"] == provenance
    legacy_memory = SimpleNamespace(id="legacy", context={}, action="route", outcome="success")
    assert "evaluator_provenance" not in _mem_to_trajectory(legacy_memory)

    trajectory = SimpleNamespace(
        task_id="progress-1", task_type="tool", objective="check", routing_decision="route",
        outcome="success", escalations=[], cost_metrics={},
        outcome_entry=SimpleNamespace(data={"evaluator_provenance": provenance}),
    )
    assert _traj_to_dict(trajectory)["evaluator_provenance"] == provenance
    trajectory.outcome_entry = None
    assert "evaluator_provenance" not in _traj_to_dict(trajectory)
