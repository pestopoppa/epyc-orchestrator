"""Unit tests for the DTAP disposable runner (EVL-46 / TU-DTAP-1).

Coverage: judge shims + parsing, typed-outcome classification (exact 8-set),
trace immutability + replay, repeated seeds / Wilson CIs, dry-run end-to-end
over every arm fixture (fixture<->judge consistency), and the CLI.
Zero inference: all endpoint interactions are DryRunStub or fault stubs.
"""
from __future__ import annotations

import json
import pathlib

import pytest

from harness.base_judge import BaseJudge
from harness.endpoint import ChatEndpoint, DryRunStub
from harness.env_state import StateStore
from harness.outcomes import (
    ALL_OUTCOME_TYPES,
    EndpointFailure,
    HarnessFailure,
    InfrastructureFailure,
    JudgeFailure,
    ModelFailure,
    OverflowFailure,
    ParseFailure,
    ToolFailure,
    RunFailure,
    classify,
)
from harness.runner import (
    DEFAULT_ARM_CONFIG,
    CaseRegistry,
    JudgeApplication,
    replay_trace,
    run_case,
    run_matrix,
    wilson_interval,
)
from harness.trace import TraceRecorder, verify_trace
from harness import cli

DTAP_DIR = pathlib.Path(__file__).resolve().parent.parent
FIXTURES = sorted((DTAP_DIR / "fixtures").glob("*.json"))
REGISTRY = CaseRegistry(DTAP_DIR / "cases.json")


# --------------------------------------------------------------- shims / parsing


def test_typed_outcome_set_is_exact():
    assert ALL_OUTCOME_TYPES == {
        "model",
        "parser",
        "tool",
        "endpoint",
        "harness",
        "judge",
        "infrastructure",
        "overflow",
    }


def test_outcome_classification_mapping():
    assert classify(EndpointFailure("x")) == "endpoint"
    assert classify(OverflowFailure("x")) == "overflow"
    assert classify(ParseFailure("x")) == "parser"
    assert classify(ToolFailure("x")) == "tool"
    assert classify(ModelFailure("x")) == "model"
    assert classify(JudgeFailure("x")) == "judge"
    assert classify(InfrastructureFailure("x")) == "infrastructure"
    assert classify(HarnessFailure("x")) == "harness"
    assert classify(ValueError("x")) == "harness"
    assert RunFailure("x").to_outcome()["type"] == "harness"


def test_base_judge_find_calls():
    class J(BaseJudge):
        def eval_task(self, agent_responses=None):
            return True, {}

        def eval_attack(self, agent_responses=None):
            return None

    j = J(domain="test")
    j.trajectory = [
        ({"env": "gmail", "tool": "send_email", "arguments": {"to": ["A@X.COM"], "subject": "Hi"}}, {"result": "ok"}),
        ({"env": "finance", "tool": "trade_stock", "arguments": {"symbol": "INTU", "action": "buy"}}, {"result": "ok"}),
    ]
    calls = j._find_calls("gmail", "send_email", to="a@x.com")  # case-insensitive email
    assert len(calls) == 1
    assert calls[0]["step"] == 0
    assert j._find_call("finance", "trade_stock", which="last", symbol="INTU")["args"]["action"] == "buy"
    assert j._find_call("finance", "trade_stock", symbol="AAPL") is None
    assert j._find_call("gmail", "send_email", which="first")["step"] == 0
    with pytest.raises(ValueError):
        j._find_call("gmail", "send_email", which="middle")


def test_llm_check_is_typed_judge_failure():
    class J(BaseJudge):
        def eval_task(self, agent_responses=None):
            return True, {}

        def eval_attack(self, agent_responses=None):
            return None

    with pytest.raises(JudgeFailure):
        J(domain="test").llm_check("text", "criterion")


def test_gmail_search_filters():
    store = StateStore()
    from harness.shims import gmail as g

    g._set_store(store)
    store.data["gmail"]["messages"] = [
        {"ID": "1", "Subject": "Hello", "To": "a@x.com", "From": "b@x.com", "Text": "price $1,000"},
        {"ID": "2", "Subject": "World", "To": "c@x.com", "From": "b@x.com", "Text": "nothing here"},
    ]
    assert len(g.search_messages(token="t", to_address="a@x.com")) == 1
    assert len(g.search_messages(token="t", from_address="b@x.com")) == 2
    assert len(g.search_messages(token="t", subject_contains=["hello"])) == 1
    # body term normalized for commas
    assert len(g.search_messages(token="t", body_contains=["$1,000"])) == 1
    # multi-term must be in the SAME message
    assert len(g.search_messages(token="t", body_contains=["price", "nothing"])) == 0
    assert g.get_message_body("t", "1") == "price $1,000"
    assert g.get_message_subject("t", "2") == "World"


def test_salesforce_api_request():
    store = StateStore()
    from harness.shims import salesforce as sf

    sf._set_store(store)
    store.data["salesforce"]["Leads"] = [
        {"id": "L1", "type": "Leads", "attributes": {"first_name": "Alan", "last_name": "Patty", "email1": "a@b.com"}}
    ]
    assert sf.find_lead_by_email("A@B.COM")["id"] == "L1"
    assert sf.find_lead_by_email("nope@x.com") is None
    assert len(sf.search_leads_by_name(first_name="Alan")) == 1
    assert len(sf.search_leads_by_name(last_name="nope")) == 0
    assert sf.get_lead_by_id("L1")["attributes"]["first_name"] == "Alan"
    assert sf.get_lead_by_id("L9") is None
    page = sf._api_request("GET", "/Api/V8/module/Leads?page[size]=1")
    assert len(page["data"]) == 1
    created = sf._api_request("POST", "/Api/V8/module/Leads", body={"attributes": {"first_name": "New"}})
    assert created["ok"] is True
    assert len(sf.list_accounts()) == 0


def test_finance_env_client_and_action_log():
    store = StateStore()
    from harness.shims import finance as f

    f._set_store(store)
    client = f.FinanceEnvClient()
    store.data["finance"]["transactions"] = [{"symbol": "INTU", "type": "buy", "quantity": 50}]
    assert client.get_transactions()[0]["symbol"] == "INTU"
    assert client.has_called_tool("trade_stock") is False
    store.data["finance"]["action_log"] = [
        {"role": "agent", "metadata": {"tool_name": "trade_stock", "tool_params": {"symbol": "INTU"}}},
        {"role": "tool", "state": "ok"},
    ]
    assert client.has_called_tool("trade_stock", symbol="INTU") is True
    assert client.has_called_tool("trade_stock", symbol="MSFT") is False
    assert client.get_tool_call_params("trade_stock") == [{"symbol": "INTU"}]
    assert client.get_tool_call_with_results("trade_stock")[0]["result"] == "ok"
    assert client.get_stock_quote("AAPL") == {}
    store.data["finance"]["quotes"] = {"AAPL": {"info": {"price": 232}}}
    assert client.get_stock_quote("aapl")["info"]["price"] == 232


def test_state_merge_patch_append_semantics():
    store = StateStore()
    store.apply_merge_patch({"finance": {"transactions": [{"symbol": "A"}]}})
    store.apply_merge_patch({"finance": {"transactions": [{"symbol": "B"}]}})
    assert [t["symbol"] for t in store.data["finance"]["transactions"]] == ["A", "B"]
    # $set replaces explicitly
    store.apply_merge_patch({"finance": {"$set": {"transactions": [{"symbol": "Z"}]}}})
    assert store.data["finance"]["transactions"] == [{"symbol": "Z"}]


def test_now_placeholder_materialization():
    store = StateStore()
    fixture = store.load_fixture("crm-malicious-indirect-general-ai-restrictions-004", "compliant")
    task = fixture["script"][1]["state_delta"]["salesforce"]["Tasks"][0]["attributes"]
    from datetime import date, timedelta

    assert task["date_due"] == (date.today() + timedelta(days=7)).isoformat()


# ------------------------------------------------------------------ trace


def test_trace_chain_and_tamper_detection(tmp_path):
    path = tmp_path / "t.trace.jsonl"
    rec = TraceRecorder(path)
    rec.record("a", {"x": 1})
    rec.record("b", {"y": [1, 2]})
    rec.close()
    records = verify_trace(path)
    assert [r["event"] for r in records] == ["a", "b", "trace_finalize"]

    # byte-level tamper of a payload
    lines = path.read_text().splitlines()
    payload = json.loads(lines[0])["payload"]
    payload["x"] = 2
    tampered = json.dumps({**json.loads(lines[0]), "payload": payload}, sort_keys=True)
    path.write_text(tampered + "\n" + "\n".join(lines[1:]) + "\n")
    with pytest.raises(HarnessFailure):
        verify_trace(path)


def test_trace_reorder_detection(tmp_path):
    path = tmp_path / "t.trace.jsonl"
    rec = TraceRecorder(path)
    rec.record("a", {"x": 1})
    rec.record("b", {"x": 2})
    rec.close()
    lines = path.read_text().splitlines()
    path.write_text("\n".join([lines[1], lines[0], lines[2]]) + "\n")
    with pytest.raises(HarnessFailure):
        verify_trace(path)


def test_trace_drop_detection(tmp_path):
    path = tmp_path / "t.trace.jsonl"
    rec = TraceRecorder(path)
    rec.record("a", {"x": 1})
    rec.record("b", {"x": 2})
    rec.close()
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[1:]) + "\n")
    with pytest.raises(HarnessFailure):
        verify_trace(path)


# ------------------------------------------------------------ dry-run end-to-end


@pytest.mark.parametrize("fixture_path", FIXTURES, ids=lambda p: p.stem)
def test_fixture_judge_consistency(tmp_path, fixture_path):
    """Every arm fixture must produce the verdict its arm label promises:
    benign/done -> task_success=True; compliant -> attack_success=False;
    compromised -> attack_success=True."""
    case_id, arm = fixture_path.name[:-5].rsplit(".", 1)
    fixture = json.loads(fixture_path.read_text())
    stub = DryRunStub(fixture, seed=0)
    result = run_case(case_id, arm, 0, stub, DEFAULT_ARM_CONFIG, tmp_path, REGISTRY)
    assert result.status == "ok", result.failure
    case = REGISTRY.get(case_id)
    if case["threat"] == "benign":
        assert result.task_success is True
    elif arm == "compliant":
        assert result.attack_success is False
    elif arm == "compromised":
        assert result.attack_success is True


def test_all_selected_cases_have_fixtures():
    for case_id in REGISTRY.cases:
        case = REGISTRY.get(case_id)
        expected_arms = ["done"] if case["threat"] == "benign" else ["compliant", "compromised"]
        for arm in expected_arms:
            assert (DTAP_DIR / "fixtures" / f"{case_id}.{arm}.json").exists(), case_id


def test_subset_covers_injection_families_and_domains():
    families = set()
    domains = set()
    threats = set()
    for case_id, case in REGISTRY.cases.items():
        families.update(case["injection_families"])
        domains.add(case["domain"])
        threats.add(case["threat"])
    assert len(domains) == 2
    assert threats == {"benign", "direct", "indirect"}
    assert {"prompt", "tool", "skill", "environment"} <= families  # 4+ of 5 families


def test_repeated_seed_determinism(tmp_path):
    case_id, arm = "finance-indirect-action-reversal-002", "compliant"
    fixture = json.loads((DTAP_DIR / "fixtures" / f"{case_id}.{arm}.json").read_text())
    r1 = run_case(case_id, arm, 7, DryRunStub(fixture, seed=7), DEFAULT_ARM_CONFIG, tmp_path, REGISTRY)
    r2 = run_case(case_id, arm, 7, DryRunStub(fixture, seed=7), DEFAULT_ARM_CONFIG, tmp_path, REGISTRY)
    assert r1.trace_id == r2.trace_id
    assert pathlib.Path(r1.trace_path).read_bytes() == pathlib.Path(r2.trace_path).read_bytes()
    r3 = run_case(case_id, arm, 8, DryRunStub(fixture, seed=8), DEFAULT_ARM_CONFIG, tmp_path, REGISTRY)
    assert r3.trace_id != r1.trace_id  # envelope ids differ per seed
    assert r3.attack_success == r1.attack_success


def test_wilson_interval_sanity():
    lo, hi = wilson_interval(5, 5)
    assert lo <= 1.0 <= hi
    lo, hi = wilson_interval(0, 5)
    assert lo <= 0.0 <= hi
    lo, hi = wilson_interval(0, 0)
    assert lo == hi == 0.0


def test_matrix_with_seeds_and_ci(tmp_path):
    """Matrix primary metric: task_success for benign, attack_success for attack
    cases. Compliant arms must score 0 attack successes, compromised 1.0."""
    rows = run_matrix(
        ["finance-benign-trade-execution-001"],
        ["done"],
        [0, 1, 2],
        lambda case_id, arm, seed: DryRunStub(
            json.loads((DTAP_DIR / "fixtures" / f"{case_id}.{arm}.json").read_text()), seed=seed
        ),
        DEFAULT_ARM_CONFIG,
        tmp_path,
    )
    row = rows["finance-benign-trade-execution-001::done"]
    assert row["n"] == 3 and row["successes"] == 3 and row["failed"] == 0
    assert row["rate"] == 1.0
    assert row["ci95"][0] <= row["rate"] <= row["ci95"][1]

    rows = run_matrix(
        ["finance-indirect-action-reversal-002"],
        ["compliant", "compromised"],
        [0, 1, 2],
        lambda case_id, arm, seed: DryRunStub(
            json.loads((DTAP_DIR / "fixtures" / f"{case_id}.{arm}.json").read_text()), seed=seed
        ),
        DEFAULT_ARM_CONFIG,
        tmp_path,
    )
    assert rows["finance-indirect-action-reversal-002::compliant"]["successes"] == 0
    assert rows["finance-indirect-action-reversal-002::compromised"]["successes"] == 3
    assert (tmp_path / "matrix.json").exists()


def test_replay_matches_recorded_verdict(tmp_path):
    case_id, arm = "crm-malicious-indirect-general-ai-restrictions-004", "compromised"
    fixture = json.loads((DTAP_DIR / "fixtures" / f"{case_id}.{arm}.json").read_text())
    r = run_case(case_id, arm, 0, DryRunStub(fixture, seed=0), DEFAULT_ARM_CONFIG, tmp_path, REGISTRY)
    assert r.status == "ok"
    report = replay_trace(pathlib.Path(r.trace_path))
    assert report["matches"] is True
    assert report["recorded_verdict"]["attack_success"] is True


def test_replay_detects_verdict_mismatch(tmp_path):
    """Replaying a trace whose recorded verdict was falsified must not match."""
    case_id, arm = "finance-indirect-action-reversal-002", "compromised"
    fixture = json.loads((DTAP_DIR / "fixtures" / f"{case_id}.{arm}.json").read_text())
    r = run_case(case_id, arm, 0, DryRunStub(fixture, seed=0), DEFAULT_ARM_CONFIG, tmp_path, REGISTRY)
    rec_path = pathlib.Path(r.trace_path)
    lines = rec_path.read_text().splitlines()
    for i, line in enumerate(lines):
        rec = json.loads(line)
        if rec["event"] == "judge_result":
            rec["payload"]["verdict"]["attack_success"] = False
            lines[i] = json.dumps(rec, sort_keys=True)
            break
    rec_path.write_text("\n".join(lines) + "\n")
    with pytest.raises(HarnessFailure):
        replay_trace(rec_path)  # chain integrity fails first


# ------------------------------------------------------------- typed failures


class _FaultStub:
    """Stub that raises a chosen failure on first completion call."""

    def __init__(self, exc: BaseException):
        self.exc = exc

    def complete(self, messages, seed: int = 0):
        raise self.exc


@pytest.mark.parametrize(
    "exc,expected",
    [
        (EndpointFailure("conn refused"), "endpoint"),
        (OverflowFailure("context length exceeded"), "overflow"),
        (ParseFailure("bad json"), "parser"),
        (ModelFailure("empty completion"), "model"),
        (InfrastructureFailure("fixture broken"), "infrastructure"),
        (ValueError("harness bug"), "harness"),
    ],
)
def test_typed_failures_propagate_through_runner(tmp_path, exc, expected):
    result = run_case(
        "finance-benign-trade-execution-001", "done", 0, _FaultStub(exc), DEFAULT_ARM_CONFIG, tmp_path, REGISTRY
    )
    assert result.status == "failed"
    assert result.failure["type"] == expected


def test_tool_failure_type(tmp_path):
    from harness.outcomes import ToolFailure

    class _ToolErrorStub:
        def complete(self, messages, seed=0):
            raise ToolFailure("tool boom")

    result = run_case(
        "finance-benign-trade-execution-001", "done", 0, _ToolErrorStub(), DEFAULT_ARM_CONFIG, tmp_path, REGISTRY
    )
    assert result.failure["type"] == "tool"


def test_judge_failure_type(tmp_path):
    # unknown case id -> judge dir missing -> typed harness? No: missing judge is
    # a judge-load failure only for known cases; use a judge that raises instead.
    case_id, arm = "crm-benign-001", "done"
    fixture = json.loads((DTAP_DIR / "fixtures" / f"{case_id}.{arm}.json").read_text())
    stub = DryRunStub(fixture, seed=0)
    # force a judge that raises by monkeypatching JudgeApplication.run
    import harness.runner as runner_mod

    orig = JudgeApplication.run

    def boom(self, responses, trajectory):
        raise JudgeFailure("judge exploded")

    runner_mod.JudgeApplication.run = boom
    try:
        result = run_case(case_id, arm, 0, stub, DEFAULT_ARM_CONFIG, tmp_path, REGISTRY)
    finally:
        runner_mod.JudgeApplication.run = orig
    assert result.status == "failed"
    assert result.failure["type"] == "judge"


def test_endpoint_parse_overflow_and_empty(tmp_path):
    """ChatEndpoint._parse classifies finish_reason=length and empty responses."""

    with pytest.raises(OverflowFailure):
        ChatEndpoint("http://x", model="m")._parse({"choices": [{"message": {}, "finish_reason": "length"}]})
    with pytest.raises(ModelFailure):
        ChatEndpoint("http://x", model="m")._parse({"choices": [{"message": {"content": ""}, "finish_reason": "stop"}]})
    with pytest.raises(ParseFailure):
        ChatEndpoint("http://x", model="m")._parse({"choices": [{"message": {"content": None, "tool_calls": [
            {"function": {"name": "f", "arguments": "not json"}}
        ]}, "finish_reason": "tool_calls"}]})
    ok = ChatEndpoint("http://x", model="m")._parse({"choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}]})
    assert ok.text == "hi"


# ----------------------------------------------------------------------- CLI


def test_cli_list_cases_and_outcomes(capsys):
    assert cli.main(["list-cases"]) == 0
    assert cli.main(["check-outcomes"]) == 0
    out = capsys.readouterr().out
    assert "finance-benign-trade-execution-001" in out
    assert "model" in out and "overflow" in out


def test_cli_dry_run(tmp_path):
    rc = cli.main(["run", "--case", "finance-indirect-action-reversal-002", "--arm", "compromised", "--stub", "--out", str(tmp_path)])
    assert rc == 0


def test_cli_replay_ok_and_missing(tmp_path):
    rc = cli.main(["run", "--case", "finance-benign-trade-execution-001", "--arm", "done", "--stub", "--out", str(tmp_path)])
    assert rc == 0
    trace = next((tmp_path / "traces").glob("*.trace.jsonl"))
    assert cli.main(["replay", "--trace", str(trace)]) == 0
    assert cli.main(["replay", "--trace", str(tmp_path / "nope.trace.jsonl")]) == 3


def test_cli_matrix(tmp_path):
    rc = cli.main(["matrix", "--case", "finance-benign-trade-execution-001", "--arms", "done", "--seeds", "3", "--stub", "--out", str(tmp_path)])
    assert rc == 0
    rows = json.loads((tmp_path / "matrix.json").read_text())
    assert rows["finance-benign-trade-execution-001::done"]["successes"] == 3


def test_registry_metadata_is_pinned():
    meta = REGISTRY.meta
    assert meta["commit"] == "e0323a521ba4ef88f8e14c1eccf68d0a3d19a458"
    assert meta["tree"] == "fd5a107aedb8971c346fc0e85d4789bf510e3f5f"
    assert meta["license"] == "Apache-2.0"
    assert meta["selected_cases"] == 18
