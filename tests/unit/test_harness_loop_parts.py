"""HIL-6h concrete sibling parts: editors, credibility judge, fenced /chat evaluator, report rules.

Every transport is an injected fake: no test reaches a network, a model, or a CLI.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from src.harness_loop import substrate
from src.harness_loop.chat_evaluator import ChatArmEvaluator, EvaluationRefused, ItemScore
from src.harness_loop.credibility_judge import (
    CREDIBILITY_RESPONSE_SCHEMA,
    CROSS_FAMILY_JUDGE,
    PINNED_JUDGE,
    CredibilityMatrixJudge,
    JudgeRefused,
    JudgeResponseInvalid,
    cloud_cli_transport,
)
from src.harness_loop.driver import DECOY_CANARY, CycleSpec, SiblingLoopDriver, TripwireTripped
from src.harness_loop.editor import (
    GEPA_SOURCE_PROMPT_SHA256,
    LOCAL_EDITOR_META_PROMPT,
    LOCAL_EDITOR_META_PROMPT_SHA256,
    LOCAL_EDITOR_ROLE,
    EditorMetaPrompt,
    FamilyCorrelationRefused,
    LocalEditorProposer,
    ProposerRefused,
    ReferenceEditorProposer,
    family_correlation_note,
    render_editor_prompt,
)
from src.harness_loop.interfaces import (
    ArmResult,
    ArmSpec,
    Candidate,
    ProposerView,
    SplitRef,
    TokenCount,
    TraceRef,
    outcome_keys_in,
    sha256_text,
)
from src.harness_loop.pilot import PilotDesign, unmet_requirements
from src.harness_loop.report import (
    ADEQUATE,
    ADOPTION_UNKNOWN,
    EXERCISED,
    NOT_EXERCISED,
    UNDERPOWERED,
    adoption_readout,
    power_readout,
    result_labels,
)
from src.harness_loop.transport import (
    LOCAL_EGRESS,
    EgressClearance,
    EgressRefused,
    TransportError,
    http_json_transport,
    registry_role_model,
    require_clearance,
    token_count_from_usage,
)
from src.typed_decisions.cloud_judges import CloudJudgeSpec, load_registry

LOCAL_MODEL = "Qwen3.6-35B-A3B-MTP-Q8_0"
FEATURE = "HS4-3.6-summary-prompt"
TARGET = "orchestration/prompts/compaction.md"
INCUMBENT = "Summarise the session.\nKeep tool outputs short.\n"
ANTHROPIC_EGRESS = "anthropic (OAuth via claude CLI)"
PINS = {"tool_surface_sha256": "a" * 64, "served_context_window": 65536,
        "compaction_policy_sha256": "c" * 64, "wall_time_budget_s": 900.0, "turn_budget": 12,
        "final_result_selection_rule": "last FINAL()"}


def _view(text=INCUMBENT, excerpt="worker truncated the tool output at 4 KiB", rejected=()):
    return ProposerView(FEATURE, "3.6", TARGET, "prompt", text,
                        traces=(TraceRef("t1", "worker", ("truncation", "tool_error"), excerpt),),
                        selection_rule="first 20 traces with any L2 event, by trace id",
                        rejected_history=tuple(rejected))


class FakeV1:
    """Records bodies; replies with an OpenAI chat-completions shaped object."""

    def __init__(self, content, usage=None):
        self.content = content
        self.usage = usage
        self.bodies = []

    def __call__(self, body):
        self.bodies.append(json.loads(json.dumps(body)))
        out = {"choices": [{"index": 0, "message": {"role": "assistant", "content": self.content}}]}
        if self.usage is not None:
            out["usage"] = self.usage
        return out


def _reply(*texts, rationale="tighter wording"):
    return json.dumps({"candidates": [{"text": t, "rationale": rationale} for t in texts]})


# ── transport seam ───────────────────────────────────────────────────────────


def test_token_count_never_invents():
    full = token_count_from_usage({"prompt_tokens": 900, "completion_tokens": 40,
                                   "prompt_tokens_details": {"cached_tokens": 600}}, "proposer")
    assert (full.uncached_prefill, full.cached_prefill, full.decode, full.seat) == (300, 600, 40, "proposer")
    assert full.total() == 940
    # No cache count reported: the prompt cannot be split, so prefill stays "not captured".
    split_unknown = token_count_from_usage({"prompt_tokens": 900, "completion_tokens": 40}, "proposer")
    assert (split_unknown.uncached_prefill, split_unknown.cached_prefill, split_unknown.decode) == (None, None, 40)
    assert split_unknown.total() is None
    # The orchestrator reports an unreported count as 0: that is "not captured", never zero.
    zeros = token_count_from_usage({"prompt_tokens": 0, "completion_tokens": 0}, "judge")
    assert zeros.decode is None and zeros.total() is None
    assert token_count_from_usage(None, "judge") == TokenCount(seat="judge")


def test_egress_clearance_rules():
    require_clearance(None, LOCAL_EGRESS)  # nothing leaves the host
    with pytest.raises(EgressRefused):
        require_clearance(None, ANTHROPIC_EGRESS)
    c = EgressClearance(ANTHROPIC_EGRESS, "suite-x", "operator", frozenset({"q1"}))
    require_clearance(c, ANTHROPIC_EGRESS, ["q1"])
    with pytest.raises(EgressRefused):
        require_clearance(c, ANTHROPIC_EGRESS, ["q1", "q2"])
    with pytest.raises(EgressRefused):
        require_clearance(c, "openai (ChatGPT account via codex CLI)", ["q1"])
    with pytest.raises(ValueError):
        EgressClearance(ANTHROPIC_EGRESS, "", "operator")


def test_http_transport_posts_to_the_given_base_without_a_socket():
    import httpx

    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"ok": True})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    post = http_json_transport("/v1/chat/completions", base_url="http://api.test", client=client)
    assert post({"model": "frontdoor"}) == {"ok": True}
    assert seen == {"url": "http://api.test/v1/chat/completions", "body": {"model": "frontdoor"}}

    def bad(request):
        return httpx.Response(503, json={"detail": "role_parked"})

    post = http_json_transport("/chat", base_url="http://api.test",
                               client=httpx.Client(transport=httpx.MockTransport(bad)))
    with pytest.raises(TransportError):
        post({})


def test_registry_role_model_reads_the_registry(tmp_path):
    reg = tmp_path / "model_registry.yaml"
    reg.write_text("roles:\n  frontdoor:\n    model:\n      name: Qwen3.6-35B-A3B-MTP-Q8_0\n")
    assert registry_role_model("frontdoor", reg) == LOCAL_MODEL
    assert registry_role_model("absent", reg) is None
    assert registry_role_model("frontdoor", tmp_path / "missing.yaml") is None
    # The real runtime view names a model for the default local-editor role.
    assert registry_role_model(LOCAL_EDITOR_ROLE)


# ── editors ───────────────────────────────────────────────────────────────────


def test_meta_prompt_is_pinned_and_ported_from_the_published_source():
    assert LOCAL_EDITOR_META_PROMPT.sha256 == LOCAL_EDITOR_META_PROMPT_SHA256
    with pytest.raises(ValueError):
        replace(LOCAL_EDITOR_META_PROMPT, template=LOCAL_EDITOR_META_PROMPT.template + " ")
    with pytest.raises(ValueError):
        EditorMetaPrompt("x", "s", "no placeholders", sha256_text(json.dumps(["s", "no placeholders"])))
    # item 7: the port carries no outcome/feedback slot
    assert "feedback" not in LOCAL_EDITOR_META_PROMPT.template.lower()
    gepa = pytest.importorskip("gepa.strategies.instruction_proposal")
    src = gepa.InstructionProposalSignature.default_prompt_template
    assert sha256_text(src) == GEPA_SOURCE_PROMPT_SHA256
    assert src.splitlines()[0] == LOCAL_EDITOR_META_PROMPT.template.splitlines()[0]


def test_local_editor_request_shape_is_outcome_free(tmp_path):
    reg = tmp_path / "model_registry.yaml"
    reg.write_text(f"roles:\n  frontdoor:\n    model:\n      name: {LOCAL_MODEL}\n")
    fake = FakeV1(_reply("Summarise the session.\nQuote tool errors verbatim.\n"),
                  usage={"prompt_tokens": 800, "completion_tokens": 60,
                         "prompt_tokens_details": {"cached_tokens": 500}})
    p = LocalEditorProposer(transport=fake, registry_path=str(reg), seed=7)
    assert p.model_id == LOCAL_MODEL and p.role == "frontdoor"
    out = p.propose(_view(rejected=["- Keep tool outputs short.\n+ Drop tool outputs.\n"]),
                    max_candidates=3)
    body = fake.bodies[0]
    assert body["model"] == body["x_force_role"] == "frontdoor"
    assert body["x_tool_mode"] == "client" and body["seed"] == 7 and body["stream"] is False
    assert "eval_fence" not in body  # the proposer is not an eval arm
    assert outcome_keys_in(body) == []
    prompt = body["messages"][1]["content"]
    assert INCUMBENT.strip() in prompt and "worker truncated the tool output" in prompt
    assert "truncation, tool_error" in prompt and "Drop tool outputs" in prompt
    assert DECOY_CANARY not in json.dumps(body)
    assert body["messages"][0]["content"] == LOCAL_EDITOR_META_PROMPT.system
    (cand,) = out.candidates
    assert cand.proposer_model == LOCAL_MODEL and cand.original_text == INCUMBENT
    assert cand.rationale == "tighter wording" and cand.candidate_id.startswith(FEATURE + ":")
    assert out.tokens == (TokenCount(300, 60, 500, "proposer"),)
    assert p.last_parse.status == "parsed"
    assert p.identity()["meta_prompt_sha256"] == LOCAL_EDITOR_META_PROMPT_SHA256


def test_decoy_canary_or_answer_marker_never_reaches_the_transport():
    fake = FakeV1(_reply("x"))
    p = LocalEditorProposer(transport=fake, model_id=LOCAL_MODEL, answer_markers=("GOLD-ANSWER-",))
    with pytest.raises(TripwireTripped):
        p.propose(_view(excerpt=f"rubric says {DECOY_CANARY}"), max_candidates=2)
    with pytest.raises(TripwireTripped):
        p.propose(_view(excerpt="GOLD-ANSWER-42"), max_candidates=2)
    with pytest.raises(TripwireTripped):
        p.propose(ProposerView(FEATURE, "3.6", TARGET, "prompt", INCUMBENT,
                               traces=(TraceRef("t2", "worker", ("score",)),),
                               selection_rule="all"), max_candidates=2)
    assert fake.bodies == []


def test_quoted_text_cannot_inject_a_placeholder():
    view = _view(text="Use <trajectory_index> literally.\n")
    prompt = render_editor_prompt(view, max_candidates=2)
    assert "Use <trajectory_index> literally." in prompt


def test_parse_drops_noops_duplicates_and_overflow_and_reports_unparseable():
    texts = [INCUMBENT, "A\n", "A\n", "", "B\n", "C\n"]
    fake = FakeV1(json.dumps({"candidates": [{"text": t} for t in texts] + ["junk"]}))
    p = LocalEditorProposer(transport=fake, model_id=LOCAL_MODEL)
    out = p.propose(_view(), max_candidates=2)
    assert [c.mutated_text for c in out.candidates] == ["A\n", "B\n"]
    assert all(c.rationale == "" for c in out.candidates)
    assert p.last_parse.status == "partial"
    assert {d.split(":")[1] for d in p.last_parse.dropped} == {
        "no_op", "duplicate", "empty_text", "over_max_candidates", "not_an_object"}
    # tokens not reported -> not captured
    assert out.tokens[0].total() is None and out.tokens[0].seat == "proposer"

    p = LocalEditorProposer(transport=FakeV1("I cannot help with that."), model_id=LOCAL_MODEL)
    assert p.propose(_view(), max_candidates=2).candidates == ()
    assert p.last_parse.status == "unparseable"
    p = LocalEditorProposer(transport=lambda body: {"choices": []}, model_id=LOCAL_MODEL)
    assert p.propose(_view(), max_candidates=2).candidates == ()
    assert p.last_parse.status == "no_response"
    p = LocalEditorProposer(transport=FakeV1("```json\n" + _reply("D\n") + "\n```"), model_id=LOCAL_MODEL)
    assert [c.mutated_text for c in p.propose(_view(), max_candidates=2).candidates] == ["D\n"]


def test_local_editor_refuses_unknown_role_and_cloud_egress(tmp_path):
    reg = tmp_path / "r.yaml"
    reg.write_text("roles: {}\n")
    with pytest.raises(ProposerRefused):
        LocalEditorProposer(transport=FakeV1(""), registry_path=str(reg))
    with pytest.raises(ProposerRefused):
        LocalEditorProposer(transport=FakeV1(""), model_id=LOCAL_MODEL, egress=ANTHROPIC_EGRESS)


def test_reference_editor_family_correlation_check():
    clearance = EgressClearance(ANTHROPIC_EGRESS, "suite-x", "operator")
    kw = dict(transport=FakeV1(_reply("R\n")), judge_model="claude-sonnet-5-5",
              local_editor_model=LOCAL_MODEL, egress=ANTHROPIC_EGRESS, egress_clearance=clearance)
    with pytest.raises(FamilyCorrelationRefused):
        ReferenceEditorProposer(model_id="claude-opus-5-5", **kw)
    ref = ReferenceEditorProposer(model_id="claude-opus-5-5", correlation_acknowledged=True, **kw)
    assert "anthropic" in ref.correlation_note and ref.identity()["correlation_note"]
    out = ref.propose(_view(), max_candidates=1)
    assert out.candidates[0].proposer_model == "claude-opus-5-5"
    assert "x_force_role" not in ref.last_request
    with pytest.raises(ProposerRefused):  # P2: different from the local editor
        ReferenceEditorProposer(model_id=LOCAL_MODEL, **kw)
    with pytest.raises(ProposerRefused):  # P3: never the judge itself
        ReferenceEditorProposer(model_id="claude-sonnet-5-5", correlation_acknowledged=True, **kw)
    with pytest.raises(EgressRefused):  # content leaves the host only with a clearance
        ReferenceEditorProposer(model_id="claude-opus-5-5", correlation_acknowledged=True,
                                **{**kw, "egress_clearance": None})
    openai_ref = ReferenceEditorProposer(
        model_id="gpt-5.6-terra", **{**kw, "egress": "openai (ChatGPT account via codex CLI)",
                                     "egress_clearance": EgressClearance(
                                         "openai (ChatGPT account via codex CLI)", "suite-x", "op")})
    assert openai_ref.correlation_note is None
    assert family_correlation_note("gpt-5.6-terra", "claude-sonnet-5-5") is None


# ── credibility judge ─────────────────────────────────────────────────────────


def _cands(*ids):
    return [Candidate(i, FEATURE, "3.6", TARGET, "prompt", INCUMBENT, INCUMBENT + i, LOCAL_MODEL,
                      rationale=f"SELF-REPORT-{i}") for i in ids]


class FakeJudgeTransport:
    def __init__(self, payload, usage=None):
        self.payload = payload
        self.usage = usage
        self.calls = []

    def __call__(self, spec, prompt, schema, model):
        self.calls.append({"spec": spec, "prompt": prompt, "schema": schema, "model": model})
        return {"payload": self.payload, "usage": self.usage, "resolved_model": model}


def _matrix_payload(grid, chosen):
    return {"rows": [{"candidate_id": c, "cells": [{"instance_id": i, "verdict": v, "reason": "r"}
                                                   for i, v in row.items()]}
                     for c, row in grid.items()], "chosen": chosen}


def _judge(transport, **over):
    kw = dict(transport=transport, proposer_models=[LOCAL_MODEL], local_editor_model=LOCAL_MODEL,
              egress_clearance=EgressClearance(ANTHROPIC_EGRESS, "suite-x", "operator"),
              evidence_fn=lambda c, iid: f"{c.candidate_id} on {iid}: answered", task_fn=lambda iid: f"task {iid}")
    kw.update(over)
    return CredibilityMatrixJudge(**kw)


def test_judge_reads_the_pinned_registry_entry_and_surfaces_verified_live():
    reg = load_registry()
    assert reg[PINNED_JUDGE].model == "claude-sonnet-5-5"
    assert reg[CROSS_FAMILY_JUDGE].model.startswith("gpt-")
    j = _judge(FakeJudgeTransport({}))
    assert j.model_id == "claude-sonnet-5-5" and j.egress == ANTHROPIC_EGRESS
    if not reg[PINNED_JUDGE].verified_live:
        assert j.verified_live is False and any("verified_live=false" in c for c in j.caveats)


def test_judge_matrix_is_complete_validated_and_argmax_checked():
    grid = {"a": {"q1": "credible", "q2": "credible"}, "b": {"q1": "not_credible", "q2": "unresolved"}}
    fake = FakeJudgeTransport(_matrix_payload(grid, "a"))
    j = _judge(fake)
    m = j.judge(_cands("a", "b"), ["q1", "q2"])
    m.validate(["a", "b"], ["q1", "q2"])
    assert m.chosen == "a" and m.choice_is_argmax() is True and m.judge_model == "claude-sonnet-5-5"
    assert m.tokens == (TokenCount(seat="judge"),) and m.tokens[0].total() is None
    call = fake.calls[0]
    assert call["schema"] is CREDIBILITY_RESPONSE_SCHEMA and call["model"] == "claude-sonnet-5-5"
    assert "SELF-REPORT" not in call["prompt"]  # item 17c: the rationale never reaches the judge
    assert "a on q2: answered" in call["prompt"] and "task q1" in call["prompt"]
    rec = j.call_log[-1]
    assert rec["egress"] == ANTHROPIC_EGRESS and rec["verified_live"] == j.verified_live
    assert rec["outcome"] == "ok" and rec["argmax_ok"] is True and rec["clearance"]["suite_id"] == "suite-x"
    # a named choice that is not the argmax of its own counts is kept and flagged, not corrected
    j2 = _judge(FakeJudgeTransport(_matrix_payload(grid, "b")))
    m2 = j2.judge(_cands("a", "b"), ["q1", "q2"])
    assert m2.chosen == "b" and m2.choice_is_argmax() is False and j2.call_log[-1]["argmax_ok"] is False


@pytest.mark.parametrize("payload", [
    _matrix_payload({"a": {"q1": "credible"}}, None),                               # empty cell
    _matrix_payload({"a": {"q1": "credible", "q2": "maybe"}}, None),                # bad verdict
    _matrix_payload({"a": {"q1": "credible", "q2": "credible"}, "z": {"q1": "credible"}}, None),
    _matrix_payload({"a": {"q1": "credible", "q2": "credible"}}, "z"),              # chosen unknown
    {"rows": [{"candidate_id": "a", "cells": [
        {"instance_id": "q1", "verdict": "credible", "reason": ""},
        {"instance_id": "q1", "verdict": "credible", "reason": ""},
        {"instance_id": "q2", "verdict": "credible", "reason": ""}]}], "chosen": None},  # duplicate
    "not an object",
])
def test_judge_refuses_an_incomplete_or_malformed_matrix(payload):
    j = _judge(FakeJudgeTransport(payload))
    with pytest.raises(JudgeResponseInvalid):
        j.judge(_cands("a"), ["q1", "q2"])
    assert j.call_log[-1]["outcome"] == "failed" and j.call_log[-1]["egress"] == ANTHROPIC_EGRESS


def test_judge_enforces_p3_and_egress_at_construction():
    t = FakeJudgeTransport({})
    with pytest.raises(JudgeRefused):  # unpinned alias
        _judge(t, judge_name="sonnet-low")
    with pytest.raises(JudgeRefused):  # judge model is also a proposer
        _judge(t, proposer_models=[LOCAL_MODEL, "claude-sonnet-5-5"])
    with pytest.raises(JudgeRefused):  # same family as the local editor
        _judge(t, local_editor_model="claude-haiku-5", proposer_models=["claude-haiku-5"])
    with pytest.raises(JudgeRefused):  # P3 caution 1, not acknowledged
        _judge(t, reference_editor_model="claude-opus-5-5")
    j = _judge(t, reference_editor_model="claude-opus-5-5", correlation_acknowledged=True)
    assert j.correlation_note and j.correlation_note in j.caveats
    with pytest.raises(EgressRefused):
        _judge(t, egress_clearance=None)
    with pytest.raises(EgressRefused):
        _judge(t, egress_clearance=EgressClearance("openai (ChatGPT account via codex CLI)", "s", "op"))
    # the cross-family option clears an Anthropic reference editor without an acknowledgement
    openai_egress = load_registry()[CROSS_FAMILY_JUDGE].egress
    jx = _judge(t, judge_name=CROSS_FAMILY_JUDGE, reference_editor_model="claude-opus-5-5",
                egress_clearance=EgressClearance(openai_egress, "suite-x", "operator"))
    assert jx.correlation_note is None


def test_judge_refuses_instances_outside_the_clearance_before_any_call():
    fake = FakeJudgeTransport(_matrix_payload({"a": {"q1": "credible", "q9": "credible"}}, "a"))
    j = _judge(fake, egress_clearance=EgressClearance(ANTHROPIC_EGRESS, "suite-x", "operator",
                                                      frozenset({"q1"})))
    with pytest.raises(EgressRefused):
        j.judge(_cands("a"), ["q1", "q9"])
    assert fake.calls == [] and j.call_log[-1]["outcome"] == "refused"


def test_cloud_cli_transport_wraps_the_registry_runner_with_an_injected_process():
    spec = CloudJudgeSpec(name=PINNED_JUDGE, transport="claude_cli", model="claude-sonnet-5-5",
                          egress=ANTHROPIC_EGRESS)
    payload = _matrix_payload({"a": {"q1": "credible"}}, "a")
    seen = {}

    def run_fn(argv, stdin_text, timeout_s, cwd):
        seen["argv"] = argv
        return 0, json.dumps({"structured_output": payload, "modelUsage": {"claude-sonnet-5-5": {}}}), ""

    out = cloud_cli_transport(run_fn)(spec, "prompt", CREDIBILITY_RESPONSE_SCHEMA, "claude-sonnet-5-5")
    assert out["payload"] == payload and out["usage"] is None
    assert "--json-schema" in seen["argv"] and "claude-sonnet-5-5" in seen["argv"]

    def failing(argv, stdin_text, timeout_s, cwd):
        return 1, "", "boom"

    with pytest.raises(TransportError):
        cloud_cli_transport(failing)(spec, "p", CREDIBILITY_RESPONSE_SCHEMA, "claude-sonnet-5-5")


# ── fenced /chat evaluator ────────────────────────────────────────────────────

ITEMS = {f"q{i}": {"prompt": f"question {i}", "gold": f"g{i}"} for i in range(4)}


def _arm(text=INCUMBENT, arm_id="inc"):
    return ArmSpec(arm_id, "incumbent", FEATURE, TARGET, sha256_text(text), dict(PINS))


class FakeChat:
    def __init__(self, fence_state="active", fail=(), tokens=30):
        self.fence_state = fence_state
        self.fail = set(fail)
        self.tokens = tokens
        self.bodies = []

    def __call__(self, body):
        self.bodies.append(dict(body))
        qid = body["request_id"].rsplit("-", 1)[-1]
        if qid in self.fail:
            raise TransportError("connection reset")
        out = {"answer": f"g{qid[1:]}" if qid != "q3" else "wrong", "turns": 1,
               "tokens_generated": self.tokens, "tools_called": ["compact"] if qid != "q0" else []}
        if self.fence_state is not None:
            out["eval_fence"] = {"state": self.fence_state, "touched_paths": [], "denied_paths": [],
                                 "denied_count": 0, "enforcement": "hook-only"}
        return out


def _builder(arm, arm_text, item):
    return {"prompt": item["prompt"], "force_role": "frontdoor", "force_mode": "direct",
            "x_orchestrator_prompt_root": f"/scratch/{arm.content_sha256[:12]}"}


def _scorer(item, response):
    return ItemScore(passed=response["answer"] == item["gold"],
                     fired="compact" in response.get("tools_called", []))


def _evaluator(chat, **over):
    kw = dict(transport=chat, items=ITEMS, request_builder=_builder, scorer=_scorer,
              content_for=lambda arm: INCUMBENT, metric="accuracy", unit="fraction",
              direction="higher_better", verification_signal="none",
              infra_fingerprint_fn=lambda: {"regime": "cpu-full"})
    kw.update(over)
    return ChatArmEvaluator(**kw)


def test_evaluator_fills_the_arm_result_through_a_fenced_chat():
    chat = FakeChat(fail={"q2"})
    split = SplitRef("V", "validation", frozenset(ITEMS))
    r = _evaluator(chat).evaluate(_arm(), split)
    assert all(b["eval_fence"] is True and b["workload_class"] == "eval_batch" for b in chat.bodies)
    assert len(chat.bodies) == 4
    assert (r.arm_id, r.split_id, r.metric, r.unit, r.direction) == ("inc", "V", "accuracy", "fraction", "higher_better")
    assert r.item_outcomes == {"q0": True, "q1": True, "q2": None, "q3": False}
    assert (r.scored, r.attempted) == (3, 4) and r.value == pytest.approx(2 / 3)
    assert r.treatment_fired == 2  # q1, q3 fired; q0 did not; q2 never answered
    assert r.fence_state == "active" and r.verification_signal == "none"
    assert r.infra_fingerprint == {"regime": "cpu-full"}
    (tok,) = r.tokens
    assert tok.seat == "solver" and tok.decode == 90 and tok.uncached_prefill is None and tok.total() is None


@pytest.mark.parametrize("state", ["unarmed", None])
def test_evaluator_refuses_a_non_active_fence_echo(state):
    with pytest.raises(EvaluationRefused):
        _evaluator(FakeChat(fence_state=state)).evaluate(_arm(), SplitRef("V", "validation", frozenset(ITEMS)))


def test_evaluator_refuses_a_disarm_and_unpinned_bytes():
    chat = FakeChat()
    split = SplitRef("V", "validation", frozenset(ITEMS))
    ev = _evaluator(chat, request_builder=lambda a, t, i: {"prompt": i["prompt"], "eval_fence": False})
    with pytest.raises(ValueError):
        ev.evaluate(_arm(), split)
    with pytest.raises(EvaluationRefused):
        _evaluator(chat, content_for=lambda arm: INCUMBENT + "tampered").evaluate(_arm(), split)
    with pytest.raises(EvaluationRefused):
        _evaluator(chat).evaluate(_arm(), SplitRef("X", "validation", frozenset({"q99"})))
    assert chat.bodies == []


def test_evaluator_leaves_unobservable_adoption_unknown():
    ev = _evaluator(FakeChat(tokens=0), scorer=lambda item, resp: ItemScore(True, None))
    r = ev.evaluate(_arm(), SplitRef("V", "validation", frozenset(ITEMS)))
    assert r.treatment_fired is None and r.tokens[0].decode is None
    assert ev.last_diagnostics.fired_unobservable == sorted(ITEMS)
    all_failed = _evaluator(FakeChat(fail=set(ITEMS))).evaluate(_arm(), SplitRef("V", "validation", frozenset(ITEMS)))
    assert all_failed.scored == 0 and all_failed.value is None and all_failed.fence_state == "unobserved"


def test_evaluator_constructor_validates_declarations():
    for bad in ({"direction": "up"}, {"verification_signal": "maybe"}, {"unit": ""}):
        with pytest.raises(ValueError):
            _evaluator(FakeChat(), **bad)


# ── the parts compose with the driver (fakes only) ────────────────────────────


def test_parts_run_one_driver_cycle_on_fakes(tmp_path):
    def split(sid, role, a, b):
        return SplitRef(sid, role, frozenset(f"q{i}" for i in range(a, b)))

    design = PilotDesign(
        feature_id=FEATURE, home="3.6", primary_metric="accuracy", primary_unit="fraction",
        primary_direction="higher_better", train=split("T1", "train", 10, 14),
        validation=split("V", "validation", 0, 4), arrival=split("T2", "arrival", 20, 24),
        test=split("TEST", "test", 30, 34), cycles=2, stopping_rule="2 cycles",
        local_editor_model=LOCAL_MODEL, reference_editor_model="gpt-5.6-terra",
        judge_model="cloud:" + PINNED_JUDGE, matched_rollout_baseline=True, unmodified_control=True,
        verification_arms=frozenset({"with", "none", "self_judged"}), runs=2, mde=0.05, paired_n=4,
        expected_discordant=2, min_win_share=0.6, token_seats=frozenset({"proposer", "judge", "solver"}),
        headroom_note="n", exclusions_declared=True, regression_set_id="r", per_role_arm_set=True,
        context_policy_controls=frozenset({"no_management", "wording_only"}))
    assert unmet_requirements(design) == []
    better = INCUMBENT + "Quote tool errors verbatim.\n"
    texts = {sha256_text(t): t for t in (INCUMBENT, better)}
    items = {f"q{i}": {"prompt": f"p{i}", "gold": "ok"} for i in list(range(0, 4)) + list(range(20, 24))}

    def chat(body):
        good = body["x_orchestrator_prompt_root"].endswith(sha256_text(better)[:12])
        qn = int(body["request_id"].rsplit("-q", 1)[-1])
        return {"answer": "ok" if (good or qn % 2 == 0) else "no", "tokens_generated": 10,
                "tools_called": ["compact"], "eval_fence": {"state": "active", "denied_count": 0}}

    def builder(arm, arm_text, item):
        return {"prompt": item["prompt"], "x_orchestrator_prompt_root": f"/scratch/{arm.content_sha256[:12]}"}

    evaluator = _evaluator(chat, items=items, request_builder=builder,
                           content_for=lambda arm: texts[arm.content_sha256], verification_signal="with",
                           infra_fingerprint_fn=None)
    proposer = LocalEditorProposer(transport=FakeV1(_reply("Quote tool errors verbatim.\n" + INCUMBENT, better),
                                                    usage={"prompt_tokens": 500, "completion_tokens": 50,
                                                           "prompt_tokens_details": {"cached_tokens": 100}}),
                                   model_id=LOCAL_MODEL)
    texts[sha256_text("Quote tool errors verbatim.\n" + INCUMBENT)] = "Quote tool errors verbatim.\n" + INCUMBENT

    class GridTransport:
        def __call__(self, spec, prompt, schema, model):
            ids = [c["candidate_id"] for c in json.loads(prompt.split("):\n", 1)[1])["candidates"]]
            grid = {c: {f"q{i}": "credible" for i in range(4)} for c in ids}
            return {"payload": _matrix_payload(grid, ids[0]), "usage": None}

    judge = _judge(GridTransport(), reference_editor_model="gpt-5.6-terra",
                   proposer_models=[LOCAL_MODEL, "gpt-5.6-terra"])
    journal = substrate.open_loop_journal(tmp_path / "loop")
    drv = SiblingLoopDriver(design=design, journal=journal, journal_dir=tmp_path / "loop",
                            proposer=proposer, evaluator=evaluator, judge=judge, leakage=lambda t: None)
    inc = ArmSpec("inc1", "incumbent", FEATURE, TARGET, sha256_text(INCUMBENT), dict(PINS))
    out = drv.run_cycle(_view(), CycleSpec(1, design.train, inc, INCUMBENT, dict(PINS), max_candidates=2))
    assert out.outcome == "keep" and out.committed_text == better
    assert out.judge_argmax_ok is True
    assert out.tokens_by_seat["proposer"] == 550 and out.tokens_by_seat["judge"] is None
    assert out.tokens_by_seat["solver"] is None  # /chat reports no prefill: not captured, not zero
    assert out.arrival is not None and out.arrival.fence_state == "active"


# ── report rules ──────────────────────────────────────────────────────────────


def _result(outcomes, fired=1, arm_id="a"):
    return ArmResult(arm_id, "V", "accuracy", None, "fraction", "higher_better",
                     scored=sum(v is not None for v in outcomes.values()), attempted=len(outcomes),
                     item_outcomes=outcomes, tokens=(TokenCount(seat="solver"),), treatment_fired=fired,
                     fence_state="active")


def test_zero_adoption_is_labelled_not_exercised():
    r0 = adoption_readout(_result({"q1": True}, fired=0))
    assert r0.label == NOT_EXERCISED and r0.null_informative is False and "uninformative" in r0.note
    assert adoption_readout(_result({"q1": True}, fired=None)).label == ADOPTION_UNKNOWN
    assert adoption_readout(_result({"q1": True}, fired=1)).label == EXERCISED
    labels = result_labels(_result({"q1": True}, fired=0), correlation_note="P3 note", caveats=("c",))
    assert labels["adoption"] == NOT_EXERCISED and labels["correlation_note"] == "P3 note"
    assert labels["decision_grade"] == "observation"


def test_near_ceiling_power_statement_and_underpowered_label():
    inc = _result({f"q{i}": i >= 5 for i in range(100)})  # pass share 0.95, 5 incumbent failures
    p = power_readout(inc, paired_n=100, expected_discordant=None, min_win_share=None)
    assert p.near_ceiling and p.label == UNDERPOWERED and p.headroom_items == 5
    p = power_readout(inc, paired_n=100, expected_discordant=20, min_win_share=0.65)
    assert p.required_wins == 13 and p.label == UNDERPOWERED
    p = power_readout(inc, paired_n=100, expected_discordant=8, min_win_share=0.6)
    assert p.required_wins == 5 and p.label == ADEQUATE
    cand_flat = _result({f"q{i}": i >= 5 for i in range(100)}, arm_id="b")
    p = power_readout(inc, paired_n=100, expected_discordant=8, min_win_share=0.6, candidate_validation=cand_flat)
    assert p.observed_discordant == 0 and p.label == UNDERPOWERED
    cand = _result({f"q{i}": (i >= 5) != (i < 8) or i < 5 for i in range(100)}, arm_id="c")
    p = power_readout(inc, paired_n=100, expected_discordant=5, min_win_share=0.6, candidate_validation=cand)
    assert p.observed_discordant >= 5 and p.label == ADEQUATE
    mid = _result({f"q{i}": i % 2 == 0 for i in range(10)})
    assert power_readout(mid, paired_n=10, expected_discordant=None, min_win_share=None).label == ADEQUATE
    assert power_readout(_result({"q1": None}), paired_n=1, expected_discordant=1,
                         min_win_share=0.5).label == UNDERPOWERED
