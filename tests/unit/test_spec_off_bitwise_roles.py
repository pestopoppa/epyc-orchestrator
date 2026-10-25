"""Q38-T10: per-request ``speculative.n_max: 0`` for bitwise-required callers.

Operator ruling 2026-10-20 (per-role parity). The ``spec_off_bitwise_roles`` flag
defaults OFF; with it OFF every payload here must be unchanged. With it ON exactly
the bitwise-required callers (eval_tower llm_judge / rubric judge, model_grader,
coherence_judge, judge_redundancy + routing_replay CLIs) send ``n_max: 0``.
Serving callers (LlamaServerBackend with no override) never do.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT / "scripts" / "benchmark", REPO_ROOT / "scripts" / "autopilot"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import debug_scorer  # noqa: E402
import eval_tower  # noqa: E402
import seeding_orchestrator  # noqa: E402

from src import spec_policy  # noqa: E402
from src.backends.llama_server import LlamaServerBackend, ServerConfig  # noqa: E402
from src.features import _FEATURE_REGISTRY, reset_features  # noqa: E402
from src.model_server import InferenceRequest  # noqa: E402
from src.pipeline_monitor import model_grader  # noqa: E402
from src.registry_loader import (  # noqa: E402
    AccelerationConfig,
    MemoryConfig,
    ModelConfig,
    PerformanceMetrics,
    RoleConfig,
)

KEY = "speculative.n_max"
ENV = "ORCHESTRATOR_SPEC_OFF_BITWISE_ROLES"


@pytest.fixture(autouse=True)
def _flag_off_by_default(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.delenv("ORCHESTRATOR_FEATURE_SPEC_OFF_BITWISE_ROLES", raising=False)
    reset_features()
    yield
    reset_features()


@pytest.fixture
def flag_on(monkeypatch):
    monkeypatch.setenv(ENV, "1")
    reset_features()
    assert spec_policy.bitwise_roles_enabled() is True


@pytest.fixture
def role_config():
    return RoleConfig(
        name="test_role",
        tier="C",
        description="t",
        model=ModelConfig(name="m", path="m.gguf", quant="Q4_K_M", size_gb=1.0),
        acceleration=AccelerationConfig(type="baseline", temperature=0.0),
        performance=PerformanceMetrics(baseline_tps=10.0),
        memory=MemoryConfig(residency="hot"),
    )


# ── flag + helper ──────────────────────────────────────────────────────────


def test_flag_registered_default_on_in_prod_off_in_test():
    """Flipped ON in prod 2026-10-20 after the Q38-T10.1 control arm passed; test default stays OFF."""
    from src.features import get_features

    spec = next(s for s in _FEATURE_REGISTRY if s.name == "spec_off_bitwise_roles")
    assert spec.default_test is False and spec.default_prod is True
    assert get_features(production=True).spec_off_bitwise_roles is True
    assert get_features(production=False).spec_off_bitwise_roles is False
    assert spec_policy.bitwise_roles_enabled() is False
    assert spec_policy.bitwise_n_max() is None
    assert spec_policy.bitwise_call_kwargs() == {}


def test_server_key_is_the_flat_dotted_schema_key():
    assert spec_policy.SERVER_KEY == KEY


def test_flag_on_helpers(flag_on):
    assert spec_policy.bitwise_n_max() == 0
    assert spec_policy.bitwise_call_kwargs() == {"speculative_n_max": 0}


def test_apply_to_payload_noop_without_override():
    payload = {"prompt": "x", "n_predict": 4}
    before = json.dumps(payload, sort_keys=True)
    spec_policy.apply_to_payload(payload, None)
    assert json.dumps(payload, sort_keys=True) == before


# ── backend payloads: serving callers are never affected ───────────────────


def _backend(chat: bool) -> LlamaServerBackend:
    return LlamaServerBackend(config=ServerConfig(base_url="http://test:8080", use_chat_completions=chat))


def _capture_chat_payload(backend, role_config, request, stream: bool) -> dict:
    captured: dict = {}

    class _Stop(Exception):
        pass

    def _grab(*args, **kwargs):
        body = kwargs.get("json")
        if body is None and len(args) >= 3:
            body = args[2]
        captured.update(body or {})
        raise _Stop()

    target = "stream" if stream else "post"
    with patch.object(backend.client, target, side_effect=_grab):
        try:
            if stream:
                backend.infer_stream_text(role_config, request)
            else:
                backend._infer_chat_completions(role_config, request, 0.0)
        except Exception:
            pass
    return captured


def _all_lane_payloads(role_config, request) -> dict[str, dict]:
    return {
        "completion": _backend(False)._build_payload(role_config, request),
        "chat": _capture_chat_payload(_backend(True), role_config, request, stream=False),
        "chat_stream": _capture_chat_payload(_backend(True), role_config, request, stream=True),
    }


def test_flag_off_backend_payloads_unchanged_on_every_lane(role_config):
    base = InferenceRequest(role="test_role", prompt="Hello", n_tokens=16)
    explicit_none = InferenceRequest(role="test_role", prompt="Hello", n_tokens=16, speculative_n_max=None)
    a, b = _all_lane_payloads(role_config, base), _all_lane_payloads(role_config, explicit_none)
    for lane, payload in a.items():
        assert payload, f"{lane}: payload not captured"
        assert KEY not in payload, lane
        assert json.dumps(payload, sort_keys=True) == json.dumps(b[lane], sort_keys=True), lane


def test_flag_on_serving_backend_calls_still_carry_no_override(role_config, flag_on):
    """Serving paths never install a scope and never set the request field."""
    request = InferenceRequest(role="frontdoor", prompt="Hello", n_tokens=16)
    for lane, payload in _all_lane_payloads(role_config, request).items():
        assert payload and KEY not in payload, lane


def test_explicit_request_value_and_scope_reach_every_lane(role_config):
    request = InferenceRequest(role="test_role", prompt="Hello", n_tokens=16, speculative_n_max=0)
    for lane, payload in _all_lane_payloads(role_config, request).items():
        assert payload[KEY] == 0 and isinstance(payload[KEY], int), lane
    plain = InferenceRequest(role="test_role", prompt="Hello", n_tokens=16)
    with spec_policy.speculative_n_max_override(0):
        for lane, payload in _all_lane_payloads(role_config, plain).items():
            assert payload[KEY] == 0, lane
    # scope exited -> unchanged again
    for lane, payload in _all_lane_payloads(role_config, plain).items():
        assert KEY not in payload, lane


# ── ChatRequest + route plumbing ───────────────────────────────────────────


def test_chat_request_field_defaults_none_and_validates():
    from pydantic import ValidationError

    from src.api.models import ChatRequest

    assert ChatRequest(prompt="x").speculative_n_max is None
    assert ChatRequest(prompt="x", speculative_n_max=0).speculative_n_max == 0
    with pytest.raises(ValidationError):
        ChatRequest(prompt="x", speculative_n_max=-1)


def test_route_override_none_is_a_noop():
    with spec_policy.speculative_n_max_override(None):
        assert spec_policy.current_override() is None
    with spec_policy.speculative_n_max_override(0):
        assert spec_policy.current_override() == 0
    assert spec_policy.current_override() is None


# ── call_orchestrator_forced: direct + representative callers ──────────────

_CALLER_SHAPES = [
    pytest.param({}, id="seeding-default"),
    pytest.param({"scoring_method": "rubric_judge", "request_priority": "background",
                  "workload_class": "eval_batch", "allow_delegation": False}, id="rubric-judge"),
    pytest.param({"workload_class": "eval_batch", "n_probs": 5, "max_tokens": 64}, id="eval-generation"),
    pytest.param({"output_schema": {"type": "object"}}, id="model-grader"),
    pytest.param({"tools": [{"type": "function"}], "tool_choice": "auto"}, id="tool-mode"),
    pytest.param({"cache_prompt": False, "stop_sequences": ["</answer>"]}, id="stop-seqs"),
    pytest.param({"eval_fence": True, "batch_id": "b1"}, id="fenced-batch"),
]


def _forced_payload(**kwargs) -> dict:
    import httpx

    captured: dict = {}

    def _handler(request):
        captured["payload"] = json.loads(request.content)
        return httpx.Response(200, json={"answer": "ok"})

    with httpx.Client(transport=httpx.MockTransport(_handler)) as client:
        seeding_orchestrator.call_orchestrator_forced(
            prompt="p", force_role="worker_general", timeout=300, client=client, **kwargs
        )
    return captured["payload"]


@pytest.mark.parametrize("shape", _CALLER_SHAPES)
def test_call_orchestrator_forced_flag_off_payload_unchanged(shape, monkeypatch):
    monkeypatch.delenv("AUTOPILOT_EVAL_MIN_LLAMA_BUDGET_S", raising=False)
    without = _forced_payload(**shape)
    with_none = _forced_payload(speculative_n_max=None, **shape)
    assert "speculative_n_max" not in without
    without.pop("client_deadline_unix_s"), with_none.pop("client_deadline_unix_s")
    assert json.dumps(without, sort_keys=True) == json.dumps(with_none, sort_keys=True)


def test_call_orchestrator_forced_sends_field_when_given(monkeypatch):
    monkeypatch.delenv("AUTOPILOT_EVAL_MIN_LLAMA_BUDGET_S", raising=False)
    assert _forced_payload(speculative_n_max=0)["speculative_n_max"] == 0


# ── eval_tower rubric judge ────────────────────────────────────────────────


def _run_rubric(monkeypatch) -> list[dict]:
    calls: list[dict] = []

    def _fake(**kwargs):
        calls.append(kwargs)
        return {"answer": "", "error": "stop"}

    monkeypatch.setattr(eval_tower, "call_orchestrator_forced", _fake)
    monkeypatch.setenv("AUTOPILOT_RUBRIC_JUDGE_ROLES", "architect_critic")
    monkeypatch.setattr(eval_tower, "check_cross_family_status", lambda g, r: (True, "ok"))
    monkeypatch.setattr(eval_tower, "_load_orchestrator_debug_scorer", lambda: SimpleNamespace(CONSTRAIN_JUDGE_OUTPUT=False))
    eval_tower.EvalTower._rubric_scores_for_answer(
        SimpleNamespace(url="http://x", timeout=30, watcher=None),
        q={"prompt": "q", "expected_contains": ["a"]},
        answer="a",
        generator_model="gen",
        tool_events=[],
        client=None,
    )
    return calls


def test_rubric_judge_flag_off_call_has_no_field(monkeypatch):
    calls = _run_rubric(monkeypatch)
    assert calls and "speculative_n_max" not in calls[0]


def test_rubric_judge_flag_on_sends_zero(monkeypatch, flag_on):
    calls = _run_rubric(monkeypatch)
    assert calls and calls[0]["speculative_n_max"] == 0


def test_eval_generation_call_never_carries_field(monkeypatch, flag_on):
    """The eval GENERATION call (call_kwargs path) is not a judge; it must not opt in."""
    src = (REPO_ROOT / "scripts" / "autopilot" / "eval_tower.py").read_text()
    gen = src[src.index("resp = call_orchestrator_forced(**call_kwargs, **_bitwise_generation_spec_kwargs())") - 6000 :
              src.index("resp = call_orchestrator_forced(**call_kwargs, **_bitwise_generation_spec_kwargs())")]
    # Q38-T10.2(b): the only opt-in is the env-gated helper on the call line itself.
    assert "speculative_n_max" not in gen and "_bitwise_spec" not in gen


def test_eval_tower_opt_in_sites_are_exactly_the_two_rubric_judge_calls():
    src = (REPO_ROOT / "scripts" / "autopilot" / "eval_tower.py").read_text()
    sites = [m.start() for m in re.finditer(r"\*\*_bitwise_spec_kwargs\(\)", src)]
    assert len(sites) == 2
    for pos in sites:
        window = src[pos - 1500 : pos]
        assert 'scoring_method="rubric_judge' in window


# ── debug_scorer llm_judge (orchestrator + direct lanes) ───────────────────


def _judge_post(monkeypatch, config) -> dict:
    import httpx

    captured: dict = {}

    def _post(url, json=None, timeout=None, **kw):  # noqa: A002
        captured["url"], captured["json"] = url, json
        req = httpx.Request("POST", url)
        body = {"answer": "true"} if url.endswith("/chat") else {"choices": [{"message": {"content": "true"}}]}
        return httpx.Response(200, json=body, request=req)

    monkeypatch.setattr(httpx, "post", _post)
    debug_scorer.request_llm_judge_text("judge?", config, max_tokens=8, output_schema=None)
    return captured


def test_llm_judge_orchestrator_lane(monkeypatch):
    monkeypatch.delenv("LLM_JUDGE_ROLE", raising=False)
    off = _judge_post(monkeypatch, {})
    assert off["url"].endswith("/chat") and "speculative_n_max" not in off["json"]
    monkeypatch.setenv(ENV, "1")
    reset_features()
    on = _judge_post(monkeypatch, {})
    assert on["json"]["speculative_n_max"] == 0


def test_llm_judge_direct_server_lane(monkeypatch):
    cfg = {"judge_url": "http://127.0.0.1:59999"}
    off = _judge_post(monkeypatch, cfg)
    assert off["url"].endswith("/v1/chat/completions") and KEY not in off["json"]
    monkeypatch.setenv(ENV, "1")
    reset_features()
    on = _judge_post(monkeypatch, cfg)
    assert on["json"][KEY] == 0


# ── model_grader ───────────────────────────────────────────────────────────


def _grade(monkeypatch) -> dict:
    calls: list[dict] = []
    module = ModuleType("seeding_orchestrator")

    def call_orchestrator_forced(**kwargs):
        calls.append(kwargs)
        return {"answer": "A"}

    module.call_orchestrator_forced = call_orchestrator_forced
    monkeypatch.setitem(sys.modules, "seeding_orchestrator", module)
    model_grader.grade_answer(
        {"spec_name": "q", "prompt_template": "Q: {question}", "choice_strings": ["A"], "choice_scores": {"A": 1.0}},
        {"question_id": "q1"},
    )
    return calls[0]


def test_model_grader_flag_off_unchanged(monkeypatch):
    assert "speculative_n_max" not in _grade(monkeypatch)


def test_model_grader_flag_on_sends_zero(monkeypatch, flag_on):
    assert _grade(monkeypatch)["speculative_n_max"] == 0


# ── coherence_judge + CLIs: scope installed iff flag on ────────────────────


def _coherence_override_seen() -> list:
    from src.typed_decisions import coherence_judge as cj

    seen: list = []

    def _fake_score(*a, **k):
        seen.append(spec_policy.current_override())
        raise cj.JudgeRefused("probe", "probe", status_code=503)

    judge = cj.CoherenceJudge(
        primitives_fn=lambda: object(),
        holds_fn=lambda: [],
        served_model_fn=lambda p, r: "served",
        log_path=Path("/dev/null"),
    )
    with patch.object(cj, "_score_local", _fake_score):
        with pytest.raises(cj.JudgeRefused):
            judge.judge(cj.JudgeRequest(prompt="p", base_output="a", candidate_output="b", allow_uncalibrated=True))
    return seen


def test_coherence_judge_scope_off_then_on(monkeypatch):
    assert _coherence_override_seen() == [None]
    monkeypatch.setenv(ENV, "1")
    reset_features()
    assert _coherence_override_seen() == [0]


def _cli_override_seen(which: str) -> list:
    seen: list = []
    if which == "judge_redundancy":
        from src.typed_decisions import judge_redundancy as mod

        def _fake(*a, **k):
            seen.append(spec_policy.current_override())
            return {}

        with patch.object(mod, "run_judge_redundancy", _fake):
            assert mod.main(["--dry-run"]) == 0
    else:
        from src.typed_decisions import routing_replay as mod

        def _fake(*a, **k):
            seen.append(spec_policy.current_override())
            return {}

        with (
            patch.object(mod, "run_replay", _fake),
            patch.object(mod, "resolve_snapshot", lambda p: SimpleNamespace(db_path="x", provenance=lambda: {})),
            patch.object(mod, "load_rows", lambda p: []),
        ):
            assert mod.main(["--dry-run", "--snapshot", "x"]) == 0
    return seen


@pytest.mark.parametrize("which", ["judge_redundancy", "routing_replay"])
def test_replay_clis_scope_off_then_on(which, monkeypatch):
    assert _cli_override_seen(which) == [None]
    monkeypatch.setenv(ENV, "1")
    reset_features()
    assert _cli_override_seen(which) == [0]


# ── structural guard: only the declared callers opt in ─────────────────────

_ALLOWED_OPT_IN = {
    "src/spec_policy.py",
    "src/backends/llama_server.py",
    "src/api/routes/chat.py",
    "src/api/models/requests.py",
    "src/inference/model_server.py",
    "src/features.py",
    "src/pipeline_monitor/model_grader.py",
    "src/typed_decisions/coherence_judge.py",
    "src/typed_decisions/judge_redundancy.py",
    "src/typed_decisions/routing_replay.py",
    "scripts/benchmark/seeding_orchestrator.py",
    "scripts/benchmark/debug_scorer.py",
    "scripts/autopilot/eval_tower.py",
    "scripts/benchmark/q38_t10_1_control_arm.py",  # control script (not run here)
}
_NEEDLE = re.compile(r"bitwise_scope|bitwise_n_max|bitwise_call_kwargs|_bitwise_spec|speculative_n_max|spec_off_bitwise_roles")


def test_no_serving_path_opts_in():
    hits = set()
    for root in ("src", "scripts"):
        for path in (REPO_ROOT / root).rglob("*.py"):
            if "deprecated" in path.parts:
                continue
            try:
                if _NEEDLE.search(path.read_text()):
                    hits.add(str(path.relative_to(REPO_ROOT)))
            except (OSError, UnicodeDecodeError):
                continue
    assert hits <= _ALLOWED_OPT_IN, sorted(hits - _ALLOWED_OPT_IN)


# ── caller table proofs for call_orchestrator_forced (AST) ─────────────────


def test_speculative_n_max_is_last_defaulted_param_so_positional_calls_cannot_shift():
    import inspect

    params = list(inspect.signature(seeding_orchestrator.call_orchestrator_forced).parameters.values())
    assert params[-1].name == "speculative_n_max" and params[-1].default is None
    assert params[-2].name == "eval_fence"  # previous last param unchanged


def test_only_bitwise_callers_of_call_orchestrator_forced_opt_in():
    import ast

    opted: dict[str, int] = {}
    total = 0
    for root in ("src", "scripts"):
        for path in (REPO_ROOT / root).rglob("*.py"):
            if "deprecated" in path.parts:
                continue
            try:
                tree = ast.parse(path.read_text())
            except (SyntaxError, UnicodeDecodeError, OSError):
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                fn = node.func
                name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", "")
                if name != "call_orchestrator_forced":
                    continue
                total += 1
                for kw in node.keywords:
                    if kw.arg == "speculative_n_max" or (
                        kw.arg is None and "bitwise" in ast.unparse(kw.value)
                    ):
                        rel = str(path.relative_to(REPO_ROOT))
                        opted[rel] = opted.get(rel, 0) + 1
    assert total >= 12  # the table is not vacuous
    assert opted == {"scripts/autopilot/eval_tower.py": 3, "src/pipeline_monitor/model_grader.py": 1}


# ── Q38-T10.1 control-arm script (never launched here) ─────────────────────


def test_control_arm_script_builds_flat_key_and_refuses_production_ports(tmp_path, capsys):
    import importlib.util

    path = REPO_ROOT / "scripts" / "benchmark" / "q38_t10_1_control_arm.py"
    spec = importlib.util.spec_from_file_location("q38_t10_1_control_arm", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.SPEC_KEY == spec_policy.SERVER_KEY
    assert mod.body("p", 8, 0, 0)[KEY] == 0 and KEY not in mod.body("p", 8, 0, None)
    assert mod.main(["--out", str(tmp_path), "--port", "8083"]) == 2  # production port: refuses, starts nothing
    assert mod.main(["--out", str(tmp_path), "--print-commands"]) == 0
    assert "--spec-type none" in capsys.readouterr().out
    assert mod.first_div([1, 2, 3], [1, 2, 4]) == 2 and mod.first_div([1], [1]) is None


# ── Q38-T10.2(b): EvalTower generation opt-in (env-gated) ──────────────────


def test_eval_generation_opt_in_is_env_gated(monkeypatch):
    et = eval_tower
    monkeypatch.setattr(et, "_bitwise_spec_kwargs", lambda: {"speculative_n_max": 0})
    monkeypatch.delenv(et.EVAL_BITWISE_GENERATION_ENV, raising=False)
    assert et._bitwise_generation_spec_kwargs() == {}
    for off in ("", "0", "false", "no"):
        monkeypatch.setenv(et.EVAL_BITWISE_GENERATION_ENV, off)
        assert et._bitwise_generation_spec_kwargs() == {}
    for on in ("1", "true", "YES", "on"):
        monkeypatch.setenv(et.EVAL_BITWISE_GENERATION_ENV, on)
        assert et._bitwise_generation_spec_kwargs() == {"speculative_n_max": 0}
    monkeypatch.setattr(et, "_bitwise_spec_kwargs", lambda: {})  # flag OFF -> still unchanged
    assert et._bitwise_generation_spec_kwargs() == {}
