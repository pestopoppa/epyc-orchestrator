#!/usr/bin/env python3
"""Hosted-only strict producer controls; no APP bootstrap, models or native binaries."""
import hashlib
import ast
import asyncio
import sys
import types
import __future__
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

PRODUCER = Path(__file__).resolve().parents[2] / "src/runtime/hg5_request_event.py"
spec = importlib.util.spec_from_file_location("hg5_fixture_producer", PRODUCER)
events = importlib.util.module_from_spec(spec)
spec.loader.exec_module(events)


class WriterControls(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.env = patch.dict(os.environ, {"HG5_REQUEST_EVENT_DIRECTORY": str(self.directory),
                                          "HG5_REQUEST_EVENT_SOURCE_CONTEXT": ""})
        self.env.start()
        self.plan = SimpleNamespace(requested="force_architect_general", enabled=True,
                                    disabled_reason=None, from_role="frontdoor",
                                    final_answer_role="frontdoor", target_role="architect_general", error=None)

    def tearDown(self):
        self.env.stop()
        self.temp.cleanup()

    def write(self):
        capture = events.begin_capture()
        capture["stage"] = "direct"
        capture["feature_enabled"] = True
        path = events.finish_capture(capture, request_id="fixture-request", plan=self.plan,
                                     counters=events.snapshot(None))
        return capture, path, json.loads(path.read_bytes())

    def test_default_off(self):
        with patch.dict(os.environ, {"HG5_REQUEST_EVENT_DIRECTORY": ""}):
            self.assertIsNone(events.begin_capture())
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_missing_counters_are_null(self):
        self.assertEqual(events.snapshot(None), {key: None for key in events.COUNTERS})

    def test_invalid_types_nonfinite_negative_are_null(self):
        owner = SimpleNamespace(total_calls=True, total_prompt_tokens_reported=1.5,
                                total_tokens_generated=-1, total_prompt_eval_ms=float("nan"),
                                total_generation_ms=float("inf"))
        self.assertTrue(all(value is None for value in events.snapshot(owner).values()))

    def test_getter_failure_is_unknown(self):
        class Missing:
            @property
            def total_calls(self):
                raise RuntimeError("unavailable")
        self.assertIsNone(events.snapshot(Missing())["calls"])

    def test_known_zero_is_measured_zero(self):
        before = {key: 0 for key in events.COUNTERS}
        self.assertEqual(events.counter_delta(before, before), before)

    def test_counter_reset_is_unknown(self):
        before = {key: 2 for key in events.COUNTERS}
        after = {key: 1 for key in events.COUNTERS}
        self.assertTrue(all(value is None for value in events.counter_delta(before, after).values()))

    def test_actual_delta(self):
        before = {key: 2 for key in events.COUNTERS}
        after = {key: 5 for key in events.COUNTERS}
        self.assertEqual(events.counter_delta(before, after), {key: 3 for key in before})

    def test_seal_and_private_immutable_custody(self):
        capture, path, envelope = self.write()
        self.assertEqual(envelope["body_sha256"], hashlib.sha256(events.canonical(envelope["body"])).hexdigest())
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        self.assertEqual(path.stat().st_nlink, 1)
        original = path.read_bytes()
        with self.assertRaises(FileExistsError):
            events.finish_capture(capture, request_id="fixture-request", plan=self.plan, counters=events.snapshot(None))
        self.assertEqual(path.read_bytes(), original)

    def test_unknown_source_and_runtime_not_fabricated(self):
        _, _, envelope = self.write()
        self.assertEqual(envelope["body"]["source"], {"origin": "unknown", "commit": None,
                                                     "tree": None, "producer_sha256": None})
        self.assertEqual(envelope["body"]["runtime_origin"], "unknown")

    def test_exact_source_authored_context(self):
        context = {"commit": "1" * 40, "tree": "2" * 40,
                   "producer_sha256": hashlib.sha256(PRODUCER.read_bytes()).hexdigest()}
        with patch.dict(os.environ, {"HG5_REQUEST_EVENT_SOURCE_CONTEXT": json.dumps(context)}):
            _, _, envelope = self.write()
        self.assertEqual(envelope["body"]["source"], {"origin": "source_authored_run_context", **context})
        self.assertEqual(envelope["body"]["runtime_origin"], "unknown")

    def test_wrong_producer_source_context_refuses(self):
        context = {"commit": "1" * 40, "tree": "2" * 40, "producer_sha256": "0" * 64}
        with patch.dict(os.environ, {"HG5_REQUEST_EVENT_SOURCE_CONTEXT": json.dumps(context)}):
            with self.assertRaises(ValueError):
                events.begin_capture()
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_extra_source_context_refuses(self):
        with patch.dict(os.environ, {"HG5_REQUEST_EVENT_SOURCE_CONTEXT": '{"extra":true}'}):
            with self.assertRaises(ValueError):
                events.begin_capture()

    def test_non_private_directory_refuses(self):
        self.directory.chmod(0o755)
        with self.assertRaises(ValueError):
            self.write()
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_disabled_request_carries_eligibility_and_reason(self):
        self.plan.enabled = False
        self.plan.disabled_reason = "flag_off"
        _, _, envelope = self.write()
        body = envelope["body"]
        self.assertFalse(body["eligible"])
        self.assertEqual(body["disabled_reason"], "flag_off")
        self.assertEqual(body["steps"], [])

    def test_failure_and_frontdoor_fallback_preserved(self):
        self.plan.error = "RuntimeError: forced consultant call failed"
        _, _, envelope = self.write()
        self.assertEqual(envelope["body"]["final_role"], "frontdoor")
        self.assertEqual(envelope["body"]["failure"], {"status": "reported", "exception_type": None, "message": None})

    def test_exception_failure_is_sealed_without_plan_error_text(self):
        capture = events.begin_capture()
        capture["stage"] = "direct"
        capture["feature_enabled"] = True
        capture["step_before"] = {key: None for key in events.COUNTERS}
        events.record_step(capture, None, trigger="quality_escalation", initial_role="frontdoor",
                           target_role="coder_escalation", outcome="failed")
        path = events.finish_capture(capture, request_id="fixture-request", plan=self.plan,
                                     counters=events.snapshot(None), failure_type="RuntimeError",
                                     failure_status="request_failed")
        envelope = json.loads(path.read_bytes())
        self.assertEqual(envelope["body"]["failure"], {"status": "request_failed",
                         "exception_type": "RuntimeError", "message": None})
        self.assertEqual(envelope["body"]["steps"][0]["outcome"], "failed")
        self.assertEqual(envelope["body"]["steps"][0]["call_status"], "unknown")
        with self.assertRaises(FileExistsError):
            events.finish_capture(capture, request_id="fixture-request", plan=self.plan,
                                  counters=events.snapshot(None), failure_type="RuntimeError",
                                  failure_status="request_failed")
        self.assertEqual(len(list(self.directory.glob("*.json"))), 1)

    def test_forced_step_uses_actual_integer_and_timing_deltas(self):
        capture = events.begin_capture()
        capture["step_before"] = {key: 0 for key in events.COUNTERS}
        owner = SimpleNamespace(total_calls=1, total_prompt_tokens_reported=8,
                                total_tokens_generated=3, total_prompt_eval_ms=12.5,
                                total_generation_ms=2.0)
        events.record_step(capture, owner, trigger="caller_forced", initial_role="frontdoor",
                           target_role="architect_general", outcome="adopted")
        self.assertEqual(capture["steps"][0]["delta"], {"calls": 1, "prompt_tokens": 8,
                         "completion_tokens": 3, "prompt_ms": 12.5, "generation_ms": 2.0})
        self.assertEqual(capture["steps"][0]["outcome"], "adopted")

    def test_quality_failed_call_retains_actual_cost(self):
        capture = events.begin_capture()
        capture["step_before"] = {key: 0 for key in events.COUNTERS}
        owner = SimpleNamespace(total_calls=1, total_prompt_tokens_reported=8,
                                total_tokens_generated=0, total_prompt_eval_ms=12.5,
                                total_generation_ms=0)
        events.record_step(capture, owner, trigger="quality_escalation", initial_role="frontdoor",
                           target_role="coder_escalation", outcome="failed")
        self.assertEqual(capture["steps"][0]["outcome"], "failed")
        self.assertEqual(capture["steps"][0]["delta"]["calls"], 1)
        self.assertEqual(capture["steps"][0]["delta"]["prompt_ms"], 12.5)

    def test_no_call_is_distinct_from_unknown(self):
        capture = events.begin_capture()
        capture["step_before"] = {key: 0 for key in events.COUNTERS}
        owner = SimpleNamespace(**{attribute: 0 for attribute in events.COUNTERS.values()})
        events.record_step(capture, owner, trigger="quality_escalation", initial_role="frontdoor",
                           target_role="coder_escalation", outcome="not_adopted")
        self.assertEqual(capture["steps"][0]["outcome"], "not_adopted")
        self.assertEqual(capture["steps"][0]["call_status"], "not_called")
        self.assertEqual(capture["steps"][0]["delta"]["calls"], 0)

    def test_unknown_call_counter_does_not_fabricate_no_call(self):
        capture = events.begin_capture()
        events.record_step(capture, None, trigger="caller_forced", initial_role="frontdoor",
                           target_role="architect_general", outcome="failed")
        self.assertEqual(capture["steps"][0]["outcome"], "failed")
        self.assertEqual(capture["steps"][0]["call_status"], "unknown")
        self.assertIsNone(capture["steps"][0]["delta"]["calls"])

    def test_zero_calls_do_not_erase_known_exception_outcome(self):
        capture = events.begin_capture()
        capture["step_before"] = {key: 0 for key in events.COUNTERS}
        owner = SimpleNamespace(**{attribute: 0 for attribute in events.COUNTERS.values()})
        events.record_step(capture, owner, trigger="quality_escalation", initial_role="frontdoor",
                           target_role="coder_escalation", outcome="failed")
        self.assertEqual(capture["steps"][0]["outcome"], "failed")
        self.assertEqual(capture["steps"][0]["call_status"], "not_called")
        self.assertEqual(capture["steps"][0]["delta"]["calls"], 0)

    def test_canonical_refuses_nonfinite(self):
        with self.assertRaises(ValueError):
            events.canonical({"counter": float("nan")})


class ActualSourceIntegrationControls(unittest.TestCase):
    """Execute the original wrapper/function bodies; replace only backend/tap effects."""
    @classmethod
    def setUpClass(cls):
        cls.app_root = PRODUCER.parents[2]
        path = cls.app_root / "src/api/routes/v1_escalation.py"
        name = "hg5_actual_escalation_source_controls"
        spec = importlib.util.spec_from_file_location(name, path)
        cls.v1 = importlib.util.module_from_spec(spec)
        sys.modules[name] = cls.v1
        spec.loader.exec_module(cls.v1)
        route_source = (cls.app_root / "src/api/routes/openai_compat.py").read_text()
        tree = ast.parse(route_source)
        original = next(node for node in tree.body if isinstance(node, ast.AsyncFunctionDef)
                        and node.name == "_escalate_v1_answer")
        namespace = {"asyncio": asyncio, "escalate_answer": cls.v1.escalate_answer,
                     "STAGE_DIRECT": cls.v1.STAGE_DIRECT, "STAGE_REPL": cls.v1.STAGE_REPL}
        # Compile the exact existing body unchanged; postponed annotations avoid API bootstrap.
        exec(compile(ast.Module(body=[original], type_ignores=[]), str(cls.app_root / "src/api/routes/openai_compat.py"),
                     "exec", flags=__future__.annotations.compiler_flag), namespace)
        cls.route_wrapper = staticmethod(namespace["_escalate_v1_answer"])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.primitives = SimpleNamespace(total_calls=0, total_prompt_tokens_reported=0,
                                           total_tokens_generated=0, total_prompt_eval_ms=0,
                                           total_generation_ms=0, server_urls={})
        self.legacy_emitter = unittest.mock.Mock(return_value=True)
        tap = types.ModuleType("src.runtime.inference_tap")
        tap.emit_request_event = self.legacy_emitter
        stages = SimpleNamespace(_quality_escalate=self.backend)
        pipeline = types.ModuleType("src.api.routes.chat_pipeline")
        pipeline.stages = stages
        self.stage = stages
        self.fail = False
        self.env = patch.dict(os.environ, {"HG5_REQUEST_EVENT_DIRECTORY": str(self.directory),
                                          "HG5_REQUEST_EVENT_SOURCE_CONTEXT": ""})
        self.modules = patch.dict(sys.modules, {"src.runtime.hg5_request_event": events,
                                                "src.runtime.inference_tap": tap,
                                                "src.api.routes.chat_pipeline": pipeline})
        self.env.start()
        self.modules.start()

    def tearDown(self):
        self.modules.stop()
        self.env.stop()
        self.temp.cleanup()

    def backend(self, answer, prompt, primitives, role, **kwargs):
        primitives.total_calls += 1
        primitives.total_prompt_tokens_reported += 8
        primitives.total_tokens_generated += 3
        primitives.total_prompt_eval_ms += 12.5
        primitives.total_generation_ms += 2
        if self.fail:
            raise RuntimeError("fixture backend failure")
        return "consultant answer", self.v1.Role.ARCHITECT_GENERAL

    def plan(self, mode="force_architect_general", flag=True):
        return self.v1.plan_v1_escalation(flag_on=flag, requested=mode,
                    role=self.v1.Role.FRONTDOOR, role_override=False, image_input=False)

    def run_wrapper(self, plan, stage="direct"):
        return asyncio.run(self.route_wrapper(plan, stage, answer="frontdoor answer", question="fixture",
                        direct_prompt="fixture", primitives=self.primitives, state=None, chat_id="fixture"))

    def record(self, plan):
        return self.v1.record_escalation(plan, chat_id="fixture", request_keys={}, primitives=self.primitives)

    def body(self):
        paths = list(self.directory.glob("*.json"))
        self.assertEqual(len(paths), 1)
        envelope = json.loads(paths[0].read_bytes())
        self.assertEqual(envelope["body_sha256"], hashlib.sha256(events.canonical(envelope["body"])).hexdigest())
        return envelope["body"]

    def test_actual_quality_exception_seals_once_and_clears_capture(self):
        self.fail = True
        plan = self.plan("auto")
        with patch.object(events, "finish_capture", wraps=events.finish_capture) as writer:
            self.assertEqual(self.run_wrapper(plan), "frontdoor answer")
            self.assertIsNone(plan.strict_capture)
            self.record(plan)
            self.record(plan)
            self.assertEqual(writer.call_count, 1)
        body = self.body()
        self.assertEqual(body["failure"], {"status": "request_failed", "exception_type": "RuntimeError",
                                          "message": None})
        self.assertEqual(body["steps"][0]["trigger"], "quality_escalation")
        self.assertEqual(body["steps"][0]["outcome"], "failed")
        self.assertEqual(body["steps"][0]["delta"]["calls"], 1)
        self.assertEqual(body["final_role"], "frontdoor")

    def test_actual_force_failure_keeps_bounded_frontdoor_fallback(self):
        self.fail = True
        plan = self.plan()
        self.assertEqual(self.run_wrapper(plan), "frontdoor answer")
        legacy = self.record(plan)
        self.assertEqual(legacy["error"], "RuntimeError: forced consultant call failed")
        body = self.body()
        self.assertEqual(body["failure"]["exception_type"], "RuntimeError")
        self.assertEqual(body["steps"][0]["trigger"], "caller_forced")
        self.assertEqual(body["final_role"], "frontdoor")
        self.assertEqual(self.primitives.total_calls, 1)

    def test_actual_disabled_wrapper_reaches_record_finalizer(self):
        plan = self.plan(flag=False)
        with patch.object(events, "finish_capture", wraps=events.finish_capture) as writer:
            self.assertEqual(self.run_wrapper(plan), "frontdoor answer")
            self.assertIsNotNone(plan.strict_capture)
            self.record(plan)
            self.assertIsNone(plan.strict_capture)
            self.record(plan)
            self.assertEqual(writer.call_count, 1)
        self.assertFalse(self.body()["eligible"])
        self.assertEqual(self.body()["disabled_reason"], "flag_off")
        self.assertEqual(self.primitives.total_calls, 0)

    def test_actual_stage_none_wrapper_reaches_record_finalizer(self):
        plan = self.plan()
        self.assertEqual(self.run_wrapper(plan, stage=None), "frontdoor answer")
        self.assertIsNotNone(plan.strict_capture)
        self.record(plan)
        self.assertIsNone(plan.strict_capture)
        self.assertFalse(self.body()["eligible"])
        self.assertEqual(self.body()["disabled_reason"], "force_requires_completed_direct_answer")
        self.assertEqual(self.primitives.total_calls, 0)

    def test_actual_emitter_exception_preserves_answer_and_legacy_receipt(self):
        plan = self.plan()
        with patch.object(events, "finish_capture", side_effect=OSError("fixture disk refusal")) as writer:
            self.assertEqual(self.run_wrapper(plan), "consultant answer")
            legacy = self.record(plan)
            again = self.record(plan)
            self.assertEqual(writer.call_count, 1)
        self.assertIsNone(plan.strict_capture)
        self.assertEqual(legacy, again)
        self.assertEqual(legacy["final_answer_role"], "architect_general")
        self.assertEqual(list(self.directory.glob("*.json")), [])

    def test_actual_success_then_metadata_render_does_not_attempt_again(self):
        plan = self.plan()
        with patch.object(events, "finish_capture", wraps=events.finish_capture) as writer:
            self.assertEqual(self.run_wrapper(plan), "consultant answer")
            self.record(plan)
            self.record(plan)
            self.assertEqual(writer.call_count, 1)
        self.assertIsNone(plan.strict_capture)
        self.assertEqual(self.body()["final_role"], "architect_general")
        self.assertEqual(self.body()["steps"][0]["outcome"], "adopted")
        self.assertEqual(self.primitives.total_calls, 1)


# These use the real existing TestClient fixture and route generator. All backend
# completions remain the accepted FakePrimitives implementation; never live inference.
import pytest

@pytest.fixture
def hg5_actual_api_env(monkeypatch, tmp_path):
    app_root = PRODUCER.parents[2]
    path = app_root / "tests/unit/test_v1_escalation.py"
    name = "hg5_existing_API_fixture_helpers"
    spec = importlib.util.spec_from_file_location(name, path)
    legacy = importlib.util.module_from_spec(spec)
    sys.modules[name] = legacy
    spec.loader.exec_module(legacy)
    directory = tmp_path / "strict-events"
    directory.mkdir(mode=0o700)
    monkeypatch.setenv("HG5_REQUEST_EVENT_DIRECTORY", str(directory))
    monkeypatch.setenv("HG5_REQUEST_EVENT_SOURCE_CONTEXT", "")
    monkeypatch.setitem(sys.modules, "src.runtime.hg5_request_event", events)
    generator = legacy.env.__wrapped__(monkeypatch, tmp_path)
    env = next(generator)
    try:
        yield legacy, env, directory
    finally:
        try:
            next(generator)
        except StopIteration:
            pass


@pytest.mark.parametrize("stream", [False, True])
def test_actual_API_success_and_stream_seal_one_event(hg5_actual_api_env, stream):
    legacy, env, directory = hg5_actual_api_env
    holder = legacy._install(env, answers={"architect_general": legacy.QUALITY_ANSWER})
    with patch.object(events, "finish_capture", wraps=events.finish_capture) as writer:
        response = env.client.post("/v1/chat/completions", json=legacy._body(
            x_escalation="force_architect_general", stream=stream,
            stream_options={"include_usage": True} if stream else None))
        assert response.status_code == 200
        assert writer.call_count == 1
    paths = list(directory.glob("*.json"))
    assert len(paths) == 1
    body = json.loads(paths[0].read_bytes())["body"]
    assert body["final_role"] == "architect_general"
    assert body["steps"][0]["trigger"] == "caller_forced"
    assert body["steps"][0]["delta"]["calls"] == 1
    assert [call["role"] for call in holder["fake"].calls] == ["frontdoor", "architect_general"]
    if stream:
        chunks = [json.loads(line[6:]) for line in response.text.splitlines()
                  if line.startswith("data: ") and line != "data: [DONE]"]
        content = "".join(chunk["choices"][0]["delta"].get("content") or ""
                          for chunk in chunks if chunk["choices"])
        assert content == legacy.QUALITY_ANSWER
    else:
        assert response.json()["choices"][0]["message"]["content"] == legacy.QUALITY_ANSWER


def test_actual_API_upstream_failure_is_outside_completed_post_answer_scope(hg5_actual_api_env):
    legacy, env, directory = hg5_actual_api_env
    legacy._install(env)
    def upstream_failure(*args, **kwargs):
        raise RuntimeError("fixture upstream failure before post-answer")
    env.monkeypatch.setattr(legacy.FakePrimitives, "chat_completion_call", upstream_failure)
    with patch.object(events, "finish_capture", wraps=events.finish_capture) as writer:
        response = env.client.post("/v1/chat/completions", json=legacy._body(
            x_escalation="force_architect_general"))
        assert response.status_code == 502
        assert writer.call_count == 0
    assert list(directory.glob("*.json")) == []


if __name__ == "__main__":
    unittest.main()
