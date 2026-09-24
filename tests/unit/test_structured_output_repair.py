"""Tests for the shared structured-output repair helper (TD-21).

Covers: `fish_json` extraction edge cases, the `parse_with_repair` state
machine (parsed / declined / repaired / failed paths, TD-21.30(b) partial
objects, never-fabricates-a-value), the `http_chat_completer` wire payload
shape, and the (site, status) telemetry counter.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.structured_output.repair import (
    STRUCTURED_OUTPUT_REPAIR_COUNTS,
    RepairResult,
    fish_json,
    http_chat_completer,
    parse_with_repair,
    primitives_completer,
    reset_counts_for_tests,
)


@pytest.fixture(autouse=True)
def _clear_counts():
    reset_counts_for_tests()
    yield
    reset_counts_for_tests()


# --------------------------------------------------------------------------- fish_json


class TestFishJson:
    def test_fenced_json_block(self):
        text = 'Here you go:\n```json\n{"a": 1, "b": 2}\n```\nThanks.'
        assert fish_json(text) == {"a": 1, "b": 2}

    def test_bare_fence_no_language_tag(self):
        text = 'Result:\n```\n{"a": 1}\n```'
        assert fish_json(text) == {"a": 1}

    def test_prose_around_json_no_fence(self):
        text = "The answer, after some thought, is {\"x\": 1} and that's final."
        assert fish_json(text) == {"x": 1}

    def test_nested_braces_in_strings_do_not_break_depth(self):
        text = '{"a": "value with } a brace and { another", "b": 2}'
        assert fish_json(text) == {"a": "value with } a brace and { another", "b": 2}

    def test_escaped_quote_inside_string_does_not_end_string_early(self):
        text = r'{"a": "she said \"hi\" to } me", "b": 3}'
        value = fish_json(text)
        assert value == {"a": 'she said "hi" to } me', "b": 3}

    def test_last_object_wins(self):
        text = '{"a": 1} some text in between {"a": 2}'
        assert fish_json(text) == {"a": 2}

    def test_array_kind(self):
        text = 'Items: [1, 2, 3] done.'
        assert fish_json(text, kind="array") == [1, 2, 3]

    def test_array_kind_rejects_object(self):
        text = '{"a": 1}'
        assert fish_json(text, kind="array") is None

    def test_object_kind_rejects_array(self):
        text = "[1, 2, 3]"
        assert fish_json(text, kind="object") is None

    def test_any_kind_accepts_either(self):
        assert fish_json("[1, 2]", kind="any") == [1, 2]
        assert fish_json('{"a": 1}', kind="any") == {"a": 1}

    def test_garbage_returns_none(self):
        assert fish_json("this is not json at all, sorry") is None

    def test_empty_string_returns_none(self):
        assert fish_json("") is None

    def test_never_raises_on_malformed_input(self):
        # unterminated string, dangling braces, mixed junk
        assert fish_json('{"a": "unterminated, {[[[') is None

    def test_truncated_fenced_block_repaired_via_existing_repair_idiom(self):
        # A fenced block is handed to `_repair_json_text` even when its own
        # brackets are not balanced (the fence delimiters, not brace-matching,
        # define its span) -- this is exactly the truncated-generation shape
        # `_repair_json_text`'s trailing-comma + missing-closer path targets.
        text = 'Here:\n```json\n{"a": 1, "b": 2,\n```\nend'
        assert fish_json(text) == {"a": 1, "b": 2}

    def test_fence_preferred_over_looser_balanced_match_elsewhere(self):
        text = (
            'noise {"stale": true} more noise\n'
            '```json\n{"fresh": true}\n```\n'
            'trailing {"also_stale": true}'
        )
        # fenced block wins even though it isn't the textually-last balanced span
        assert fish_json(text) == {"fresh": True}

    def test_nested_object_inside_top_level_array(self):
        text = '[{"a": 1}, {"b": 2}]'
        assert fish_json(text, kind="array") == [{"a": 1}, {"b": 2}]


# --------------------------------------------------------------------------- parse_with_repair: parsed path


SIMPLE_SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "count": {"type": "integer"}},
    "required": ["name", "count"],
}


class TestParsedPath:
    def test_clean_json_parses_with_zero_calls(self):
        complete = MagicMock(side_effect=AssertionError("must not be called"))
        result = parse_with_repair(
            'Reply: {"name": "widget", "count": 3}',
            schema=SIMPLE_SCHEMA,
            complete=complete,
            site="test.parsed",
        )
        assert result.status == "parsed"
        assert result.value == {"name": "widget", "count": 3}
        assert result.repair_calls == 0
        assert result.reason == ""
        complete.assert_not_called()
        assert STRUCTURED_OUTPUT_REPAIR_COUNTS[("test.parsed", "parsed")] == 1

    def test_additional_properties_defaulted_closed_rejects_extra_key(self):
        # No additionalProperties on SIMPLE_SCHEMA; TD-21.30(d) closes it, so
        # an extra key on the fished object should NOT be accepted as "parsed"
        # (falls through to repair, which then fails with no completer help).
        def complete(messages, schema):
            raise RuntimeError("no local server in this test")

        result = parse_with_repair(
            'Reply: {"name": "widget", "count": 3, "extra": "nope"}',
            schema=SIMPLE_SCHEMA,
            complete=complete,
            site="test.closed",
        )
        assert result.status == "failed"
        assert result.value is None


# --------------------------------------------------------------------------- TD-21.30(b): partial/wrong-type objects


class TestPartialObjectGoesToRepair:
    def test_wrong_type_required_field_is_not_accepted_as_parsed(self):
        # `count` is present (so a naive "required keys covered" check would
        # accept this) but has the WRONG TYPE -- TD-21.30(b) requires full
        # schema validation, so this must NOT short-circuit as "parsed".
        raw = 'Reply: {"name": "widget", "count": "three"}'

        def complete(messages, schema):
            return json.dumps({"name": "widget", "count": 3})

        result = parse_with_repair(raw, schema=SIMPLE_SCHEMA, complete=complete, site="test.b")
        assert result.status == "repaired"
        assert result.value == {"name": "widget", "count": 3}
        assert result.repair_calls == 1

    def test_missing_required_field_goes_to_repair(self):
        raw = 'Reply: {"name": "widget"}'

        def complete(messages, schema):
            return json.dumps({"name": "widget", "count": 7})

        result = parse_with_repair(raw, schema=SIMPLE_SCHEMA, complete=complete, site="test.b2")
        assert result.status == "repaired"
        assert result.value == {"name": "widget", "count": 7}


# --------------------------------------------------------------------------- decline path


DECLINE_SCHEMA = {
    "type": "object",
    "properties": {"mechanism_id": {"type": "string"}},
    "required": ["mechanism_id"],
    "additionalProperties": False,
}


class TestDeclinePath:
    def test_explicit_decline_returns_declined_and_never_extracts(self):
        raw = "I looked into this but found nothing worth proposing; I decline to submit a hypothesis."
        calls = []

        def complete(messages, schema):
            calls.append(schema)
            # First call is the decline probe; must never be asked twice for
            # extraction after an explicit decline.
            return json.dumps({"explicitly_declines": True, "reason": "found nothing"})

        result = parse_with_repair(
            raw,
            schema=DECLINE_SCHEMA,
            complete=complete,
            decline_question="Does the report explicitly decline to propose anything?",
            site="test.decline",
        )
        assert result.status == "declined"
        assert result.value is None
        assert result.reason == "found nothing"
        assert result.repair_calls == 1
        assert len(calls) == 1  # extraction never invoked

    def test_non_decline_falls_through_to_extraction(self):
        raw = "I propose renaming the buffer to widen alignment."

        def complete(messages, schema):
            if schema.get("properties", {}).get("explicitly_declines"):
                return json.dumps({"explicitly_declines": False, "reason": ""})
            return json.dumps({"mechanism_id": "widen-alignment"})

        result = parse_with_repair(
            raw,
            schema=DECLINE_SCHEMA,
            complete=complete,
            decline_question="Does the report explicitly decline to propose anything?",
            site="test.decline2",
        )
        assert result.status == "repaired"
        assert result.value == {"mechanism_id": "widen-alignment"}
        assert result.repair_calls == 2


# --------------------------------------------------------------------------- failure paths


class TestFailurePaths:
    def test_bad_json_from_extraction_turn_fails_without_fabricating(self):
        def complete(messages, schema):
            return "not json at all"

        result = parse_with_repair("garbage garbage", schema=SIMPLE_SCHEMA, complete=complete, site="test.f1")
        assert result.status == "failed"
        assert result.value is None
        assert "no parseable JSON" in result.reason

    def test_schema_invalid_extraction_result_fails(self):
        def complete(messages, schema):
            return json.dumps({"name": "widget"})  # missing required `count`

        result = parse_with_repair("garbage garbage", schema=SIMPLE_SCHEMA, complete=complete, site="test.f2")
        assert result.status == "failed"
        assert result.value is None

    def test_transport_exception_is_caught_and_typed_as_failed(self):
        def complete(messages, schema):
            raise ConnectionError("server unreachable")

        result = parse_with_repair("garbage garbage", schema=SIMPLE_SCHEMA, complete=complete, site="test.f3")
        assert result.status == "failed"
        assert result.value is None
        assert "transport_error" in result.reason
        assert "ConnectionError" in result.reason

    def test_never_returns_fabricated_default_value(self):
        def complete(messages, schema):
            raise TimeoutError("timed out")

        result = parse_with_repair("garbage", schema=SIMPLE_SCHEMA, complete=complete, site="test.f4")
        assert result.value is None  # never {} or a partial guess


# --------------------------------------------------------------------------- counters


class TestCounters:
    def test_counters_increment_per_site_and_status(self):
        def ok_complete(messages, schema):
            return json.dumps({"name": "w", "count": 1})

        parse_with_repair('{"name": "w", "count": 1}', schema=SIMPLE_SCHEMA, complete=ok_complete, site="site.a")
        parse_with_repair("garbage", schema=SIMPLE_SCHEMA, complete=ok_complete, site="site.a")
        parse_with_repair("garbage", schema=SIMPLE_SCHEMA, complete=ok_complete, site="site.b")

        assert STRUCTURED_OUTPUT_REPAIR_COUNTS[("site.a", "parsed")] == 1
        assert STRUCTURED_OUTPUT_REPAIR_COUNTS[("site.a", "repaired")] == 1
        assert STRUCTURED_OUTPUT_REPAIR_COUNTS[("site.b", "repaired")] == 1


# --------------------------------------------------------------------------- http_chat_completer wire shape


class TestHttpChatCompleter:
    def test_payload_shape(self):
        captured = {}

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return json.dumps(
                    {"choices": [{"message": {"content": '{"ok": true}'}}]}
                ).encode("utf-8")

        def fake_urlopen(request, timeout=None):
            captured["url"] = request.full_url
            captured["timeout"] = timeout
            captured["body"] = json.loads(request.data.decode("utf-8"))
            return FakeResponse()

        with patch("src.structured_output.repair.urllib.request.urlopen", side_effect=fake_urlopen):
            complete = http_chat_completer("http://127.0.0.1:8083/v1", model="qwen3.8-27b", timeout_s=42)
            content = complete(
                [{"role": "system", "content": "sys"}, {"role": "user", "content": "usr"}],
                SIMPLE_SCHEMA,
            )

        assert content == '{"ok": true}'
        assert captured["url"] == "http://127.0.0.1:8083/v1/chat/completions"
        assert captured["timeout"] == 42
        body = captured["body"]
        assert body["temperature"] == 0
        assert body["chat_template_kwargs"] == {"enable_thinking": False}
        assert body["model"] == "qwen3.8-27b"
        assert body["response_format"]["type"] == "json_schema"
        assert body["response_format"]["json_schema"]["schema"] == SIMPLE_SCHEMA
        assert body["messages"][0]["content"] == "sys"
        assert body["messages"][1]["content"] == "usr"

    def test_model_omitted_when_not_given(self):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return json.dumps(
                    {"choices": [{"message": {"content": "{}"}}]}
                ).encode("utf-8")

        captured = {}

        def fake_urlopen(request, timeout=None):
            captured["body"] = json.loads(request.data.decode("utf-8"))
            return FakeResponse()

        with patch("src.structured_output.repair.urllib.request.urlopen", side_effect=fake_urlopen):
            complete = http_chat_completer("http://127.0.0.1:8083")
            complete([{"role": "user", "content": "hi"}], SIMPLE_SCHEMA)

        assert "model" not in captured["body"]

    def test_transport_error_propagates_to_caller(self):
        import urllib.error

        def fake_urlopen(request, timeout=None):
            raise urllib.error.URLError("connection refused")

        with patch("src.structured_output.repair.urllib.request.urlopen", side_effect=fake_urlopen):
            complete = http_chat_completer("http://127.0.0.1:8083")
            with pytest.raises(urllib.error.URLError):
                complete([{"role": "user", "content": "hi"}], SIMPLE_SCHEMA)


# --------------------------------------------------------------------------- primitives_completer


class TestPrimitivesCompleter:
    def test_renders_messages_into_a_single_prompt_and_forwards_kwargs(self):
        primitives = MagicMock()
        primitives.llm_call.return_value = '{"ok": true}'

        complete = primitives_completer(primitives, role="worker")
        content = complete(
            [{"role": "system", "content": "sys line"}, {"role": "user", "content": "usr line"}],
            SIMPLE_SCHEMA,
        )

        assert content == '{"ok": true}'
        primitives.llm_call.assert_called_once()
        args, kwargs = primitives.llm_call.call_args
        prompt = args[0]
        assert "sys line" in prompt
        assert "usr line" in prompt
        assert kwargs["role"] == "worker"
        assert kwargs["json_schema"] == SIMPLE_SCHEMA
        assert kwargs["temperature"] == 0.0
        assert kwargs["skip_suffix"] is True

    def test_transport_exception_from_llm_call_propagates(self):
        primitives = MagicMock()
        primitives.llm_call.side_effect = RuntimeError("backend down")

        complete = primitives_completer(primitives, role="worker")
        with pytest.raises(RuntimeError):
            complete([{"role": "user", "content": "hi"}], SIMPLE_SCHEMA)


# --------------------------------------------------------------------------- integration: parse_with_repair + primitives_completer


class TestParseWithRepairUsesInjectedCompleter:
    def test_full_pipeline_with_primitives_completer(self):
        primitives = MagicMock()
        primitives.llm_call.return_value = json.dumps({"name": "widget", "count": 5})

        complete = primitives_completer(primitives, role="worker")
        result = parse_with_repair(
            "some unparseable prose about a widget",
            schema=SIMPLE_SCHEMA,
            complete=complete,
            site="test.integration",
        )
        assert result.status == "repaired"
        assert result.value == {"name": "widget", "count": 5}
