"""SPEC-GRAM-JSON-1 enabler: json_schema capture in serving_calls (flag serving_calls_json_schema_capture).

Local files only, no network. Flag off must leave the row byte-identical and touch no files.
"""

from __future__ import annotations

import hashlib
import json
import types
from pathlib import Path

import pytest

from src.backends import serving_calls as sc
from src.features import reset_features
from src.model_server import InferenceRequest, InferenceResult

SCHEMA = {"type": "object", "properties": {"b": {"type": "integer"}, "a": {"type": "string"}},
          "required": ["a"]}
SCHEMA_REORDERED = {"required": ["a"], "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
                    "type": "object"}
FLAG_ENV = "ORCHESTRATOR_SERVING_CALLS_JSON_SCHEMA_CAPTURE"


@pytest.fixture
def log_file(monkeypatch, tmp_path) -> Path:
    path = tmp_path / "serving_calls" / "serving_calls.jsonl"
    monkeypatch.setenv(sc.LOG_ENV, str(path))
    monkeypatch.setenv("ORCHESTRATOR_PATHS_LOG_DIR", str(tmp_path))
    monkeypatch.delenv(FLAG_ENV, raising=False)
    sc._SIDECAR_CACHE.clear()
    sc.clear_staged()
    reset_features()
    yield path
    sc.clear_staged()
    reset_features()


def _flag(monkeypatch, on: bool) -> None:
    monkeypatch.setenv(FLAG_ENV, "1" if on else "0")
    reset_features()


class _Backend:
    def __init__(self, output: str = '{"a":"x"}'):
        self.output = output
        self.config = types.SimpleNamespace(base_url="http://localhost:8070")

    @sc.recorded_call("infer")
    def infer(self, role_config, request):
        return InferenceResult(role="frontdoor", output=self.output, tokens_generated=5,
                               generation_speed=1.0, elapsed_time=0.1, success=True)


def _rc():
    return types.SimpleNamespace(name="frontdoor", model=types.SimpleNamespace(name="Qwen-test"))


def _rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def _call(schema=SCHEMA, prompt="give json", output='{"a":"x"}'):
    sc.stage_caller(role="frontdoor", request_id="r1", workload_class="interactive")
    _Backend(output).infer(_rc(), InferenceRequest(role="frontdoor", prompt=prompt, n_tokens=32,
                                                   json_schema=schema))


def test_flag_off_row_byte_identical_and_no_files(log_file, monkeypatch, tmp_path):
    """Off: the line equals the pre-change path (build_record + write_record) byte for byte."""
    _flag(monkeypatch, False)
    monkeypatch.setattr(sc.uuid, "uuid4", lambda: types.SimpleNamespace(hex="ab" * 16))
    monkeypatch.setattr(sc.time, "time", lambda: 1000.0)
    _call()
    [row] = _rows(log_file)
    assert "json_schema_capture" not in row
    assert not (tmp_path / "serving_calls" / sc.SCHEMA_STORE_DIR).exists()
    assert not (tmp_path / "serving_calls" / sc.BODY_DIR).exists()

    expected_path = tmp_path / "expected.jsonl"
    monkeypatch.setenv(sc.LOG_ENV, str(expected_path))
    req = InferenceRequest(role="frontdoor", prompt="give json", n_tokens=32, json_schema=SCHEMA)
    staged = {"role": "frontdoor", "request_id": "r1", "workload_class": "interactive", "_ts0": 1000.0}
    rec = sc.build_record(method="infer", role_config=_rc(), request=req,
                          base_url="http://localhost:8070", ts_start=1000.0, ts_end=1000.0,
                          result=_Backend().infer.__wrapped__(_Backend(), _rc(), req),
                          staged=staged, notes={})
    sc.write_record(rec)
    assert log_file.read_bytes() == expected_path.read_bytes()


def test_flag_on_row_fields_schema_store_and_body(log_file, monkeypatch, tmp_path):
    _flag(monkeypatch, True)
    _call()
    [row] = _rows(log_file)
    cap = row["json_schema_capture"]
    canonical = json.dumps(SCHEMA, sort_keys=True, separators=(",", ":"))
    sha = hashlib.sha256(canonical.encode()).hexdigest()
    assert cap["schema_sha256"] == sha
    assert "properties" not in json.dumps(cap)  # only the sha in the row, never the schema
    assert cap["output_text"] == '{"a":"x"}' and cap["output_chars"] == 9
    assert cap["request_class"] == "json:frontdoor"
    assert cap["prompt_sha256"] == hashlib.sha256(b"give json").hexdigest()
    assert cap["prompt_chars"] == 9
    assert cap["prompt_body"] == f"{sc.BODY_DIR}/{row['record_id']}.json"

    base = tmp_path / "serving_calls"
    assert json.loads((base / sc.SCHEMA_STORE_DIR / f"{sha}.json").read_text()) == SCHEMA
    body = json.loads((base / cap["prompt_body"]).read_text())
    assert body["record_id"] == row["record_id"]
    assert body["prompt_text"] == "give json" and body["prompt_sha256"] == cap["prompt_sha256"]
    assert body["prompt_kind"] == "prompt"
    assert sc.record_digest(row) == row["record_sha256"]


def test_schema_store_dedup_across_calls_and_key_order(log_file, monkeypatch, tmp_path):
    _flag(monkeypatch, True)
    _call(SCHEMA, prompt="p1")
    _call(SCHEMA_REORDERED, prompt="p2")
    rows = _rows(log_file)
    assert rows[0]["json_schema_capture"]["schema_sha256"] == rows[1]["json_schema_capture"]["schema_sha256"]
    assert len(list((tmp_path / "serving_calls" / sc.SCHEMA_STORE_DIR).glob("*.json"))) == 1
    assert len(list((tmp_path / "serving_calls" / sc.BODY_DIR).glob("*.json"))) == 2


def test_raw_schema_preserves_property_order_round_trip(log_file, monkeypatch, tmp_path):
    _flag(monkeypatch, True)
    _call(SCHEMA)  # properties b before a
    [row] = _rows(log_file)
    cap = row["json_schema_capture"]
    raw_path = tmp_path / "serving_calls" / sc.SCHEMA_STORE_DIR / sc.SCHEMA_RAW_SUBDIR / f"{cap['schema_raw_sha256']}.json"
    text = raw_path.read_text()
    assert hashlib.sha256(text.encode()).hexdigest() == cap["schema_raw_sha256"]
    assert cap["schema_raw_chars"] == len(text)
    assert list(json.loads(text)["properties"]) == ["b", "a"]
    assert text == json.dumps(SCHEMA, separators=(",", ":"))
    canon = tmp_path / "serving_calls" / sc.SCHEMA_STORE_DIR / f"{cap['schema_sha256']}.json"
    assert list(json.loads(canon.read_text())["properties"]) == ["a", "b"]


def test_reordered_schemas_share_canonical_sha_distinct_raw_sha(log_file, monkeypatch, tmp_path):
    _flag(monkeypatch, True)
    _call(SCHEMA, prompt="p1")
    _call(SCHEMA_REORDERED, prompt="p2")
    _call(SCHEMA, prompt="p3")
    caps = [r["json_schema_capture"] for r in _rows(log_file)]
    assert caps[0]["schema_sha256"] == caps[1]["schema_sha256"]
    assert caps[0]["schema_raw_sha256"] != caps[1]["schema_raw_sha256"]
    assert caps[0]["schema_raw_sha256"] == caps[2]["schema_raw_sha256"]
    store = tmp_path / "serving_calls" / sc.SCHEMA_STORE_DIR
    assert len(list(store.glob("*.json"))) == 1
    assert len(list((store / sc.SCHEMA_RAW_SUBDIR).glob("*.json"))) == 2


def test_flag_on_without_json_schema_is_unchanged(log_file, monkeypatch, tmp_path):
    _flag(monkeypatch, True)
    sc.stage_caller(role="frontdoor")
    _Backend().infer(_rc(), InferenceRequest(role="frontdoor", prompt="x", n_tokens=8))
    [row] = _rows(log_file)
    assert "json_schema_capture" not in row
    assert not (tmp_path / "serving_calls" / sc.BODY_DIR).exists()


def test_tool_call_class_and_chat_payload_body(log_file, monkeypatch, tmp_path):
    _flag(monkeypatch, True)
    sc.stage_caller(role="frontdoor", client_class="opencode")
    payload = {"tools": [{"type": "function", "function": {"name": "t"}}],
               "messages": [{"role": "user", "content": "hi"}]}
    _Backend().infer(_rc(), InferenceRequest(role="frontdoor", prompt="x", n_tokens=8,
                                             json_schema=SCHEMA, chat_payload=payload))
    [row] = _rows(log_file)
    cap = row["json_schema_capture"]
    assert cap["request_class"] == "tool_call:opencode"
    assert cap["request_class_source"] == "caller.client_class"
    body = json.loads((tmp_path / "serving_calls" / cap["prompt_body"]).read_text())
    assert body["prompt_kind"] == "chat_payload" and '"function"' in body["prompt_text"]


def test_kill_switch_disables_capture(monkeypatch, tmp_path):
    monkeypatch.setenv(sc.LOG_ENV, "off")
    monkeypatch.setenv("ORCHESTRATOR_PATHS_LOG_DIR", str(tmp_path))
    _flag(monkeypatch, True)
    _call()
    assert not any(tmp_path.iterdir())
    reset_features()


def test_rotation_prunes_bodies_of_dropped_log(log_file, monkeypatch, tmp_path):
    _flag(monkeypatch, True)
    monkeypatch.setenv(sc.KEEP_ENV, "1")
    _call()
    [row] = _rows(log_file)
    body = tmp_path / "serving_calls" / row["json_schema_capture"]["prompt_body"]
    assert body.exists()
    monkeypatch.setattr(sc, "_rotation", lambda: (1, 1))
    _call(prompt="second")  # rotates: first file becomes .1 (== keep)
    assert body.exists()
    _call(prompt="third")  # rotates again: .1 dropped, its body pruned
    assert not body.exists()
