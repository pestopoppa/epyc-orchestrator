"""Model-free tests for the retained-source KPF-26 prospective smoke packet."""
import hashlib
import json

import pytest

from scripts.trace import kpf26_paired_replay as paired
from scripts.trace import kpf26_prepare_smoke as prep
from scripts.trace import kpf26_scout_executor as executor


def test_packet_freezes_real_source_and_actual_pins(tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"not an inference model")
    binary_hash = "a" * 64
    server = {"binary_sha256": binary_hash, "argv": ["server", str(model)], "ggml": {"lib": "b" * 64}}
    path = prep.prepare(tmp_path / "packet", pid=123, url="http://127.0.0.1:1", role="architect_critic",
                        model=model, grant="test-grant", source="1" * 40, snapshot=server,
                        verify_endpoint=False)
    m = json.loads(path.read_text())
    paired.validate_manifest(m, path.parent)
    assert m["provenance"]["kind"] == "prospective-scout-smoke-derived-from-hs19a"
    assert "not a historical scout replay" in m["provenance"]["interpretation"]
    assert m["pins"]["server_binary_sha256"] == binary_hash
    assert m["pins"]["model_sha256"] == hashlib.sha256(model.read_bytes()).hexdigest()
    assert m["pins"]["executor_sha256"] == prep.digest_file(prep.EXECUTOR)
    assert m["runtime"]["off"]["grant_id"] == m["runtime"]["on"]["grant_id"] == "test-grant"
    assert m["runtime"]["off"]["window_id"] != m["runtime"]["on"]["window_id"]
    from src.api.models.requests import ChatRequest
    api_request = json.loads((path.parent / m["api_request"]["path"]).read_text())
    validated = ChatRequest.model_validate(api_request)
    assert validated.scouts.reserve_slots == 1 and len(validated.scouts.targets) == 2
    assert validated.real_mode and not validated.mock_mode
    task = m["tasks"][0]
    request = json.loads((path.parent / task["request"]["path"]).read_text())
    expected = json.loads((path.parent / task["expected"]["path"]).read_text())
    assert request["scope_root"] == str((path.parent / "scope").resolve())
    assert executor.score_summary("RELEASE_CODENAME is amber-heron-47", executor.validate_scorer(expected, 2)[0])
    assert not executor.score_summary("I do not know", executor.validate_scorer(expected, 2)[0])
    (path.parent / "request.json").write_text("tampered")
    with pytest.raises(ValueError, match="pin mismatch"):
        paired.validate_manifest(m, path.parent)
    with pytest.raises(ValueError, match="output already exists"):
        prep.prepare(tmp_path / "packet", pid=123, url="http://127.0.0.1:1", role="architect_critic",
                     model=model, grant="test-grant", source="1" * 40, snapshot=server, verify_endpoint=False)


def test_rejects_unsafe_grant_and_unmapped_server(tmp_path):
    model = tmp_path / "model"
    model.write_bytes(b"x")
    with pytest.raises(ValueError, match="grant"):
        prep.prepare(tmp_path / "one", pid=1, url="http://127.0.0.1:1", role="a", model=model,
                     grant="../bad", source="1" * 40, snapshot={}, verify_endpoint=False)
    with pytest.raises(ValueError, match="mapped ggml"):
        prep.prepare(tmp_path / "two", pid=1, url="http://127.0.0.1:1", role="a", model=model,
                     grant="safe", source="1" * 40, snapshot={"argv": [str(model)], "ggml": {}},
                     verify_endpoint=False)
