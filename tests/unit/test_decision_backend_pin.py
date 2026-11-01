import hashlib
import importlib.util
from pathlib import Path
import struct

import pytest

spec = importlib.util.spec_from_file_location(
    "decision_backend_pin",
    Path(__file__).parents[2] / "src/typed_decisions/decision_backend_pin.py",
)
pin = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pin)


def s(value):
    value = value.encode()
    return struct.pack("<Q", len(value)) + value


def gguf(path, pairs):
    data = b"GGUF" + struct.pack("<IQQ", 3, 0, len(pairs))
    for key, kind, value in pairs:
        data += s(key) + struct.pack("<I", kind)
        data += s(value) if kind == 8 else struct.pack("<f", value)
    path.write_bytes(data)
    return data


def test_capture_reads_all_architecture_temperatures_and_hash(tmp_path):
    model = tmp_path / "m.gguf"
    data = gguf(
        model,
        [
            ("general.architecture", 8, "lfm3"),
            ("lfm3.decision.temperature.choice", 6, 2.0),
            ("lfm3.decision.temperature.noul", 8, "1.5"),
            ("other.decision.temperature.choice", 6, 9.0),
            ("decision.temperature.score", 6, 3.0),
        ],
    )
    receipt = pin.capture_decision_backend_pin(
        model, ["server", "--parallel=3", "--batch-size", "512", "-ub=128"], "a" * 40
    )
    assert receipt["gguf_sha256"] == hashlib.sha256(data).hexdigest()
    assert receipt["decision_temperatures"] == {
        "lfm3.decision.temperature.choice": 2.0,
        "lfm3.decision.temperature.noul": "1.5",
        "other.decision.temperature.choice": 9.0,
        "decision.temperature.score": 3.0,
    }
    assert receipt["model_architecture"] == "lfm3"
    assert receipt["native_temperature_prefix"] == "lfm3.decision.temperature."
    assert receipt["server_argv_flags"] == {"np": 3, "batch": 512, "ubatch": 128}


def test_empty_temperatures_is_distinct_from_missing_architecture(tmp_path):
    model = tmp_path / "m.gguf"
    gguf(model, [("general.architecture", 8, "lfm3")])
    assert pin.read_decision_temperatures(model) == ("lfm3", "lfm3.decision.temperature.", {})
    gguf(model, [("another", 8, "lfm3")])
    with pytest.raises(ValueError, match="architecture"):
        pin.read_decision_temperatures(model)


@pytest.mark.parametrize(
    "argv",
    [
        ["server", "-np", "1", "-b", "8"],
        ["server", "-np", "1", "--parallel=2", "-b", "8", "-ub", "4"],
        ["server", "-np", "1", "-b", "8", "-ub", "0"],
    ],
)
def test_flags_fail_closed(argv):
    with pytest.raises(ValueError):
        pin.parse_process_flags(argv)


def test_invalid_or_truncated_metadata_fails(tmp_path):
    model = tmp_path / "m.gguf"
    gguf(
        model,
        [
            ("general.architecture", 8, "lfm3"),
            ("lfm3.decision.temperature.choice", 6, -1),
        ],
    )
    with pytest.raises(ValueError, match="temperature"):
        pin.read_decision_temperatures(model)
    model.write_bytes(model.read_bytes()[:-1])
    with pytest.raises(ValueError, match="truncated"):
        pin.read_decision_temperatures(model)


def test_duplicate_and_wrong_architecture_metadata(tmp_path):
    model = tmp_path / "m.gguf"
    gguf(
        model,
        [("general.architecture", 8, "lfm3"), ("general.architecture", 8, "lfm3")],
    )
    with pytest.raises(ValueError, match="duplicate"):
        pin.read_decision_temperatures(model)
    gguf(
        model,
        [
            ("general.architecture", 8, "lfm3"),
            ("other.decision.temperature.choice", 6, 2.0),
        ],
    )
    assert pin.read_decision_temperatures(model) == ("lfm3", "lfm3.decision.temperature.", {"other.decision.temperature.choice": 2.0})


def test_nonlocal_or_nonorigin_endpoint_fails_closed():
    for url in (
        "http://example.com:1234",
        "https://127.0.0.1:1234",
        "http://127.0.0.1:1234/proxy",
        "http://user@localhost:1234",
    ):
        with pytest.raises(ValueError, match="localhost"):
            pin.listener_inode(1, url)
