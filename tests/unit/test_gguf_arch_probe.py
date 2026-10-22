"""RTE-PFX-LB1: the GGUF header probe on a synthetic file (no model, no weights)."""
import importlib.util
import struct
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "probe", Path(__file__).resolve().parents[2] / "scripts/analysis/gguf_arch_probe.py")
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)


def _s(x: str) -> bytes:
    b = x.encode()
    return struct.pack("<Q", len(b)) + b


def _write(path, arch, extra):
    kvs = [(_s("general.architecture"), struct.pack("<I", 8) + _s(arch))]
    for k, v in extra.items():
        kvs.append((_s(k), struct.pack("<I", 4) + struct.pack("<I", v)))
    with open(path, "wb") as f:
        f.write(b"GGUF" + struct.pack("<IQQ", 3, 0, len(kvs)))
        for k, v in kvs:
            f.write(k + v)


def test_hybrid_detected(tmp_path):
    p = tmp_path / "h.gguf"
    _write(p, "qwen35", {"qwen35.ssm.state_size": 128, "qwen35.block_count": 64})
    r = probe.probe(str(p))
    assert r["architecture"] == "qwen35" and r["recurrent_hybrid"] is True and r["block_count"] == 64


def test_pure_attention_not_hybrid(tmp_path):
    p = tmp_path / "t.gguf"
    _write(p, "llama", {"llama.block_count": 32})
    assert probe.probe(str(p))["recurrent_hybrid"] is False
