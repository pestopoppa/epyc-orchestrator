#!/usr/bin/env python3
"""RTE-PFX-LB1: read a GGUF's architecture metadata WITHOUT loading the model.

Reads only the header key/value block (pure ``struct``; no llama.cpp, no tensors, no weights
mapped) and reports ``general.architecture`` plus every ``*.ssm.*`` / ``*full_attention*`` /
``*linear*`` key, which is what separates a recurrent hybrid (needs a junction checkpoint to
reuse a prefix; plain position-keyed KV reuse does not apply to its recurrent state) from a
pure-attention transformer. For a split GGUF pass the first shard (``-00001-of-N``).
"""
from __future__ import annotations

import json
import struct
import sys

_SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
_HYBRID_MARKERS = (".ssm.", "full_attention", "linear_attention", "recurrent", "delta_net", "deltanet", ".linear_")


def _read_str(f) -> str:
    (n,) = struct.unpack("<Q", f.read(8))
    return f.read(n).decode("utf-8", "replace")


def _read_value(f, t: int, keep: bool):
    if t in _SCALARS:
        fmt = _SCALARS[t]
        (v,) = struct.unpack(fmt, f.read(struct.calcsize(fmt)))
        return v
    if t == 8:
        s = _read_str(f)
        return s if keep else None
    if t == 9:
        (et,) = struct.unpack("<I", f.read(4))
        (n,) = struct.unpack("<Q", f.read(8))
        out = [_read_value(f, et, keep and n <= 64) for _ in range(n)]
        return out if keep and n <= 64 else f"<array of {n}>"
    raise ValueError(f"unknown GGUF value type {t}")


def probe(path: str) -> dict:
    kv: dict = {}
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError("not a GGUF file")
        version, _n_tensors, n_kv = struct.unpack("<IQQ", f.read(20))
        for _ in range(n_kv):
            key = _read_str(f)
            (t,) = struct.unpack("<I", f.read(4))
            keep = key.startswith("general.") or any(m in key for m in _HYBRID_MARKERS) or key.endswith(
                (".block_count", ".context_length", ".expert_count"))
            kv[key] = _read_value(f, t, keep)
    arch = kv.get("general.architecture")
    hybrid_keys = {k: v for k, v in kv.items() if any(m in k for m in _HYBRID_MARKERS)}
    return {
        "path": path,
        "gguf_version": version,
        "architecture": arch,
        "name": kv.get("general.name"),
        "block_count": kv.get(f"{arch}.block_count"),
        "recurrent_hybrid": bool(hybrid_keys),
        "hybrid_keys": hybrid_keys,
    }


if __name__ == "__main__":
    print(json.dumps([probe(p) for p in sys.argv[1:]], indent=1, default=str))
