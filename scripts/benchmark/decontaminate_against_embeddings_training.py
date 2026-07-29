#!/usr/bin/env python3
"""LightOn-compatible retrieval-eval decontamination (exact xxHash64 + 13-grams).

Normalizes text as lower-case Unicode NFKD with collapsed whitespace.  A sample
is rejected when its exact normalized-text hash is in training, or when at
least 50% of its word-level 13-grams occur in training.  This is a static file
transform: it never starts a model, server, or GPU workload.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

MASK = (1 << 64) - 1
P1, P2, P3, P4, P5 = (
    11400714785074694791,
    14029467366897019727,
    1609587929392839161,
    9650029242287828579,
    2870177450012600261,
)


def _rotl(x: int, n: int) -> int:
    x &= MASK
    return ((x << n) | (x >> (64 - n))) & MASK


def _round(acc: int, value: int) -> int:
    acc = (acc + value * P2) & MASK
    acc = _rotl(acc, 31)
    return (acc * P1) & MASK


def _merge(acc: int, value: int) -> int:
    acc ^= _round(0, value)
    return (_rotl(acc, 27) * P1 + P4) & MASK


def xxhash64(data: bytes, seed: int = 0) -> int:
    """Portable xxHash64, seed 0 by default (no optional package required)."""
    length, i = len(data), 0
    if length >= 32:
        v1, v2, v3, v4 = (seed + P1 + P2) & MASK, (seed + P2) & MASK, seed, (seed - P1) & MASK
        limit = length - 32
        while i <= limit:
            v1 = _round(v1, int.from_bytes(data[i : i + 8], "little"))
            v2 = _round(v2, int.from_bytes(data[i + 8 : i + 16], "little"))
            v3 = _round(v3, int.from_bytes(data[i + 16 : i + 24], "little"))
            v4 = _round(v4, int.from_bytes(data[i + 24 : i + 32], "little"))
            i += 32
        h = (_rotl(v1, 1) + _rotl(v2, 7) + _rotl(v3, 12) + _rotl(v4, 18)) & MASK
        for value in (v1, v2, v3, v4):
            h = _merge(h, value)
    else:
        h = (seed + P5) & MASK
    h = (h + length) & MASK
    while i + 8 <= length:
        k = _round(0, int.from_bytes(data[i : i + 8], "little"))
        h ^= k
        h = (_rotl(h, 27) * P1 + P4) & MASK
        i += 8
    if i + 4 <= length:
        h ^= (int.from_bytes(data[i : i + 4], "little") * P1) & MASK
        h = (_rotl(h, 23) * P2 + P3) & MASK
        i += 4
    while i < length:
        h ^= data[i] * P5
        h = (_rotl(h, 11) * P1) & MASK
        i += 1
    h ^= h >> 33
    h = (h * P2) & MASK
    h ^= h >> 29
    h = (h * P3) & MASK
    return (h ^ (h >> 32)) & MASK


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKD", text).lower().split())


def grams(text: str, n: int = 13) -> set[int]:
    words = re.findall(r"\S+", normalize(text))
    return {xxhash64(" ".join(words[i : i + n]).encode()) for i in range(len(words) - n + 1)}


def containment(sample: str, training_grams: set[int]) -> float:
    sample_grams = grams(sample)
    return 0.0 if not sample_grams else len(sample_grams & training_grams) / len(sample_grams)


def _text(row: dict, fields: list[str]) -> str:
    return "\n".join(str(row.get(field, "")) for field in fields if row.get(field) is not None)


def _rows(path: Path):
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def run(
    training: Path, evaluation: Path, output: Path, fields: list[str], threshold: float = 0.5
) -> dict:
    if not fields:
        raise ValueError("at least one --text-field is required")
    train_texts = [_text(row, fields) for row in _rows(training)]
    exact = {xxhash64(normalize(text).encode()) for text in train_texts}
    training_grams = set().union(*(grams(text) for text in train_texts)) if train_texts else set()
    kept = rejected = 0
    with output.open("w") as handle:
        for row in _rows(evaluation):
            text = _text(row, fields)
            exact_hit = xxhash64(normalize(text).encode()) in exact
            ratio = containment(text, training_grams)
            contaminated = exact_hit or ratio >= threshold
            row["_decontamination"] = {
                "method": "xxhash64_exact_then_word_13gram",
                "normalization": "lowercase_unicode_nfkd_whitespace_collapsed",
                "threshold": threshold,
                "exact_hash_hit": exact_hit,
                "containment": ratio,
                "contaminated": contaminated,
            }
            if contaminated:
                rejected += 1
            else:
                kept += 1
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    return {
        "schema": "epyc.retrieval_decontamination.v1",
        "training_rows": len(train_texts),
        "kept": kept,
        "rejected": rejected,
        "threshold": threshold,
        "output": str(output),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--text-field", action="append", default=[])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be in [0, 1]")
    report = run(args.training, args.evaluation, args.output, args.text_field, args.threshold)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.write_text(payload)
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
