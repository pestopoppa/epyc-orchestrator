from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def stable_hash(payload: Any, *, n: int = 16) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno}: invalid JSONL row: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{lineno}: JSONL row must be an object")
        row["_source_line"] = lineno
        rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
            count += 1
    return count


def write_manifest(
    path: Path,
    *,
    builder: str,
    generated_at: str,
    source_path: Path,
    output_path: Path,
    counts: dict[str, Any],
    options: dict[str, Any],
) -> None:
    manifest = {
        "schema_version": "dataset_builder_manifest.v1",
        "builder": builder,
        "generated_at": generated_at,
        "source_path": str(source_path),
        "source_sha256": file_sha256(source_path) if source_path.exists() else None,
        "output_path": str(output_path),
        "counts": counts,
        "options": options,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
