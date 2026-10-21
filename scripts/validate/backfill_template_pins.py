#!/usr/bin/env python3
"""SEC-ART-4 backfill planner: compute `served_template_sha256` for every launched GGUF role.

For each `server_mode.<role>` that serves a GGUF (aliases skipped: they launch nothing), the EFFECTIVE template is
what llama-server will render: the bytes of `chat_template_file` when declared, else the GGUF's embedded
`tokenizer.chat_template` (header-only read, a few MB of I/O, never tensor data). The plan lists the sha256, its byte
length, and -- where the registry already carries the 12-hex `served_template_sha256_12` -- whether that prefix
agrees. A disagreement is a FINDING (the old pin was stale or described another artifact), never silently
overwritten. Apply a plan with `apply_registry_integrity_pins.py`.

    backfill_template_pins.py [--registry master-or-lean.yaml] [--out plan.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.registry import artifact_integrity as ai  # noqa: E402

MODELS_DIR = "/mnt/raid0/llm/models"


def _server_model_path(cfg: dict[str, Any]) -> str | None:
    model = cfg.get("model")
    path = model.get("path") if isinstance(model, dict) else model
    if not (isinstance(path, str) and path.endswith(".gguf")):
        return None
    return path if path.startswith("/") else os.path.join(MODELS_DIR, path)


def plan(registry: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for role, cfg in sorted((registry.get("server_mode") or {}).items()):
        if not isinstance(cfg, dict) or cfg.get("alias_of"):
            continue
        gguf = _server_model_path(cfg)
        if gguf is None:
            continue
        row: dict[str, Any] = {"role": role, "gguf": gguf}
        tfile = cfg.get("chat_template_file")
        try:
            if tfile:
                data, source = Path(tfile).read_bytes(), f"file:{tfile}"
            elif os.path.exists(gguf):
                data, source = ai.read_gguf_chat_template(gguf), f"gguf:{gguf}"
            else:
                rows.append({**row, "status": "missing_gguf"})
                continue
        except (OSError, ai.ArtifactIntegrityError) as exc:
            rows.append({**row, "status": "unreadable", "error": str(exc)})
            continue
        if data is None:
            rows.append({**row, "status": "no_template", "source": source})
            continue
        digest = ai.template_sha256(data)
        row.update(source=source, bytes=len(data), served_template_sha256=digest)
        block = cfg.get("chat_template") if isinstance(cfg.get("chat_template"), dict) else {}
        old = block.get("served_template_sha256_12")
        have = block.get("served_template_sha256")
        if have and have != digest:
            row["status"] = "CONFLICT_existing_64hex_differs"
        elif old and str(old) != digest[:12]:
            row["status"] = "CONFLICT_existing_12hex_differs"
            row["existing_12"] = str(old)
        else:
            row["status"] = "pin"
        rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--registry", default=str(Path(__file__).resolve().parents[2] / "orchestration/model_registry.yaml"))
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    reg = yaml.safe_load(Path(a.registry).read_text())
    rows = plan(reg)
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    doc = json.dumps({"counts": counts, "rows": rows}, indent=2)
    if a.out:
        Path(a.out).write_text(doc + "\n")
    else:
        print(doc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
