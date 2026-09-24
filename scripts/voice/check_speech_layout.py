#!/usr/bin/env python3
"""Check the speech aux services' declared CPU layout against scripts/voice/speech_layouts.yaml.

Exit 0 when the declared (threads, cpuset) of whisper (and tts, when both are CPU-pinned)
satisfies the acceptance rule stated in that file; non-zero with the reason otherwise.
Read-only: parses the launch manifest, starts nothing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
LAYOUTS = REPO / "scripts" / "voice" / "speech_layouts.yaml"
MANIFEST = REPO / "orchestration" / "launch_manifest.yaml"


def _threads(entry: dict, name: str) -> int | None:
    argv = [str(t) for t in entry.get("argv", [])]
    if name == "whisper":
        return int(argv[argv.index("-t") + 1]) if "-t" in argv else None
    nprocs = (entry.get("env") or {}).get("SHIM_NPROCS")
    return int(nprocs) // 2 if nprocs else None  # qwentts uses hardware_concurrency()/2


def declared_layouts(manifest: Path = MANIFEST) -> dict[str, dict]:
    services = {s["name"]: s for s in yaml.safe_load(manifest.read_text())["aux_services"]}
    out = {}
    for name in ("whisper", "tts"):
        entry = services.get(name) or {}
        if entry.get("cpuset"):
            out[name] = {"threads": _threads(entry, name), "cpuset": str(entry["cpuset"])}
    return out


def check(manifest: Path = MANIFEST, layouts: Path = LAYOUTS) -> list[str]:
    table = yaml.safe_load(layouts.read_text())
    decl = declared_layouts(manifest)
    problems: list[str] = []
    stt, tts = decl.get("whisper"), decl.get("tts")
    if stt and stt in table.get("stt_known_bad", []):
        problems.append(f"whisper {stt} is a KNOWN-BAD layout (encode hangs)")
    if stt and tts:
        if not any(p["stt"] == stt and p["tts"] == tts for p in table.get("concurrent_ok", [])):
            problems.append(f"whisper {stt} + tts {tts} is not a measured concurrent_ok pair")
    elif stt:
        if stt not in table.get("stt_solo_ok", []):
            problems.append(f"whisper {stt} is not a measured stt_solo_ok layout")
    elif tts:
        if not any(p["tts"] == tts for p in table.get("concurrent_ok", [])):
            problems.append(f"tts {tts} has no measured layout")
    return problems


if __name__ == "__main__":
    issues = check()
    for issue in issues:
        print(f"REFUSING: {issue}", file=sys.stderr)
    print("speech layout OK" if not issues else "speech layout NOT measured-safe")
    sys.exit(1 if issues else 0)
