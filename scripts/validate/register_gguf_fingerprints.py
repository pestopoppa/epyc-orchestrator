#!/usr/bin/env python3
"""SEC-ART-2 registration: hash each lineup GGUF in full ONCE and record its stat fingerprint.

    register_gguf_fingerprints.py register [--role R]... [--path P]... [--adopt-unpinned] [--registry F]
    register_gguf_fingerprints.py check    [--registry F]          # stat only, never reads GGUF bytes

`register` full-hashes every GGUF named by a lineup role (every shard of a split set) and writes
(size, mtime_ns, inode, sha256) to the host-local ledger (`ARTIFACT_FINGERPRINT_LEDGER`, default
/mnt/raid0/llm/cache/artifact_fingerprints.json). A digest that differs from the registry pin is a hard
failure and records NOTHING. A file with no pin is skipped unless `--adopt-unpinned`, which records the digest as
a PROPOSAL (`pin_sha256: null`) and prints it for review into `roles.<role>.model.sha256` -- an adopted hash
is not a verification of anything upstream.

The launcher (`orchestrator_stack.build_server_command` -> `artifact_integrity.verify_launch_argv`) then compares only the
stat fingerprint and re-hashes a file only when it changed. Hashing 30-90 GB reads the whole file: run it under
the CPU scheduling policy (a correctness/IO task, not a timing window) and never during a timing run.

`register` takes an ordinary flock-protected ledger update per file; it never touches a running server.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.registry import artifact_integrity as ai  # noqa: E402


def _targets(registry: dict, roles: list[str] | None, paths: list[str] | None) -> list[tuple[str, ai.GgufPin | None, str]]:
    """[(role label, pin, file)] for every file to process, split sets expanded to every shard."""
    index = ai.pin_index(registry)
    out: list[tuple[str, ai.GgufPin | None, str]] = []
    seen: set[str] = set()
    if paths:
        for p in paths:
            pin = index.get(os.path.realpath(p))
            for f in ai.shard_paths(p):
                if os.path.realpath(f) not in seen:
                    seen.add(os.path.realpath(f))
                    out.append((pin.role if pin else "-", pin, f))
        return out
    for pin in index.values():
        if roles and pin.role not in roles:
            continue
        for f in ai.shard_paths(pin.path):
            if os.path.realpath(f) not in seen:
                seen.add(os.path.realpath(f))
                out.append((pin.role, pin, f))
    return out


def _expected(pin: ai.GgufPin | None, f: str) -> str | None:
    if pin is None:
        return None
    if os.path.realpath(f) == os.path.realpath(pin.path):
        return pin.sha256
    return pin.shard_sha256.get(os.path.basename(f))


def cmd_register(args: argparse.Namespace) -> int:
    registry = ai.load_registry(args.registry)
    rc = 0
    rows = []
    for role, pin, f in _targets(registry, args.role, args.path):
        want = _expected(pin, f)
        row = {"role": role, "file": f, "pin": want}
        if not os.path.exists(f):
            rows.append({**row, "status": "missing_file"})
            rc = 1
            continue
        if want is None and not args.adopt_unpinned:
            rows.append({**row, "status": "skipped_unpinned (pass --adopt-unpinned to record a proposal)"})
            continue
        try:
            entry = ai.register_gguf(f, pin_sha256=want)
        except ai.ArtifactIntegrityError as exc:
            rows.append({**row, "status": f"REFUSED: {exc}"})
            rc = 1
            continue
        rows.append({**row, "status": "registered" if want else "adopted_proposal",
                     "sha256": entry["sha256"], "hash_seconds": entry["hash_seconds"]})
    print(json.dumps({"ledger": str(ai.ledger_path()), "rows": rows}, indent=1))
    return rc


def cmd_check(args: argparse.Namespace) -> int:
    registry = ai.load_registry(args.registry)
    ledger = ai.load_ledger()
    rc = 0
    rows = []
    for role, pin, f in _targets(registry, args.role, args.path):
        want = _expected(pin, f)
        real = os.path.realpath(f)
        entry = ledger.get(real)
        if not os.path.exists(f):
            status = "missing_file"
            rc = 1
        elif want is None:
            status = "unpinned"
        elif entry is None:
            status = "pinned_unregistered"
        else:
            fp = ai.stat_fingerprint(real)
            same = all(entry.get(k) == fp[k] for k in ("size", "mtime_ns", "ino"))
            status = "ok" if same and entry.get("sha256") == want else "CHANGED_OR_PIN_DRIFT"
            rc = rc or (0 if status == "ok" else 1)
        rows.append({"role": role, "file": f, "status": status})
    print(json.dumps({"ledger": str(ai.ledger_path()), "rows": rows}, indent=1))
    return rc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("register", "check"):
        sp = sub.add_parser(name)
        sp.add_argument("--registry", help="lean registry (default orchestration/model_registry.yaml)")
        sp.add_argument("--role", action="append")
        sp.add_argument("--path", action="append")
        if name == "register":
            sp.add_argument("--adopt-unpinned", action="store_true")
    a = ap.parse_args(argv)
    return cmd_register(a) if a.cmd == "register" else cmd_check(a)


if __name__ == "__main__":
    sys.exit(main())
