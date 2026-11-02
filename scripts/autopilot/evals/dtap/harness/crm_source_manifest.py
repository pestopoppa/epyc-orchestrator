"""Pre-capture source manifest: bind Git blobs to currently readable files.

This records on-disk source custody, not already-imported Python bytecode.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

SCHEMA = "dtap.crm_source_manifest.v1"
DTAP = "scripts/autopilot/evals/dtap/"
FILES = tuple(DTAP + path for path in (
    "harness/crm_pair_capture.py", "harness/crm_pair_seal.py", "harness/crm_pair.py",
    "harness/crm_receipt.py", "harness/crm_services.py", "harness/crm_provision.py",
    "harness/crm_source_manifest.py", "harness/runner.py", "harness/endpoint.py",
    "harness/tool_contract.py", "harness/tool_schemas.json", "harness/judge_guard.py",
    "cases.json", "manifest.json", "judges/crm-malicious-indirect-general-ai-restrictions-004/judge.py",
    "tools/crm_prepare.py", "tools/crm_row_fixture.py", "tools/crm_qual_server.py", "tools/crm_qual_worker.py",
)) + ("src/runtime/region_claim_receipt.py", "src/runtime/region_lock_cli.py",
      "src/runtime/cpu_region_lock.py")


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], check=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, timeout=30).stdout


def _mapping_hash(mapping):
    return _sha(json.dumps(mapping, sort_keys=True, separators=(",", ":")).encode())


def source_facts(root, commit):
    root = Path(root).resolve()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("source commit must be a full SHA")
    if _git(root, "rev-parse", "HEAD").decode().strip() != commit:
        raise ValueError("source checkout HEAD does not match declared commit")
    files = {}
    for name in FILES:
        current = (root / name).read_bytes()
        committed = _git(root, "show", f"{commit}:{name}")
        if current != committed:
            raise ValueError(f"working source differs from Git blob: {name}")
        files[name] = _sha(current)
    return {"schema": SCHEMA, "repo_path": str(root), "source_commit": commit,
            "files": files, "source_tree_sha256": _mapping_hash(files)}


def write_manifest(path, root, commit):
    facts = source_facts(root, commit)
    data = (json.dumps(facts, sort_keys=True, separators=(",", ":")) + "\n").encode()
    path = Path(path)
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return facts


def verify_manifest(path, commit, expected_tree_sha256):
    record = json.loads(Path(path).read_bytes())
    if not isinstance(record, dict) or record.get("schema") != SCHEMA:
        raise ValueError("invalid CRM source manifest")
    if record.get("source_commit") != commit or record.get("source_tree_sha256") != expected_tree_sha256:
        raise ValueError("CRM source identity differs from provenance")
    actual = source_facts(record["repo_path"], commit)
    if record != actual:
        raise ValueError("CRM source manifest differs from committed working files")
    return record
