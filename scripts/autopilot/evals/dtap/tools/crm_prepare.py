#!/usr/bin/env python3
"""Hash candidate inputs before any compute grant; not runtime identity proof."""
import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from harness.crm_source_manifest import write_manifest


def file_fact(path):
    path = Path(path).resolve(strict=True)
    before = path.stat()
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise RuntimeError("input changed while hashing")
    return {"path": str(path), "sha256": h.hexdigest(), "device": after.st_dev,
            "inode": after.st_ino, "size": after.st_size, "mtime_ns": after.st_mtime_ns,
            "ctime_ns": after.st_ctime_ns}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "commit", "model", "server", "template", "source-manifest", "input-manifest"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    if os.environ.get("FILLER_ID") or os.environ.get("FILLER_DRYRUN"):
        raise RuntimeError("input hashing must run outside grant or dry-run")
    source = write_manifest(a.source_manifest, a.repo, a.commit)
    record = {"schema": "dtap.crm_input_preparation.v1", "scope": "preflight_only_not_runtime_binding",
              "created_utc": datetime.now(timezone.utc).isoformat(), "source_commit": source["source_commit"],
              "source_tree_sha256": source["source_tree_sha256"],
              "model": file_fact(a.model), "server": file_fact(a.server), "template": file_fact(a.template)}
    data = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode()
    fd = os.open(a.input_manifest, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    directory_fd = os.open(Path(a.input_manifest).parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
