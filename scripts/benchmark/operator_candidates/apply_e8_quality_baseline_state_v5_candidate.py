#!/usr/bin/env python3
"""Proposed minimal human-owned applier amendment: accept protocol v5."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import importlib.util
import os
from pathlib import Path
import stat
import sys


APPLIER = Path("/mnt/raid0/llm/epyc-root/artifacts/operator/apply_e8_quality_baseline_state.py")
spec = importlib.util.spec_from_file_location("e8_v5_base_applier", APPLIER)
if spec is None or spec.loader is None:
    raise SystemExit("ERROR: cannot import canonical E8 state applier")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
module.EXPECTED_PROTOCOL = "e8_quality_full_pool_tier_baseline.v5"

TRUST_LOCK = Path(
    os.environ.get(
        "E8_V5_TRUST_LOCK", "/run/lock/epyc-measurement-trust-boundary.lock"
    )
)


@contextmanager
def measurement_trust_boundary_lock():
    """Acquire the campaign lock or verify the wrapper's inherited lock FD."""
    if "E8_V5_TRUST_LOCK" in os.environ and not (
        os.environ.get("E8_V5_TEST_MODE") == "1"
        and os.environ.get("PYTEST_CURRENT_TEST")
    ):
        raise module.ApplyError("noncanonical measurement lock is pytest-only")

    inherited_text = os.environ.get("EPYC_MEASUREMENT_TRUST_LOCK_FD")
    inherited = inherited_text is not None
    if inherited:
        try:
            fd = int(inherited_text)
        except (TypeError, ValueError) as exc:
            raise module.ApplyError("inherited measurement lock FD is malformed") from exc
    else:
        try:
            nofollow = getattr(os, "O_NOFOLLOW", 0)
            parent_fd = os.open(
                TRUST_LOCK.parent, os.O_RDONLY | os.O_DIRECTORY | nofollow
            )
            try:
                fd = os.open(
                    TRUST_LOCK.name,
                    os.O_RDWR | os.O_CREAT | nofollow,
                    0o660,
                    dir_fd=parent_fd,
                )
                os.fsync(fd)
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
        except OSError as exc:
            raise module.ApplyError(
                f"cannot open measurement trust-boundary lock: {TRUST_LOCK}"
            ) from exc

    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise module.ApplyError(
                f"measurement trust-boundary lock is already held: {TRUST_LOCK}"
            ) from exc
        named = os.stat(TRUST_LOCK, follow_symlinks=False)
        held = os.fstat(fd)
        if (
            not stat.S_ISREG(named.st_mode)
            or (named.st_dev, named.st_ino) != (held.st_dev, held.st_ino)
        ):
            raise module.ApplyError(
                "measurement trust-boundary lock inode changed during acquisition"
            )
        yield
    finally:
        if not inherited:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)


if __name__ == "__main__":
    try:
        with measurement_trust_boundary_lock():
            raise SystemExit(module.main())
    except module.ApplyError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
