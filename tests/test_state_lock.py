"""Tests for the cross-process autopilot_state.json write lock (H4 fix)."""
import errno
import fcntl
import os
import time

import pytest

from scripts.autopilot.state_lock import state_write_lock


def test_yields_true_when_acquired(tmp_path):
    sp = tmp_path / "autopilot_state.json"
    with state_write_lock(sp) as held:
        assert held is True


def test_lockfile_created_next_to_state(tmp_path):
    sp = tmp_path / "autopilot_state.json"
    with state_write_lock(sp):
        assert (tmp_path / "autopilot_state.json.lock").exists()


def test_mutual_exclusion_while_held(tmp_path):
    sp = tmp_path / "autopilot_state.json"
    lock_path = f"{sp}.lock"
    with state_write_lock(sp) as held:
        assert held is True
        # A second independent fd must NOT be able to take LOCK_EX (non-blocking).
        other = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            with pytest.raises(OSError) as ei:
                fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
            assert ei.value.errno in (errno.EAGAIN, errno.EACCES, errno.EWOULDBLOCK)
        finally:
            os.close(other)


def test_released_after_block(tmp_path):
    sp = tmp_path / "autopilot_state.json"
    lock_path = f"{sp}.lock"
    with state_write_lock(sp):
        pass
    # After the block the lock is free — another fd can take it immediately.
    other = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)  # must not raise
        fcntl.flock(other, fcntl.LOCK_UN)
    finally:
        os.close(other)


def test_released_on_exception(tmp_path):
    sp = tmp_path / "autopilot_state.json"
    lock_path = f"{sp}.lock"
    with pytest.raises(ValueError):
        with state_write_lock(sp):
            raise ValueError("boom")
    other = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)  # lock was released
        fcntl.flock(other, fcntl.LOCK_UN)
    finally:
        os.close(other)


def test_fail_open_on_timeout(tmp_path):
    sp = tmp_path / "autopilot_state.json"
    lock_path = f"{sp}.lock"
    # Hold the lock on a separate fd, then a short-timeout acquire must fail OPEN
    # (yield False) rather than hang or raise.
    holder = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    fcntl.flock(holder, fcntl.LOCK_EX)
    try:
        t0 = time.monotonic()
        with state_write_lock(sp, timeout=0.2) as held:
            assert held is False  # failed open
        elapsed = time.monotonic() - t0
        assert 0.2 <= elapsed < 3.0  # waited ~timeout then proceeded, did not hang
    finally:
        fcntl.flock(holder, fcntl.LOCK_UN)
        os.close(holder)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-p", "no:xdist"]))
