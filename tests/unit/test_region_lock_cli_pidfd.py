"""Regression tests for region-lock's PIDFD-only child signal forwarding."""
from __future__ import annotations

import contextlib
import io
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from unittest import mock


from src.runtime import region_lock_cli


class FakeChild:
    pid = 4242

    def __init__(self, wait_result=0, wait_hook=None):
        self.wait_result = wait_result
        self.wait_hook = wait_hook
        self.waited = False

    def wait(self):
        self.waited = True
        if self.wait_hook:
            self.wait_hook()
        return self.wait_result


class WaitFailureChild:
    pid = 4343

    def __init__(self):
        self.wait_calls = 0
        self.reaped = False

    def poll(self):
        return 0 if self.reaped else None

    def wait(self, timeout=None):
        self.wait_calls += 1
        if self.wait_calls == 1:
            raise RuntimeError("simulated wait interruption")
        if timeout is not None:
            self.reaped = True
            return -signal.SIGTERM
        self.reaped = True
        return -signal.SIGTERM


class PersistentWaitFailureChild:
    pid = 4545

    def __init__(self):
        self.wait_calls = 0
        self.kill_sent = False
        self.reaped = False

    def wait(self, timeout=None):
        self.wait_calls += 1
        if not self.kill_sent or self.wait_calls < 6:
            raise RuntimeError("persistent simulated wait failure")
        self.reaped = True
        return -signal.SIGKILL


class SustainedWaitFailureChild:
    pid = 4555

    def __init__(self):
        self.wait_calls = 0
        self.kill_sent = False
        self.post_kill_failures = 0
        self.reaped = False

    def wait(self, timeout=None):
        self.wait_calls += 1
        if not self.kill_sent:
            raise RuntimeError("initial wait failure with a long diagnostic " + "x" * 500)
        self.post_kill_failures += 1
        if self.post_kill_failures <= 1000:
            raise RuntimeError("persistent final wait error")
        self.reaped = True
        return -signal.SIGKILL


def fake_signal_registration(handlers):
    def set_handler(signum, handler):
        previous = handlers.get(signum, ("previous", signum))
        handlers[signum] = handler
        return previous
    return set_handler


def test_queued_signal_before_pidfd_open_is_forwarded_through_pidfd():
    handlers, sends, closed = {}, [], []
    child = FakeChild(wait_hook=lambda: handlers[signal.SIGTERM](signal.SIGTERM, None))

    def popen(_command):
        # Simulate TERM after Popen returns but before pidfd_open completes.
        handlers[signal.SIGINT](signal.SIGINT, None)
        return child

    stderr = io.StringIO()
    with mock.patch.object(region_lock_cli.subprocess, "Popen", side_effect=popen), \
         mock.patch.object(region_lock_cli.signal, "signal", side_effect=fake_signal_registration(handlers)), \
         mock.patch.object(region_lock_cli.os, "pidfd_open", return_value=70), \
         mock.patch.object(region_lock_cli.signal, "pidfd_send_signal",
                           side_effect=lambda fd, sig, info, flags: sends.append((fd, sig))), \
         mock.patch.object(region_lock_cli.os, "close", side_effect=lambda fd: closed.append(fd)), \
         contextlib.redirect_stderr(stderr):
        rc = region_lock_cli._spawn(["owned-child"])

    assert rc == 0
    assert child.waited
    assert sends == [(70, signal.SIGINT), (70, signal.SIGTERM)]
    assert closed == [70]
    assert "forwarded signal(s) [<Signals.SIGINT: 2>, <Signals.SIGTERM: 15>]" in stderr.getvalue()


def test_pidfd_open_refusal_waits_owned_child_and_has_no_numeric_fallback():
    handlers, numeric_calls, sends = {}, [], []
    child = FakeChild(wait_hook=lambda: handlers[signal.SIGTERM](signal.SIGTERM, None))

    def open_refused(_pid, _flags):
        raise PermissionError("pidfd denied")

    stderr = io.StringIO()
    with mock.patch.object(region_lock_cli.subprocess, "Popen", return_value=child), \
         mock.patch.object(region_lock_cli.signal, "signal", side_effect=fake_signal_registration(handlers)), \
         mock.patch.object(region_lock_cli.os, "pidfd_open", side_effect=open_refused), \
         mock.patch.object(region_lock_cli.signal, "pidfd_send_signal",
                           side_effect=lambda *args: sends.append(args)), \
         contextlib.redirect_stderr(stderr):
        rc = region_lock_cli._spawn(["owned-child"])

    assert rc == region_lock_cli.EX_TEMPFAIL
    assert child.waited
    assert sends == []
    assert not hasattr(child, "send_signal")
    assert "cannot pin child pidfd" in stderr.getvalue()
    assert "signals not forwarded without child pidfd" in stderr.getvalue()


def test_pidfd_open_refusal_without_signal_still_returns_nonzero():
    child = FakeChild()
    stderr = io.StringIO()
    with mock.patch.object(region_lock_cli.subprocess, "Popen", return_value=child), \
         mock.patch.object(region_lock_cli.signal, "signal", side_effect=fake_signal_registration({})), \
         mock.patch.object(region_lock_cli.os, "pidfd_open",
                           side_effect=PermissionError("pidfd denied")), \
         contextlib.redirect_stderr(stderr):
        rc = region_lock_cli._spawn(["owned-child"])

    assert rc == region_lock_cli.EX_TEMPFAIL
    assert child.waited
    assert "cannot pin child pidfd" in stderr.getvalue()


def test_missing_pidfd_api_refuses_before_spawning_child():
    popen_calls = []
    stderr = io.StringIO()
    with mock.patch.object(region_lock_cli.os, "pidfd_open", create=True, new=None), \
         mock.patch.object(region_lock_cli.signal, "pidfd_send_signal", create=True, new=None), \
         mock.patch.object(region_lock_cli.subprocess, "Popen",
                           side_effect=lambda *args, **kwargs: popen_calls.append(args)), \
         contextlib.redirect_stderr(stderr):
        rc = region_lock_cli._spawn(["must-not-start"])

    assert rc == region_lock_cli.EX_TEMPFAIL
    assert popen_calls == []
    assert "pidfd signaling unavailable" in stderr.getvalue()


def test_partial_signal_handler_registration_failure_refuses_before_spawn():
    handlers, popen_calls = {}, []

    def set_handler(signum, handler):
        if signum == signal.SIGTERM:
            raise ValueError("not main thread")
        previous = handlers.get(signum, ("previous", signum))
        handlers[signum] = handler
        return previous

    stderr = io.StringIO()
    with mock.patch.object(region_lock_cli.signal, "signal", side_effect=set_handler), \
         mock.patch.object(region_lock_cli.subprocess, "Popen",
                           side_effect=lambda *args, **kwargs: popen_calls.append(args)), \
         contextlib.redirect_stderr(stderr):
        rc = region_lock_cli._spawn(["must-not-start"])

    assert rc == region_lock_cli.EX_TEMPFAIL
    assert popen_calls == []
    assert handlers.get(signal.SIGINT) == ("previous", signal.SIGINT)
    assert "cannot install signal handler" in stderr.getvalue()
    assert "refusing child start" in stderr.getvalue()


def test_pidfd_send_error_is_reported_after_child_wait():
    handlers, closed = {}, []
    child = FakeChild(wait_hook=lambda: handlers[signal.SIGTERM](signal.SIGTERM, None))

    def send_refused(*_args):
        raise PermissionError("pidfd send denied")

    stderr = io.StringIO()
    with mock.patch.object(region_lock_cli.subprocess, "Popen", return_value=child), \
         mock.patch.object(region_lock_cli.signal, "signal", side_effect=fake_signal_registration(handlers)), \
         mock.patch.object(region_lock_cli.os, "pidfd_open", return_value=71), \
         mock.patch.object(region_lock_cli.signal, "pidfd_send_signal", side_effect=send_refused), \
         mock.patch.object(region_lock_cli.os, "close", side_effect=lambda fd: closed.append(fd)), \
         contextlib.redirect_stderr(stderr):
        rc = region_lock_cli._spawn(["owned-child"])

    assert rc == region_lock_cli.EX_TEMPFAIL
    assert child.waited
    assert closed == [71]
    assert "signal forwarding refused" in stderr.getvalue()
    assert "pidfd send denied" in stderr.getvalue()


def test_real_owned_child_receives_signal_only_via_open_pidfd():
    handlers = {}
    pidfd_ready = threading.Event()
    real_popen = subprocess.Popen
    real_pidfd_open = os.pidfd_open
    children = []
    trigger = None

    def spawn(command):
        child = real_popen(command, stdin=subprocess.DEVNULL,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        children.append(child)

        def deliver_after_pin():
            assert pidfd_ready.wait(timeout=3), "_spawn never pinned the owned child"
            time.sleep(0.05)
            handlers[signal.SIGTERM](signal.SIGTERM, None)

        nonlocal trigger
        trigger = threading.Thread(target=deliver_after_pin, daemon=True)
        trigger.start()
        return child

    def open_child_pidfd(pid, flags):
        fd = real_pidfd_open(pid, flags)
        pidfd_ready.set()
        return fd

    def set_handler(signum, handler):
        previous = handlers.get(signum, ("previous", signum))
        handlers[signum] = handler
        return previous

    stderr = io.StringIO()
    try:
        with mock.patch.object(region_lock_cli.subprocess, "Popen", side_effect=spawn), \
             mock.patch.object(region_lock_cli.signal, "signal", side_effect=set_handler), \
             mock.patch.object(region_lock_cli.os, "pidfd_open", side_effect=open_child_pidfd), \
             contextlib.redirect_stderr(stderr):
            rc = region_lock_cli._spawn([
                sys.executable, "-B", "-c", "import time; time.sleep(10)"
            ])
        assert rc == 128 + signal.SIGTERM
        assert "forwarded signal(s) [<Signals.SIGTERM: 15>]" in stderr.getvalue()
        assert children and children[0].poll() is not None
    finally:
        if trigger is not None:
            trigger.join(timeout=2)
        for child in children:
            if child.poll() is None:
                fd = real_pidfd_open(child.pid, 0)
                try:
                    signal.pidfd_send_signal(fd, signal.SIGKILL, None, 0)
                finally:
                    os.close(fd)
                child.wait(timeout=3)


def test_wait_exception_drains_only_captured_child_via_pidfd_before_return():
    handlers, sends, closed = {}, [], []
    child = WaitFailureChild()
    stderr = io.StringIO()
    with mock.patch.object(region_lock_cli.subprocess, "Popen", return_value=child), \
         mock.patch.object(region_lock_cli.signal, "signal", side_effect=fake_signal_registration(handlers)), \
         mock.patch.object(region_lock_cli.os, "pidfd_open", return_value=72), \
         mock.patch.object(region_lock_cli.signal, "pidfd_send_signal",
                           side_effect=lambda fd, sig, info, flags: sends.append((fd, sig))), \
         mock.patch.object(region_lock_cli.os, "close", side_effect=lambda fd: closed.append(fd)), \
         contextlib.redirect_stderr(stderr):
        try:
            region_lock_cli._spawn(["owned-child"])
        except RuntimeError as exc:
            assert str(exc) == "simulated wait interruption"
        else:
            raise AssertionError("wait exception was swallowed")

    assert child.reaped, "_spawn returned before its owned child was drained"
    assert sends == [(72, signal.SIGTERM)]
    assert closed == [72]


def test_persistent_wait_errors_keep_lease_until_pidfd_kill_and_reap():
    handlers, sends, closed = {}, [], []
    child = PersistentWaitFailureChild()
    stderr = io.StringIO()
    lease_released_after_reap = []

    def send(fd, signum, _info, _flags):
        sends.append((fd, signum))
        if signum == signal.SIGKILL:
            child.kill_sent = True

    class Lease:
        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            lease_released_after_reap.append(child.reaped)

    with mock.patch.object(region_lock_cli, "_SPAWN_WAIT_POLL_S", 0.001), \
         mock.patch.object(region_lock_cli, "_SPAWN_TERM_GRACE_S", 0.005), \
         mock.patch.object(region_lock_cli.subprocess, "Popen", return_value=child), \
         mock.patch.object(region_lock_cli.signal, "signal", side_effect=fake_signal_registration(handlers)), \
         mock.patch.object(region_lock_cli.os, "pidfd_open", return_value=73), \
         mock.patch.object(region_lock_cli.signal, "pidfd_send_signal", side_effect=send), \
         mock.patch.object(region_lock_cli.os, "close", side_effect=lambda fd: closed.append(fd)), \
         contextlib.redirect_stderr(stderr):
        try:
            with Lease():
                region_lock_cli._spawn(["owned-child"])
        except RuntimeError as exc:
            assert str(exc) == "persistent simulated wait failure"
        else:
            raise AssertionError("original wait failure was swallowed")

    assert child.reaped
    assert lease_released_after_reap == [True]
    assert sends == [(73, signal.SIGTERM), (73, signal.SIGKILL)]
    assert closed == [73]


def test_pidfd_open_failure_retries_wait_errors_without_numeric_signal():
    handlers = {}

    class OpenFailureWaitChild:
        pid = 4646

        def __init__(self):
            self.wait_calls = 0
            self.reaped = False

        def wait(self, timeout=None):
            self.wait_calls += 1
            if self.wait_calls <= 3:
                raise RuntimeError("transient wait failure without pidfd")
            self.reaped = True
            return 0

    child = OpenFailureWaitChild()
    numeric_calls, sends = [], []
    stderr = io.StringIO()
    with mock.patch.object(region_lock_cli, "_SPAWN_WAIT_POLL_S", 0.001), \
         mock.patch.object(region_lock_cli.subprocess, "Popen", return_value=child), \
         mock.patch.object(region_lock_cli.signal, "signal", side_effect=fake_signal_registration(handlers)), \
         mock.patch.object(region_lock_cli.os, "pidfd_open", side_effect=PermissionError("pidfd denied")), \
         mock.patch.object(region_lock_cli.signal, "pidfd_send_signal", side_effect=lambda *a: sends.append(a)), \
         mock.patch.object(region_lock_cli.os, "kill", side_effect=lambda *a: numeric_calls.append(a)), \
         contextlib.redirect_stderr(stderr):
        try:
            region_lock_cli._spawn(["owned-child"])
        except RuntimeError as exc:
            assert str(exc) == "transient wait failure without pidfd"
        else:
            raise AssertionError("original wait failure was swallowed")

    assert child.reaped and child.wait_calls >= 4
    assert numeric_calls == []
    assert sends == []


def test_sustained_wait_and_backoff_errors_are_bounded_until_reap():
    handlers, sends = {}, []
    child = SustainedWaitFailureChild()
    stderr = io.StringIO()
    lease_released_after_reap = []
    sleep_calls = []

    def send(fd, signum, _info, _flags):
        sends.append((fd, signum))
        if signum == signal.SIGKILL:
            child.kill_sent = True

    def interrupted_backoff(_delay):
        sleep_calls.append(1)
        if len(sleep_calls) <= 5:
            raise InterruptedError("backoff interrupted")

    class Lease:
        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            lease_released_after_reap.append(child.reaped)

    with mock.patch.object(region_lock_cli, "_SPAWN_WAIT_POLL_S", 0.0001), \
         mock.patch.object(region_lock_cli, "_SPAWN_TERM_GRACE_S", 0.001), \
         mock.patch.object(region_lock_cli.time, "sleep", side_effect=interrupted_backoff), \
         mock.patch.object(region_lock_cli.subprocess, "Popen", return_value=child), \
         mock.patch.object(region_lock_cli.signal, "signal", side_effect=fake_signal_registration(handlers)), \
         mock.patch.object(region_lock_cli.os, "pidfd_open", return_value=74), \
         mock.patch.object(region_lock_cli.signal, "pidfd_send_signal", side_effect=send), \
         mock.patch.object(region_lock_cli.os, "close"), \
         contextlib.redirect_stderr(stderr):
        try:
            with Lease():
                region_lock_cli._spawn(["owned-child"])
        except RuntimeError as exc:
            assert str(exc).startswith("initial wait failure")
        else:
            raise AssertionError("original wait failure was swallowed")

    assert child.reaped
    assert child.post_kill_failures == 1001
    assert lease_released_after_reap == [True]
    assert sends == [(74, signal.SIGTERM), (74, signal.SIGKILL)]
    assert len(sleep_calls) >= 5
    summary = [line for line in stderr.getvalue().splitlines() if "child-reap retry" in line]
    assert len(summary) <= 4
    assert all(len(line) < 400 for line in summary)
    assert any("final wait after SIGKILL" in line and "count=999" in line for line in summary)
    assert any("wait after signal 9" in line and "count=1" in line for line in summary)


def test_source_has_no_numeric_child_signal_fallback():
    source = Path(region_lock_cli.__file__).read_text(encoding="utf-8")
    spawn = source.split("def _spawn(", 1)[1].split("\ndef _status(", 1)[0]
    assert "proc.send_signal(" not in spawn
    assert "os.kill(" not in spawn
