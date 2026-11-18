"""D-RI-CONTAIN-T2: an opt-in killable, resource-limited CPython child lane for REPL cells.

Flag ``ORCHESTRATOR_REPL_CHILD_LANE`` (default OFF). With the flag off nothing in this module
runs: ``REPLEnvironment`` takes exactly the in-process path it always took.

Design
------
* ONE warm child per REPLEnvironment (== per request), started lazily by the first cell and
  kept for the request's later cells. It is launched through ``exec_sandbox.wrap`` (Landlock +
  seccomp: no sockets, credential read-deny, foreign /proc hidden), in its own session, with a
  private scratch cwd/HOME and a minimal environment (``exec_containment.child_env``). The
  child runs ``_repl_child_main.py`` (stdlib only, no ``src`` imports) under ``python -I -S``.
* Budgets, each verified, never assumed: RLIMIT_AS (set by the shim, re-read by the child),
  RLIMIT_CPU / NOFILE / FSIZE / CORE=0 (set by the child, reported in its ``ready`` frame and
  compared by the parent), and, when a writable cgroup v2 directory is available, an owned
  cgroup with ``memory.max`` / ``memory.swap.max=0`` / ``pids.max`` (controllers are enabled
  only if the parent cgroup already delegates them; this module NEVER writes a parent's
  ``cgroup.subtree_control``). ``ORCHESTRATOR_REPL_CHILD_CGROUP=require`` makes a missing
  cgroup limit fatal; ``off`` skips it. If a required control cannot be applied the lane
  REFUSES the cell. It never falls back to inline execution.
* The child holds no authority. Every callback is a bounded JSON value message that the parent
  authorises against a parent-owned table (``BROKER``): request identity (a per-cell nonce),
  monotonic sequence, deadline, optional lease validator, per-snippet tool-call budget, argument
  validation, and result size. EVERY callable the in-process REPL exposes is bridged (file writers,
  TOOL/CALL, llm_call/llm_batch, run_shell/run_python_code (each in its own exec_sandbox child),
  SCRIPT, web, list/search/describe_tool, the CM-2..CM-5 helpers when their flags are on...): the
  surface is enumerated from the in-process globals builder by ``tool_broker.bridged_tools`` (shared
  with other lanes) and each call runs in the parent through the very callable an in-process cell
  would call. Only non-callable preloads (numpy/scipy modules) are stubs that raise.
* State lives in the parent. After every cell the child returns its variables and artifacts through
  a typed JSON codec (``_repl_child_main.encode``/``decode``): plain JSON values; tuple, set,
  frozenset, bytes, complex, Counter, OrderedDict, defaultdict, deque, Decimal, Fraction,
  datetime/date/time/timedelta and non-str-keyed dicts as tagged records (real types come back in
  the parent); allowlisted stdlib modules/functions/classes by reference; and the model's own
  top-level ``def`` / ``class`` / ``name = <expr>`` definitions as carried SOURCE, re-executed in
  the next child after a crash, with ``inst`` records for instances of those classes. The parent
  stores carried source as ``LaneOpaque`` and never runs it; the source must pass the REPL's AST
  security gate first. The parent mirrors state into ``env._globals`` / ``env.artifacts``, so
  ``get_state()``, checkpoints, SQLite persistence, signing and fencing are untouched. Only what
  truly cannot cross (generators, bound/builtin methods, closures, ``__slots__``/metaclass
  instances, cycles, wrapped callables such as ``lru_cache``, a def nested under ``if``/``for``)
  rolls the cell back with a named error; its output is kept.
* Structured (React) mode: its single-tool fall-through ``exec`` runs in the lane too. The tools the
  in-process path exposes become parent-owned callbacks (captured from the parent's globals at
  child start, so the child cannot rebind them); budget, argument validation and write dedupe are
  unchanged, and only the model's Python around the call is contained.
* Callbacks run on a worker thread bounded by the cell deadline (any calling thread; request
  contextvars propagated); a pathological regex is bounded by ``safe_regex``. Errors surface as
  ``<Type>: {"origin": ..., "message": ...}`` (``exception_schema``), a bounded, defanged schema.
* Lifetime: ``closes_lanes`` / ``lane_scope`` reap every lane a request created on every exit path;
  ``env._lane_lease_validator`` (``HeldSessionLease.validator``) revokes admission and kills the
  child when the session lease or fencing token is lost mid-cell.
* Deadline: the parent kills the child (cgroup.kill + killpg SIGKILL, reap, then a scan for
  survivors in the child's session/cgroup) and the environment goes terminal exactly like the
  in-process timeout (``mark_timed_out``). Crash / OOM / protocol violation: the child is
  reaped, the cell returns an error, and the next cell starts a fresh child restored from the
  parent's last completed state (bounded by ``ORCHESTRATOR_REPL_CHILD_MAX_RESPAWNS``).
* IPC is length-prefixed JSON; the length is checked against the cap BEFORE any payload byte
  is accumulated, in both directions; the child's stdout/stderr capture is capped at the source.
"""

from __future__ import annotations

import atexit
import contextvars
import errno
import ctypes
import json
import logging
import math
import os
import secrets
import select
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import threading
import time
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from src.repl_environment import tool_broker

logger = logging.getLogger(__name__)

FLAG_ENV = "ORCHESTRATOR_REPL_CHILD_LANE"
_TRUE = {"1", "true", "yes", "on"}
CHILD_SCRIPT = Path(__file__).with_name("_repl_child_main.py")

#: The two control signals the child implements itself (they end the cell); the parent executes the
#: matching REPLEnvironment method. Every other tool is enumerated by ``tool_broker.bridged_tools``.
BROKER: dict[str, str] = dict(tool_broker.CONTROL)
_CHILD_DEFINED = frozenset(tool_broker.CONTROL)
_CORE_NAMES = tool_broker.CORE_NAMES


def enabled() -> bool:
    from src.repl_environment.tool_arg_validation import feature_on

    return feature_on("repl_child_lane", FLAG_ENV)


class LaneError(RuntimeError):
    """Base for lane failures."""


class LaneUnavailable(LaneError):
    """Containment cannot be established or a control cannot be applied: refuse, never run inline."""


class LaneCrash(LaneError):
    """The child died or violated the protocol; it was reaped."""


class LaneDeadline(LaneError):
    """The wall-clock deadline elapsed; the child was killed."""


class LaneRevoked(LaneError):
    """Cancelled or lease lost; the child was killed."""


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class LaneConfig:
    as_mb: int = 2048
    cpu_s: int = 600
    nofile: int = 64
    fsize_mb: int = 8
    output_chars: int = 4_000_000
    context_cap_chars: int = 32 * 1024 * 1024
    max_frame: int = 16 * 1024 * 1024
    max_inbound_frame: int = 90 * 1024 * 1024
    max_reply_bytes: int = 4 * 1024 * 1024
    spawn_timeout_s: float = 20.0
    max_respawns: int = 5
    max_calls_per_cell: int = 5000
    cgroup_mode: str = "auto"  # auto | require | off
    cgroup_memory_mb: int = 0  # 0 -> as_mb + 512
    cgroup_pids_max: int = 64

    @classmethod
    def from_env(cls) -> "LaneConfig":
        mode = os.environ.get("ORCHESTRATOR_REPL_CHILD_CGROUP", "auto").strip().lower()
        if mode not in ("auto", "require", "off"):
            mode = "auto"
        return cls(
            as_mb=_env_int("ORCHESTRATOR_REPL_CHILD_AS_MB", cls.as_mb),
            cpu_s=_env_int("ORCHESTRATOR_REPL_CHILD_CPU_S", cls.cpu_s),
            max_respawns=_env_int("ORCHESTRATOR_REPL_CHILD_MAX_RESPAWNS", cls.max_respawns),
            max_calls_per_cell=_env_int("ORCHESTRATOR_REPL_CHILD_MAX_CALLS", cls.max_calls_per_cell),
            cgroup_mode=mode,
            cgroup_memory_mb=_env_int("ORCHESTRATOR_REPL_CHILD_MEM_MB", 0),
            cgroup_pids_max=_env_int("ORCHESTRATOR_REPL_CHILD_PIDS_MAX", cls.cgroup_pids_max),
        )


# ── framing ──────────────────────────────────────────────────────────────────


class _Reader:
    """Non-blocking length-prefixed frame reader. The payload length is checked against
    ``max_frame`` as soon as the 4-byte header is complete and no more than the declared
    payload is ever read, so the buffer is bounded before accumulation."""

    def __init__(self, fd: int, max_frame: int):
        self.fd = fd
        self.max_frame = max_frame
        self.buf = bytearray()
        os.set_blocking(fd, False)

    def poll(self, timeout: float) -> dict | None:
        end = time.monotonic() + max(0.0, timeout)
        while True:
            frame = self._parse()
            if frame is not None:
                return frame
            want = 4 - len(self.buf)
            if want <= 0:
                (n,) = struct.unpack(">I", bytes(self.buf[:4]))
                if n > self.max_frame:
                    raise LaneCrash(f"child frame of {n} bytes exceeds the {self.max_frame} byte cap")
                want = 4 + n - len(self.buf)
            wait = end - time.monotonic()
            r, _, _ = select.select([self.fd], [], [], max(0.0, wait))
            if not r:
                return None
            try:
                chunk = os.read(self.fd, min(1 << 20, want))
            except BlockingIOError:
                continue
            except OSError as exc:
                raise LaneCrash(f"read from child failed: {exc}") from None
            if not chunk:
                raise LaneCrash("child closed its pipe")
            self.buf += chunk

    def _parse(self) -> dict | None:
        if len(self.buf) < 4:
            return None
        (n,) = struct.unpack(">I", bytes(self.buf[:4]))
        if n > self.max_frame:
            raise LaneCrash(f"child frame of {n} bytes exceeds the {self.max_frame} byte cap")
        if len(self.buf) < 4 + n:
            return None
        payload = bytes(self.buf[4:4 + n])
        del self.buf[:4 + n]
        try:
            obj = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise LaneCrash(f"child sent an undecodable frame: {exc}") from None
        if not isinstance(obj, dict):
            raise LaneCrash("child sent a non-object frame")
        return obj


# ── cgroup v2 ────────────────────────────────────────────────────────────────


def _own_cgroup_dir(root: str) -> Path | None:
    try:
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            if line.startswith("0::"):
                rel = line[3:].strip().lstrip("/")
                return Path(root) / rel if rel else Path(root)
    except OSError:
        return None
    return None


class _Cgroup:
    """An owned leaf cgroup under a writable delegated cgroup v2 directory."""

    def __init__(self, path: Path, controllers: set[str]):
        self.path = path
        self.controllers = controllers
        self.memory_applied = False
        self.pids_applied = False

    @classmethod
    def create(cls, cfg: LaneConfig) -> "_Cgroup | None":
        """A new cgroup, or None when unavailable (``auto``). Raises LaneUnavailable in ``require``
        mode when the memory limit cannot be applied. Never edits a parent's subtree_control."""
        if cfg.cgroup_mode == "off":
            return None
        root = os.environ.get("ORCHESTRATOR_REPL_CHILD_CGROUP_ROOT", "/sys/fs/cgroup")
        mode = cfg.cgroup_mode
        parent = Path(os.environ["ORCHESTRATOR_REPL_CHILD_CGROUP_PARENT"]) if os.environ.get(
            "ORCHESTRATOR_REPL_CHILD_CGROUP_PARENT") else _own_cgroup_dir(root)
        why = ""
        if parent is None or not parent.is_dir():
            why = "no cgroup v2 directory found"
        elif not os.access(parent, os.W_OK | os.X_OK):
            why = f"{parent} is not writable"
        if why:
            if mode == "require":
                raise LaneUnavailable(f"cgroup v2 limit required but unavailable: {why}")
            logger.info("repl child lane: no cgroup (%s); RLIMIT_AS only", why)
            return None
        assert parent is not None
        try:
            ctl = set((parent / "cgroup.subtree_control").read_text().split())
        except OSError:
            ctl = set()
        path = parent / f"epyc-repl-{os.getpid()}-{secrets.token_hex(4)}"
        try:
            path.mkdir(mode=0o700)
        except OSError as exc:
            if mode == "require":
                raise LaneUnavailable(f"cgroup v2 limit required but mkdir {path} failed: {exc}") from None
            logger.info("repl child lane: cgroup mkdir failed (%s); RLIMIT_AS only", exc)
            return None
        cg = cls(path, ctl)
        mem_mb = cfg.cgroup_memory_mb or (cfg.as_mb + 512)
        try:
            if "memory" in ctl and (path / "memory.max").exists():
                (path / "memory.max").write_text(str(mem_mb * 1024 * 1024))
                for knob, val in (("memory.swap.max", "0"), ("memory.oom.group", "1")):
                    if (path / knob).exists():
                        (path / knob).write_text(val)
                cg.memory_applied = True
            if "pids" in ctl and (path / "pids.max").exists() and cfg.cgroup_pids_max > 0:
                (path / "pids.max").write_text(str(cfg.cgroup_pids_max))
                cg.pids_applied = True
        except OSError as exc:
            cg.destroy()
            if mode == "require":
                raise LaneUnavailable(f"cgroup limit write failed: {exc}") from None
            logger.info("repl child lane: cgroup limit write failed (%s); RLIMIT_AS only", exc)
            return None
        if mode == "require" and not cg.memory_applied:
            cg.destroy()
            raise LaneUnavailable(
                f"cgroup v2 memory limit required but the memory controller is not delegated in {parent}"
            )
        return cg

    def attach(self, pid: int) -> None:
        (self.path / "cgroup.procs").write_text(str(pid))

    def members(self) -> list[int]:
        try:
            return [int(x) for x in (self.path / "cgroup.procs").read_text().split()]
        except (OSError, ValueError):
            return []

    def oom_kills(self) -> int:
        try:
            for line in (self.path / "memory.events").read_text().splitlines():
                if line.startswith("oom_kill "):
                    return int(line.split()[1])
        except (OSError, ValueError):
            pass
        return 0

    def kill_all(self) -> None:
        try:
            (self.path / "cgroup.kill").write_text("1")
        except OSError:
            for pid in self.members():
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass

    def destroy(self) -> bool:
        """rmdir the (empty) cgroup; True when it is gone."""
        for _ in range(40):
            try:
                self.path.rmdir()
                return True
            except FileNotFoundError:
                return True
            except OSError as exc:
                if exc.errno not in (errno.EBUSY, errno.ENOTEMPTY):
                    return False
                time.sleep(0.05)
        return False


def _session_members(pgid: int) -> list[int]:
    """Live pids whose process group or session is ``pgid`` (/proc scan; our own child's tree only)."""
    out: list[int] = []
    try:
        names = os.listdir("/proc")
    except OSError:
        return out
    for name in names:
        if not name.isdigit():
            continue
        try:
            stat = Path(f"/proc/{name}/stat").read_text()
        except OSError:
            continue
        rest = stat[stat.rfind(")") + 2:].split()
        try:
            state, _ppid, pgrp, sid = rest[0], rest[1], int(rest[2]), int(rest[3])
        except (IndexError, ValueError):
            continue
        if state != "Z" and (pgrp == pgid or sid == pgid):
            out.append(int(name))
    return out


# ── the child process ────────────────────────────────────────────────────────

_live: "weakref.WeakSet[_ChildProc]" = weakref.WeakSet()


def _close_all() -> None:
    for proc in list(_live):
        try:
            proc.terminate()
        except Exception:  # noqa: BLE001
            pass


atexit.register(_close_all)


class _ChildProc:
    """The spawned child and everything that must be torn down with it. Holds no env reference."""

    def __init__(self, cfg: LaneConfig):
        self.cfg = cfg
        self.popen: subprocess.Popen | None = None
        self.pid = 0
        self.cgroup: _Cgroup | None = None
        self.scratch: str | None = None
        self.err_file: Any = None
        self.reader: _Reader | None = None
        self.limits: dict[str, Any] = {}
        self.report: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._done = False
        _live.add(self)

    def __del__(self) -> None:  # last resort: never leave an orphan if the lane is dropped
        try:
            self.terminate()
        except Exception:  # noqa: BLE001
            pass

    # spawn ---------------------------------------------------------------
    def spawn(self, task_id: str | None) -> None:
        from src.repl_environment import exec_sandbox
        from src.repl_environment.exec_containment import child_env

        cfg = self.cfg
        if cfg.as_mb <= 0:
            raise LaneUnavailable("RLIMIT_AS cannot be disabled for the contained lane (ORCHESTRATOR_REPL_CHILD_AS_MB<=0)")
        if not CHILD_SCRIPT.is_file():
            raise LaneUnavailable(f"child script missing: {CHILD_SCRIPT}")
        self.scratch = tempfile.mkdtemp(prefix="epyc-repl-lane-")
        os.chmod(self.scratch, 0o700)
        try:
            argv = [sys.executable, "-I", "-S", str(CHILD_SCRIPT)]
            try:
                wrapped, extra = exec_sandbox.wrap(argv, policy=exec_sandbox.Policy(), extra_allow=[self.scratch])
            except exec_sandbox.ContainmentUnavailable as exc:
                raise LaneUnavailable(exec_sandbox.describe_refusal(exc)) from None
            env = child_env(extra=extra, task_id=task_id)
            env["EPYC_EXEC_RLIMIT_AS_MB"] = str(cfg.as_mb)
            limits = {
                "as_bytes": cfg.as_mb * 1024 * 1024,
                "cpu": cfg.cpu_s,
                "nofile": cfg.nofile,
                "fsize": cfg.fsize_mb * 1024 * 1024,
                "core": 0,
            }
            if cfg.cpu_s <= 0:
                limits.pop("cpu")
            env["EPYC_REPL_CHILD_LIMITS"] = json.dumps(limits)
            self.limits = limits
            self.err_file = open(os.path.join(self.scratch, "stderr.log"), "w+b")
            self.cgroup = _Cgroup.create(cfg)
            self.popen = subprocess.Popen(
                wrapped, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.err_file,
                cwd=self.scratch, env=env, close_fds=True, start_new_session=True,
            )
            self.pid = self.popen.pid
            if self.cgroup is not None:
                try:
                    self.cgroup.attach(self.pid)
                except OSError as exc:
                    if cfg.cgroup_mode == "require":
                        raise LaneUnavailable(f"could not move the child into its cgroup: {exc}") from None
                    logger.info("repl child lane: cgroup attach failed (%s); continuing with RLIMIT_AS only", exc)
                    self.cgroup.kill_all()
                    self.cgroup.destroy()
                    self.cgroup = None
            assert self.popen.stdout is not None
            self.reader = _Reader(self.popen.stdout.fileno(), cfg.max_frame)
        except BaseException:
            self.terminate()
            raise

    # io ------------------------------------------------------------------
    def send(self, obj: dict, deadline: float) -> None:
        assert self.popen is not None and self.popen.stdin is not None
        data = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if len(data) > self.cfg.max_inbound_frame:
            raise LaneUnavailable(f"message of {len(data)} bytes exceeds the lane's {self.cfg.max_inbound_frame} byte cap")
        view = memoryview(struct.pack(">I", len(data)) + data)
        fd = self.popen.stdin.fileno()
        os.set_blocking(fd, False)
        while view:
            if self.popen.poll() is not None:
                raise LaneCrash("child exited while receiving a message")
            left = deadline - time.monotonic()
            if left <= 0:
                raise LaneDeadline("deadline while sending to the child")
            _, w, _ = select.select([], [fd], [], min(left, 0.1))
            if not w:
                continue
            try:
                n = os.write(fd, view[: 1 << 20])
            except BlockingIOError:
                continue
            except (BrokenPipeError, OSError) as exc:
                raise LaneCrash(f"write to child failed: {exc}") from None
            view = view[n:]

    # teardown ------------------------------------------------------------
    def terminate(self) -> dict[str, Any]:
        """Kill and reap ONLY this child (captured pid / its own session / its own cgroup),
        verify nothing it owns survives, remove scratch + cgroup. Idempotent."""
        with self._lock:
            if self._done:
                return self.report
            self._done = True
        rep: dict[str, Any] = {"pid": self.pid, "reaped": True, "survivors": [], "cgroup_removed": None}
        if self.cgroup is not None:
            self.cgroup.kill_all()
        if self.popen is not None and self.pid:
            # The group id is the captured child pid. Once the child is reaped its pid may be
            # recycled, so only signal the group while it is unreaped or still has members (the
            # kernel never recycles a pid that is in use as a pgid/session id).
            if self.popen.returncode is None or _session_members(self.pid):
                try:
                    os.killpg(self.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
            try:
                self.popen.wait(timeout=5)
            except subprocess.TimeoutExpired:
                rep["reaped"] = False
            for stream in (self.popen.stdin, self.popen.stdout):
                try:
                    if stream:
                        stream.close()
                except OSError:
                    pass
            deadline = time.monotonic() + 3.0
            survivors = _session_members(self.pid)
            while survivors and time.monotonic() < deadline:
                for pid in survivors:
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except OSError:
                        pass
                time.sleep(0.05)
                survivors = _session_members(self.pid)
            rep["survivors"] = survivors
            rep["returncode"] = self.popen.returncode
        if self.cgroup is not None:
            left = self.cgroup.members()
            if left:
                rep["survivors"] = sorted(set(rep["survivors"]) | set(left))
            rep["oom_kills"] = self.cgroup.oom_kills()
            rep["cgroup_removed"] = self.cgroup.destroy()
        if self.err_file is not None:
            try:
                self.err_file.close()
            except OSError:
                pass
        if self.scratch:
            shutil.rmtree(self.scratch, ignore_errors=True)
        self.report = rep
        if rep["survivors"] or not rep["reaped"]:
            logger.error("repl child lane: teardown incomplete: %s", rep)
        return rep

    def stderr_tail(self, limit: int = 600) -> str:
        path = os.path.join(self.scratch or "", "stderr.log")
        try:
            with open(path, "rb") as fh:
                fh.seek(0, os.SEEK_END)
                fh.seek(max(0, fh.tell() - limit))
                return fh.read().decode("utf-8", "replace").strip()
        except OSError:
            return ""

    def exit_description(self) -> str:
        rc = self.popen.poll() if self.popen else None
        if rc is None:
            return "alive"
        if rc < 0:
            try:
                name = signal.Signals(-rc).name
            except ValueError:
                name = str(-rc)
            desc = f"killed by {name}"
            if self.cgroup is not None and self.cgroup.oom_kills():
                desc += " (cgroup OOM kill)"
            return desc
        return f"exit status {rc}"


# ── the lane ─────────────────────────────────────────────────────────────────


def _codec():
    from src.repl_environment import _repl_child_main as m

    return m


def _canon(enc: Any) -> str:
    return json.dumps(enc, sort_keys=True, ensure_ascii=True, allow_nan=True)


def _make_ctx(env: Any, allow_opaque: bool, obj_fields: bool = False):
    """Parent-side codec context. Carried definitions are only ever STORED (as ``LaneOpaque``),
    never executed here, and their source must pass the REPL's own AST security gate first.
    An ``obj`` record (a tool-result object the child holds) is inert data: it is stored as
    ``LaneOpaque`` anywhere, or, with ``obj_fields`` (tool ARGUMENTS), flattened to its field dict."""
    m = _codec()

    class ParentCtx(m._Ctx):
        safe_modules = frozenset(env.SAFE_IMPORT_MODULES)

        def opaque_encode(self, v):
            raise m.Unsupported(m._why(v))

        def opaque_decode(self, tag, enc):
            if tag == "obj":
                if obj_fields:
                    fields = enc.get("v")
                    if type(fields) is not dict:
                        raise m.Unsupported("malformed obj record")
                    return m.decode(fields, self, 1)
                return m.LaneOpaque(enc)
            if not allow_opaque:
                raise m.Unsupported("a %s record cannot appear here" % tag)
            if tag == "def":
                from src.repl_environment.types import REPLSecurityError

                src = enc.get("src")
                if type(src) is not str or not isinstance(enc.get("name"), str) or len(src) > m.MAX_DEF_SOURCE:
                    raise m.Unsupported("malformed definition record")
                try:
                    env._validate_code(src)
                except REPLSecurityError as exc:
                    raise m.Unsupported(f"carried definition {enc['name']!r} failed the AST security gate: {exc}") from None
            return m.LaneOpaque(enc)

    return ParentCtx()


_UNTRUSTED_DETAIL_CAP = 300


def _clean_child_text(text: str) -> str:
    """Child-controlled text (stderr tail) that reaches an error message: defanged, capped, one line."""
    from src.security.untrusted_text import sanitize_tool_text

    return " | ".join(sanitize_tool_text(text, cap=_UNTRUSTED_DETAIL_CAP).splitlines())


class ChildLane:
    """Parent side of the lane for one ``REPLEnvironment``."""

    def __init__(self, env: Any, cfg: LaneConfig | None = None,
                 lease_validator: Callable[[], bool] | None = None):
        self._env_ref = weakref.ref(env)
        self.cfg = cfg or LaneConfig.from_env()
        self.lease_validator = lease_validator
        self._proc: _ChildProc | None = None
        self._cancel = threading.Event()
        self.respawns = 0
        self.spawns = 0
        self._intentional_restart = False
        self._state_objs: tuple[Any, Any] | None = None
        self._art_view: dict[str, str] = {}   # what the child holds: key -> canonical JSON of its encoding
        self._art_start: dict[str, str] = {}  # the parent's encodable artifacts when the current cell began
        self.admission_open = False
        self._nonce: str | None = None
        self._structured = False
        self._tool_fns: dict[str, Callable[..., Any]] = {}
        self._expect_seq = 0
        self._calls = 0
        self.last_teardown: dict[str, Any] = {}
        self._finalizer = weakref.finalize(env, _finalize_lane, weakref.ref(self))

    # public ---------------------------------------------------------------
    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc else None

    @property
    def limits(self) -> dict[str, Any]:
        p = self._proc
        if not p:
            return {}
        out = dict(p.limits)
        out["child_reported"] = p.report.get("child_limits") if p.report else None
        out["cgroup"] = ({"path": str(p.cgroup.path), "memory": p.cgroup.memory_applied,
                          "pids": p.cgroup.pids_applied} if p.cgroup else None)
        return out

    def cancel(self) -> None:
        """Thread-safe: the cell in flight is killed at the next poll (<= ~100 ms)."""
        self._cancel.set()

    def close(self) -> dict[str, Any]:
        self.admission_open = False
        proc, self._proc = self._proc, None
        if proc is None:
            return self.last_teardown
        self.last_teardown = proc.terminate()
        return self.last_teardown

    def run_cell(self, code: str, deadline: float) -> dict:
        """Run one cell in the (warm) child; returns the child's ``done`` frame.
        Raises LaneUnavailable / LaneCrash / LaneDeadline / LaneRevoked; the child is already
        reaped when the latter three propagate."""
        env = self._env_ref()
        if env is None:
            raise LaneCrash("the REPL environment is gone")
        self._cancel.clear()
        held = self._state_objs
        if self._proc is not None and held is not None and (held[0] is not env._globals or held[1] is not env.artifacts):
            self.discard_child()  # reset()/checkpoint restore rebuilt the parent's state: re-seed the child
        proc = self._ensure_started(env)
        nonce = secrets.token_hex(12)
        self._nonce, self._expect_seq, self._calls = nonce, 0, 0
        self.admission_open = True
        try:
            assert proc.reader is not None
            cell = {"t": "cell", "id": nonce, "code": code}
            patch = self._artifact_patch(env)
            if patch:
                cell["art_patch"] = patch
            proc.send(cell, deadline)
            while True:
                if self._cancel.is_set():
                    raise LaneRevoked("cancelled")
                left = deadline - time.monotonic()
                if left <= 0:
                    raise LaneDeadline(f"execution timed out after {env.config.timeout_seconds}s")
                if self.lease_validator is not None and not self._lease_ok():
                    raise LaneRevoked("lease lost")
                try:
                    frame = proc.reader.poll(min(left, 0.1))
                except LaneCrash:
                    raise
                if frame is None:
                    if proc.popen is not None and proc.popen.poll() is not None:
                        raise LaneCrash(f"child {proc.exit_description()}")
                    continue
                if frame.get("id") != nonce:
                    raise LaneCrash("stale or forged frame (request identity mismatch)")
                kind = frame.get("t")
                if kind == "call":
                    proc.send(self._serve_call(env, frame, deadline), deadline)
                elif kind == "done":
                    return frame
                else:
                    raise LaneCrash(f"unexpected frame type {kind!r}")
        except LaneUnavailable:
            self._reap("unavailable")
            raise
        except LaneCrash as exc:
            detail = _clean_child_text(proc.stderr_tail())
            if proc.popen is not None:
                try:
                    proc.popen.wait(timeout=0.5)  # a closed pipe can beat the exit status by a moment
                except subprocess.TimeoutExpired:
                    pass
            desc = proc.exit_description() if proc.popen and proc.popen.poll() is not None else "protocol error"
            self._reap("crash")
            raise LaneCrash(f"{exc} [{desc}]" + (f" stderr: {detail}" if detail else "")) from None
        except (LaneDeadline, LaneRevoked):
            self._reap("deadline/revoked")
            raise
        finally:
            self.admission_open = False
            self._nonce = None

    # internals ------------------------------------------------------------
    def _lease_ok(self) -> bool:
        try:
            return bool(self.lease_validator())  # type: ignore[misc]
        except Exception:  # noqa: BLE001 - an unreadable lease is a lost lease
            return False

    def _reap(self, why: str) -> None:
        self.admission_open = False
        proc, self._proc = self._proc, None
        if proc is not None:
            self.last_teardown = proc.terminate()
            self.last_teardown["why"] = why

    def discard_child(self) -> None:
        """Kill the warm child so the next cell restarts from the parent's state (state rollback)."""
        self._reap("rollback")
        self._intentional_restart = True

    def _ensure_started(self, env: Any) -> _ChildProc:
        if self._proc is not None and self._proc.popen is not None and self._proc.popen.poll() is None:
            return self._proc
        if self._proc is not None:
            self._reap("dead")
        if self.spawns and not self._intentional_restart and self.respawns >= self.cfg.max_respawns:
            raise LaneUnavailable(f"the contained child crashed {self.respawns} times in this request; refusing to respawn")
        init = self._build_init(env)
        self._state_objs = (env._globals, env.artifacts)
        proc = _ChildProc(self.cfg)
        self._proc = proc
        end = time.monotonic() + self.cfg.spawn_timeout_s
        try:
            proc.spawn(getattr(env, "task_id", None))
            proc.send(init, end)
            assert proc.reader is not None
            ready = None
            while ready is None:
                if time.monotonic() > end:
                    raise LaneCrash("child did not become ready in time")
                ready = proc.reader.poll(0.1)
                if ready is None and proc.popen is not None and proc.popen.poll() is not None:
                    raise LaneCrash(f"child {proc.exit_description()} before ready")
            if ready.get("t") != "ready":
                raise LaneCrash("child did not send ready")
            self._verify_limits(proc, ready.get("limits") or {})
        except LaneUnavailable:
            self._reap("start-refused")
            raise
        except (LaneCrash, LaneDeadline) as exc:
            detail = _clean_child_text(proc.stderr_tail())
            self._reap("start-failed")
            # exit 126 from the shim / child == a control could not be applied: fail closed
            raise LaneUnavailable(f"contained child failed to start: {exc}" + (f" ({detail})" if detail else "")) from None
        except BaseException:
            self._reap("start-error")
            raise
        if self.spawns and not self._intentional_restart:
            self.respawns += 1
        self._intentional_restart = False
        self.spawns += 1
        return proc

    def _verify_limits(self, proc: _ChildProc, got: dict) -> None:
        want = proc.limits
        for key, val in want.items():
            have = got.get(key)
            have = have if not isinstance(have, list) else have[0]
            if have != val:
                raise LaneUnavailable(f"limit {key} not applied: wanted {val}, child reports {have}")
        proc.report = {"child_limits": got}

    def _build_init(self, env: Any) -> dict:
        m = _codec()
        if getattr(env, "_context_bundle", None) is not None:
            raise LaneUnavailable("a context bundle is attached; the contained lane supports string context only")
        ctx = env.context
        if not isinstance(ctx, str):
            raise LaneUnavailable("context is not a string; unsupported in the contained lane")
        if len(ctx) > self.cfg.context_cap_chars:
            raise LaneUnavailable(f"context of {len(ctx)} chars exceeds the lane cap of {self.cfg.context_cap_chars}")
        base = tool_broker.tool_names(env)
        pctx = _make_ctx(env, allow_opaque=True)
        state: dict[str, Any] = {}
        bad: list[str] = []
        for key, val in env._globals.items():
            if key in base or key == "__builtins__":
                continue
            try:
                state[key] = m.encode(val, pctx)
            except m.Unsupported as exc:
                bad.append(f"{key} ({exc})")
        arts = self._encodable_artifacts(env)
        self._art_view = {k: _canon(v) for k, v in arts.items()}
        self._art_start = dict(self._art_view)
        if bad:
            raise LaneUnavailable(
                "unsupported REPL state cannot be restored into the contained lane: " + ", ".join(sorted(bad)[:10])
            )
        self._structured = bool(getattr(env, "_structured_mode", False))
        # Free-form AND structured (React) mode broker the same surface: every callable the in-process
        # path exposes. The parent still owns each call (budget, argument validation, write dedupe,
        # untrusted-text handling, deadline, lease); the child only contains the Python around it.
        # The callables are captured NOW from the parent's own globals, so nothing the child sends
        # can rebind what a name resolves to.
        self._tool_fns = tool_broker.bridged_tools(env)
        allow = sorted(self._tool_fns)
        refuse = tool_broker.refused_names(env, self._tool_fns)
        return {
            "t": "init", "context": ctx, "state": state, "artifacts": arts,
            "builtins": sorted(env.config.allowed_builtins), "safe_modules": sorted(env.SAFE_IMPORT_MODULES),
            "allow": allow, "refuse": refuse, "caps": {"output_chars": self.cfg.output_chars},
        }

    # callbacks ------------------------------------------------------------
    def _serve_call(self, env: Any, frame: dict, deadline: float) -> dict:
        m = _codec()
        nonce = frame["id"]
        seq = frame.get("seq")

        def err(tp: str, msg: str, args: list | None = None) -> dict:
            return {"t": "reply", "id": nonce, "seq": seq, "ok": False, "err": {"type": tp, "msg": msg, "args": args}}

        if not self.admission_open or nonce != self._nonce:
            raise LaneCrash("callback outside an admitted cell")
        self._expect_seq += 1
        if seq != self._expect_seq:
            raise LaneCrash(f"callback sequence {seq!r}, expected {self._expect_seq}")
        self._calls += 1
        if self._calls > self.cfg.max_calls_per_cell:
            return err("RuntimeError", f"callback cap exceeded: more than {self.cfg.max_calls_per_cell} callbacks in one cell")
        if env.timed_out:
            raise LaneRevoked("environment is terminal")
        name = frame.get("name")
        known = isinstance(name, str) and (name in BROKER or name in self._tool_fns)
        if not known:
            return err("NotImplementedError", f"{name!r} is not available in the contained REPL lane")
        try:  # tool ARGUMENTS: a tool-result object the model passes back crosses as its field dict
            args = m.decode(frame.get("args"), _make_ctx(env, allow_opaque=False, obj_fields=True))
            kwargs = m.decode(frame.get("kwargs"), _make_ctx(env, allow_opaque=False, obj_fields=True))
        except m.Unsupported as exc:
            return err("TypeError", f"{name}(): bad argument encoding ({exc})")
        if not isinstance(args, list) or not isinstance(kwargs, dict) or not all(isinstance(k, str) for k in kwargs):
            return err("TypeError", f"{name}(): malformed call")
        if self.lease_validator is not None and not self._lease_ok():
            raise LaneRevoked("lease lost")
        try:
            from src.repl_environment.types import FinalSignal

            if name in self._tool_fns:
                # The in-process callable: its wrapper charges the snippet budget (``_invoke_tool`` charges
                # TOOL/CALL itself), and ``tool_broker.call`` enforces argument validation.
                fn = self._tool_fns[name]
            else:
                fn = getattr(env, BROKER[name])  # FINAL / FINAL_VAR: control signals, not charged
            if name == "FINAL_VAR":
                arts = frame.get("artifacts")
                if not isinstance(arts, dict):
                    return err("ValueError", "FINAL_VAR requires the artifact snapshot")
                self._merge_artifacts(env, arts)
            if isinstance(frame.get("state"), dict):  # SCRIPT & co: the callee reads the live variables
                self._apply_live_state(env, frame["state"])
            thunk = lambda: tool_broker.call(name, fn, args, kwargs)  # noqa: E731
            if name in _CHILD_DEFINED:
                value = thunk()
            else:
                value = self._run_bounded(thunk, deadline)
        except FinalSignal as fs:
            return {"t": "reply", "id": nonce, "seq": seq, "ok": True, "final": fs.answer}
        except (LaneDeadline, LaneRevoked):
            raise
        except Exception as exc:  # noqa: BLE001 - tool errors are data for the child's cell, never fatal
            return {"t": "reply", "id": nonce, "seq": seq, "ok": False, "err": tool_broker.error_record(exc)}
        try:
            enc = m.encode(value, tool_broker.reply_ctx(m))
            blob = json.dumps(enc, ensure_ascii=True, allow_nan=True)
        except (m.Unsupported, TypeError, ValueError) as exc:
            return err("TypeError", f"{name}() returned a value the contained lane cannot carry: {exc}")
        if len(blob) > self.cfg.max_reply_bytes:
            return err("ValueError", f"{name}() result of {len(blob)} chars exceeds the lane reply cap of {self.cfg.max_reply_bytes}")
        return {"t": "reply", "id": nonce, "seq": seq, "ok": True, "value": enc}

    def _apply_live_state(self, env: Any, state: dict) -> None:
        """Mirror the child's live variables into ``env._globals`` BEFORE a tool that reads them runs
        (best effort per key: a value that cannot cross keeps the parent's copy; nothing is deleted)."""
        m = _codec()
        base = tool_broker.tool_names(env)
        pctx = _make_ctx(env, allow_opaque=True)
        for key, val in state.items():
            if key in base or key == "__builtins__" or not isinstance(key, str):
                continue
            try:
                env._globals[key] = m.decode(val, pctx)
            except m.Unsupported:
                continue

    def _run_bounded(self, thunk: Callable[[], Any], deadline: float) -> Any:
        """Run a callback on a worker thread and wait for it at most until ``deadline``. While it runs
        the wait is sliced so a cancel or a lost session lease kills the cell promptly (a long
        ``llm_call`` must not outlive its lease), exactly as between callbacks.

        Thread-safe on any calling thread (unlike SIGALRM, which only exists on the main thread and,
        like an async exception, cannot interrupt a C-level call such as a pathological regex; that
        case is bounded separately by ``safe_regex``). On overrun the worker is sent an async
        exception to unwind Python-level work, the caller raises ``LaneDeadline`` (the child is killed
        and the environment goes terminal, so a late worker result is never used), and the thread is
        abandoned as a daemon. The request's contextvars are propagated to the worker."""
        left = deadline - time.monotonic()
        if left <= 0:
            raise LaneDeadline("deadline reached before the callback ran")
        box: dict[str, Any] = {}
        cv = contextvars.copy_context()

        def target() -> None:
            try:
                box["v"] = cv.run(thunk)
            except _CallbackAbort:
                box["aborted"] = True
            except BaseException as exc:  # noqa: BLE001 - re-raised in the caller's thread
                box["e"] = exc

        worker = threading.Thread(target=target, name="repl-lane-callback", daemon=True)
        worker.start()
        revoked = ""
        while worker.is_alive():
            slice_s = min(0.1, deadline - time.monotonic())
            if slice_s <= 0:
                break
            worker.join(slice_s)
            if not worker.is_alive():
                break
            if self._cancel.is_set():
                revoked = "cancelled"
            elif self.lease_validator is not None and not self._lease_ok():
                revoked = "lease lost"
            if revoked:
                break
        if worker.is_alive():
            if worker.ident is not None:
                ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(worker.ident), ctypes.py_object(_CallbackAbort))
            if revoked:
                raise LaneRevoked(revoked)
            raise LaneDeadline("deadline reached inside a callback")
        if "e" in box:
            raise box["e"]
        return box.get("v")

    @staticmethod
    def _encodable_artifacts(env: Any) -> dict[str, Any]:
        """The parent's artifacts that can cross (key -> encoding). Anything else (a tool's
        ``_tool_outputs`` ledger holding arbitrary objects, ...) is parent-owned: the child never sees
        it and its absence from the child's result is never read as a deletion."""
        m = _codec()
        ctx = _make_ctx(env, allow_opaque=False)
        out: dict[str, Any] = {}
        for k, v in list(env.artifacts.items()):
            if not isinstance(k, str):
                continue
            try:
                out[k] = m.encode(v, ctx)
            except m.Unsupported:
                continue
        return out

    def _artifact_patch(self, env: Any) -> dict[str, Any] | None:
        """What changed in the parent's artifacts since the child last saw them (a tool wrote
        ``_tool_outputs`` in a previous cell, or code outside the lane edited them)."""
        cur = self._encodable_artifacts(env)
        canon = {k: _canon(v) for k, v in cur.items()}
        patch = {
            "set": {k: cur[k] for k in cur if self._art_view.get(k) != canon[k]},
            "del": [k for k in self._art_view if k not in cur],
        }
        self._art_view = canon
        self._art_start = dict(canon)
        return patch if (patch["set"] or patch["del"]) else None

    def _merge_artifacts(self, env: Any, child_arts: dict) -> None:
        """Three-way merge of ``env.artifacts``: the child's edits win for keys it changed; keys a
        brokered tool wrote in the parent during the cell, and parent-only keys, are kept."""
        m = _codec()
        start = self._art_start
        child_canon = {k: _canon(v) for k, v in child_arts.items()}
        decoded = m.decode(child_arts, _make_ctx(env, allow_opaque=False))
        live = env.artifacts
        merged: dict[str, Any] = {}
        for k in list(live):  # parent-only keys and keys the parent changed/created
            if k in child_canon:
                continue
            if k in start:  # shipped, and the child deleted it
                continue
            merged[k] = live[k]
        for k, v in decoded.items():
            if k in start and child_canon[k] == start[k]:
                if k in live:  # the child left it alone: the parent's (possibly tool-updated) value stands
                    merged[k] = live[k]
            else:
                merged[k] = v
        live.clear()
        live.update(merged)
        self._art_view = dict(child_canon)
        self._art_start = dict(child_canon)

    # state ----------------------------------------------------------------
    def sync_state(self, env: Any, done: dict) -> None:
        """Mirror the child's variables and artifacts into the parent (the authoritative copy)."""
        m = _codec()
        base = tool_broker.tool_names(env)
        pctx = _make_ctx(env, allow_opaque=True)
        state = done.get("state") or {}
        decoded = {key: m.decode(val, pctx) for key, val in state.items()}  # may raise Unsupported
        for key in [k for k in env._globals if k not in base and k != "__builtins__"]:
            if key not in decoded:
                del env._globals[key]
        for key, val in decoded.items():
            if key not in base:
                env._globals[key] = val
        arts = done.get("artifacts")
        if isinstance(arts, dict):
            self._merge_artifacts(env, arts)


class _CallbackAbort(BaseException):
    """Injected into an abandoned callback worker to unwind Python-level work."""


def _finalize_lane(lane_ref: "weakref.ref[ChildLane]") -> None:
    lane = lane_ref()
    if lane is not None:
        lane.close()


# ── request scope: close every lane a request created, on every exit path ────

_scope_var: "contextvars.ContextVar[list | None]" = contextvars.ContextVar("repl_lane_scope", default=None)


class lane_scope:
    """Context manager (sync or async) that owns the lanes created inside it and reaps them on exit.

    Nest-safe: an inner scope collects its own lanes. Exiting from a different ``Context`` than the
    one that entered (an async generator finalised by the GC) closes the lanes all the same and only
    skips the ``ContextVar`` reset."""

    def __init__(self) -> None:
        self.lanes: list = []
        self._token: Any = None

    def __enter__(self) -> "lane_scope":
        self._token = _scope_var.set(self.lanes)
        return self

    def __exit__(self, *exc: Any) -> None:
        try:
            for ref in self.lanes:
                lane = ref()
                if lane is not None:
                    try:
                        lane.close()
                    except Exception:  # noqa: BLE001
                        logger.warning("repl child lane close failed", exc_info=True)
        finally:
            self.lanes.clear()
            try:
                _scope_var.reset(self._token)
            except ValueError:
                pass

    async def __aenter__(self) -> "lane_scope":
        return self.__enter__()

    async def __aexit__(self, *exc: Any) -> None:
        self.__exit__(*exc)


def closes_lanes(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator for request entry points (coroutine, async generator or plain function): every
    contained child lane created while the call runs is killed and reaped when it ends, however it
    ends. With the lane flag off it calls ``fn`` untouched."""
    import functools
    import inspect

    if inspect.isasyncgenfunction(fn):

        @functools.wraps(fn)
        def agen_wrapper(*args: Any, **kwargs: Any):
            if not enabled():
                return fn(*args, **kwargs)

            async def scoped():
                with lane_scope():
                    gen = fn(*args, **kwargs)
                    try:
                        async for item in gen:
                            yield item
                    finally:
                        await gen.aclose()

            return scoped()

        return agen_wrapper
    if inspect.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def co_wrapper(*args: Any, **kwargs: Any):
            if not enabled():
                return await fn(*args, **kwargs)
            with lane_scope():
                return await fn(*args, **kwargs)

        return co_wrapper

    @functools.wraps(fn)
    def sync_wrapper(*args: Any, **kwargs: Any):
        if not enabled():
            return fn(*args, **kwargs)
        with lane_scope():
            return fn(*args, **kwargs)

    return sync_wrapper


# ── REPLEnvironment integration ──────────────────────────────────────────────


def get_lane(env: Any) -> ChildLane:
    lane = getattr(env, "_child_lane", None)
    if lane is None:
        lane = ChildLane(env, lease_validator=getattr(env, "_lane_lease_validator", None))
        env._child_lane = lane
        scope = _scope_var.get()
        if scope is not None:
            scope.append(weakref.ref(lane))
    return lane


def close_lane(env: Any) -> dict[str, Any]:
    lane = getattr(env, "_child_lane", None)
    return lane.close() if lane is not None else {}


def _lane_error(env: Any, start: float, msg: str):
    from src.repl_environment.types import ExecutionResult

    return ExecutionResult(output="", is_final=False, error=msg, elapsed_seconds=time.perf_counter() - start)


def _run(env: Any, code: str, start_time: float):
    """(done frame, None) or (None, ExecutionResult) when the cell could not complete."""
    lane = get_lane(env)
    deadline = time.monotonic() + env.config.timeout_seconds
    try:
        return lane.run_cell(code, deadline), None
    except LaneDeadline:
        env.mark_timed_out()
        return None, env._timeout_result(start_time)
    except LaneUnavailable as exc:
        return None, _lane_error(env, start_time, f"[ERROR: CONTAINED REPL LANE unavailable: {exc}; the cell was NOT run (no inline fallback)]")
    except LaneRevoked as exc:
        return None, _lane_error(env, start_time, f"[ERROR: REPLLaneRevoked: {exc}; the contained child was killed]")
    except LaneCrash as exc:
        return None, _lane_error(env, start_time, f"[ERROR: REPLChildCrashed: {exc}; the next cell restarts from the last completed state]")


def _rollback(env: Any, lane: ChildLane, stdout: str, start_time: float, why: str):
    from src.repl_environment.types import ExecutionResult

    lane.discard_child()  # the cell's state effects are discarded; the next cell restores from the parent
    return ExecutionResult(
        output=env._bundle_cap_output(stdout), is_final=False,
        error=env._error_text_for_result(
            "REPLLaneStateError: the contained lane carries JSON-compatible values, tuples/sets/Counter-style "
            "containers, stdlib references and top-level def/class/assignment definitions; this cell's state "
            f"changes were discarded because of: {_clean_child_text(why)}"
        ),
        elapsed_seconds=time.perf_counter() - start_time,
    )


def _settle(env: Any, done: dict, start_time: float):
    """Apply the child's state to the parent. Returns an ExecutionResult when the cell must be
    rolled back (unsupported state), else None."""
    m = _codec()
    lane = env._child_lane
    violations = done.get("violations") or []
    status = done.get("status")
    if violations and status != "final":
        why = ", ".join(f"{n}: {t}" for n, t in violations[:10])
        return _rollback(env, lane, done.get("stdout") or "", start_time, why)
    if not violations:
        try:
            lane.sync_state(env, done)
        except m.Unsupported as exc:
            if status != "final":
                return _rollback(env, lane, done.get("stdout") or "", start_time, str(exc))
    return None


def _origin(done: dict) -> str:
    tool = done.get("etool")
    return f"tool:{tool}" if isinstance(tool, str) and tool.isidentifier() else "model_code"


def execute_in_lane(env: Any, code: str, start_time: float):
    """Run one already-validated cell in the contained child and shape the result exactly like
    the in-process path (same ExecutionResult, same output post-processing)."""
    from src.repl_environment import exception_schema
    from src.repl_environment.redaction import redact_if_enabled
    from src.repl_environment.types import ExecutionResult

    done, early = _run(env, code, start_time)
    if early is not None:
        return early
    rolled = _settle(env, done, start_time)
    if rolled is not None:
        return rolled
    status = done.get("status")
    stdout, stderr = done.get("stdout") or "", done.get("stderr") or ""
    elapsed = time.perf_counter() - start_time

    if status == "final":
        return ExecutionResult(output=env._bundle_cap_output(stdout), is_final=True,
                               final_answer=done.get("final"), elapsed_seconds=elapsed)
    if status == "syntax":
        hint = env._tool_hint_if_relevant(code, None)
        return ExecutionResult(output="", is_final=False, error=f"SyntaxError: {done.get('emsg')}{hint}", elapsed_seconds=elapsed)
    if status == "error":
        hint = env._tool_hint_if_relevant(code, None)
        text = exception_schema.format_exception(done.get("etype") or "Exception", done.get("emsg") or "", _origin(done), hint)
        return ExecutionResult(
            output=env._bundle_cap_output(stdout), is_final=False,
            error=env._error_text_for_result(text), elapsed_seconds=elapsed,
        )
    output = stdout
    if stderr:
        output += "\n[STDERR]\n" + stderr
    output = redact_if_enabled(output)
    output = env._bundle_cap_output(output)
    output = env._spill_output(output)
    return ExecutionResult(output=output, is_final=False, elapsed_seconds=elapsed)


def execute_structured_in_lane(env: Any, code: str, start_time: float, tool_calls: list[str]):
    """Structured (React) mode fall-through: the single-tool / simple-expression cell that the
    in-process path ``exec``s. Same result shaping as ``_execute_structured``'s own exec branch."""
    from src.repl_environment import exception_schema
    from src.repl_environment.redaction import redact_if_enabled
    from src.repl_environment.types import ExecutionResult

    try:
        done, early = _run(env, code, start_time)
        if early is not None:
            return early
        rolled = _settle(env, done, start_time)
        if rolled is not None:
            return rolled
        status = done.get("status")
        stdout, stderr = done.get("stdout") or "", done.get("stderr") or ""
        elapsed = time.perf_counter() - start_time
        if status == "final":
            return ExecutionResult(output=stdout, is_final=True, final_answer=done.get("final"), elapsed_seconds=elapsed)
        if status == "syntax":
            hint = env._tool_hint_if_relevant(code, None)
            return ExecutionResult(output="", is_final=False, error=f"SyntaxError: {done.get('emsg')}{hint}", elapsed_seconds=elapsed)
        if status == "error":
            hint = env._tool_hint_if_relevant(code, None)
            text = exception_schema.format_exception(done.get("etype") or "Exception", done.get("emsg") or "", _origin(done), hint)
            return ExecutionResult(output=env._bundle_cap_output(stdout), is_final=False, error=text, elapsed_seconds=elapsed)
        output = stdout
        if stderr:
            output += "\n[STDERR]\n" + stderr
        output = redact_if_enabled(output)
        observation = f"Observation: {output.strip()}" if output else "Observation: [No output]"
        observation = env._spill_output(observation)
        tool_name = tool_calls[0] if tool_calls else ""
        observation = env._maybe_append_suggestions(tool_name, observation)
        return ExecutionResult(output=observation, is_final=False, elapsed_seconds=elapsed)
    finally:
        env._finalize_active_chain_meta()
