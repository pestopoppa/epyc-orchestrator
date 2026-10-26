#!/usr/bin/python3
"""CPU admission and nontruncating writer flock before any GitNexus work."""
import fcntl
import ctypes
import importlib.util
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import time

HERE = Path(__file__).absolute().parent
QUARTER_SIGNALS = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)


class Termination(Exception):
    def __init__(self, signum):
        self.signum = signum
        super().__init__(f"received signal {signum}")


def terminate_quarter(signum, _frame):
    raise Termination(signum)


def parent_death_guard(expected_parent, previous_mask):
    # Called only in the forked child of this single-threaded supervisor.
    # KILL survives Node signal-handler changes and stops all its worker threads.
    libc = ctypes.CDLL(None, use_errno=True)
    prctl = libc.prctl
    prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
    prctl.restype = ctypes.c_int
    if prctl(1, signal.SIGKILL, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
        os._exit(64)
    # The parent might have died between fork and prctl. Do not exec in that case.
    if os.getppid() != expected_parent:
        os._exit(64)
    signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)


def cpu_guard():
    spec = importlib.util.spec_from_file_location("epyc_gitnexus_guard", HERE / "gitnexus-guard.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def quarter_arguments(args, cpus):
    result = []
    i = 0
    while i < len(args):
        item = args[i]
        if item == "--" or item.startswith(("--embedding", "--skills")):
            raise Refusal("unreviewed quarter analyze arguments")
        if item == "--workers" or item.startswith("--workers="):
            if item == "--workers":
                i += 1
                if i == len(args):
                    raise Refusal("missing workers value")
                value = args[i]
            else:
                value = item.split("=", 1)[1]
            if not value.isascii() or not value.isdigit() or not 1 <= int(value) <= len(cpus):
                raise Refusal("workers exceeds the granted quarter")
        else:
            result.append(item)
        i += 1
    # Explicit CLI values win over GNX's environment knob, so own both channels.
    return [*result, "--workers", "4"]


def quarter_run(repo, args, fd, admission, current):
    if repo != "app":
        raise Refusal("quarter route is APP-only; ROOT retains whole-host admission")
    admission.installed_source()
    if frozenset(os.sched_getaffinity(0)) != current["cpus"]:
        raise Refusal("writer is not pinned exactly to the granted CPUs")
    args = quarter_arguments(args, current["cpus"])
    env = os.environ.copy()
    for key in tuple(env):
        if key.startswith("GITNEXUS_TEST_") or key in {"NODE_PATH", "NODE_REPL_EXTERNAL_MODULE", "LD_PRELOAD"}:
            raise Refusal("unreviewed Node startup environment")
    env.update({
        "UV_THREADPOOL_SIZE": "1", "GITNEXUS_WORKER_POOL_SIZE": "4",
        "GITNEXUS_PARSE_CHUNK_CONCURRENCY": "1", "GITNEXUS_EMBEDDING_THREADS": "1",
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "NODE_OPTIONS": "--max-old-space-size=8192", "GITNEXUS_LBUG_EXTENSION_INSTALL": "never",
        "LD_PRELOAD": admission.native_artifact(),
        "GITNEXUS_LBUG_BUFFER_BYTES": str(admission.DB_BUFFER_BYTES),
        "GITNEXUS_LBUG_MAX_DB_SIZE": str(admission.DB_MAX_BYTES),
    })
    command = [admission.NODE, "--v8-pool-size=1", str(HERE / "gitnexus-quarter.cjs"),
               "app", str(admission.GNX / "dist/cli/index.js"), "analyze",
               "--skip-agents-md", "--skip-skills", *args]
    if list(Path("/mnt/raid0/llm/tmp/compute-flags").glob("HOLD_*")):
        raise Refusal("HOLD prevents quarter Node startup")
    child = None
    previous_handlers = {sig: signal.getsignal(sig) for sig in QUARTER_SIGNALS}
    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, QUARTER_SIGNALS)
    try:
        for sig in QUARTER_SIGNALS:
            signal.signal(sig, terminate_quarter)
        parent_pid = os.getpid()
        child = subprocess.Popen(command, env=env, pass_fds=(fd,),
                                 preexec_fn=lambda: parent_death_guard(parent_pid, previous_mask))
        # Deliver pending TERM/INT/HUP only after the captured child is assigned
        # inside this try/finally. The child separately restored its own mask.
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
        while child.poll() is None:
            if list(Path("/mnt/raid0/llm/tmp/compute-flags").glob("HOLD_*")):
                raise Refusal("HOLD appeared during quarter analyze")
            if admission.claim() != current:
                raise Refusal("quarter claim changed during analyze")
            try:
                admission.node_admission(current, child.pid)
            except (FileNotFoundError, ProcessLookupError):
                # The captured child may have exited between poll and /proc.
                if child.poll() is None:
                    raise
            time.sleep(0.25)
        return child.wait()
    finally:
        signal.pthread_sigmask(signal.SIG_BLOCK, QUARTER_SIGNALS)
        # Only this Popen's captured Node PID is ever signalled or reaped.
        try:
            if child is not None and child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=5)
            if child is not None and child.poll() is None:
                raise Refusal("captured Node child did not exit")
        finally:
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)


def guard(repo):
    result = subprocess.run(['/usr/bin/python3', '-I', str(HERE / 'gitnexus-guard.py'), '--repo', repo])
    if result.returncode:
        raise Refusal('CPU admission refused')


class Refusal(Exception):
    pass


def writer_lock(repo):
    # /tmp itself must be a nonsymlink directory, with descriptor-relative opens.
    root = os.open('/', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        directory = os.open('tmp', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=root)
    finally:
        os.close(root)
    fd = None
    try:
        if repo == 'root':
            name = 'gitnexus-epyc-root-analyze.lock'
        else:
            # Preserve APP's existing repository-basename lock identity.
            top = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], text=True).strip()
            name = f'gitnexus-{Path(top).name}-analyze.lock'
        fd = os.open(name, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, 0o600, dir_fd=directory)
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode) or opened.st_uid != os.getuid() or opened.st_nlink != 1:
            raise Refusal('writer lock is not an owned single-link regular file')
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        current = os.stat(name, dir_fd=directory, follow_symlinks=False)
        check = os.stat('/tmp', follow_symlinks=False)
        original = os.fstat(directory)
        if ((current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino) or
            (check.st_dev, check.st_ino) != (original.st_dev, original.st_ino)):
            raise Refusal('writer pathname changed')
        result, fd = fd, None
        return result
    finally:
        if fd is not None:
            os.close(fd)
        os.close(directory)


def main():
    fd = None
    try:
        if len(sys.argv) < 4 or sys.argv[1] != '--repo' or sys.argv[2] not in {'root', 'app'} or sys.argv[3] != '--':
            raise Refusal('usage: gitnexus-writer.py --repo root|app -- [analyze arguments]')
        repo = sys.argv[2]
        admission = cpu_guard()
        current = admission.claim()  # No Node can run until real CPU flocks admit.
        if current["region"] is None:
            guard(repo)
        elif repo != "app":
            raise Refusal("quarter route is APP-only")
        try:
            fd = writer_lock(repo)
        except BlockingIOError:
            print('gitnexus analyze writer lock is already held; wait for it to finish.', file=sys.stderr)
            return 75
        if admission.claim() != current:
            raise Refusal("CPU claim changed before writer work")
        if current["region"] is not None:
            return quarter_run(repo, sys.argv[4:], fd, admission, current)
        guard(repo)  # Existing whole-host behavior remains valid.
        if repo == 'root':
            patched = subprocess.run(['node', str(HERE / 'gitnexus-patch.js')], pass_fds=(fd,))
            if patched.returncode == 64:
                return 64
            # Preserve best-effort handling of other patcher failures.
            args = ['--skip-agents-md']
            help_result = subprocess.run(['gitnexus', 'analyze', '--help'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, pass_fds=(fd,))
            if b'--skip-skills' in help_result.stdout:
                args.append('--skip-skills')
        else:
            args = ['--skip-agents-md', '--skip-skills']
        os.set_inheritable(fd, True)
        os.execvp('gitnexus', ['gitnexus', 'analyze', *args, *sys.argv[4:]])
    except Termination as exc:
        print(f'gitnexus writer: terminated: {exc}', file=sys.stderr)
        return 128 + exc.signum
    except Exception as exc:
        print(f'gitnexus writer: denied: {exc}', file=sys.stderr)
        return 64
    finally:
        if fd is not None:
            os.close(fd)


if __name__ == '__main__':
    raise SystemExit(main())
