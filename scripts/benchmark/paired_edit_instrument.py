"""Opt-in paired instruction experiment; no model calls or suite integration.

The prompt contains only the initial file and selected instruction.  The evaluator
keeps the shared oracle and reference solutions out of that prompt.  Execution uses
SEC-EXEC containment regardless of the feature flag and refuses if unavailable.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import tempfile
from dataclasses import dataclass

from src.repl_environment import exec_containment, exec_sandbox


@dataclass(frozen=True)
class Fixture:
    id: str
    initial: str
    descriptive: str
    minimal: str
    entry: str
    cases: tuple
    gold: str
    mutant: str


FIXTURES = (
    Fixture('clamp', 'def clamp(n, low, high):\n    return n\n',
            'Return low when n is below low, high when n is above high, and n otherwise; endpoints are inclusive.',
            'Clamp n to the inclusive bounds.', 'clamp',
            (((-2, 0, 5), 0), ((7, 0, 5), 5), ((3, 0, 5), 3), ((0, 0, 5), 0), ((5, 0, 5), 5)),
            'def clamp(n, low, high):\n    return max(low, min(n, high))\n',
            'def clamp(n, low, high):\n    return min(n, high)\n'),
    Fixture('unique', 'def unique(items):\n    return items\n',
            'Return a new list containing each item once, preserving its first occurrence order; do not modify the input.',
            'Deduplicate items in first-seen order.', 'unique',
            ((([3, 1, 3, 2, 1],), [3, 1, 2]), (([],), []), (([2, 2],), [2])),
            'def unique(items):\n    return list(dict.fromkeys(items))\n',
            'def unique(items):\n    return sorted(set(items))\n'),
)


def prompt(fixture: Fixture, variant: str) -> str:
    if variant not in ('descriptive', 'minimal'):
        raise ValueError('unknown instruction variant')
    return f"{getattr(fixture, variant)}\n\nReturn the complete Python file.\n\n{fixture.initial}"


# Candidate receives one input at a time. Expected values and the verdict stay
# in the trusted parent. Strict JSON framing rejects early exit and extra output.
_HARNESS = """import json, os, resource, sys
try:
    if os.geteuid() == 0:
        raise RuntimeError('root bypasses RLIMIT_NPROC')
    with open('/proc/self/status') as status:
        effective = next(int(line.split()[1], 16) for line in status if line.startswith('CapEff:'))
    if effective & ((1 << 21) | (1 << 24)):
        raise RuntimeError('capabilities bypass RLIMIT_NPROC')
    resource.setrlimit(resource.RLIMIT_NPROC, (1, 1))
    resource.setrlimit(resource.RLIMIT_FSIZE, (65536, 65536))
    resource.setrlimit(resource.RLIMIT_AS, (536870912, 536870912))
    if resource.getrlimit(resource.RLIMIT_NPROC) != (1, 1):
        raise RuntimeError('RLIMIT_NPROC not enforced')
except Exception:
    print('paired-edit: process control unavailable', file=sys.stderr)
    sys.exit(126)
p = json.load(sys.stdin)
ns = {}
exec(compile(p['code'], '<attempt>', 'exec'), ns)
result = ns[p['entry']](*p['args'])
print(json.dumps({'result': result, 'args': p['args']}, allow_nan=False))
"""


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f'invalid JSON constant: {value}')


def evaluate(fixture: Fixture, attempt: str, timeout: float = 5.0) -> str:
    """Return pass, fail, timeout, or unavailable. Never invoke a model."""
    if not isinstance(attempt, str) or not attempt.strip():
        return 'fail'
    if not fixture.cases or not fixture.entry.isidentifier() or not 0 < timeout <= 30:
        raise ValueError('invalid oracle or timeout')
    try:
        argv, extra = exec_sandbox.wrap(
            ['/usr/bin/python3', '-I', '-c', _HARNESS], policy=exec_sandbox.Policy(),
            deny_paths=[*exec_sandbox.credential_paths(), '/home', '/workspace', '/mnt'],
        )
        env = exec_containment.child_env(extra)
        with tempfile.TemporaryDirectory(prefix='paired-edit-') as scratch:
            for args, expected in fixture.cases:
                payload = json.dumps({'code': attempt, 'entry': fixture.entry, 'args': args})
                with tempfile.TemporaryFile(mode='w+') as out, tempfile.TemporaryFile(mode='w+') as err:
                    process = subprocess.Popen(argv, stdin=subprocess.PIPE, text=True, stdout=out, stderr=err,
                                               cwd=scratch, env=env, start_new_session=True)
                    # The harness blocks on stdin until custody is established.
                    # Keep its exact UID and birth ticks for the exceptional fallback.
                    proc_dir = f'/proc/{process.pid}'
                    uid, birth = None, None
                    try:
                        uid = os.stat(proc_dir).st_uid
                        with open(f'{proc_dir}/stat') as stat_file:
                            birth = stat_file.read().rsplit(') ', 1)[1].split()[19]
                        if uid != os.getuid():
                            raise OSError('unexpected child UID')
                        pidfd = os.pidfd_open(process.pid)
                    except OSError:
                        process.stdin.close()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            current_uid = os.stat(proc_dir).st_uid
                            with open(f'{proc_dir}/stat') as stat_file:
                                current_birth = stat_file.read().rsplit(') ', 1)[1].split()[19]
                            if current_uid != uid or current_birth != birth or uid != os.getuid():
                                raise RuntimeError('child identity changed during cleanup')
                            os.kill(process.pid, signal.SIGKILL)
                            process.wait(timeout=5)
                        return 'unavailable'
                    try:
                        try:
                            process.communicate(payload, timeout=timeout)
                        except subprocess.TimeoutExpired:
                            try:
                                signal.pidfd_send_signal(pidfd, signal.SIGKILL)
                            except ProcessLookupError:
                                pass  # Exited between timeout and signal; still reap it.
                            process.communicate()
                            return 'timeout'
                    finally:
                        try:
                            if process.poll() is None:
                                try:
                                    signal.pidfd_send_signal(pidfd, signal.SIGKILL)
                                except ProcessLookupError:
                                    pass
                                except OSError:
                                    current_uid = os.stat(proc_dir).st_uid
                                    with open(f'{proc_dir}/stat') as stat_file:
                                        current_birth = stat_file.read().rsplit(') ', 1)[1].split()[19]
                                    if current_uid != uid or current_birth != birth or uid != os.getuid():
                                        raise RuntimeError('child identity changed during cleanup')
                                    os.kill(process.pid, signal.SIGKILL)
                                process.wait(timeout=5)
                        finally:
                            os.close(pidfd)
                    err.seek(0)
                    note = err.read(4096)
                    if (exec_sandbox.shim_failure_note(process.returncode, note)
                            or (process.returncode == 126 and 'paired-edit: process control unavailable' in note)):
                        return 'unavailable'
                    if process.returncode != 0:
                        return 'fail'
                    out.seek(0, 2)
                    if out.tell() > 65536:
                        return 'fail'
                    out.seek(0)
                    try:
                        response = json.loads(out.read(), object_pairs_hook=_unique_object,
                                              parse_constant=_invalid_constant)
                    except (ValueError, UnicodeError, RecursionError):
                        return 'fail'
                    if (not isinstance(response, dict) or set(response) != {'result', 'args'}
                            or json.dumps(response['result'], sort_keys=True) != json.dumps(expected, sort_keys=True)
                            or json.dumps(response['args']) != json.dumps(args)):
                        return 'fail'
        return 'pass'
    except (exec_sandbox.ContainmentUnavailable, OSError):
        return 'unavailable'
