"""Opt-in paired instruction experiment; no model calls or suite integration.

The prompt contains only the initial file and selected instruction.  The evaluator
keeps the shared oracle and reference solutions out of that prompt.  Execution uses
SEC-EXEC containment regardless of the feature flag and refuses if unavailable.
"""
from __future__ import annotations

import json
import os
import secrets
import signal
import subprocess
import sys
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


# Runs inside the existing containment shim. Oracle is passed to the child only at
# evaluation time, never to prompt(). A malformed/empty attempt cannot pass.
_HARNESS = '''import copy, json, sys\np = json.load(sys.stdin)\nns = {}\nexec(compile(p["code"], "<attempt>", "exec"), ns)\nf = ns[p["entry"]]\nfor args, expected in p["cases"]:\n    original = copy.deepcopy(args)\n    if f(*args) != expected or args != original:\n        sys.exit(1)\nprint(p["token"])\n'''


def evaluate(fixture: Fixture, attempt: str, timeout: float = 5.0) -> str:
    """Return pass, fail, timeout, or unavailable. Never invoke a model."""
    if not isinstance(attempt, str) or not attempt.strip():
        return 'fail'
    if not fixture.cases or not fixture.entry.isidentifier() or not 0 < timeout <= 30:
        raise ValueError('invalid oracle or timeout')
    try:
        argv, extra = exec_sandbox.wrap([sys.executable, '-I', '-c', _HARNESS], policy=exec_sandbox.Policy())
        env = exec_containment.child_env(extra)
        with tempfile.TemporaryDirectory(prefix='paired-edit-') as scratch:
            # Files instead of PIPEs prevent unbounded output accumulation in memory.
            with tempfile.TemporaryFile(mode='w+') as out, tempfile.TemporaryFile(mode='w+') as err:
                token = secrets.token_hex(32)
                process = subprocess.Popen(argv, stdin=subprocess.PIPE, text=True, stdout=out, stderr=err,
                                           cwd=scratch, env=env, start_new_session=True)
                try:
                    process.communicate(json.dumps({'code': attempt, 'entry': fixture.entry,
                                                    'cases': fixture.cases, 'token': token}), timeout=timeout)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.communicate()
                    return 'timeout'
                err.seek(0)
                note = err.read(4096)
                out.seek(0, 2)
                size = out.tell()
                out.seek(max(0, size - 256))
                tail = out.read()
        if exec_sandbox.shim_failure_note(process.returncode, note):
            return 'unavailable'
        return 'pass' if process.returncode == 0 and tail.splitlines()[-1:] == [token] else 'fail'
    except subprocess.TimeoutExpired:
        return 'timeout'
    except (exec_sandbox.ContainmentUnavailable, OSError):
        return 'unavailable'
