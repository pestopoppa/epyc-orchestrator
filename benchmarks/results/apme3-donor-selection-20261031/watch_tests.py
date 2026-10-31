#!/usr/bin/env python3
"""Wait for R6 admission, then run the reviewed model-free controls once."""
import datetime
import fcntl
import hashlib
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/mnt/raid0/llm/tmp/codex3-apme3-20261031')
ATTEMPT = ROOT / 'watch-attempt-02'
CAPSULE = Path('/mnt/raid0/llm/tmp/codex3-k4a-20261031/capsule')
sys.path.insert(0, str(CAPSULE))
import prep_admission

TERMINAL = ATTEMPT / 'watch-terminal.json'
LOG = ATTEMPT / 'watch-pytest.log'
STARTUP = ATTEMPT / 'watch-startup.json'
GUARD = ATTEMPT / 'watch-lease.lock'
cancelled = False

def on_signal(_signum, _frame):
    global cancelled
    cancelled = True

signal.signal(signal.SIGTERM, on_signal)
signal.signal(signal.SIGINT, on_signal)
SOURCE = [
    'scripts/autopilot/actions.py',
    'scripts/autopilot/crossover_diagnostics.py', 'scripts/autopilot/species/mutation_graph.py',
    'tests/unit/test_mutation_diagnostics.py', 'tests/unit/test_mutation_graph.py',
]
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    started = time.monotonic()
    birth = int(Path('/proc/self/stat').read_text().rsplit(') ', 1)[1].split()[19])
    outcome = {'state': 'failure', 'watcher_pid': os.getpid(), 'watcher_birth': birth,
               'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    lease = GUARD.open('a+', encoding='utf-8')
    try:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return 1
    if TERMINAL.exists() or LOG.exists() or STARTUP.exists():
        return 1
    try:
        with LOG.open('x', encoding='utf-8'):
            pass
        with STARTUP.open('x', encoding='utf-8') as startup:
            json.dump(outcome, startup, indent=2, sort_keys=True)
            startup.write('\n')
        outcome['watcher_sha256'] = digest(Path(__file__))
        outcome['source_sha256'] = {path: digest(ROOT / path) for path in SOURCE}
        outcome['supervisor_sha256'] = digest(CAPSULE / 'prep_supervisor.py')
        outcome['admission_sha256'] = digest(CAPSULE / 'prep_admission.py')
        with open('/mnt/raid0/llm/tmp/compute-flags/prep-locks/26-27.lock', 'a+', encoding='utf-8') as lock:
            while time.monotonic() - started < 5400:
                if cancelled:
                    outcome['error'] = 'watcher interrupted before test launch'
                    break
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    time.sleep(15)
                    continue
                pair = None
                try:
                    for candidate in ((26, 27), (50, 51), (74, 75), (2, 3), (122, 123)):
                        try:
                            prep_admission.live(set(candidate))
                        except (ValueError, OSError):
                            continue
                        pair = candidate
                        break
                    if pair is None:
                        fcntl.flock(lock, fcntl.LOCK_UN)
                        time.sleep(15)
                        continue
                    lock.seek(0)
                    lock.truncate()
                    lock.write(f'{pair[0]},{pair[1]}\n')
                    lock.flush()
                    outcome['cpus'] = list(pair)
                    if {path: digest(ROOT / path) for path in SOURCE} != outcome['source_sha256']:
                        raise RuntimeError('source changed while waiting for admission')
                    env = os.environ.copy()
                    env.update(OMP_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1',
                               PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
                               PYTHONPATH=(str(ROOT) + ':/mnt/raid0/llm/tmp/codex3-k4a-20261031/testdeps:'
                                           '/mnt/raid0/llm/epyc-orchestrator/.venv/lib/python3.11/site-packages'))
                    argv = ['/usr/bin/taskset', '-c', f'{pair[0]},{pair[1]}', '/usr/bin/python3',
                            str(CAPSULE / 'prep_supervisor.py'), '900', '--',
                            '/home/daniele/.local/share/uv/python/cpython-3.11-linux-x86_64-gnu/bin/python3.11',
                            '-m', 'pytest', '-p', 'no:cacheprovider', '-q',
                            '--basetemp=' + str(ATTEMPT / 'pytest-tmp'),
                            'tests/unit/test_mutation_diagnostics.py', 'tests/unit/test_mutation_graph.py']
                    outcome['argv'] = argv
                    with LOG.open('a', encoding='utf-8') as output:
                        outcome['supervisor_pre_sha256'] = digest(CAPSULE / 'prep_supervisor.py')
                        outcome['admission_pre_sha256'] = digest(CAPSULE / 'prep_admission.py')
                        if (outcome['supervisor_pre_sha256'] != outcome['supervisor_sha256']
                                or outcome['admission_pre_sha256'] != outcome['admission_sha256']):
                            raise RuntimeError('supervisor or admission changed before execution')
                        child = subprocess.run(argv, cwd=ROOT, env=env, stdout=output,
                                               stderr=subprocess.STDOUT, check=False)
                    outcome['returncode'] = child.returncode
                    outcome['supervisor_post_sha256'] = digest(CAPSULE / 'prep_supervisor.py')
                    outcome['admission_post_sha256'] = digest(CAPSULE / 'prep_admission.py')
                    outcome['log_sha256'] = digest(LOG)
                    outcome['state'] = 'success' if child.returncode == 0 and not cancelled else 'failure'
                    if (outcome['supervisor_post_sha256'] != outcome['supervisor_sha256']
                            or outcome['admission_post_sha256'] != outcome['admission_sha256']):
                        outcome['state'] = 'failure'
                        outcome['error'] = 'supervisor or admission changed during execution'
                    if cancelled:
                        outcome['error'] = 'watcher interrupted; supervised child was allowed to finish'
                    if {path: digest(ROOT / path) for path in SOURCE} != outcome['source_sha256']:
                        outcome['state'] = 'failure'
                        outcome['error'] = 'source changed during test execution'
                    break
                finally:
                    fcntl.flock(lock, fcntl.LOCK_UN)
            else:
                outcome['error'] = 'admission or mutex did not become available within 5400 seconds'
    except Exception as exc:
        outcome['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        outcome['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        temp = ATTEMPT / f'watch-terminal.{os.getpid()}.tmp'
        with temp.open('x', encoding='utf-8') as output:
            json.dump(outcome, output, indent=2, sort_keys=True)
            output.write('\n')
        try:
            os.link(temp, TERMINAL)
        finally:
            temp.unlink()
    return 0 if outcome['state'] == 'success' else 1

if __name__ == '__main__':
    sys.exit(main())
