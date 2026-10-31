import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


def sampler(dead):
    file = Path(__file__).with_name('prep_supervisor.py')
    tree = ast.parse(file.read_text())
    function = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'sample_tree')
    namespace = {
        'os': SimpleNamespace(getpid=lambda: 123, pidfd_open=lambda _: 10, close=lambda _: None),
        'Path': Path,
        'identity': lambda _: (_ for _ in ()).throw(ProcessLookupError(3, 'No such process')),
        'dead_pidfd': lambda _: dead,
        'exited_processes': 0,
        'exited_threads': 0,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(file), 'exec'), namespace)
    return namespace


def test_esrch_is_accepted_only_after_pidfd_reports_exit():
    env = sampler(True)
    assert env['sample_tree']() == (0, 1)
    assert env['exited_processes'] == 1


def test_esrch_on_live_pidfd_refuses():
    env = sampler(False)
    with pytest.raises(RuntimeError, match='live process became unreadable'):
        env['sample_tree']()
