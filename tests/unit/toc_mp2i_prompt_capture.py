"""Capture an actual flag-off graph prompt with a fake model; no inference."""
import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.features import Features, set_features
from src.graph.helpers import _execute_turn
from src.graph.state import TaskDeps, TaskState
from src.repl_environment import REPLEnvironment


class Recorder:
    def __init__(self):
        self.prompts = []
    def llm_call(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return 'FINAL(42)'


def capture(mode, scratch):
    set_features(Features(session_log=True, output_spill_to_file=True))
    repl = REPLEnvironment(context='same-context', task_id='parity')
    repl._session_artifact_nonce = 'fixednonce'
    repl._session_artifact_dir = Path(scratch) / 'fixed'
    repl._session_artifact_base = Path(scratch)
    repl._session_artifact_dir.mkdir(parents=True, exist_ok=True)
    state = TaskState(task_id='parity', prompt='What is the final answer?')
    if mode == 'summary':
        state.session_summary_cache = 'Earlier summary with exact fact 42.'
    if mode == 'spill':
        state.last_output = 'HEAD-' + ('output-block-' * 300) + '-TAIL'
    worker = Recorder()
    ctx = SimpleNamespace(state=state, deps=TaskDeps(repl=repl, primitives=worker))
    with patch('src.graph.helpers._answer_force_due', return_value=False):
        asyncio.run(_execute_turn(ctx, 'worker_general'))
    assert len(worker.prompts) == 1
    return worker.prompts[0].replace(str(Path(scratch)), '<TMP>')


if __name__ == '__main__':
    print(json.dumps({mode: capture(mode, sys.argv[1]) for mode in ('ordinary', 'summary', 'spill')}, sort_keys=True))


def capture_schema(scratch):
    import os
    from src.graph import trajectory_artifact as trajectory

    os.environ['ORCHESTRATOR_TRAJECTORY_ARTIFACT_DIR'] = str(scratch)
    set_features(Features(trajectory_artifact=True))
    trajectory._reset_for_tests()
    state = TaskState(task_id='schema-parity', turns=1)
    with patch.object(trajectory, '_served_n_ctx', return_value=(1024, 'fixture')):
        trajectory.note_call(state, role='worker', prompt='prompt', meta={'prompt_tokens': 8, 'tokens': 2})
        trajectory.note_turn(state, role='worker', code='print(1)', output='1', tool_calls=['peek'])
        trajectory.finalize(state, '42', True)
    rows = trajectory.read_events(Path(scratch) / 'schema-parity.trajectory.jsonl')
    for row in rows:
        row.pop('ts', None)
    return rows


def capture_repl_spill(scratch):
    from src.repl_environment import REPLConfig

    set_features(Features())
    config = REPLConfig(output_cap=200, spill_dir=str(scratch))
    repl = REPLEnvironment(context='same-context', task_id='parity', config=config)
    result = repl.execute('print("line-needle\\n" * 100)')
    assert not result.error
    return result.output.replace(str(Path(scratch)), '<TMP>')
