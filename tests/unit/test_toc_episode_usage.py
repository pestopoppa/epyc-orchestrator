"""Native request ledger boundaries, including unknowns and cache replay."""
from contextvars import ContextVar
from types import SimpleNamespace

from src.graph import episode_usage
from src.llm_primitives.usage import UsageLedger

SOURCE = 'llama_server:/completion.tokens_predicted'


def _native(ledger, prompt, output, *, censored=False):
    call = ledger.begin('worker')
    ledger.finish(call, prompt_tokens=prompt, completion_tokens=output,
                  provenance='server_reported', source=SOURCE, censored=censored)


def _context(task='task'):
    ledger = UsageLedger(request_id='request')
    primitives = SimpleNamespace(_request_task_id_ctx=ContextVar('task', default=task),
                                 _usage_ledger_ctx=ContextVar('ledger', default=ledger))
    return ledger, primitives, SimpleNamespace(llm_primitives=primitives)


def test_episode_excludes_previous_calls_and_cache_replay():
    ledger, primitives, repl = _context()
    _native(ledger, 1000, 2000)
    baseline = episode_usage.begin(primitives, repl, 'task')
    _native(ledger, 10, 2)  # graph turn
    _native(ledger, 20, 4)  # compressor
    _native(ledger, 30, 6)  # retrieval call
    cached = ledger.begin('worker')
    ledger.finish(cached, prompt_tokens=99, completion_tokens=88,
                  provenance='cached_replay_of_server_reported', censored=False)
    result = episode_usage.finish(baseline, primitives, repl, 'task')
    assert result == {'status': 'complete', 'backend_attempts': 3, 'cached_replays': 1,
                      'observed_input_calls': 3, 'observed_output_calls': 3,
                      'total_input_tokens': 60, 'total_output_tokens': 12, 'total_tokens': 72}


def test_unknowns_pending_and_cross_request_never_become_zero():
    ledger, primitives, repl = _context()
    baseline = episode_usage.begin(primitives, repl, 'task')
    _native(ledger, 10, 2)
    ledger.begin('pending')
    result = episode_usage.finish(baseline, primitives, repl, 'task')
    assert result['status'] == 'partial' and result['backend_attempts'] == 2
    assert result['observed_input_calls'] == 1 and result['total_input_tokens'] is None
    assert result['total_output_tokens'] is None and result['total_tokens'] is None
    primitives._request_task_id_ctx.set('other')
    assert episode_usage.finish(baseline, primitives, repl, 'task')['status'] == 'unknown'
    assert episode_usage.begin(primitives, repl, 'task') is None


def test_partial_per_axis_and_wrong_primitive():
    ledger, primitives, repl = _context()
    baseline = episode_usage.begin(primitives, repl, 'task')
    _native(ledger, 10, None)
    result = episode_usage.finish(baseline, primitives, repl, 'task')
    assert result['status'] == 'partial' and result['total_input_tokens'] == 10
    assert result['total_output_tokens'] is None and result['total_tokens'] is None
    repl.llm_primitives = object()
    assert episode_usage.finish(baseline, primitives, repl, 'task')['status'] == 'unknown'


def test_bound_ledger_precedence_is_not_misattributed():
    _, primitives, repl = _context()
    baseline = episode_usage.begin(primitives, repl, 'task')
    primitives._bound_request_usage = UsageLedger(request_id='different')
    assert episode_usage.begin(primitives, repl, 'task') is None
    assert episode_usage.finish(baseline, primitives, repl, 'task')['status'] == 'unknown'


def test_concurrent_request_contexts_remain_separate():
    import asyncio

    primitives = SimpleNamespace(_request_task_id_ctx=ContextVar('concurrent_task'),
                                 _usage_ledger_ctx=ContextVar('concurrent_ledger'))
    repl = SimpleNamespace(llm_primitives=primitives)

    async def request(task, count):
        ledger = UsageLedger(request_id=task)
        task_token = primitives._request_task_id_ctx.set(task)
        ledger_token = primitives._usage_ledger_ctx.set(ledger)
        try:
            baseline = episode_usage.begin(primitives, repl, task)
            await asyncio.sleep(0)
            await asyncio.to_thread(_native, ledger, count, 1)
            await asyncio.sleep(0)
            return episode_usage.finish(baseline, primitives, repl, task)
        finally:
            primitives._usage_ledger_ctx.reset(ledger_token)
            primitives._request_task_id_ctx.reset(task_token)

    async def both():
        return await asyncio.gather(request('a', 11), request('b', 22))

    first, second = asyncio.run(both())
    assert first['total_input_tokens'] == 11 and first['backend_attempts'] == 1
    assert second['total_input_tokens'] == 22 and second['backend_attempts'] == 1
