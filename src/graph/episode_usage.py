"""Prospective native backend usage for one opt-in compression episode."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Baseline:
    ledger: Any
    task_id: str
    call_ids: frozenset[str]


def begin(primitives: Any, repl: Any, task_id: str) -> Baseline | None:
    """Bind only the active request's ledger and the REPL's actual primitives."""
    try:
        from src.llm_primitives.usage import UsageLedger

        if (not task_id or repl.llm_primitives is not primitives
                or getattr(primitives, '_bound_request_usage', None) is not None):
            return None
        if primitives._request_task_id_ctx.get() != task_id:
            return None
        ledger = primitives._usage_ledger_ctx.get()
        if not isinstance(ledger, UsageLedger):
            return None
        records = ledger.records()
        return Baseline(ledger, task_id, frozenset(record['call_id'] for record in records))
    except (AttributeError, KeyError, TypeError):
        return None


def finish(baseline: Baseline | None, primitives: Any, repl: Any, task_id: str) -> dict[str, Any]:
    """Never turn missing, pending, estimated or foreign usage into exact totals."""
    unknown = {
        'status': 'unknown', 'backend_attempts': None, 'cached_replays': None,
        'observed_input_calls': None, 'observed_output_calls': None,
        'total_input_tokens': None, 'total_output_tokens': None, 'total_tokens': None,
    }
    try:
        if (baseline is None or baseline.task_id != task_id or repl.llm_primitives is not primitives
                or getattr(primitives, '_bound_request_usage', None) is not None
                or primitives._request_task_id_ctx.get() != task_id
                or primitives._usage_ledger_ctx.get() is not baseline.ledger):
            return unknown
        records = [r for r in baseline.ledger.records() if r['call_id'] not in baseline.call_ids]
        replays = [r for r in records if r['provenance'] == 'cached_replay_of_server_reported'
                   and not r['pending'] and not r['censored']]
        attempts = [r for r in records if r not in replays]

        def observed(record: dict, key: str) -> bool:
            value = record.get(key)
            return (record.get('provenance') == 'server_reported' and not record.get('pending')
                    and not record.get('censored') and type(value) is int and value >= 0)

        ins = [r['prompt_tokens'] for r in attempts if observed(r, 'prompt_tokens')]
        outs = [r['completion_tokens'] for r in attempts if observed(r, 'completion_tokens')]
        # An empty record set cannot prove that the expected graph call was observed.
        complete_in = bool(records) and len(ins) == len(attempts)
        complete_out = bool(records) and len(outs) == len(attempts)
        total_in = sum(ins) if complete_in else None
        total_out = sum(outs) if complete_out else None
        return {
            'status': 'complete' if complete_in and complete_out else 'partial',
            'backend_attempts': len(attempts), 'cached_replays': len(replays),
            'observed_input_calls': len(ins), 'observed_output_calls': len(outs),
            'total_input_tokens': total_in, 'total_output_tokens': total_out,
            'total_tokens': total_in + total_out if total_in is not None and total_out is not None else None,
        }
    except (AttributeError, KeyError, TypeError, ValueError):
        return unknown
