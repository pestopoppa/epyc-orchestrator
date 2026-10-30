"""Experimental request-local generation reservation ledger; not integrated."""
from __future__ import annotations

from contextvars import ContextVar, copy_context
from dataclasses import dataclass
from threading import Lock
from typing import Callable, TypeVar


class BudgetUnavailable(RuntimeError):
    """No safe generation allowance is available at present."""


@dataclass(frozen=True)
class BudgetSnapshot:
    limit: int
    used_exact: int
    reserved: int
    unknown: bool
    exhausted: bool


class RequestTokenBudget:
    """Reserve before dispatch; only exact completed usage can release allowance.

    On unknown/partial/error the reservation is never refunded and this request
    refuses every future reservation. Already-dispatched calls must still settle.
    """
    def __init__(self, limit: int):
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError('limit must be a positive integer')
        self._limit = limit
        self._used = 0
        self._pending: dict[object, int] = {}
        self._unknown = False
        self._lock = Lock()

    def reserve(self, requested: int) -> tuple[object, int]:
        if isinstance(requested, bool) or not isinstance(requested, int) or requested <= 0:
            raise ValueError('requested must be a positive integer')
        with self._lock:
            if self._unknown:
                raise BudgetUnavailable('usage unknown')
            available = self._limit - self._used - sum(self._pending.values())
            if available <= 0:
                raise BudgetUnavailable('no unreserved tokens')
            allowance = min(requested, available)
            ticket = object()
            self._pending[ticket] = allowance
            return ticket, allowance

    def settle(self, ticket: object, *, exact_usage: int | None, complete: bool) -> None:
        with self._lock:
            if ticket not in self._pending:
                raise ValueError('unknown or already settled reservation')
            allowed = self._pending.pop(ticket)
            exact = (complete and isinstance(exact_usage, int) and not isinstance(exact_usage, bool)
                     and 0 <= exact_usage <= allowed)
            if exact:
                self._used += exact_usage
            else:
                # Conservatively retain the whole allowance as spent.  It is not
                # represented as exact use, and can never establish exhaustion.
                self._unknown = True

    def snapshot(self) -> BudgetSnapshot:
        with self._lock:
            reserved = sum(self._pending.values())
            return BudgetSnapshot(self._limit, self._used, reserved, self._unknown,
                                  not self._unknown and not reserved and self._used == self._limit)


_current_budget: ContextVar[RequestTokenBudget | None] = ContextVar('request_token_budget', default=None)
T = TypeVar('T')


def bind_budget(budget: RequestTokenBudget):
    return _current_budget.set(budget)


def reset_budget(token) -> None:
    _current_budget.reset(token)


def current_budget() -> RequestTokenBudget | None:
    return _current_budget.get()


def submit_with_context(executor, fn: Callable[..., T], /, *args, **kwargs):
    """Capture a fresh Context per submission; threads do not inherit it automatically."""
    context = copy_context()
    return executor.submit(context.run, fn, *args, **kwargs)

# One thread-local receipt per backend attempt; context is copied into workers.
_attempt_receipt: ContextVar[list[tuple[object, object]] | None] = ContextVar('budget_attempt_receipt', default=None)

def note_usage(value, source):
    receipt = _attempt_receipt.get()
    if receipt is not None:
        receipt.append((value, source))
