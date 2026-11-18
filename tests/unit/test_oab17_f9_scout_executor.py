"""OAB-17 F9: scouts run on a dedicated bounded executor, not the loop default."""
from __future__ import annotations

import asyncio
import contextvars
import threading

from src.api.routes.chat_pipeline import scout_stage as ss

_VAR = contextvars.ContextVar("oab17_f9", default=None)


def _run(coro):
    return asyncio.run(coro)


def test_runs_on_scout_pool_with_context_propagated():
    async def go():
        _VAR.set("kept")
        return await ss._scout_thread(lambda: (threading.current_thread().name, _VAR.get()))
    name, value = _run(go())
    assert name.startswith("scout") and value == "kept"


def test_default_executor_untouched_and_pool_is_bounded():
    async def go():
        loop = asyncio.get_running_loop()
        marker = await loop.run_in_executor(None, lambda: threading.current_thread().name)
        scout = await ss._scout_thread(lambda: threading.current_thread().name)
        return marker, scout
    default_name, scout_name = _run(go())
    assert not default_name.startswith("scout") and scout_name.startswith("scout")
    assert ss._scout_executor()._max_workers == ss.SCOUT_EXECUTOR_WORKERS
