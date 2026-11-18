"""Pytest plugin: run an existing REPL test suite with EVERY ``REPLEnvironment`` routed through the Monty lane.

Usage (needs ``pydantic-monty``)::

    pytest -p tests.monty_lane_parity tests/unit/test_repl_environment.py ...

Every environment a test constructs gets the lane installed at the end of ``__init__`` (as ``create_repl_environment``
does in production), with the flag forced ON for the lane even when a test resets the feature registry. The suite's
own assertions are the parity oracle: they must pass unchanged, through Monty or through the CPython fallback.
At the end the plugin prints the route / fallback shares it observed and, with ``MONTY_PARITY_REPORT=<path>``,
writes them as JSON.
"""

from __future__ import annotations

import collections
import json
import os

import pytest

_LOGS: list = []  # route logs only: holding adapters would pin their workers
_FAILED_INSTALLS: list[str] = []
_CURRENT = {"nodeid": "", "whitebox": False}
#: Tests that drive the CPython executor's own internals (they hook ``builtins.exec`` on ``repl._globals`` to inject
#: a deadline). They test the CPython lane by construction, so their environments are left on CPython and counted.
WHITEBOX_CPYTHON = (
    "TestExecuteTimeout::test_alarm_timeout_marks_terminal_and_refuses_later_execute",
)
_WHITEBOX_RUNS: list[str] = []
_NODE_OF: dict[int, str] = {}


def pytest_runtest_setup(item) -> None:
    _CURRENT["nodeid"] = item.nodeid
    _CURRENT["whitebox"] = any(item.nodeid.endswith(w) for w in WHITEBOX_CPYTHON)
    if _CURRENT["whitebox"]:
        _WHITEBOX_RUNS.append(item.nodeid)


def pytest_configure(config: pytest.Config) -> None:
    import pydantic_monty  # noqa: F401  - fail loudly: a parity run without Monty proves nothing

    os.environ.setdefault("ORCHESTRATOR_MONTY_LANE_LOG", "off")
    from src.repl_environment import environment as env_mod
    from src.repl_environment import monty_adapter

    monty_adapter.monty_enabled = lambda: True  # forced for the lane, whatever a test does to the registry
    orig_init = env_mod.REPLEnvironment.__init__

    def init(self, *a, **kw):
        orig_init(self, *a, **kw)
        if type(self).__name__ == "_RestrictedREPLEnvironment" or _CURRENT["whitebox"]:
            return
        try:
            ad = monty_adapter.maybe_monty_adapter(self, telemetry=False)
        except Exception as exc:  # noqa: BLE001
            _FAILED_INSTALLS.append(f"{type(exc).__name__}: {exc}")
            return
        if ad is None:
            _FAILED_INSTALLS.append(type(self).__name__)
        else:
            _LOGS.append(ad.route_log)
            _NODE_OF[id(ad.route_log)] = _CURRENT["nodeid"]

    env_mod.REPLEnvironment.__init__ = init


def summary() -> dict:
    rows = [r for log in _LOGS for r in log]
    routes = collections.Counter(r["route"] for r in rows)
    in_subset = [r for r in rows if r["screen"] == "in_subset"]
    fallbacks = collections.Counter((r["fallback"] or "").split(":")[0] + ":" + ((r["fallback"] or "").split(":") + [""])[1]
                                    for r in in_subset if r["fallback"])
    screens = collections.Counter(r["screen"].split(":")[0] + ":" + (r["screen"].split(":") + [""])[1]
                                  for r in rows if r["screen"] != "in_subset")
    return {
        "environments": len(_LOGS),
        "install_failures": len(_FAILED_INSTALLS),
        "whitebox_cpython_only_tests": len(_WHITEBOX_RUNS),
        "cells": len(rows),
        "routes": dict(routes),
        "screened_in": len(in_subset),
        "fell_back": sum(fallbacks.values()),
        "fallback_reasons": dict(fallbacks.most_common()),
        "screen_out_reasons": dict(screens.most_common()),
    }


def pytest_terminal_summary(terminalreporter, exitstatus, config) -> None:  # noqa: ARG001
    s = summary()
    terminalreporter.write_line("monty-lane parity: " + json.dumps(s, sort_keys=True))
    path = os.environ.get("MONTY_PARITY_REPORT")
    if path:
        with open(path, "w") as fh:
            json.dump(s, fh, indent=1, sort_keys=True)
    from src.repl_environment.monty_adapter import shutdown_pools

    shutdown_pools()


def pytest_sessionfinish(session, exitstatus) -> None:  # noqa: ARG001
    path = os.environ.get("MONTY_PARITY_ROWS")
    if path:
        with open(path, "w") as fh:
            for log in _LOGS:
                for r in log:
                    fh.write(json.dumps({**r, "test": _NODE_OF.get(id(log), "")}, sort_keys=True, default=str) + "\n")
