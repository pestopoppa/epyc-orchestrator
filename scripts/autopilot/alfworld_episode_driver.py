#!/usr/bin/env python3
"""ALFWorld episode driver (AP-ALFWORLD-SUITE): runs INSIDE the native ALFWorld venv, never imported by the orchestrator.

The Life-Harness native ALFWorld environment lives in its own Python 3.10 venv
(``<lh_root>/AgentBench/.native/alfworld-venv``: textworld, jericho, fast-downward, nltk, numpy, ...). The orchestrator
venv must not import it, so ``scripts/autopilot/alfworld_suite.py`` launches THIS file as a subprocess and talks to it
through two files: a JSON request in, a JSONL row per finished episode out (written as each episode ends, so a killed
batch still leaves every completed episode behind).

Python-3.10 compatible, stdlib only plus the AgentBench modules the ``agentbench`` env adapter imports lazily.

Episode semantics are AgentBench's own ALFWorld loop (``src/server/tasks/alfworld/task.py::alfworld_run``) driven by
the served model through an OpenAI-compatible ``/chat/completions`` endpoint with one client tool, ``take_action``:

* the task instruction text IS ``ALFWorld.get_task_instruction()`` (imported, not copied), the first user turn is
  ``'Here is your task. ' + observation + ' AVAILABLE ACTIONS: ...'``, the tool result is the next observation plus
  the admissible actions;
* the chosen action goes through AgentBench's ``process_action`` (exact match, else nearest admissible by BLEU);
* a turn with no tool call costs a step and gets the same nudge AgentBench injects;
* the same action three turns in a row ends the episode (AgentBench's ``AGENT_INVALID_ACTION``).

Row statuses use the vocabulary of ``bsv2_accept_rigor.STATUSES`` so the R4 status split reads them directly:

* ``completed``       the game ended (won, or the env reported done without a win);
* ``solver_limit``    step cap, per-episode wall budget, or the endpoint's context-length refusal;
* ``candidate_error`` the model repeated one action three times in a row;
* ``infra_error``     the endpoint was unreachable / answered 5xx / the env raised: no verdict about the model.

``success`` is 1 iff the environment reported ``won`` (episode success = 1, everything else 0).

Env adapters: ``--env-adapter agentbench`` (default) or ``module:callable`` where the callable takes the AgentBench root
and returns an object with ``task_instruction() -> str``, ``process_action(action, choices) -> str``,
``open(game_path) -> episode`` and ``episode.reset() -> (observation, admissible)``,
``episode.step(action) -> (observation, won, done, admissible)``, ``episode.close()``. The tests use a fake one.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

TOOL = {
    "type": "function",
    "function": {
        "name": "take_action",
        "description": "Take an action.",
        "parameters": {
            "type": "object",
            "properties": {"action": {"type": "string", "description": "The action you would like to take"}},
            "required": ["action"],
            "additionalProperties": False,
        },
    },
}
NO_TOOL_NUDGE = "You MUST call the take_action tool — do NOT output plain text without a tool call."
BAD_TOOL_NUDGE = "No valid tool calls found. Please call a tool instead."
REPEAT_LIMIT = 3

_NO_PROXY = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class ModelError(Exception):
    """The model endpoint failed; ``kind`` is ``infra`` or ``context``."""

    def __init__(self, kind: str, detail: str):
        super().__init__(detail)
        self.kind = kind
        self.detail = detail


# ---------------------------------------------------------------- env adapters
class AgentBenchAdapter:
    """The real native env: AgentBench's ALFWorld wrapper, task text and action matcher (lazy imports)."""

    def __init__(self, root: str):
        import yaml

        self.root = Path(root)
        sys.path.insert(0, str(self.root))
        os.environ["ALFWORLD_DATA"] = str(self.root / "data" / "alfworld")
        from src.server.tasks.alfworld.environment import AlfworldEnvWrapper
        from src.server.tasks.alfworld.task import ALFWorld
        from src.server.tasks.alfworld.utils import process_action, process_ob

        config = yaml.safe_load((self.root / "src/server/tasks/alfworld/configs/base_config.yaml").read_text())
        self._wrapper = AlfworldEnvWrapper(config)
        self._instruction = ALFWorld.get_task_instruction()
        self._process_action = process_action
        self._process_ob = process_ob

    def task_instruction(self) -> str:
        return self._instruction

    def process_action(self, action: str, choices: list) -> str:
        return self._process_action(action, choices)

    def open(self, game_path: str) -> "_AgentBenchEpisode":
        return _AgentBenchEpisode(self, self._wrapper.create_env(game_path))


class _AgentBenchEpisode:
    def __init__(self, adapter: AgentBenchAdapter, env: Any):
        self._a, self._env = adapter, env

    def reset(self) -> tuple[str, list]:
        obs, info = self._a._wrapper.reset_env(self._env)
        text = "\n".join(obs[0].split("\n\n")[1:])  # AgentBench drops the banner paragraph
        return text, list(info.get("admissible_commands", [[]])[0])

    def step(self, action: str) -> tuple[str, bool, bool, list]:
        obs, _reward, done, info = self._a._wrapper.step_env(self._env, action)
        return (self._a._process_ob(obs[0]), bool(info["won"][0]), bool(done[0]),
                list(info.get("admissible_commands", [[]])[0]))

    def close(self) -> None:
        self._a._wrapper.close_env(self._env)


def load_adapter(spec: str, root: str) -> Any:
    if spec == "agentbench":
        return AgentBenchAdapter(root)
    module, _, attr = spec.partition(":")
    if not module or not attr:
        raise ValueError(f"env adapter must be 'agentbench' or 'module:callable', got {spec!r}")
    sys.path.insert(0, root)  # a custom adapter module lives in (or beside) the AgentBench root
    return getattr(importlib.import_module(module), attr)(root)


# ---------------------------------------------------------------- model call
def _post(url: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"})
    try:
        with _NO_PROXY.open(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", "replace")[:400]
        if exc.code == 400 and "context" in text.lower():
            raise ModelError("context", f"http_400_context: {text}")
        raise ModelError("infra", f"http_{exc.code}: {text}")
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise ModelError("infra", f"{type(exc).__name__}: {exc}")


def call_model(req: dict, messages: list, timeout: float) -> dict:
    body = {"model": req["model"], "messages": messages, "tools": [TOOL], "tool_choice": "auto",
            "temperature": req.get("temperature", 0.0), "max_tokens": req.get("max_tokens", 1024)}
    if req.get("force_role"):
        body["x_force_role"] = req["force_role"]
    url = req["endpoint"].rstrip("/") + "/chat/completions"
    try:
        return _post(url, body, timeout)
    except ModelError as first:
        if first.kind != "infra" or "http_4" in first.detail:
            raise
        return _post(url, body, timeout)  # one retry for a transient transport / 5xx failure


# ---------------------------------------------------------------- one episode
def _tool_action(message: dict) -> tuple[str | None, str | None]:
    """(action, call_id) from the first tool call, or (None, call_id) when the arguments are unusable."""
    calls = message.get("tool_calls") or []
    if not calls:
        return None, None
    call = calls[0]
    call_id = str(call.get("id") or "call_0")
    try:
        args = json.loads(call["function"]["arguments"])
        value = list(args.values())[0]
        return (str(value) if value is not None else None), call_id
    except (KeyError, IndexError, TypeError, ValueError, AttributeError):
        return None, call_id


def run_episode(adapter: Any, req: dict, ep: dict, *, model=call_model, clock=time.monotonic) -> dict:
    started = clock()
    row: dict[str, Any] = {"qid": ep["qid"], "index": ep.get("index"), "game": ep.get("game"), "success": 0,
                           "status": "infra_error", "detail": "", "steps": 0, "model_calls": 0,
                           "tokens_prompt": 0, "tokens_completion": 0, "elapsed_s": 0.0}
    max_steps, wall = int(req["max_steps"]), float(req["episode_wall_s"])
    episode = None
    try:
        try:
            episode = adapter.open(ep["game"])
            observation, admissible = episode.reset()
        except Exception as exc:  # noqa: BLE001 - env failure carries no verdict about the model
            row.update(status="infra_error", detail=f"env_error: {type(exc).__name__}: {exc}"[:300])
            return row
        avail = lambda acts: " AVAILABLE ACTIONS: " + "\n".join(acts) + "\n"  # noqa: E731
        messages: list[dict] = [{"role": "system", "content": adapter.task_instruction()},
                                {"role": "user", "content": "Here is your task. " + observation + avail(admissible)}]
        recent: list[str] = []
        for step in range(1, max_steps + 1):
            remaining = wall - (clock() - started)
            if remaining <= 0:
                row.update(status="solver_limit", detail="episode_wall")
                return row
            row["steps"] = step
            try:
                resp = model(req, messages, min(float(req["request_timeout_s"]), remaining))
            except ModelError as exc:
                row.update(status="solver_limit" if exc.kind == "context" else "infra_error",
                           detail=("context_limit: " if exc.kind == "context" else "model_error: ") + exc.detail[:300])
                return row
            row["model_calls"] += 1
            usage = resp.get("usage") or {}
            row["tokens_prompt"] += int(usage.get("prompt_tokens") or 0)
            row["tokens_completion"] += int(usage.get("completion_tokens") or 0)
            try:
                message = resp["choices"][0]["message"]
            except (KeyError, IndexError, TypeError):
                row.update(status="infra_error", detail="malformed_model_response")
                return row
            messages.append({"role": "assistant", "content": message.get("content") or "",
                             **({"tool_calls": message["tool_calls"]} if message.get("tool_calls") else {})})
            action, call_id = _tool_action(message)
            if action is None:
                messages.append({"role": "user", "content": NO_TOOL_NUDGE if call_id is None else BAD_TOOL_NUDGE})
                if call_id is not None:  # a tool call with unusable arguments still needs its tool turn answered
                    messages.insert(len(messages) - 1, {"role": "tool", "tool_call_id": call_id, "content": "invalid arguments"})
                continue
            action = adapter.process_action(action, admissible)
            try:
                observation, won, done, admissible = episode.step(action)
            except Exception as exc:  # noqa: BLE001
                row.update(status="infra_error", detail=f"env_error: {type(exc).__name__}: {exc}"[:300])
                return row
            messages.append({"role": "tool", "tool_call_id": call_id, "content": observation + avail(admissible)})
            recent.append(action)
            if won:
                row.update(success=1, status="completed", detail="won")
                return row
            if done:
                row.update(status="completed", detail="done_without_win")
                return row
            if len(recent) >= REPEAT_LIMIT and len(set(recent[-REPEAT_LIMIT:])) == 1:
                row.update(status="candidate_error", detail="repeated_action")
                return row
        row.update(status="solver_limit", detail="step_cap")
        return row
    finally:
        row["elapsed_s"] = round(clock() - started, 3)
        if episode is not None:
            try:
                episode.close()
            except Exception:  # noqa: BLE001 - closing must never mask the verdict
                pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--request", required=True, help="JSON request file")
    parser.add_argument("--out", required=True, help="JSONL file, one row appended per finished episode")
    args = parser.parse_args(argv)
    req = json.loads(Path(args.request).read_text())
    adapter = load_adapter(req.get("env_adapter", "agentbench"), req["agentbench_root"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ep in req["episodes"]:
        row = run_episode(adapter, req, ep)
        with out.open("a") as fh:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
    return 0


if __name__ == "__main__":
    sys.exit(main())
