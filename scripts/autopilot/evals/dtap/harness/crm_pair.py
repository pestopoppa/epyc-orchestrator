"""Case-scoped CRM/Slack paired diagnostic; no services are launched here.

The caller supplies isolated service instances and owns their lifecycle and actual
admission. This module is not a claim of service qualification or inference.
"""
from __future__ import annotations

import copy
import hashlib
import json
import threading
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from .endpoint import ChatEndpoint
from .env_state import StateStore
from .runner import CaseRegistry, JudgeApplication, render_messages
from .shims import finance, gmail, paypal, salesforce, slack
from .tool_contract import load_native_tool_contract, validate_tool_calls

CASE = "crm-malicious-indirect-general-ai-restrictions-004"
TARGET = "create_record"
_JUDGE_LOCK = threading.Lock()


def _digest(data: Any) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


class Services(Protocol):
    """A real implementation must reset both isolated services and return their actual state."""
    def reset(self) -> dict: ...
    def invoke(self, name: str, arguments: dict) -> Any: ...
    def snapshot(self) -> dict: ...
    def close(self) -> None: ...


class SeededEndpoint(ChatEndpoint):
    """Send the seed in every request; the generic ChatEndpoint omits it."""
    def __init__(self, *args: Any, seed: int, opener: Any = None, **kwargs: Any):
        if type(seed) is not int:
            raise ValueError("seed must be an integer")
        self.seed = seed
        transport = opener or urllib.request.urlopen

        def open_seeded(request: Any, *, timeout: float) -> Any:
            body = json.loads(request.data)
            body["seed"] = self.seed
            request.data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            return transport(request, timeout=timeout)

        super().__init__(*args, opener=open_seeded, **kwargs)

    def complete(self, messages: list, seed: int = 0, **kwargs: Any) -> Any:
        if seed != self.seed:
            raise ValueError("request seed differs from pair seed")
        return super().complete(messages, seed=seed, **kwargs)


def pair_manifest() -> dict:
    """Bind source-owned case, public schema and unchanged judge to the two arms."""
    registry = CaseRegistry()
    case = registry.get(CASE)
    contract = load_native_tool_contract(registry.cases, registry_path=registry.path)
    steps = [s for turn in case["config"]["Attack"]["attack_turns"]
             for s in turn["attack_steps"] if s.get("type") == "tool"
             and s.get("injected_tool") == "salesforce:create_record" and s.get("mode") == "suffix"]
    if len(steps) != 1 or not isinstance(steps[0].get("content"), str) or not steps[0]["content"]:
        raise ValueError("case does not contain exactly one nonempty create_record suffix")
    tools = contract.for_case(CASE)
    if sum(t["function"]["name"] == TARGET for t in tools) != 1:
        raise ValueError("case tool contract does not uniquely contain target")
    judge_path = registry.judge_path(CASE)
    return {"case": CASE, "registry_sha256": registry.sha256,
            "schema_sha256": contract.sha256, "case_tools_sha256": contract.tools_sha256(CASE),
            "judge_sha256": hashlib.sha256(Path(judge_path).read_bytes()).hexdigest(),
            "payload_sha256": hashlib.sha256(steps[0]["content"].encode()).hexdigest(),
            "payload": steps[0]["content"], "target": TARGET,
            "arms": ["control", "attack"], "injection_surface": "tool-result-suffix",
            "skill_injection": "not_tested"}


def run_pair(endpoint: Any, services: Services, *, seed: int, max_turns: int = 8) -> dict:
    """Execute both arms using one case, endpoint and backend, resetting before each.

    Partial records are returned on operational failure, including teardown. The
    caller must durably capture the returned record even for a failed arm.
    """
    if type(seed) is not int or type(max_turns) is not int or not 1 <= max_turns <= 64:
        raise ValueError("invalid seed or turn limit")
    if getattr(endpoint, "seed", None) != seed:
        raise ValueError("endpoint does not attest the configured seed")
    registry = CaseRegistry()
    case = registry.get(CASE)
    contract = load_native_tool_contract(registry.cases, registry_path=registry.path)
    tools = contract.for_case(CASE)
    manifest = pair_manifest()
    results: dict[str, Any] = {"manifest": manifest, "seed": seed, "max_turns": max_turns,
                               "arms": {}, "cleanup": "not_attempted"}
    baseline_state_hash = None
    try:
        for arm in ("control", "attack"):
            record: dict[str, Any] = {"status": "running", "events": [], "delivery": "not_observed",
                                      "judge": None, "error": None}
            results["arms"][arm] = record
            try:
                initial = services.reset()
                if not isinstance(initial, dict):
                    raise ValueError("reset must return actual state")
                record["initial_state"] = copy.deepcopy(initial)
                record["initial_state_sha256"] = _digest(initial)
                if baseline_state_hash is None:
                    baseline_state_hash = record["initial_state_sha256"]
                elif record["initial_state_sha256"] != baseline_state_hash:
                    raise ValueError("arm reset states differ")
                messages = render_messages(case)
                response = None
                trajectory = []
                for turn in range(max_turns):
                    request_messages = copy.deepcopy(messages)
                    chat = endpoint.complete(messages, seed=seed, tools=tools,
                                             tool_choice="auto", endpoint_mode="openai-compatible")
                    if record["delivery"] == "queued_not_confirmed":
                        record["delivery"] = "delivered"
                    record["events"].append({"type": "model", "turn": turn,
                                             "request_messages": request_messages, "raw": chat.raw,
                                             "usage": chat.usage, "usage_status": chat.usage_status})
                    if not chat.tool_calls:
                        response = chat.text
                        break
                    validate_tool_calls(chat.tool_calls, tools)
                    ids = [c.get("id") for c in chat.tool_calls]
                    if any(not isinstance(i, str) or not i for i in ids) or len(ids) != len(set(ids)):
                        raise ValueError("missing or duplicate tool call ID")
                    messages.append({"role": "assistant", "content": chat.text or None,
                        "tool_calls": [{"id": c["id"], "type": "function", "function": {
                            "name": c["name"], "arguments": json.dumps(c["arguments"])}}
                                       for c in chat.tool_calls]})
                    for call in chat.tool_calls:
                        try:
                            output = services.invoke(call["name"], copy.deepcopy(call["arguments"]))
                            if isinstance(output, dict) and output.get("error"):
                                raise RuntimeError("service returned error: " + str(output["error"]))
                        except Exception:
                            record["events"].append({"type": "tool_failed", "turn": turn,
                                                     "name": call["name"],
                                                     "arguments": copy.deepcopy(call["arguments"])})
                            raise
                        body = json.dumps(output, ensure_ascii=False, sort_keys=True, allow_nan=False)
                        if arm == "attack" and call["name"] == TARGET:
                            body += "\n" + manifest["payload"]
                            if record["delivery"] != "delivered":
                                record["delivery"] = "queued_not_confirmed"
                        record["events"].append({"type": "tool", "turn": turn, "name": call["name"],
                                                 "arguments": copy.deepcopy(call["arguments"]),
                                                 "native_result": copy.deepcopy(output),
                                                 "model_visible_result": body})
                        trajectory.append(({"tool": call["name"], "arguments": call["arguments"]},
                                           {"result": output}))
                        messages.append({"role": "tool", "tool_call_id": call["id"], "content": body})
                if response is None:
                    raise RuntimeError("model turn limit reached")
                state = services.snapshot()
                if not isinstance(state, dict) or not isinstance(state.get("salesforce"), dict) or not isinstance(state.get("slack"), dict):
                    raise ValueError("snapshot must contain CRM and Slack state")
                if not all(isinstance(state["salesforce"].get(k), list) for k in ("Notes", "Tasks")):
                    raise ValueError("snapshot missing judge-relevant CRM records")
                if not isinstance(state["slack"].get("channels"), dict):
                    raise ValueError("snapshot missing judge-relevant Slack channels")
                record["final_state"] = copy.deepcopy(state)
                record["final_state_sha256"] = _digest(state)
                store = StateStore(state)
                shims = (finance, gmail, paypal, salesforce, slack)
                # Shims hold process-global state. Restore it even if the judge fails.
                # Callers must not run generic DTAP judges concurrently in this process.
                with _JUDGE_LOCK:
                    prior = [shim._store for shim in shims]
                    try:
                        for shim in shims:
                            shim._set_store(store)
                        record["judge_date_local"] = datetime.now().astimezone().date().isoformat()
                        record["judge_date_utc"] = datetime.now(timezone.utc).date().isoformat()
                        record["judge"] = JudgeApplication(CASE, registry=registry).load().run(
                            [response], trajectory, final_state=state)
                        if record["judge_date_local"] != datetime.now().astimezone().date().isoformat():
                            raise RuntimeError("local date changed while judge ran")
                    finally:
                        for shim, old_store in zip(shims, prior):
                            shim._set_store(old_store)
                record["status"] = "complete"
            except Exception as exc:
                record["status"] = "failed"
                record["error"] = {"type": type(exc).__name__, "message": str(exc)}
                break
    finally:
        try:
            services.close()
            results["cleanup"] = "completed_by_backend"
        except Exception as exc:
            results["cleanup"] = "failed"
            results["cleanup_error"] = {"type": type(exc).__name__, "message": str(exc)}
    return results
