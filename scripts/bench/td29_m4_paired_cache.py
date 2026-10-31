#!/usr/bin/env python3
"""TD-29.M4: pair the original TD-4 tool call with its real REPL shadow dispatch.

Use a private one-slot champion sidecar.  Every case generates the original
free-form tool arguments, then `_ContextMixin._dispatch_tool` executes those
arguments unchanged and submits the typed shadow.  The benchmark drains that
shadow before advancing, so each receipt belongs to one exact pair.  `native`
is an explicit private-sidecar override; dispatch itself pins JSON for Q4.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

import src.config
from src.features import Features, reset_features, set_features
from src.registry.tool_registry import Tool, ToolCategory
from src.repl_environment.context import _ContextMixin
from src.typed_decisions.call_recorder import read_last_meta
from src.typed_decisions.measure import _live_primitives
from src.typed_decisions.shadow import drain_shadow
from src.typed_decisions.tool_args_pilot import (
    _DECODE_SEED,
    _FREE_FORM_N_TOKENS,
    _free_form_prompt,
    _extract_json_object,
    _parse_free_form_arguments,
    build_cases,
)


class _Registry:
    def __init__(self, case: Any) -> None:
        parameters = {
            name: {**spec, "required": name in case.parameters.get("required", ())}
            for name, spec in case.parameters["properties"].items()
        }
        self._tools = {
            case.tool: Tool(
                name=case.tool,
                description="TD-4 original case",
                category=ToolCategory.DATA,
                parameters=parameters,
                handler=None,
            )
        }

    def can_use_tool(self, role: str, tool_name: str, context: Any = None) -> bool:
        return tool_name in self._tools

    def invoke(self, tool_name: str, role: str, **kwargs: Any) -> dict[str, Any]:
        tool = self._tools[tool_name]
        arguments = {
            key: value
            for key, value in kwargs.items()
            if key not in {"caller_type", "chain_id", "chain_index", "context"}
        }
        errors = tool.validate_args(arguments)
        if errors:
            raise ValueError("; ".join(errors))
        return {"observed_arguments": arguments}


class _Repl(_ContextMixin):
    def __init__(self, primitives: Any, case: Any, role: str) -> None:
        self.llm_primitives = primitives
        self.tool_registry = _Registry(case)
        self.context = case.state
        self.role = role
        self._active_tool_chain_id = None
        self._active_tool_chain_index = 0


class _Capture:
    """Capture calls at the class seam so the shadow's primitives copy is seen."""

    def __init__(self, primitives: Any) -> None:
        self.cls = type(primitives)
        self.original = self.cls.llm_call
        self.calls: list[dict[str, Any]] = []
        self.stage = ""
        self.lock = threading.Lock()
        self.local = threading.local()
        self.original_send = httpx.Client.send

    def __enter__(self) -> "_Capture":
        original = self.original
        original_send = self.original_send
        capture = self

        def observed_send(
            client: httpx.Client, request: httpx.Request, *args: Any, **kwargs: Any
        ) -> Any:
            record = getattr(capture.local, "record", None)
            if record is not None and request.url.path == "/v1/chat/completions":
                record["wire_route"] = request.url.path
                record["wire_body_sha256"] = hashlib.sha256(request.content).hexdigest()
                record["wire_request_json"] = json.loads(request.content)
            return original_send(client, request, *args, **kwargs)

        def observed(instance: Any, prompt: str, **kwargs: Any) -> Any:
            record: dict[str, Any] = {
                "stage": capture.stage,
                "prompt": prompt,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "role": kwargs.get("role"),
                "n_tokens": kwargs.get("n_tokens"),
            }
            try:
                capture.local.record = record
                reply = original(instance, prompt, **kwargs)
                record["reply"] = reply
                return reply
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                capture.local.record = None
                meta = read_last_meta(instance)
                record["prompt_n"] = meta.get("prompt_n") if meta else None
                record["cache_n"] = meta.get("cache_n") if meta else None
                record["prompt_ms"] = meta.get("prompt_ms") if meta else None
                record["meta"] = meta
                with capture.lock:
                    capture.calls.append(record)

        self.cls.llm_call = observed
        httpx.Client.send = observed_send
        return self

    def __exit__(self, *_: Any) -> None:
        self.cls.llm_call = self.original
        httpx.Client.send = self.original_send


def _tokenize(url: str, prompt: str) -> list[int]:
    response = httpx.post(
        f"{url.rstrip('/')}/tokenize",
        json={"content": prompt, "add_special": True, "parse_special": True, "with_pieces": False},
        timeout=30.0,
    )
    response.raise_for_status()
    tokens = response.json().get("tokens")
    if not isinstance(tokens, list) or any(not isinstance(token, int) for token in tokens):
        raise ValueError("sidecar /tokenize returned no exact token ids")
    return tokens


def _prefix(left: bytes | list[int], right: bytes | list[int]) -> int:
    return next(
        (i for i, (a, b) in enumerate(zip(left, right)) if a != b), min(len(left), len(right))
    )


def _prefix_record(url: str, calls: list[dict[str, Any]]) -> dict[str, Any] | None:
    tool = next((call for call in calls if call["stage"] == "tool_call"), None)
    typed = next((call for call in calls if call["stage"] == "typed_shadow"), None)
    if tool is None or typed is None:
        return None
    rendered = []
    for call in (tool, typed):
        if call.get("wire_route") != "/v1/chat/completions":
            raise ValueError("actual chat wire request was not captured")
        response = httpx.post(
            f"{url.rstrip('/')}/apply-template",
            json=call["wire_request_json"],
            timeout=30.0,
        )
        response.raise_for_status()
        prompt = response.json().get("prompt")
        if not isinstance(prompt, str):
            raise ValueError("sidecar /apply-template did not return a text prompt")
        rendered.append(prompt)
    left, right = rendered
    left_ids, right_ids = _tokenize(url, left), _tokenize(url, right)
    caller_left, caller_right = tool["prompt"], typed["prompt"]
    common_bytes = _prefix(left.encode(), right.encode())
    return {
        "source": "same-sidecar /apply-template on captured /v1/chat/completions JSON",
        "common_bytes": common_bytes,
        "common_tokens": _prefix(left_ids, right_ids),
        "caller_common_bytes": _prefix(caller_left.encode(), caller_right.encode()),
        "rendered_tool_prompt": left,
        "rendered_typed_prompt": right,
        "rendered_tool_sha256": hashlib.sha256(left.encode()).hexdigest(),
        "rendered_typed_sha256": hashlib.sha256(right.encode()).hexdigest(),
        "tool_token_count_matches_server": len(left_ids) == tool["prompt_n"] + tool["cache_n"],
        "typed_token_count_matches_server": len(right_ids) == typed["prompt_n"] + typed["cache_n"],
        "tool_tokens": len(left_ids),
        "typed_tokens": len(right_ids),
        "first_tool_text": left[:100],
        "first_typed_text": right[:100],
        "first_tool_bytes_at_break": left.encode()[common_bytes : common_bytes + 32].decode(
            errors="backslashreplace"
        ),
        "first_typed_bytes_at_break": right.encode()[common_bytes : common_bytes + 32].decode(
            errors="backslashreplace"
        ),
    }


def _original_free_form_arguments(
    raw_text: str,
    parameters: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    """Keep parsed model kwargs even when validation rejects them.

    The pilot parser intentionally returns no arguments after a schema error.
    This dispatch observer must instead expose the original invalid object to
    the real registry and shadow, while retaining that error for pair validity.
    No parseable object means no dispatch; never invent empty arguments.
    """
    validated, error = _parse_free_form_arguments(raw_text, parameters)
    if validated is not None:
        return validated, error
    candidate = _extract_json_object(raw_text)
    if candidate is None:
        return None, error
    try:
        original = json.loads(candidate)
    except json.JSONDecodeError:
        return None, error
    return (original if isinstance(original, dict) else None), error


def run(
    *,
    url: str,
    output: Path,
    role: str,
    mode: str,
    limit: int = 18,
    deadline_epoch: float | None = None,
    stopfile: Path | None = None,
) -> dict[str, Any]:
    cases = build_cases()
    if len(cases) != 18 or not 1 <= limit <= 18:
        raise ValueError("TD-4 cohort must contain exactly 18 cases; limit is smoke-only")
    if mode not in {"json", "native"}:
        raise ValueError("mode must be json or native")
    primitives = _live_primitives({role: url})
    primitives.cache_prompt = True
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt_path = output.parent / "typed_decisions" / "decision_receipts.jsonl"
    prior = []
    if output.exists():
        prior = [json.loads(line) for line in output.read_text().splitlines() if line]
        if [row["case_id"] for row in prior] != [case.case_id for case in cases[: len(prior)]]:
            raise ValueError("existing receipt does not match the ordered TD-4 cohort")
    original_get_config = src.config.get_config
    import src.typed_decisions.tool_args_integration as integration

    class _PathsProxy:
        def __init__(self, real: Any) -> None:
            self.real = real
            self.artifacts_dir = output.parent

        def __getattr__(self, name: str) -> Any:
            return getattr(self.real, name)

    class _ConfigProxy:
        def __init__(self, real: Any) -> None:
            self.real = real
            self.paths = _PathsProxy(real.paths)

        def __getattr__(self, name: str) -> Any:
            return getattr(self.real, name)

    original_submit = integration.submit_tool_argument_shadow
    if mode == "native":

        def private_native_submit(**kwargs: Any) -> bool:
            kwargs["mode"] = "native"
            return original_submit(**kwargs)

        integration.submit_tool_argument_shadow = private_native_submit
    src.config.get_config = lambda: _ConfigProxy(original_get_config())
    set_features(Features(typed_decisions_tool_args=True))
    completed = len(prior)
    missing = sum(row.get("prefix") is None for row in prior)
    invalid = sum(not row.get("pair_valid", False) for row in prior)
    try:
        with output.open("a", encoding="utf-8") as sink:
            for case in cases[completed:limit]:
                if (deadline_epoch is not None and time.time() >= deadline_epoch) or (
                    stopfile is not None and stopfile.exists()
                ):
                    break
                repl = _Repl(primitives, case, role)
                before_receipts = (
                    len(receipt_path.read_text().splitlines()) if receipt_path.exists() else 0
                )
                with _Capture(primitives) as capture:
                    capture.stage = "tool_call"
                    tool_error = None
                    try:
                        raw = primitives.llm_call(
                            _free_form_prompt(case),
                            role=role,
                            n_tokens=_FREE_FORM_N_TOKENS,
                            temperature=0.0,
                            seed=_DECODE_SEED,
                        )
                        kwargs, tool_error = _original_free_form_arguments(
                            str(raw or ""), case.parameters
                        )
                    except Exception as exc:
                        kwargs = None
                        tool_error = f"{type(exc).__name__}: {exc}"
                    capture.stage = "typed_shadow"
                    dispatch_error = None
                    if kwargs is None:
                        dispatch_error = "not_dispatched: free-form arguments unavailable"
                    else:
                        try:
                            repl._dispatch_tool(case.tool, **kwargs)
                        except Exception as exc:
                            dispatch_error = f"{type(exc).__name__}: {exc}"
                    drain_shadow()
                receipt = None
                if receipt_path.exists():
                    lines = receipt_path.read_text().splitlines()
                    if len(lines) > before_receipts:
                        receipt = json.loads(lines[-1])
                prefix_error = None
                try:
                    pair = _prefix_record(url, capture.calls)
                except Exception as exc:
                    pair = None
                    prefix_error = f"{type(exc).__name__}: {exc}"
                if pair is None:
                    missing += 1
                paired_calls = [
                    call for call in capture.calls if call["stage"] in {"tool_call", "typed_shadow"}
                ]
                pair_valid = (
                    pair is not None
                    and pair["tool_token_count_matches_server"]
                    and pair["typed_token_count_matches_server"]
                    and len(paired_calls) >= 2
                    and all(
                        isinstance(call.get("prompt_n"), int)
                        and isinstance(call.get("cache_n"), int)
                        for call in paired_calls
                    )
                    and tool_error is None
                    and dispatch_error is None
                    and receipt is not None
                    and receipt.get("validation_result") == "accepted"
                )
                if not pair_valid:
                    invalid += 1
                row = {
                    "schema": "td29_m4_actual_shadow_pair.v1",
                    "recorded_at": datetime.now(timezone.utc).isoformat(),
                    "case_id": case.case_id,
                    "family": case.family,
                    "tool": case.tool,
                    "mode": mode,
                    "tool_error": tool_error,
                    "dispatch_error": dispatch_error,
                    "pair_valid": pair_valid,
                    "expected": case.expected,
                    "receipt": receipt,
                    "prefix": pair,
                    "prefix_error": prefix_error,
                    "calls": capture.calls,
                }
                sink.write(json.dumps(row, sort_keys=True, default=str) + "\n")
                sink.flush()
                completed += 1
    finally:
        integration.submit_tool_argument_shadow = original_submit
        src.config.get_config = original_get_config
        reset_features()
    return {
        "completed": completed,
        "expected": len(cases),
        "missing_pairs": missing,
        "invalid_pairs": invalid,
        "smoke_only": limit < 18,
        "partial": limit == 18 and completed < 18,
        "output": str(output),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--role", default="frontdoor")
    parser.add_argument("--mode", choices=("json", "native"), default="native")
    parser.add_argument("--smoke-first-case", action="store_true")
    parser.add_argument("--deadline-epoch", type=float)
    parser.add_argument("--stopfile", type=Path)
    args = parser.parse_args()
    result = run(
        url=args.server_url,
        output=args.output,
        role=args.role,
        mode=args.mode,
        limit=1 if args.smoke_first_case else 18,
        deadline_epoch=args.deadline_epoch,
        stopfile=args.stopfile,
    )
    print(json.dumps(result, sort_keys=True))
    if result["invalid_pairs"]:
        return 1
    return 75 if result["partial"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
