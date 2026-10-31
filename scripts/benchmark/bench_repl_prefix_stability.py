#!/usr/bin/env python3
"""RTE prefix cache replay of *natively captured* REPL turns.

Capture using ``await capture_turn(ctx, role)`` at the native graph turn boundary;
write the returned records as a JSON array. The caller must advance graph state
normally between turns. Captures contain prompt contents and may be sensitive.
Replay needs an exclusive, single-slot server and does not assess task quality.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import dataclasses
import hashlib
import inspect
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def source_hashes():
    paths = sorted(p for folder in (ROOT / "src", ROOT / "orchestration" / "prompts",
                                  ROOT / "orchestration" / "tools", ROOT / "scripts" / "server")
                   for p in folder.rglob("*") if p.is_file() and not p.is_symlink()
                   and "__pycache__" not in p.parts and not p.name.endswith(".pyc"))
    paths.append(Path(__file__).resolve())
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


async def capture_turn(ctx, role):
    """Capture one real ``_execute_turn`` at its inference boundary.

    Runs its actual primitive and REPL. This is process-local instrumentation:
    never call concurrently with other graph turns. It refuses transformations
    that prevent an unambiguous replacement of the native builder span.
    """
    from src.features import features
    from src.graph import helpers
    from src.prompt_builders.builder import PromptBuilder
    from src.security.control_tokens import neutralise_control_tokens

    original_builder = PromptBuilder.build_root_lm_prompt
    original_call = ctx.deps.primitives.llm_call
    observed = {}

    def builder(self, *args, **kwargs):
        if observed.get("builder") is not None or kwargs.get("as_structured"):
            raise ValueError("unsupported multiple or structured builder invocation")
        inputs = dict(kwargs)
        if args:
            raise ValueError("unexpected positional builder inputs")
        structured = original_builder(self, **kwargs, as_structured=True)
        raw = dataclasses.asdict(structured)
        native_order = bool(features().prefix_stable_order)
        native = structured.to_string(prefix_stable=native_order)
        config = dataclasses.asdict(self.config)
        config["style"] = self.config.style.value
        observed["builder"] = {"inputs": inputs, "structured": raw, "config": config,
                               "counter_tail": bool(features().repl_turn_counter_tail),
                               "deferred": bool(features().deferred_tool_results),
                               "native": native, "native_order": native_order}
        return native

    def note_prompt(prompt):
        if "prompt" in observed:
            raise ValueError("multiple inference calls in one turn are unsupported")
        observed["prompt"] = prompt

    if inspect.iscoroutinefunction(original_call):
        async def call(prompt, *args, **kwargs):
            note_prompt(prompt)
            value = await original_call(prompt, *args, **kwargs)
            observed["generated"] = value
            return value
    else:
        def call(prompt, *args, **kwargs):
            note_prompt(prompt)
            value = original_call(prompt, *args, **kwargs)
            observed["generated"] = value
            return value

    components = {}
    component_names = ("_auto_gather_context", "_workspace_prompt_block",
                       "_session_log_prompt_block", "_budget_pressure_warnings",
                       "_answer_force_due")
    def tap(name, original):
        def wrapped(*args, **kwargs):
            if name in components:
                raise ValueError(f"multiple calls to {name}")
            value = original(*args, **kwargs)
            components[name] = value
            return value
        return wrapped
    with ExitStack() as stack:
        stack.enter_context(patch.object(PromptBuilder, "build_root_lm_prompt", builder))
        stack.enter_context(patch.object(ctx.deps.primitives, "llm_call", call))
        for name in component_names:
            stack.enter_context(patch.object(helpers, name, tap(name, getattr(helpers, name))))
        output, error, final, artifacts = await helpers._execute_turn(ctx, role)
    if set(components) != set(component_names):
        raise ValueError("native composition component was not observed")
    if "builder" not in observed or "prompt" not in observed or not isinstance(observed.get("generated"), str):
        raise ValueError("native builder or inference boundary was not reached")
    from src.prompt_builders.types import RootLMPrompt
    b = observed["builder"]
    if b["inputs"]["turn"] != ctx.state.turns - 1:
        raise ValueError("turn index mismatch")
    if b["inputs"]["turn"] and b["inputs"].get("corpus_context"):
        raise ValueError("corpus context on later turn")
    native_base = neutralise_control_tokens(b["native"])
    prompt = observed["prompt"]
    if prompt.count(native_base) != 1:
        raise ValueError("native builder span transformed or ambiguous")
    for name in component_names[:-1]:
        value = components[name]
        if not isinstance(value, str) or (value and neutralise_control_tokens(value) not in prompt):
            raise ValueError(f"native component {name} missing from prompt")
    if type(components["_answer_force_due"]) is not bool:
        raise ValueError("unknown force-answer state")
    structured = RootLMPrompt(**b["structured"])
    prompts = {}
    for name, stable in (("legacy", False), ("stable", True)):
        base = neutralise_control_tokens(structured.to_string(prefix_stable=stable))
        prompts[name] = prompt.replace(native_base, base, 1)
    native_name = "stable" if b["native_order"] else "legacy"
    if prompts[native_name] != prompt:
        raise ValueError("native prompt mismatch")
    record = {"schema": 1, "source_hashes": source_hashes(), "role": str(role),
              "turn": b["inputs"]["turn"], "builder_inputs": b["inputs"],
              "structured": b["structured"], "builder_config": b["config"],
              "counter_tail": b["counter_tail"], "deferred": b["deferred"], "components": components, "native_prompt": prompt,
              "native_order": native_name, "prompts": prompts,
              "generated": observed["generated"],
              "execution": {"output": output, "error": error, "final": final},
              "artifacts": artifacts}
    record["sha256"] = digest(record)
    return record, (output, error, final, artifacts)


def validate_records(records):
    from src.prompt_builders import builder as builder_module
    from src.prompt_builders.types import PromptConfig, PromptStyle, RootLMPrompt
    from src.security.control_tokens import neutralise_control_tokens

    if not isinstance(records, list) or not records:
        raise ValueError("at least one captured turn is required")
    current = source_hashes()
    for name, module in tuple(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if path and (name == "src" or name.startswith("src.")):
            if not Path(path).resolve().is_relative_to(ROOT.resolve()):
                raise ValueError(f"foreign native module: {name}")
    for turn, item in enumerate(records):
        if not isinstance(item, dict) or item.get("schema") != 1:
            raise ValueError("invalid capture schema")
        body = dict(item)
        claimed = body.pop("sha256", None)
        if claimed != digest(body) or item.get("source_hashes") != current:
            raise ValueError("capture digest or native source mismatch")
        if item.get("turn") != turn or item["builder_inputs"].get("turn") != turn:
            raise ValueError("noncontiguous turn sequence")
        if turn and item["builder_inputs"].get("corpus_context"):
            raise ValueError("corpus context on later turn")
        if turn and records[turn - 1]["execution"]["final"]:
            raise ValueError("capture continues after final answer")
        if turn and item["builder_inputs"].get("last_output") != records[turn - 1]["execution"]["output"]:
            raise ValueError("last output differs from native execution")
        # Errors and spilled/compressed outputs can be transformed by the live caller;
        # without a binding to those transformations the capture is not comparable.
        if turn and (item["builder_inputs"].get("last_error") or "") != (records[turn - 1]["execution"]["error"] or ""):
            raise ValueError("last error differs from native execution")
        config = dict(item["builder_config"])
        config["style"] = PromptStyle(config["style"])
        if config["style"] != PromptStyle.MINIMAL or type(item["counter_tail"]) is not bool or type(item["deferred"]) is not bool:
            raise ValueError("invalid native builder configuration")
        with patch.object(builder_module, "_get_features", lambda: SimpleNamespace(
            repl_turn_counter_tail=item["counter_tail"], deferred_tool_results=item["deferred"])):
            rebuilt = builder_module.PromptBuilder(PromptConfig(**config)).build_root_lm_prompt(
                **item["builder_inputs"], as_structured=True)
        if dataclasses.asdict(rebuilt) != item["structured"]:
            raise ValueError("raw builder inputs do not match structured sections")
        base = RootLMPrompt(**item["structured"])
        name = item["native_order"]
        if name not in ("legacy", "stable"):
            raise ValueError("unknown native ordering")
        native_base = neutralise_control_tokens(base.to_string(prefix_stable=name == "stable"))
        native = item["native_prompt"]
        required = ("_auto_gather_context", "_workspace_prompt_block",
                    "_session_log_prompt_block", "_budget_pressure_warnings")
        components = item["components"]
        if set(components) != set(required) | {"_answer_force_due"} or type(components["_answer_force_due"]) is not bool:
            raise ValueError("incomplete native composition")
        for key in required:
            value = components[key]
            if not isinstance(value, str) or (value and neutralise_control_tokens(value) not in native):
                raise ValueError(f"unbound native component: {key}")
        if native.count(native_base) != 1:
            raise ValueError("native span ambiguous")
        expected = {n: native.replace(native_base, neutralise_control_tokens(
            base.to_string(prefix_stable=s)), 1) for n, s in (("legacy", False), ("stable", True))}
        if item["prompts"] != expected:
            raise ValueError("rendered prompts do not match native components")
    return records


def exchange(client, method, url, journal, context, payload=None):
    """Durably record an attempt before I/O and its response or unknown outcome."""
    attempt = dict(context, event="request", method=method, url=url, payload=payload)
    journal(attempt)
    try:
        if method == "GET":
            response = client.get(url)
        else:
            response = client.post(url, json=payload) if payload is not None else client.post(url)
    except Exception as exc:
        journal(dict(context, event="response", outcome="unknown", error=repr(exc)))
        raise
    try:
        body = response.json()
    except Exception as exc:
        journal(dict(context, event="response", status=response.status_code,
                     outcome="non_json", raw_text=response.text, error=repr(exc)))
        raise ValueError("server returned non-JSON response") from exc
    journal(dict(context, event="response", status=response.status_code,
                 outcome="received", native_json=body))
    response.raise_for_status()
    return body


def idle_slot(client, url, journal, order, phase):
    slots = exchange(client, "GET", url + "/slots", journal,
                     {"arm": order, "phase": phase})
    if not isinstance(slots, list) or len(slots) != 1 or slots[0].get("id") != 0:
        raise ValueError("exclusive single slot 0 required")
    if type(slots[0].get("is_processing")) is not bool or slots[0]["is_processing"]:
        raise ValueError("slot processing state unknown or busy")


def reset_slot(client, url, journal, order):
    idle_slot(client, url, journal, order, "before_reset")
    receipt = exchange(client, "POST", url + "/slots/0?action=erase", journal,
                       {"arm": order, "phase": "reset"})
    if (not isinstance(receipt, dict) or "error" in receipt
            or type(receipt.get("id_slot")) is not int or receipt["id_slot"] != 0
            or type(receipt.get("n_erased")) is not int or receipt["n_erased"] < 0):
        raise ValueError("invalid slot erase receipt")
    idle_slot(client, url, journal, order, "after_reset")


def metric(data):
    timings = data.get("timings")
    if not isinstance(timings, dict):
        raise ValueError("missing timings")
    total = data.get("tokens_evaluated")
    cached = timings.get("cache_n")
    processed = timings.get("prompt_n")
    ms = timings.get("prompt_ms")
    if any(type(v) is not int or v < 0 for v in (total, cached, processed)):
        raise ValueError("missing or invalid token metrics")
    if total <= 0 or cached + processed != total:
        raise ValueError("inconsistent token metrics")
    if isinstance(ms, bool) or not isinstance(ms, (int, float)) or not math.isfinite(ms) or ms < 0:
        raise ValueError("missing or invalid prompt timing")
    return {"total": total, "cached": cached, "processed": processed, "prompt_ms": ms}


def run_pair(client, server, records, n_predict, journal):
    """Replay matched native captures. The journal records every exposure."""
    order = None
    turn = None
    try:
        validate_records(records)
        if type(n_predict) is not int or n_predict < 1:
            raise ValueError("n_predict must be positive")
        url = server.rstrip("/")
        results = {}
        journal({"event": "pair_start", "capture_hash": digest(records),
                 "kind": "matched_native_replay"})
        for order in ("legacy", "stable"):
            turn = None
            reset_slot(client, url, journal, order)
            rows = []
            for record in records:
                turn = record["turn"]
                prompt = record["prompts"][order]
                payload = {"prompt": prompt, "n_predict": n_predict, "temperature": 0.0,
                           "cache_prompt": True, "id_slot": 0, "stop": ["\n```\n"]}
                context = {"arm": order, "phase": "completion", "turn": turn,
                           "capture_sha256": record["sha256"],
                           "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}
                start = time.monotonic()
                data = exchange(client, "POST", url + "/completion", journal, context, payload)
                row = metric(data)
                if not rows and row["cached"] != 0:
                    raise ValueError("slot reset did not yield a cold first turn")
                row.update(turn=turn, capture_sha256=record["sha256"],
                           http_ms=(time.monotonic() - start) * 1000)
                journal(dict(context, event="validated_metrics", metrics=row))
                rows.append(row)
            results[order] = rows
        result = {"capture_hash": digest(records), "arms": results}
        journal({"event": "pair_complete", "capture_hash": result["capture_hash"]})
        return result
    except Exception as exc:
        journal({"event": "pair_failed", "arm": order, "turn": turn, "error": repr(exc)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures", required=True, type=Path)
    parser.add_argument("--server", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--n-predict", type=int, default=8)
    args = parser.parse_args()
    import httpx

    journal_path = Path(str(args.output) + ".journal.jsonl")
    # Exclusive private files exist before the first request. An interrupted run
    # retains an in-progress manifest and the journal's last durable boundary.
    try:
        fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except Exception as exc:
        print(f"RTE pair failed before journal creation: {exc}", file=sys.stderr)
        return 1
    with os.fdopen(fd, "w") as manifest:
        def write_manifest(value):
            manifest.seek(0)
            manifest.truncate()
            json.dump(value, manifest, indent=2)
            manifest.write("\n")
            manifest.flush()
            os.fsync(manifest.fileno())

        write_manifest({"status": "in_progress", "journal": str(journal_path),
                        "kind": "matched_native_replay"})
        partial = []
        try:
            journal_fd = os.open(journal_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND, 0o600)
            with os.fdopen(journal_fd, "w") as journal_file:
                def journal(event):
                    journal_file.write(json.dumps(event, ensure_ascii=False) + "\n")
                    journal_file.flush()
                    os.fsync(journal_file.fileno())
                    if event.get("event") == "validated_metrics":
                        partial.append(event)
                records = json.loads(args.captures.read_text())
                with httpx.Client(timeout=120.0) as client:
                    result = run_pair(client, args.server, records, args.n_predict, journal)
            write_manifest({"status": "complete", "kind": "matched_native_replay",
                            "journal": str(journal_path), "result": result})
        except Exception as exc:
            write_manifest({"status": "failed", "kind": "matched_native_replay",
                            "journal": str(journal_path), "error": repr(exc),
                            "partial_observations": partial})
            print(f"RTE pair failed: {exc}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
