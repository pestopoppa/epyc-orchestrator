#!/usr/bin/env python3
"""Dry-run-first AXA-2 live cutover / continuity harness.

Default mode writes an operator bundle only. ``--execute`` talks to already
running llama-server endpoints and records a v1 re-prefill cutover smoke plus a
CPU-vs-GPU continuity comparison. The harness never starts servers, never
builds kernels, and never touches production v6.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.gpu_lease import GpuLeaseManager

_TELEPORT_PATH = REPO_ROOT / "src" / "llm_primitives" / "teleport.py"
_TELEPORT_SPEC = importlib.util.spec_from_file_location("axa2_live_cutover_teleport", _TELEPORT_PATH)
if _TELEPORT_SPEC is None or _TELEPORT_SPEC.loader is None:
    raise RuntimeError(f"failed to load teleport policy module from {_TELEPORT_PATH}")
_TELEPORT_MODULE = importlib.util.module_from_spec(_TELEPORT_SPEC)
sys.modules[_TELEPORT_SPEC.name] = _TELEPORT_MODULE
_TELEPORT_SPEC.loader.exec_module(_TELEPORT_MODULE)

TeleportInputs = _TELEPORT_MODULE.TeleportInputs
TeleportPolicy = _TELEPORT_MODULE.TeleportPolicy
decide_teleport = _TELEPORT_MODULE.decide_teleport


SCHEMA = "epyc.axa2_live_cutover_bundle.v1"
DEFAULT_OUTPUT_BASE = REPO_ROOT / "orchestration" / "reports"
DEFAULT_EXECUTION_OUTPUT_BASE = DEFAULT_OUTPUT_BASE / "axa2_live_cutover_runs"
DEFAULT_PROMPT = (
    "Write a deterministic two sentence validation note. "
    "Mention AXA-2 once and end with the word done."
)

Runner = Callable[..., subprocess.CompletedProcess[str]]


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def canonical_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def jsonable(value: Any) -> Any:
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    return value


def run_capture(
    argv: list[str],
    *,
    cwd: Path | None = None,
    runner: Runner = subprocess.run,
    timeout: float = 20.0,
) -> dict[str, Any]:
    try:
        proc = runner(
            argv,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "argv": argv,
            "cwd": str(cwd) if cwd else None,
            "ok": False,
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
        }
    return {
        "argv": argv,
        "cwd": str(cwd) if cwd else None,
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def git_state(repo: Path = REPO_ROOT, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    head = run_capture(["git", "rev-parse", "HEAD"], cwd=repo, runner=runner)
    branch = run_capture(["git", "branch", "--show-current"], cwd=repo, runner=runner)
    status = run_capture(["git", "status", "--porcelain=v1", "--untracked-files=no"], cwd=repo, runner=runner)
    return {
        "path": str(repo),
        "branch": branch["stdout"].strip() if branch["ok"] else None,
        "head": head["stdout"].strip() if head["ok"] else None,
        "tracked_dirty_lines": len([line for line in status["stdout"].splitlines() if line.strip()]) if status["ok"] else None,
        "commands": {"head": head, "branch": branch, "status": status},
    }


def process_snapshot(*, runner: Runner = subprocess.run) -> dict[str, Any]:
    return {
        "captured_at": utc_now(),
        "ps": run_capture(["ps", "-eo", "pid=,comm=,args="], runner=runner, timeout=10),
        "rocm_smi_showpids": run_capture(["rocm-smi", "--showpids"], runner=runner, timeout=20),
    }


def read_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file:
        return Path(args.prompt_file).expanduser().read_text(encoding="utf-8")
    return args.prompt or DEFAULT_PROMPT


def parse_csv_set(value: str | None) -> frozenset[str]:
    if not value:
        return frozenset()
    return frozenset(item.strip() for item in value.split(",") if item.strip())


def build_policy(args: argparse.Namespace) -> TeleportPolicy:
    return TeleportPolicy(
        enabled=args.policy_enabled,
        quant_policy=args.quant_policy,
        long_running_trigger_tokens=args.long_running_trigger_tokens,
        min_resident_remaining_tokens=args.min_resident_remaining_tokens,
        min_cold_remaining_tokens=args.min_cold_remaining_tokens,
        min_speedup=args.min_speedup,
        allowed_roles=parse_csv_set(args.role_allowlist),
        allowed_quant_change_roles=parse_csv_set(args.quant_change_role_allowlist),
    )


def build_inputs(args: argparse.Namespace) -> TeleportInputs:
    return TeleportInputs(
        role=args.role,
        generated_tokens=args.generated_tokens,
        estimated_remaining_tokens=args.estimated_remaining_tokens,
        cpu_tps=args.cpu_tps,
        gpu_tps=args.gpu_tps,
        gpu_available=args.gpu_available,
        gpu_resident=args.gpu_resident,
        cpu_quant=args.cpu_quant,
        gpu_quant=args.gpu_quant,
        catch_up_supported=False,
        metadata={
            "trace_id": args.trace_id,
            "protocol_note": "v1 re-prefill cutover only; no spec-dec catch-up",
        },
    )


def request_payload(prompt: str, *, n_predict: int, seed: int) -> dict[str, Any]:
    return {
        "prompt": prompt,
        "n_predict": n_predict,
        "seed": seed,
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
        "cache_prompt": False,
        "stream": False,
    }


def _endpoint(base_url: str) -> str:
    return base_url.rstrip("/") + "/completion"


def post_completion(base_url: str, payload: dict[str, Any], *, timeout_s: float) -> dict[str, Any]:
    data = canonical_json(payload).encode("utf-8")
    req = urllib.request.Request(
        _endpoint(base_url),
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as response:
            body = response.read().decode("utf-8", errors="replace")
            status = response.status
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return {
            "ok": False,
            "status": exc.code,
            "elapsed_s": time.monotonic() - started,
            "body": body,
            "error": str(exc),
        }
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return {
            "ok": False,
            "status": None,
            "elapsed_s": time.monotonic() - started,
            "body": "",
            "error": str(exc),
        }
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        parsed = None
    return {
        "ok": True,
        "status": status,
        "elapsed_s": time.monotonic() - started,
        "body": body,
        "json": parsed,
    }


def extract_text(response: dict[str, Any]) -> str:
    parsed = response.get("json")
    if isinstance(parsed, dict):
        if isinstance(parsed.get("content"), str):
            return parsed["content"]
        choices = parsed.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                if isinstance(first.get("text"), str):
                    return first["text"]
                message = first.get("message")
                if isinstance(message, dict) and isinstance(message.get("content"), str):
                    return message["content"]
    return ""


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def first_char_divergence(left: str, right: str) -> int | None:
    for idx, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return idx
    if len(left) == len(right):
        return None
    return min(len(left), len(right))


def operator_output_preamble(args: argparse.Namespace) -> str:
    if args.execution_output:
        execution_dir = Path(args.execution_output).expanduser().resolve()
        return f'export AXA2_EXEC_OUTPUT="${{AXA2_EXEC_OUTPUT:-{execution_dir}}}"'
    return "\n".join(
        [
            ': "${AXA2_RUN_ID:=axa2-live-cutover-$(date -u +%Y%m%dT%H%M%SZ)}"',
            f'export AXA2_EXECUTION_BASE="${{AXA2_EXECUTION_BASE:-{DEFAULT_EXECUTION_OUTPUT_BASE}}}"',
            'export AXA2_EXEC_OUTPUT="${AXA2_EXEC_OUTPUT:-${AXA2_EXECUTION_BASE}/${AXA2_RUN_ID}}"',
        ]
    )


def write_operator_script(output_dir: Path, args: argparse.Namespace) -> None:
    script_path = output_dir / "operator_run.sh"
    argv = [
        "python3",
        str(Path(__file__).resolve()),
        "--execute",
        "--output",
        "__AXA2_EXEC_OUTPUT__",
        "--policy-enabled" if args.policy_enabled else "",
        "--role",
        args.role,
        "--quant-policy",
        args.quant_policy,
        "--cpu-quant",
        args.cpu_quant,
        "--gpu-quant",
        args.gpu_quant,
        "--generated-tokens",
        str(args.generated_tokens),
        "--estimated-remaining-tokens",
        str(args.estimated_remaining_tokens),
        "--cpu-tps",
        str(args.cpu_tps),
        "--gpu-tps",
        str(args.gpu_tps),
        "--cpu-prefix-tokens",
        str(args.cpu_prefix_tokens),
        "--gpu-suffix-tokens",
        str(args.gpu_suffix_tokens),
        "--continuity-tokens",
        str(args.continuity_tokens),
        "--seed",
        str(args.seed),
    ]
    if args.role_allowlist:
        argv.extend(["--role-allowlist", args.role_allowlist])
    if args.quant_change_role_allowlist:
        argv.extend(["--quant-change-role-allowlist", args.quant_change_role_allowlist])
    argv = [item for item in argv if item]
    invocation = shlex.join(argv).replace("__AXA2_EXEC_OUTPUT__", '"$AXA2_EXEC_OUTPUT"')
    script = "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "",
            "# AXA-2 operator gate: this talks to already-running servers only.",
            "# It does not start servers, build kernels, restart AutoPilot, or touch production v6.",
            operator_output_preamble(args),
            'CPU_URL="${CPU_URL:?set CPU_URL to the CPU llama-server base URL}"',
            'GPU_URL="${GPU_URL:?set GPU_URL to the GPU llama-server base URL}"',
            invocation + ' --cpu-url "$CPU_URL" --gpu-url "$GPU_URL"',
            "",
        ]
    )
    script_path.write_text(script, encoding="utf-8")
    script_path.chmod(0o755)


def dry_summary(args: argparse.Namespace, output_dir: Path, prompt: str) -> dict[str, Any]:
    policy = build_policy(args)
    inputs = build_inputs(args)
    decision = decide_teleport(policy, inputs)
    return {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "status": "prepared_no_inference",
        "execute": False,
        "output_dir": str(output_dir),
        "operator_execution": {
            "mode": "static" if args.execution_output else "dynamic_timestamped",
            "default_output": str(Path(args.execution_output).expanduser().resolve())
            if args.execution_output
            else "${AXA2_EXECUTION_BASE}/${AXA2_RUN_ID}",
            "output_base": None if args.execution_output else str(DEFAULT_EXECUTION_OUTPUT_BASE),
        },
        "no_inference": True,
        "no_server_start": True,
        "production_v6_touch_authorized": False,
        "trace_id": args.trace_id,
        "prompt_sha256": sha256_text(prompt),
        "policy": jsonable(asdict(policy)),
        "inputs": jsonable(asdict(inputs)),
        "decision": jsonable(asdict(decision)),
        "required_live_artifacts": [
            "preflight/process_snapshot.json",
            "policy_decision.json",
            "events.jsonl",
            "requests/cpu_prefix.request.json",
            "responses/cpu_prefix.response.json",
            "requests/gpu_suffix.request.json",
            "responses/gpu_suffix.response.json",
            "responses/continuity_cpu.response.json",
            "responses/continuity_gpu.response.json",
            "postflight/process_snapshot.json",
        ],
        "environment": {
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "repo": git_state(),
        },
    }


def write_dry_bundle(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt = read_prompt(args)
    summary = dry_summary(args, output_dir, prompt)
    (output_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    (output_dir / "summary.json").write_text(canonical_json(summary), encoding="utf-8")
    (output_dir / "policy_decision.json").write_text(canonical_json(summary["decision"]), encoding="utf-8")
    write_operator_script(output_dir, args)
    return summary


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value), encoding="utf-8")


def append_event(path: Path, event: str, payload: dict[str, Any] | None = None) -> None:
    row = {"ts": utc_now(), "event": event, "payload": payload or {}}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def execute_bundle(args: argparse.Namespace) -> dict[str, Any]:
    if not args.cpu_url or not args.gpu_url:
        raise SystemExit("--execute requires --cpu-url and --gpu-url")

    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ("preflight", "postflight", "requests", "responses"):
        (output_dir / subdir).mkdir(exist_ok=True)

    prompt = read_prompt(args)
    policy = build_policy(args)
    inputs = build_inputs(args)
    decision = decide_teleport(policy, inputs)
    events_path = output_dir / "events.jsonl"
    if events_path.exists():
        events_path.unlink()

    write_json(output_dir / "preflight" / "process_snapshot.json", process_snapshot())
    write_json(output_dir / "policy_decision.json", jsonable(asdict(decision)))
    append_event(events_path, "teleport_candidate", {"decision": jsonable(asdict(decision))})

    status = "policy_declined"
    cpu_prefix_text = ""
    gpu_suffix_text = ""
    continuity_cpu_text = ""
    continuity_gpu_text = ""
    lease_released = None

    if decision.should_cutover:
        lease = GpuLeaseManager()
        acquire = lease.acquire(args.trace_id, reason="axa2_live_cutover_smoke", wait=False)
        append_event(events_path, "gpu_lease_acquired", {"acquired": acquire.acquired, "reason": acquire.reason})
        if not acquire.acquired or acquire.handle is None:
            status = "lease_unavailable"
            append_event(events_path, "fallback", {"reason": status})
        else:
            try:
                cpu_prefix_payload = request_payload(prompt, n_predict=args.cpu_prefix_tokens, seed=args.seed)
                write_json(output_dir / "requests" / "cpu_prefix.request.json", cpu_prefix_payload)
                cpu_prefix = post_completion(args.cpu_url, cpu_prefix_payload, timeout_s=args.timeout_s)
                write_json(output_dir / "responses" / "cpu_prefix.response.json", cpu_prefix)
                cpu_prefix_text = extract_text(cpu_prefix)

                if not cpu_prefix.get("ok"):
                    status = "cpu_prefix_failed"
                    append_event(events_path, "fallback", {"reason": status})
                else:
                    gpu_prompt = prompt + cpu_prefix_text
                    append_event(events_path, "gpu_prefill_start", {"prefix_chars": len(gpu_prompt)})
                    gpu_suffix_payload = request_payload(gpu_prompt, n_predict=args.gpu_suffix_tokens, seed=args.seed)
                    write_json(output_dir / "requests" / "gpu_suffix.request.json", gpu_suffix_payload)
                    gpu_suffix = post_completion(args.gpu_url, gpu_suffix_payload, timeout_s=args.timeout_s)
                    write_json(output_dir / "responses" / "gpu_suffix.response.json", gpu_suffix)
                    append_event(
                        events_path,
                        "gpu_prefill_end",
                        {"ok": gpu_suffix.get("ok"), "elapsed_s": gpu_suffix.get("elapsed_s")},
                    )
                    gpu_suffix_text = extract_text(gpu_suffix)
                    if not gpu_suffix.get("ok"):
                        status = "gpu_suffix_failed"
                        append_event(events_path, "fallback", {"reason": status})
                    else:
                        append_event(
                            events_path,
                            "cutover",
                            {
                                "cpu_prefix_sha256": sha256_text(cpu_prefix_text),
                                "gpu_suffix_sha256": sha256_text(gpu_suffix_text),
                            },
                        )
                        status = "executed"
            finally:
                lease_released = acquire.handle.release()
                append_event(events_path, "lease_released", {"released": lease_released})
    else:
        append_event(events_path, "fallback", {"reason": decision.reason})

    continuity_payload = request_payload(prompt, n_predict=args.continuity_tokens, seed=args.seed)
    write_json(output_dir / "requests" / "continuity.request.json", continuity_payload)
    continuity_cpu = post_completion(args.cpu_url, continuity_payload, timeout_s=args.timeout_s)
    continuity_gpu = post_completion(args.gpu_url, continuity_payload, timeout_s=args.timeout_s)
    write_json(output_dir / "responses" / "continuity_cpu.response.json", continuity_cpu)
    write_json(output_dir / "responses" / "continuity_gpu.response.json", continuity_gpu)
    continuity_cpu_text = extract_text(continuity_cpu)
    continuity_gpu_text = extract_text(continuity_gpu)

    continuity = {
        "seed": args.seed,
        "n_predict": args.continuity_tokens,
        "cpu_output_sha256": sha256_text(continuity_cpu_text),
        "gpu_output_sha256": sha256_text(continuity_gpu_text),
        "first_char_divergence": first_char_divergence(continuity_cpu_text, continuity_gpu_text),
        "cpu_chars": len(continuity_cpu_text),
        "gpu_chars": len(continuity_gpu_text),
    }
    write_json(output_dir / "postflight" / "process_snapshot.json", process_snapshot())

    summary = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "status": status,
        "execute": True,
        "output_dir": str(output_dir),
        "production_v6_touch_authorized": False,
        "no_server_start": True,
        "trace_id": args.trace_id,
        "policy": jsonable(asdict(policy)),
        "inputs": jsonable(asdict(inputs)),
        "decision": jsonable(asdict(decision)),
        "lease_released": lease_released,
        "cutover": {
            "cpu_prefix_sha256": sha256_text(cpu_prefix_text),
            "gpu_suffix_sha256": sha256_text(gpu_suffix_text),
            "combined_output_sha256": sha256_text(cpu_prefix_text + gpu_suffix_text),
            "cpu_prefix_chars": len(cpu_prefix_text),
            "gpu_suffix_chars": len(gpu_suffix_text),
        },
        "continuity": continuity,
        "artifact_grade": "observation_until_p_gpu_1_ratified",
        "environment": {
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "repo": git_state(),
        },
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_BASE / f"axa2_live_cutover_bundle_{utc_stamp()}"))
    parser.add_argument(
        "--execution-output",
        default="",
        help=(
            "Optional static artifact directory used by generated operator_run.sh. "
            "If omitted, operator_run.sh creates a fresh timestamped directory under "
            f"{DEFAULT_EXECUTION_OUTPUT_BASE}."
        ),
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--trace-id", default=f"axa2-live-{utc_stamp()}")
    parser.add_argument("--prompt")
    parser.add_argument("--prompt-file")
    parser.add_argument("--cpu-url", default="")
    parser.add_argument("--gpu-url", default="")
    parser.add_argument("--timeout-s", type=float, default=240.0)
    parser.add_argument("--policy-enabled", action="store_true")
    parser.add_argument("--role", default="architect_general")
    parser.add_argument("--role-allowlist", default="architect_general")
    parser.add_argument("--quant-policy", default="same_quant_only")
    parser.add_argument("--quant-change-role-allowlist", default="")
    parser.add_argument("--long-running-trigger-tokens", type=int, default=128)
    parser.add_argument("--min-resident-remaining-tokens", type=int, default=150)
    parser.add_argument("--min-cold-remaining-tokens", type=int, default=350)
    parser.add_argument("--min-speedup", type=float, default=1.05)
    parser.add_argument("--generated-tokens", type=int, default=200)
    parser.add_argument("--estimated-remaining-tokens", type=int, default=500)
    parser.add_argument("--cpu-tps", type=float, default=20.0)
    parser.add_argument("--gpu-tps", type=float, default=44.0)
    parser.add_argument("--gpu-available", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gpu-resident", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cpu-quant", default="q4_k_m")
    parser.add_argument("--gpu-quant", default="q4_k_m")
    parser.add_argument("--cpu-prefix-tokens", type=int, default=64)
    parser.add_argument("--gpu-suffix-tokens", type=int, default=128)
    parser.add_argument("--continuity-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = execute_bundle(args) if args.execute else write_dry_bundle(args)
    print(summary["output_dir"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
