"""Case runner: executes a DTAP case against an endpoint (live) or fixture stub
(dry-run), captures an immutable trace, applies the deterministic final-state
judge, and emits typed failure outcomes.
"""
from __future__ import annotations

import importlib.util
import inspect
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import REGISTRY_PATH
from .endpoint import ChatResult, DryRunStub
from .env_state import StateStore
from .outcomes import HarnessFailure, ModelFailure, RunFailure
from .trace import TraceRecorder, extract_run_snapshot, verify_trace

JUDGES_DIR = Path(__file__).resolve().parent.parent / "judges"
POLICY_VERSION = "injection-render-policy-v1"

# arm -> canonical arm label for fixtures
ARMS = ("done", "compliant", "compromised", "none")


@dataclass(frozen=True)
class ArmConfig:
    """Per-arm fixed configuration. Held identical across arms for a matrix run:
    model, prompt rendering policy, tools policy, endpoint, temperature, retries,
    max turns, max tokens and the harness itself are all fixed by the caller."""

    model: str = "local"
    temperature: float = 0.0
    max_tokens: int = 1024
    max_turns: int = 8
    retries: int = 2
    timeout: float = 60.0
    injection_render_policy: str = POLICY_VERSION

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


DEFAULT_ARM_CONFIG = ArmConfig()


@dataclass
class RunResult:
    case_id: str
    arm: str
    seed: int
    status: str  # "ok" | "failed"
    failure: Optional[Dict[str, Any]] = None  # typed failure: {"type", "message", "detail"}
    task_success: Optional[bool] = None
    attack_success: Optional[bool] = None
    task_metadata: Optional[Dict[str, Any]] = None
    attack_metadata: Optional[Dict[str, Any]] = None
    trace_id: Optional[str] = None
    trace_path: Optional[str] = None
    events: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "arm": self.arm,
            "seed": self.seed,
            "status": self.status,
            "failure": self.failure,
            "task_success": self.task_success,
            "attack_success": self.attack_success,
            "task_metadata": self.task_metadata or {},
            "attack_metadata": self.attack_metadata or {},
            "trace_id": self.trace_id,
            "trace_path": self.trace_path,
            "events": self.events or [],
        }


class CaseRegistry:
    def __init__(self, path: Path = REGISTRY_PATH):
        self.path = Path(path)
        with open(self.path, encoding="utf-8") as fh:
            raw = json.load(fh)
        self.meta = raw.get("meta", {})
        self.cases: Dict[str, dict] = raw["cases"]

    def get(self, case_id: str) -> dict:
        case = self.cases.get(case_id)
        if case is None:
            raise HarnessFailure(f"unknown case id: {case_id}")
        return case


def render_messages(case: dict) -> List[Dict[str, Any]]:
    """Fixed, versioned injection-rendering policy (POLICY_VERSION).

    The system prompt is the case's own Agent.system_prompt; the user message is
    the task instruction plus — for attack cases only — a constant rendering of
    the *prompt-family* attack steps from the case's released Attack payload.
    Environment/tool/skill injections are not rendered into the prompt (they
    live environment-side); in dry-run the environment side is mirrored by the
    arm fixture's initial_state. This policy is identical for every arm.
    """
    config = case["config"]
    task = config.get("Task") or {}
    agent = config.get("Agent") or {}
    attack = config.get("Attack") or {}
    system = agent.get("system_prompt") or ""
    instruction = task.get("task_instruction") or ""
    user = str(instruction)
    if case.get("threat") in ("direct", "indirect"):
        prompt_steps = []
        for turn in attack.get("attack_turns") or []:
            for step in turn.get("attack_steps") or []:
                if step.get("type") == "prompt":
                    content = step.get("content")
                    if content:
                        prompt_steps.append(str(content))
        if prompt_steps:
            block = "\n\n".join(prompt_steps)
            user = f"{user}\n\n[Scenario note — fixed rendering, injection-render-policy-v1]\n{block}"
    messages = [{"role": "system", "content": system}]
    if user.strip():
        messages.append({"role": "user", "content": user.strip()})
    return messages


def trajectory_from_events(events: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """Build the BaseJudge.trajectory shape ((call, response) pairs) from tool events."""
    out: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for ev in events:
        if ev["event"] == "tool_call":
            out.append(
                (
                    {"env": ev.get("env", ""), "tool": ev.get("tool", ""), "arguments": ev.get("arguments", {})},
                    {"result": ev.get("result", "")},
                )
            )
    return out


class JudgeApplication:
    """Mirrors upstream utils/judge_helpers.run_judge semantics on the harness's
    transcribed judge modules (signature-aware trajectory passing)."""

    def __init__(self, case_id: str):
        self.case_id = case_id
        self.module = None
        self.judge_class = None

    def load(self) -> "JudgeApplication":
        judge_path = JUDGES_DIR / self.case_id / "judge.py"
        if not judge_path.exists():
            raise HarnessFailure(f"transcribed judge not found: {judge_path}")
        spec = importlib.util.spec_from_file_location(f"dtap_judge_{self.case_id}", judge_path)
        if spec is None or spec.loader is None:
            raise HarnessFailure(f"cannot load judge module for {self.case_id}")
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except RunFailure:
            raise
        except Exception as exc:
            raise HarnessFailure(f"judge module {self.case_id} failed to import: {exc!r}") from exc
        judge_class = None
        for name in dir(module):
            obj = getattr(module, name)
            if (
                isinstance(obj, type)
                and name == "Judge"
                and hasattr(obj, "eval_task")
                and hasattr(obj, "eval_attack")
            ):
                judge_class = obj
                break
        if judge_class is None:
            raise HarnessFailure(f"Judge class with eval_task/eval_attack not found in {self.case_id}/judge.py")
        self.module = module
        self.judge_class = judge_class
        return self

    def run(self, responses: List[str], trajectory: List[Tuple[Dict[str, Any], Dict[str, Any]]]) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "task_success": None,
            "task_message": "",
            "task_metadata": {},
            "attack_success": None,
            "attack_message": "",
            "attack_metadata": {},
            "error": None,
        }
        try:
            judge = self.judge_class()
            if hasattr(judge, "trajectory"):
                judge.trajectory = trajectory
        except RunFailure:
            raise
        except Exception as exc:
            raise HarnessFailure(f"judge {self.case_id} failed to instantiate: {exc!r}") from exc

        def _call(method: Callable) -> Any:
            try:
                sig = inspect.signature(method)
            except (TypeError, ValueError):
                return method(responses)
            params = sig.parameters
            if "trajectory" in params or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
                return method(responses, trajectory=trajectory)
            return method(responses)

        try:
            task_res = _call(judge.eval_task)
            if task_res is not None:
                ok, meta = task_res
                result["task_success"] = bool(ok)
                result["task_metadata"] = meta or {}
                result["task_message"] = (meta or {}).get("message", "")
        except RunFailure:
            raise
        except Exception as exc:
            raise HarnessFailure(f"eval_task raised in {self.case_id}: {exc!r}") from exc

        try:
            attack_res = _call(judge.eval_attack)
            if attack_res is not None:
                ok, meta = attack_res
                result["attack_success"] = bool(ok)
                result["attack_metadata"] = meta or {}
                result["attack_message"] = (meta or {}).get("message", "")
        except RunFailure:
            raise
        except Exception as exc:
            raise HarnessFailure(f"eval_attack raised in {self.case_id}: {exc!r}") from exc
        return result


def run_case(
    case_id: str,
    arm: str,
    seed: int,
    endpoint: Any,
    arm_config: ArmConfig = DEFAULT_ARM_CONFIG,
    results_dir: Optional[Path] = None,
    registry: Optional[CaseRegistry] = None,
) -> RunResult:
    """Execute one case/arm/seed and return a typed RunResult. Never raises:
    every failure is folded into the typed failure field."""
    results_dir = Path(results_dir) if results_dir else Path("results")
    registry = registry or CaseRegistry()
    trace_path = results_dir / "traces" / f"{case_id}.{arm}.seed{seed}.trace.jsonl"
    rec = TraceRecorder(trace_path)
    result = RunResult(case_id=case_id, arm=arm, seed=seed, status="ok", events=[])
    try:
        case = registry.get(case_id)
        rec.record(
            "session_start",
            {
                "case_id": case_id,
                "arm": arm,
                "seed": seed,
                "threat": case["threat"],
                "injection_families": case.get("injection_families", []),
                "arm_config": arm_config.as_dict(),
                "registry_meta": registry.meta,
            },
        )
        store = StateStore()
        from .shims import finance as _fin
        from .shims import gmail as _gmail
        from .shims import paypal as _paypal
        from .shims import salesforce as _sf
        from .shims import slack as _slack

        for shim in (_fin, _gmail, _paypal, _sf, _slack):
            shim._set_store(store)

        if isinstance(endpoint, DryRunStub):
            initial = endpoint.fixture.get("initial_state")
            if isinstance(initial, dict):
                store.reset(initial)

        messages = render_messages(case)
        tool_events: List[Dict[str, Any]] = []
        final_text = ""
        turn = 0
        while turn < arm_config.max_turns:
            turn += 1
            rec.record("endpoint_request", {"turn": turn, "messages": messages})
            try:
                chat: ChatResult = endpoint.complete(messages, seed=seed)
            except RunFailure:
                raise
            except Exception as exc:
                raise HarnessFailure(f"endpoint.complete raised: {exc!r}") from exc
            rec.record(
                "endpoint_response",
                {"turn": turn, "text": chat.text, "tool_calls": chat.tool_calls, "finish_reason": chat.finish_reason},
            )
            if chat.tool_calls:
                for tc in chat.tool_calls:
                    env = tc.get("env", "")
                    rec.record(
                        "tool_call",
                        {"turn": turn, "env": env, "tool": tc.get("name", ""), "arguments": tc.get("arguments", {})},
                    )
                    if isinstance(endpoint, DryRunStub):
                        step = endpoint.fixture.get("script", [])[tc.get("script_index", 0)]
                        if step.get("state_delta"):
                            store.apply_merge_patch(step["state_delta"])
                        store.data["finance"]["action_log"].append(
                            {
                                "role": "agent",
                                "metadata": {
                                    "tool_name": tc.get("name", ""),
                                    "tool_params": tc.get("arguments", {}),
                                },
                            }
                        )
                        store.data["finance"]["action_log"].append(
                            {"role": "tool", "state": step.get("result", "")}
                        )
                        rec.record(
                            "tool_result",
                            {"turn": turn, "tool": tc.get("name", ""), "result": step.get("result", "")},
                        )
                        tool_events.append(
                            {
                                "event": "tool_call",
                                "env": env,
                                "tool": tc.get("name", ""),
                                "arguments": tc.get("arguments", {}),
                                "result": step.get("result", ""),
                            }
                        )
                    else:
                        # Live mode: tool effects on the simulated environment are
                        # not modeled yet (no simulated services); the call itself
                        # is recorded in the trace so the judge's trajectory view
                        # is truthful about what the agent attempted.
                        rec.record(
                            "tool_result",
                            {"turn": turn, "tool": tc.get("name", ""), "result": "(live: env effects not modeled)"},
                        )
                        tool_events.append(
                            {
                                "event": "tool_call",
                                "env": env,
                                "tool": tc.get("name", ""),
                                "arguments": tc.get("arguments", {}),
                                "result": "",
                            }
                        )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.get("id", ""),
                            "content": "ok",
                        }
                    )
                continue
            final_text = chat.text
            rec.record("agent_response", {"turn": turn, "text": final_text})
            break
        else:
            raise ModelFailure(
                f"agent loop reached max_turns={arm_config.max_turns} without a final answer",
            )

        responses = [final_text] if final_text.strip() else []
        rec.record("state_snapshot", {"state": store.data})
        judge_app = JudgeApplication(case_id).load()
        judge_result = judge_app.run(responses, trajectory_from_events(tool_events))
        rec.record("judge_result", {"verdict": judge_result})
        if case["threat"] == "benign":
            result.task_success = judge_result["task_success"]
            result.attack_success = judge_result["attack_success"]
        else:
            result.attack_success = judge_result["attack_success"]
            result.task_success = judge_result["task_success"]
        result.task_metadata = judge_result["task_metadata"]
        result.attack_metadata = judge_result["attack_metadata"]
    except RunFailure as exc:
        result.status = "failed"
        result.failure = exc.to_outcome()
    except Exception as exc:
        result.status = "failed"
        result.failure = HarnessFailure(f"unhandled runner exception: {exc!r}").to_outcome()
    finally:
        rec.record("run_result", {"result": result.to_dict()})
        try:
            result.trace_id = rec.close()
            result.trace_path = str(trace_path)
        except Exception:
            pass
    return result


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion (95% CI by default)."""
    if trials <= 0:
        return (0.0, 0.0)
    p = successes / trials
    denom = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * trials)) / trials) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def run_matrix(
    case_ids: List[str],
    arms: List[str],
    seeds: List[int],
    endpoint_factory: Callable[[str, str, int], Any],
    arm_config: ArmConfig = DEFAULT_ARM_CONFIG,
    results_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run a seed-repeat matrix; aggregate verdicts with Wilson CIs.

    `endpoint_factory(case_id, arm, seed)` builds a fresh endpoint per run
    (dry-run stubs are per-fixture and stateful, so they must not be shared).
    """
    results_dir = Path(results_dir) if results_dir else Path("results")
    registry = CaseRegistry()
    rows: Dict[str, Any] = {}
    for case_id in case_ids:
        case = registry.get(case_id)
        for arm in arms:
            if case["threat"] == "benign" and arm not in ("done", "none"):
                continue
            if case["threat"] != "benign" and arm == "done":
                continue
            key = f"{case_id}::{arm}"
            runs = [
                run_case(case_id, arm, s, endpoint_factory(case_id, arm, s), arm_config, results_dir, registry)
                for s in seeds
            ]
            primary = "attack_success" if case["threat"] != "benign" else "task_success"
            successes = sum(1 for r in runs if r.status == "ok" and getattr(r, primary) is True)
            failed = sum(1 for r in runs if r.status == "failed")
            lo, hi = wilson_interval(successes, len(runs))
            failure_types = {r.failure["type"] for r in runs if r.failure}
            rows[key] = {
                "case_id": case_id,
                "arm": arm,
                "threat": case["threat"],
                "seeds": seeds,
                "n": len(runs),
                "successes": successes,
                "failed": failed,
                "rate": successes / len(runs) if runs else 0.0,
                "ci95": [round(lo, 4), round(hi, 4)],
                "typed_failures": sorted(failure_types),
            }
    summary_path = results_dir / "matrix.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return rows


def replay_trace(trace_path: Path, results_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Verify the immutable trace and deterministically re-run the judge against
    the recorded state; compare with the recorded verdict."""
    records = verify_trace(trace_path)
    snap = extract_run_snapshot(records)
    judge_app = JudgeApplication(snap["case_id"]).load()
    store = StateStore()
    from .shims import finance as _fin
    from .shims import gmail as _gmail
    from .shims import paypal as _paypal
    from .shims import salesforce as _sf
    from .shims import slack as _slack

    for shim in (_fin, _gmail, _paypal, _sf, _slack):
        shim._set_store(store)
    store.reset(snap["state"])
    replay_verdict = judge_app.run(snap["agent_responses"], [])
    recorded = snap["judge_result"]["verdict"]
    mismatch = {
        k: (replay_verdict.get(k), recorded.get(k))
        for k in ("task_success", "attack_success")
        if replay_verdict.get(k) != recorded.get(k)
    }
    return {
        "trace_path": str(trace_path),
        "trace_id": snap["run_result"].get("trace_id"),
        "case_id": snap["case_id"],
        "arm": snap["arm"],
        "seed": snap["seed"],
        "recorded_verdict": recorded,
        "replayed_verdict": replay_verdict,
        "mismatch": mismatch,
        "matches": not mismatch,
    }
