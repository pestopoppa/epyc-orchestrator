#!/usr/bin/env python3
"""Gate Runner for automated quality gates.

This module runs quality gates (linting, testing, schema validation) and
parses their output to provide structured failure information for agent feedback.

Usage:
    from src.gate_runner import GateRunner

    runner = GateRunner()
    results = runner.run_all_gates()

    if not all(r.passed for r in results):
        failed = [r for r in results if not r.passed]
        for r in failed:
            print(f"Gate '{r.gate_name}' failed: {r.errors}")
"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import sys
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

import yaml

from src.config import _registry_timeout

# The CJ-8 gate vocabulary. `scripts/` is not an installed package, so it is
# loaded by path — the same idiom epyc-root's granite preflight uses.
_GVV_PATH = Path(__file__).resolve().parent.parent / "scripts" / "benchmark" / "gate_verdict_vocab.py"
_GVV_SPEC = importlib.util.spec_from_file_location("gate_verdict_vocab", _GVV_PATH)
if _GVV_SPEC is None or _GVV_SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load the CJ-8 gate vocabulary from {_GVV_PATH}")
gate_verdict_vocab = importlib.util.module_from_spec(_GVV_SPEC)
sys.modules.setdefault("gate_verdict_vocab", gate_verdict_vocab)
_GVV_SPEC.loader.exec_module(gate_verdict_vocab)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from orchestration.repl_memory.progress_logger import ProgressLogger

# Default gate timeout from registry
_GATE_TIMEOUT = int(_registry_timeout("tools", "gate_runner", 60))

# Per-gate timeouts from registry
_GATE_FORMAT_TIMEOUT = int(_registry_timeout("gates", "format", 30))
_GATE_LINT_TIMEOUT = int(_registry_timeout("gates", "lint", 60))
_GATE_TYPECHECK_TIMEOUT = int(_registry_timeout("gates", "typecheck", 120))
_GATE_UNIT_TIMEOUT = int(_registry_timeout("gates", "unit", 180))


@dataclass
class GateConfig:
    """Configuration for a single gate.

    Timeout default from model_registry.yaml (runtime_defaults.timeouts.tools.gate_runner).
    """

    name: str
    command: str
    timeout: int = _GATE_TIMEOUT
    required: bool = True
    retry_count: int = 0
    description: str = ""
    parallelizable: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GateConfig:
        """Create GateConfig from dictionary."""
        return cls(
            name=data["name"],
            command=data["command"],
            timeout=data.get("timeout", _GATE_TIMEOUT),
            required=data.get("required", True),
            retry_count=data.get("retry_count", 0),
            description=data.get("description", ""),
            parallelizable=data.get("parallelizable", False),
        )


@dataclass
class GateResult:
    """Result of running a single gate.

    CJ-8 (2026-09-07): THE VERDICT IS THREE-VALUED, and ``passed`` is no longer
    the whole of it.

    Before this, a gate that TIMED OUT, a gate whose subprocess RAISED, and a
    gate NAME THAT DOES NOT EXIST all produced ``passed=False`` — the same value
    a genuine lint failure produces. Downstream that boolean became an
    ``EventType.GATE_FAILED`` event, and ``q_reward.compute_reward`` charges
    -0.1 per ``GATE_FAILED``. So a harness timeout was converted into negative
    learning signal ABOUT THE MODEL, which never did anything wrong: the check
    never ran. Infra failure is the ABSENCE of a measurement, never a bad one.

    ``passed`` KEEPS ITS MEANING AND ITS VALUE. An undecidable gate still
    reports ``passed=False``, so every existing consumer — the retry loop, the
    stop-on-required-failure rule, ``all_passed`` at the API boundary, the
    ``get_summary`` tally — blocks exactly as it did before. Out-of-coverage is
    RENAMED, never downgraded to a pass. What is new is ``verdict``/``cause``
    riding alongside, so a consumer that needs the distinction (the reward
    writer, the verification-report bridge) can read it instead of inferring it
    from ``exit_code == -1`` and a substring of ``errors[0]``.
    """

    gate_name: str
    passed: bool
    exit_code: int
    output: str
    elapsed_seconds: float
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    attempt: int = 1
    required: bool = True
    #: CJ-8 verdict. Defaults from ``passed`` so every existing construction
    #: site (and every pickled/reconstructed result) keeps its old meaning; only
    #: the sites that KNOW the check did not run pass ``out-of-coverage``.
    verdict: str | None = None
    #: Mandatory on ``out-of-coverage``, forbidden otherwise.
    cause: str | None = None

    def __post_init__(self) -> None:
        if self.verdict is None:
            self.verdict = (
                gate_verdict_vocab.VERDICT_PASS if self.passed
                else gate_verdict_vocab.VERDICT_FAIL
            )
        if self.verdict == gate_verdict_vocab.VERDICT_OUT_OF_COVERAGE:
            if self.cause is None:
                raise GateRunnerError(
                    f"gate {self.gate_name!r}: 'out-of-coverage' without a cause "
                    f"code. An undecided gate with no cause names no remedy — "
                    f"one of {gate_verdict_vocab.CAUSES!r}."
                )
            if self.cause not in gate_verdict_vocab.CAUSES:
                raise GateRunnerError(
                    f"gate {self.gate_name!r}: cause {self.cause!r} is outside "
                    f"the closed registry {gate_verdict_vocab.CAUSES!r}."
                )
            if self.passed:
                raise GateRunnerError(
                    f"gate {self.gate_name!r}: an undecided gate can never be "
                    f"passed=True. Out-of-coverage stays BLOCKING."
                )
        elif self.cause is not None:
            raise GateRunnerError(
                f"gate {self.gate_name!r}: verdict {self.verdict!r} is a DECISION "
                f"and must not carry a cause code (got {self.cause!r})."
            )

    @property
    def decided(self) -> bool:
        """True iff this gate actually reached a verdict on its subject."""
        return gate_verdict_vocab.is_decided(self.verdict or "")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d = {
            "gate_name": self.gate_name,
            "passed": self.passed,
            "exit_code": self.exit_code,
            "elapsed_seconds": self.elapsed_seconds,
            "errors": self.errors,
            "warnings": self.warnings,
            "attempt": self.attempt,
            "required": self.required,
            "verdict": self.verdict,
        }
        if self.cause is not None:
            d["cause"] = self.cause
            d["cause_means"] = gate_verdict_vocab.CAUSE_MEANINGS[self.cause]
        return d

    @property
    def summary(self) -> str:
        """Get a one-line summary of the result."""
        if not self.decided:
            status = f"NOT-DECIDED:{self.cause}"
        else:
            status = "PASSED" if self.passed else "FAILED"
        return f"[{status}] {self.gate_name} ({self.elapsed_seconds:.1f}s)"


class GateRunnerError(Exception):
    """Error in gate runner."""

    pass


class GateRunner:
    """Runs quality gates and parses output.

    Loads gate configuration from a YAML file and executes gates
    in order, collecting structured results.

    Config source (RD-4): ``config/gates.yaml`` now exists and codifies the
    exact gate set below. When present it is loaded; when absent
    ``_get_default_gates()`` is the fallback. The two are kept byte-for-behaviour
    equal (same names/commands/timeouts/flags) — introducing the file changed no
    default behaviour; ``tests/test_verifier_adapter.py::TestGatesYamlParity``
    guards that parity. Consumed by ``src/proactive_delegation/verifier_adapter.py``,
    which maps ``verifier_requests[]`` onto named gates.
    """

    DEFAULT_CONFIG_PATH = Path("config/gates.yaml")

    def __init__(
        self,
        config_path: Path | str | None = None,
        working_dir: Path | str | None = None,
        progress_logger: "ProgressLogger | None" = None,
    ):
        """Initialize the gate runner.

        Args:
            config_path: Path to gates.yaml configuration file.
            working_dir: Working directory for running commands.
            progress_logger: Optional ProgressLogger for MemRL integration.
        """
        self.config_path = Path(config_path) if config_path else self.DEFAULT_CONFIG_PATH
        self.working_dir = Path(working_dir) if working_dir else Path.cwd()
        self.progress_logger = progress_logger
        self.gates: list[GateConfig] = []
        self._load_config()

    def _load_config(self) -> None:
        """Load gate configuration from YAML file."""
        if not self.config_path.exists():
            # Use default gates if config doesn't exist
            self.gates = self._get_default_gates()
            return

        try:
            with open(self.config_path) as f:
                config = yaml.safe_load(f)

            self.gates = [GateConfig.from_dict(gate_data) for gate_data in config.get("gates", [])]
        except Exception as e:
            logger.error("Failed to load gate config from %s: %s", self.config_path, e)
            raise GateRunnerError(f"Failed to load config: {e}")

    def _get_default_gates(self) -> list[GateConfig]:
        """Get default gate configuration.

        Timeouts from model_registry.yaml (runtime_defaults.timeouts.gates.*).
        """
        return [
            GateConfig(
                name="format",
                command="make format-check",
                timeout=_GATE_FORMAT_TIMEOUT,
                required=True,
                description="Check code formatting",
                parallelizable=True,
            ),
            GateConfig(
                name="lint",
                command="make lint",
                timeout=_GATE_LINT_TIMEOUT,
                required=True,
                description="Run linters",
                parallelizable=True,
            ),
            GateConfig(
                name="typecheck",
                command="make typecheck",
                timeout=_GATE_TYPECHECK_TIMEOUT,
                required=False,
                description="Run type checker",
            ),
            GateConfig(
                name="unit",
                command="make test-unit",
                timeout=_GATE_UNIT_TIMEOUT,
                required=True,
                description="Run unit tests",
            ),
        ]

    def run_gate(
        self,
        gate: GateConfig,
        attempt: int = 1,
        task_id: str | None = None,
        agent_tier: str = "C",
        agent_role: str = "worker",
    ) -> GateResult:
        """Run a single gate and parse output.

        Args:
            gate: Gate configuration.
            attempt: Current attempt number (for retries).
            task_id: Optional task ID for MemRL logging.
            agent_tier: Agent tier for logging (default "C" for workers).
            agent_role: Agent role for logging.

        Returns:
            GateResult with parsed output.
        """
        start_time = time.perf_counter()

        try:
            import shlex

            result = subprocess.run(
                shlex.split(gate.command),
                capture_output=True,
                text=True,
                timeout=gate.timeout,
                cwd=self.working_dir,
            )

            elapsed = time.perf_counter() - start_time
            output = result.stdout + "\n" + result.stderr
            passed = result.returncode == 0

            # Parse errors and warnings from output
            errors, warnings = self._parse_output(output, gate.name)

            gate_result = GateResult(
                gate_name=gate.name,
                passed=passed,
                exit_code=result.returncode,
                output=output,
                elapsed_seconds=elapsed,
                errors=errors if not passed else [],
                warnings=warnings,
                attempt=attempt,
                required=gate.required,
            )

        except subprocess.TimeoutExpired:
            # CJ-8. The gate exceeded its budget BEFORE deciding. It did not rule
            # against the subject; it failed to rule. `passed` stays False so the
            # retry/stop-on-failure/`all_passed` logic still BLOCKS.
            elapsed = time.perf_counter() - start_time
            gate_result = GateResult(
                gate_name=gate.name,
                passed=False,
                exit_code=-1,
                output=f"Timeout after {gate.timeout}s",
                elapsed_seconds=elapsed,
                errors=[f"Gate timed out after {gate.timeout}s"],
                attempt=attempt,
                required=gate.required,
                verdict=gate_verdict_vocab.VERDICT_OUT_OF_COVERAGE,
                cause=gate_verdict_vocab.CAUSE_TIMEOUT,
            )

        except Exception as e:
            # CJ-8. The INSTRUMENT raised — a defect in the checker, not evidence
            # about the subject. Still blocking (`passed=False`), now named.
            elapsed = time.perf_counter() - start_time
            gate_result = GateResult(
                gate_name=gate.name,
                passed=False,
                exit_code=-1,
                output=str(e),
                elapsed_seconds=elapsed,
                errors=[f"Gate execution error: {e}"],
                attempt=attempt,
                required=gate.required,
                verdict=gate_verdict_vocab.VERDICT_OUT_OF_COVERAGE,
                cause=gate_verdict_vocab.CAUSE_CHECKER_ERROR,
            )

        # Log gate result for MemRL if configured
        if self.progress_logger and task_id:
            error_msg = gate_result.errors[0] if gate_result.errors else None
            # CJ-8. The verdict rides the call. Without it the logger collapses
            # to GATE_PASSED/GATE_FAILED and `q_reward` charges -0.1 for a
            # timeout — negative learning signal about a model that never got
            # checked. `log_gate_result` accepts `verdict`/`cause` optionally, so
            # a logger that predates this change is unaffected.
            self.progress_logger.log_gate_result(
                task_id=task_id,
                gate_name=gate.name,
                passed=gate_result.passed,
                agent_tier=agent_tier,
                agent_role=agent_role,
                error_message=error_msg,
                verdict=gate_result.verdict,
                cause=gate_result.cause,
            )

        return gate_result

    def _parse_output(self, output: str, gate_name: str) -> tuple[list[str], list[str]]:
        """Parse gate output for errors and warnings.

        Args:
            output: Raw gate output.
            gate_name: Name of the gate (for context-specific parsing).

        Returns:
            Tuple of (errors, warnings) lists.
        """
        errors = []
        warnings = []

        for line in output.split("\n"):
            line_lower = line.lower()

            # Skip common log prefixes
            if any(prefix in line_lower for prefix in ["build:", "main:", "info:"]):
                continue

            # Detect errors
            if any(pattern in line_lower for pattern in ["error:", "error ", "failed", "fatal"]):
                # Clean up the error message
                clean_line = line.strip()
                if clean_line and clean_line not in errors:
                    errors.append(clean_line)

            # Detect warnings
            elif "warning:" in line_lower or "warn " in line_lower:
                clean_line = line.strip()
                if clean_line and clean_line not in warnings:
                    warnings.append(clean_line)

        # Limit to most relevant errors
        if len(errors) > 20:
            errors = errors[:20] + [f"... and {len(errors) - 20} more errors"]

        return errors, warnings

    def run_all_gates(
        self,
        stop_on_first_failure: bool = True,
        required_only: bool = False,
        task_id: str | None = None,
        agent_tier: str = "C",
        agent_role: str = "worker",
    ) -> list[GateResult]:
        """Run all configured gates.

        Args:
            stop_on_first_failure: Stop after first required gate fails.
            required_only: Only run gates marked as required.
            task_id: Optional task ID for MemRL logging.
            agent_tier: Agent tier for logging.
            agent_role: Agent role for logging.

        Returns:
            List of GateResult objects.
        """
        results = []

        for gate in self.gates:
            if required_only and not gate.required:
                continue

            # Run with retries
            for attempt in range(1, gate.retry_count + 2):  # +2 for first attempt
                result = self.run_gate(
                    gate,
                    attempt=attempt,
                    task_id=task_id,
                    agent_tier=agent_tier,
                    agent_role=agent_role,
                )
                if result.passed:
                    break

            results.append(result)

            # Check if we should stop
            if stop_on_first_failure and not result.passed and gate.required:
                break

        return results

    async def run_all_gates_parallel(
        self,
        stop_on_first_failure: bool = True,
        required_only: bool = False,
        task_id: str | None = None,
        agent_tier: str = "C",
        agent_role: str = "worker",
    ) -> list[GateResult]:
        """Run gates with lightweight gates in parallel, heavy gates sequential.

        Gates marked ``parallelizable=True`` run concurrently in Phase 1.
        Remaining gates run sequentially in Phase 2 (same as run_all_gates).

        Args:
            stop_on_first_failure: Stop after first required gate fails.
            required_only: Only run gates marked as required.
            task_id: Optional task ID for MemRL logging.
            agent_tier: Agent tier for logging.
            agent_role: Agent role for logging.

        Returns:
            List of GateResult objects in config order.
        """
        gates = self.gates
        if required_only:
            gates = [g for g in gates if g.required]

        # Partition into parallel and sequential
        parallel_gates = [g for g in gates if g.parallelizable]
        sequential_gates = [g for g in gates if not g.parallelizable]

        # Track results by gate name for ordered output
        results_map: dict[str, GateResult] = {}

        # Phase 1: Run parallelizable gates concurrently
        if parallel_gates:

            async def _run_gate_async(gate: GateConfig) -> GateResult:
                return await asyncio.to_thread(
                    self.run_gate,
                    gate,
                    1,
                    task_id,
                    agent_tier,
                    agent_role,
                )

            parallel_results = await asyncio.gather(
                *[_run_gate_async(g) for g in parallel_gates],
            )

            # Retry failures sequentially
            for gate, result in zip(parallel_gates, parallel_results):
                if not result.passed and gate.retry_count > 0:
                    for attempt in range(2, gate.retry_count + 2):
                        result = self.run_gate(
                            gate,
                            attempt,
                            task_id,
                            agent_tier,
                            agent_role,
                        )
                        if result.passed:
                            break
                results_map[gate.name] = result

            # Check if any required parallel gate failed
            if stop_on_first_failure:
                for gate in parallel_gates:
                    r = results_map[gate.name]
                    if not r.passed and gate.required:
                        # Return results in config order (parallel only)
                        return [results_map[g.name] for g in gates if g.name in results_map]

        # Phase 2: Run sequential gates in order
        for gate in sequential_gates:
            for attempt in range(1, gate.retry_count + 2):
                result = self.run_gate(
                    gate,
                    attempt,
                    task_id,
                    agent_tier,
                    agent_role,
                )
                if result.passed:
                    break

            results_map[gate.name] = result

            if stop_on_first_failure and not result.passed and gate.required:
                break

        # Return results in original config order
        return [results_map[g.name] for g in gates if g.name in results_map]

    def run_gates_by_name(self, gate_names: list[str]) -> list[GateResult]:
        """Run specific gates by name.

        Args:
            gate_names: List of gate names to run.

        Returns:
            List of GateResult objects.
        """
        results = []

        for name in gate_names:
            gate = next((g for g in self.gates if g.name == name), None)
            if gate is None:
                # CJ-8. A gate name that does not exist says NOTHING about the
                # subject: no check ran. Reporting it as a failed check pointed
                # every remedy at the model instead of at the caller's gate list.
                results.append(
                    GateResult(
                        gate_name=name,
                        passed=False,
                        exit_code=-1,
                        output="",
                        elapsed_seconds=0,
                        errors=[f"Unknown gate: {name}"],
                        verdict=gate_verdict_vocab.VERDICT_OUT_OF_COVERAGE,
                        cause=gate_verdict_vocab.CAUSE_UNSUPPORTED,
                    )
                )
            else:
                results.append(self.run_gate(gate))

        return results

    def get_gate_names(self) -> list[str]:
        """Get list of configured gate names."""
        return [g.name for g in self.gates]

    def get_summary(self, results: list[GateResult]) -> str:
        """Get a summary of gate results.

        Args:
            results: List of GateResult objects.

        Returns:
            Multi-line summary string.
        """
        lines = ["Gate Results:"]
        lines.append("-" * 40)

        for r in results:
            lines.append(r.summary)
            if r.errors:
                for e in r.errors[:3]:
                    lines.append(f"  - {e[:100]}")

        total = len(results)
        passed = sum(1 for r in results if r.passed)
        # CJ-8. `failed` used to be `total - passed`, which folded every gate
        # that never ran into the failed count. Undecided gates are now named
        # separately; they still block, they are simply no longer mislabelled.
        undecided = sum(1 for r in results if not r.decided)
        failed = total - passed - undecided

        lines.append("-" * 40)
        line = f"Total: {passed}/{total} passed, {failed} failed"
        if undecided:
            line += f", {undecided} not decided (out-of-coverage)"
        lines.append(line)

        return "\n".join(lines)
