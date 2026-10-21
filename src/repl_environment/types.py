"""Data types, exceptions, and constants for the REPL environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from src.config import _registry_timeout

if TYPE_CHECKING:
    pass


# Structured delimiters for tool output isolation.
# These wrap stdout from tool functions (my_role, route_advice, list_dir, recall, _invoke_tool)
# so _strip_tool_outputs() can use regex instead of fragile exact-string matching.
TOOL_OUTPUT_START = "<<<TOOL_OUTPUT>>>"
TOOL_OUTPUT_END = "<<<END_TOOL_OUTPUT>>>"


_TOOL_MARKER_ZWSP = "\u200b"  # same escape char as security.control_tokens.ZWSP


def neutralise_tool_markers(text: str) -> str:
    """Break any literal TOOL_OUTPUT begin/end marker inside ``text``.

    Content wrapped by :func:`wrap_tool_output` may come from attacker-controlled
    pages or documents; an embedded ``<<<END_TOOL_OUTPUT>>>`` would close the block
    early and let the tail pose as orchestrator structure.  A ZWSP is inserted
    after the first ``<`` of each marker (as ``control_tokens`` does for chat
    tokens).  Idempotent (the escaped form no longer matches) and a no-op, same
    object returned, for text without a marker.
    """
    if not text or "<<<" not in text:
        return text
    out = text.replace(TOOL_OUTPUT_START, "<" + _TOOL_MARKER_ZWSP + TOOL_OUTPUT_START[1:])
    out = out.replace(TOOL_OUTPUT_END, "<" + _TOOL_MARKER_ZWSP + TOOL_OUTPUT_END[1:])
    return text if out == text else out


def wrap_tool_output(output: str) -> str:
    """Wrap a tool output string with structured delimiters.

    Any begin/end marker already present in ``output`` is neutralised first so the
    content cannot terminate the block early.

    Args:
        output: The raw tool output string.

    Returns:
        Delimited output string for reliable stripping.
    """
    return f"{TOOL_OUTPUT_START}{neutralise_tool_markers(output)}{TOOL_OUTPUT_END}"


class REPLError(Exception):
    """Error during REPL execution."""

    pass


class REPLTimeout(REPLError):
    """Execution timed out."""

    pass


class REPLSecurityError(REPLError):
    """Attempted to execute dangerous code."""

    pass


class FinalSignal(Exception):
    """Signal that FINAL() was called with the answer."""

    def __init__(self, answer: str):
        self.answer = answer
        super().__init__(answer)


@dataclass
class ExplorationEvent:
    """A single exploration event for logging."""

    function: str  # peek, grep, llm_call, llm_batch
    args: dict[str, Any]  # Arguments passed
    result_size: int  # Size of result (chars or items)
    timestamp: float  # Time of call
    token_estimate: int = 0  # Estimated tokens used


@dataclass
class ExplorationLog:
    """Log of exploration events for a REPL session."""

    events: list[ExplorationEvent] = field(default_factory=list)
    total_exploration_tokens: int = 0
    # HIL-15: optional observed-dependency tracker (src/trace/dependency_log.py). None == off.
    dependency_log: Any = None
    # HIL-13a: optional step-event recorder (src/trace/step_events.py). None == off.
    step_events: Any = None

    def add_event(
        self,
        function: str,
        args: dict[str, Any],
        result: Any,
    ) -> None:
        """Add an exploration event to the log."""
        import time

        # Estimate result size
        if isinstance(result, str):
            result_size = len(result)
        elif isinstance(result, list):
            result_size = len(result)
        else:
            result_size = 0

        # Rough token estimate (4 chars per token)
        token_estimate = result_size // 4

        event = ExplorationEvent(
            function=function,
            args=args,
            result_size=result_size,
            timestamp=time.time(),
            token_estimate=token_estimate,
        )
        self.events.append(event)
        self.total_exploration_tokens += token_estimate
        if self.dependency_log is not None:
            self.dependency_log.observe(function, args, result, ok=(result != "refused"))
        if self.step_events is not None and not self.step_events.in_call():   # HIL-13a, default off
            self.step_events.observe_builtin(function, args, result)

    def get_strategy_summary(self) -> dict[str, Any]:
        """Get a summary of the exploration strategy used."""
        function_counts: dict[str, int] = {}
        for event in self.events:
            function_counts[event.function] = function_counts.get(event.function, 0) + 1

        return {
            "total_events": len(self.events),
            "function_counts": function_counts,
            "total_tokens": self.total_exploration_tokens,
            "strategy_type": self._classify_strategy(function_counts),
        }

    def get_token_efficiency(self, result_tokens: int) -> dict[str, Any]:
        """Calculate token efficiency metrics.

        Token efficiency = result_tokens / exploration_tokens
        Higher is better - means we got more useful output per exploration token spent.

        Args:
            result_tokens: Tokens in the final result (estimated as len(result)/4).

        Returns:
            Dictionary with efficiency metrics.
        """
        if self.total_exploration_tokens == 0:
            efficiency = float("inf") if result_tokens > 0 else 0.0
        else:
            efficiency = result_tokens / self.total_exploration_tokens

        return {
            "exploration_tokens": self.total_exploration_tokens,
            "result_tokens": result_tokens,
            "efficiency_ratio": round(efficiency, 3),
            "total_events": len(self.events),
        }

    def _classify_strategy(self, counts: dict[str, int]) -> str:
        """Classify the exploration strategy based on function usage."""
        if not counts:
            return "none"
        if counts.get("llm_call", 0) > 0 or counts.get("llm_batch", 0) > 0:
            return "delegated"  # Used sub-LLM calls
        if counts.get("grep", 0) > counts.get("peek", 0):
            return "search"  # Primarily used grep
        if counts.get("peek", 0) > 0:
            return "scan"  # Primarily used peek
        return "mixed"


@dataclass
class REPLConfig:
    """Configuration for the REPL environment.

    Timeout default from model_registry.yaml (runtime_defaults.timeouts.repl.session).
    """

    timeout_seconds: int = field(
        default_factory=lambda: int(_registry_timeout("repl", "session", 600))
    )
    output_cap: int = 8192
    spill_dir: str = ""
    max_grep_results: int = 100

    def __post_init__(self) -> None:
        if not self.spill_dir:
            try:
                from src.config import get_config
                self.spill_dir = str(get_config().paths.tmp_dir / "repl_output")
            except Exception:
                self.spill_dir = "/mnt/raid0/llm/tmp/repl_output"
    # Forced exploration validation (prevent premature FINAL)
    # Default False for backwards compatibility - enable for production use
    require_exploration_before_final: bool = False
    min_exploration_calls: int = 1  # Minimum peek/grep/llm_call before FINAL
    # TOON encoding for tool outputs (reduces tokens by ~55% on structured data)
    use_toon_encoding: bool = (
        True  # Enabled after TTFT benchmark: 55.6% token reduction, 41.8% latency improvement
    )
    # Structured mode: React-style one-tool-per-turn execution
    # When True, enforces single tool call per execute() and returns observation.
    # Replaces separate React mode with unified REPL that can operate in structured fashion.
    structured_mode: bool = False
    allowed_builtins: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {
                # Safe builtins
                "abs",
                "all",
                "any",
                "ascii",
                "bin",
                "bool",
                "bytes",
                "chr",
                "dict",
                "divmod",
                "enumerate",
                "filter",
                "float",
                "format",
                "frozenset",
                "hash",
                "hex",
                "int",
                "isinstance",
                "issubclass",
                "iter",
                "len",
                "list",
                "map",
                "max",
                "min",
                "next",
                "oct",
                "ord",
                "pow",
                "print",
                "range",
                "repr",
                "reversed",
                "round",
                "set",
                "slice",
                "sorted",
                "str",
                "sum",
                "tuple",
                "type",
                "zip",
                # Exceptions (needed for try/except)
                "Exception",
                "ValueError",
                "TypeError",
                "KeyError",
                "IndexError",
                "AttributeError",
                "RuntimeError",
                "StopIteration",
                "ZeroDivisionError",
                "NameError",
                "FileNotFoundError",
                "IOError",
                "OSError",
                # Internals (needed for class/super in exec())
                "__build_class__",
                # Constants
                "True",
                "False",
                "None",
            }
        )
    )


@dataclass
class ExecutionResult:
    """Result of executing code in the REPL."""

    output: str
    is_final: bool
    final_answer: str | None = None
    error: str | None = None
    elapsed_seconds: float = 0.0
