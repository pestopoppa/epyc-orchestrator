"""OpenAI-compatible models for the orchestrator API."""

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class OpenAIMessage(BaseModel):
    """OpenAI message format."""

    role: str = Field(..., description="Role: system, user, assistant, tool")
    content: str | list | None = Field(
        default=None,
        description="Message content: string or multipart content array "
        '(e.g. [{"type": "text", "text": "..."}, {"type": "image_url", ...}])',
    )
    tool_calls: list[dict[str, Any]] | None = Field(
        default=None,
        description="Assistant tool calls in OpenAI chat-completions format",
    )
    tool_call_id: str | None = Field(
        default=None,
        description="Tool-call id for role=tool result messages",
    )
    name: str | None = Field(
        default=None,
        description="Optional participant or tool name",
    )

    @model_validator(mode="after")
    def _require_content_or_tool_call(self) -> "OpenAIMessage":
        if self.content is None and not self.tool_calls:
            raise ValueError("content is required unless assistant tool_calls are present")
        return self


# HS-OD-1: standard OpenAI body fields this API does not honour must be REFUSED
# when honouring them would have changed the output — never silently dropped.
# Pydantic's default extra='ignore' was discarding response_format without error,
# so any JSON-mode client got prose with a 200 and no diagnostic. Value-sensitive
# on purpose: an explicit no-op (n=1, penalty 0.0, response_format {"type":"text"},
# empty stop list) is accepted so SDK clients that spell out defaults keep
# working; only a request whose semantics we would silently change is refused.
# Fields with no output effect (user, metadata, stream_options) stay ignored.
_UNHONOURED_SEMANTIC_FIELDS: dict = {
    "response_format": (
        lambda v: v is not None and not (isinstance(v, dict) and v.get("type") == "text"),
        "JSON mode is not implemented on this seam; remove response_format "
        'or send {"type": "text"}',
    ),
    "n": (lambda v: v is not None and v != 1, "only n=1 is supported"),
    "stop": (lambda v: bool(v), "stop sequences are not forwarded to the backend"),
    "logprobs": (lambda v: bool(v), "logprobs are not returned"),
    "top_logprobs": (lambda v: v is not None, "logprobs are not returned"),
    "logit_bias": (lambda v: bool(v), "logit_bias is not forwarded to the backend"),
    "presence_penalty": (
        lambda v: v not in (None, 0, 0.0),
        "sampling penalties are not forwarded to the backend",
    ),
    "frequency_penalty": (
        lambda v: v not in (None, 0, 0.0),
        "sampling penalties are not forwarded to the backend",
    ),
    "functions": (
        lambda v: bool(v),
        "legacy function calling is not supported; use tools",
    ),
    "function_call": (
        lambda v: v is not None,
        "legacy function calling is not supported; use tool_choice",
    ),
}


# HS-4 P0.2: opaque client identifiers — printable, no whitespace, bounded.
_REQUEST_KEY_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:@+/=-]*$"
_CLIENT_TOOL_CHOICE_STRINGS = frozenset({"auto", "none", "required"})


class OpenAIChatRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    model: str = Field(default="orchestrator", description="Model/role to use")
    messages: list[OpenAIMessage] = Field(..., description="Conversation messages")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    top_k: int | None = Field(
        default=None,
        ge=1,
        description="Orchestrator extension: llama.cpp top-k sampling override.",
    )
    seed: int | None = Field(default=None, description="Optional deterministic decode seed")
    max_tokens: int = Field(default=1024, ge=1, le=32768)
    stream: bool = Field(default=False, description="Enable streaming")
    tools: list[dict[str, Any]] | None = Field(
        default=None,
        description="OpenAI native tool definitions. Function tools are bridged to REPL CALL().",
    )
    tool_choice: str | dict[str, Any] | None = Field(
        default=None,
        description="OpenAI tool choice policy, e.g. 'auto', 'none', 'required', or function choice.",
    )
    # Extension fields — orchestrator routing overrides
    x_orchestrator_role: str | None = Field(
        default=None,
        description="Force specific orchestrator role, bypassing frontdoor routing. "
        "Values: any role from /v1/models (e.g. 'architect_general', 'worker_math').",
    )
    x_max_escalation: str | None = Field(
        default=None,
        description="Requested escalation-tier cap. Values: 'A' (frontdoor only), 'B1' (coder), "
        "'B2' (architect), 'C' (worker). METADATA ONLY on /v1 today: the value is recorded "
        "in routing metadata and NOT enforced -- role/override resolution applies no "
        "escalation cap, and client tool mode performs no escalation at all. Enforcement "
        "is HS-4 P4 work; until then this field does not prevent anything.",
    )
    x_force_model: str | None = Field(
        default=None,
        description="Force a specific model by registry name (e.g. 'architect_qwen2_5_72b'), "
        "bypassing all routing logic. Takes precedence over x_orchestrator_role.",
    )
    x_disable_repl: bool = Field(
        default=False,
        description="Skip REPL code execution — force direct text response only.",
    )
    x_show_routing: bool = Field(default=False, description="Include routing metadata")
    # HS-4 P0.2 — typed session/arm keys. Each value is validated (422 on a bad
    # one), echoed into x_orchestrator_metadata["request_keys"] and stamped onto
    # the inference-tap trace. Absent keys change nothing.
    x_session_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        pattern=_REQUEST_KEY_ID_PATTERN,
        description="Client conversation/session id (HS-4). Recorded; P1/P3 key their stores on it.",
    )
    x_user_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        pattern=_REQUEST_KEY_ID_PATTERN,
        description="Client user id (HS-4). Recorded; P2 keys the user profile on it.",
    )
    x_memory: Literal["on", "off"] | None = Field(
        default=None,
        description="Memory-injection arm (HS-4). Recorded only until HS-4 P2 ships: "
        "nothing is injected on /v1 today, so 'on' is reported as "
        "memory_injection='not_implemented' in the metadata.",
    )
    x_tool_mode: Literal["repl", "client"] | None = Field(
        default=None,
        description="Tool execution mode (HS-4 P0.1). 'repl' (default when absent): client "
        "tools are bridged to the orchestrator REPL CALL(). 'client': tools, tool_choice and "
        "tool history are forwarded to the backend and tool_calls are returned for the "
        "client to execute.",
    )

    @model_validator(mode="after")
    def _validate_client_tool_choice(self) -> "OpenAIChatRequest":
        # Only client mode is strict: the default REPL bridge keeps today's
        # permissive handling byte-for-byte (HS-4 P0.1(b)).
        if self.x_tool_mode != "client" or self.tool_choice is None:
            return self
        choice = self.tool_choice
        if isinstance(choice, str):
            if choice not in _CLIENT_TOOL_CHOICE_STRINGS:
                raise ValueError(
                    f"tool_choice {choice!r} is not one of "
                    f"{sorted(_CLIENT_TOOL_CHOICE_STRINGS)} (x_tool_mode='client')"
                )
            if choice == "required" and not self.tools:
                raise ValueError("tool_choice 'required' needs a non-empty tools list")
            return self
        func = choice.get("function") if choice.get("type") == "function" else None
        name = func.get("name") if isinstance(func, dict) else None
        if not isinstance(name, str) or not name:
            raise ValueError(
                "tool_choice object must be "
                '{"type": "function", "function": {"name": ...}} (x_tool_mode=\'client\')'
            )
        declared = {
            (t.get("function") or {}).get("name")
            for t in (self.tools or [])
            if isinstance(t, dict) and isinstance(t.get("function"), dict)
        }
        if name not in declared:
            raise ValueError(f"tool_choice names undeclared tool {name!r}")
        return self

    @model_validator(mode="before")
    @classmethod
    def _alias_max_completion_tokens(cls, data):
        # OpenAI deprecated max_tokens in favour of max_completion_tokens; SDK
        # clients send either. Runs mode="before" because after validation the
        # default max_tokens=1024 is indistinguishable from an explicit one.
        if isinstance(data, dict) and data.get("max_completion_tokens") is not None:
            if data.get("max_tokens") is not None:
                raise ValueError(
                    "max_tokens and max_completion_tokens were both supplied; "
                    "send exactly one"
                )
            data = dict(data)
            data["max_tokens"] = data.pop("max_completion_tokens")
        return data

    @model_validator(mode="before")
    @classmethod
    def _refuse_unhonoured_semantic_fields(cls, data):
        if isinstance(data, dict):
            for field, (would_change_output, reason) in _UNHONOURED_SEMANTIC_FIELDS.items():
                if field in data and would_change_output(data[field]):
                    raise ValueError(
                        f"'{field}' is not honoured by this API and is not a "
                        f"no-op in this request: {reason}. Refusing rather than "
                        "silently dropping it (HS-OD-1)."
                    )
        return data


class OpenAIChoice(BaseModel):
    """OpenAI choice object."""

    index: int = 0
    message: OpenAIMessage | None = None
    delta: dict[str, str] | None = None
    finish_reason: str | None = None


class OpenAIUsage(BaseModel):
    """OpenAI usage statistics."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class OpenAIChatResponse(BaseModel):
    """OpenAI-compatible chat completion response."""

    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:8]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "orchestrator"
    choices: list[OpenAIChoice]
    usage: OpenAIUsage | None = None
    # Extension fields
    x_orchestrator_metadata: dict[str, Any] | None = None


class OpenAIModelInfo(BaseModel):
    """OpenAI model info."""

    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "orchestrator"


class OpenAIModelsResponse(BaseModel):
    """OpenAI models list response."""

    object: str = "list"
    data: list[OpenAIModelInfo]
