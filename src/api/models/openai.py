"""OpenAI-compatible models for the orchestrator API."""

import time
import uuid
from typing import Any

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
        description="Cap escalation tier. Values: 'A' (frontdoor only), 'B1' (coder), "
        "'B2' (architect), 'C' (worker). Prevents escalation beyond the specified tier.",
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
