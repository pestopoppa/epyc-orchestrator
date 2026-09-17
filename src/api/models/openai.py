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
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Decode temperature. Forwarded to the backend ONLY when sent explicitly; "
        "the schema default 0.0 is NOT forwarded, so an omitted temperature uses the "
        "backend's per-role default. Ignored on image (vision) requests, which take no "
        "sampling overrides.",
    )
    top_p: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling override. Forwarded when set; ignored on image (vision) "
        "requests, which take no sampling overrides.",
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        description="Orchestrator extension: llama.cpp top-k sampling override. Forwarded when "
        "set; ignored on image (vision) requests, which take no sampling overrides.",
    )
    seed: int | None = Field(
        default=None,
        description="Optional deterministic decode seed. Forwarded when set; ignored on image "
        "(vision) requests, which take no sampling overrides.",
    )
    max_tokens: int = Field(
        default=1024,
        ge=1,
        le=32768,
        description="Generation cap ONLY with x_tool_mode='client' or x_disable_repl=true. In the "
        "default REPL mode it is NOT the token budget: each REPL turn generates up to a "
        "fixed 1024 tokens and max_tokens only sets the turn count (max_tokens // 500, "
        "clamped to 1..5). Ignored on image (vision) requests. max_completion_tokens is "
        "accepted as an alias (422 if both are sent).",
    )
    stream: bool = Field(default=False, description="Enable streaming")
    tools: list[dict[str, Any]] | None = Field(
        default=None,
        description="OpenAI native tool definitions. Forwarded to the backend verbatim ONLY with "
        "x_tool_mode='client'. In the default REPL mode they are rendered into the prompt as "
        "CALL() instructions for the orchestrator REPL and are never returned as tool_calls "
        "(metadata native_tool_contract='internal_repl_execution'). With x_disable_repl=true "
        "they are still rendered as prompt text but there is no REPL to execute them.",
    )
    tool_choice: str | dict[str, Any] | None = Field(
        default=None,
        description="OpenAI tool choice policy, e.g. 'auto', 'none', 'required', or function choice. "
        "Validated (422) and forwarded to the backend ONLY with x_tool_mode='client'. In the "
        "default REPL mode 'none' suppresses the rendered tool block and any other value is "
        "rendered as advisory prompt text, not enforced.",
    )
    # Extension fields — orchestrator routing overrides
    x_orchestrator_role: str | None = Field(
        default=None,
        description="Force specific orchestrator role. Values: any role from /v1/models (e.g. "
        "'architect_general', 'worker_math'). Honoured as the backend role on the text and "
        "client-tool paths; NOT validated against /v1/models here, so an unknown value is "
        "passed through to the backend lookup rather than refused with a 422. On image "
        "(vision) requests only 'worker_vision'/'vision_escalation' constrain the server; any "
        "other role is ignored by the vision path.",
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
        description="Highest-precedence ROLE override (x_force_model > x_orchestrator_role > "
        "model). Despite the name it does NOT select a model by registry name: on /v1 the "
        "value is treated exactly like x_orchestrator_role -- normalised as a role label and "
        "looked up in the role->server map -- so a registry model name (e.g. "
        "'architect_qwen2_5_72b') is not resolved to a model. Send a role from /v1/models.",
    )
    x_disable_repl: bool = Field(
        default=False,
        description="Skip REPL code execution -- direct text response only. Honoured on the text "
        "path. Not consulted with x_tool_mode='client' (which never uses the REPL) or on image "
        "(vision) requests. Any tools sent alongside it are rendered as prompt text with no "
        "executor.",
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
        description="Client conversation/session id (HS-4). Validated (422 on a bad value). "
        "RECORDED ONLY: stamped onto the inference-tap trace and echoed in "
        "x_orchestrator_metadata.request_keys when x_show_routing=true. No store is keyed on "
        "it on /v1 today (HS-4 P1/P3 are future work). Its ABSENCE is refused with a 422 for "
        "x_tool_mode='client' or an OpenCode user-agent when the v1_client_session_guard "
        "flag is on.",
    )
    x_user_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        pattern=_REQUEST_KEY_ID_PATTERN,
        description="Client user id (HS-4). Validated (422 on a bad value). RECORDED ONLY: "
        "stamped onto the inference-tap trace and echoed in x_orchestrator_metadata."
        "request_keys when x_show_routing=true. No user profile exists on /v1 today (HS-4 P2 "
        "is future work); the value changes nothing about the response.",
    )
    x_memory: Literal["on", "off"] | None = Field(
        default=None,
        description="Memory-injection arm (HS-4). Recorded only until HS-4 P2 ships: "
        "nothing is injected on /v1 today, so 'on' is reported as "
        "memory_injection='not_implemented' in the metadata (visible only with "
        "x_show_routing=true).",
    )
    x_tool_mode: Literal["repl", "client"] | None = Field(
        default=None,
        description="Tool execution mode (HS-4 P0.1). 'repl' (default when absent): client "
        "tools are rendered into the prompt as orchestrator REPL CALL() instructions and "
        "tool_calls are never returned (with x_disable_repl=true nothing executes them). "
        "'client': tools, tool_choice and tool history are forwarded to the backend and "
        "tool_calls are returned for the client to execute; tool_choice is validated (422), "
        "image input is refused (400), and x_session_id may be required (422, "
        "v1_client_session_guard flag). Neither mode escalates.",
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
