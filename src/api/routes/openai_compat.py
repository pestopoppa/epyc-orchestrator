"""OpenAI-compatible endpoints for the orchestrator API.

These endpoints allow tools like Aider, LM Studio, and other OpenAI-compatible
clients to use our orchestrator backend for inference.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from base64 import b64decode
from binascii import Error as Base64Error
from dataclasses import dataclass
from typing import Any, AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from src.api.dependencies import dep_app_state
from src.api.models import (
    OpenAIChatRequest,
    OpenAIChatResponse,
    OpenAIChoice,
    OpenAIMessage,
    OpenAIModelInfo,
    OpenAIModelsResponse,
    OpenAIUsage,
)
from src.api.routes.chat_pipeline.routing_decision import normalize_ingress_role
from src.api.state import AppState
from src.prompt_builders import (
    build_root_lm_prompt,
    extract_code_from_response,
    auto_wrap_final,
)
from src.registry.stack_priors import (
    live_stack_role_records,
    stack_prior_primary_port,
    stack_prior_serving,
)
from src.repl_environment import REPLEnvironment
from src.roles import Role

logger = logging.getLogger(__name__)

router = APIRouter()


@dataclass(frozen=True)
class _OpenAIContentParts:
    text: str
    image_base64: str | None = None


def _parse_image_data_url(url: str) -> str:
    header, sep, payload = url.partition(",")
    header_l = header.lower()
    if sep != "," or not header_l.startswith("data:image/") or ";base64" not in header_l:
        raise ValueError(
            "OpenAI image_url content must use a data:image/...;base64 URL"
        )
    payload = payload.strip()
    try:
        b64decode(payload, validate=True)
    except (Base64Error, ValueError) as exc:
        raise ValueError("OpenAI image_url content contains invalid base64") from exc
    return payload


def _extract_openai_content(content: str | list | None, *, parse_images: bool) -> _OpenAIContentParts:
    """Extract text and, when requested, one data-URL image from OpenAI content."""
    if content is None:
        return _OpenAIContentParts(text="")
    if isinstance(content, str):
        return _OpenAIContentParts(text=content)
    if not isinstance(content, list):
        return _OpenAIContentParts(text="")

    text_parts: list[str] = []
    image_base64: str | None = None
    for part in content:
        if not isinstance(part, dict):
            continue
        part_type = part.get("type")
        if part_type == "text":
            text = part.get("text", "")
            if isinstance(text, str):
                text_parts.append(text)
            continue
        if part_type != "image_url" or not parse_images:
            continue

        raw_image_url = part.get("image_url")
        if isinstance(raw_image_url, dict):
            url = raw_image_url.get("url")
        else:
            url = raw_image_url
        if not isinstance(url, str) or not url:
            raise ValueError("OpenAI image_url content must include image_url.url")
        if image_base64 is not None:
            raise ValueError("Only one OpenAI image_url part is supported per request")
        image_base64 = _parse_image_data_url(url)

    return _OpenAIContentParts(text=" ".join(text_parts), image_base64=image_base64)


def _extract_text(content: str | list | None) -> str:
    """Extract text from OpenAI content field (string or multipart array)."""
    return _extract_openai_content(content, parse_images=False).text


def _history_message_dict(message: OpenAIMessage) -> dict[str, Any]:
    data: dict[str, Any] = {
        "role": message.role,
        "content": _extract_text(message.content) or "",
    }
    if message.tool_calls:
        data["tool_calls"] = message.tool_calls
    if message.tool_call_id:
        data["tool_call_id"] = message.tool_call_id
    if message.name:
        data["name"] = message.name
    return data


def _tool_function(tool: dict[str, Any]) -> dict[str, Any] | None:
    if tool.get("type") == "function":
        func = tool.get("function")
        return func if isinstance(func, dict) else None
    if "name" in tool:
        return tool
    return None


def _tool_choice_name(tool_choice: str | dict[str, Any] | None) -> str | None:
    if isinstance(tool_choice, str):
        return tool_choice
    if not isinstance(tool_choice, dict):
        return None
    func = tool_choice.get("function")
    if isinstance(func, dict) and isinstance(func.get("name"), str):
        return func["name"]
    if isinstance(tool_choice.get("name"), str):
        return tool_choice["name"]
    return None


def _format_tool_call(tool_call: dict[str, Any]) -> str:
    func = tool_call.get("function") if isinstance(tool_call, dict) else None
    func = func if isinstance(func, dict) else {}
    name = func.get("name") or tool_call.get("name") or "unknown_tool"
    args = func.get("arguments")
    if isinstance(args, (dict, list)):
        args_text = json.dumps(args, sort_keys=True)
    elif isinstance(args, str) and args:
        args_text = args
    else:
        args_text = "{}"
    call_id = tool_call.get("id")
    prefix = f"{call_id}: " if call_id else ""
    return f"{prefix}{name}({args_text})"


def _format_native_tools_for_repl(
    tools: list[dict[str, Any]] | None,
    tool_choice: str | dict[str, Any] | None,
) -> str | None:
    if not tools:
        return None
    choice = _tool_choice_name(tool_choice)
    if choice == "none":
        return None

    lines = [
        "OpenAI native tools were supplied by the caller.",
        "Use the existing REPL bridge to execute function tools as Python code:",
        '  result = CALL("tool_name", arg=value)',
        "Do not invent tool results; call the tool before FINAL when the answer depends on it.",
    ]
    if choice and choice not in {"auto", "none"}:
        lines.append(f"Tool choice policy: {choice}.")
    lines.append("Available function tools:")

    added = 0
    for tool in tools:
        func = _tool_function(tool)
        if not func:
            continue
        name = func.get("name")
        if not isinstance(name, str) or not name:
            continue
        desc = func.get("description")
        params = func.get("parameters")
        suffix = f" - {desc}" if isinstance(desc, str) and desc else ""
        lines.append(f"- {name}{suffix}")
        if isinstance(params, dict) and params:
            lines.append(f"  parameters: {json.dumps(params, sort_keys=True)}")
        added += 1

    if added == 0:
        return None
    return "\n".join(lines)


def _context_parts_from_history(
    history_messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    tool_choice: str | dict[str, Any] | None,
) -> list[str]:
    context_parts: list[str] = []
    for msg in history_messages:
        role = str(msg.get("role", "user"))
        content = str(msg.get("content", ""))
        tool_calls = msg.get("tool_calls")
        if role == "tool":
            label = msg.get("name") or msg.get("tool_call_id") or "tool"
            if content:
                context_parts.append(f"Tool result {label}: {content}")
            continue
        role_label = role.capitalize()
        if content:
            context_parts.append(f"{role_label}: {content}")
        if isinstance(tool_calls, list) and tool_calls:
            calls = "; ".join(
                _format_tool_call(tc) for tc in tool_calls if isinstance(tc, dict)
            )
            if calls:
                context_parts.append(f"{role_label} tool_calls: {calls}")

    native_tools = _format_native_tools_for_repl(tools, tool_choice)
    if native_tools:
        context_parts.append(native_tools)
    return context_parts


def _executed_tool_metadata(repl: Any | None) -> dict[str, Any]:
    """Return request-local internal REPL tool telemetry for OpenAI metadata."""
    if repl is None:
        return {"tools_used": 0, "tools_called": []}

    invocations = list(getattr(repl, "_invoked_tools", None) or [])
    tools_called: list[str] = []
    for invocation in invocations:
        name = getattr(invocation, "tool_name", None) or getattr(invocation, "name", None)
        if isinstance(name, str) and name:
            tools_called.append(name)

    try:
        repl_count = int(getattr(repl, "_tool_invocations", 0) or 0)
    except (TypeError, ValueError):
        repl_count = 0
    return {
        "tools_used": max(repl_count, len(tools_called)),
        "tools_called": tools_called,
    }


def _apply_openai_tool_contract_metadata(
    meta: dict[str, Any],
    *,
    request_tools: list[dict[str, Any]] | None,
    repl: Any | None,
) -> dict[str, Any]:
    tool_meta = _executed_tool_metadata(repl)
    if request_tools is not None:
        meta["native_tool_contract"] = "internal_repl_execution"
        meta["response_tool_calls"] = "not_emitted"
    if request_tools is not None or tool_meta["tools_used"]:
        meta.update(tool_meta)
    return meta


def _combined_prompt_with_context(prompt: str, context: str | None) -> str:
    if context:
        return f"{context}\n\nUser: {prompt}"
    return prompt


def _sampling_kwargs(request: OpenAIChatRequest) -> dict[str, Any]:
    """Return only caller-explicit sampling controls for downstream inference."""
    explicit_fields = getattr(request, "model_fields_set", set())
    kwargs: dict[str, Any] = {}
    if "temperature" in explicit_fields:
        kwargs["temperature"] = request.temperature
    if request.seed is not None:
        kwargs["seed"] = request.seed
    if request.top_p is not None:
        kwargs["top_p"] = request.top_p
    if request.top_k is not None:
        kwargs["top_k"] = request.top_k
    return kwargs


def _sampling_metadata(sampling_kwargs: dict[str, Any]) -> dict[str, Any]:
    if not sampling_kwargs:
        return {}
    return {"sampling": dict(sorted(sampling_kwargs.items()))}


def _role_name(role: str | Role) -> str:
    return role.value if isinstance(role, Role) else str(role)


async def _run_openai_vision_completion(
    *,
    prompt: str,
    context: str | None,
    image_base64: str,
    role: str | Role,
    primitives: Any,
    state: AppState,
    task_id: str,
) -> str:
    from src.api.models import ChatRequest
    from src.api.routes.chat_vision import _handle_vision_request

    role_id = _role_name(role)
    force_server = role_id if role_id in {"worker_vision", "vision_escalation"} else None
    vision_prompt = _combined_prompt_with_context(prompt or "Describe the image.", context)
    vision_request = ChatRequest(
        prompt=vision_prompt,
        mock_mode=False,
        real_mode=True,
        role=role_id,
        image_base64=image_base64,
    )
    return await _handle_vision_request(
        vision_request,
        primitives,
        state,
        task_id=task_id,
        force_server=force_server,
    )


COMPATIBILITY_MODEL_ALIASES = ("orchestrator", "architect", "worker")


def _canonical_role_name(role: str) -> str:
    canonical = normalize_ingress_role(role)
    if isinstance(canonical, Role):
        return canonical.value
    return str(canonical)


def _degraded_available_roles() -> list[str]:
    """Return degraded concrete roles when generated stack priors are absent.

    Concrete live roles intentionally do not fall back to stack_manifest
    constants here; those are launch inputs, not the /v1/models truth source.
    ``available_roles()`` still exposes compatibility aliases in degraded mode.
    """
    return []


def _ordered_live_role_ids(records: dict[str, dict]) -> list[str]:
    return [
        role
        for role, _record in sorted(
            records.items(),
            key=lambda item: (
                0 if item[0] == "frontdoor" else 1,
                stack_prior_primary_port(stack_prior_serving(item[1])) or 1_000_000,
                item[0],
            ),
        )
    ]


def _live_stack_role_ids() -> list[str]:
    """Read deployed role IDs from the generated stack-priors contract."""
    try:
        records = live_stack_role_records()
    except Exception as exc:
        logger.debug("Could not load stack priors for OpenAI models list: %s", exc)
        return []

    return _ordered_live_role_ids(records)


def available_roles() -> list[str]:
    """Return OpenAI-compatible model IDs from live stack truth plus aliases."""
    role_ids = [_canonical_role_name(role) for role in (_live_stack_role_ids() or _degraded_available_roles())]
    return list(dict.fromkeys([*COMPATIBILITY_MODEL_ALIASES, *role_ids]))


@router.get("/models", response_model=OpenAIModelsResponse)
async def list_models() -> OpenAIModelsResponse:
    """List available models (roles) in OpenAI format."""
    return OpenAIModelsResponse(data=[OpenAIModelInfo(id=role) for role in available_roles()])


@router.post("/chat/completions", response_model=None)
async def openai_chat_completions(
    request: OpenAIChatRequest,
    state: AppState = Depends(dep_app_state),
):
    """OpenAI-compatible chat completions endpoint.

    Supports both streaming and non-streaming modes.
    The 'model' field maps to orchestrator roles:
    - orchestrator: Auto-routing via frontdoor
    - frontdoor: Direct to frontdoor
    - coder: Direct to coder specialist
    - etc.

    For Aider integration:
    - Configure ~/.aider.conf.yml with openai-api-base: http://localhost:8000/v1
    - Aider will use this endpoint for all LLM calls
    """

    # Extract the last user message as the prompt
    user_messages = [m for m in request.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message provided")

    try:
        prompt_parts = _extract_openai_content(user_messages[-1].content, parse_images=True)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    prompt = prompt_parts.text

    # Build conversation context from message history
    # B2: Apply context compression on structured messages before flattening
    history_messages = list(request.messages[:-1])
    from src.features import features as _feat
    if _feat().context_compression and len(history_messages) > 8:
        try:
            from src.context_compression import ContextCompressor
            _compressor = ContextCompressor()
            _result = _compressor.compress(
                [_history_message_dict(m) for m in history_messages]
            )
            if _result.tool_outputs_summarized > 0 or _result.tool_pairs_fixed > 0:
                import logging
                logging.getLogger(__name__).info(
                    "B2 context compression: %d outputs summarized, %d pairs fixed",
                    _result.tool_outputs_summarized, _result.tool_pairs_fixed,
                )
            history_messages_dicts = _result.messages
        except Exception:
            history_messages_dicts = [_history_message_dict(m) for m in history_messages]
    else:
        history_messages_dicts = [_history_message_dict(m) for m in history_messages]

    context_parts = _context_parts_from_history(
        history_messages_dicts,
        request.tools,
        request.tool_choice,
    )
    context = "\n\n".join(context_parts) if context_parts else None

    # Map model to role — x_force_model > x_orchestrator_role > model field
    if request.x_force_model:
        role = request.x_force_model
    elif request.x_orchestrator_role:
        role = request.x_orchestrator_role
    elif request.model in ("orchestrator", "gpt-4", "gpt-3.5-turbo", "claude-3"):
        role = Role.FRONTDOOR
    else:
        role = request.model
    role = normalize_ingress_role(role)

    # Escalation cap and REPL disable flags — pass through to metadata
    max_escalation = request.x_max_escalation
    disable_repl = request.x_disable_repl
    sampling_kwargs = _sampling_kwargs(request)

    chat_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())

    # Determine if we should use real inference
    # Real mode requires: registry loaded AND mock_mode disabled via env
    from src.features import features
    from src.config import get_config

    f = features()
    use_real_mode = (
        state.registry is not None and not f.mock_mode  # Respect mock_mode feature flag
    )

    # Build real primitives with server_urls (matching /chat endpoint pattern)
    primitives = None
    if use_real_mode:
        try:
            from src.llm_primitives import LLMPrimitives

            server_urls = get_config().server_urls.as_dict()
            primitives = LLMPrimitives(
                mock_mode=False,
                server_urls=server_urls,
                registry=state.registry,
                health_tracker=state.health_tracker,
                admission_controller=getattr(state, "admission", None),
            )
        except Exception as e:
            logger.warning("Failed to create LLMPrimitives: %s", e)
            primitives = None

    if request.stream:
        # Streaming mode with real orchestration
        async def generate_stream() -> AsyncGenerator[str, None]:
            start_time = time.perf_counter()
            total_tokens = 0
            response_text = ""

            if not use_real_mode:
                # Mock mode fallback
                mock_response = f"[MOCK] Processed via {role}: {prompt[:100]}..."
                for i, char in enumerate(mock_response):
                    chunk = {
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": request.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": char}
                                if i > 0
                                else {"role": "assistant", "content": char},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                response_text = mock_response
            else:
                # Real orchestration with streaming
                repl_for_metadata: REPLEnvironment | None = None
                if primitives is None:
                    error_msg = "LLM primitives not initialized — check server_urls config"
                    chunk = {
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": request.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"role": "assistant", "content": error_msg},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                    response_text = error_msg

                if primitives:
                    # Build combined context
                    combined_context = _combined_prompt_with_context(prompt, context)

                    if prompt_parts.image_base64:
                        try:
                            response_text = await _run_openai_vision_completion(
                                prompt=prompt,
                                context=context,
                                image_base64=prompt_parts.image_base64,
                                role=role,
                                primitives=primitives,
                                state=state,
                                task_id=chat_id,
                            )
                        except Exception as e:
                            response_text = f"[ERROR] Vision request failed: {e}"
                        total_tokens = primitives.total_tokens_generated
                    elif disable_repl:
                        # Direct LLM call — no REPL, no code execution
                        try:
                            response_text = primitives.llm_call(
                                combined_context, role=role,
                                n_tokens=request.max_tokens,
                                **sampling_kwargs,
                            )
                        except Exception as e:
                            response_text = f"[ERROR] Direct call failed: {e}"
                        total_tokens = primitives.total_tokens_generated
                    else:
                        # Create REPL environment
                        repl = REPLEnvironment(
                            context=combined_context,
                            llm_primitives=primitives,
                            tool_registry=state.tool_registry,
                            script_registry=state.script_registry,
                            role=role,
                        )
                        repl_for_metadata = repl

                        # Run orchestration loop (simplified for streaming)
                        max_turns = request.max_tokens // 500 if request.max_tokens else 3
                        max_turns = min(max(max_turns, 1), 5)

                        for turn in range(max_turns):
                            repl_state = repl.get_state()
                            root_prompt = build_root_lm_prompt(
                                state=repl_state,
                                original_prompt=prompt,
                                last_output="",
                                last_error="",
                                turn=turn,
                            )

                            try:
                                code = primitives.llm_call(
                                    root_prompt,
                                    role=role,
                                    n_tokens=1024,
                                    **sampling_kwargs,
                                )
                                code = extract_code_from_response(code)
                                code = auto_wrap_final(code)
                            except Exception as e:
                                code = f'FINAL("Error during generation: {e}")'

                            # Execute in REPL
                            result = repl.execute(code)

                            if result.is_final:
                                response_text = result.final_answer or ""
                                break
                            elif result.output:
                                response_text = result.output
                        else:
                            # Max turns reached
                            response_text = response_text or f"[Completed {max_turns} turns]"

                        total_tokens = primitives.total_tokens_generated

                    # Stream the response character by character (OpenAI format)
                    first_chunk = True
                    for char in response_text:
                        chunk = {
                            "id": chat_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": request.model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"role": "assistant", "content": char}
                                    if first_chunk
                                    else {"content": char},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        first_chunk = False
                        if request.x_show_routing:
                            chunk["x_role"] = role
                        yield f"data: {json.dumps(chunk)}\n\n"

            # Final chunk with finish_reason
            final_chunk = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
            }
            if request.x_show_routing:
                meta = {
                    "role": role,
                    "elapsed_seconds": time.perf_counter() - start_time,
                    "tokens": total_tokens,
                }
                if max_escalation:
                    meta["max_escalation"] = max_escalation
                if disable_repl:
                    meta["repl_disabled"] = True
                meta.update(_sampling_metadata(sampling_kwargs))
                _apply_openai_tool_contract_metadata(
                    meta,
                    request_tools=request.tools,
                    repl=locals().get("repl_for_metadata"),
                )
                final_chunk["x_orchestrator_metadata"] = meta
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
    else:
        # Non-streaming mode with real orchestration
        start_time = time.perf_counter()
        total_tokens = 0

        if not use_real_mode:
            # Mock mode fallback
            response_text = f"[MOCK] Processed via {role}: {prompt[:100]}..."
            repl_for_metadata = None
        else:
            # Real orchestration
            repl_for_metadata = None
            try:
                if primitives is None:
                    raise HTTPException(
                        status_code=503,
                        detail="LLM primitives not initialized — check server_urls config",
                    )

                combined_context = _combined_prompt_with_context(prompt, context)

                if prompt_parts.image_base64:
                    response_text = await _run_openai_vision_completion(
                        prompt=prompt,
                        context=context,
                        image_base64=prompt_parts.image_base64,
                        role=role,
                        primitives=primitives,
                        state=state,
                        task_id=chat_id,
                    )
                elif disable_repl:
                    # Direct LLM call — no REPL, no code execution
                    response_text = primitives.llm_call(
                        combined_context, role=role,
                        n_tokens=request.max_tokens,
                        **sampling_kwargs,
                    )
                else:
                    repl = REPLEnvironment(
                        context=combined_context,
                        llm_primitives=primitives,
                        tool_registry=state.tool_registry,
                        script_registry=state.script_registry,
                        role=role,
                    )
                    repl_for_metadata = repl

                    max_turns = request.max_tokens // 500 if request.max_tokens else 3
                    max_turns = min(max(max_turns, 1), 5)

                    response_text = ""
                    for turn in range(max_turns):
                        repl_state = repl.get_state()
                        root_prompt = build_root_lm_prompt(
                            state=repl_state,
                            original_prompt=prompt,
                            last_output="",
                            last_error="",
                            turn=turn,
                        )

                        code = primitives.llm_call(
                            root_prompt,
                            role=role,
                            n_tokens=1024,
                            **sampling_kwargs,
                        )
                        code = extract_code_from_response(code)
                        code = auto_wrap_final(code)

                        result = repl.execute(code)

                        if result.is_final:
                            response_text = result.final_answer or ""
                            break
                        elif result.output:
                            response_text = result.output

                total_tokens = primitives.total_tokens_generated

            except Exception as e:
                response_text = f"[ERROR] Backend failed: {e}"

        elapsed = time.perf_counter() - start_time

        return OpenAIChatResponse(
            id=chat_id,
            created=created,
            model=request.model,
            choices=[
                OpenAIChoice(
                    index=0,
                    message=OpenAIMessage(role="assistant", content=response_text),
                    finish_reason="stop",
                )
            ],
            usage=OpenAIUsage(
                prompt_tokens=len(prompt) // 4,
                completion_tokens=total_tokens or len(response_text) // 4,
                total_tokens=(len(prompt) // 4) + (total_tokens or len(response_text) // 4),
            ),
            x_orchestrator_metadata=_apply_openai_tool_contract_metadata(
                {
                    "role": role,
                    "elapsed_seconds": elapsed,
                    **({"max_escalation": max_escalation} if max_escalation else {}),
                    **({"repl_disabled": True} if disable_repl else {}),
                    **_sampling_metadata(sampling_kwargs),
                },
                request_tools=request.tools,
                repl=repl_for_metadata,
            )
            if request.x_show_routing
            else None,
        )


@router.get("/models/{model_id}")
async def get_model(model_id: str) -> OpenAIModelInfo:
    """Get info for a specific model."""
    if model_id not in available_roles():
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    return OpenAIModelInfo(id=model_id)
