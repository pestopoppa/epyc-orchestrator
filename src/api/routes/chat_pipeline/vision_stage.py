"""Pipeline stage 6: Vision preprocessing + multimodal VL routing.

Runs DocumentPreprocessor to extract text, sections, and figures from
images/documents. Stores results on routing.document_result so that
_execute_repl() can use DocumentREPLEnvironment.

Stage 7.5 (_execute_vision_multimodal): Routes vision-role requests through
the actual multimodal VL handler instead of text-only paths. Without this,
_execute_direct/_execute_repl discard image data and VL models answer blind.
"""

from __future__ import annotations

import base64
import logging
import time
from pathlib import Path

from src.api.models import ChatRequest, ChatResponse
from src.api.routes.chat_pipeline.telemetry import (
    llm_completion_meta,
    work_completion_meta,
)
from src.api.routes.chat_utils import RoutingResult
from src.api.services.memrl import score_completed_task
from src.api.routes.vision_serving import (
    VISION_ROLES as _LEGACY_VISION_ROLES,
    fallback_vl_port_for_role as _fallback_vl_port_for_role,  # noqa: F401 - test compat
    stack_prior_vl_ports as _shared_stack_prior_vl_ports,
    vl_port_for_role as _shared_vl_port_for_role,
    vision_roles as _shared_vision_roles,
)
from src.api.structured_logging import task_extra
from src.llm_primitives import LLMPrimitives

_DEFAULT_STACK_PRIORS_PATH = (
    Path(__file__).resolve().parents[4] / "orchestration" / "derived" / "stack_priors.yaml"
)

log = logging.getLogger(__name__)
_VISION_ROLES = _LEGACY_VISION_ROLES


def _stack_prior_vl_ports(stack_priors_path: Path = _DEFAULT_STACK_PRIORS_PATH) -> dict[str, int]:
    ports = _shared_stack_prior_vl_ports(stack_priors_path)
    if not ports:
        log.warning("Using fallback VL ports; no live stack-prior VL ports found")
    return ports


def _vision_roles(stack_priors_path: Path = _DEFAULT_STACK_PRIORS_PATH) -> frozenset[str]:
    return _shared_vision_roles(stack_priors_path)


def _vl_port_for_role(
    role: str,
    stack_priors_path: Path = _DEFAULT_STACK_PRIORS_PATH,
) -> int:
    return _shared_vl_port_for_role(role, stack_priors_path)


async def _execute_vision(
    request: ChatRequest,
    routing: RoutingResult,
    primitives: LLMPrimitives,
    state,
    start_time: float,
) -> ChatResponse | None:
    """Preprocess vision inputs through the document pipeline.

    Instead of answering directly, this stage runs DocumentPreprocessor to
    extract text, sections, and figures from the image/document. The result
    is stored on routing.document_result so that _execute_repl() can use
    DocumentREPLEnvironment with structured document access tools.

    Returns None to fall through to REPL mode (the normal path).
    Only returns ChatResponse on unrecoverable errors.
    """
    has_vision_input = request.image_path or request.image_base64 or request.files
    if not (request.real_mode and has_vision_input):
        return None

    from src.services.document_preprocessor import (
        DocumentPreprocessor,
        PreprocessingConfig,
    )

    config = PreprocessingConfig(
        extract_figures=True,
        describe_figures=True,
    )
    preprocessor = DocumentPreprocessor(config=config)

    try:
        if request.image_path:
            log.info(
                "Vision preprocessing file: %s",
                request.image_path,
                extra=task_extra(
                    task_id=routing.task_id, stage="execute", mode="vision_preprocess"
                ),
            )
            result = await preprocessor.preprocess_file(request.image_path)
        elif request.image_base64:
            log.info(
                "Vision preprocessing base64 image",
                extra=task_extra(
                    task_id=routing.task_id, stage="execute", mode="vision_preprocess"
                ),
            )
            task_ir = {"inputs": [{"type": "base64", "value": request.image_base64}]}
            result = await preprocessor.preprocess(task_ir)
        elif request.files:
            log.info(
                "Vision preprocessing %d files",
                len(request.files),
                extra=task_extra(
                    task_id=routing.task_id, stage="execute", mode="vision_preprocess"
                ),
            )
            task_ir = {"inputs": [{"type": "path", "value": f} for f in request.files]}
            result = await preprocessor.preprocess(task_ir)
        else:
            result = None

        if result and result.success and result.document_result:
            routing.document_result = result
            log.info(
                "Document preprocessing succeeded: %d sections, %d figures",
                len(result.document_result.sections),
                len(result.document_result.figures),
                extra=task_extra(
                    task_id=routing.task_id, stage="execute", mode="vision_preprocess"
                ),
            )
            return None  # Fall through to REPL with document context

        # Preprocessing returned but without usable document result
        warn_msg = result.error if result else "unknown"
        log.warning(
            "Document preprocessing failed: %s",
            warn_msg,
            extra=task_extra(
                task_id=routing.task_id,
                stage="execute",
                mode="vision_preprocess",
                error_type="preprocess_failed",
            ),
        )

    except Exception as e:
        log.warning(
            "Vision preprocessing exception: %s: %s",
            type(e).__name__,
            e,
            extra=task_extra(
                task_id=routing.task_id,
                stage="execute",
                mode="vision_preprocess",
                error_type=type(e).__name__,
            ),
        )

    # Preprocessing failed — inject context note and fall through to text modes
    image_ref = request.image_path or "(base64 image)"
    request.context = (request.context or "") + (
        f"\n\n[IMAGE: {image_ref} — Document pipeline failed. Answering without OCR context.]"
    )
    return None  # Fall through to standard orchestration


def _vision_unavailable_response(
    request: ChatRequest,
    routing: RoutingResult,
    initial_role,
    execution_mode: str,
    start_time: float,
    detail: str,
) -> ChatResponse:
    """Build an in-band vision-unavailable error response for an image request.

    A request that CARRIES image data must NEVER silently fall through to a
    text-only path when the multimodal handler fails: the VL model would then
    answer BLIND (image discarded), and a blind answer is an ordinary string
    the eval SCORES as wrong — the mechanism behind the vl-suite 0/376. Instead
    emit an in-band ``[ERROR: vision_unavailable: ...]`` marker. The eval
    tower's REL-1 ``_inband_error_text`` guard keys on the ``[ERROR:`` prefix
    and EXCLUDES the row honestly (attributable failure) rather than scoring a
    wrong answer, and interactive callers see an explicit failure code. The
    detail carries the exception class + a truncated message for triage.
    """
    elapsed = time.perf_counter() - start_time
    answer = f"[ERROR: vision_unavailable: {detail}]"
    return ChatResponse(
        answer=answer,
        turns=1,
        tokens_used=0,
        elapsed_seconds=elapsed,
        mock_mode=False,
        real_mode=request.real_mode,
        routed_to=str(initial_role),
        role_history=[str(initial_role)],
        routing_strategy=routing.routing_strategy,
        mode=execution_mode,
        tokens_generated=0,
        formalization_applied=routing.formalization_applied,
        skills_retrieved=len(routing.skill_ids),
        skill_ids=routing.skill_ids,
        # "unavailable" → 503 under _annotate_error; set explicitly so the
        # response is self-describing for any caller that skips finalization.
        error_code=503,
        error_detail=answer,
    )


async def _execute_vision_multimodal(
    request: ChatRequest,
    routing: RoutingResult,
    primitives: LLMPrimitives,
    state,
    start_time: float,
    initial_role,
    execution_mode: str,
) -> ChatResponse | None:
    """Route vision-role requests through multimodal VL handlers.

    Text-only paths (_execute_direct, _execute_repl) discard image data.
    When a vision role has image data, this routes through:
    - _handle_vision_request (direct mode): OCR + multimodal VL completion
    - _vision_react_mode_answer (repl mode): multimodal ReAct tool loop

    Returns None only when this is NOT a vision-with-image request (the caller
    then handles it as ordinary text). When the request DOES carry image data
    but the multimodal handler fails, it returns an in-band
    ``[ERROR: vision_unavailable: ...]`` ChatResponse instead of falling
    through to a blind text answer (see ``_vision_unavailable_response``).
    """
    if str(initial_role) not in _vision_roles():
        return None
    if not (request.image_path or request.image_base64):
        return None
    # Past this point the request carries image data destined for a vision
    # role: it must never silently degrade to text-only (blind answering).

    from src.api.routes.chat_vision import (
        _handle_vision_request,
        _vision_react_mode_answer,
    )

    task_id = routing.task_id
    tools_used = 0
    tools_called: list[str] = []

    try:
        if execution_mode == "repl":
            # Vision ReAct: multimodal tool loop with image
            image_b64 = request.image_base64
            if not image_b64 and request.image_path:
                from src.api.routes.path_validation import validate_api_path

                img_path = validate_api_path(request.image_path)
                if not img_path.exists():
                    log.error(
                        "Vision image not found: %s — emitting in-band "
                        "vision_unavailable (NOT falling through to blind text)",
                        request.image_path,
                        extra=task_extra(
                            task_id=task_id,
                            role=str(initial_role),
                            stage="execute",
                            mode="vision_multimodal",
                            error_type="image_not_found",
                        ),
                    )
                    return _vision_unavailable_response(
                        request, routing, initial_role, execution_mode, start_time,
                        f"FileNotFoundError: {str(request.image_path)[:120]}",
                    )
                image_b64 = base64.b64encode(img_path.read_bytes()).decode("utf-8")

            if not image_b64:
                return _vision_unavailable_response(
                    request, routing, initial_role, execution_mode, start_time,
                    "ValueError: empty image payload",
                )

            # Detect MIME type from header bytes
            mime_type = "image/jpeg"
            try:
                raw = base64.b64decode(image_b64[:32])
                if raw[:4] == b"\x89PNG":
                    mime_type = "image/png"
                elif raw[:4] == b"RIFF":
                    mime_type = "image/webp"
            except Exception:
                pass

            vl_port = _vl_port_for_role(str(initial_role))
            answer, tools_used, tools_called = await _vision_react_mode_answer(
                prompt=request.prompt,
                image_b64=image_b64,
                mime_type=mime_type,
                context=request.context or "",
                vl_port=vl_port,
            )
        else:
            # Direct VL: multimodal completion (OCR + image + text → VL model)
            answer = await _handle_vision_request(
                request, primitives, state, task_id,
                force_server=str(initial_role),
            )

    except Exception as e:
        # The request carries image data (guaranteed above). The multimodal
        # handler exhausted its VL backends and raised. Do NOT return None:
        # that would fall through to the text-only stages, which discard the
        # image and let the model answer BLIND — a blind answer is scored, not
        # excluded, which is exactly the vl-suite 0/376 poisoning. Emit an
        # in-band vision_unavailable marker so the eval EXCLUDES the row.
        image_ref = request.image_path or "(base64 image)"
        detail = f"{type(e).__name__}: {str(e)[:120]}"
        vl_backend = None
        try:
            vl_backend = _vl_port_for_role(str(initial_role))
        except Exception:
            vl_backend = "unresolved"
        log.error(
            "Vision multimodal failed for image %s via %s (vl_port=%s): %s — "
            "emitting in-band vision_unavailable (NOT falling through to blind text)",
            image_ref,
            str(initial_role),
            vl_backend,
            detail,
            extra=task_extra(
                task_id=task_id,
                role=str(initial_role),
                stage="execute",
                mode="vision_multimodal",
                error_type=type(e).__name__,
            ),
        )
        return _vision_unavailable_response(
            request, routing, initial_role, execution_mode, start_time, detail,
        )

    elapsed = time.perf_counter() - start_time
    state.increment_request(mock_mode=False, turns=1)

    if state.progress_logger:
        state.progress_logger.log_task_completed(
            task_id=task_id,
            success=True,
            details=f"Vision multimodal ({initial_role}:{execution_mode}), {elapsed:.3f}s",
            completion_meta={
                "producer_role": str(initial_role),
                "delegation_lineage": [str(initial_role)],
                "final_answer_role": str(initial_role),
                **llm_completion_meta(primitives),
                # M-11a2b. The answer only — image bytes are NOT work payload and
                # must never enter the episodic store.
                **work_completion_meta(answer=answer),
            },
        )
        score_completed_task(
            state,
            task_id,
            force_role=request.force_role,
            real_mode=request.real_mode,
        )

    # Estimate tokens from answer length — vision backends don't expose
    # completion token counts, but word count is a reasonable proxy.
    tokens_est = len(answer.split()) if answer else 0

    return ChatResponse(
        answer=answer,
        turns=1,
        tokens_used=tokens_est,
        elapsed_seconds=elapsed,
        mock_mode=False,
        real_mode=request.real_mode,
        routed_to=str(initial_role),
        role_history=[str(initial_role)],
        routing_strategy=routing.routing_strategy,
        mode=execution_mode,
        tokens_generated=tokens_est,
        formalization_applied=routing.formalization_applied,
        tools_used=tools_used,
        tools_called=tools_called,
        skills_retrieved=len(routing.skill_ids),
        skill_ids=routing.skill_ids,
    )
