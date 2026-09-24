"""Domain-specific exceptions for the orchestrator system.

Replaces generic ``except Exception`` catches with typed exceptions that
allow callers to handle specific failure modes.  Infrastructure exceptions
(timeouts, connection failures) are distinguished from application-level
errors (bad input, delegation failure) so callers can retry vs. bail out.
"""


class OrchestratorError(Exception):
    """Base exception for all orchestrator errors."""


# -- Inference / LLM --------------------------------------------------------

class InferenceError(OrchestratorError):
    """LLM call failed (timeout, backend down, malformed response)."""


class InferenceTimeoutError(InferenceError):
    """LLM call timed out."""


class BackendUnavailableError(InferenceError):
    """Backend server is unreachable or returned 502/503."""


class ContextOverflowError(InferenceError, RuntimeError):
    """llama-server refused or aborted a request because the KV context ran out.

    Two server failures map here (frozen tree ``production-consolidated-v10``,
    ``tools/server/server-context.cpp``):

    * ``kind == "request_too_large"`` — HTTP 400, ``type:
      exceed_context_size_error``: the prompt alone does not fit the slot's
      ``n_ctx`` ("request (N tokens) exceeds the available context size (M
      tokens), try increasing it", :3303-3310; or "input (N tokens) is larger
      than the max context size (M tokens). skipping", :3292-3299). Retrying the
      same request on the same role can never succeed.
    * ``kind == "pool_exhausted"`` — HTTP 500 / mid-stream error, ``type:
      server_error``, message "Context size has been exceeded." (:3759-3764):
      ``llama_decode`` found no free KV cells even at batch size 1. Under a
      unified KV pool this is concurrency (other in-flight requests filled the
      shared pool) and every in-flight request fails with it; the request may
      well fit on its own.

    Subclasses ``RuntimeError`` as well as ``InferenceError`` so every existing
    ``except RuntimeError`` caller still treats it as an inference failure,
    while callers that care can catch the specific type.
    """

    REQUEST_TOO_LARGE = "request_too_large"
    POOL_EXHAUSTED = "pool_exhausted"

    def __init__(
        self,
        message: str,
        *,
        kind: str,
        role: str = "",
        backend_url: str = "",
        n_prompt_tokens: int | None = None,
        n_ctx: int | None = None,
        http_status: int | None = None,
        server_message: str = "",
        source: str = "server",
        recovery: list[dict] | None = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.role = role
        self.backend_url = backend_url
        self.n_prompt_tokens = n_prompt_tokens
        self.n_ctx = n_ctx
        self.http_status = http_status
        self.server_message = server_message
        # "server": the llama-server said so. "admission": the orchestrator's
        # shared-pool admission refused to send it (no server call was made).
        self.source = source
        # Ordered record of the recovery attempts already made (backoff
        # retries, reroutes), so the final error says what was tried.
        self.recovery: list[dict] = list(recovery or [])

    @property
    def retryable(self) -> bool:
        """True when waiting can help (pool pressure), False when it cannot."""
        return self.kind == self.POOL_EXHAUSTED

    def to_dict(self) -> dict:
        return {
            "error": "context_overflow",
            "kind": self.kind,
            "role": self.role,
            "backend_url": self.backend_url,
            "n_prompt_tokens": self.n_prompt_tokens,
            "n_ctx": self.n_ctx,
            "http_status": self.http_status,
            "server_message": self.server_message,
            "source": self.source,
            "recovery": list(self.recovery),
            "detail": str(self),
        }


# -- Delegation / Routing ---------------------------------------------------

class DelegationError(OrchestratorError):
    """Architect delegation failed (bad decision, specialist failure)."""


class DelegationLoopError(DelegationError):
    """Delegation entered a zero-progress loop."""


# -- Vision ------------------------------------------------------------------

class VisionAnalysisError(OrchestratorError):
    """Vision pipeline failed (OCR, VL inference, image processing)."""


# -- Archive / Document ------------------------------------------------------

class ArchiveExtractionError(OrchestratorError):
    """Archive extraction failed (corrupt, too large, unsupported format)."""


class DocumentProcessingError(OrchestratorError):
    """Document preprocessing or parsing failed."""


# -- REPL / Execution -------------------------------------------------------

class REPLExecutionError(OrchestratorError):
    """REPL code execution failed."""


class REPLTimeoutError(REPLExecutionError):
    """REPL execution timed out."""


# -- Configuration -----------------------------------------------------------

class ConfigurationError(OrchestratorError):
    """Configuration is invalid or missing required values."""


class RegistryError(OrchestratorError):
    """Model registry loading or validation failed."""
