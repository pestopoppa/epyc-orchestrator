"""Per-request speculative-decoding override for bitwise-required callers (Q38-T10).

Operator ruling 2026-10-20 (Q38-T10, per-role parity): eval, judge, replay and
machine-parsed roles must produce output identical to non-speculative decoding.
The minimal mechanism is llama-server's per-request ``speculative.n_max`` field
(``tools/server/server-schema.cpp``: a FLAT top-level key whose literal name
contains a dot; ``0`` disables speculation for that request).

Contract:
  * Nothing here changes any payload unless the feature flag
    ``spec_off_bitwise_roles`` is ON (default OFF) or a caller sets an override
    explicitly. With neither, ``apply_to_payload`` is a no-op: request bodies are
    byte-identical to before.
  * Serving paths never call ``bitwise_n_max`` / ``bitwise_scope``; they carry no
    override, so they are never affected.
  * Do NOT turn the flag on until Q38-T10.1's control arm has shown that
    per-request ``n_max: 0`` is token-identical to a spec-off server.
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
from typing import Any, Iterator

logger = logging.getLogger(__name__)

#: The llama-server request key (flat, dotted — NOT a nested ``speculative`` object).
SERVER_KEY = "speculative.n_max"

_OVERRIDE: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "spec_policy_n_max_override", default=None
)


def bitwise_roles_enabled() -> bool:
    """Whether the ``spec_off_bitwise_roles`` flag is ON (False on any error)."""
    try:
        from src.features import features

        return bool(features().spec_off_bitwise_roles)
    except Exception:  # fail to the unchanged (OFF) behaviour
        logger.debug("spec_off_bitwise_roles unreadable; treating as OFF", exc_info=True)
        return False


def bitwise_n_max() -> int | None:
    """``0`` when the flag is ON (bitwise-required caller must disable speculation), else None."""
    return 0 if bitwise_roles_enabled() else None


def bitwise_call_kwargs() -> dict[str, int]:
    """``{"speculative_n_max": 0}`` when the flag is ON, else ``{}`` (call left unchanged)."""
    n_max = bitwise_n_max()
    return {} if n_max is None else {"speculative_n_max": n_max}


def current_override() -> int | None:
    """The in-context override installed by ``speculative_n_max_override``, or None."""
    return _OVERRIDE.get()


@contextlib.contextmanager
def speculative_n_max_override(value: int | None) -> Iterator[None]:
    """Install ``value`` for backend calls made within this context (None = no-op)."""
    if value is None:
        yield
        return
    token = _OVERRIDE.set(int(value))
    try:
        yield
    finally:
        _OVERRIDE.reset(token)


def bitwise_scope() -> contextlib.AbstractContextManager[None]:
    """Context for an in-process bitwise-required caller: n_max 0 if flag ON, else no-op."""
    return speculative_n_max_override(bitwise_n_max())


def apply_to_payload(payload: dict[str, Any], request_value: int | None = None) -> None:
    """Set ``speculative.n_max`` on a llama-server payload iff an override applies.

    Precedence: explicit per-request value, then the in-context override. Absent
    both, the payload is left untouched (byte-identical to the legacy body).
    """
    value = request_value if request_value is not None else _OVERRIDE.get()
    if value is not None:
        payload[SERVER_KEY] = max(0, int(value))
