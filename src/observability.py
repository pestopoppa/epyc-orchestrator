"""Shared observability helpers for degraded and suppressed execution paths."""

from __future__ import annotations

import json
import logging
from typing import Any


def classify_exception(exc: Exception) -> tuple[str, str]:
    """Return a stable ``(reason, detail)`` pair for an exception."""
    detail = f"{type(exc).__name__}: {exc}"

    try:
        import httpx

        if isinstance(exc, httpx.ConnectTimeout):
            return "connect_timeout", detail
        if isinstance(exc, httpx.ReadTimeout):
            return "read_timeout", detail
        if isinstance(exc, httpx.TimeoutException):
            return "timeout", detail
        if isinstance(exc, httpx.ConnectError):
            return "connect_error", detail
        if isinstance(exc, httpx.HTTPStatusError):
            return "http_status", detail
        if isinstance(exc, httpx.RequestError):
            return "request_error", detail
    except Exception:
        pass

    if isinstance(exc, ImportError):
        return "missing_dependency", detail
    if isinstance(exc, PermissionError):
        return "permission_error", detail
    if isinstance(exc, FileNotFoundError):
        return "file_not_found", detail
    if isinstance(exc, OSError):
        return "filesystem_error", detail
    if isinstance(exc, json.JSONDecodeError):
        return "invalid_json", detail
    if isinstance(exc, ValueError):
        return "invalid_value", detail
    if isinstance(exc, RuntimeError):
        return "runtime_error", detail
    return "unexpected_error", detail


def _merge_context(context: dict[str, Any] | None, extra: dict[str, Any]) -> dict[str, Any]:
    merged = dict(context or {})
    merged.update({k: v for k, v in extra.items() if v is not None and v != ""})
    return merged


def log_suppressed(
    logger: logging.Logger,
    message: str,
    exc: Exception,
    *,
    context: dict[str, Any] | None = None,
) -> tuple[str, str]:
    """Log a suppressed exception with structured context."""
    reason, detail = classify_exception(exc)
    logger.warning(
        "%s [%s]: %s",
        message,
        reason,
        detail,
        extra=_merge_context(context, {"failure_reason": reason, "failure_detail": detail}),
    )
    return reason, detail


def log_degradation(
    logger: logging.Logger,
    message: str,
    *,
    reason: str,
    detail: str = "",
    context: dict[str, Any] | None = None,
) -> None:
    """Log a degraded execution path."""
    logger.warning(
        "%s [%s]%s",
        message,
        reason,
        f": {detail}" if detail else "",
        extra=_merge_context(context, {"failure_reason": reason, "failure_detail": detail}),
    )


def log_partial_success(
    logger: logging.Logger,
    message: str,
    *,
    reason: str,
    detail: str = "",
    context: dict[str, Any] | None = None,
) -> None:
    """Log a partial-but-usable execution result."""
    logger.warning(
        "%s [%s]%s",
        message,
        reason,
        f": {detail}" if detail else "",
        extra=_merge_context(context, {"partial": True, "failure_reason": reason, "failure_detail": detail}),
    )
