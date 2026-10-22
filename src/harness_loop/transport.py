"""I/O seam shared by the sibling loop's concrete parts (HIL-6h). Build-only; nothing here runs.

Every concrete proposer, judge and evaluator in this package takes an injected **transport**: a
plain callable that turns one request mapping into one response mapping. Unit tests pass fakes, so
no test can reach a network, a model, or a CLI. The factories below build the real transports a
later, separately authorized entrypoint (HIL-6j) would inject; this package never calls them.

* :func:`http_json_transport` POSTs JSON to the orchestrator API (``/v1/chat/completions`` for the
  editors, ``/chat`` for the evaluator). The base URL comes from the orchestrator config
  (``src.config.get_config().server_urls.api_url``), never from a hardcoded port.
* :func:`registry_role_model` reads which model a role serves from the compiled runtime registry,
  so the editor's model id (used by the P3 family checks) is read, not typed.
* :func:`token_count_from_usage` maps an OpenAI ``usage`` block to a :class:`TokenCount` without
  inventing anything: a count the server did not report stays ``None`` (HIL-20 addendum item 2).
* :class:`EgressClearance` is the explicit allow flag for content leaving the host (P3 caution 2).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from src.harness_loop.interfaces import TokenCount

#: One request mapping in, one response mapping out. A transport raises on a transport failure.
JsonTransport = Callable[[Mapping[str, Any]], Mapping[str, Any]]

V1_CHAT_PATH = "/v1/chat/completions"
CHAT_PATH = "/chat"

_REPO_ROOT = Path(__file__).resolve().parents[2]
#: The compiled lean runtime view (``orchestrator_stack.py start`` regenerates it from the master).
RUNTIME_REGISTRY = _REPO_ROOT / "orchestration" / "model_registry.yaml"

#: Egress label for a call that stays on the EPYC host (orchestrator ``/v1`` or ``/chat``).
LOCAL_EGRESS = "local (EPYC orchestrator)"


class TransportError(RuntimeError):
    """A transport could not deliver a request or returned no JSON object."""


class EgressRefused(PermissionError):
    """Content would leave the host for an egress it is not cleared for (P3 caution 2)."""


# ── configuration lookups (read-only) ─────────────────────────────────────────


def default_api_base_url() -> str:
    """The orchestrator API base URL from the orchestrator's own config (no hardcoded port)."""
    from src.config import get_config

    return str(get_config().server_urls.api_url).rstrip("/")


def registry_role_model(role: str, registry_path: Path | str | None = None) -> str | None:
    """``roles.<role>.model.name`` from the runtime registry, or ``None`` when absent.

    Read-only. ``None`` means "not found", never a guess; a caller that needs the id refuses.
    """
    import yaml

    path = Path(registry_path) if registry_path is not None else RUNTIME_REGISTRY
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except (OSError, yaml.YAMLError):
        return None
    entry = (data.get("roles") or {}).get(role)
    if not isinstance(entry, Mapping):
        return None
    model = entry.get("model")
    name = model.get("name") if isinstance(model, Mapping) else None
    return str(name) if name else None


# ── transports (built, never called by this package) ──────────────────────────


def http_json_transport(
    path: str,
    *,
    base_url: str | None = None,
    timeout_s: float = 600.0,
    client: Any = None,
) -> JsonTransport:
    """A JSON POST transport to ``<base_url><path>``. Building it opens nothing.

    ``client`` is an ``httpx.Client``-compatible object (tests pass one over
    ``httpx.MockTransport``); when absent, one short-lived client is opened per call.
    """

    def _post(body: Mapping[str, Any]) -> Mapping[str, Any]:
        import httpx

        url = f"{(base_url or default_api_base_url()).rstrip('/')}{path}"
        try:
            if client is not None:
                resp = client.post(url, json=dict(body), timeout=timeout_s)
            else:
                with httpx.Client(timeout=timeout_s) as c:
                    resp = c.post(url, json=dict(body))
        except httpx.HTTPError as exc:
            raise TransportError(f"POST {url} failed: {exc}") from exc
        try:
            data = resp.json()
        except ValueError as exc:
            raise TransportError(f"POST {url}: HTTP {resp.status_code}, body is not JSON") from exc
        if not isinstance(data, Mapping):
            raise TransportError(f"POST {url}: HTTP {resp.status_code}, JSON is not an object")
        if resp.status_code >= 400 and "eval_fence" not in data:
            raise TransportError(f"POST {url}: HTTP {resp.status_code}: {str(data)[:300]}")
        return data

    return _post


# ── token accounting ──────────────────────────────────────────────────────────


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    n = int(value)
    return n if n > 0 else None


def _non_negative_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    n = int(value)
    return n if n >= 0 else None


def token_count_from_usage(usage: Any, seat: str) -> TokenCount:
    """Project an OpenAI ``usage`` block onto one seat's :class:`TokenCount`.

    The orchestrator reports a count the backend did not report as ``0`` (``_client_usage``:
    "a consumer gating on >= 1 refuses an unmeasured turn"), so ``0`` maps to ``None`` here.
    ``prompt_tokens_details.cached_tokens`` is present only when the backend reported a KV-cache
    reuse count; without it the prompt cannot be split into uncached and cached prefill, so both
    stay ``None`` (``TokenCount.total()`` is then ``None``: "not captured", never zero).
    """
    if not isinstance(usage, Mapping):
        return TokenCount(seat=seat)
    prompt = _positive_int(usage.get("prompt_tokens"))
    decode = _positive_int(usage.get("completion_tokens"))
    details = usage.get("prompt_tokens_details")
    cached = _non_negative_int(details.get("cached_tokens")) if isinstance(details, Mapping) else None
    uncached = None
    if prompt is not None and cached is not None and cached <= prompt:
        uncached = prompt - cached
    else:
        cached = None
    return TokenCount(uncached_prefill=uncached, decode=decode, cached_prefill=cached, seat=seat)


# ── egress clearance (P3 caution 2) ───────────────────────────────────────────


@dataclass(frozen=True)
class EgressClearance:
    """An explicit, human-issued clearance for sending ``suite_id`` content over ``egress``.

    ``instance_ids`` ``None`` clears every instance of the suite; otherwise only those listed.
    The egress string must equal the registry entry's ``egress`` field byte for byte.
    """

    egress: str
    suite_id: str
    cleared_by: str
    instance_ids: frozenset[str] | None = None

    def __post_init__(self) -> None:
        if not (self.egress.strip() and self.suite_id.strip() and self.cleared_by.strip()):
            raise ValueError("an egress clearance names the egress, the suite and who cleared it")

    def uncovered(self, egress: str, instance_ids: Iterable[str] = ()) -> list[str]:
        """Reasons this clearance does not cover the call (empty == covered)."""
        out: list[str] = []
        if egress != self.egress:
            out.append(f"egress {egress!r} != cleared egress {self.egress!r}")
        if self.instance_ids is not None:
            missing = sorted(set(instance_ids) - self.instance_ids)
            if missing:
                out.append(f"instances not cleared for egress: {missing[:10]}")
        return out

    def as_record(self) -> dict[str, Any]:
        return {
            "egress": self.egress,
            "suite_id": self.suite_id,
            "cleared_by": self.cleared_by,
            "instance_scope": "suite" if self.instance_ids is None else sorted(self.instance_ids),
        }


def require_clearance(
    clearance: EgressClearance | None, egress: str, instance_ids: Iterable[str] = ()
) -> None:
    """Refuse unless ``clearance`` covers ``egress`` (and the instances). Local egress needs none."""
    if egress == LOCAL_EGRESS:
        return
    if clearance is None:
        raise EgressRefused(f"no egress clearance for {egress!r}; content stays on the host")
    reasons = clearance.uncovered(egress, instance_ids)
    if reasons:
        raise EgressRefused("; ".join(reasons))


# ── parsing ───────────────────────────────────────────────────────────────────


def extract_json_object(text: str) -> dict[str, Any] | None:
    """The JSON object in ``text`` (whole text, a fenced block, or the outermost braces)."""
    text = (text or "").strip()
    if not text:
        return None
    candidates = [text]
    if "```" in text:
        for chunk in text.split("```")[1::2]:
            chunk = chunk.strip()
            if chunk.lower().startswith("json"):
                chunk = chunk[4:].strip()
            candidates.append(chunk)
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        candidates.append(text[start : end + 1])
    for c in candidates:
        try:
            value = json.loads(c)
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    return None


def openai_message_text(response: Mapping[str, Any]) -> str | None:
    """``choices[0].message.content`` of an OpenAI chat-completions response, else ``None``."""
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        return None
    msg = choices[0].get("message")
    content = msg.get("content") if isinstance(msg, Mapping) else None
    return content if isinstance(content, str) else None


__all__ = [
    "CHAT_PATH",
    "LOCAL_EGRESS",
    "RUNTIME_REGISTRY",
    "V1_CHAT_PATH",
    "EgressClearance",
    "EgressRefused",
    "JsonTransport",
    "TransportError",
    "default_api_base_url",
    "extract_json_object",
    "http_json_transport",
    "openai_message_text",
    "registry_role_model",
    "require_clearance",
    "token_count_from_usage",
]
