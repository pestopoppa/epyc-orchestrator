"""The ``local:champion_sidecar`` backend of the coherence judge (2026-10-04).

Why it exists. Native single-token scoring (TD-29) reads the verdict's probability from
the server's post-sampling top-k. Production v10 serves its MTP / DFlash2 roles with
speculative decoding, and the v10 spec-accept path fills NO token probabilities
(TD-1d.5), so on production the judge can only score in the JSON arm. The CHAMPION
build carries SW-9 (``2b57340bf``) and does return them. Operator decision 2026-10-04:
point the judge at the champion (it becomes v11), not at production v10. No promotion.

What it talks to. The TD-29 champion SIDECAR: a CPU side instance of the champion
llama-server on a spare port, launched ONLY by
``/mnt/raid0/llm/tmp/champion-sidecar/launch_champion_sidecar.sh`` (window-gated:
AutoKernel CPU window open, or an announced DS41 pause; verifies binary sha256, version,
SW-9 ancestry and ggml linkage, then writes ``sidecar-<port>.state.json``). The
launcher's defaults: build ``/mnt/raid0/llm/kernels/builds/cpu-20260925-90c12df42``
(``llama-server 10308 (90c12df42)``), production frontdoor's model
``Qwen3.6-35B-A3B-MTP-Q8_0.gguf`` with frontdoor's live recipe (MTP speculation on,
``-np 4``), ``numactl --interleave=2,3 --physcpubind=48-87 -t 40`` (DS41-pause profile
``SIDECAR_CPUSET=48-71,80-87 SIDECAR_THREADS=32``), port 8199.

THIS MODULE NEVER STARTS, STOPS OR SIGNALS A PROCESS. It probes ``GET /health`` and
``GET /props`` (no inference), reads the launcher's state file, and when the sidecar is
down it refuses with the launch command for a human or the main session to run.

Scoring path = TD-29's: ``run_typed_decisions(mode="native")`` with single-token keys,
through an ``LLMPrimitives`` whose one role points at the sidecar URL over
``/v1/chat/completions`` (frontdoor's lane, as in TD-29's ``--server-url
frontdoor=<sidecar>``). The role is named :data:`SIDECAR_ROLE`, NOT ``frontdoor``: the
sidecar is not a topology instance, and under ``ORCHESTRATOR_PER_REGION_LOCKS=1`` a
``frontdoor`` call would take production frontdoor's CPU region locks (TD-29's driver ran
with region locks off and a private lock file for the same reason). An unknown role
resolves to no region lock; the judge's measurement-window guard covers the sidecar.

Prefill. The judge is the sidecar's only client, so every sidecar call is pinned to ONE
server slot (``id_slot``, ``ORCHESTRATOR_COHERENCE_JUDGE_SIDECAR_SLOT``, default 0, must
be < the launcher's ``-np 4``) with ``cache_prompt: true``: successive judgements of the
same base output reuse that slot's KV prefix (fixed head + prompt + base). Production
roles are NEVER pinned (pinning hurts the unified pool's admission); they keep the
server's own slot choice and llama-server's default ``cache_prompt: true``.

Champion identity. A sidecar counts as the champion only when ``/props`` ``build_info``
contains the expected champion commit: ``ORCHESTRATOR_COHERENCE_JUDGE_CHAMPION_COMMIT``,
else the launcher state file's ``server_commit`` (the launcher proved SW-9 is its
ancestor), else :data:`DEFAULT_CHAMPION_COMMIT`. The recorded build id is the live
``build_info`` (e.g. ``b10308-90c12df42``).

When v11 ships SW-9 in production, this backend becomes unnecessary: see the v11 note in
``coherence_judge.py``.
"""

from __future__ import annotations

import json
import os
import shlex
import threading
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

SIDECAR_BACKEND = "local:champion_sidecar"
SIDECAR_ROLE = "champion_sidecar"
SIDECAR_URL_ENV = "ORCHESTRATOR_COHERENCE_JUDGE_SIDECAR_URL"
SIDECAR_STATE_ENV = "ORCHESTRATOR_COHERENCE_JUDGE_SIDECAR_STATE"
SIDECAR_LAUNCHER_ENV = "ORCHESTRATOR_COHERENCE_JUDGE_SIDECAR_LAUNCHER"
CHAMPION_COMMIT_ENV = "ORCHESTRATOR_COHERENCE_JUDGE_CHAMPION_COMMIT"
SIDECAR_SLOT_ENV = "ORCHESTRATOR_COHERENCE_JUDGE_SIDECAR_SLOT"
DEFAULT_SIDECAR_SLOT = 0

DEFAULT_SIDECAR_URL = "http://127.0.0.1:8199"
SIDECAR_HOME = Path("/mnt/raid0/llm/tmp/champion-sidecar")
DEFAULT_LAUNCHER = SIDECAR_HOME / "launch_champion_sidecar.sh"
#: The SW-9-bearing champion build the launcher pins (EXPECT_COMMIT, 2026-09-25).
DEFAULT_CHAMPION_COMMIT = "90c12df42"
#: The launcher serves ``-np 4``; the prefix router must not allocate a slot id >= 4.
SIDECAR_NUM_SLOTS = 4
PROBE_TIMEOUT_S = 3.0

#: The state-file fields worth recording on a verdict (provenance, not identity).
_STATE_KEEP = (
    "schema",
    "pid",
    "port",
    "started_at",
    "server_commit",
    "sw9_commit",
    "binary_sha256",
    "build_dir",
    "props_build_info",
    "model",
    "cpuset",
    "threads",
    "numa_interleave",
    "window_mode",
)


def sidecar_url() -> str:
    return (os.environ.get(SIDECAR_URL_ENV) or DEFAULT_SIDECAR_URL).strip().rstrip("/")


def _port(url: str) -> int | None:
    try:
        return urlparse(url).port
    except ValueError:
        return None


def state_path(url: str | None = None) -> Path:
    override = os.environ.get(SIDECAR_STATE_ENV, "").strip()
    if override:
        return Path(override)
    return SIDECAR_HOME / f"sidecar-{_port(url or sidecar_url()) or 8199}.state.json"


def launch_command(url: str | None = None) -> str:
    """The exact command a human / the main session runs to bring the sidecar up."""
    launcher = os.environ.get(SIDECAR_LAUNCHER_ENV, "").strip() or str(DEFAULT_LAUNCHER)
    port = _port(url or sidecar_url())
    prefix = f"SIDECAR_PORT={port} " if port and port != 8199 else ""
    return f"{prefix}bash {shlex.quote(launcher)} start"


def _read_state(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def expected_champion_commit(state: dict[str, Any] | None) -> tuple[str, str]:
    """(commit, source) — env override, launcher state file, module default."""
    env = os.environ.get(CHAMPION_COMMIT_ENV, "").strip()
    if env:
        return env, "env"
    commit = str((state or {}).get("server_commit") or "").strip()
    if commit:
        return commit, "state_file"
    return DEFAULT_CHAMPION_COMMIT, "default"


@dataclass
class SidecarStatus:
    """What a probe saw. ``ready`` = reachable AND serving the champion build."""

    url: str
    reachable: bool
    champion: bool
    reason: str
    build_info: str | None = None
    expected_commit: str | None = None
    expected_commit_source: str | None = None
    served_model: str | None = None
    state_path: str | None = None
    state: dict[str, Any] | None = None
    launch_command: str = ""
    props_keys: list[str] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return self.reachable and self.champion

    @property
    def build_id(self) -> str | None:
        return self.build_info

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["ready"] = self.ready
        out.pop("props_keys", None)
        return out


HttpGet = Callable[[str, float], "tuple[int, Any]"]


def _http_get(url: str, timeout_s: float) -> tuple[int, Any]:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as resp:  # noqa: S310 - localhost probe
            raw = resp.read().decode("utf-8", "replace")
            status = resp.status
    except urllib.error.HTTPError as exc:
        return exc.code, None
    try:
        return status, json.loads(raw) if raw else None
    except ValueError:
        return status, None


def probe_sidecar(
    url: str | None = None,
    *,
    http_get: HttpGet | None = None,
    state_file: Path | None = None,
    timeout_s: float = PROBE_TIMEOUT_S,
) -> SidecarStatus:
    """Probe ``/health`` + ``/props`` (no inference). Never raises, never starts anything."""
    url = (url or sidecar_url()).rstrip("/")
    get = http_get or _http_get
    path = state_file or state_path(url)
    raw_state = _read_state(path)
    state = {k: raw_state.get(k) for k in _STATE_KEEP if k in raw_state} if raw_state else None
    commit, source = expected_champion_commit(raw_state)
    status = SidecarStatus(
        url=url,
        reachable=False,
        champion=False,
        reason="",
        expected_commit=commit,
        expected_commit_source=source,
        state_path=str(path),
        state=state,
        launch_command=launch_command(url),
    )
    try:
        code, body = get(url + "/health", timeout_s)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        status.reason = f"sidecar {url} not reachable ({type(exc).__name__}: {exc})"
        return status
    if code != 200:
        # llama-server answers 503 {"error": "Loading model"} while it loads.
        status.reason = f"sidecar {url} /health returned HTTP {code} (not serving yet)"
        return status
    try:
        code, props = get(url + "/props", timeout_s)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        status.reason = f"sidecar {url} /props failed ({type(exc).__name__}: {exc})"
        return status
    if code != 200 or not isinstance(props, dict):
        status.reason = f"sidecar {url} /props returned HTTP {code} without a JSON object"
        return status
    status.reachable = True
    status.props_keys = sorted(props)
    build_info = props.get("build_info")
    status.build_info = str(build_info) if build_info else None
    model_path = props.get("model_path")
    if model_path:
        status.served_model = os.path.basename(str(model_path))
    elif state and state.get("model"):
        status.served_model = os.path.basename(str(state["model"]))
    if not status.build_info:
        status.reason = f"sidecar {url} /props carries no build_info; cannot prove the champion build"
        return status
    if commit not in status.build_info:
        status.reason = (
            f"sidecar {url} serves build {status.build_info!r}, not the champion commit "
            f"{commit} ({source})"
        )
        return status
    status.champion = True
    status.reason = f"sidecar {url} serves champion build {status.build_info}"
    return status


# ---------------------------------------------------------------------------
# Primitives bound to the sidecar (built once per URL, no process side effects)
# ---------------------------------------------------------------------------


def sidecar_slot() -> int:
    raw = os.environ.get(SIDECAR_SLOT_ENV, "").strip()
    try:
        slot = int(raw) if raw else DEFAULT_SIDECAR_SLOT
    except ValueError as exc:
        raise ValueError(f"{SIDECAR_SLOT_ENV}={raw!r} is not an integer") from exc
    if not 0 <= slot < SIDECAR_NUM_SLOTS:
        raise ValueError(f"{SIDECAR_SLOT_ENV}={slot} outside the sidecar's slots 0..{SIDECAR_NUM_SLOTS - 1}")
    return slot


class PinnedSlotBackend:
    """Wraps the sidecar's backend: every request goes to slot ``slot`` with cache_prompt.

    Sets ``slot_id`` + ``pin_slot`` (the opt-in the chat lane needs to forward
    ``id_slot``) and ``cache_prompt=True`` on each request; everything else delegates.
    """

    def __init__(self, backend: Any, slot: int) -> None:
        self.backend = backend  # attribute name native._backend_base_url unwraps
        self.slot = int(slot)

    def _pin(self, request: Any) -> Any:
        from dataclasses import is_dataclass, replace

        if is_dataclass(request):
            return replace(request, slot_id=self.slot, pin_slot=True, cache_prompt=True)
        for name, value in (("slot_id", self.slot), ("pin_slot", True), ("cache_prompt", True)):
            setattr(request, name, value)
        return request

    def infer(self, role_config: Any, request: Any) -> Any:
        return self.backend.infer(role_config, self._pin(request))

    def infer_stream_text(self, role_config: Any, request: Any, on_chunk: Any = None) -> Any:
        return self.backend.infer_stream_text(role_config, self._pin(request), on_chunk=on_chunk)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.backend, name)

_PRIMITIVES: dict[str, Any] = {}
_PRIMITIVES_LOCK = threading.Lock()


def _force_chat_completions(backend: Any) -> int:
    """Set ``use_chat_completions`` on every concrete server config under ``backend``."""
    flipped = 0
    queue, seen, configs = [backend], set(), set()
    while queue:
        node = queue.pop(0)
        if node is None or id(node) in seen:
            continue
        seen.add(id(node))
        config = getattr(node, "config", None)
        if config is not None and hasattr(config, "use_chat_completions") and id(config) not in configs:
            configs.add(id(config))
            config.use_chat_completions = True
            flipped += 1
        for attr in ("backend", "_backend"):
            queue.append(getattr(node, attr, None))
    return flipped


def build_sidecar_primitives(url: str) -> Any:
    """``LLMPrimitives`` with ONE role, :data:`SIDECAR_ROLE`, at ``url`` (cached per URL).

    Constructing it opens no connection and starts nothing; the first ``llm_call``
    is the first request. ``server_urls_source="request"`` keeps the fleet layer off
    this caller-supplied URL.
    """
    url = url.rstrip("/")
    with _PRIMITIVES_LOCK:
        cached = _PRIMITIVES.get(url)
        if cached is not None:
            return cached
        from src.llm_primitives import LLMPrimitives

        primitives = LLMPrimitives(
            mock_mode=False,
            server_urls={SIDECAR_ROLE: url},
            num_slots=SIDECAR_NUM_SLOTS,
            server_urls_source="request",
        )
        backends = getattr(primitives, "_backends", None) or {}
        backend = backends.get(SIDECAR_ROLE)
        if backend is None or not _force_chat_completions(backend):
            raise RuntimeError(f"could not build a chat-completions backend for the sidecar at {url}")
        backends[SIDECAR_ROLE] = PinnedSlotBackend(backend, sidecar_slot())
        primitives.cache_prompt = True  # this instance is private to the judge
        _PRIMITIVES[url] = primitives
        return primitives


__all__ = [
    "CHAMPION_COMMIT_ENV",
    "DEFAULT_CHAMPION_COMMIT",
    "DEFAULT_SIDECAR_URL",
    "SIDECAR_BACKEND",
    "PinnedSlotBackend",
    "SIDECAR_ROLE",
    "SIDECAR_SLOT_ENV",
    "SIDECAR_URL_ENV",
    "SidecarStatus",
    "build_sidecar_primitives",
    "expected_champion_commit",
    "launch_command",
    "probe_sidecar",
    "sidecar_slot",
    "sidecar_url",
    "state_path",
]
