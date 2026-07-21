"""Backend management for LLM primitives."""

import logging
from typing import Any, Mapping

_log = logging.getLogger(__name__)


def _normalise_role_urls(url_str: str) -> tuple[str, ...]:
    """Return comparable server URLs with the `full:` marker stripped."""
    urls = [u.strip() for u in url_str.split(",") if u.strip()]
    if urls and urls[0].startswith("full:"):
        urls[0] = urls[0][len("full:"):]
    return tuple(urls)


def _infer_topology_role_for_urls(
    role: str,
    role_urls: Mapping[str, tuple[str, ...]],
    topology_roles: set[str],
) -> str:
    """Resolve logical alias roles to the physical topology role they share.

    Some config roles are logical aliases over the same llama-server pool
    (for example coder_escalation shares frontdoor ports). The region-lock
    topology is keyed by the physical pool role, so those aliases must dispatch
    under the canonical topology role while keeping their logical role for
    routing/tap metadata.
    """
    if role in topology_roles:
        return role
    urls = role_urls.get(role, ())
    if not urls:
        return role
    for candidate, candidate_urls in role_urls.items():
        if candidate == role:
            continue
        if candidate in topology_roles and candidate_urls == urls:
            return candidate
    return role


class BackendMixin:
    """Mixin for backend initialization and access methods."""

    def _init_caching_backends(self, server_urls: dict[str, str], num_slots: int) -> None:
        """Initialize CachingBackend instances for each role.

        Args:
            server_urls: Dict mapping role names to llama-server URLs.
                         Values may be comma-separated for multi-instance roles
                         (e.g. "http://localhost:8080,http://localhost:8180").
            num_slots: Number of slots per server.
        """
        try:
            from src.backends.llama_server import LlamaServerBackend, ServerConfig
            from src.backends.round_robin import RoundRobinBackend
            from src.backends.concurrency_aware import ConcurrencyAwareBackend
            from src.prefix_cache import CachingBackend, PrefixRouter

            # 2026-05-23: roles in this env var route through
            # /v1/chat/completions instead of /completion. Used for models
            # whose chat template needs server-side jinja application
            # (gemma-4 multi-channel format, etc.). Comma-separated names.
            # Default: worker_general (gemma-4-26B-A4B-it).
            # J12 (2026-05-26): frontdoor/coder_escalation/architect_general route through
            # /v1/chat/completions so their registry chat_template_kwargs.enable_thinking=false
            # applies (load-bearing, feedback_qwen3x_enable_thinking_false). On /completion the
            # kwarg is inert and Qwen3.6/3.5 emit <think> blocks (J1: degenerate 1390-token
            # thinking on hard prompts). VERIFIED (max_turns=4, reading `answer`): chat-completions
            # frontdoor answers a moderate prompt in 270 tokens / 1 turn, no <think>, vs /completion
            # 2643 tokens / 4 turns thinking-on — ~10x fewer tokens, single turn. (An earlier
            # "0 tokens" reading was a max_turns=1 probe artifact, not a backend bug.)
            # ingest_long_context is EXCLUDED — thinking-on is load-bearing for Qwen3-Next-80B.
            from src.chat_completions_roles import chat_completions_roles
            _chat_completion_roles = chat_completions_roles()  # shared SoT (was a divergent inline default)
            if _chat_completion_roles:
                _log.info(
                    "Roles using /v1/chat/completions backend: %s",
                    sorted(_chat_completion_roles),
                )

            normalized_role_urls = {
                role: _normalise_role_urls(url_str)
                for role, url_str in server_urls.items()
            }
            try:
                from src.runtime.instance_topology import get_instance_regions

                topology_roles = {
                    role
                    for role, _idx in get_instance_regions().keys()
                }
            except Exception:
                topology_roles = set()

            for role, url_str in server_urls.items():
                urls = [u.strip() for u in url_str.split(",") if u.strip()]
                topology_role = _infer_topology_role_for_urls(
                    role,
                    normalized_role_urls,
                    topology_roles,
                )

                # Pre-warm convention: first URL prefixed with "full:" denotes
                # the full-speed (1×96t) instance for ConcurrencyAwareBackend.
                # Remaining URLs are quarter (48t) instances.
                has_full = urls and urls[0].startswith("full:")
                if has_full:
                    full_url = urls[0][len("full:"):]
                    quarter_urls = urls[1:]
                else:
                    full_url = None
                    quarter_urls = urls

                _use_cc = role in _chat_completion_roles
                if _use_cc:
                    _log.info("role %s → /v1/chat/completions backend (server-side jinja)", role)

                if has_full and quarter_urls:
                    def _port_of(u: str) -> int:
                        return int(u.rsplit(":", 1)[-1]) if ":" in u else 0

                    def _mk_backend(u: str):
                        return CachingBackend(
                            LlamaServerBackend(ServerConfig(
                                base_url=u, num_slots=num_slots,
                                use_chat_completions=_use_cc,
                            )),
                            PrefixRouter(num_slots=num_slots),
                        )

                    full_port = _port_of(full_url)

                    # DISPATCH-A2: resolve each endpoint's TRUE topology index by
                    # PORT against NUMA_CONFIG. When the endpoint wired as `full:`
                    # is NOT the topology idx-0 port (a quarter impersonating the
                    # 96-core full — e.g. a quarters-only serving stack whose first
                    # port got the `full:` marker), DEMOTE it into the quarters
                    # pool at its true index rather than stranding it. This (a)
                    # makes all live quarters dispatchable (restores the N-way
                    # ceiling) and (b) makes every quarter's region lock match the
                    # server's physical cores (kills the q0-lock-on-q1-cores
                    # cross-role collision hazard). Only demote when the true index
                    # is resolvable; otherwise keep today's behavior (the
                    # ConcurrencyAwareBackend alignment guard suppresses the full).
                    from src.runtime.instance_topology import (
                        full_instance_port,
                        topology_idx_for_port,
                    )
                    idx0_port = full_instance_port(topology_role)
                    demoted_idx = (
                        topology_idx_for_port(topology_role, full_port)
                        if (idx0_port is not None and full_port != idx0_port)
                        else None
                    )

                    if demoted_idx is not None:
                        quarter_url_list = [full_url] + list(quarter_urls)
                        full_backend = None
                        ca_full_port = 0
                        _log.warning(
                            "role %s: endpoint %s wired as full: is NOT the "
                            "NUMA_CONFIG idx-0 port (%s) for topology role %s — "
                            "DEMOTED into the quarters pool at topology idx %d so "
                            "all %d quarters are dispatchable and each region lock "
                            "matches its physical cores. No full instance served.",
                            role, full_url, idx0_port, topology_role,
                            demoted_idx, len(quarter_url_list),
                        )
                    else:
                        quarter_url_list = list(quarter_urls)
                        full_backend = _mk_backend(full_url)
                        ca_full_port = full_port

                    quarter_backends = []
                    quarter_topology_idxs: list[int] = []
                    for pos, url in enumerate(quarter_url_list):
                        quarter_backends.append(_mk_backend(url))
                        t_idx = topology_idx_for_port(topology_role, _port_of(url))
                        quarter_topology_idxs.append(t_idx if t_idx is not None else pos + 1)

                    self._backends[role] = ConcurrencyAwareBackend(
                        full_backend, quarter_backends,
                        role=role,
                        full_port=ca_full_port,
                        topology_role=topology_role,
                        quarter_topology_idxs=quarter_topology_idxs,
                    )
                    if topology_role != role:
                        _log.info(
                            "Concurrency-aware backend for %s uses topology/lock role %s",
                            role, topology_role,
                        )
                    _log.info(
                        "Concurrency-aware backend for %s: %d full + %d quarters (topo idxs %s)",
                        role, 1 if full_backend is not None else 0,
                        len(quarter_backends), quarter_topology_idxs,
                    )
                elif len(quarter_urls) > 1:
                    # Multi-instance role without full: round-robin
                    backends = []
                    for url in quarter_urls:
                        config = ServerConfig(
                            base_url=url, num_slots=num_slots,
                            use_chat_completions=_use_cc,
                        )
                        backend = LlamaServerBackend(config)
                        router = PrefixRouter(num_slots=num_slots)
                        backends.append(CachingBackend(backend, router))
                    self._backends[role] = RoundRobinBackend(backends, role=role)
                    _log.info("Round-robin backend for %s: %d instances", role, len(quarter_urls))
                else:
                    # Single-instance role
                    url = quarter_urls[0] if quarter_urls else url_str
                    config = ServerConfig(
                        base_url=url, num_slots=num_slots,
                        use_chat_completions=_use_cc,
                    )
                    backend = LlamaServerBackend(config)
                    router = PrefixRouter(num_slots=num_slots)
                    self._backends[role] = CachingBackend(backend, router)

        except ImportError as e:
            _log.warning("CachingBackend not available: %s. Using legacy mode.", e)

    def get_backend(self, role: str) -> Any | None:
        """Get the CachingBackend for a role.

        Args:
            role: Role name (e.g., "worker", "coder", "frontdoor").

        Returns:
            CachingBackend instance or None if not configured.
        """
        return self._backends.get(role)

    def get_cache_stats(self) -> dict[str, dict[str, Any]]:
        """Get cache statistics for all backends.

        Returns:
            Dict mapping role to cache stats dict.
        """
        stats = {}
        for role, backend in self._backends.items():
            if hasattr(backend, "get_stats"):
                stats[role] = backend.get_stats()
        return stats
