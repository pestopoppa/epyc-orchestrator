"""Backend management for LLM primitives."""

import logging
from typing import Any

_log = logging.getLogger(__name__)


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
            import os as _os
            # J12 (2026-05-26): frontdoor/coder_escalation/architect_general route through
            # /v1/chat/completions so their registry chat_template_kwargs.enable_thinking=false
            # applies (load-bearing, feedback_qwen3x_enable_thinking_false). On /completion the
            # kwarg is inert and Qwen3.6/3.5 emit <think> blocks (J1: degenerate 1390-token
            # thinking on hard prompts). VERIFIED (max_turns=4, reading `answer`): chat-completions
            # frontdoor answers a moderate prompt in 270 tokens / 1 turn, no <think>, vs /completion
            # 2643 tokens / 4 turns thinking-on — ~10x fewer tokens, single turn. (An earlier
            # "0 tokens" reading was a max_turns=1 probe artifact, not a backend bug.)
            # ingest_long_context is EXCLUDED — thinking-on is load-bearing for Qwen3-Next-80B.
            _ccl_raw = _os.environ.get(
                "ORCHESTRATOR_USE_CHAT_COMPLETIONS_ROLES",
                "worker_general,worker_explore,worker_math,worker_summarize,worker_coder,"
                "frontdoor,coder_escalation,architect_general",
            )
            _chat_completion_roles = {
                r.strip() for r in _ccl_raw.split(",") if r.strip()
            }
            if _chat_completion_roles:
                _log.info(
                    "Roles using /v1/chat/completions backend: %s",
                    sorted(_chat_completion_roles),
                )

            for role, url_str in server_urls.items():
                urls = [u.strip() for u in url_str.split(",") if u.strip()]

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
                    # Pre-warm role: full-speed + quarter instances
                    full_config = ServerConfig(
                        base_url=full_url, num_slots=num_slots,
                        use_chat_completions=_use_cc,
                    )
                    full_backend = CachingBackend(
                        LlamaServerBackend(full_config),
                        PrefixRouter(num_slots=num_slots),
                    )
                    full_port = int(full_url.rsplit(":", 1)[-1]) if ":" in full_url else 0

                    quarter_backends = []
                    for url in quarter_urls:
                        qcfg = ServerConfig(
                            base_url=url, num_slots=num_slots,
                            use_chat_completions=_use_cc,
                        )
                        quarter_backends.append(CachingBackend(
                            LlamaServerBackend(qcfg),
                            PrefixRouter(num_slots=num_slots),
                        ))

                    self._backends[role] = ConcurrencyAwareBackend(
                        full_backend, quarter_backends,
                        role=role, full_port=full_port,
                    )
                    _log.info(
                        "Concurrency-aware backend for %s: 1 full + %d quarters",
                        role, len(quarter_backends),
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
