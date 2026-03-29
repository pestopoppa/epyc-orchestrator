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

                if has_full and quarter_urls:
                    # Pre-warm role: full-speed + quarter instances
                    full_config = ServerConfig(base_url=full_url, num_slots=num_slots)
                    full_backend = CachingBackend(
                        LlamaServerBackend(full_config),
                        PrefixRouter(num_slots=num_slots),
                    )
                    full_port = int(full_url.rsplit(":", 1)[-1]) if ":" in full_url else 0

                    quarter_backends = []
                    for url in quarter_urls:
                        qcfg = ServerConfig(base_url=url, num_slots=num_slots)
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
                        config = ServerConfig(base_url=url, num_slots=num_slots)
                        backend = LlamaServerBackend(config)
                        router = PrefixRouter(num_slots=num_slots)
                        backends.append(CachingBackend(backend, router))
                    self._backends[role] = RoundRobinBackend(backends, role=role)
                    _log.info("Round-robin backend for %s: %d instances", role, len(quarter_urls))
                else:
                    # Single-instance role
                    url = quarter_urls[0] if quarter_urls else url_str
                    config = ServerConfig(base_url=url, num_slots=num_slots)
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
