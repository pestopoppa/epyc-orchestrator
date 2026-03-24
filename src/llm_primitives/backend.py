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
            from src.prefix_cache import CachingBackend, PrefixRouter

            for role, url_str in server_urls.items():
                urls = [u.strip() for u in url_str.split(",") if u.strip()]

                if len(urls) > 1:
                    # Multi-instance role: create one backend per URL, wrap in round-robin
                    backends = []
                    for url in urls:
                        config = ServerConfig(base_url=url, num_slots=num_slots)
                        backend = LlamaServerBackend(config)
                        router = PrefixRouter(num_slots=num_slots)
                        backends.append(CachingBackend(backend, router))
                    self._backends[role] = RoundRobinBackend(backends, role=role)
                    _log.info("Round-robin backend for %s: %d instances", role, len(urls))
                else:
                    # Single-instance role
                    config = ServerConfig(base_url=urls[0], num_slots=num_slots)
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
