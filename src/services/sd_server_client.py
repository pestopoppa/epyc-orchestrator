"""Async HTTP client for sd-server (stable-diffusion.cpp native).

Replaces comfyui_client.py for the ERNIE-Image-Turbo path. sd-server's
sdapi/v1/txt2img is synchronous (blocks the HTTP connection until the
image is ready), so the client interface is simpler than ComfyUI's
queue/poll pattern. Health probe uses /sdapi/v1/samplers (sd-server has
no dedicated /health endpoint).
"""

from __future__ import annotations

import logging
from typing import Any

import httpx


LOG = logging.getLogger(__name__)

DEFAULT_SD_SERVER_URL = "http://127.0.0.1:8190"
DEFAULT_HEALTH_TIMEOUT = 5.0
# CPU-realistic ceiling: 1024² @ 8 steps lands ~3-4 min on sd.cpp; 60 min
# covers larger resolutions / contention with other stack workloads.
DEFAULT_INFERENCE_TIMEOUT = 60 * 60.0


class SDServerError(Exception):
    """Error from sd-server."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class SDServerUnavailable(SDServerError):
    """sd-server is not reachable."""


class SDServerClient:
    """Async HTTP client for sd-server's sdapi/v1/txt2img endpoint.

    Usage:
        client = SDServerClient()
        if await client.health_check():
            result = await client.txt2img(
                prompt="a cat",
                width=512, height=512, steps=4, cfg_scale=1.0, seed=42,
            )
            # result["images"] is list of base64 strings
    """

    def __init__(self, base_url: str = DEFAULT_SD_SERVER_URL):
        self.base_url = base_url.rstrip("/")
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(DEFAULT_INFERENCE_TIMEOUT, connect=10.0),
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> SDServerClient:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    async def health_check(self) -> bool:
        """Probe /sdapi/v1/samplers — cheap readiness signal."""
        try:
            client = await self._get_client()
            resp = await client.get("/sdapi/v1/samplers", timeout=DEFAULT_HEALTH_TIMEOUT)
            return resp.status_code == 200
        except (httpx.RequestError, httpx.TimeoutException):
            return False

    async def txt2img(
        self,
        prompt: str,
        *,
        width: int = 1024,
        height: int = 1024,
        steps: int = 8,
        cfg_scale: float = 1.0,
        seed: int = -1,
        batch_size: int = 1,
        negative_prompt: str = "",
        sampler_name: str | None = None,
        scheduler: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """POST /sdapi/v1/txt2img and return the parsed JSON response.

        Response shape (sd-webui-compatible):
            {
              "images": [<base64-png>, ...],
              "parameters": {...},
              "info": "..."
            }

        Raises SDServerUnavailable on connect error, SDServerError on 4xx/5xx.
        """
        body: dict[str, Any] = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "seed": seed,
            "batch_size": batch_size,
        }
        if sampler_name is not None:
            body["sampler_name"] = sampler_name
        if scheduler is not None:
            body["scheduler"] = scheduler
        if extra:
            body.update(extra)

        client = await self._get_client()
        try:
            resp = await client.post("/sdapi/v1/txt2img", json=body)
        except httpx.RequestError as exc:
            raise SDServerUnavailable(
                f"Could not reach sd-server at {self.base_url}: {exc}"
            ) from exc

        if resp.status_code >= 400:
            raise SDServerError(
                f"sd-server txt2img returned HTTP {resp.status_code}: {resp.text[:500]}",
                status_code=resp.status_code,
            )
        return resp.json()


_default_client: SDServerClient | None = None


def get_sd_server_client() -> SDServerClient:
    """Module-level singleton accessor."""
    global _default_client
    if _default_client is None:
        _default_client = SDServerClient()
    return _default_client
