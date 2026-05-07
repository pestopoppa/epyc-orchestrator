"""Async HTTP client for the ComfyUI diffusion-inference server.

ComfyUI exposes a queue-based REST API:
  POST /prompt           submit a workflow (returns prompt_id)
  GET  /history/{id}     poll for completion + outputs
  GET  /view             fetch a generated image by filename
  GET  /system_stats     readiness probe (no /health; non-OpenAI-compatible)

Mirrors the shape of src/services/document_client.py for consistency.
"""

from __future__ import annotations

import asyncio
import logging
import urllib.parse
from typing import Any

import httpx


LOG = logging.getLogger(__name__)

DEFAULT_COMFYUI_URL = "http://127.0.0.1:8188"
DEFAULT_HEALTH_TIMEOUT = 5.0
DEFAULT_SUBMIT_TIMEOUT = 30.0
DEFAULT_POLL_INTERVAL_SEC = 5.0
# CPU-realistic ceiling: 1024² @ 8 steps lands ~8 min; 60 min covers larger
# resolutions and prompt-enhancer overhead.
DEFAULT_INFERENCE_TIMEOUT = 60 * 60.0


class ComfyUIError(Exception):
    """Error from the ComfyUI server."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class ComfyUIUnavailable(ComfyUIError):
    """ComfyUI server is not reachable."""


class ComfyUIQueueRejected(ComfyUIError):
    """Server returned a 4xx on /prompt — workflow JSON was invalid."""


class ComfyUITimeout(ComfyUIError):
    """Inference did not complete within the timeout."""


class ComfyUIClient:
    """Async HTTP client for ComfyUI.

    Usage:
        client = ComfyUIClient()
        if await client.health_check():
            prompt_id = await client.submit_workflow(workflow)
            outputs = await client.wait_for_completion(prompt_id)
            for image_meta in extract_images(outputs):
                bytes_ = await client.fetch_image(**image_meta)
    """

    def __init__(self, base_url: str = DEFAULT_COMFYUI_URL):
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

    async def __aenter__(self) -> ComfyUIClient:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    async def health_check(self) -> bool:
        """Probe /system_stats — ComfyUI's readiness endpoint."""
        try:
            client = await self._get_client()
            resp = await client.get("/system_stats", timeout=DEFAULT_HEALTH_TIMEOUT)
            return resp.status_code == 200
        except (httpx.RequestError, httpx.TimeoutException):
            return False

    async def submit_workflow(
        self,
        workflow: dict[str, Any],
        client_id: str | None = None,
    ) -> str:
        """POST a workflow to /prompt; return the assigned prompt_id."""
        client = await self._get_client()
        body: dict[str, Any] = {"prompt": workflow}
        if client_id:
            body["client_id"] = client_id
        try:
            resp = await client.post("/prompt", json=body, timeout=DEFAULT_SUBMIT_TIMEOUT)
        except httpx.RequestError as exc:
            raise ComfyUIUnavailable(f"Could not reach ComfyUI at {self.base_url}: {exc}") from exc
        if resp.status_code >= 400:
            raise ComfyUIQueueRejected(
                f"ComfyUI rejected workflow (HTTP {resp.status_code}): {resp.text[:500]}",
                status_code=resp.status_code,
            )
        data = resp.json()
        prompt_id = data.get("prompt_id")
        if not prompt_id:
            raise ComfyUIError(f"No prompt_id in response: {data!r}")
        return prompt_id

    async def get_history(self, prompt_id: str) -> dict[str, Any] | None:
        """Return the history entry for a prompt, or None if not yet present."""
        client = await self._get_client()
        resp = await client.get(f"/history/{prompt_id}", timeout=DEFAULT_HEALTH_TIMEOUT)
        if resp.status_code != 200:
            raise ComfyUIError(
                f"GET /history/{prompt_id} returned HTTP {resp.status_code}",
                status_code=resp.status_code,
            )
        history = resp.json()
        return history.get(prompt_id)

    async def wait_for_completion(
        self,
        prompt_id: str,
        timeout: float = DEFAULT_INFERENCE_TIMEOUT,
        poll_interval: float = DEFAULT_POLL_INTERVAL_SEC,
    ) -> dict[str, Any]:
        """Block until the prompt completes; return outputs dict.

        Raises:
            ComfyUITimeout: if completion is not seen within `timeout`.
            ComfyUIError: if the server reports an execution error.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            entry = await self.get_history(prompt_id)
            if entry is not None:
                status = entry.get("status", {})
                if status.get("completed"):
                    return entry.get("outputs", {})
                if status.get("status_str") == "error":
                    msgs = status.get("messages", [])
                    raise ComfyUIError(f"Sampler error: {msgs!r}")
            if loop.time() >= deadline:
                raise ComfyUITimeout(
                    f"Prompt {prompt_id} did not complete within {timeout:.0f}s"
                )
            await asyncio.sleep(poll_interval)

    async def fetch_image(
        self,
        filename: str,
        subfolder: str = "",
        type: str = "output",
    ) -> bytes:
        """GET /view to retrieve a generated image as bytes."""
        client = await self._get_client()
        params = urllib.parse.urlencode(
            {"filename": filename, "subfolder": subfolder, "type": type}
        )
        resp = await client.get(f"/view?{params}")
        if resp.status_code != 200:
            raise ComfyUIError(
                f"GET /view returned HTTP {resp.status_code}",
                status_code=resp.status_code,
            )
        return resp.content


def extract_images(outputs: dict[str, Any]) -> list[dict[str, str]]:
    """From a /history outputs dict, list image-metadata records.

    Each record has keys: filename, subfolder, type — directly suitable for
    passing to ComfyUIClient.fetch_image().
    """
    images: list[dict[str, str]] = []
    for _node_id, node_out in outputs.items():
        for img in node_out.get("images", []) or []:
            images.append({
                "filename": img["filename"],
                "subfolder": img.get("subfolder", ""),
                "type": img.get("type", "output"),
            })
    return images


_default_client: ComfyUIClient | None = None


def get_comfyui_client() -> ComfyUIClient:
    """Module-level singleton accessor (mirrors document_client.get_document_client)."""
    global _default_client
    if _default_client is None:
        _default_client = ComfyUIClient()
    return _default_client
