"""High-level image-generation orchestration (sd-server backend).

Replaces the prior ComfyUI workflow-construction path 2026-05-07.
sd-server's sdapi/v1/txt2img is a single HTTP request returning the
image as base64, so this module collapses to a thin wrapper around
SDServerClient.

Public interface preserved: `ImageGenerator.generate(request) -> ImageGenerateResult`
so the Hermes plugin and any existing callers don't need to change.

The Ministral3 prompt-enhancer chain that ComfyUI's workflow JSON ran
in-graph is NOT yet wired through sd-server (sd.cpp does not currently
expose the enhancer LLM as a separate generation step). The `enhance`
parameter is accepted for interface compatibility but is a no-op here;
the user's prompt is sent through verbatim. This is acceptable for
already-rich prompts (>= ~50 words); for short prompts a future revision
should call the Ministral3 GGUF directly via llama-cli to expand them
before submitting to sd-server.

CLI invocation:
    python -m src.services.image_generator --prompt "..." \
        --width 1024 --height 1024 --steps 8
"""

from __future__ import annotations

import asyncio
import base64
import logging
import random
import time
from typing import Any

from src.models.image import (
    ImageGenerateRequest,
    ImageGenerateResult,
    encode_image_bytes,
    output_dir_for_today,
)
from src.services.sd_server_client import (
    SDServerClient,
    SDServerError,
)


LOG = logging.getLogger(__name__)


class ImageGenerator:
    """Build an sd-server txt2img request, submit it, save the result."""

    def __init__(self, client: SDServerClient | None = None):
        self.client = client or SDServerClient()

    async def generate(self, request: ImageGenerateRequest) -> ImageGenerateResult:
        seed = request.seed if request.seed is not None else random.randint(0, 2**31 - 1)
        # NOTE: sd-server's prompt-enhancer integration is not yet exposed;
        # `enhance` is recorded but does not modify the request.
        use_enhancer = request.resolve_enhance()
        if use_enhancer:
            LOG.info(
                "ImageGenerator: enhance=auto resolved True but sd-server backend "
                "does not run the Ministral3 enhancer in-graph; passing prompt verbatim"
            )

        started = time.monotonic()

        try:
            resp = await self.client.txt2img(
                prompt=request.prompt,
                width=request.width,
                height=request.height,
                steps=request.steps,
                cfg_scale=request.cfg,
                seed=seed,
                batch_size=request.batch_size,
                sampler_name=request.sampler if request.sampler else None,
                scheduler=request.scheduler if request.scheduler else None,
            )
        except SDServerError as exc:
            return ImageGenerateResult(
                prompt_id=None,
                image_path=None,
                image_bytes_b64=None,
                width=request.width,
                height=request.height,
                seed_used=seed,
                steps=request.steps,
                elapsed_sec=time.monotonic() - started,
                enhancer_used=False,
                error=str(exc),
            )

        images_b64 = resp.get("images") or []
        if not images_b64:
            return ImageGenerateResult(
                prompt_id=None,
                image_path=None,
                image_bytes_b64=None,
                width=request.width,
                height=request.height,
                seed_used=seed,
                steps=request.steps,
                elapsed_sec=time.monotonic() - started,
                enhancer_used=False,
                error="No images in sd-server response",
            )

        first_b64 = images_b64[0]
        # Some sd-webui-compat servers prefix with "data:image/png;base64,";
        # strip if present.
        if first_b64.startswith("data:"):
            first_b64 = first_b64.split(",", 1)[1]
        image_bytes = base64.b64decode(first_b64)
        elapsed = time.monotonic() - started

        out_dir = output_dir_for_today()
        # sd-server has no native prompt_id; mint one from seed+timestamp.
        prompt_id = f"sd-{int(time.time())}-{seed}"
        out_path = out_dir / f"{prompt_id}.png"
        out_path.write_bytes(image_bytes)

        return ImageGenerateResult(
            prompt_id=prompt_id,
            image_path=str(out_path),
            image_bytes_b64=encode_image_bytes(image_bytes),
            width=request.width,
            height=request.height,
            seed_used=seed,
            steps=request.steps,
            elapsed_sec=elapsed,
            enhanced_prompt=None,
            enhancer_used=False,  # sd-server backend does not run the enhancer chain
            metadata={
                "backend": "sd_server",
                "sd_server_info": resp.get("info", "")[:500] if isinstance(resp.get("info"), str) else None,
                "image_count": len(images_b64),
            },
        )


def _cli_main() -> int:
    """End-to-end CLI for testing."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Generate one image via sd-server (ERNIE-Image-Turbo).")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--enhance", choices=["auto", "true", "false"], default="auto")
    args = parser.parse_args()

    enhance: Any = args.enhance
    if args.enhance == "true":
        enhance = True
    elif args.enhance == "false":
        enhance = False

    req = ImageGenerateRequest(
        prompt=args.prompt,
        width=args.width,
        height=args.height,
        seed=args.seed,
        steps=args.steps,
        cfg=args.cfg,
        enhance=enhance,
    )

    async def run() -> ImageGenerateResult:
        gen = ImageGenerator()
        try:
            return await gen.generate(req)
        finally:
            await gen.client.close()

    result = asyncio.run(run())
    payload = result.to_dict()
    payload.pop("image_bytes_b64", None)
    print(json.dumps(payload, indent=2, default=str))
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(_cli_main())
