"""CPU-only Qwen-Image-2.1 service with the legacy sdapi HTTP contract.

This process deliberately imports Diffusers/Torch only after startup begins.
It binds to loopback, loads only the already-downloaded local model bundle,
and serializes requests because the full-precision CPU pipeline is memory-heavy.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

LOG = logging.getLogger("qwen_image_server")
MODEL_PATH = os.environ.get(
    "QWEN_IMAGE_MODEL_PATH", "/mnt/raid0/llm/models/diffusion/qwen-image-2.1"
)
MAX_BODY_BYTES = 1_048_576
MAX_PIXELS = 4_500_000
MAX_STEPS = 80
MAX_BATCH_SIZE = 4
DEFAULT_REFERENCE_ROOTS = (
    "/mnt/raid0/llm/output/images",
    "/mnt/raid0/llm/epyc-root/tmp",
)
MAX_REFERENCE_IMAGE_BYTES = 32 * 1024 * 1024
MAX_REFERENCE_IMAGE_PIXELS = 50_000_000
MAX_REFERENCE_TOTAL_PIXELS = 60_000_000


def validate_request(body: Any) -> dict[str, Any]:
    if not isinstance(body, dict):
        raise ValueError("request body must be a JSON object")
    prompt = body.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")

    def integer(name: str, default: int, low: int, high: int) -> int:
        value = body.get(name, default)
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            raise ValueError(f"{name} must be an integer from {low} to {high}")
        return value

    width = integer("width", 1024, 256, 3072)
    height = integer("height", 1024, 256, 3072)
    if width % 16 or height % 16:
        raise ValueError("width and height must be multiples of 16")
    if width * height > MAX_PIXELS:
        raise ValueError(f"requested image exceeds the CPU pixel limit of {MAX_PIXELS:,}")

    steps = integer("steps", 40, 1, MAX_STEPS)
    batch_size = integer("batch_size", 1, 1, MAX_BATCH_SIZE)
    seed = body.get("seed", -1)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    cfg_scale = body.get("cfg_scale", 1.0)
    if isinstance(cfg_scale, bool) or not isinstance(cfg_scale, (int, float)):
        raise ValueError("cfg_scale must be numeric")
    if not 1.0 <= float(cfg_scale) <= 20.0:
        raise ValueError("cfg_scale must be between 1 and 20")
    reference_images = body.get("reference_images", [])
    if (
        not isinstance(reference_images, list)
        or len(reference_images) > 10
        or any(not isinstance(path, str) or not path.strip() for path in reference_images)
    ):
        raise ValueError("reference_images must be a list of at most 10 non-empty file paths")

    return {
        "prompt": prompt,
        "width": width,
        "height": height,
        "steps": steps,
        "batch_size": batch_size,
        "seed": seed,
        "cfg_scale": float(cfg_scale),
        "negative_prompt": body.get("negative_prompt", ""),
        "reference_images": reference_images,
    }


def load_reference_images(paths: list[str]) -> list[Any]:
    """Load user-selected images only from explicitly allowed local roots."""
    if not paths:
        return []

    from PIL import Image

    roots_value = os.environ.get("QWEN_IMAGE_REFERENCE_ROOTS")
    root_values = roots_value.split(os.pathsep) if roots_value else DEFAULT_REFERENCE_ROOTS
    roots = [Path(value).expanduser().resolve() for value in root_values if value]
    images = []
    total_pixels = 0
    for value in paths:
        try:
            path = Path(value).expanduser().resolve(strict=True)
        except OSError as exc:
            raise ValueError("reference image path does not exist or cannot be resolved") from exc
        if not path.is_file() or not any(path.is_relative_to(root) for root in roots):
            raise ValueError("reference image must be a regular file under an allowed image-input root")
        if path.stat().st_size > MAX_REFERENCE_IMAGE_BYTES:
            raise ValueError("reference image exceeds the 32 MiB input limit")
        try:
            with Image.open(path) as source:
                if source.format not in {"JPEG", "PNG", "WEBP"}:
                    raise ValueError("reference images must be JPEG, PNG, or WebP")
                if source.width * source.height > MAX_REFERENCE_IMAGE_PIXELS:
                    raise ValueError("reference image exceeds the 50-megapixel input limit")
                total_pixels += source.width * source.height
                if total_pixels > MAX_REFERENCE_TOTAL_PIXELS:
                    raise ValueError("reference images exceed the 60-megapixel combined input limit")
                source.load()
                images.append(source.convert("RGB"))
        except (OSError, SyntaxError) as exc:
            raise ValueError(f"could not read reference image: {path.name}") from exc
    return images


def generate_images(body: Any, pipeline: Any, torch_module: Any) -> dict[str, Any]:
    """Run one validated request and return the legacy sdapi response shape."""
    request = validate_request(body)
    seed = request["seed"] if request["seed"] >= 0 else secrets.randbelow(2**31)
    generator = torch_module.Generator(device="cpu").manual_seed(seed)
    kwargs: dict[str, Any] = {
        "prompt": request["prompt"],
        "width": request["width"],
        "height": request["height"],
        "num_inference_steps": request["steps"],
        "num_images_per_prompt": request["batch_size"],
        "generator": generator,
    }
    if request["cfg_scale"] > 1.0:
        kwargs["true_cfg_scale"] = request["cfg_scale"]
        kwargs["negative_prompt"] = request["negative_prompt"] or " "
    references = load_reference_images(request["reference_images"])
    if references:
        kwargs["image"] = references

    started = time.monotonic()
    result = pipeline(**kwargs)
    elapsed = time.monotonic() - started

    encoded_images: list[str] = []
    for image in result.images:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        encoded_images.append(base64.b64encode(buffer.getvalue()).decode("ascii"))
    info = {
        "model": "Qwen-Image-2.1",
        "seed": seed,
        "width": request["width"],
        "height": request["height"],
        "steps": request["steps"],
        "device": "cpu",
        "generation_seconds": round(elapsed, 3),
        "reference_count": len(references),
    }
    return {
        "images": encoded_images,
        "parameters": {**request, "seed": seed},
        "info": json.dumps(info, separators=(",", ":")),
    }


class _Handler(BaseHTTPRequestHandler):
    server_version = "QwenImage21/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        LOG.info("%s - %s", self.address_string(), fmt % args)

    def _json(self, status: int, payload: Any) -> None:
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path == "/sdapi/v1/samplers":
            self._json(200, [{"name": "FlowMatchEulerDiscreteScheduler"}])
        elif self.path == "/health":
            self._json(200, {"status": "ok", "model": "Qwen-Image-2.1", "device": "cpu"})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path != "/sdapi/v1/txt2img":
            self._json(404, {"error": "not found"})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= MAX_BODY_BYTES:
                raise ValueError(f"request body must be between 1 and {MAX_BODY_BYTES} bytes")
            body = json.loads(self.rfile.read(size))
            with self.server.inference_lock:  # type: ignore[attr-defined]
                response = generate_images(
                    body,
                    self.server.pipeline,  # type: ignore[attr-defined]
                    self.server.torch_module,  # type: ignore[attr-defined]
                )
            self._json(200, response)
        except (ValueError, json.JSONDecodeError) as exc:
            self._json(400, {"error": str(exc)})
        except Exception:
            LOG.exception("Qwen-Image-2.1 request failed")
            self._json(500, {"error": "Qwen image generation failed"})


class _QwenHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main() -> int:
    import torch
    from diffusers import QwenImage21Pipeline

    torch.set_num_threads(int(os.environ.get("QWEN_CPU_THREADS", "96")))
    torch.set_num_interop_threads(1)
    LOG.info("loading Qwen-Image-2.1 from local path %s on CPU", MODEL_PATH)
    pipeline = QwenImage21Pipeline.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).to("cpu")

    host = os.environ.get("QWEN_IMAGE_LISTEN", "127.0.0.1")
    port = int(os.environ.get("QWEN_IMAGE_PORT", "8190"))
    server = _QwenHTTPServer((host, port), _Handler)
    server.pipeline = pipeline  # type: ignore[attr-defined]
    server.torch_module = torch  # type: ignore[attr-defined]
    server.inference_lock = threading.Lock()  # type: ignore[attr-defined]
    LOG.info("Qwen-Image-2.1 ready on http://%s:%d (CPU-only)", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    raise SystemExit(main())
