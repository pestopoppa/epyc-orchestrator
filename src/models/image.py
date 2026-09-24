"""Data models for image generation via the local Qwen-Image-2.1 service.

Mirrors the shape of src/models/document.py — frozen dataclasses with
explicit error fields and serialization helpers.
"""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal


# Recommended aspect-ratio sizes from the Qwen-Image-2.1 model card.
RECOMMENDED_SIZES = [
    (2048, 2048),
    (2400, 1792), (1792, 2400),
    (2528, 1696), (1696, 2528),
    (2752, 1536), (1536, 2752),
]

EnhancePolicy = Literal["auto", True, False]

TEXT_SURFACE_TERMS = (
    "poster",
    "infographic",
    "flyer",
    "brochure",
    "sign",
    "signage",
    "banner",
    "label",
    "logo",
    "typography",
    "headline",
    "title text",
    "caption",
    "menu",
    "certificate",
    "presentation slide",
    "slide",
    "sticker",
    "packaging",
)

COMPOSITIONAL_SUPPRESSORS = (
    "left of",
    "right of",
    "beside",
    "next to",
    "between",
    "behind",
    "in front of",
    "above",
    "below",
    "under",
    "on top of",
    "holding",
    "wearing",
    "sitting on",
    "standing on",
    "spatial relationship",
    "object count",
    "geneval",
    "compositional",
)


def _contains_term(prompt: str, term: str) -> bool:
    return re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", prompt) is not None


@dataclass(frozen=True)
class ImageGenerateRequest:
    """A single image-generation request.

    `enhance` is retained for API compatibility with the former ERNIE stack.
    The policy is recorded in metadata but Qwen currently receives the prompt
    unchanged; no prompt enhancer is executed.
    """

    prompt: str
    width: int = 1024
    height: int = 1024
    seed: int | None = None
    steps: int = 40
    cfg: float = 1.0
    sampler: str = "euler"
    scheduler: str = "simple"
    enhance: EnhancePolicy = "auto"
    batch_size: int = 1
    reference_images: tuple[str, ...] = ()

    def enhance_auto_reason(self) -> str:
        """Return the deterministic reason used by the auto enhancer policy."""
        if self.enhance is True:
            return "forced_true"
        if self.enhance is False:
            return "forced_false"

        prompt = " ".join(self.prompt.casefold().split())
        if any(_contains_term(prompt, term) for term in TEXT_SURFACE_TERMS):
            return "text_surface"
        if any(_contains_term(prompt, term) for term in COMPOSITIONAL_SUPPRESSORS):
            return "compositional_scene"
        if len(prompt.split()) < 50:
            return "short_prompt"
        return "rich_prompt"

    def resolve_enhance(self) -> bool:
        """Resolve the enhance policy to a concrete bool."""
        if self.enhance == "auto":
            return self.enhance_auto_reason() in {"text_surface", "short_prompt"}
        return bool(self.enhance)


@dataclass(frozen=True)
class ImageGenerateResult:
    """Result of an image-generation request.

    Either `image_path` + `image_bytes_b64` are populated (success) OR
    `error` is set (failure). `enhanced_prompt` is only present if the
    prompt enhancer ran; `enhancer_used` is the source of truth for
    whether it actually fired.
    """

    prompt_id: str | None
    image_path: str | None
    image_bytes_b64: str | None
    width: int
    height: int
    seed_used: int | None
    steps: int
    elapsed_sec: float
    enhanced_prompt: str | None = None
    enhancer_used: bool = False
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.error is None and self.image_path is not None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dict (suitable for tool responses)."""
        return {
            "prompt_id": self.prompt_id,
            "image_path": self.image_path,
            "image_bytes_b64": self.image_bytes_b64,
            "width": self.width,
            "height": self.height,
            "seed_used": self.seed_used,
            "steps": self.steps,
            "elapsed_sec": self.elapsed_sec,
            "enhanced_prompt": self.enhanced_prompt,
            "enhancer_used": self.enhancer_used,
            "error": self.error,
            "metadata": self.metadata,
        }


def encode_image_bytes(data: bytes) -> str:
    """Base64-encode image bytes for inline transport in tool responses."""
    return base64.b64encode(data).decode("ascii")


def output_dir_for_today(root: Path | str = "/mnt/raid0/llm/output/images") -> Path:
    """Return /mnt/raid0/llm/output/images/YYYY-MM-DD/, creating if needed.

    Daily-bucketed durable storage for generated images. Mirrors the
    OCR pipeline's tmp-dir convention but uses a date-bucketed permanent
    location instead of a flat tmp dir.
    """
    root_path = Path(root)
    today_path = root_path / datetime.now().strftime("%Y-%m-%d")
    today_path.mkdir(parents=True, exist_ok=True)
    return today_path
