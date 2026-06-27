import base64
from pathlib import Path

import pytest

from src.models.image import ImageGenerateRequest
from src.services.image_generator import ImageGenerator


def test_auto_enhance_prefers_text_surfaces() -> None:
    request = ImageGenerateRequest(prompt="Create an infographic about kernel v6 speedups.")

    assert request.enhance_auto_reason() == "text_surface"
    assert request.resolve_enhance() is True


def test_auto_enhance_suppresses_compositional_prompts() -> None:
    request = ImageGenerateRequest(prompt="A red cube to the left of a blue sphere on a table.")

    assert request.enhance_auto_reason() == "compositional_scene"
    assert request.resolve_enhance() is False


def test_auto_enhance_keeps_short_prompt_fallback() -> None:
    request = ImageGenerateRequest(prompt="A watercolor city street at sunrise.")

    assert request.enhance_auto_reason() == "short_prompt"
    assert request.resolve_enhance() is True


def test_auto_enhance_disables_already_rich_prompt() -> None:
    prompt = " ".join(f"detail{i}" for i in range(60))
    request = ImageGenerateRequest(prompt=prompt)

    assert request.enhance_auto_reason() == "rich_prompt"
    assert request.resolve_enhance() is False


class FakeSDClient:
    async def txt2img(self, **kwargs):
        self.kwargs = kwargs
        return {
            "images": [base64.b64encode(b"fake image bytes").decode("ascii")],
            "info": "ok",
        }


@pytest.mark.asyncio
async def test_generator_records_unwired_enhancer_policy(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("src.services.image_generator.output_dir_for_today", lambda: tmp_path)
    client = FakeSDClient()
    generator = ImageGenerator(client=client)

    result = await generator.generate(
        ImageGenerateRequest(
            prompt="A red cube to the left of a blue sphere.",
            seed=123,
        )
    )

    assert result.success
    assert result.enhancer_used is False
    assert result.metadata["enhance_policy"] == "auto"
    assert result.metadata["enhance_resolved"] is False
    assert result.metadata["enhance_auto_reason"] == "compositional_scene"
    assert result.metadata["enhancer_available"] is False
    assert client.kwargs["prompt"] == "A red cube to the left of a blue sphere."
