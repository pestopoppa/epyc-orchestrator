"""Vision-Language description analyzer using live VL servers or llama-mtmd-cli."""

from __future__ import annotations

import base64
import json
import logging
import mimetypes
import os
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from PIL import Image

from src.vision.analyzers.base import Analyzer, AnalyzerResult
from src.vision.config import (
    LLAMA_MTMD_CLI,
    VL_MODEL_PATH,
    VL_MMPROJ_PATH,
    VL_SERVER_PORT,
    DEFAULT_VL_MAX_TOKENS,
    DEFAULT_VL_THREADS,
    VL_INFERENCE_TIMEOUT,
    TEMP_JPEG_QUALITY,
    VISION_CACHE_DIR,
)

logger = logging.getLogger(__name__)

_VL_BACKEND_ENV = "ORCHESTRATOR_VISION_VL_BACKEND"
_VALID_VL_BACKENDS = {"auto", "server", "cli"}


class VLDescribeAnalyzer(Analyzer):
    """Generate natural language descriptions of images using VL model.

    Prefers the resident multimodal llama-server, falling back to
    llama-mtmd-cli when configured for ``auto`` and no server is available.
    """

    def __init__(
        self,
        prompt: str = "Describe this image briefly. Note people, setting, and activities.",
        max_tokens: int = DEFAULT_VL_MAX_TOKENS,
        model_path: Path | str | None = None,
        mmproj_path: Path | str | None = None,
        threads: int = DEFAULT_VL_THREADS,
        backend: str | None = None,
        server_port: int = VL_SERVER_PORT,
        **config: Any,
    ):
        """Initialize VL description analyzer.

        Args:
            prompt: Prompt for the VL model.
            max_tokens: Maximum tokens to generate.
            model_path: Path to GGUF model (default: Qwen2.5-VL-7B).
            mmproj_path: Path to multimodal projector.
            threads: Number of threads for inference.
            backend: "auto", "server", or "cli" (default: env/auto).
            server_port: llama-server port for OpenAI-compatible VL inference.
            **config: Additional configuration.
        """
        super().__init__(**config)
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.model_path = Path(model_path) if model_path else VL_MODEL_PATH
        self.mmproj_path = Path(mmproj_path) if mmproj_path else VL_MMPROJ_PATH
        self.threads = threads
        self.backend = (backend or os.environ.get(_VL_BACKEND_ENV, "auto")).lower()
        if self.backend not in _VALID_VL_BACKENDS:
            raise ValueError(
                f"Invalid VL backend {self.backend!r}; expected one of "
                f"{', '.join(sorted(_VALID_VL_BACKENDS))}"
            )
        self.server_port = server_port

    @property
    def name(self) -> str:
        return "vl_describe"

    def initialize(self) -> None:
        """Verify the configured backend is usable."""
        if self.backend != "cli":
            if self._server_available():
                super().initialize()
                return
            if self.backend == "server":
                raise ConnectionError(f"VL server is not available on port {self.server_port}")

        cli_path = self._resolve_mtmd_cli()
        if not cli_path:
            raise FileNotFoundError(f"llama-mtmd-cli not found at {LLAMA_MTMD_CLI}")
        if not self.model_path.exists():
            raise FileNotFoundError(f"VL model not found at {self.model_path}")
        if not self.mmproj_path.exists():
            raise FileNotFoundError(f"mmproj not found at {self.mmproj_path}")

        super().initialize()

    def analyze(self, image: Image.Image, path: Path | None = None) -> AnalyzerResult:
        """Generate description for image.

        Args:
            image: PIL Image to describe.
            path: Optional path to original file (used directly if available).

        Returns:
            AnalyzerResult with description text.
        """
        self.ensure_initialized()
        start = time.perf_counter()
        temp_file: tempfile._TemporaryFileWrapper | None = None

        try:
            # Use original path if available, otherwise save temp file
            if path and path.exists():
                image_path = str(path)
            else:
                VISION_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                temp_file = tempfile.NamedTemporaryFile(
                    suffix=".jpg",
                    dir=VISION_CACHE_DIR,
                    delete=False,
                )
                image.save(temp_file.name, format="JPEG", quality=TEMP_JPEG_QUALITY)
                image_path = temp_file.name

            if self.backend != "cli":
                server_result = self._analyze_with_server(Path(image_path), start)
                if server_result.success or self.backend == "server":
                    return server_result
                logger.warning("VL server path failed; falling back to CLI: %s", server_result.error)

            return self._analyze_with_cli(Path(image_path), start)

        except subprocess.TimeoutExpired:
            return AnalyzerResult(
                analyzer_name=self.name,
                success=False,
                error="VL inference timed out",
                processing_time_ms=(time.perf_counter() - start) * 1000,
            )
        except Exception as e:
            logger.error(f"VL description failed: {e}")
            return AnalyzerResult(
                analyzer_name=self.name,
                success=False,
                error=str(e),
                processing_time_ms=(time.perf_counter() - start) * 1000,
            )
        finally:
            if temp_file:
                Path(temp_file.name).unlink(missing_ok=True)

    def _analyze_with_server(self, image_path: Path, start: float) -> AnalyzerResult:
        """Run inference through the resident multimodal llama-server."""
        try:
            image_b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
            mime_type = mimetypes.guess_type(str(image_path))[0] or "image/jpeg"
            payload = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime_type};base64,{image_b64}"},
                            },
                            {"type": "text", "text": self.prompt},
                        ],
                    }
                ],
                "max_tokens": self.max_tokens,
                "temperature": 0.0,
                "stream": False,
            }
            request = urllib.request.Request(
                f"http://127.0.0.1:{self.server_port}/v1/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=VL_INFERENCE_TIMEOUT) as response:
                body = response.read().decode("utf-8")
                if response.status != 200:
                    return self._error_result(f"VL server HTTP {response.status}: {body}", start)

            data = json.loads(body)
            description = self._extract_server_content(data)
            if not description:
                return self._error_result("VL server returned no content", start)

            return self._success_result({"description": self._clean_output(description)}, start)
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            return self._error_result(f"VL server inference failed: {exc}", start)

    def _analyze_with_cli(self, image_path: Path, start: float) -> AnalyzerResult:
        """Run inference through llama-mtmd-cli."""
        cli_path = self._resolve_mtmd_cli()
        if not cli_path:
            return self._error_result(f"llama-mtmd-cli not found at {LLAMA_MTMD_CLI}", start)

        cmd = [
            str(cli_path),
            "-m",
            str(self.model_path),
            "--mmproj",
            str(self.mmproj_path),
            "--image",
            str(image_path),
            "-p",
            self.prompt,
            "-n",
            str(self.max_tokens),
            "-t",
            str(self.threads),
            "--temp",
            "0.0",
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=VL_INFERENCE_TIMEOUT,
        )

        if result.returncode != 0:
            error = result.stderr or "Unknown error"
            logger.error(f"VL inference failed: {error}")
            return self._error_result(error, start)

        return self._success_result({"description": self._clean_output(result.stdout.strip())}, start)

    def _server_available(self) -> bool:
        """Return True when the configured VL server health endpoint responds."""
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{self.server_port}/health",
                timeout=2.0,
            ) as response:
                return 200 <= response.status < 500
        except OSError:
            return False

    def _resolve_mtmd_cli(self) -> Path | None:
        """Resolve configured or known local llama-mtmd-cli build paths."""
        candidates = [
            LLAMA_MTMD_CLI,
            LLAMA_MTMD_CLI.parents[2] / "build-v2" / "bin" / "llama-mtmd-cli",
            LLAMA_MTMD_CLI.parents[2] / "build_libomp_pgo_bolt" / "bin" / "llama-mtmd-cli",
            LLAMA_MTMD_CLI.parents[2] / "build_libomp_pgo_use" / "bin" / "llama-mtmd-cli",
            LLAMA_MTMD_CLI.parents[2] / "build-blis52" / "bin" / "llama-mtmd-cli",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    @staticmethod
    def _extract_server_content(data: dict[str, Any]) -> str:
        choices = data.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            content = message.get("content")
            if isinstance(content, str):
                return content.strip()
            text = choices[0].get("text")
            if isinstance(text, str):
                return text.strip()
        content = data.get("content")
        return content.strip() if isinstance(content, str) else ""

    def _clean_output(self, text: str) -> str:
        """Clean VL model output artifacts.

        Args:
            text: Raw model output.

        Returns:
            Cleaned description text.
        """
        # Remove thinking tags if present
        import re

        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

        # Remove assistant/user prefixes
        text = re.sub(r"^(Assistant:|assistant:|User:|user:)\s*", "", text)

        # Remove leading/trailing whitespace
        text = text.strip()

        return text


class VLOCRAnalyzer(VLDescribeAnalyzer):
    """Extract text from images using VL model.

    Specialized prompt for OCR-like text extraction.
    """

    def __init__(self, **config: Any):
        prompt = (
            "Read and transcribe all visible text in this image exactly as it appears. "
            "Preserve formatting including line breaks and indentation where possible. "
            "If the image contains mathematical formulas, transcribe them in LaTeX format."
        )
        super().__init__(prompt=prompt, max_tokens=1024, **config)

    @property
    def name(self) -> str:
        return "vl_ocr"


class VLStructuredAnalyzer(VLDescribeAnalyzer):
    """Extract structured data from images (forms, receipts, tables).

    Returns JSON-formatted extraction results.
    """

    def __init__(self, schema_hint: str = "", **config: Any):
        """Initialize structured extraction analyzer.

        Args:
            schema_hint: Optional JSON schema or field hints for extraction.
            **config: Additional configuration.
        """
        self.schema_hint = schema_hint
        prompt = (
            "Extract structured information from this image and return it as valid JSON. "
            "Include fields like: date, total, items, names, addresses, etc. as applicable. "
            f"{schema_hint}"
        )
        super().__init__(prompt=prompt, max_tokens=1024, **config)

    @property
    def name(self) -> str:
        return "vl_structured"

    def analyze(self, image: Image.Image, path: Path | None = None) -> AnalyzerResult:
        """Extract structured data from image."""
        result = super().analyze(image, path)

        if result.success and result.data.get("description"):
            # Try to parse as JSON
            import json

            try:
                # Find JSON in the response
                text = result.data["description"]
                # Look for JSON block
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0]
                elif "```" in text:
                    text = text.split("```")[1].split("```")[0]

                structured = json.loads(text.strip())
                result.data["structured"] = structured
            except json.JSONDecodeError:
                # Return raw text if JSON parsing fails
                result.data["structured"] = None
                result.data["parse_error"] = "Failed to parse JSON from response"

        return result
