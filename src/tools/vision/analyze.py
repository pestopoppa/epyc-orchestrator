"""Vision tool handlers for tool registry integration.

Thin wrappers around VisionPipeline that expose vision capabilities as
invocable tools. Model-agnostic: VL operations check for mmproj availability
at runtime rather than hardcoding role permissions.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default analyzers when none specified
_DEFAULT_ANALYZERS = ["exif", "vl_describe"]

# Map short names to AnalyzerType enum values
_ANALYZER_NAME_MAP = {
    "exif": "exif_extract",
    "face_detect": "face_detect",
    "face_embed": "face_embed",
    "vl_describe": "vl_describe",
    "vl_ocr": "vl_ocr",
    "vl_structured": "vl_structured",
    "clip_embed": "clip_embed",
}

# Analyzers that require a multimodal model (mmproj)
_VL_ANALYZERS = {"vl_describe", "vl_ocr", "vl_structured"}


def _load_registry_roles() -> dict:
    """Load roles from model registry. Isolated for testability."""
    from src.registry.registry_loader import RegistryLoader
    loader = RegistryLoader()
    loader.load()
    return loader.roles


def _check_multimodal_available() -> bool:
    """Check if a multimodal model is available in the active stack.

    Checks the model registry for any active role with mmproj_path set,
    making this model-agnostic — works with any VL model, not just
    hardcoded vision roles.
    """
    try:
        roles = _load_registry_roles()
        for role in roles.values():
            if role.model and role.model.mmproj_path:
                return True
        return False
    except Exception:
        # Registry not loaded (e.g., in tests or standalone mode)
        return False


def _get_pipeline():
    """Get or create the VisionPipeline singleton."""
    from src.vision.pipeline import VisionPipeline

    # Use module-level singleton via dependency injection pattern
    if not hasattr(_get_pipeline, "_instance"):
        _get_pipeline._instance = VisionPipeline()
    return _get_pipeline._instance


def _parse_analyzers(analyzers_str: str | None) -> list[str]:
    """Parse comma-separated analyzer string into list of AnalyzerType values."""
    if not analyzers_str:
        return _DEFAULT_ANALYZERS

    names = [n.strip().lower() for n in analyzers_str.split(",") if n.strip()]
    result = []
    for name in names:
        mapped = _ANALYZER_NAME_MAP.get(name, name)
        result.append(mapped)
    return result


def vision_analyze(
    image_path: str,
    analyzers: str | None = None,
    vl_prompt: str | None = None,
) -> str:
    """Analyze an image using the vision pipeline.

    Args:
        image_path: Path to image file.
        analyzers: Comma-separated analyzer names. Default: exif,vl_describe.
        vl_prompt: Custom prompt for VL description.

    Returns:
        JSON string with analysis results.
    """
    path = Path(image_path)
    if not path.exists():
        return json.dumps({"error": f"File not found: {image_path}"})

    analyzer_list = _parse_analyzers(analyzers)

    # Check if VL analyzers requested but no multimodal model available
    requested_vl = set(analyzer_list) & _VL_ANALYZERS
    if requested_vl and not _check_multimodal_available():
        # Fall back to non-VL analyzers only
        analyzer_list = [a for a in analyzer_list if a not in _VL_ANALYZERS]
        if not analyzer_list:
            return json.dumps({
                "error": "VL analyzers requested but no multimodal model (with mmproj) is active. "
                         "Available non-VL analyzers: exif, face_detect, face_embed, clip_embed",
            })
        logger.warning(
            "VL analyzers %s skipped — no multimodal model active. "
            "Running non-VL analyzers only: %s",
            requested_vl, analyzer_list,
        )

    try:
        from src.vision.models import AnalyzerType

        enum_analyzers = []
        for name in analyzer_list:
            try:
                enum_analyzers.append(AnalyzerType(name))
            except ValueError:
                return json.dumps({"error": f"Unknown analyzer: {name}"})

        pipeline = _get_pipeline()
        if not pipeline.is_initialized:
            pipeline.initialize(enum_analyzers)

        result = pipeline.analyze(
            image=path,
            analyzers=enum_analyzers,
            vl_prompt=vl_prompt,
        )

        return json.dumps(result.to_dict(), indent=2, default=str)

    except Exception as e:
        logger.exception("Vision analysis failed for %s", image_path)
        return json.dumps({"error": f"Analysis failed: {type(e).__name__}: {e}"})


def vision_search(query: str, limit: int = 5) -> str:
    """Search analyzed images by text query.

    Args:
        query: Natural language search query.
        limit: Maximum results (default 5).

    Returns:
        JSON string with search results.
    """
    try:
        from src.vision.search import VisionSearch

        search = VisionSearch()
        results = search.search_by_text(query, limit=limit)

        return json.dumps(
            {"query": query, "results": [r.to_dict() for r in results]},
            indent=2,
            default=str,
        )

    except Exception as e:
        logger.exception("Vision search failed for query: %s", query)
        return json.dumps({"error": f"Search failed: {type(e).__name__}: {e}"})


def vision_face_identify(image_path: str, threshold: str = "0.6") -> str:
    """Identify known faces in an image.

    Args:
        image_path: Path to image file.
        threshold: Similarity threshold (default 0.6).

    Returns:
        JSON string with identified faces.
    """
    path = Path(image_path)
    if not path.exists():
        return json.dumps({"error": f"File not found: {image_path}"})

    try:
        thresh = float(threshold)
    except ValueError:
        return json.dumps({"error": f"Invalid threshold: {threshold}"})

    try:
        from src.vision.models import AnalyzerType

        pipeline = _get_pipeline()
        if not pipeline.is_initialized:
            pipeline.initialize([AnalyzerType.FACE_DETECT, AnalyzerType.FACE_EMBED])

        result = pipeline.analyze(
            image=path,
            analyzers=[AnalyzerType.FACE_DETECT, AnalyzerType.FACE_EMBED],
        )

        faces = []
        if result.faces:
            from src.vision.search import VisionSearch
            search = VisionSearch()

            for face in result.faces:
                if face.embedding is not None:
                    matches = search.search_by_face(
                        face.embedding, threshold=thresh, limit=3,
                    )
                    faces.append({
                        "bbox": face.bbox.to_dict() if face.bbox else None,
                        "confidence": face.confidence,
                        "matches": [
                            {"person": m.person_name, "similarity": m.similarity}
                            for m in matches
                        ],
                    })

        return json.dumps(
            {"image": image_path, "faces_found": len(faces), "faces": faces},
            indent=2,
            default=str,
        )

    except Exception as e:
        logger.exception("Face identification failed for %s", image_path)
        return json.dumps({"error": f"Face ID failed: {type(e).__name__}: {e}"})
