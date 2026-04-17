"""Tests for vision tool plugin registration and handlers."""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.registry.tool_loader import ToolPluginLoader


class TestVisionPluginLoading:
    """Test that the vision plugin loads correctly via ToolPluginLoader."""

    def test_plugin_discovered(self):
        """Vision plugin is discovered and loaded."""
        loader = ToolPluginLoader()
        n = loader.discover_plugins(Path("src/tools"))
        assert n >= 1
        assert "vision" in loader.plugins

    def test_plugin_has_three_tools(self):
        """Vision plugin declares 3 tools."""
        loader = ToolPluginLoader()
        loader.discover_plugins(Path("src/tools"))
        manifest = loader.plugins["vision"]
        assert len(manifest.tools) == 3
        tool_names = {t.name for t in manifest.tools}
        assert tool_names == {"vision_analyze", "vision_search", "vision_face_identify"}

    def test_tools_are_specialized_category(self):
        """All vision tools have specialized category."""
        loader = ToolPluginLoader()
        loader.discover_plugins(Path("src/tools"))
        for tool in loader.plugins["vision"].tools:
            assert tool.category == "specialized"

    def test_vision_analyze_handler_loadable(self):
        """Handler function for vision_analyze can be imported."""
        loader = ToolPluginLoader()
        loader.discover_plugins(Path("src/tools"))
        handler = loader.get_handler("vision_analyze")
        assert handler is not None
        assert callable(handler)

    def test_vision_search_handler_loadable(self):
        """Handler function for vision_search can be imported."""
        loader = ToolPluginLoader()
        loader.discover_plugins(Path("src/tools"))
        handler = loader.get_handler("vision_search")
        assert handler is not None
        assert callable(handler)


class TestVisionAnalyzeHandler:
    """Test vision_analyze handler logic."""

    def test_file_not_found(self):
        """Missing file returns error JSON."""
        from src.tools.vision.analyze import vision_analyze

        result = vision_analyze("/nonexistent/path/image.jpg")
        parsed = json.loads(result)
        assert "error" in parsed
        assert "not found" in parsed["error"].lower()

    def test_unknown_analyzer_returns_error(self, tmp_path):
        """Unknown analyzer name returns error."""
        from src.tools.vision.analyze import vision_analyze

        img = tmp_path / "test.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)  # Minimal JPEG header

        result = vision_analyze(str(img), analyzers="nonexistent_analyzer")
        parsed = json.loads(result)
        assert "error" in parsed
        assert "unknown analyzer" in parsed["error"].lower()

    @patch("src.tools.vision.analyze._check_multimodal_available", return_value=False)
    def test_vl_analyzers_skipped_without_multimodal(self, mock_check, tmp_path):
        """VL analyzers gracefully skipped when no multimodal model active."""
        from src.tools.vision.analyze import vision_analyze

        img = tmp_path / "test.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

        # Request only VL analyzers with no multimodal model
        result = vision_analyze(str(img), analyzers="vl_describe")
        parsed = json.loads(result)
        assert "error" in parsed
        assert "multimodal" in parsed["error"].lower()

    @patch("src.tools.vision.analyze._check_multimodal_available", return_value=False)
    def test_mixed_analyzers_fallback_to_non_vl(self, mock_check, tmp_path):
        """Mixed VL + non-VL analyzers: VL silently dropped, non-VL proceed."""
        from src.tools.vision.analyze import vision_analyze

        img = tmp_path / "test.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

        # Mock the pipeline to avoid actual analysis
        mock_pipeline = MagicMock()
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"analyzers_run": ["exif_extract"]}
        mock_pipeline.is_initialized = True
        mock_pipeline.analyze.return_value = mock_result

        with patch("src.tools.vision.analyze._get_pipeline", return_value=mock_pipeline):
            result = vision_analyze(str(img), analyzers="exif,vl_describe")
            parsed = json.loads(result)
            assert "error" not in parsed


class TestMultimodalDetection:
    """Test model-agnostic multimodal capability detection."""

    def test_detects_mmproj_model(self):
        """Detects multimodal capability via mmproj_path."""
        from src.tools.vision.analyze import _check_multimodal_available

        mock_role = MagicMock()
        mock_role.model.mmproj_path = "/path/to/mmproj.gguf"
        mock_roles = {"worker_vision": mock_role}

        with patch("src.tools.vision.analyze._load_registry_roles", return_value=mock_roles):
            assert _check_multimodal_available() is True

    def test_no_multimodal_when_no_mmproj(self):
        """No multimodal when no role has mmproj_path."""
        from src.tools.vision.analyze import _check_multimodal_available

        mock_role = MagicMock()
        mock_role.model.mmproj_path = None
        mock_roles = {"frontdoor": mock_role}

        with patch("src.tools.vision.analyze._load_registry_roles", return_value=mock_roles):
            assert _check_multimodal_available() is False

    def test_no_registry_returns_false(self):
        """Registry load failure returns False gracefully."""
        from src.tools.vision.analyze import _check_multimodal_available

        with patch("src.tools.vision.analyze._load_registry_roles", side_effect=Exception("no registry")):
            assert _check_multimodal_available() is False
