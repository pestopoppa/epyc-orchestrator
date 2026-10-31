import ast
import builtins
import importlib.util
import re
import tomllib
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]


def load(path, missing=None):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == missing:
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return original(name, *args, **kwargs)

    spec = importlib.util.spec_from_file_location("dependency_test_module", ROOT / path)
    module = importlib.util.module_from_spec(spec)
    with patch("builtins.__import__", guarded):
        spec.loader.exec_module(module)
    return module


class BoundaryTests(unittest.TestCase):
    def test_direct_imports_have_base_distributions(self):
        sources = {
            "src/services/worker_pool.py": {"aiohttp": "aiohttp"},
            "src/db/models/vision.py": {"sqlalchemy": "sqlalchemy"},
            "src/vision/search.py": {"sqlalchemy": "sqlalchemy"},
            "src/graph/langgraph/nodes.py": {"langchain_core": "langchain-core"},
            "src/api/dashboard_cors.py": {"starlette": "starlette"},
        }
        project = tomllib.loads((ROOT / "pyproject.toml").read_text())
        declared = {
            re.split(r"[<>=!~;\[]", item, 1)[0].strip().lower()
            for item in project["project"]["dependencies"]
        }
        for path, mappings in sources.items():
            tree = ast.parse((ROOT / path).read_text())
            roots = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    roots.update(alias.name.split(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    roots.add(node.module.split(".")[0])
            for module, distribution in mappings.items():
                if module in roots:
                    self.assertIn(distribution, declared, f"{path} imports {module}")

    def test_missing_ui(self):
        with self.assertRaisesRegex(ModuleNotFoundError, r"pip install -e.*\[gradio-ui\]"):
            load("src/gradio_ui.py", "gradio")

    @unittest.skipUnless(importlib.util.find_spec("pypdfium2"), "OCR PDF dependency not installed")
    def test_missing_ocr_imports(self):
        for dependency in ("pypdfium2", "python_multipart"):
            with self.subTest(dependency=dependency):
                with self.assertRaisesRegex(ModuleNotFoundError, r"pip install -e.*\[lightonocr\]"):
                    load("src/services/lightonocr_server.py", dependency)

    @unittest.skipUnless(
        all(
            importlib.util.find_spec(name)
            for name in ("pypdfium2", "python_multipart", "fastapi", "PIL", "uvicorn")
        ),
        "OCR import dependencies not installed",
    )
    def test_missing_ocr_model_deps(self):
        module = load("src/services/lightonocr_server.py")
        original = builtins.__import__
        for dependency in ("torch", "torchvision", "transformers"):

            def guarded(name, *args, **kwargs):
                if name == dependency:
                    raise ModuleNotFoundError(name, name=name)
                if name in {"torch", "torchvision"}:
                    return types.ModuleType(name)
                return original(name, *args, **kwargs)

            with self.subTest(dependency=dependency), patch("builtins.__import__", guarded):
                with self.assertRaisesRegex(ModuleNotFoundError, r"pip install -e.*\[lightonocr\]"):
                    module.get_model()

    @unittest.skipUnless(importlib.util.find_spec("gradio"), "Gradio UI extra not installed")
    def test_ui_build_and_stream_format(self):
        src = types.ModuleType("src")
        src.__path__ = []
        config = types.ModuleType("src.config")
        config.get_config = lambda: types.SimpleNamespace(
            server_urls=types.SimpleNamespace(api_url="http://localhost")
        )
        with patch.dict(sys.modules, {"src": src, "src.config": config}):
            module = load("src/gradio_ui.py")
            ui = module.create_ui()
            self.assertIsNotNone(ui)

            class Response:
                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    pass

                def raise_for_status(self):
                    pass

                def iter_lines(self):
                    return iter(['data: {"type":"token","content":"hello"}', "data: [DONE]"])

            class Client:
                def __init__(self, *args, **kwargs):
                    pass

                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    pass

                def stream(self, *args, **kwargs):
                    return Response()

            with patch.object(module.httpx, "Client", Client):
                result = list(module.stream_chat_response("hi", [], "http://localhost"))
            self.assertEqual(
                result[-1][0],
                [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
            )
