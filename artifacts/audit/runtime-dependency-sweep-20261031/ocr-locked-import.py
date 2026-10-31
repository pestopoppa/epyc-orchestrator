"""Model-free OCR import proof. Does not load or construct a model."""

import hashlib
import importlib.metadata
import importlib.util
import sys
from pathlib import Path

import pypdfium2
import python_multipart
import torch
import torchvision
import transformers
from packaging.version import Version
from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor
from transformers.models.pixtral.image_processing_pixtral import PixtralImageProcessor

root = Path(__file__).resolve().parents[3]
source = root / "src/services/lightonocr_server.py"
assert Version(torch.__version__).base_version == "2.9.0"
assert Version(torchvision.__version__).base_version == "0.24.0"
assert transformers.__version__ == "5.22.1"
assert "+cpu" in torch.__version__ and "+cpu" in torchvision.__version__
spec = importlib.util.spec_from_file_location("lightonocr_import_proof", source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert module.app is not None and module.model is None and module.processor is None
print("python", sys.version)
for name, mod in (
    ("torch", torch),
    ("torchvision", torchvision),
    ("transformers", transformers),
    ("pypdfium2", pypdfium2),
    ("python-multipart", python_multipart),
):
    print(name, importlib.metadata.version(name), mod.__file__)
print(
    "classes",
    LightOnOcrForConditionalGeneration.__name__,
    LightOnOcrProcessor.__name__,
    PixtralImageProcessor.__name__,
)
print("server_source_sha256", hashlib.sha256(source.read_bytes()).hexdigest())
print("server_imported_no_model", module.model is None and module.processor is None)
