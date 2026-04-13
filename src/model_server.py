# Backward-compatible shim — actual module moved to src/inference/model_server.py
from src.inference.model_server import *  # noqa: F401,F403
from src.inference.model_server import InferenceResult, ModelServer  # explicit re-exports for type checkers
