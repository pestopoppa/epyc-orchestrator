# Backward-compatible shim — actual module at src/inference/model_server.py
import importlib as _il, sys as _sys
_real = _il.import_module("src.inference.model_server")
_real.__name__ = __name__
_sys.modules[__name__] = _real
