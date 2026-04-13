# Backward-compatible shim — actual module at src/runtime/inference_lock.py
import importlib as _il, sys as _sys
_real = _il.import_module("src.runtime.inference_lock")
_real.__name__ = __name__
_sys.modules[__name__] = _real
