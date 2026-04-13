# Backward-compatible shim — actual module moved to src/runtime/inference_tap.py
# Replace this module in sys.modules so monkeypatch and attribute access
# go directly to the real module (not a copy of its namespace).
import importlib as _il, sys as _sys
_real = _il.import_module("src.runtime.inference_tap")
_real.__name__ = __name__  # preserve original module name for logging
_sys.modules[__name__] = _real
