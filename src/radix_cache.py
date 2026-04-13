# Backward-compatible shim — actual module at src/inference/radix_cache.py
import importlib as _il, sys as _sys
_real = _il.import_module("src.inference.radix_cache")
_real.__name__ = __name__
_sys.modules[__name__] = _real
