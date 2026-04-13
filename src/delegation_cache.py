# Backward-compatible shim — actual module at src/orchestration/delegation_cache.py
import importlib as _il, sys as _sys
_real = _il.import_module("src.orchestration.delegation_cache")
_real.__name__ = __name__
_sys.modules[__name__] = _real
