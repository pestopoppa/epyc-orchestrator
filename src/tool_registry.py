# Backward-compatible shim — actual module at src/registry/tool_registry.py
import importlib as _il, sys as _sys
_real = _il.import_module("src.registry.tool_registry")
_real.__name__ = __name__
_sys.modules[__name__] = _real
