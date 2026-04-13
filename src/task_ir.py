# Backward-compatible shim — actual module at src/orchestration/task_ir.py
import importlib as _il, sys as _sys
_real = _il.import_module("src.orchestration.task_ir")
_real.__name__ = __name__
_sys.modules[__name__] = _real
