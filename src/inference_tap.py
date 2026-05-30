# Backward-compatible shim — actual module is src/runtime/inference_tap.py.
#
# 2026-05-23: replaced the prior sys.modules-swap pattern (which had a
# concurrent-import race: between `_real = import_module(...)` and
# `sys.modules[__name__] = _real`, a second concurrent import that hit
# `from src.inference_tap import is_active` could see the half-built
# shim namespace and fail with `cannot import name 'is_active'`).
#
# This version uses explicit eager re-exports — every consumer-visible
# symbol is bound on the shim module itself, so attribute lookup
# succeeds even if Python's module system is mid-init somewhere else.
# The original sys.modules swap is retained as a best-effort optimization
# so monkeypatches still affect the real module, but it's no longer
# load-bearing for correctness.

from src.runtime.inference_tap import (  # noqa: F401  (re-export)
    annotate_current_tap,
    is_active,
    stream_mode,
    should_stream_role,
    tap_section,
)

# Best-effort sys.modules pin so test monkeypatches that import the
# canonical path see the same module object. Race-safe in the sense
# that callers fetching `is_active` via `from src.inference_tap import
# is_active` get the eagerly-bound name above regardless of swap state.
import importlib as _il
import sys as _sys
try:
    _real = _il.import_module("src.runtime.inference_tap")
    _sys.modules[__name__] = _real
except Exception:
    pass
