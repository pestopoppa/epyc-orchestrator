"""Resolve the epyc-root checkout that holds the belief-kernel (vidya) carriers.

``/workspace`` is a shared clone that can lag origin/main for hundreds of commits and then lacks
the newest ``scripts/vidya/adapters/*.py`` writers entirely; a producer that hard-defaults to it
fails at the finalize boundary, after the run has spent its compute. Resolution order:

1. an explicit path (argv) wins verbatim;
2. env ``EPYC_ROOT`` when set and non-empty (verbatim: a wrong value must fail loudly, not fall through);
3. ``/mnt/raid0/llm/views/epyc-root-main`` when it exists AND holds the carrier;
4. ``/workspace``.

Stdlib only, no side effects at import.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_VAR = "EPYC_ROOT"
MAIN_VIEW = Path("/mnt/raid0/llm/views/epyc-root-main")
FALLBACK = Path("/workspace")
DEFAULT_CARRIER = "scripts/vidya/adapters/harness_improvement.py"


def resolve_epyc_root(explicit: str | Path | None = None, *,
                      carrier: str = DEFAULT_CARRIER) -> Path:
    """The epyc-root checkout to load belief writers from (see module docstring)."""
    if explicit:
        return Path(explicit)
    env = os.environ.get(ENV_VAR, "").strip()
    if env:
        return Path(env)
    if (MAIN_VIEW / carrier).is_file():
        return MAIN_VIEW
    return FALLBACK


def default_epyc_root(carrier: str = DEFAULT_CARRIER) -> str:
    """String form for argparse defaults."""
    return str(resolve_epyc_root(carrier=carrier))


def load_adapter(name: str, explicit: str | Path | None = None):
    """Import ``<root>/scripts/vidya/adapters/<name>.py`` (a belief-kernel writer) as ``adapters.<name>``.

    Raises FileNotFoundError naming the resolved root when the writer is not there, so a producer
    fails loudly at its write boundary rather than silently emitting nothing.
    """
    import importlib
    import sys

    carrier = f"scripts/vidya/adapters/{name}.py"
    root = resolve_epyc_root(explicit, carrier=carrier)
    vidya = root / "scripts" / "vidya"
    if not (root / carrier).is_file():
        raise FileNotFoundError(f"belief writer {name} not found under {root} (set EPYC_ROOT or --epyc-root)")
    if str(vidya) not in sys.path:
        sys.path.insert(0, str(vidya))
    return importlib.import_module(f"adapters.{name}")
