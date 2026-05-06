"""Thin entry-point: `python scripts/trace/cli.py ...` or `python -m src.trace.cli ...`."""

from __future__ import annotations

import sys
from pathlib import Path

# Make src/ importable when invoked directly.
_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.trace.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
