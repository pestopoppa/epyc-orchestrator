"""Declared HIL-6 pilot instances (build only; nothing here runs a loop).

Each module holds one pilot design as data plus the pure helpers that freeze it. A design is a
:class:`src.harness_loop.pilot.PilotDesign`; the driver refuses to construct while
``unmet_requirements(design)`` is non-empty. The 2026-09-16 deferment on RUNNING a
harness-improvement loop is in force: there is no entrypoint here, and no non-test module imports
this package.
"""

from __future__ import annotations

__all__ = ["first_feature"]
