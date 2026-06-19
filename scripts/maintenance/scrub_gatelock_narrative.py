#!/usr/bin/env python3
"""Retired gate-lock narrative scrubber.

The original version of this one-shot maintenance script rewrote
``autopilot_journal.jsonl``, edited generated short-term memory, deleted
StrategyStore rows, and rebuilt the FAISS/FTS mirrors in place. That conflicts
with the append-only evidence-plane contract.

Use append-only supersession events and journal-backed repair/report tools
instead:

    python3 scripts/autopilot/scrub_journal.py --dry-run ...
    python3 scripts/autopilot/scrub_journal.py ...
    python3 scripts/autopilot/archive_authority_report.py --strict

Historical implementation remains recoverable from git history.
"""

from __future__ import annotations

import sys
from textwrap import dedent


RETIREMENT_MESSAGE = """\
scrub_gatelock_narrative.py is retired.

It used in-place mutation of the AutoPilot journal, generated STM, strategies
SQLite rows, and FAISS/FTS mirrors. Those writes are forbidden by the current
append-only evidence-plane contract.

Use scripts/autopilot/scrub_journal.py for append-only supersession events, then
validate with scripts/autopilot/archive_authority_report.py --strict. Recover
the old one-shot script from git history only for offline forensics, never
against live runtime state.
"""


def main() -> int:
    sys.stderr.write(dedent(RETIREMENT_MESSAGE))
    if not RETIREMENT_MESSAGE.endswith("\n"):
        sys.stderr.write("\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
