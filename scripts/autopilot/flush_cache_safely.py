"""Operator-facing wrapper for `host_health.flush_cache_with_pause()`.

The canonical way to invoke a deliberate page-cache flush without polluting
the autopilot journal. Pauses autopilot via state.json (which the 2026-05-24
loop fix actually honors), runs `sudo /usr/local/sbin/autopilot-flush-cache`,
NUMA-interleave-rewarms every active role's primary GGUF serially, then
restores the pre-flush paused state.

Use this instead of calling `sudo /usr/local/sbin/autopilot-flush-cache`
directly. The bare flush leaves all role GGUFs cold + risks NUMA-pinned
pages on the next non-NUMA-aware re-read (per
`feedback_drop_caches_numa_eviction`).

Usage:
    python scripts/autopilot/flush_cache_safely.py
    python scripts/autopilot/flush_cache_safely.py --no-rewarm   # flush only
    python scripts/autopilot/flush_cache_safely.py --verbose
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure module path resolves regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.autopilot.host_health import flush_cache_with_pause, _DEFAULT_REWARM_GGUFS


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--no-rewarm", action="store_true",
                   help="skip NUMA-interleave rewarm (NOT recommended; next "
                        "non-NUMA-aware re-read will pin to one NUMA node)")
    p.add_argument("--verbose", "-v", action="store_true")
    p.add_argument("--state-path", type=Path, default=None,
                   help="override autopilot_state.json location")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    result = flush_cache_with_pause(
        state_path=args.state_path,
        rewarm=not args.no_rewarm,
        rewarm_paths=_DEFAULT_REWARM_GGUFS,
    )
    print(json.dumps({k: (v if not isinstance(v, dict) else {Path(p).name: ok for p, ok in v.items()})
                      for k, v in result.items()}, indent=2))
    return 0 if result.get("flush_ok") else 1


if __name__ == "__main__":
    sys.exit(main())
