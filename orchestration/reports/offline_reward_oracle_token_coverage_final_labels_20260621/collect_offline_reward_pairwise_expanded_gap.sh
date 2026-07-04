#!/bin/bash
set -euo pipefail

RUN_TS="${A9_COLLECTION_TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
if [[ ! "$RUN_TS" =~ ^[0-9]{8}T[0-9]{6}Z$ ]]; then
  echo "invalid A9 collection timestamp: $RUN_TS" >&2
  exit 64
fi
if pgrep -af 'scripts/autopilot/autopilot.py start' >/dev/null; then
  echo 'refusing A9 collection while AutoPilot is active' >&2
  exit 75
fi
cd /mnt/raid0/llm/epyc-orchestrator
