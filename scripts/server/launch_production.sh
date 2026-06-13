#!/bin/bash
set -euo pipefail

# Deterministic production orchestrator stack launcher
# Usage: ./launch_production.sh [--full | --minimal | --with-burst | --dev]
# Default: Full HOT tier with all architects (~510GB, 45% of 1130GB RAM)
#
# This script provides a simple, deterministic way to launch the orchestrator
# stack without requiring agent interpretation.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source environment library for path variables
# shellcheck source=../lib/env.sh
source "${SCRIPT_DIR}/../lib/env.sh"

STACK_PY="$SCRIPT_DIR/orchestrator_stack.py"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

print_manifest_summary() {
  local tier="${1:-hot}"
  PYTHONPATH="$REPO_ROOT" python3 - "$tier" <<'PY'
import sys

from scripts.server.stack_manifest import HOT_SERVERS, WARM_SERVERS

tier = sys.argv[1]
servers = HOT_SERVERS if tier == "hot" else WARM_SERVERS

seen: set[tuple[str, ...]] = set()
for server in servers:
    roles = tuple(server.get("roles") or ())
    if not roles or roles in seen:
        continue
    seen.add(roles)
    port = server.get("port")
    flags: list[str] = []
    if server.get("worker_pool"):
        flags.append(f"worker_pool:{server.get('worker_type')}")
    if server.get("vision"):
        flags.append(f"vision:{server.get('vision_type')}")
    if server.get("embedding"):
        flags.append("embedding")
    suffix = f" ({', '.join(flags)})" if flags else ""
    print(f"  - {', '.join(roles)} ({port}){suffix}")
PY
}

# Validate script exists
if [[ ! -f "$STACK_PY" ]]; then
  echo "ERROR: orchestrator_stack.py not found at $STACK_PY"
  exit 1
fi

# Parse mode
MODE="${1:---full}"

echo "============================================================"
echo "PRODUCTION ORCHESTRATOR STACK LAUNCHER"
echo "============================================================"
echo ""

case "$MODE" in
  --full)
    echo "Mode: FULL production stack (manifest-defined HOT tier)"
    echo ""
    echo "HOT launch groups:"
    print_manifest_summary hot
    echo ""
    python3 "$STACK_PY" start
    ;;
  --minimal)
    echo "Mode: MINIMAL legacy alias (manifest-defined HOT tier only)"
    echo ""
    echo "HOT launch groups:"
    print_manifest_summary hot
    echo ""
    echo "NOTE: The stack manager no longer exposes a separate core-only tier."
    echo ""
    python3 "$STACK_PY" start --hot-only
    ;;
  --with-burst)
    echo "Mode: FULL + burst worker"
    echo ""
    echo "HOT launch groups:"
    print_manifest_summary hot
    echo ""
    echo "WARM launch groups requested:"
    print_manifest_summary warm
    echo ""
    python3 "$STACK_PY" start --include-warm worker_fast
    ;;
  --dev)
    echo "Mode: DEV (single 0.5B model)"
    echo ""
    echo "Single model for testing/development"
    echo ""
    python3 "$STACK_PY" start --dev
    ;;
  --status)
    echo "Checking stack status..."
    echo ""
    python3 "$STACK_PY" status
    exit 0
    ;;
  --stop)
    echo "Stopping all components..."
    echo ""
    python3 "$STACK_PY" stop --all
    exit 0
    ;;
  --help | -h)
    echo "Usage: $0 [--full | --minimal | --with-burst | --dev | --status | --stop]"
    echo ""
    echo "Modes:"
    echo "  --full (default)  Manifest-defined HOT tier"
    echo "  --minimal         Legacy alias for HOT tier only, launched via --hot-only"
    echo "  --with-burst      HOT tier + manifest-defined warm burst worker"
    echo "  --dev             Single 0.5B model for testing"
    echo ""
    echo "Commands:"
    echo "  --status          Show current stack status"
    echo "  --stop            Stop all running components"
    echo ""
    echo "Launch inventory comes from scripts/server/stack_manifest.py."
    echo "Use --status after launch for live processes, ports, and residency."
    exit 0
    ;;
  *)
    echo "Unknown mode: $MODE"
    echo "Use --help for usage information"
    exit 1
    ;;
esac

# Verify health
echo ""
echo "============================================================"
echo "HEALTH CHECK"
echo "============================================================"
python3 "$STACK_PY" status
