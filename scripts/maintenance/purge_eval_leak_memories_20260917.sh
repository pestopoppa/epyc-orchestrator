#!/bin/bash
# purge_eval_leak_memories_20260917.sh — operator wrapper for purge_eval_leak_memories_20260917.py
#
# Removes the HumanEval/55 and simpleqa_general_00912/00795 (Aschoff) eval-pool leaks from every
# episodic.db (keeping FAISS + id_map consistent) and redacts them from autopilot_state backups.
# See the Python docstring for the full contract.
#
#   Inventory (default, writes only the signature cache):  bash purge_eval_leak_memories_20260917.sh
#   Offline stores + state backups:                         ... --apply
#   Live store (API STOPPED, AutoPilot stopped):            ... --apply --include-live
#   production_best v10 (after RATIFY-CKPT-PROMPT-LEAK):    ... --apply --include-pinned
#   Post-state check (read-only, exit 1 on any leak):       ... --verify [--include-live] [--include-pinned]
#
# Environment: ORCH (orchestrator data tree, default /mnt/raid0/llm/epyc-orchestrator),
# BACKUP_ROOT (default /mnt/raid0/llm/backups/episodic-leak-20260917), PYTHON (needs faiss+numpy;
# default: $ORCH/.venv/bin/python).
# Exit: 0 ok, 1 a store/file failed or a leak remains (--verify), 2 refused, 3 live store refused
# because the API/AutoPilot is up (the operator procedure is printed).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORCH="${ORCH:-/mnt/raid0/llm/epyc-orchestrator}"
PYTHON="${PYTHON:-$ORCH/.venv/bin/python}"
export ORCH
export BACKUP_ROOT="${BACKUP_ROOT:-/mnt/raid0/llm/backups/episodic-leak-20260917}"

if ! "$PYTHON" -c 'import faiss, numpy' 2>/dev/null; then
  echo "REFUSED: $PYTHON cannot import faiss+numpy (set PYTHON=...)" >&2
  exit 2
fi

mkdir -p "$BACKUP_ROOT"
LOG="$BACKUP_ROOT/run-$(date -u +%Y%m%dT%H%M%SZ).log"
set +e
"$PYTHON" "$HERE/purge_eval_leak_memories_20260917.py" --orch-root "$ORCH" "$@" 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e
echo "[log] $LOG (exit $rc)"
exit "$rc"
