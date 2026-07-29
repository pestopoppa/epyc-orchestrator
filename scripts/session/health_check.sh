#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

status=0

check_file() {
  local path="$1"
  if [[ -e "$path" ]]; then
    printf 'ok: %s\n' "$path"
  else
    printf 'missing: %s\n' "$path" >&2
    status=1
  fi
}

check_command() {
  local command_name="$1"
  if command -v "$command_name" >/dev/null 2>&1; then
    printf 'ok: command %s\n' "$command_name"
  else
    printf 'missing: command %s\n' "$command_name" >&2
    status=1
  fi
}

check_python() {
  local path="$1"
  if python3 -m py_compile "$path"; then
    printf 'ok: py_compile %s\n' "$path"
  else
    status=1
  fi
}

check_command python3
check_command uv
check_file pyproject.toml
check_file uv.lock
check_file Makefile
check_file orchestration/model_registry.yaml
check_file scripts/server/orchestrator_stack.py
check_file scripts/autopilot/autopilot.py
check_file scripts/security/audit_repository.py

check_python scripts/server/stack_health.py
check_python scripts/autopilot/phase_health_report.py
check_python scripts/security/audit_repository.py

if timeout 90s python3 scripts/maintenance/check_episodic_integrity.py --semantic --require-semantic; then
  printf 'ok: episodic semantic integrity\n'
else
  printf 'failed: episodic semantic integrity\n' >&2
  status=1
fi

if [[ -x scripts/security/audit_repository.py ]]; then
  scripts/security/audit_repository.py >/dev/null
else
  python3 scripts/security/audit_repository.py >/dev/null
fi
printf 'ok: security audit\n'

exit "$status"
