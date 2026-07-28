#!/bin/bash
# Proposed one-step human v5 protocol ratification and baseline apply transaction.
set -euo pipefail

ORCH="/mnt/raid0/llm/epyc-orchestrator"
ROOT="/mnt/raid0/llm/epyc-root"
SOURCE_ROOT="${E8_V5_SOURCE_ROOT:-$ORCH}"
PYTHON="$ORCH/.venv/bin/python"
RUNNER="$SOURCE_ROOT/scripts/benchmark/run_e8_quality_baseline_v5.py"
RESUME_RUNNER="$SOURCE_ROOT/scripts/benchmark/resume_e8_quality_baseline_v5.py"
BASE_RUNNER="$SOURCE_ROOT/scripts/benchmark/run_e8_quality_baseline_reseed.py"
VALIDATOR="$SOURCE_ROOT/scripts/benchmark/operator_candidates/prepare_e8_quality_baseline_v5_candidate.sh"
VALIDATOR_PY="$SOURCE_ROOT/scripts/benchmark/validate_e8_quality_baseline_v5.py"
APPLIER="$SOURCE_ROOT/scripts/benchmark/operator_candidates/apply_e8_quality_baseline_state_v5_candidate.py"
CANONICAL_APPLIER="$ROOT/artifacts/operator/apply_e8_quality_baseline_state.py"
STATE="$ORCH/orchestration/autopilot_state.json"
LOCK="/mnt/raid0/llm/tmp/e8-quality-baseline-v5-apply.lock"
TOKEN="ATTEST-E8-QUALITY-V5-GENERATION-TAIL-AND-BASELINE-APPLY-20260727"

fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
sha() { sha256sum -- "$1" | awk '{print $1}'; }
verify_reviewed_bindings() {
    for binding in \
        "E8_V5_WRAPPER_SHA256:$0" \
        "E8_V5_RUNNER_SHA256:$RUNNER" \
        "E8_V5_RESUME_RUNNER_SHA256:$RESUME_RUNNER" \
        "E8_V5_BASE_RUNNER_SHA256:$BASE_RUNNER" \
        "E8_V5_VALIDATOR_SHA256:$VALIDATOR" \
        "E8_V5_VALIDATOR_PY_SHA256:$VALIDATOR_PY" \
        "E8_V5_APPLIER_SHA256:$APPLIER" \
        "E8_V5_CANONICAL_APPLIER_SHA256:$CANONICAL_APPLIER"; do
        name="${binding%%:*}"
        path="${binding#*:}"
        expected="${!name:-}"
        [[ "$expected" =~ ^[0-9a-f]{64}$ && -f "$path" && "$(sha "$path")" == "$expected" ]] ||
            fail "reviewed artifact pin differs: $name"
    done
}
case "${1:-}" in
    --attest)
        [[ $# -eq 8 && "$3" == "--evidence" && "$5" == "--expected-pre-state-sha256" && "$7" == "--expected-candidate-state-sha256" ]] ||
            fail 'usage: ratify_and_apply_e8_quality_baseline_v5.sh --attest TOKEN --evidence EVIDENCE --expected-pre-state-sha256 SHA --expected-candidate-state-sha256 SHA'
        [[ "$2" == "$TOKEN" ]] || fail "attestation token differs; use: $TOKEN"
        MODE="attest"
        EVIDENCE="$4"
        EXPECTED_PRE="$6"
        EXPECTED_CANDIDATE="$8"
        ;;
    --prevalidate)
        [[ $# -eq 7 && "$2" == "--evidence" && "$4" == "--expected-pre-state-sha256" && "$6" == "--expected-candidate-state-sha256" ]] ||
            fail 'usage: ratify_and_apply_e8_quality_baseline_v5.sh --prevalidate --evidence EVIDENCE --expected-pre-state-sha256 SHA --expected-candidate-state-sha256 SHA'
        MODE="prevalidate"
        EVIDENCE="$3"
        EXPECTED_PRE="$5"
        EXPECTED_CANDIDATE="$7"
        ;;
    *)
        fail 'first argument must be --attest or --prevalidate'
        ;;
esac
[[ "$EVIDENCE" = /* && -f "$EVIDENCE" ]] || fail 'evidence must be an existing absolute path'
[[ "$EXPECTED_PRE" =~ ^[0-9a-f]{64}$ && "$EXPECTED_CANDIDATE" =~ ^[0-9a-f]{64}$ ]] ||
    fail 'reviewed state hashes must be lowercase SHA-256'
[[ -x "$PYTHON" ]] || fail 'canonical orchestrator venv is unavailable'
[[ "$(readlink -f -- "$PYTHON")" == "$(readlink -f -- "$ORCH/.venv/bin/python")" ]] ||
    fail 'canonical orchestrator venv identity differs'
verify_reviewed_bindings
[[ "${E8_V5_ORCHESTRATOR_HEAD:-}" =~ ^[0-9a-f]{40}$ && "$(git -C "$SOURCE_ROOT" rev-parse HEAD)" == "$E8_V5_ORCHESTRATOR_HEAD" ]] ||
    fail 'canonical orchestrator HEAD differs from the reviewed source pin'
[[ -z "$(git -C "$SOURCE_ROOT" status --porcelain)" ]] ||
    fail 'canonical orchestrator tracked worktree is not clean'

if [[ "$MODE" == "attest" ]]; then
    exec 9>"$LOCK"
    flock -n 9 || fail 'another v5 apply owns the lock'
fi
bash "$VALIDATOR" --validate-evidence "$EVIDENCE"

EVIDENCE_SHA256="$(sha "$EVIDENCE")"
TRANSACTION="$ROOT/artifacts/operator/e8_quality_baseline_state_v5_${EVIDENCE_SHA256}.transaction"
ATTESTATION="$ROOT/artifacts/operator/e8_quality_baseline_state_v5_${EVIDENCE_SHA256}.apply_attestation.json"
PROTOCOL_RECEIPT="$ROOT/artifacts/operator/e8_quality_baseline_protocol_v5_${EVIDENCE_SHA256}.ratification.json"
COMMON=(
    --state "$STATE"
    --evidence "$EVIDENCE"
    --canonical-evidence "$EVIDENCE"
    --validator "$VALIDATOR"
    --transaction-dir "$TRANSACTION"
    --attestation "$ATTESTATION"
    --expected-pre-state-sha256 "$EXPECTED_PRE"
    --expected-candidate-state-sha256 "$EXPECTED_CANDIDATE"
)
verify_reviewed_bindings
E8_BASELINE_APPLY_TOKEN="$TOKEN" PYTHONOPTIMIZE=0 "$PYTHON" "$APPLIER" "${COMMON[@]}" --validate-only
if [[ "$MODE" == "prevalidate" ]]; then
    printf 'E8 v5 ratify-and-apply prevalidation passed; no state or receipt changed.\n'
    exit 0
fi

PYTHONOPTIMIZE=0 "$PYTHON" - \
    "$PROTOCOL_RECEIPT" "$EVIDENCE" "$0" "$RUNNER" "$RESUME_RUNNER" "$BASE_RUNNER" "$VALIDATOR" \
    "$VALIDATOR_PY" "$APPLIER" "$CANONICAL_APPLIER" \
    "$EXPECTED_PRE" "$EXPECTED_CANDIDATE" "$TOKEN" <<'PY'
import hashlib, json, os
from datetime import datetime, timezone
from pathlib import Path
import sys
output, evidence, wrapper, runner, resume_runner, base_runner, validator, validator_py, applier, canonical_applier = map(
    Path, sys.argv[1:11]
)
pre, candidate, token = sys.argv[11:]
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
manifest = json.loads(evidence.read_text())
protocol = Path(manifest["protocol_candidate"]["path"])
payload = {
    "schema": "epyc.operator_e8_quality_baseline_v5_protocol_ratification.v1",
    "decision": token,
    "ratified_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "evidence": str(evidence.resolve()), "evidence_sha256": sha(evidence),
    "protocol_candidate": str(protocol.resolve()), "protocol_candidate_sha256": sha(protocol),
    "protocol_id": "e8_quality_full_pool_tier_baseline.v5",
    "reviewed_artifact_sha256": {
        "wrapper": sha(wrapper),
        "runner": sha(runner),
        "resume_runner": sha(resume_runner),
        "base_runner": sha(base_runner),
        "validator": sha(validator),
        "validator_python": sha(validator_py),
        "applier_adapter": sha(applier),
        "canonical_applier": sha(canonical_applier),
    },
    "pre_state_sha256": pre, "candidate_state_sha256": candidate,
}
if output.exists():
    existing = json.loads(output.read_text())
    existing_without_time = dict(existing)
    ratified_at = existing_without_time.pop("ratified_at", None)
    expected_without_time = dict(payload)
    expected_without_time.pop("ratified_at")
    if not isinstance(ratified_at, str) or existing_without_time != expected_without_time:
        raise RuntimeError("existing protocol ratification differs from this reviewed transaction")
    raise SystemExit(0)
data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
fd = os.open(output, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
try:
    offset = 0
    while offset < len(data):
        written = os.write(fd, memoryview(data)[offset:])
        if written <= 0 or written > len(data) - offset:
            raise OSError(f"receipt write made invalid progress: {written}")
        offset += written
    os.fsync(fd)
finally:
    os.close(fd)
dir_fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
try:
    os.fsync(dir_fd)
finally:
    os.close(dir_fd)
PY

verify_reviewed_bindings
E8_BASELINE_APPLY_TOKEN="$TOKEN" PYTHONOPTIMIZE=0 "$PYTHON" "$APPLIER" "${COMMON[@]}" --attest "$TOKEN"
