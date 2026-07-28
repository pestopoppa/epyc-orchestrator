#!/bin/bash
# Proposed one-step human v5 protocol ratification and baseline apply transaction.
set -euo pipefail

ORCH="/mnt/raid0/llm/epyc-orchestrator"
CANONICAL_ROOT="/mnt/raid0/llm/epyc-root"
ROOT="${E8_V5_OPERATOR_ROOT:-$CANONICAL_ROOT}"
SOURCE_ROOT_RAW="${E8_V5_SOURCE_ROOT:-$ORCH}"
PYTHON_RAW="${E8_V5_PYTHON:-$ORCH/.venv/bin/python}"
[[ "$SOURCE_ROOT_RAW" = /* && "$PYTHON_RAW" = /* ]] || {
    printf 'ERROR: E8 v5 source root and interpreter must be absolute paths\n' >&2
    exit 1
}
SOURCE_ROOT="$(realpath -e -- "$SOURCE_ROOT_RAW")"
PYTHON="$(readlink -f -- "$PYTHON_RAW")"
CANONICAL_PYTHON="$(readlink -f -- "$ORCH/.venv/bin/python")"
WRAPPER="$(readlink -f -- "$0")"
EXPECTED_WRAPPER="$SOURCE_ROOT/scripts/benchmark/operator_candidates/ratify_and_apply_e8_quality_baseline_v5.sh"
RUNNER="$SOURCE_ROOT/scripts/benchmark/run_e8_quality_baseline_v5.py"
PRODUCER="$SOURCE_ROOT/scripts/benchmark/terminalize_e8_quality_baseline_source.py"
RESUME_RUNNER="$SOURCE_ROOT/scripts/benchmark/resume_e8_quality_baseline_v5.py"
BASE_RUNNER="$SOURCE_ROOT/scripts/benchmark/run_e8_quality_baseline_reseed.py"
RECOVERY_RUNNER="$SOURCE_ROOT/scripts/benchmark/recover_e8_quality_baseline_v5_partial_r2.py"
FINALIZER_RUNNER="$SOURCE_ROOT/scripts/benchmark/finalize_e8_quality_baseline_v5_recovery_r2.py"
SUCCESSOR_RUNNER="$SOURCE_ROOT/scripts/benchmark/prepare_e8_quality_baseline_v5_partial_r2_successor.py"
RACE_RETRY_RUNNER="$SOURCE_ROOT/scripts/benchmark/prepare_e8_quality_baseline_v5_partial_r2_race_retry.py"
VALIDATOR="$SOURCE_ROOT/scripts/benchmark/operator_candidates/prepare_e8_quality_baseline_v5_candidate.sh"
VALIDATOR_PY="$SOURCE_ROOT/scripts/benchmark/validate_e8_quality_baseline_v5.py"
APPLIER="$SOURCE_ROOT/scripts/benchmark/operator_candidates/apply_e8_quality_baseline_state_v5_candidate.py"
CANONICAL_APPLIER="$CANONICAL_ROOT/artifacts/operator/apply_e8_quality_baseline_state.py"
STATE="${E8_V5_STATE:-$ORCH/orchestration/autopilot_state.json}"
LOCK="${E8_V5_LOCK_PATH:-/mnt/raid0/llm/tmp/e8-quality-baseline-v5-apply.lock}"
TOKEN="ATTEST-E8-QUALITY-V5-GENERATION-TAIL-AND-BASELINE-APPLY-20260727"

fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
sha() { sha256sum -- "$1" | awk '{print $1}'; }
TEST_SANDBOX=0
override_count=0
for variable in E8_V5_OPERATOR_ROOT E8_V5_STATE E8_V5_LOCK_PATH; do
    [[ -n "${!variable:-}" ]] && ((override_count += 1))
done
if (( override_count == 3 )); then
    [[ "${E8_V5_TEST_MODE:-}" == "1" && -n "${PYTEST_CURRENT_TEST:-}" ]] ||
        fail 'noncanonical state/artifact paths are pytest-only'
    [[ -d "$ROOT" && -f "$STATE" && -d "$(dirname -- "$LOCK")" ]] ||
        fail 'pytest-only root, state, and lock parent must already exist'
    [[ ! -L "$ROOT" && ! -L "$STATE" && ! -L "$(dirname -- "$LOCK")" && ! -L "$LOCK" ]] ||
        fail 'pytest-only paths must not be symlinks'
    ROOT_LEXICAL="$(realpath -ms -- "$ROOT")"
    STATE_LEXICAL="$(realpath -ms -- "$STATE")"
    LOCK_PARENT_LEXICAL="$(realpath -ms -- "$(dirname -- "$LOCK")")"
    ROOT_RESOLVED="$(realpath -e -- "$ROOT")"
    STATE_RESOLVED="$(realpath -e -- "$STATE")"
    LOCK_PARENT_RESOLVED="$(realpath -e -- "$(dirname -- "$LOCK")")"
    [[ "$ROOT_LEXICAL" == "$ROOT_RESOLVED" && "$STATE_LEXICAL" == "$STATE_RESOLVED" && "$LOCK_PARENT_LEXICAL" == "$LOCK_PARENT_RESOLVED" ]] ||
        fail 'pytest-only paths must not traverse symlinked components'
    [[ "$ROOT_RESOLVED" == /tmp/* && "$STATE_RESOLVED" == /tmp/* && "$LOCK_PARENT_RESOLVED" == /tmp/* ]] ||
        fail 'pytest-only resolved paths must remain below /tmp'
    [[ "$(stat -c '%d:%i' -- "$STATE_RESOLVED")" != "$(stat -c '%d:%i' -- "$ORCH/orchestration/autopilot_state.json")" ]] ||
        fail 'pytest-only state must not resolve to the canonical production state inode'
    ROOT="$ROOT_RESOLVED"
    STATE="$STATE_RESOLVED"
    LOCK="$LOCK_PARENT_RESOLVED/$(basename -- "$LOCK")"
    TEST_SANDBOX=1
elif (( override_count != 0 )); then
    fail 'test sandbox requires all of E8_V5_OPERATOR_ROOT, E8_V5_STATE, and E8_V5_LOCK_PATH'
fi
TEST_REVIEW_PATH="${E8_V5_TEST_REVIEW_PATH:-}"
if [[ -n "$TEST_REVIEW_PATH" ]]; then
    [[ "${E8_V5_TEST_MODE:-}" == "1" && -n "${PYTEST_CURRENT_TEST:-}" && "$TEST_REVIEW_PATH" = /tmp/* ]] ||
        fail 'noncanonical state-review path is pytest-only and must remain below /tmp'
    [[ ! -L "$TEST_REVIEW_PATH" && -d "$(dirname -- "$TEST_REVIEW_PATH")" ]] ||
        fail 'pytest-only state-review path must have a real existing parent'
    TEST_REVIEW_PATH="$(realpath -ms -- "$TEST_REVIEW_PATH")"
    [[ "$TEST_REVIEW_PATH" == /tmp/* ]] ||
        fail 'pytest-only state-review path escapes /tmp after canonicalization'
fi

reviewed_artifact_bindings_json() {
    env -u PYTHONPATH -u PYTHONHOME -u PYTHONSTARTUP PYTHONNOUSERSITE=1 PYTHONOPTIMIZE=0 "$PYTHON" -I - \
        "$WRAPPER" "$PRODUCER" "$RUNNER" "$RESUME_RUNNER" "$BASE_RUNNER" \
        "$RECOVERY_RUNNER" "$FINALIZER_RUNNER" "$SUCCESSOR_RUNNER" "$RACE_RETRY_RUNNER" "$VALIDATOR" \
        "$VALIDATOR_PY" "$APPLIER" "$CANONICAL_APPLIER" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

names = (
    "wrapper", "producer", "runner", "resume_runner", "base_runner",
    "recovery_runner", "finalizer_runner", "successor_runner", "race_retry_runner", "validator",
    "validator_python", "applier_adapter", "canonical_applier",
)
print(json.dumps({name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                  for name, path in zip(names, sys.argv[1:], strict=True)}, sort_keys=True))
PY
}

verify_clean_source_root() {
    [[ "${E8_V5_ORCHESTRATOR_HEAD:-}" =~ ^[0-9a-f]{40}$ && "$(git -C "$SOURCE_ROOT" rev-parse HEAD)" == "$E8_V5_ORCHESTRATOR_HEAD" ]] ||
        fail 'canonical orchestrator HEAD differs from the reviewed source pin'
    [[ -z "$(git -C "$SOURCE_ROOT" status --porcelain=v1 --untracked-files=all)" ]] ||
        fail 'reviewed source worktree is not clean'
}

verify_reviewed_bindings() {
    [[ "$WRAPPER" == "$(readlink -f -- "$EXPECTED_WRAPPER")" ]] ||
        fail 'reviewed source root does not own the invoked wrapper'
    [[ -d "$SOURCE_ROOT/.git" || -f "$SOURCE_ROOT/.git" ]] ||
        fail 'reviewed source root is not a git worktree'
    [[ -x "$PYTHON" && "$PYTHON" == "$CANONICAL_PYTHON" ]] ||
        fail 'canonical orchestrator venv identity differs'
    verify_clean_source_root
    for binding in \
        "E8_V5_WRAPPER_SHA256:$WRAPPER" \
        "E8_V5_PRODUCER_SHA256:$PRODUCER" \
        "E8_V5_RUNNER_SHA256:$RUNNER" \
        "E8_V5_RESUME_RUNNER_SHA256:$RESUME_RUNNER" \
        "E8_V5_BASE_RUNNER_SHA256:$BASE_RUNNER" \
        "E8_V5_RECOVERY_RUNNER_SHA256:$RECOVERY_RUNNER" \
        "E8_V5_FINALIZER_RUNNER_SHA256:$FINALIZER_RUNNER" \
        "E8_V5_SUCCESSOR_RUNNER_SHA256:$SUCCESSOR_RUNNER" \
        "E8_V5_RACE_RETRY_RUNNER_SHA256:$RACE_RETRY_RUNNER" \
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
    --stage-state-review)
        [[ $# -eq 3 && "$2" == "--evidence" ]] ||
            fail 'usage: ratify_and_apply_e8_quality_baseline_v5.sh --stage-state-review --evidence EVIDENCE'
        MODE="stage"
        EVIDENCE="$3"
        EXPECTED_PRE=""
        EXPECTED_CANDIDATE=""
        ;;
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
        fail 'first argument must be --stage-state-review, --attest, or --prevalidate'
        ;;
esac
[[ -z "$TEST_REVIEW_PATH" || "$MODE" != "attest" ]] ||
    fail 'pytest-only state-review path is not permitted for --attest'
[[ "$EVIDENCE" = /* && -f "$EVIDENCE" ]] || fail 'evidence must be an existing absolute path'
if [[ "$MODE" != "stage" ]]; then
    [[ "$EXPECTED_PRE" =~ ^[0-9a-f]{64}$ && "$EXPECTED_CANDIDATE" =~ ^[0-9a-f]{64}$ ]] ||
        fail 'reviewed state hashes must be lowercase SHA-256'
fi
verify_reviewed_bindings
if [[ "$MODE" == "attest" ]]; then
    exec 9>"$LOCK"
    flock -n 9 || fail 'another v5 apply owns the lock'
fi
env -u PYTHONPATH -u PYTHONHOME -u PYTHONSTARTUP PYTHONNOUSERSITE=1 \
    bash "$VALIDATOR" --validate-evidence "$EVIDENCE"

EVIDENCE_SHA256="$(sha "$EVIDENCE")"
TRANSACTION="$ROOT/artifacts/operator/e8_quality_baseline_state_v5_${EVIDENCE_SHA256}.transaction"
ATTESTATION="$ROOT/artifacts/operator/e8_quality_baseline_state_v5_${EVIDENCE_SHA256}.apply_attestation.json"
PROTOCOL_RECEIPT="$ROOT/artifacts/operator/e8_quality_baseline_protocol_v5_${EVIDENCE_SHA256}.ratification.json"
STATE_REVIEW="${TEST_REVIEW_PATH:-$ROOT/artifacts/operator/e8_quality_baseline_state_v5_${EVIDENCE_SHA256}.state_candidate_review.json}"
REVIEWED_ARTIFACTS="$(reviewed_artifact_bindings_json)"

stage_state_review() {
    env -u PYTHONPATH -u PYTHONHOME -u PYTHONSTARTUP PYTHONNOUSERSITE=1 PYTHONOPTIMIZE=0 "$PYTHON" -I - \
        "$APPLIER" "$STATE" "$EVIDENCE" "$VALIDATOR" "$STATE_REVIEW" "$SOURCE_ROOT" \
        "$E8_V5_ORCHESTRATOR_HEAD" "$PYTHON" "$REVIEWED_ARTIFACTS" <<'PY'
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

(
    adapter_raw,
    state_raw,
    evidence_raw,
    validator_raw,
    output_raw,
    source_root_raw,
    source_head,
    interpreter_raw,
    bindings_json,
) = sys.argv[1:]
adapter_path = Path(adapter_raw)
state_path = Path(state_raw)
evidence_path = Path(evidence_raw)
validator_path = Path(validator_raw)
output_path = Path(output_raw)
source_root = Path(source_root_raw)
interpreter = Path(interpreter_raw)
bindings = json.loads(bindings_json)
if not isinstance(bindings, dict) or set(bindings) != {
    "wrapper", "producer", "runner", "resume_runner", "base_runner",
    "recovery_runner", "finalizer_runner", "successor_runner", "race_retry_runner", "validator",
    "validator_python", "applier_adapter", "canonical_applier",
} or not all(isinstance(value, str) and len(value) == 64 for value in bindings.values()):
    raise SystemExit("ERROR: reviewed artifact binding set is malformed")
spec = importlib.util.spec_from_file_location("e8_v5_state_review_adapter", adapter_path)
if spec is None or spec.loader is None:
    raise SystemExit("ERROR: cannot import reviewed E8-v5 applier adapter")
adapter = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = adapter
spec.loader.exec_module(adapter)
canonical = adapter.module
environment = {
    key: value for key, value in os.environ.items()
    if key not in {"PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"}
}
environment["PYTHONNOUSERSITE"] = "1"
environment["PYTHONOPTIMIZE"] = "0"
review = canonical.state_candidate_review_payload(
    state_path,
    evidence_path,
    validator_path,
    lambda: canonical.run_evidence_validator(validator_path, evidence_path, environment),
)
payload = {
    "schema": "epyc.e8_quality_baseline_v5_state_review.v1",
    "source": {"root": str(source_root.resolve()), "head": source_head},
    "interpreter": {
        "path": str(interpreter.resolve()),
        "sha256": hashlib.sha256(interpreter.read_bytes()).hexdigest(),
    },
    "reviewed_artifact_sha256": bindings,
    "state_candidate_review": review,
}
data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
if output_path.exists() or output_path.is_symlink():
    if output_path.is_symlink() or not output_path.is_file():
        raise SystemExit("ERROR: existing state-candidate review must be a regular non-symlink file")
    if output_path.read_bytes() != data:
        raise SystemExit("ERROR: existing state-candidate review differs from current pre-state/evidence")
    raise SystemExit(0)
fd = os.open(output_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
try:
    offset = 0
    while offset < len(data):
        written = os.write(fd, memoryview(data)[offset:])
        if written <= 0 or written > len(data) - offset:
            raise OSError(f"review write made invalid progress: {written}")
        offset += written
    os.fsync(fd)
finally:
    os.close(fd)
directory_fd = os.open(output_path.parent, os.O_RDONLY | os.O_DIRECTORY)
try:
    os.fsync(directory_fd)
finally:
    os.close(directory_fd)
PY
}

validate_state_review() {
    env -u PYTHONPATH -u PYTHONHOME -u PYTHONSTARTUP PYTHONNOUSERSITE=1 PYTHONOPTIMIZE=0 "$PYTHON" -I - \
        "$APPLIER" "$STATE" "$EVIDENCE" "$VALIDATOR" "$STATE_REVIEW" "$SOURCE_ROOT" \
        "$E8_V5_ORCHESTRATOR_HEAD" "$PYTHON" "$REVIEWED_ARTIFACTS" "$EXPECTED_PRE" "$EXPECTED_CANDIDATE" <<'PY'
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

(
    adapter_raw,
    state_raw,
    evidence_raw,
    validator_raw,
    review_raw,
    source_root_raw,
    source_head,
    interpreter_raw,
    bindings_json,
    expected_pre,
    expected_candidate,
) = sys.argv[1:]
adapter_path = Path(adapter_raw)
state_path = Path(state_raw)
evidence_path = Path(evidence_raw)
validator_path = Path(validator_raw)
review_path = Path(review_raw)
source_root = Path(source_root_raw)
interpreter = Path(interpreter_raw)
if review_path.is_symlink() or not review_path.is_file():
    raise SystemExit("ERROR: state-candidate review must be a regular non-symlink file; run --stage-state-review first")
spec = importlib.util.spec_from_file_location("e8_v5_state_review_validator", adapter_path)
if spec is None or spec.loader is None:
    raise SystemExit("ERROR: cannot import reviewed E8-v5 applier adapter")
adapter = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = adapter
spec.loader.exec_module(adapter)
canonical = adapter.module
try:
    stored_bytes = review_path.read_bytes()
    stored = json.loads(stored_bytes)
    expected_bindings = json.loads(bindings_json)
except (OSError, ValueError, json.JSONDecodeError) as exc:
    raise SystemExit(f"ERROR: cannot read state-candidate review: {exc}") from exc
expected_keys = {
    "schema", "source", "interpreter", "reviewed_artifact_sha256", "state_candidate_review"
}
if (
    not isinstance(stored, dict)
    or set(stored) != expected_keys
    or stored.get("schema") != "epyc.e8_quality_baseline_v5_state_review.v1"
    or stored.get("source") != {"root": str(source_root.resolve()), "head": source_head}
    or stored.get("interpreter") != {
        "path": str(interpreter.resolve()),
        "sha256": hashlib.sha256(interpreter.read_bytes()).hexdigest(),
    }
    or stored.get("reviewed_artifact_sha256") != expected_bindings
):
    raise SystemExit("ERROR: state-candidate review binding differs")
environment = {
    key: value for key, value in os.environ.items()
    if key not in {"PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"}
}
environment["PYTHONNOUSERSITE"] = "1"
environment["PYTHONOPTIMIZE"] = "0"
fresh = canonical.state_candidate_review_payload(
    state_path,
    evidence_path,
    validator_path,
    lambda: canonical.run_evidence_validator(validator_path, evidence_path, environment),
)
if stored.get("state_candidate_review") != fresh:
    raise SystemExit("ERROR: state-candidate review differs from a fresh exact recomputation")
if fresh["pre_state_sha256"] != expected_pre:
    raise SystemExit("ERROR: live pre-state differs from the reviewed pre-state")
if fresh["candidate_state_sha256"] != expected_candidate:
    raise SystemExit("ERROR: derived candidate differs from the reviewed candidate")
if len(fresh["exact_state_diff"]) != 6:
    raise SystemExit("ERROR: candidate review is not the exact six-row state diff")
PY
}

verify_state_review_pin() {
    [[ -f "$STATE_REVIEW" && ! -L "$STATE_REVIEW" ]] ||
        fail 'state-candidate review must remain a regular non-symlink file'
    [[ "$(sha "$STATE_REVIEW")" == "$STATE_REVIEW_SHA256" ]] ||
        fail 'state-candidate review changed after validation'
}

if [[ "$MODE" == "stage" ]]; then
    stage_state_review
    printf 'E8 v5 state-candidate review staged: %s\n' "$STATE_REVIEW"
    exit 0
fi

validate_state_review
STATE_REVIEW_SHA256="$(sha "$STATE_REVIEW")"
verify_state_review_pin
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
verify_state_review_pin
E8_BASELINE_APPLY_TOKEN="$TOKEN" env -u PYTHONPATH -u PYTHONHOME -u PYTHONSTARTUP PYTHONNOUSERSITE=1 \
    PYTHONOPTIMIZE=0 "$PYTHON" -I "$APPLIER" "${COMMON[@]}" --validate-only
if [[ "$MODE" == "prevalidate" ]]; then
    printf 'E8 v5 ratify-and-apply prevalidation passed; no state or receipt changed.\n'
    exit 0
fi

verify_reviewed_bindings
verify_state_review_pin
env -u PYTHONPATH -u PYTHONHOME -u PYTHONSTARTUP PYTHONNOUSERSITE=1 PYTHONOPTIMIZE=0 "$PYTHON" -I - \
    "$PROTOCOL_RECEIPT" "$EVIDENCE" "$WRAPPER" "$PRODUCER" "$RUNNER" "$RESUME_RUNNER" "$BASE_RUNNER" \
    "$RECOVERY_RUNNER" "$FINALIZER_RUNNER" "$SUCCESSOR_RUNNER" "$RACE_RETRY_RUNNER" "$VALIDATOR" "$VALIDATOR_PY" "$APPLIER" "$CANONICAL_APPLIER" "$STATE_REVIEW" \
    "$SOURCE_ROOT" "$E8_V5_ORCHESTRATOR_HEAD" "$EXPECTED_PRE" "$EXPECTED_CANDIDATE" "$STATE_REVIEW_SHA256" "$TOKEN" <<'PY'
import hashlib, json, os
from datetime import datetime, timezone
from pathlib import Path
import sys
output, evidence, wrapper, producer, runner, resume_runner, base_runner, recovery_runner, finalizer_runner, successor_runner, race_retry_runner, validator, validator_py, applier, canonical_applier, state_review, source_root = map(
    Path, sys.argv[1:18]
)
source_head, pre, candidate, state_review_sha256, token = sys.argv[18:]
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
    "source_root": str(source_root.resolve()),
    "source_head": source_head,
    "state_candidate_review": str(state_review.resolve()),
    "state_candidate_review_sha256": state_review_sha256,
    "reviewed_artifact_sha256": {
        "wrapper": sha(wrapper),
        "producer": sha(producer),
        "runner": sha(runner),
        "resume_runner": sha(resume_runner),
        "base_runner": sha(base_runner),
        "recovery_runner": sha(recovery_runner),
        "finalizer_runner": sha(finalizer_runner),
        "successor_runner": sha(successor_runner),
        "race_retry_runner": sha(race_retry_runner),
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
verify_state_review_pin
E8_BASELINE_APPLY_TOKEN="$TOKEN" env -u PYTHONPATH -u PYTHONHOME -u PYTHONSTARTUP PYTHONNOUSERSITE=1 \
    PYTHONOPTIMIZE=0 "$PYTHON" -I "$APPLIER" "${COMMON[@]}" --attest "$TOKEN"
