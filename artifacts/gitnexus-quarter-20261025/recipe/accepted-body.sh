#!/bin/bash
set -euo pipefail
readonly packet=/mnt/raid0/llm/tmp/codex-ni-20261025-gnx-native-cap-fix/r2
readonly output="$packet/result/attempt-2"
readonly orch="$packet/repos/epyc-orchestrator"
readonly source_snapshot=/mnt/raid0/llm/tmp/codex-ni-20261025-usage-provenance/orch/.gitnexus
readonly base=49c1f749014d5290367413f2898d4e37e2bd1016
stage=initial-checks
[[ ! -e "$output" ]] || { printf 'fresh result path required; no blind retry\n'; exit 65; }
mkdir -p "$output" "$packet/runtime/tmp" "$packet/gnx-home" "$packet/repos"
exec > >(tee -a "$output/run.log") 2>&1
finish() {
 local rc=$?
 jq -n --arg stage "$stage" --argjson rc "$rc" --arg cpus "${FILLER_CPUS:-}" \
  '{stage:$stage,rc:$rc,cpus:$cpus,status:(if $rc==0 then "PASS" else "FAIL" end),index_run:false,installed_package_write:false,published:false}' > "$output/RESULT.json"
 (( rc == 0 )) || touch "$packet/FAILED.flag"
}
trap finish EXIT
: "${FILLER_CPUS:?}" "${FILLER_ID:?}" "${FILLER_DEADLINE:?}"
[[ "$FILLER_ID" == codex-ni-gnx-quarter-guard-20261025 ]]
check_boundary() {
 for hold in /mnt/raid0/llm/tmp/compute-flags/HOLD_*; do [[ ! -e "$hold" ]] || exit 70; done
 [[ ! -e "$packet/STOP" ]]
 (( $(date +%s) + 60 < FILLER_DEADLINE ))
}
check_boundary
cd "$packet"
sha256sum -c PACKET.sha256
sha256sum -c evidence/preserved-v2.sha256
sha256sum -c evidence/preserved-r1-failure.sha256
sha256sum -c evidence/installed-source.sha256
manifest_sha=$(sha256sum PACKET.sha256 | cut -d' ' -f1)
jq -e --arg sha "$manifest_sha" '.approved==true and .packet_sha256==$sha' MAIN-APPROVAL.json > /dev/null
export TMPDIR="$packet/runtime/tmp" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export UV_THREADPOOL_SIZE=1 GITNEXUS_WORKER_POOL_SIZE=4 GITNEXUS_PARSE_CHUNK_CONCURRENCY=1 GITNEXUS_EMBEDDING_THREADS=1
export NODE_OPTIONS=--max-old-space-size=8192 GITNEXUS_LBUG_EXTENSION_INSTALL=never
export GITNEXUS_HOME="$packet/gnx-home"
actual=$(awk '/Cpus_allowed_list/ {print $2}' /proc/self/status)
printf 'grant=%s actual=%s\n' "$FILLER_CPUS" "$actual" > "$output/initial-affinity.txt"
stage=compiler-first-native-resource-control-gate
check_boundary
/bin/bash "$packet/grant-fixtures/build-native.sh" "$packet/proposal/scripts" "$output/build"
stage=granted-native-init-preflight
check_boundary
timeout --signal=TERM --kill-after=5s 120s /usr/bin/python3 -I "$packet/grant-fixtures/run-node.py" \
 "$packet/proposal/scripts" native-db "$output/native-init-preflight" > "$output/native-init-preflight.log" 2>&1
[[ $(rg -c '^GNX_NATIVE_CAP pid=[0-9]+ requested=0 applied=1 observed=1$' "$output/native-init-preflight.log") -ge 2 ]]
jq -e '.status=="PASS" and .buffer_manager_bytes==67108864 and .max_db_bytes==268435456' \
 "$output/native-init-preflight/native-db-proof.json" > /dev/null
stage=canonical-github-private-clone
# Source acquisition is the only network operation; all fixtures are offline.
[[ ! -e "$orch" ]] || { printf 'refuse existing clone; diagnose and review before requeue\n'; exit 65; }
timeout --signal=TERM --kill-after=5s 150s git -c pack.threads=1 -c index.threads=1 clone --no-checkout \
 https://github.com/pestopoppa/epyc-orchestrator.git "$orch"
git -C "$orch" rev-parse origin/main > "$output/origin-main-pin.txt"
# Fresh canonical acquisition is required. Unrelated owner changes are fine;
# any target drift stops instead of silently replacing the reviewed baseline.
git -C "$orch" diff --exit-code "$base" origin/main -- \
 scripts/gitnexus-guard.py scripts/gitnexus-writer.py scripts/gitnexus-analyze.sh \
 scripts/gitnexus-quarter.cjs scripts/gitnexus-quarter-runtime.cjs scripts/gitnexus-lbug-cap.c \
 tests/unit/test_gitnexus_quarter.py tests/unit/test_gitnexus_native_cap.py \
 > "$output/unchanged-targets-vs-origin-main.diff"
git -C "$orch" checkout --detach "$base"
[[ "$(git -C "$orch" remote get-url origin)" == https://github.com/pestopoppa/epyc-orchestrator.git ]]
[[ "$(git -C "$orch" rev-parse HEAD)" == "$base" ]]
[[ -z "$(git -C "$orch" status --porcelain)" ]]
stage=private-graph-snapshot
# Copy only the already-private historical snapshot. Never open/recover/delete
# a canonical shared registry, metadata or index. Its stale status is preserved.
sha256sum "$source_snapshot/meta.json" "$source_snapshot/lbug" > "$output/source-private-snapshot.sha256"
mkdir "$orch/.gitnexus"
for file in "$source_snapshot"/lbug*; do
 [[ -f "$file" && ! -L "$file" ]]
 cp --reflink=auto "$file" "$orch/.gitnexus/"
done
jq --arg path "$orch" '.repoPath=$path' "$source_snapshot/meta.json" > "$orch/.gitnexus/meta.json"
jq -n --arg path "$orch" --slurpfile meta "$orch/.gitnexus/meta.json" \
 '[($meta[0]+{name:"gnx-quarter-guard-private",path:$path,storagePath:($path+"/.gitnexus")})]' > "$GITNEXUS_HOME/registry.json"
stage=first-private-gnx-status-and-scoped-impact
check_boundary
(cd "$orch" && timeout --signal=TERM --kill-after=5s 45s /usr/bin/python3 -I \
 "$packet/grant-fixtures/run-node.py" "$packet/proposal/scripts" query status) > "$output/private-status.txt" 2>&1
impact() {
 local symbol=$1 file=$2 label=$3 rc=0
 check_boundary
 (cd "$orch" && timeout --signal=TERM --kill-after=5s 60s /usr/bin/python3 -I \
  "$packet/grant-fixtures/run-node.py" "$packet/proposal/scripts" query impact "$symbol" \
  --repo gnx-quarter-guard-private --file "$file" --direction upstream --depth 3 --limit 100) \
  > "$output/$label.json" 2> "$output/$label.stderr" || rc=$?
 printf '%s\n' "$rc" > "$output/$label.rc"
}
impact admit scripts/gitnexus-guard.py guard-admit
impact valid_payload scripts/gitnexus-guard.py guard-valid-payload
impact guard scripts/gitnexus-writer.py writer-guard
impact main scripts/gitnexus-writer.py writer-main
/usr/bin/python3 -I "$packet/grant-fixtures/graph-gate.py" "$output" "$packet/MAIN-APPROVAL.json" "$packet/BASE-CALLERS.txt"
sha256sum -c "$output/source-private-snapshot.sha256"
stage=private-proposal-apply
check_boundary
git -C "$orch" apply --check "$packet/PROPOSED.patch"
git -C "$orch" apply "$packet/PROPOSED.patch"
git -C "$orch" diff --check
git -C "$orch" diff --stat > "$output/applied-diff-stat.txt"
cp "$packet/proposal/scripts/gitnexus-lbug-cap.so" "$orch/scripts/"
cp "$packet/proposal/scripts/gitnexus-lbug-cap.manifest.json" "$orch/scripts/"
stage=python-policy-fixtures
timeout --signal=TERM --kill-after=5s 90s /usr/bin/python3 -m unittest discover \
 -s "$orch/tests/unit" -p test_gitnexus_quarter.py -v > "$output/python-fixtures.log" 2>&1
timeout --signal=TERM --kill-after=5s 90s /usr/bin/python3 -m unittest discover \
 -s "$orch/tests/unit" -p test_gitnexus_native_cap.py -v >> "$output/python-fixtures.log" 2>&1
stage=granted-node-syntax
for file in "$orch/scripts/gitnexus-quarter.cjs" "$orch/scripts/gitnexus-quarter-runtime.cjs" \
 "$packet/grant-fixtures/native-db.cjs" "$packet/grant-fixtures/query.cjs" "$packet/grant-fixtures/node-hold.cjs"; do
 check_boundary
 timeout --signal=TERM --kill-after=5s 15s /usr/local/bin/node --v8-pool-size=1 --check "$file"
done
stage=granted-js-resource-bound-controls
check_boundary
timeout --signal=TERM --kill-after=5s 30s /usr/local/bin/node --v8-pool-size=1 \
 "$packet/grant-fixtures/runtime-controls.cjs" "$orch/scripts" > "$output/runtime-controls.log" 2>&1
stage=actual-native-db-cap-and-os-mask
check_boundary
timeout --signal=TERM --kill-after=5s 120s /usr/bin/python3 -I "$packet/grant-fixtures/run-node.py" \
 "$orch/scripts" native-db "$output/native" > "$output/native-db.log" 2>&1
[[ $(rg -c '^GNX_NATIVE_CAP pid=[0-9]+ requested=0 applied=1 observed=1$' "$output/native-db.log") -ge 2 ]]
jq -e '.status=="PASS" and .buffer_manager_bytes==67108864 and .max_db_bytes==268435456' \
 "$output/native/native-db-proof.json" > /dev/null
stage=actual-production-bootstrap-help
check_boundary
(cd "$orch" && timeout --signal=TERM --kill-after=5s 45s /bin/bash scripts/gitnexus-analyze.sh --help) \
 > "$output/actual-bootstrap-help.log" 2>&1
rg -q '"gitnexus_admission": "quarter"' "$output/actual-bootstrap-help.log"
stage=actual-node-supervisor-lifetime
check_boundary
timeout --signal=TERM --kill-after=5s 120s /usr/bin/python3 -I "$packet/grant-fixtures/lifetime.py" \
 "$orch/scripts" "$output/lifetime" > "$output/lifetime.log" 2>&1
jq -e '.status=="PASS" and (.receipts|length)==4 and all(.receipts[];.node_reaped==true and .proc_absent==true and .node_inherited_lock_fd==true and .writer_lock_released==true)' \
 "$output/lifetime/lifetime-proof.json" > /dev/null
stage=source-and-completion-proof
sha256sum -c "$output/source-private-snapshot.sha256"
sha256sum -c "$output/build/artifacts.sha256"
sha256sum -c evidence/preserved-v2.sha256
sha256sum -c evidence/preserved-r1-failure.sha256
git -C "$orch" diff --check
[[ ! -e "$orch/.gitnexus/parse-cache" ]]
stage=complete-native-resource-control-fixtures-only
touch "$packet/COMPLETE"
