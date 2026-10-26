#!/bin/bash
set -euo pipefail
readonly scripts=${1:?owned scripts directory}
readonly evidence=${2:?fresh build evidence directory}
: "${FILLER_CPUS:?}" "${FILLER_ID:?}" "${FILLER_DEADLINE:?}"
[[ "$FILLER_ID" == codex-ni-gnx-quarter-guard-20261025 ]]
for hold in /mnt/raid0/llm/tmp/compute-flags/HOLD_*; do [[ ! -e "$hold" ]] || exit 70; done
(( $(date +%s) + 120 < FILLER_DEADLINE ))
[[ ! -e "$scripts/gitnexus-lbug-cap.so" && ! -e "$scripts/gitnexus-lbug-cap.manifest.json" ]]
mkdir -p "$evidence"
/usr/bin/python3 -I - "$scripts" <<'PY'
import importlib.util, os, pathlib, sys
scripts = pathlib.Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("guard", scripts / "gitnexus-guard.py")
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)
claim = guard.claim()
if claim["region"] is None or frozenset(os.sched_getaffinity(0)) != claim["cpus"]:
    raise RuntimeError("build requires the exact single-quarter canonical claim")
guard.installed_source()
import hashlib
if hashlib.sha256((scripts / "gitnexus-lbug-cap.c").read_bytes()).hexdigest() != guard.NATIVE_SOURCE_SHA:
    raise RuntimeError("reviewed shim source changed")
PY
cc --version > "$evidence/compiler.txt"
printf 'compiler_first_gate=begin source=%s output=%s flags=-std=c11,-O2,-fPIC,-shared,-Wall,-Wextra,-Werror,--no-undefined,-z,relro,-z,now,-ldl wall_cap=60\n' \
 "$scripts/gitnexus-lbug-cap.c" "$scripts/gitnexus-lbug-cap.so" > "$evidence/compiler-command.txt"
cat "$evidence/compiler-command.txt"
compile_rc=0
timeout --signal=TERM --kill-after=5s 60s cc -std=c11 -O2 -fPIC -shared \
 -Wall -Wextra -Werror -Wl,--no-undefined,-z,relro,-z,now \
 "$scripts/gitnexus-lbug-cap.c" -o "$scripts/gitnexus-lbug-cap.so" -ldl \
 > "$evidence/compiler.log" 2>&1 || compile_rc=$?
printf '%s\n' "$compile_rc" > "$evidence/compiler.rc"
cat "$evidence/compiler.log"
printf 'compiler_first_gate_rc=%s\n' "$compile_rc"
(( compile_rc == 0 )) || exit "$compile_rc"
nm -D --defined-only "$scripts/gitnexus-lbug-cap.so" > "$evidence/exported-symbols.txt"
awk '{print $3}' "$evidence/exported-symbols.txt" | sort > "$evidence/exported-names.txt"
printf '%s\n' _ZN4lbug4main12SystemConfigC1Emmbbmbmbbbbb _ZN4lbug4main12SystemConfigC2Emmbbmbmbbbbb \
 > "$evidence/expected-names.txt"
diff -u "$evidence/expected-names.txt" "$evidence/exported-names.txt"
/usr/bin/python3 -I - "$scripts" <<'PY'
import hashlib, importlib.util, json, pathlib, sys
scripts = pathlib.Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("guard", scripts / "gitnexus-guard.py")
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)
manifest = {"schema_version": 1, "source_sha256": guard.NATIVE_SOURCE_SHA,
            "native_sha256": guard.NATIVE_ELF_SHA, "max_num_threads": 1,
            "shim_sha256": hashlib.sha256((scripts / "gitnexus-lbug-cap.so").read_bytes()).hexdigest()}
(scripts / "gitnexus-lbug-cap.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
guard.native_artifact()
PY
sha256sum "$scripts/gitnexus-lbug-cap.so" "$scripts/gitnexus-lbug-cap.manifest.json" > "$evidence/artifacts.sha256"
