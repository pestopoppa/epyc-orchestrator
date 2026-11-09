#!/bin/bash
# TD-16 filler JOB (row td16-label-word-bias-*). Run by filler_run15.sh under a grant; it is NOT a launcher.
# One PRIVATE champion-sidecar llama-server (operator ruling 6, routing-program.md "GATE: champion-sidecar (CPU window)":
# champion CPU build kernels/builds/cpu-20260925-90c12df42, frontdoor's model and MTP recipe, spare port 8199, granted
# CPUs only) + scripts/benchmark/td16_label_word_bias.py. Correctness/calibration class: no timing claim.
# Resumable: the driver appends one unit per (arm, batch) to RESULTS_JSONL and skips units already there.
# Build identity by CONTENT (not path): sha256 of the resolved llama-server AND a version string naming the commit, both
# checked before launch; plus verify_ggml_linkage.sh (own ggml). Test seams (env): LLAMA_SERVER MODEL PORT OUT PY SKIP_LINKAGE.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_SERVER="${LLAMA_SERVER:-/mnt/raid0/llm/kernels/builds/cpu-20260925-90c12df42/bin/llama-server}"
EXPECT_SHA="${EXPECT_SHA:-302df9d744215ef4accb9ee4b7fff45ca2dfca835abed39568b2e86292cc0b92}"
EXPECT_VERSION="${EXPECT_VERSION:-10308 (90c12df42)}"
MODEL="${MODEL:-/mnt/raid0/llm/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf}"
PORT="${PORT:-8199}"
ROLE="${ROLE:-worker_general}"
OUT="${OUT:?OUT (fresh output dir) is required}"
RESULTS_JSONL="${RESULTS_JSONL:-$OUT/results.jsonl}"
PY="${PY:-/workspace/repos/epyc-orchestrator/.venv/bin/python}"
FIXTURE="${FIXTURE:-/mnt/raid0/llm/epyc-root/research/fixtures/td16-hunch-20261102/td16-lookalikes.jsonl}"
: "${FILLER_CPUS:?}" "${FILLER_STOPFILE:?}" "${FILLER_DEADLINE:?}" "${FILLER_PIDFILE:?}"
THREADS="${FILLER_THREADS:-8}"
mkdir -p "$OUT"
[ -e "$FILLER_STOPFILE" ] && {
  echo "stop-file"
  exit 0
}
[ -s "$OUT/RESULT.json" ] && python3 -I -c "import json,sys; d=json.load(open('$OUT/RESULT.json')); sys.exit(0 if d.get('complete') and d.get('valid') is True else 1)" && {
  echo "already complete"
  exit 0
}
[ "$(date -u +%s)" -ge "$FILLER_DEADLINE" ] && {
  echo "deadline: nothing started"
  exit 0
}
[ -n "${FILLER_LIB:-}" ] && . "$FILLER_LIB"

REAL_BIN="$(readlink -f "$LLAMA_SERVER")"
BUILD_DIR="$(dirname "$REAL_BIN")"
if [ -z "${SKIP_LINKAGE:-}" ]; then
  GOT_SHA="$(sha256sum "$REAL_BIN" | cut -d' ' -f1)"
  [ "$GOT_SHA" = "$EXPECT_SHA" ] || {
    echo "binary content mismatch: $REAL_BIN sha256 $GOT_SHA != $EXPECT_SHA"
    exit 5
  }
  VER="$(env -u LD_LIBRARY_PATH LD_LIBRARY_PATH="$BUILD_DIR" "$REAL_BIN" --version 2>&1 | head -n 3)"
  case "$VER" in *"$EXPECT_VERSION"*) ;; *)
    echo "version string lacks '$EXPECT_VERSION': $VER"
    exit 5
    ;;
  esac
  printf '{"sha256": "%s", "real_path": "%s", "version": "%s"}\n' "$GOT_SHA" "$REAL_BIN" "$(echo "$VER" | tr '\n' ' ')" >"$OUT/binary-proof.json"
  LD_LIBRARY_PATH="$BUILD_DIR" bash /workspace/repos/epyc-inference-research/scripts/utils/verify_ggml_linkage.sh "$REAL_BIN" "$BUILD_DIR" >"$OUT/linkage.txt" 2>&1 ||
    {
      echo "linkage FAIL (see $OUT/linkage.txt)"
      exit 5
    }
fi
/usr/bin/python3 -I "$HERE/own_port_proof.py" listening "$PORT" && {
  echo "port $PORT already listening: not ours, refusing"
  exit 4
}

# own process group; the PID is captured and registered, and is the ONLY thing ever signalled
LD_LIBRARY_PATH="$BUILD_DIR" setsid taskset -c "$FILLER_CPUS" nice -n 19 "$REAL_BIN" -m "$MODEL" -t "$THREADS" -np 1 -c 8192 \
  --jinja --host 127.0.0.1 --port "$PORT" --no-webui -ngl 0 --spec-type draft-mtp --spec-draft-n-max 4 >"$OUT/server.log" 2>&1 &
SPID=$!
echo "$SPID" >>"$FILLER_PIDFILE"
cleanup() {
  kill -TERM -- "-$SPID" 2>/dev/null || kill -TERM "$SPID" 2>/dev/null
  for _ in $(seq 1 20); do
    kill -0 "$SPID" 2>/dev/null || break
    sleep 1
  done
  kill -0 "$SPID" 2>/dev/null && {
    kill -KILL -- "-$SPID" 2>/dev/null || kill -KILL "$SPID" 2>/dev/null
    sleep 1
  }
  if kill -0 "$SPID" 2>/dev/null; then echo "server $SPID STILL ALIVE"; else echo "server $SPID dead"; fi
}
trap cleanup EXIT

# own-pid proof WITHOUT ss (the keeper's job env has none; fl_wait_own_port would wait forever): /health answers AND the LISTEN
# socket on PORT is in OUR server pid's fd table (own_port_proof.py reads /proc)
for _ in $(seq 1 180); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && /usr/bin/python3 -I "$HERE/own_port_proof.py" owner "$PORT" "$SPID" && break
  kill -0 "$SPID" 2>/dev/null || {
    echo "server died"
    exit 6
  }
  sleep 5
done
/usr/bin/python3 -I "$HERE/own_port_proof.py" owner "$PORT" "$SPID" || {
  echo "own-pid proof failed"
  exit 7
}

WINDOW_CONFIRMED=1 "$PY" "$HERE/td16_label_word_bias.py" --fixture "$FIXTURE" --role "$ROLE" \
  --server-url "$ROLE=http://127.0.0.1:$PORT" --results "$RESULTS_JSONL" --out "$OUT/RESULT.json" \
  --stop-file "$FILLER_STOPFILE" --deadline-epoch "$FILLER_DEADLINE"
rc=$?
echo "driver rc=$rc (0 complete+valid, 3 voided arm: driver/prompt failure not a model result, 75 partial)"
exit $rc
