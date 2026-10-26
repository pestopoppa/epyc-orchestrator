#!/bin/bash
# D-RI-PREVIEW-1 filler JOB (queue row s5-dripreview1). Run by filler_run.sh under a grant; it is NOT a launcher.
#   filler_run.sh --id s5-dripreview1 --class cpu --port 9461 --min-window 2400 --stop <STOP> \
#       --cmd 'bash scripts/benchmark/dripreview1_filler_job.sh'
# One PRIVATE llama-server (the served worker GGUF, CPU, on the granted region only) + scripts/benchmark/run_repl_preview.py.
# Correctness-class: the question is whether the model retypes/blind-fits, not how fast it is, so no quiet host is needed.
# Resumable: every sample is appended to OUT/cells/*.jsonl; a window that ends mid-run is continued by the next grant.
# Contract (from filler_run.sh): FILLER_CPUS FILLER_THREADS FILLER_END FILLER_DEADLINE FILLER_STOPFILE FILLER_PIDFILE FILLER_LIB.
# Test seams (env): LLAMA_SERVER, MODEL, PORT, OUT, PY. Spec: artifacts/queue-builders-3-20261021/dripreview1/PREDECLARED.md (root).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_SERVER="${LLAMA_SERVER:-/mnt/raid0/llm/kernels/production/cpu/llama-server}"
MODEL="${MODEL:-/mnt/raid0/llm/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf}"
PORT="${PORT:-9461}"
ROLE="${ROLE:-worker_general}"
OUT="${OUT:-/mnt/raid0/llm/tmp/dripreview1-out}"
PY="${PY:-/workspace/repos/epyc-orchestrator/.venv/bin/python}"
N="${N:-30}"
: "${FILLER_CPUS:?}" "${FILLER_STOPFILE:?}" "${FILLER_DEADLINE:?}" "${FILLER_PIDFILE:?}"
THREADS="${FILLER_THREADS:-8}"
mkdir -p "$OUT"
[ -e "$FILLER_STOPFILE" ] && {
  echo "stop-file"
  exit 0
}
[ -s "$OUT/result.json" ] && python3 -I -c "import json,sys; sys.exit(0 if json.load(open('$OUT/result.json')).get('complete') else 1)" && {
  echo "already complete"
  exit 0
}
[ "$(date -u +%s)" -ge "$FILLER_DEADLINE" ] && {
  echo "deadline: nothing started"
  exit 0
}
[ -n "${FILLER_LIB:-}" ] && . "$FILLER_LIB"

# linkage proof: the store build must resolve its OWN ggml (three-ggml-generations hazard)
BUILD_DIR="$(dirname "$(readlink -f "$LLAMA_SERVER")")"
if [ -z "${SKIP_LINKAGE:-}" ]; then
  LD_LIBRARY_PATH="$BUILD_DIR" bash /workspace/repos/epyc-inference-research/scripts/utils/verify_ggml_linkage.sh "$LLAMA_SERVER" "$BUILD_DIR" >"$OUT/linkage.txt" 2>&1 ||
    {
      echo "linkage FAIL (see $OUT/linkage.txt)"
      exit 5
    }
fi
ss -ltn "sport = :$PORT" | grep -q LISTEN && {
  echo "port $PORT already listening: not ours, refusing"
  exit 4
}

# own process group; the PID is captured and registered, and is the ONLY thing ever signalled
LD_LIBRARY_PATH="$BUILD_DIR" setsid taskset -c "$FILLER_CPUS" nice -n 19 "$LLAMA_SERVER" -m "$MODEL" -t "$THREADS" -np 1 -c 8192 \
  --jinja --host 127.0.0.1 --port "$PORT" --no-webui -ngl 0 >"$OUT/server.log" 2>&1 &
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

if declare -F fl_wait_own_port >/dev/null; then
  fl_wait_own_port "$PORT" "$SPID" 900 || exit $?
else # standalone / test: same proof inline
  for _ in $(seq 1 180); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && ss -ltnp "sport = :$PORT" | grep -q "pid=$SPID," && break
    kill -0 "$SPID" 2>/dev/null || {
      echo "server died"
      exit 6
    }
    sleep 5
  done
  ss -ltnp "sport = :$PORT" | grep -q "pid=$SPID," || {
    echo "own-pid proof failed"
    exit 7
  }
fi

WINDOW_CONFIRMED=1 "$PY" "$HERE/run_repl_preview.py" --role "$ROLE=http://127.0.0.1:$PORT@$SPID" --n "$N" \
  --cell-dir "$OUT/cells" --stop-file "$FILLER_STOPFILE" --deadline-epoch "$FILLER_DEADLINE" --out "$OUT/result.json"
rc=$?
echo "driver rc=$rc"
exit $rc
