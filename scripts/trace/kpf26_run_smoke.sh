#!/bin/bash
# Run only inside the exact active owner-issued filler grant bound in the packet.
set -euo pipefail
if [ "$#" -ne 1 ]; then
    echo "usage: $0 PACKET_DIRECTORY" >&2
    exit 2
fi
PACKET=$(realpath "$1")
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${FILLER_ID:?requires the owner-issued filler launcher}"
: "${FILLER_END:?requires owner hard end}"
: "${FILLER_DEADLINE:?requires owner deadline}"
: "${FILLER_STOPFILE:?requires owner stop file}"
[ ! -e "$FILLER_STOPFILE" ] || { echo "stop file active" >&2; exit 1; }
exec python3 "$HERE/kpf26_paired_replay.py" --manifest "$PACKET/manifest.json" \
    --executor "$HERE/kpf26_scout_executor.py" --output "$PACKET/result"
