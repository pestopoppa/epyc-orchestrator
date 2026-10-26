# GitNexus quarter resource-control proof — 2026-10-25

MAIN accepted the R2 isolated proof, `RESULT.json` rc0 on exact CPUs72-95, at
`/mnt/raid0/llm/tmp/codex-ni-20261025-gnx-native-cap-fix/r2/result/attempt-2/`.
This is verified resource-boundary/dependency evidence. No analyze/re-index,
USAGE validation, installed-package mutation, deployment or publication was
performed by that proof job. The source adoption and real fresh index remain
separate owner actions.

The installed Ladybug Node binding supplied the default startup-thread value0;
the pinned native SystemConfig constructor expanded it through
hardware_concurrency/get_nprocs and built its initialization pool before a
Connection existed. A Connection query cap alone was insufficient. The ELF
call chain and exact Linux GLOBAL loader/source fingerprints establish that
mechanism; an earlier failed run did not record a numeric actual task count,
so this finding does not invent one for it.

The accepted shim forwards every constructor argument except maxNumThreads=1.
Its C-source digest is
`4e64dcd9aa5ff1b2dff6b7dfb0cc9a79641e0333593c96792872fbc8c03ab29b`.
The exact accepted seven-file source/test closure is preserved in
`artifacts/gitnexus-quarter-20261025/ACCEPTED-SOURCE.sha256`.

Actual native checks observed SQL current_setting('threads')=1 after async/sync
initialization and an oversized public setter. Each DB constructor emitted
requested0/applied1/observed1. Native DB tasks peaked at6 in the recorded
samples; every recorded task mask was72-95. DB resources were64MiB buffer and
256MiB maximum file. External supervisor samples retained the same canonical
grant binding. Separate TERM/INT/HUP/SIGKILL cases reaped the captured Node,
proved its inherited writer-lock FD and proved release after exit.

The applied-source Python policy controls passed29 quarter tests and11 native
artifact/mapping tests. The earlier allowed preflight additionally passed3
graph-gate controls. The private graph read found no HIGH/CRITICAL result, but
its historical snapshot was stale. MAIN approved the exact static caller
closure54df21f6 only for private guard/native fixtures; this is not a fresh
application-wide graph verdict.

Two failures remain preserved. The V2 native run failed the real task-count
gate at database init. The first native-shim attempt failed compilation:
discarding write's warn_unused_result triggered -Werror. R2 checked its ssize_t
result while retaining unconditional exit64 and every warning/resource gate.
The proof ran a compiler-first gate, native-init preflight, then the remaining
private graph/apply/bootstrap/native/lifetime phases. Nothing was silently
retried or relabeled PASS.

Selected original receipts, logs, source fingerprints, caller closure and
artifact manifest are under `artifacts/gitnexus-quarter-20261025/`; no generated
binary is committed there. These are point-in-time fixture observations,
not throughput, inference noninterference, full-index memory or full-index
thread-count measurements. Publication of this finding does not make the
planning loop wired; committed-conclusion capture and the existing dependency
evidence contract govern that separately.

The published `exported-symbols.txt` is a derived symbol-table carrier: its
address column uses explicit hexadecimal integers with a `0x` prefix and no
zero padding. Numeric addresses, symbol types, symbol names and row order are
identical to the original `nm -D --defined-only` output; this text is not a raw
tool receipt. The original raw file remains privately preserved at
`/mnt/raid0/llm/tmp/codex-ni-20261025-gnx-native-cap-fix/publication/gates-r2/payload/artifacts/gitnexus-quarter-20261025/exported-symbols.txt`,
SHA-256 `e00ae8481215cd1de6a34041ef868dc137d232ee894dfda03e003308626bf283`.
The publication carrier changes no executable, native manifest, accepted source
closure or original validation receipt. Original validation packet hashes
continue to identify their original private files. The explicit numeric type
also avoids treating zero-padded ELF addresses as untyped account numbers;
the normal publication hook remains unchanged and mandatory.
