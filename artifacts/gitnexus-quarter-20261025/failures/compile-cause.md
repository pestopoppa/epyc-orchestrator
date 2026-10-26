# Exact cause of the 2026-10-25 native-cap attempt-1 failure

The official one-quarter grant started **18:27:31Z on CPUs48-71** and ended
**18:27:34Z, rc1**. The owner summarized this as about18:28Z.
`../result/attempt-1/RESULT.json` identifies stage
`grant-only-private-native-resource-control-build`.

Failing command, from `../grant-fixtures/build-native.sh`:

```sh
timeout --signal=TERM --kill-after=5s 60s cc -std=c11 -O2 -fPIC -shared \
 -Wall -Wextra -Werror -Wl,--no-undefined,-z,relro,-z,now \
 /mnt/raid0/llm/tmp/codex-ni-20261025-gnx-native-cap-fix/proposal/scripts/gitnexus-lbug-cap.c \
 -o /mnt/raid0/llm/tmp/codex-ni-20261025-gnx-native-cap-fix/proposal/scripts/gitnexus-lbug-cap.so -ldl
```

Exact diagnostic in `../result/attempt-1/run.log`:

```text
gitnexus-lbug-cap.c:26:11: error: ignoring return value of 'write' declared with attribute 'warn_unused_result' [-Werror=unused-result]
   26 |     (void)write(STDERR_FILENO, message, sizeof(message) - 1);
cc1: all warnings being treated as errors
```

The recorded compiler is Ubuntu GCC15.2.0. The actual compilation diagnosed
`write` as requiring its result to be used; casting the call to void did not
satisfy that contract. The local `/usr/include/unistd.h` declaration carries
`__wur`; `sys/cdefs.h` defines it as `warn_unused_result` when fortification is
enabled. No extra assertion about which default enabled fortification is needed
to establish this failure. This was the agent-authored refusal diagnostic path.

No `.so` or build manifest was produced. `result/attempt-1/build/` contains only
`compiler.txt`; `repos/` and `gnx-home/` are empty. The clone, private graph,
Node/native DB init/query, bootstrap and lifetime phases were never reached.
This is a compile-time defect, not an observed loader or native-init failure.
There is no compiler/native PASS to report.

The separate R2 candidate assigns the `ssize_t` return and checks for a full
write. A short/failed diagnostic still calls `_exit(64)` immediately; a full
write reaches the unconditional `_exit(64)` below. All warning flags, native
source/ABI/caller identity, constructor1 cap, exact CPU/mapping checks, DB bounds
and actual<=24-task gate remain intact. The guard's C-source SHA must change to
the corrected source SHA. Original packet99aa30e2/source00827265 and actual FAIL
artifacts are preserved and fingerprinted by `evidence/preserved-r1-failure.sha256`.

HOLD_GPU_FOR_QUIET_20261025 is active. This receipt and candidate are text-only:
no preflight, compiler, Node/GNX/native execution, source apply, retry or arming
occurred during this diagnosis. MAIN's relay attempt exited75 under HOLD; no
owner acknowledgement is claimed here.
