# GitNexus on one CPU quarter

The orchestrator's sanctioned entry point remains `scripts/gitnexus-analyze.sh`.
The whole-host path retains its existing admission contract. The APP quarter
path requires one live official build-role grant for exactly24 primary CPUs,
the matching GLOBAL/build ancestor flocks and single-region payload, the actual
Node executable, and every observed Node task pinned to exactly those CPUs.
ROOT's index still uses the whole-host path.

Quarter startup fixes the V8 pool and libuv pool at1, parse workers at4, parse
chunk concurrency at1, and embedding threads at1. The writer owns Node's
startup environment and the CLI worker argument, refuses secondary embedding
or skills workloads, and supervises the captured child. Actual task count must
remain<=24. Termination, claim/HOLD changes or a placement mismatch reap that
captured child; the inherited writer flock remains live until it exits.

## Required native artifact

Ladybug0.17.1's public Node Database constructor exposes memory/file bounds but
no startup-thread argument. Its native default initializes a hardware-sized
pool before a Connection's query cap can apply. The quarter-only preload
interposes the pinned SystemConfig C1/C2 constructor ABI and forwards every
argument unchanged except maxNumThreads=1. It checks the actual native/caller
image and reads back the original constructor's config field. The pinned Linux
loader uses RTLD_LAZY|RTLD_GLOBAL. The installed npm package is never patched.

Only the captured quarter Node receives the preload. Bootstrap verifies its
mapped inode before importing the CLI and strips preload propagation to Python
and later subprocesses. Missing artifacts, changed C/native/source pins or an
unexpected mapping refuse. A supported query cap1 also applies to Connection's
async/sync initialization and public setter.

The required `.so` and runtime manifest are generated files, intentionally not
committed. Reproduce them **inside a new official canonical build grant**, with
the whole body pinned to its exact primary quarter and no HOLD:

```bash
bash scripts/gitnexus-build-native-cap.sh \
  "$(pwd)/scripts" /absolute/owned/evidence-directory
```

The official launcher supplies FILLER_ID, FILLER_CPUS and FILLER_DEADLINE. The
builder verifies the actual canonical grant/claim/mask, installed source/native
pins and C-source digest before invoking the single-thread compiler. It refuses
existing output artifacts. Preserve old artifacts before any separately
reviewed replacement; never overwrite an active artifact or change native pins
to get past a refusal.

The accepted source/C/flags/toolchain produced:

| Artifact | SHA-256 |
| --- | --- |
| scripts/gitnexus-lbug-cap.so | 194f0e3ac638ffb70000faec959ced1ce5d152c1d9240567b7f5d0ba5416e2e4 |
| scripts/gitnexus-lbug-cap.manifest.json | f01a1541005a55941147eb2b5f80840454a0dde78d1c04e12abd06829282bfff |
| Native Ladybug lbugjs.node | 85b101251f07b052927443a994c69c23621f7c10159687145fa15b06decf7b49 |

A different rebuild digest requires review. This acceptance does not transfer
to an upgraded native package or a changed C shim. Never commit a generated
binary as a substitute for its source, manifest or actual native proof.

## Memory and adoption

The quarter writer explicitly supplies512MiB DB buffer and8GiB maximum DB file,
plus an8GiB V8 heap ceiling. GitNexus's installed factory otherwise passes
buffer0, selecting a host-derived default. The owned Database wrapper supplies
the reviewed nonzero bounds and rejects larger requests. The native fixture
uses a separate64MiB buffer/256MiB maximum file; historical read-only graph
queries use64MiB/2GiB. Those fixtures do not establish a full-index memory peak.

Owner adoption is separate from source publication:

1. Review and integrate the explicit codex source commit into the owner checkout.
2. In an official grant, generate the artifact/manifest using this builder, or
   copy the preserved accepted pair after checking source/native/artifact pins.
   Only the owner writes the shared checkout. Keep artifacts untracked.
3. Verify the generated pair and source/native SHA pins. The quarter path cannot
   run until both artifacts exist and pass admission; whole-host startup does
   not require them.
4. Arm a separate granted fresh re-index through the wrapper. Preserve its
   actual placement/count and index-freshness receipt. A refused quarter never
   becomes an implicit whole-host run.
5. Only a real fresh index supports the subsequent usage-provenance impact and
   caller coverage review. Actual HIGH/CRITICAL stops for the operator's review;
   source publication and native fixture acceptance do not authorize USAGE
   execution, merge, API reload or deployment.

The accepted isolated evidence and its limits are in
[the native proof finding](../reference/gitnexus-quarter-native-proof-20261025.md).
