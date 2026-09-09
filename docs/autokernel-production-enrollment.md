# AutoKernel production enrollment export

`scripts.server.autokernel_enrollment` is an offline description of the current
production launcher. It calls the real `build_server_command` with
`prepare_runtime_dirs=False`; it does not start a server, create a slot directory,
pre-evict memory, acquire a grant, or read the live process environment.

The context factory pins the exact master/lean registries, launch manifest, topology,
compiled priors and launcher/environment/runtime sources already loaded by the process.
Export refuses source drift, stale master-to-lean compilation, and stale compiled-prior
dependencies. Version 1 consequently supports only the exact current context, not an
arbitrary historical source tree.

The factory records the supplied `--revision` as `caller_declared`; it is not a Git
verification. Disk-byte SHA-256 pins and a separate digest of the callable code actually
loaded by Python are both retained, so a revision label cannot substitute for either.
Export also re-loads the pinned configuration and compares it with the tables already
loaded in Python; changing a manifest after import is a refusal, not mixed provenance.

```bash
ORCHESTRATOR_STACK_REEXEC=1 PYTHONPATH=. \
  /mnt/raid0/llm/epyc-orchestrator/.venv/bin/python \
  -m scripts.server.autokernel_enrollment \
  --master-registry /path/to/epyc-inference-research/orchestration/model_registry.yaml \
  --revision <exact-research-revision> --instance-mode both \
  --out /tmp/production-enrollment.json
```

The empty role list means the complete live production fleet. `--role` adds an explicit
launcher-only tenant and `--seed-role` adds an optional seed; neither can remove or
shadow production obligations. Whisper and Qwen TTS obligations are retained with
`speech_instrument_unsupported` because this serving instrument cannot measure them.

Artifact pins are an optional JSON array of `{use,path,sha256}` objects. The exporter
does not hash model files, binaries, or DSOs. Missing pins yield `waiting_artifact` per
target. Pins are declarations by default; `--verify-artifacts` explicitly hashes each
unique pinned path once for this export and records per-target failures, but still grants
no admission authority. Runtime
environment input is a closed loader/OMP/GGML allowlist; unrelated parent values and
credentials are refused rather than serialized.

`--out` is the enrollable form: it creates a private, create-only
`production-enrollment.json.recipes/` bundle containing one canonical JSON launch recipe
per distinct target, then records each file's actual byte SHA-256 in the export. Files and
directories are fsynced, exact retries recover, and a conflicting existing file, symlink,
ownership/mode, or tampered sidecar refuses. Stdout remains a noncreating diagnostic; a
dry consumer will not synthesize recipe identity from the enclosing export's digest.

Each row separates `command_argv` (the canonical builder output) from `argv` (declared
NUMA prefix plus command), and records aliases, obligations, environment/unsets,
artifact identities, workload shape, runtime requirements and the unperformed
pre-eviction requirement. The export is lifecycle intent only; placement, residency and
contention remain unproved until an authorized run captures them.
