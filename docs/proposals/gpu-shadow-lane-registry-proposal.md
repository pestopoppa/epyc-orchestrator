# PROPOSAL — GPU shadow lane registry block + launch wiring (NOT APPLIED)

**Filed**: 2026-07-28 (gpu-serving-tie-in-program P0-7).
**Status**: PROPOSAL ONLY. The registry is **FROZEN** (program decision D3) and
none of the blocks below are applied anywhere. This file exists so that
activation later is *copy-apply + three-gates*, exactly the way
`epyc-root/docs/runbooks/vision-escalation-minicpmo-promotion.md` Step 1
pre-writes its State-B literals. The choreography that consumes this file is
`docs/gpu-shadow-lane.md` §7.

**No live edits authorized by this file**: not to
`epyc-inference-research/orchestration/model_registry.yaml` (master), not to
the compiled lean, not to `stack_manifest.py` / `stack_numa.py` /
`orchestrator_stack.py`.

---

## 0. Naming decisions (rationale)

| Name | What | Why |
|---|---|---|
| `gpu_shadow_lane` | **Launcher role** (process identity: `ROLE_LAUNCH_META` + `NUMA_CONFIG` + `PORT_MAP` keys) | The lane is role-agnostic — the slot outlives any tenant. Mirrors the `eval_batch_frontdoor` precedent: a `launcher_only` warm entry that borrows its model priors from a *source role* and is never started by a normal `start`. A stable lane name means tenant/duty swaps never rename the process identity. |
| `coder_escalation_shadow` | **Registry role** (master-registry block = the TENANT as data) | Registry roles are duty-named (`frontdoor`, `coder_escalation`, `vision_escalation`), never model-named. The first tenancy candidacy is coder-escalation-shaped (P3-1 bake-off; a P3-3 win rebinds `coder_escalation` itself). `_shadow` states the D3 invariant in the name and sorts adjacent to `coder_escalation`. Rejected alternatives: `gpu_coder_lane` (device-in-role-name breaks convention; devices live under `server:`), `coder_escalation_27b` (model-in-role-name violates the model-not-role rule). |
| Port `18100` | Lane TCP port | The explicit-only warm-lane high-port convention (precedent: `eval_batch_frontdoor` = 18070). 18100 is clear of 18070 and of the operator-owned external 18072, and outside every production 8xxx range. |
| `ORCHESTRATOR_FEATURE_GPU_SHADOW_LANE` | Feature flag (`gpu_shadow_lane` in `src/features.py`) | Already landed, default-off in test AND prod (`eval_batch_serving` pattern). Gates the np_ceiling loader + any future routing consumption; launching stays explicit-only regardless. |

Tenant swap (stock-27B → FF after the D7/P3 bake-off, identical 28.7 GB
footprint) = replace the `model:` block in §1 below (+ Step 1–6 choreography).
Duty swap = new registry role block + repoint the single
`GPU_SHADOW_LANE_TENANT_ROLE` constant (§2).

---

## 1. Master-registry role block (ready to apply)

Target: `/mnt/raid0/llm/epyc-inference-research/orchestration/model_registry.yaml`,
`roles:` section. Insert the block **after `coder_escalation_q8` and before
`ingest_long_context`** (adjacent to the coder_escalation family; anchor
textually — line numbers drift). The master is the ONLY registry you
hand-edit; the lean is auto-compiled at every `orchestrator_stack.py start`
(hand-editing the lean is the 2026-07-18 clobber-exposed failure the vision
runbook documents).

```yaml
  coder_escalation_shadow:
    tier: B
    port: 18100
    description: "GPU shadow lane tenant - stock Qwen3.6-27B dense-hybrid Q8 resident on MI210; shadow-only (gpu-serving-tie-in D3), no production traffic until P3-3 three-gates"
    model:
      name: Qwen3.6-27B
      path: /mnt/raid0/llm/models/Qwen_Qwen3.6-27B-Q8_0.gguf
      quant: Q8_0
      size_gb: 28.7
      architecture: qwen35  # dense hybrid SSM (gated delta net + attention every 4th layer)
      ctx_max: 262144
      sha256: 5927dc06c2b19f732fb6e2a6546dff4c130b552f2ab5f91feb3daafe43897b2a
      use_chat_api: true
      disable_thinking: true
      reasoning: auto
      kv_cache:
        type_k: f16
        type_v: f16  # deliberate deviation from the qwen36_27b_q8 catalogue row (q8_0):
                     # f16 KV is the measured np_context_study_v8 lane shape and the basis
                     # of the np_ceiling arithmetic in
                     # epyc-orchestrator/orchestration/gpu_shadow_lane_np_ceiling.yaml
    candidate_roles:
    - coder
    acceleration:
      type: none  # D6: MTP is launch-bound on v8; default OFF for the escalation-dominant
                  # duty mix (measured net-negative single-stream deep-context). Mode
                  # toggle = drained-lane relaunch, never a per-request option.
      disallowed:
      - speculative_decoding
      - prompt_lookup
    performance:
      observation_grade_only: true
      np_context_grid: /mnt/raid0/llm/epyc-inference-research/artifacts/np_context_study_v8_20260727
      evidence_arm: A3_ff_fable_non_mtp_q8  # FF merge of the same dense-27B architecture
      single_stream_decode_tok_s: 26.5  # np1/L2048; aggregate saturates 102.0 t/s at np16/L2048
      np_ceiling_policy: epyc-orchestrator/orchestration/gpu_shadow_lane_np_ceiling.yaml
    memory:
      residency: warm  # explicit-only launch; never auto-started by a normal stack start
      pinned: false
    server:
      endpoint: "http://localhost:18100"
      api_format: openai
      device: ROCm0
      reasoning: 'off'
      runtime_requirements:
        binary_dir: /mnt/raid0/llm/llama.cpp/build-hip/bin
        ld_library_path:
        - /mnt/raid0/llm/llama.cpp/build-hip/bin
    notes: |
      <ACTIVATION-DATE> gpu-serving-tie-in Phase-2 lane tenant (proposal:
      epyc-orchestrator/docs/proposals/gpu-shadow-lane-registry-proposal.md;
      choreography: epyc-orchestrator/docs/gpu-shadow-lane.md). Shadow-only per
      D3: coder_escalation stays A4-bound; this role receives eval-path /
      forced-role traffic only until the P3-3 three-gates sign-off. Default
      serving shape -np 8 x 8192-token slots (phase2_resident_set np_ceiling
      row); -np/-c changes must stay within the np_ceiling policy table.
      Host threads: SMT siblings 184-191, -t 8 -tb 8 (GPU host-thread rule).
      Rollback is State A of the lane choreography (role removal + lane stop).
```

**What you deliberately do NOT add** (mirrors the vision runbook's note):
- no `server_mode` row — the lane binds via the launcher (`stack_manifest`
  role), like `vision_escalation` (`binding: stack_manifest.role` in compiled
  priors);
- no `kmp_blocktime` / `env_policy` fields — the priors compiler derives
  `env_policy: binary_override_strip_ggml` + `kmp_blocktime: 10` automatically
  whenever `binary_dir` is set (`src/registry/stack_priors.py`);
- no `process_layout.hot_resident` entry — the lane is warm/explicit-only;
- no edits to the existing `qwen36_27b_q8` catalogue entry beyond (optionally)
  a dated activation observation appended to its history (append-only
  discipline).

## 2. `stack_manifest.py` launch entries (what the activation diff ACTUALLY adds)

> **2026-07-28 P2-6 punch-list amendment:** the lane constants
> (`GPU_SHADOW_LANE_TENANT_ROLE` / `_DEVICE` / `_REASONING` / fallback shape)
> and the gated `gpu_shadow_lane` mode branch in
> `_build_servers_from_classification` are **ALREADY LANDED, inert**, in
> `stack_manifest.py` — as are the mode dispatch in
> `stack_commands.py` (cmd_start/reload), the builder + `start_server` branch
> in `orchestrator_stack.py`, the prewarm exclusion in `stack_prewarm.py`, and
> the `tenant_role` compile contract (registry_compiler / model-descriptor /
> stack-priors path, P0-1). They fire only for an entry carrying the
> `gpu_shadow_lane` mode / `tenant_role` key, and no entry exists until this
> proposal is applied (witnesses:
> `tests/unit/test_gpu_shadow_lane_compile_contract.py`,
> `tests/unit/test_gpu_shadow_lane.py::test_orchestrator_stack_has_no_lane_coupling`).
> The activation diff below is therefore SMALLER than the original P0-7
> version of this file described: two dict entries + one port here, one
> NUMA_CONFIG entry in `stack_numa.py`, nothing else.

Classification/wiring entries to apply (two dicts + one port, same file):

```python
# PORT_MAP addition:
    "gpu_shadow_lane": 18100,  # explicit-only GPU shadow lane (MI210); shadow-only per D3

# ROLE_LAUNCH_META addition (WARM tier — never started by a normal `start`):
    # gpu-serving-tie-in P2: role-agnostic MI210 shadow lane. launcher_only keeps
    # the process identity out of lean-registry/descriptor compile inputs; the
    # TENANT is the registry role named by tenant_role, which the P0-1 compile
    # contract carries through lean/descriptor/stack-prior compilation as a
    # `deployment_status: launcher_tenant` record (NEVER live_stack — routing
    # and lock consumers do not see it; only an explicit
    # `start --only gpu_shadow_lane` resolves it).
    "gpu_shadow_lane": {
        "tier": "warm",
        "mode": "gpu_shadow_lane",
        "launcher_only": True,
        "tenant_role": GPU_SHADOW_LANE_TENANT_ROLE,
    },
```

And in `stack_numa.py` `NUMA_CONFIG` (host-side pinning only; the model lives
in VRAM — no mlock):

```python
    # GPU shadow lane host threads: SMT siblings 184-191 (GPU host-thread rule;
    # the shape every np_context_study_v8 grid measured). NB 184-191 sits inside
    # NUMA_Q1B's SMT half -> contention recert at activation Step 4 must cover
    # frontdoor 8380, worker_general 8382, ingest 8485, vision_escalation 8087.
    "gpu_shadow_lane": {
        "instances": [("184-191", 18100, 8)],
        "mlock": False,
    },
```

## 3. `orchestrator_stack.py` builder (LANDED — no activation diff here)

> **2026-07-28 P2-6 punch-list amendment:** `_build_gpu_shadow_lane_command`,
> the `gpu_shadow_lane_mode` dispatch in `build_server_command`, and the
> `start_server` lane branch are **landed and inert** in
> `orchestrator_stack.py` (reachable only via the mode flag no server entry
> carries). MTP OFF per D6: the builder emits no speculative args by
> construction.

How the landed builder resolves its inputs:

- **Tenant priors**: `_stack_prior_launch(GPU_SHADOW_LANE_TENANT_ROLE)` — now
  falls back to the role's `launcher_tenant` stack-prior record (P0-1b), so an
  explicitly-requested launcher-only start resolves model path, HIP binary
  dir/LD paths, device, reasoning, and KV types WITHOUT the role being
  classified live.
- **Serving shape** (P0-1c): `-np`/`-c` come from the compiled priors cache
  (`slots`/`context_tokens`), which the stack-priors compiler fills from the
  `serving_shape` block of `orchestration/gpu_shadow_lane_np_ceiling.yaml`
  (`-np 8` × 8192-token slots ⇒ `-c 65536`) — NEVER from the CPU-mode launcher
  defaults (SERIAL_ROLES 2-slot / 32768-ctx). The builder's literal fallbacks
  (`GPU_SHADOW_LANE_FALLBACK_SLOTS/-CONTEXT_TOKENS`) mirror the same phase2
  shape and exist only for the degraded no-priors case; the preflight probe
  independently validates whatever is planned against the np_ceiling table.

CPU pinning comes from `_numa_prefix("gpu_shadow_lane")` (taskset 184-191),
applied by the launcher exactly as for every other role — the builder itself
never pins.

Reference plan (default `-np 8` × 8192-token slots, `phase2_resident_set`
row): `scripts/server/gpu_shadow_lane.py::build_tenant_launch_plan` emits the
full measured-shape argv; the preflight probe prints it in every report for
eyeball parity, and
`tests/unit/test_gpu_shadow_lane_compile_contract.py::TestGpuShadowLaneBuilder`
asserts flag-level parity between the two.

## 4. Verification checklist for the Step-2 pipeline gates

After applying §1–§2 (+ the `stack_numa.py` entry) + lean recompile, the
regenerated `orchestration/derived/stack_priors.yaml` block for
`coder_escalation_shadow` must show:
- [ ] `deployment_status: launcher_tenant` (P0-1 contract — NEVER `live_stack`; `live_stack_role_records` must not surface the role, `launcher_tenant_role_records` must)
- [ ] `launch.requirements.model_path: /mnt/raid0/llm/models/Qwen_Qwen3.6-27B-Q8_0.gguf`
- [ ] `runtime.binary_path: /mnt/raid0/llm/llama.cpp/build-hip/bin/llama-server`; `binary_dir` + `ld_library_path` same dir
- [ ] `runtime.env_policy: binary_override_strip_ggml`, `kmp_blocktime: 10` (derived — proves the binary override is active; the launcher strips `GGML_*` for non-canonical binaries)
- [ ] `runtime.flags: {flash_attn: true, device: ROCm0, reasoning: 'off'}`
- [ ] `runtime.cache: {context_tokens: 65536, slots: 8, kv_type_k: f16, kv_type_v: f16}` (= `-np 8` × 8192-token slots, sourced from the np_ceiling `serving_shape` block via the P0-1c compile path — if slots/context show the CPU defaults (2 / 32768), the serving-shape resolution failed: STOP)
- [ ] no `known_gaps` entry `Missing launcher-tenant serving shape …` (that gap = the serving_shape block failed to load). **Since P2-3d (`6defe059`) that gap has TWO distinct causes — diagnose before retrying**: (a) the block is missing/malformed, or (b) the block is present and well-formed but the shape is **outside the np_ceiling table** — `load_serving_shape()` now validates the shape against the ceiling rows in the same file and refuses `np_slots` above any tenant's saturation point, or any (np, ctx) cell no tenant/mode/budget row admits. Cause (b) is a *correct refusal*, not a plumbing failure: fix the `serving_shape` block or measure the cell, never widen the check. The raised `ValueError` names the refusing rows.
- [ ] `stack_change_pipeline.py check --run-promotion-gate` green (new gaps are NOT acceptable; documented known-gaps are)
- [ ] the State-A inertness witnesses in `tests/unit/test_gpu_shadow_lane.py::test_orchestrator_stack_has_no_lane_coupling` flipped to their State-B expectation (the manifest/NUMA/port assertions now legitimately find the lane — update the witness to assert the applied wiring instead; same designed-witness mechanism). The per-stage contract witnesses in `tests/unit/test_gpu_shadow_lane_compile_contract.py` need NO flip (they test both key-present and key-absent behavior with fixtures).
- [ ] **`git diff` in orchestrator touches ONLY**: lean/derived regenerations, `stack_manifest.py` (two dict entries + one PORT_MAP line), `stack_numa.py` (one NUMA_CONFIG entry). **Anything else = stop.** In particular the diff must NOT touch `orchestrator_stack.py`, `stack_commands.py`, or `stack_prewarm.py`: their lane plumbing (builder, `gpu_shadow_lane_mode` dispatch in cmd_start/reload, prewarm exclusion) was PRE-LANDED inert by the 2026-07-28 P2-6 punch-list commits, precisely so the activation diff stays minimal and pre-validated — a diff touching them means someone is re-implementing landed plumbing.

## 5. Rollback shape

`git revert` of the two activation commits (research: master-registry block;
orchestrator: the §2 manifest entries + `stack_numa.py` entry + regenerated
lean/priors — the pre-landed gated plumbing stays, returning to dormant),
lean recompile, Step-2 gates, `stop gpu_shadow_lane`, the Step-3a API-only
reload mirror (docs/gpu-shadow-lane.md §7 Step 7), contention recert.
Byte-for-byte the State-A shape — this file remains on disk as the dormant
proposal either way.
