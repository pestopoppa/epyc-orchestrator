# GPU Shadow Lane — role-agnostic MI210 serving slot (spec + choreography)

**Filed**: 2026-07-28, gpu-serving-tie-in-program task P0-7 (scaffolding; zero-inference).
**Status**: SCAFFOLD — nothing here is active. The feature flag
(`ORCHESTRATOR_FEATURE_GPU_SHADOW_LANE`) is default-off in test AND prod, the
registry is FROZEN (no role exists), and no launch-layer file references the
lane (witnessed by `tests/unit/test_gpu_shadow_lane.py::test_orchestrator_stack_has_no_lane_coupling`).
**Program authority**: `epyc-root/handoffs/active/gpu-serving-tie-in-program.md`
(ratified decisions D1–D10).
**Pattern template**: `epyc-root/docs/runbooks/vision-escalation-minicpmo-promotion.md`
(the State-A/State-B choreography; this spec deliberately mirrors its structure).
**Flag/probe precedent**: the `eval_batch_serving` feature (orchestrator
`7cb71a4e` default-off flag → `276a1eef` guarded probe → activation window with
rollback; see `epyc-root/handoffs/active/batched-decode-measurement.md` §E2).

---

## 1. The lane contract

**The engineering deliverable is the slot, not the tenant.** The lane is a
single resident llama-server process on the MI210 (`ROCm0`) with a small
host-side thread quarter. Everything tenant-specific — model path, quant,
sampling, duty bindings — is **registry data**:

| Contract clause | Meaning |
|---|---|
| Registry-driven tenancy | The tenant is defined ONLY by a master-registry role block (proposal: `docs/proposals/gpu-shadow-lane-registry-proposal.md`). Swapping tenants (e.g. stock-27B → FF after the P3 bake-off, D7) is a registry edit + the Step 1–6 choreography below — never a code change. |
| One resident tenant | One large tenant at a time (program P3-2). MiniCPM-o (8087 vision escalation) and whisper are separate small lanes with their own runbooks; they co-reside within the VRAM budget (§5). |
| Launch-bound modes (D6) | MTP on/off is a **launch property** of v8 (`params.speculative` is global; no request-level override). Default **MTP OFF** (escalation-dominant duty mix; MTP measured net-negative single-stream deep-context). Mode toggle = drained-lane relaunch (~1–2 min) at a boundary via this choreography — never a hot mutation. |
| Drain, never force | Lane teardown/swap happens by quiesce + drain at a request boundary (fabric axiom 4). No mid-decode preemption. |
| np policy as data | Slot-count admission (`-np` × per-slot context) is governed by `orchestration/gpu_shadow_lane_np_ceiling.yaml` (§6). A null ceiling means refuse, never extrapolate. |
| Host threads | Host-side threads pin to SMT siblings **184-191** (`-t 8 -tb 8`), never physical cores 88-95 — the GPU host-thread rule, and the shape every v8 np×context grid actually measured. |
| Production kernel only | The lane serves off the frozen production HIP tree `/mnt/raid0/llm/llama.cpp/build-hip/bin` (`llama-server --version` = `10107 (67a433bf4)`). Serving off any experimental tree violates production-kernel discipline. |

## 2. Registry states (the State-A/B choreography)

- **State A (current / rollback state)** = lane absent. No registry role, no
  `stack_manifest` wiring, feature flag off, port 18100 unbound, MI210 free
  for bench windows. This document plus the proposal file ARE State A's
  artifacts.
- **State B (active state)** = lane resident with tenant *T*: master-registry
  role block applied (Step 1), lean recompiled, `gpu_shadow_lane` launched via
  `orchestrator_stack.py start --only gpu_shadow_lane`, preflight attestation
  on file, contention recert done.
- **Tenant swap (B→B′)** and **mode toggle (MTP on↔off)** are the same Steps
  1–6 run with the new registry data; **rollback is the same runbook run
  toward State A** (Step 7). Every strategic pivot is a config change with a
  rehearsed rollback, never a rebuild.

## 3. Admission policy (program decision D1 — design provision)

Priority-ordered lane admission, highest first:

| # | Class | Status |
|---|---|---|
| 1 | **Escalations** (coder-escalation-shaped, CPU→GPU chains) | Phase-3 bake-off subject; first implemented consumer |
| 2 | **Distillation backfill** (teacher calls riding continuous batching, D8) | Post-reboot wiring; rides the same lane admission |
| 3 | **Shed batch** (worker_general-class batched work under CPU stress, ~135 t/s aggregate class) | **Named here, MAY REMAIN UNIMPLEMENTED** until after lane hardening |
| 4 | **Degraded frontdoor overflow** | **Named here, exists only as an explicit, flagged, telemetered degraded mode; MAY REMAIN UNIMPLEMENTED.** Frontdoor stays single-model on CPU. |

Admission classes 3 and 4 get their own default-off flags when (if) they are
built; declaring them here reserves the ordering so later implementation is a
policy fill-in, not a redesign.

## 4. Shadow-only invariant (program decision D3)

The registry stays frozen; `coder_escalation` stays A4-bound (Qwen3.6-35B-A3B
on CPU). The GPU lane serves **no production traffic** — all lane traffic is
eval-path / shadow duty (forced-role bench windows, P3 bake-off arms) — until
**Phase-3 bake-off evidence + operator three-gates sign-off (P3-3)**. Nothing
in this spec, the proposal file, or the scaffolding code authorizes a lineup
change by itself.

## 5. VRAM budget (program decision D2)

MI210 = 64 GB HBM2e, `ROCm0`.

| Resident tenant | VRAM |
|---|---|
| dense 27B Q8_0 (stock `Qwen_Qwen3.6-27B-Q8_0.gguf`, 28,665,067,072 B; FF alternative has identical footprint) | **28.7 GB** |
| MiniCPM-o 4.5 Q4 + vision projector (vision-escalation runbook, 8087) | **~7 GB** (K35 measured ~11% band) |
| whisper STT | **~1.6 GB** |
| **Resident total** | **≈ 37 GB** |
| **Dynamic headroom** (KV + recurrent state + compute graphs) | **≈ 27 GB** |

The dynamic budget is what the np_ceiling policy's `phase2_resident_set` row
is bound to (conservatively, 27.0 GiB). Solo-residency bench windows get the
`solo_resident` row (≈ 37.3 GiB dynamic).

## 6. np_ceiling policy (data, not code)

`orchestration/gpu_shadow_lane_np_ceiling.yaml`, loaded by
`scripts/server/gpu_shadow_lane.py::load_np_ceiling_policy` **only when the
default-off `ORCHESTRATOR_FEATURE_GPU_SHADOW_LANE` flag is set** (the
`eval_batch_serving` flag pattern). Derived from the measured v8 np×context
grids (`epyc-inference-research/artifacts/np_context_study_v8_20260727/`,
read-only aggregator; `A3_ff_fable_non_mtp_q8` = same dense-27B architecture,
MTP OFF; `A4_35b_a3b_v8_bridge` = the sequential bake-off control arm).

Key facts encoded there:
- Aggregate decode **saturates at np16** (np32 regresses −20% at L2048);
  ceilings are capped at 16 everywhere.
- Solo anchors: `np16×L32768` (f16 KV = 32 GiB) measured fit;
  `np32×L16384` (same total ctx) hit `allocation_failure` — the per-sequence
  recurrent-state + graph overhead is real and encoded conservatively.
- Phase-2 rows are **VRAM arithmetic over solo anchors, not co-resident
  measurements** — observation-grade; re-certify at P2-3 Stage-0 hardening.
- A null ceiling = no validated operating point = refuse.

MEASUREMENT discipline: the grids are observations (no promotion-gating
protocol attestation). They bound lane admission; they never gate a
promotion/rebind decision alone.

## 7. Activation choreography (operator-gated; mirrors the vision runbook)

> Every literal needed by Steps 1–3 is pre-written in
> `docs/proposals/gpu-shadow-lane-registry-proposal.md`. Nothing below may run
> without the Step-0 operator grant.

### Step 0 — Preconditions (all must hold)

| # | Precondition | How to verify |
|---|---|---|
| P1 | **Operator grant** for the activation itself + the Step-4/6 inference (MEASUREMENT policy: benchmarks via codified recipes with approval) | Written directive in session |
| P2 | **MI210 free** — no foreign compute PIDs (operator-owned external processes are never killed; wait for vacate) | `rocm-smi --showmemuse --showpids`; preflight probe |
| P3 | **Quiet window** — no AutoPilot, no EvalTower batch in flight (SIGSTOP/SIGCONT an in-flight eval runner around reload points) | `pgrep -af autopilot.py`; reports dir quiet |
| P4 | **Tenant artifacts on disk, hash-known** — stock 27B: 28,665,067,072 B, sha256 `5927dc06c2b19f732fb6e2a6546dff4c130b552f2ab5f91feb3daafe43897b2a` | `ls -l` + `sha256sum` |
| P5 | **Production v8 HIP binary healthy** — `10107 (67a433bf4)` | preflight probe (binary check) |
| P6 | **Clean git baselines** in both repos (rollback anchor is a git revert) | `git status` / `git log @{u}..` |
| P7 | **Realized fleet is the terminal lineup**; three-gates discipline applies | `orchestrator_stack.py status` + runtime-facts manifest |
| P8 | **Preflight probe clean**: `uv run python scripts/server/gpu_shadow_lane_preflight.py` exits 0 (plan-only), then `--apply` writes `preflight_attestation.json` | probe report under `orchestration/reports/gpu_shadow_lane_preflight_<ts>/` |

### Step 1 — Registry rebind (MASTER registry only) + launch constants

Apply the role block and the `stack_manifest`/`stack_numa`/`PORT_MAP`/builder
wiring **exactly as written in the proposal file**. The lean registry is
auto-compiled from master — never hand-edit the lean (the 2026-07-18
clobber-exposed lean hand-edit is the motivating failure). Then recompile
explicitly: `uv run python -m src.registry.registry_compiler --force`.

### Step 2 — Pipeline gates (no-inference)

`stack_change_pipeline.py update` + `check --run-promotion-gate`; verify the
regenerated `stack_priors.yaml` block against the proposal's verification
checklist (binary_dir → derived `env_policy: binary_override_strip_ggml`,
`kmp_blocktime: 10`, `flags: {device: ROCm0, reasoning: 'off', flash_attn: true}`).

### Step 3 — Lane start (additive, no-outage)

`orchestrator_stack.py start --only gpu_shadow_lane` over the live fleet (the
`eval_batch_frontdoor` activation shape: warm/explicit-only role, additive;
skip-healthy leaves every running server untouched). Verify: `/health` on 18100; `/proc/<pid>/cmdline` shows the
production HIP binary, `--device ROCm0`, `-fa on`, `--reasoning off`, `-np`/`-c`
within the np_ceiling policy, taskset 184-191; `rocm-smi` shows the expected
residency; every other port preserved (compare PIDs before/after).

### Step 4 — Contention recert (the step the 2026-07-17 rebind skipped)

The lane's host slice 184-191 is inside `NUMA_Q1B`'s SMT range — static
co-tenants: frontdoor quarter 8380, worker_general quarter 8382, ingest
quarter 8485, vision_escalation 8087 (the preflight probe enumerates these).
Re-bench every contention pair containing the lane + those roles
(`contention_matrix.py --roles …`), validate + freshness-gate. The topology
hash will NOT catch this change.

### Step 5 — Live-affinity + GPU residency attestation

`affinity_preflight.py` for the host quarter; `rocm-smi --showmemuse
--showpids` for the device half (VRAM within §5 budget, no foreign PIDs);
`generate_attestation.py` for the read-only running-state record.

### Step 6 — Smoke via the eval path (never live /chat)

Eval-path probe with the route **forced to the lane's role** (EvalTower window
runner `--roles` forcing, `--apply --confirm-clean-window` gating — the
vl-truth-slice pattern). Score `message.content` (reasoning off is
load-bearing). Pass = every row routed to the lane, zero error rows, decode in
the measured band for the planned (np, L) cell (e.g. np1: ~26–30 t/s
single-stream; np8×L8192: ~47 t/s aggregate). Outside band → stop, diagnose,
or roll back.

### Step 7 — Rollback (= this runbook toward State A)

`git revert` the Step-1 commits in both repos, recompile lean, re-run Step 2,
`orchestrator_stack.py stop gpu_shadow_lane` (drain at boundary), confirm port
18100 free + VRAM released, Step 4 recert (a rollback is also a lane change).
Rollback triggers: Step-3 health/cmdline mismatch; Step-6 error rows or >20%
below band; VRAM/KFD anomalies; any contention pair below the 0.65 floor.

## 8. NOT covered by this spec (explicit)

- **Tenant choice** (stock vs FF) — P3 bake-off + operator decision package.
- **Any `coder_escalation` rebind** — P3-3 three-gates only.
- **Admission classes 3/4 implementation** (shed batch, degraded overflow) —
  named in §3, deferred past lane hardening.
- **Teacher/distillation wiring** (D8) — post-reboot item.
- **MiniCPM-o promotion** — its own parked runbook (vision-escalation).
- **AutoPilot knobs on the lane** (P4) — typed knobs only, after P3-3; autopilot
  never touches launch plumbing, registry, or lane lifecycle.
- **MEASUREMENT trust-boundary artifacts** — human-amendment-only.
- **v9 kernel work** — production v8 is FROZEN; no patches.

## 9. Reporting

On activation/rollback: flip the owning checkboxes in
`epyc-root/handoffs/active/gpu-serving-tie-in-program.md` (checkbox
discipline), append to `progress/`, record contention-refresh + attestation
artifact paths in the promotion commit messages (one commit per repo).
