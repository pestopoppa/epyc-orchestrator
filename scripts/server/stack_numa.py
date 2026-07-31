"""NUMA CPU pinning topology + per-role pinning configuration.

Source of truth for "which CPUs and how many threads each role gets". Encodes
the validated EPYC 9655 topology + per-role NUMA wiring (Probe B 2026-05-04
canonical results, 2026-05-08 gemma4 MTP swap). `orchestrator_stack.py`
re-imports the constants and `_numa_prefix()` so the launcher and server-build
code keeps working unchanged.

Key findings (2026-03-18 benchmarks, refined through 2026-05-08):
- ⚠ SUPERSEDED 2026-07-30 — "Models ≤65 GB: 4×48t NUMA-quarter instances give
  6-7× aggregate throughput". That 6-7× was measured against a 1×192t
  all-SMT-threads baseline, NOT against a correctly-placed full-machine
  instance, and it was never matched on total concurrency. Directly measured
  at matched total concurrency T (llama-bench tg128, spec-dec off, drop_caches
  before every arm, kernel production-consolidated-v8 / binary 10107), one
  full-machine instance (taskset -c 0-95 + numactl --interleave=all) beats four
  quarters at EVERY rung: T=4 79.7 vs 52.9, T=8 105.1 vs 81.0, T=16 131.0 vs
  108.4, T=32 145.9 vs 143.8 aggregate tok/s.
- Models 130-250 GB: 1×96t pinning gives 1.2-1.5×. NB the historical
  "NUMA-node" wording here is a misnomer — see the NUMA_NODE0/NUMA_NODE1 note
  below; a node on this NPS4 host is 24 physical cores, i.e. one quarter.
- Using all 192t is ANTI-OPTIMAL (46-60% penalty vs 96 physical cores)
- ⚠ CORRECTED 2026-07-30 — "taskset alone is sufficient for the S4
  no-mmap/single-owner regime; shared mmap multi-instance roles need explicit
  evidence before changing memory policy". The first clause stands. The second
  understated the problem: shared-mmap fleets do not merely lack evidence for a
  memory policy, they have NO usable memory policy at all. GGUF pages are
  placed ONCE, by whichever instance faults them in first; every later instance
  inherits that placement regardless of its own --membind/--interleave.
  Measured on a quad-quarter fleet via live numa_maps: 25.6 / 25.6 / 24.2 /
  26.9 % node-local under shared mmap vs 100 % each under --no-mmap, with fleet
  decode 40.91 -> 52.13 tok/s. Consequence: fleet placement — and therefore
  fleet throughput — depends on instance START ORDER and is nondeterministic
  across reboots.
- mlock gives 30× latency improvement under memory pressure (S2)
- Total mlock budget: ~701 GB of 1.13 TB (62%), leaving ~429 GB for KV + OS

WIRING CHANGE APPLIED 2026-07-30 (operator-ratified). This block previously read
"NO WIRING CHANGE IS AUTHORISED … this module is deliberately behaviourally
unchanged … P-BENCH-PLACEMENT-1 … is STAGED, not applied". Both clauses are now
false and are corrected rather than deleted, so the transition stays auditable:

  * P-BENCH-PLACEMENT-1 was RATIFIED 2026-07-30 (epyc-root commit 07b7dcab) into
    MEASUREMENT.md §2 and the CPU annex measurement/protocols/bench-cpu.md.
    Figures taken under it are decision-grade within its scope — subject to its
    own gates, which ratification does not waive.
  * The topology was changed: quarters retired, every quarterable role now runs
    1 full + 2 halves. See the HALF FLEET block below and the per-role entries.

Two invariants are now ASSERTED AT IMPORT (see _assert_instance_invariants at
the end of this file), because both were derivable from the cpuset all along and
a mismatch is therefore always a bug: -t must equal the cpuset's PHYSICAL core
count, and any cpuset spanning more than one NPS4 node must declare a numactl
policy. Before today 15/19 instances violated the first and 17/19 the second,
including exactly the two later measured at 2.16x and 1.85x below canonical.

Analysis: epyc-root handoffs/active/numa-placement-defect-20260730.md.
Registration gate for any new model: epyc-inference-research
docs/protocols/model-registration-runbook.md (MRG-1).
"""

from __future__ import annotations


# =============================================================================
# NUMA CPU Pinning — validated via benchmarks (2026-03-18)
# =============================================================================
# EPYC 9655: 192 logical cores (96 physical + SMT), 4 NUMA nodes (NPS4), ~290 GB each.
# Node 0: cores  0-23, HT  96-119      Node 2: cores 48-71, HT 144-167
# Node 1: cores 24-47, HT 120-143      Node 3: cores 72-95, HT 168-191
#
# CORRECTED 2026-07-29 (`auditor`, P2-5l). This block previously read "2 NUMA nodes
# (~566 GB each) / Node 0: cores 0-47, HT 96-143 / Node 1: cores 48-95, HT 144-191",
# which was wrong on node count, per-node memory AND cpu ranges — and it contradicted
# the NUMA_Q* quarter definitions immediately below it, which were and are correct
# (NUMA_Q0A "0-23,96-119" is exactly node 0). Re-derived from `numactl -H` on this
# host: "available: 4 nodes (0-3)", node 0 cpus 0-23 + 96-119, node size 289860 MB.
# Nothing behavioural changed; the code was already right and only its own header
# misdescribed the machine. A misdocumented topology invariant is how the next
# placement defect gets built, which is why this was worth correcting rather than
# leaving as harmless prose.

# NUMA quarter definitions: (cpu_list, thread_count)
NUMA_Q0A = ("0-23,96-119", 48)
NUMA_Q0B = ("24-47,120-143", 48)
NUMA_Q1A = ("48-71,144-167", 48)
NUMA_Q1B = ("72-95,168-191", 48)
# ⚠ NAME IS AN NPS2-ERA ARTEFACT — THESE ARE NOT SINGLE NUMA NODES.
# Verified against live topology 2026-07-30 (`numactl --hardware`, NPS4):
#     node0 = 0-23,96-119   node1 = 24-47,120-143
#     node2 = 48-71,144-167 node3 = 72-95,168-191   (distances: local 10, remote 12)
# So NUMA_NODE0 spans node0+node1 and NUMA_NODE1 spans node2+node3. The names were
# correct under NPS2, before the 2026-04-24 NPS4 reboot; they have been misleading since.
# The quarter constants above ARE exactly the four NPS4 nodes and are correctly named.
#
# CONSEQUENCE, measured: with no numactl policy, weights first-touch onto whichever node
# loads them, so ~half a 96-thread team reads every weight cross-node. The E5 affinity
# artifact for this shape recorded pages_by_node {N0: 9226101} / total 9226101 — i.e. all
# 35.2 GiB on node0 while the threads spanned node0+node1.
#
# ⚠ The 2026-04-17 head-to-head quoted below to justify this wiring is NOT valid evidence
# for it: that run predates the NPS4 reboot (so the cpuset genuinely WAS one node then),
# and its source CSV records spec == "baseline", i.e. SPECULATIVE DECODING OFF.
# Do not cite 26.60/27.06 t/s as a current figure for this shape.
#
# The correct cpuset is under measurement (E5 re-run, 2026-07-30). Wiring intentionally
# left UNCHANGED until that reports — changing it now would break comparability with the
# recorded AutoPilot operating point we are re-anchoring against.
NUMA_NODE0 = ("0-47,96-143", 96)
NUMA_NODE1 = ("48-95,144-191", 96)
# Full-machine physical-cores-only (no SMT) — for canonical-recipe wiring
# (single-instance latency-optimal). 96 physical cores spanning all 4 NPS4 nodes.
# Pair with numactl_policy="interleave=all" so memory distributes across all 4 nodes
# (matches the canonical bench recipe used by Probe B 2026-05-04).
NUMA_FULL = ("0-95", 96)

# ── 2026-07-30 HALF FLEET (operator-ratified) ────────────────────────────────
# Same cpusets as NUMA_NODE0/NUMA_NODE1 but with the CORRECT thread count. Each
# half holds 48 PHYSICAL cores (0-47 or 48-95) plus their 48 SMT siblings, so -t
# is 48, not 96. NUMA_NODE0[1] / NUMA_NODE1[1] are both 96 and reusing them for a
# half instance silently re-introduces 2x SMT oversubscription — measured cost
# -13% per-stream and -8.5% aggregate at np=4. Do not reuse them here.
#
#   HALF_A = NPS4 nodes 0+1. GPU-DISJOINT. The GPU shadow lane pins host threads
#            to logical 184-191, which fold to physical 88-95 = region q3;
#            fold_smt_to_physical overlap with "0-47,96-143" is EMPTY. Measured
#            under a bandwidth-generating lane proxy, a 0-95 full instance lost
#            34% while this half lost nothing.
#   HALF_B = NPS4 nodes 2+3. NOT GPU-disjoint — it literally contains 72-95 and
#            168-191, so it IS a physical co-tenant of the lane.
#
# Pair each with its matching interleave list (interleave=0,1 / interleave=2,3).
# --interleave=all on a half would place weights on nodes its threads cannot
# reach locally, which is the same defect class this fleet exists to fix.
NUMA_HALF_A = ("0-47,96-143", 48)
NUMA_HALF_B = ("48-95,144-191", 48)

# ── Import-time invariants (2026-07-30) ──────────────────────────────────────
# Two of the three per-instance fields are DERIVABLE from the cpuset, so a
# mismatch is always a bug rather than a choice. Asserting here — a leaf module,
# so no import cycle — makes the 2026-07-30 defect class structurally
# unreachable rather than merely fixed.
#
# Before today 15/19 instances violated the thread rule and 17/19 the policy
# rule, including exactly the two measured at 2.16x and 1.85x below canonical.
# All 13 current instances satisfy both.
_NPS4_NODES = {0: "0-23,96-119", 1: "24-47,120-143",
               2: "48-71,144-167", 3: "72-95,168-191"}


def _parse_cpus(spec: str) -> set[int]:
    out: set[int] = set()
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))
    return out


def _nodes_touched(spec: str) -> list[int]:
    cpus = _parse_cpus(spec)
    return sorted(n for n, s in _NPS4_NODES.items() if cpus & _parse_cpus(s))


def _assert_instance_invariants() -> None:
    """Fail loudly at import if any instance contradicts its own cpuset."""
    problems: list[str] = []
    for role, cfg in NUMA_CONFIG.items():
        per = cfg.get("numactl_policy_instances") or {}
        one = cfg.get("numactl_policy")
        for idx, (cpus, port, threads) in enumerate(cfg.get("instances", [])):
            phys = len([c for c in _parse_cpus(cpus) if c < 96])
            if threads != phys:
                problems.append(
                    f"{role}[{idx}] :{port} -t {threads} but cpuset {cpus!r} holds "
                    f"{phys} PHYSICAL cores — SMT oversubscription"
                )
            nodes = _nodes_touched(cpus)
            if len(nodes) > 1 and not (per.get(idx) or one):
                problems.append(
                    f"{role}[{idx}] :{port} cpuset {cpus!r} spans NPS4 nodes {nodes} "
                    f"with NO numactl policy — weights land wherever they first touch"
                )
    if problems:
        raise AssertionError(
            "stack_numa NUMA_CONFIG invariant violation:\n  " + "\n  ".join(problems)
        )


# Per-role NUMA configurations.
# "instances" is a list of (cpu_list, port, threads) tuples.
# Roles with multiple instances get round-robin routing (requires orchestrator support).
NUMA_CONFIG: dict[str, dict] = {
    # Qwen3.5-35B-A3B Q4_K_M (19 GB) — pre-warm: 1×96t full-speed + 4×48t concurrent
    # Benchmark (2026-03-24): moe6 = 12.7 t/s at 48t. 96t TBD (expect higher per-request).
    # Pre-warm strategy (2026-03-29): 5 instances total, +19 GB (95 GB total for frontdoor).
    # Concurrency router: single session → full (96t), concurrent → quarter (48t) instances.
    "frontdoor": {
        # 2026-07-30 HALF FLEET. Quarters retired; 1 full + 2 halves.
        # Superseded the 2026-05-24 revert, whose justification (an April
        # 2026-04-17 head-to-head reading 27.06 vs 26.60 t/s) is RETRACTED: it
        # predates the 2026-04-24 NPS4 reboot, when "0-47,96-143" genuinely was
        # one NUMA node, and its source CSV records spec == "baseline".
        # Re-measured 2026-07-30 under P-BENCH-PLACEMENT-1, llama-bench tg128
        # r=10: as-wired 10.83 +/- 0.04 vs canonical 23.36 +/- 0.11 — a 2.16x
        # loss, not a 1.7% win.
        # Quarters retired because they lose at every matched concurrency, and
        # the margin WIDENS with load (aggregate tok/s at T=8: full 124.56,
        # 2xhalf 104.53, 4xquarter 89.25).
        "instances": [
            (NUMA_FULL[0], 8070, NUMA_FULL[1]),      # full   0-95, interleave=all
            (NUMA_HALF_A[0], 8080, NUMA_HALF_A[1]),  # half A nodes 0,1 — GPU-disjoint
            (NUMA_HALF_B[0], 8180, NUMA_HALF_B[1]),  # half B nodes 2,3 — GPU co-tenant
        ],
        "full_instance_idx": 0,
        # Prefer the split shape under burst, full for solo. See the enum note in
        # src/scheduling/placement_policy.py — "split" resolves to whatever
        # sub-full shape the role declares, which is now halves, not quarters.
        "placement_policy": "burst_prefer_quarters",
        "numactl_policy_instances": {
            0: "interleave=all",
            1: "interleave=0,1",
            2: "interleave=2,3",
        },
        "mlock": True,
        # TUNING SURFACE: the speculation recipe is NOT here. It lives in the
        # master registry's `acceleration` block for this role, and that single
        # field is the only thing an ngram change needs to touch.
    },
    # P-BENCH-3/A7 eval-batch serving lane. Warm/explicit-only; normal stack start
    # does not launch it. Uses the frontdoor model on a dedicated high port with
    # -np 8 so EvalTower batches can be tested without mutating the interactive
    # frontdoor process. CPU shape intentionally matches frontdoor's full instance
    # so existing frontdoor region locks remain conservative when traffic is
    # routed to this endpoint.
    "eval_batch_frontdoor": {
        # 2026-07-30: two corrections, both the same defect class as the fleet fix.
        # (1) -t was NUMA_NODE0[1] = 96 on a cpuset holding 48 PHYSICAL cores, i.e.
        #     2x SMT oversubscription. Measured cost elsewhere: -13% per-stream.
        # (2) numactl_policy was "interleave=all" on a cpuset spanning only NPS4
        #     nodes 0+1 — that places a quarter of the weights on nodes 2 and 3,
        #     which this instance's threads cannot reach locally. interleave=0,1
        #     matches the cpuset.
        # Not in the contention matrix's measured role set, so the -t change does
        # not move the matrix fingerprint.
        "instances": [
            (NUMA_HALF_A[0], 18070, NUMA_HALF_A[1]),
        ],
        "mlock": True,
        "numactl_policy": "interleave=0,1",
    },
    # coder_escalation NUMA_CONFIG entry REMOVED 2026-05-09 — consolidated onto
    # frontdoor's server (same Qwen3.6-35B-A3B Q8 GGUF since 2026-05-06 swap).
    # Historical: ports 8071 (full) + 8081/8181/8281/8381 (quarters), spec_overrides
    # {dm=32, ps=0.05} were Qwen2.5-Coder-32B-era params and didn't apply post-swap.
    # If a separate coder server is ever rostered again on a different model,
    # restore here with that model's tuned spec params.
    # Qwen3.5-122B-A10B Q4_K_M (69 GB) — 1×96t canonical (Probe B 2026-05-04)
    # Switched 2026-05-04 from 2× cross-NUMA (4.3 t/s/instance, 8.6 t/s agg) to
    # 1× full-machine canonical with numactl --interleave=all + GGML_NUMA_REPACK_INTERLEAVE=0
    # (c2 env block, see _ROLE_ENV_BLOCKS). Measured 12.19 t/s single-instance = +184%
    # per-request latency vs prior 2× wiring. Bundle:
    # epyc-inference-research/data/cpu_optimization/2026-05-04-qwen35-122b-arch-probe/
    # Reopen 4× per-NUMA-node wiring (16.86 t/s aggregate) ONLY if architect_general workload
    # shifts to 4+ concurrent batch eval — see findings_phase2.md.
    # ROLE DEFINITION (operator, 2026-07-30) — read this before "fixing" the
    # all-region hold below. This role is a **reasoning boost / critic**, invoked
    # only in select circumstances: autopilot planning when running fully local,
    # and dedicated high-value focused reasoning sessions. It is NOT a general
    # serving role and is not expected to carry routine traffic.
    #
    # Its single full instance on 0-95 holds ALL FOUR region locks, so while it
    # runs it serializes the entire CPU NUMA topology. That is INHERENT AND
    # ACCEPTED for this role, not a defect: there is no sub-full shape to demote
    # to, and a 122B at Q4_K_M has nothing smaller to be. Do not "fix" it by
    # adding a placement_policy — the other roles' burst_prefer_* exists because
    # they HAVE splits; this one does not.
    #
    # The expectation is that autopilot learns to mostly disregard it, because
    # the whole-machine lock is expensive relative to the reasoning gain on most
    # requests. For that to be learnable the cost must be ATTRIBUTABLE — a
    # whole-machine slowdown with no holder attribution is exactly the shape that
    # produced the mislearned `dual-half-negative` prior. See the autopilot
    # hygiene audit: `cpu_region_lock._acquire_one_with_timeout` already computes
    # wait seconds and resolves holders, so aggregating
    # `region_lock_wait_s_by_holder: {(role, shape_id): sec}` into the journal
    # turns this from an unattributable slowdown into a regressable per-role
    # scalar. Land that telemetry BEFORE exposing any routing knob for this role.
    "architect_general": {
        "instances": [
            (NUMA_FULL[0], 8083, NUMA_FULL[1]),  # 1×96t physical cores, all 4 NUMA nodes
        ],
        "mlock": True,
        "numactl_policy": "interleave=all",  # wraps launch with `numactl --interleave=all --`
        # 2026-06-26 v6 cutover: draft_max 24->4. The 24 was sweep-verified for the OLD Qwen3.5-0.8B
        # EXTERNAL draft; the architect now uses NEXTN self-draft (draft-mtp), whose bench-optimal
        # n-max is 4 (the +58-89% session bench). 24 over-drafts the NEXTN head (~25% accept, wasteful).
        "spec_overrides": {"draft_max": 4, "p_split": 0},
    },
    # architect_coding REMOVED 2026-05-06 — REAP-246B Q4KM scored 7/10 (70%) on coder
    # under canonical recipe, WORSE than worker_general (gemma4-26B-A4B Q4_K_M MTP at 96%)
    # AND far worse than frontdoor (Qwen3.6-35B-A3B Q8 at 97%). 139 GB warm freed.
    # Hard coding escalations now route to coder_escalation, which uses the same
    # Qwen3.6-35B-A3B Q8 model as frontdoor (shared GGUF mmap).
    # Qwen3-Next-80B-A3B Q4_K_M (45 GB) — pre-warm: 1×96t full + 4×48t quarters
    # 2026-05-24 Phase 0.5 bench: 48t single-quarter = 12.34 t/s, viable for
    # quartering despite the 45 GB GGUF being snug against per-quarter capacity.
    # Concurrency router routes solo → full (8085), concurrent → quarters
    # (8185/8285/8385/8485). Full anchor reverted to NUMA_NODE0 (same NPS4
    # half-socket pattern frontdoor uses, which the April 2026-04-17 data
    # showed beats NUMA_FULL+interleave=all by 1.7% on Qwen3.6-Q8. No direct
    # head-to-head data exists for Qwen3-Next-80B Q4, but it's an A3B MoE
    # like Qwen3.6 so cache-locality argument applies — keep parity).
    "ingest_long_context": {
        # 2026-07-30 HALF FLEET. The full previously sat on NUMA_NODE0
        # ("0-47,96-143") with NO memory policy at all, straddling NPS4 nodes 0+1
        # so every weight page landed wherever it was first touched. Measured
        # cost under P-BENCH-PLACEMENT-1: 12.42 -> 22.92 tok/s, a 1.85x loss.
        # This is also the ONE role where the split shape beats the full under
        # load rather than merely trailing it — aggregate tok/s: T=4 half 60.88
        # vs full 58.04; T=8 half 86.17 vs full 72.53. The SSM/MoE hybrid
        # behaves differently from the dense-attention MoEs.
        "instances": [
            (NUMA_FULL[0], 8085, NUMA_FULL[1]),      # full   0-95, interleave=all
            (NUMA_HALF_A[0], 8185, NUMA_HALF_A[1]),  # half A nodes 0,1 — GPU-disjoint
            (NUMA_HALF_B[0], 8285, NUMA_HALF_B[1]),  # half B nodes 2,3 — GPU co-tenant
        ],
        "full_instance_idx": 0,
        # Was ABSENT, so this role fell through to SOLO_PREFER_FULL and its full
        # instance — which holds all four region locks — could be acquired by a
        # single solo request and serialize the machine. That is the DISPATCH-A
        # shape (2026-07-21). frontdoor and worker_general were protected; this
        # role was not. Parity, not a preference: autopilot can tune it via the
        # AXA3 placement_policy knob surface once the enum is shape-agnostic.
        "placement_policy": "burst_prefer_quarters",
        "numactl_policy_instances": {
            0: "interleave=all",
            1: "interleave=0,1",
            2: "interleave=2,3",
        },
        "mlock": True,    # ~46 GB per instance — latency-critical for ingest pipeline (Stage 1 of three_stage_summarization since 2026-05-06)
    },
    # Worker: gemma4-26B-A4B Q4_K_M MTP (16 GB) — pre-warm: 1×96t + 4×48t.
    # Swapped 2026-05-08 from Qwen3-Coder-30B-A3B Q4_K_M (was 39 t/s at 48t).
    # gemma4-26B-A4B + ik_llama.cpp PR #1744 MTP: 76.5 t/s at 96t (full canonical), 95.2% draft acceptance.
    # +18pp on tool_compliance (96% vs 78%), +6pp on full suite (90% vs 84%).
    # Pre-2026-05-08: 7B f16 (until 2026-03-21), then Qwen3-Coder-30B-A3B Q4_K_M.
    # NB: full + 4 quarters share overlapping CPU sets — pick one mode at start time
    # (full instance uses 0-95; 4 quarters together also cover 0-95). See task #57.
    "worker_general": {
        "instances": [
            # 2026-05-08 swap to gemma4-26B-A4B MTP via ik_llama.cpp PR #1744:
            # full instance MUST use "0-95" (both NUMA nodes' physical cores) +
            # numactl --interleave=all to satisfy MTP's tensor-buffer NUMA expectation.
            # NUMA_NODE0's "0-47,96-143" (one-socket-with-SMT) crashed the MTP draft
            # path with "tensor buffer not set" assertion. Quarter instances retain
            # their per-quarter pinning since the full canonical recipe is incompatible
            # with the 4×concurrent design. Keep interleave scoped to idx0 so
            # --no-mmap quarters can first-touch local private pages.
            # 2026-07-30 HALF FLEET. Quarters retired — they lose at every
            # matched concurrency (gemma aggregate tok/s at T=8: full 256.77,
            # 2xhalf 200.12, 4xquarter 150.57) and the margin widens with load.
            # The full keeps "0-95" + interleave=all: gemma4 MTP asserts
            # "tensor buffer not set" on a one-socket cpuset, per the 2026-05-08
            # note this replaces.
            (NUMA_FULL[0], 8072, NUMA_FULL[1]),      # full   0-95, interleave=all
            (NUMA_HALF_A[0], 8082, NUMA_HALF_A[1]),  # half A nodes 0,1 — GPU-disjoint
            (NUMA_HALF_B[0], 8182, NUMA_HALF_B[1]),  # half B nodes 2,3 — GPU co-tenant
        ],
        "full_instance_idx": 0,
        # WP-7/J6 (2026-05-26): J5 -t48 re-bench → same_role borderline (mean
        # 0.879, all 6 quarter pairs net-positive 1.54-1.89x aggregate; gate
        # block→borderline). NOTE full="0-95" OVERLAPS all quarters, so full+
        # quarter never co-place (topology veto).
        # DISPATCH-A (2026-07-21, operator-granted): FULL_DISABLED. The full
        # (0-95) instance is NOT in the live serving stack (quarters-only; no
        # 8072 — orchestration/derived/stack_priors.yaml). Emitting a full-first
        # candidate made the burst placement SM acquire idx-0's ALL-region lock
        # (all four per-role region locks + all four global cross-role mutexes),
        # serializing the machine and starving concurrent same-role requests on
        # a 150ms poll. full="0-95" overlaps every quarter, so full can never
        # co-place with a quarter anyway: quarters-only is the correct static
        # mode until an operator redeploys a real full instance. This reclaims
        # the full's mlock and lets 4 same-role requests occupy 4 disjoint
        # quarters (design contract: big instance idle under concurrent load).
        # LINEUP RESTORATION (2026-07-23, operator-directed): the v7-cutover
        # quarter-mode launch dropping 8072 was ruled an accidental lineup
        # regression, not policy — "all models run as a full performance
        # instance and quarter instances for concurrent aggregate boost;
        # overlapping instances cannot infer concurrently." The full is being
        # redeployed (the condition named above), so FULL_DISABLED reverts to
        # BURST_PREFER_QUARTERS: solo keeps full-first for peak throughput;
        # any self-role holder demotes the full to a trailing candidate, and
        # 0-95 overlapping every quarter keeps it region-vetoed under burst.
        "placement_policy": "burst_prefer_quarters",
        "mlock": True,
        "spec_overrides": {"draft_max": 2, "p_split": 0},  # gemma4 MTP recipe (was dm=8 for Qwen3-Coder)
        # Every instance now declares a policy. Previously only idx0 did, which
        # left the sub-full instances on default first-touch — the defect class
        # that cost frontdoor and ingest_long_context ~2x for months.
        "numactl_policy_instances": {
            0: "interleave=all",   # required for gemma4 MTP idx0
            1: "interleave=0,1",
            2: "interleave=2,3",
        },
    },
    # Qwen2.5-VL-7B Q4_K_M (~4 GB) — single instance on Q0B at 24t.
    # Phase 0.5 bench (2026-05-24) showed 24t = 11.39 t/s, 48t = 11.30 t/s
    # (flat — model too small to benefit from more threads or quartering).
    # An earlier 2026-05-24 commit (6657bbd) added a 96t full instance + 4
    # quarters, all using 24t each = 16 GB unnecessary mlock budget. Reverted
    # to the original single-quarter shape: this role just doesn't have
    # enough request volume to justify concurrent-serving topology.
    # NOTE — CROSS-ROLE mmap sharing. worker_vision and vision_escalation are two
    # SEPARATE roles pinned to two DIFFERENT NPS4 nodes that point at ONE shared
    # VL GGUF. Under shared mmap its pages are placed once, by whichever loads
    # first, so at most one of them can ever be node-local: both measured 0.25
    # local. This refutes the intuition that a single-instance role is immune to
    # the shared-mmap defect — here the sharing is across roles. Each needs an
    # explicit membind AND no_mmap in the registry, or the membind is accepted
    # and silently does nothing.
    "worker_vision": {
        "instances": [(NUMA_Q0B[0], 8086, 24)],
        "mlock": True,    # ~4 GB — minimal footprint
        "numactl_policy": "membind=1",   # NUMA_Q0B == NPS4 node1
    },
    # The vision escalation lane runs on MI210 with a small 24t host quarter for
    # request/image preprocessing. Keep it distinct from worker_vision so ports
    # 8086 and 8087 can coexist without overlapping CPU masks.
    #
    # 2026-07-31: this comment previously named MiniCPM-o as the tenant. MiniCPM-o
    # is DEPRECATED and its weights are deleted — the 2026-07-18 activation policy
    # that would have flipped this lane to it is REVOKED. On a 42-question
    # OCRBench+ChartQA set MiniCPM-o scored 31/42 against the Qwen2.5-VL-7B
    # incumbent's 35/42, i.e. promotion would have been a quality regression.
    # The lane itself is tenant-agnostic and still valid; which model serves on it
    # is registry data. The forward candidate, if this is ever revisited, is
    # Qwen3-VL-30B-A3B Q4_K_M (36/42). The 24t/membind=3 sizing below was chosen
    # for a ~6 GB host-visible VL tenant and should be re-derived for any new one.
    "vision_escalation": {
        "instances": [(NUMA_Q1B[0], 8087, 24)],
        "mlock": True,    # ~6 GB host-visible + MI210 resident VL lane
        "numactl_policy": "membind=3",   # NUMA_Q1B == NPS4 node3
    },
}

# Roles that should use --mlock (requires ulimit -l unlimited in launch env)
MLOCK_ROLES = {role for role, cfg in NUMA_CONFIG.items() if cfg.get("mlock")}


def _numa_prefix(role: str, instance_idx: int = 0) -> list[str]:
    """Return CPU-pinning + memory-policy prefix for a role instance.

    Default: taskset -c <cpu_list>. The S4 benchmark found no benefit from
    adding numactl --membind for no-mmap/single-owner runs where first-touch owns
    the private copy. That finding does not generalize to shared-mmap quarter
    fleets.

    ⚠ CORRECTED 2026-07-30 (comment only — no behaviour change here). This
    docstring used to continue: "those roles rely on interleaved shared pages
    unless a role-specific A/B plus live numa_maps proof justifies changing
    memory policy". That premise is DISPROVED. Shared-mmap quarter fleets do
    NOT reliably get interleaved pages — they get whatever the FIRST instance
    to fault a page in chose. The GGUF is placed once; later instances inherit
    that placement regardless of their own --membind/--interleave, so at most
    one instance can be node-local and fleet placement depends on instance
    START ORDER.

    The evidence this docstring demanded now exists, and it is exactly a
    role-specific A/B plus live numa_maps: a quad-quarter fleet measured
    25.6 / 25.6 / 24.2 / 26.9 % node-local pages under shared mmap versus
    100 % each under --no-mmap, with fleet decode 40.91 -> 52.13 tok/s (at a
    cost of ~+141 GB RAM, since --no-mmap gives each instance a private copy).
    Instrument: llama-bench tg128, spec-dec off, drop_caches before every arm,
    kernel production-consolidated-v8 / binary 10107.

    NOT AUTHORISED AS A CHANGE. This is OBSERVATION-GRADE — P-BENCH-PLACEMENT-1
    has a MEASUREMENT.md registry entry that is STAGED, not applied, so it may
    not gate a keep / revert / deploy / promote decision. Any memory-policy or
    mmap change to a serving role is operator-gated and must be executed by the
    session that owns the inference. See the module docstring.

    If the role's NUMA_CONFIG entry has a "numactl_policy" key (e.g. "interleave=all"),
    wraps the launch with `numactl --<policy> --` ahead of taskset. Roles can
    instead provide "numactl_policy_instances" for instance-specific policy,
    such as worker_general idx0 requiring interleave while its --no-mmap
    quarter instances should rely on taskset + first-touch locality. Used for
    canonical-recipe roles like architect_general (Probe B 2026-05-04: numactl
    --interleave=all + taskset -c 0-95 = 12.19 t/s single-instance vs 4.3 t/s under
    legacy 2× cross-NUMA + first-touch).
    """
    cfg = NUMA_CONFIG.get(role)
    if cfg and instance_idx < len(cfg["instances"]):
        cpu_list = cfg["instances"][instance_idx][0]
        prefix: list[str] = []
        policy = None
        instance_policies = cfg.get("numactl_policy_instances")
        if isinstance(instance_policies, dict):
            policy = instance_policies.get(instance_idx)
            if policy is None:
                policy = instance_policies.get(str(instance_idx))
        if policy is None:
            policy = cfg.get("numactl_policy")
        if policy:
            prefix.extend(["numactl", f"--{policy}", "--"])
        prefix.extend(["taskset", "-c", cpu_list])
        return prefix
    # Fallback: no pinning (embedders, fast workers, dev mode)
    return []


_assert_instance_invariants()
