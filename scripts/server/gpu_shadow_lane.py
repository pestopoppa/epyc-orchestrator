"""Role-agnostic GPU-resident shadow lane scaffolding (gpu-serving-tie-in P0-7).

TENANT AS DATA: the lane is the engineering deliverable; which model serves on
it is registry data, swapped via the State-A/State-B choreography documented in
``docs/gpu-shadow-lane.md`` (pattern:
``epyc-root/docs/runbooks/vision-escalation-minicpmo-promotion.md``).

Everything here is DEFAULT-OFF and deliberately uncoupled from the production
launch path:

- No module under ``scripts/server`` imports this file; ``orchestrator_stack``
  has zero references to the lane (witnessed by
  ``tests/unit/test_gpu_shadow_lane.py::test_orchestrator_stack_has_no_lane_coupling``).
- The np_ceiling policy loader is gated on the default-off
  ``ORCHESTRATOR_FEATURE_GPU_SHADOW_LANE`` feature flag
  (``src/features.py::gpu_shadow_lane`` — the ``eval_batch_serving`` pattern).
- The launch-plan builder returns an argv for REPORTING inside the preflight
  probe only; nothing in this module starts a process.

Shadow-only invariant (program decision D3): the lane serves no production
traffic until the Phase-3 bake-off evidence + operator three-gates sign-off.
Registry/manifest wiring is a PROPOSAL only —
``docs/proposals/gpu-shadow-lane-registry-proposal.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.features import Features, features as _global_features

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ── Lane constants (proposal defaults; registry priors override at activation) ──
LANE_NAME = "gpu_shadow_lane"
LANE_FEATURE = "gpu_shadow_lane"
LANE_PORT = 18100  # explicit-only high-port lane range (precedent: eval_batch_frontdoor 18070); clear of operator-owned 18072
LANE_DEVICE = "ROCm0"
# GPU host-thread rule: host-side threads live on the SMT siblings 184-191,
# never on physical cores 88-95 (memory: feedback_mi210_host_threads_smt_siblings).
LANE_HOST_CPUSET = "184-191"
LANE_HOST_THREADS = 8
# Production v8 HIP tree (2026-07-25 final freeze). Serving off any other tree
# violates production-kernel discipline.
LANE_BINARY_DIR = Path("/mnt/raid0/llm/llama.cpp/build-hip/bin")
LANE_BINARY = LANE_BINARY_DIR / "llama-server"
LANE_BINARY_VERSION = "10107"
LANE_BINARY_COMMIT = "67a433bf4"

# First tenant candidate (program D2). Proposal default only: at activation the
# master-registry role block (tenancy as data) is the source of truth.
TENANT_CANDIDATE_ID = "qwen36_27b_q8"
TENANT_CANDIDATE_MODEL = "/mnt/raid0/llm/models/Qwen_Qwen3.6-27B-Q8_0.gguf"

DEFAULT_NP_CEILING_POLICY_PATH = (
    PROJECT_ROOT / "orchestration" / "gpu_shadow_lane_np_ceiling.yaml"
)

_VALID_NP_LEVELS = (1, 2, 4, 8, 16, 32)


class GpuShadowLaneDisabled(RuntimeError):
    """Raised when lane scaffolding is used while the feature flag is off."""


def lane_enabled(feats: Features | None = None) -> bool:
    """Return True when the gpu_shadow_lane feature flag is enabled."""
    resolved = feats if feats is not None else _global_features()
    return bool(getattr(resolved, LANE_FEATURE, False))


# ── np_ceiling policy table (POLICY AS DATA) ─────────────────────────────────


@dataclass(frozen=True)
class BudgetRow:
    """Ceilings validated for one dynamic-VRAM budget."""

    name: str
    dynamic_budget_gib: float
    # per-slot context bucket (tokens) -> max -np, or None (no validated point)
    ceilings: dict[int, int | None]


@dataclass(frozen=True)
class TenantPolicy:
    """Per-tenant np/context policy derived from the measured grids."""

    tenant_id: str
    evidence_arm: str
    model_path: str
    model_vram_gib: float
    kv_bytes_per_token_f16: int | None
    per_seq_overhead_gib: float | None
    compute_reserve_gib: float
    np_throughput_saturation: int
    budgets: tuple[BudgetRow, ...]


@dataclass(frozen=True)
class NpCeilingPolicy:
    """Parsed orchestration/gpu_shadow_lane_np_ceiling.yaml."""

    version: int
    lane: str
    device: str
    vram_total_gib: float
    tenants: dict[str, TenantPolicy]


def _require_mapping(value: Any, label: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"np_ceiling policy: {label} must be a mapping")
    return value


def _parse_ceilings(raw: Any, label: str) -> dict[int, int | None]:
    ceilings: dict[int, int | None] = {}
    for key, value in _require_mapping(raw, label).items():
        bucket = int(key)
        if bucket <= 0:
            raise ValueError(f"np_ceiling policy: {label} bucket {key} must be positive")
        if value is None:
            ceilings[bucket] = None
            continue
        np_value = int(value)
        if np_value not in _VALID_NP_LEVELS:
            raise ValueError(
                f"np_ceiling policy: {label} bucket {bucket} ceiling {np_value} "
                f"is not a measured np level {_VALID_NP_LEVELS}"
            )
        ceilings[bucket] = np_value
    if not ceilings:
        raise ValueError(f"np_ceiling policy: {label} has no context buckets")
    return ceilings


def _parse_tenant(tenant_id: str, raw: Any) -> TenantPolicy:
    record = _require_mapping(raw, f"tenants.{tenant_id}")
    budgets: list[BudgetRow] = []
    raw_budgets = record.get("budgets")
    if not isinstance(raw_budgets, list) or not raw_budgets:
        raise ValueError(f"np_ceiling policy: tenants.{tenant_id}.budgets must be a non-empty list")
    for idx, raw_row in enumerate(raw_budgets):
        row = _require_mapping(raw_row, f"tenants.{tenant_id}.budgets[{idx}]")
        budgets.append(
            BudgetRow(
                name=str(row["name"]),
                dynamic_budget_gib=float(row["dynamic_budget_gib"]),
                ceilings=_parse_ceilings(
                    row.get("ceilings"), f"tenants.{tenant_id}.budgets[{idx}].ceilings"
                ),
            )
        )
    kv_bytes = record.get("kv_bytes_per_token_f16")
    per_seq = record.get("per_seq_overhead_gib")
    return TenantPolicy(
        tenant_id=tenant_id,
        evidence_arm=str(record["evidence_arm"]),
        model_path=str(record["model_path"]),
        model_vram_gib=float(record["model_vram_gib"]),
        kv_bytes_per_token_f16=int(kv_bytes) if kv_bytes is not None else None,
        per_seq_overhead_gib=float(per_seq) if per_seq is not None else None,
        compute_reserve_gib=float(record.get("compute_reserve_gib", 0.0)),
        np_throughput_saturation=int(record["np_throughput_saturation"]),
        budgets=tuple(budgets),
    )


def load_np_ceiling_policy(
    path: Path | None = None,
    *,
    feats: Features | None = None,
) -> NpCeilingPolicy:
    """Load + validate the np_ceiling policy table.

    Raises GpuShadowLaneDisabled unless the gpu_shadow_lane feature flag is on
    (default-off in both test and prod), so no production code path can consume
    the policy accidentally.
    """
    if not lane_enabled(feats):
        raise GpuShadowLaneDisabled(
            "gpu_shadow_lane feature flag is off (set ORCHESTRATOR_FEATURE_GPU_SHADOW_LANE=1 "
            "or pass an explicit Features override)"
        )
    policy_path = path or DEFAULT_NP_CEILING_POLICY_PATH
    payload = _require_mapping(
        yaml.safe_load(policy_path.read_text(encoding="utf-8")), "document"
    )
    version = int(payload.get("version", 0))
    if version != 1:
        raise ValueError(f"np_ceiling policy: unsupported version {version}")
    lane = str(payload.get("lane", ""))
    if lane != LANE_NAME:
        raise ValueError(f"np_ceiling policy: lane {lane!r} != {LANE_NAME!r}")
    tenants = {
        str(tenant_id): _parse_tenant(str(tenant_id), raw)
        for tenant_id, raw in _require_mapping(payload.get("tenants"), "tenants").items()
    }
    if not tenants:
        raise ValueError("np_ceiling policy: no tenants defined")
    return NpCeilingPolicy(
        version=version,
        lane=lane,
        device=str(payload.get("device", "")),
        vram_total_gib=float(payload.get("vram_total_gib", 0.0)),
        tenants=tenants,
    )


def np_ceiling(
    policy: NpCeilingPolicy,
    tenant_id: str,
    *,
    dynamic_budget_gib: float,
    slot_context_tokens: int,
) -> int | None:
    """Return the validated -np ceiling for a tenant, or None (= refuse).

    Row selection: the most permissive budget row whose dynamic_budget_gib does
    not exceed the budget actually available (rows are evidence "safe at >=
    this budget"). Bucket selection: the smallest measured context bucket that
    covers the requested per-slot context (conservative). None means "no
    validated operating point" — callers must refuse, never extrapolate.
    """
    if slot_context_tokens <= 0:
        raise ValueError("slot_context_tokens must be positive")
    tenant = policy.tenants.get(tenant_id)
    if tenant is None:
        raise KeyError(f"unknown tenant {tenant_id!r}")
    eligible = [row for row in tenant.budgets if row.dynamic_budget_gib <= dynamic_budget_gib]
    if not eligible:
        return None
    row = max(eligible, key=lambda item: item.dynamic_budget_gib)
    for bucket in sorted(row.ceilings):
        if slot_context_tokens <= bucket:
            return row.ceilings[bucket]
    return None


def estimated_dynamic_gib(
    tenant: TenantPolicy,
    *,
    np_slots: int,
    slot_context_tokens: int,
) -> float | None:
    """Estimated dynamic VRAM (GiB) for (np, per-slot ctx), excluding weights.

    Returns None when the tenant has no KV arithmetic model (measured-cells-only
    tenants such as the A4 bridge).
    """
    if tenant.kv_bytes_per_token_f16 is None or tenant.per_seq_overhead_gib is None:
        return None
    kv_gib = (np_slots * slot_context_tokens * tenant.kv_bytes_per_token_f16) / float(1 << 30)
    return kv_gib + np_slots * tenant.per_seq_overhead_gib + tenant.compute_reserve_gib


# ── Launch plan (REPORTING ONLY — never executed by this module) ─────────────


def build_tenant_launch_plan(
    *,
    model_path: str,
    np_slots: int,
    slot_context_tokens: int,
    port: int = LANE_PORT,
) -> list[str]:
    """Informational argv for the lane, mirroring the measured grid shape.

    Matches the np_context_study_v8 server argv (production HIP binary,
    taskset 184-191, -t/-tb 8, -fa on, f16 KV, reasoning off) with MTP OFF per
    program decision D6 (launch-bound; drained-lane relaunch to toggle). Used
    by the preflight probe's report and by the activation proposal; nothing in
    this repo executes it until the operator runs the activation choreography.
    """
    total_ctx = np_slots * slot_context_tokens
    return [
        "taskset",
        "-c",
        LANE_HOST_CPUSET,
        str(LANE_BINARY),
        "-m",
        model_path,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--metrics",
        "--slots",
        "--jinja",
        "--device",
        LANE_DEVICE,
        "-ngl",
        "all",
        "-fa",
        "on",
        "-np",
        str(np_slots),
        "-c",
        str(total_ctx),
        "-t",
        str(LANE_HOST_THREADS),
        "-tb",
        str(LANE_HOST_THREADS),
        "-b",
        "2048",
        "-ub",
        "2048",
        "-ctk",
        "f16",
        "-ctv",
        "f16",
        "--reasoning",
        "off",
    ]
