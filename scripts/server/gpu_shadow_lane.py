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
TENANT_CANDIDATE_ID = "qwen36_27b_stock_q8"
TENANT_CANDIDATE_MODEL = "/mnt/raid0/llm/models/Qwen_Qwen3.6-27B-Q8_0.gguf"

DEFAULT_NP_CEILING_POLICY_PATH = (
    PROJECT_ROOT / "orchestration" / "gpu_shadow_lane_np_ceiling.yaml"
)

_VALID_NP_LEVELS = (1, 2, 4, 8, 16, 32)

# Launch modes. MTP is a LAUNCH property of v8 (program decision D6:
# ``params.speculative`` is global, there is no per-request override), and it
# moves the validated capacity frontier — the FF arm's np16 x L32768 cell fits
# with MTP off and is a capacity skip with MTP on. Ceilings are therefore
# resolved per mode, and a mode with no rows REFUSES rather than falling back
# to the other mode's frontier.
MODE_MTP_OFF = "mtp_off"
MODE_MTP_ON = "mtp_on"
VALID_MODES = (MODE_MTP_OFF, MODE_MTP_ON)


def mode_for(*, mtp: bool) -> str:
    """Map the tenancy ``mode.mtp`` boolean onto a policy mode key."""
    return MODE_MTP_ON if mtp else MODE_MTP_OFF


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
class ModePolicy:
    """Budget rows validated for ONE launch mode (mtp_off / mtp_on)."""

    mode: str
    evidence_arm: str
    evidence_basis: str
    budgets: tuple[BudgetRow, ...]


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
    # Per-mode rows. Always contains MODE_MTP_OFF (built from ``budgets``);
    # contains MODE_MTP_ON only when the tenant declares mode_overrides.mtp_on.
    modes: dict[str, ModePolicy]
    evidence_basis: str = "measured"
    model_bytes: int | None = None
    model_sha256: str | None = None
    # Self-draft depth the grid was measured at. Part of the measured identity:
    # a different depth is a different, unmeasured operating point.
    draft_n_max: int | None = None

    def mode_policy(self, mode: str) -> ModePolicy | None:
        """Rows for ``mode``, or None when that mode has no validated frontier."""
        return self.modes.get(mode)


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


def _parse_ceilings(raw: Any, label: str, *, saturation: int) -> dict[int, int | None]:
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
        # The saturation cap is ENFORCED, not merely annotated: a ceiling above
        # the tenant's measured throughput-saturation point would authorise a
        # launch that is slower AND more VRAM-hungry than the capped one. P0-7
        # documented the cap in a comment; a comment cannot fail a load.
        if np_value > saturation:
            raise ValueError(
                f"np_ceiling policy: {label} bucket {bucket} ceiling {np_value} "
                f"exceeds np_throughput_saturation {saturation}"
            )
        ceilings[bucket] = np_value
    if not ceilings:
        raise ValueError(f"np_ceiling policy: {label} has no context buckets")
    return ceilings


def _parse_budgets(
    raw: Any, label: str, *, saturation: int, allow_empty: bool = False
) -> tuple[BudgetRow, ...]:
    if raw is None or (allow_empty and isinstance(raw, list) and not raw):
        # An explicitly empty list means "this mode has no validated frontier".
        # That is a REFUSAL, encoded as data — e.g. the A4 bridge, which was
        # only ever measured with MTP on, declares no mtp_off rows.
        return ()
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"np_ceiling policy: {label} must be a non-empty list")
    rows: list[BudgetRow] = []
    for idx, raw_row in enumerate(raw):
        row = _require_mapping(raw_row, f"{label}[{idx}]")
        rows.append(
            BudgetRow(
                name=str(row["name"]),
                dynamic_budget_gib=float(row["dynamic_budget_gib"]),
                ceilings=_parse_ceilings(
                    row.get("ceilings"), f"{label}[{idx}].ceilings", saturation=saturation
                ),
            )
        )
    return tuple(rows)


def _parse_tenant(tenant_id: str, raw: Any) -> TenantPolicy:
    record = _require_mapping(raw, f"tenants.{tenant_id}")
    saturation = int(record["np_throughput_saturation"])
    if saturation not in _VALID_NP_LEVELS:
        raise ValueError(
            f"np_ceiling policy: tenants.{tenant_id}.np_throughput_saturation "
            f"{saturation} is not a measured np level {_VALID_NP_LEVELS}"
        )
    evidence_arm = str(record["evidence_arm"])
    evidence_basis = str(record.get("evidence_basis", "measured"))
    budgets = _parse_budgets(
        record.get("budgets"),
        f"tenants.{tenant_id}.budgets",
        saturation=saturation,
        allow_empty=True,
    )
    modes: dict[str, ModePolicy] = {}
    if budgets:
        modes[MODE_MTP_OFF] = ModePolicy(
            mode=MODE_MTP_OFF,
            evidence_arm=evidence_arm,
            evidence_basis=evidence_basis,
            budgets=budgets,
        )
    for mode, raw_override in _require_mapping(
        record.get("mode_overrides") or {}, f"tenants.{tenant_id}.mode_overrides"
    ).items():
        mode_key = str(mode)
        if mode_key not in VALID_MODES:
            raise ValueError(
                f"np_ceiling policy: tenants.{tenant_id}.mode_overrides has unknown "
                f"mode {mode_key!r} (valid: {VALID_MODES})"
            )
        override = _require_mapping(
            raw_override, f"tenants.{tenant_id}.mode_overrides.{mode_key}"
        )
        modes[mode_key] = ModePolicy(
            mode=mode_key,
            evidence_arm=str(override.get("evidence_arm", evidence_arm)),
            evidence_basis=str(override.get("evidence_basis", evidence_basis)),
            budgets=_parse_budgets(
                override.get("budgets"),
                f"tenants.{tenant_id}.mode_overrides.{mode_key}.budgets",
                saturation=saturation,
            ),
        )
    if not modes:
        raise ValueError(
            f"np_ceiling policy: tenants.{tenant_id} declares no validated mode "
            "(both budgets and mode_overrides are empty)"
        )
    kv_bytes = record.get("kv_bytes_per_token_f16")
    per_seq = record.get("per_seq_overhead_gib")
    model_bytes = record.get("model_bytes")
    return TenantPolicy(
        tenant_id=tenant_id,
        evidence_arm=evidence_arm,
        model_path=str(record["model_path"]),
        model_vram_gib=float(record["model_vram_gib"]),
        kv_bytes_per_token_f16=int(kv_bytes) if kv_bytes is not None else None,
        per_seq_overhead_gib=float(per_seq) if per_seq is not None else None,
        compute_reserve_gib=float(record.get("compute_reserve_gib", 0.0)),
        np_throughput_saturation=saturation,
        budgets=budgets,
        modes=modes,
        evidence_basis=evidence_basis,
        model_bytes=int(model_bytes) if model_bytes is not None else None,
        model_sha256=(
            str(record["model_sha256"]) if record.get("model_sha256") is not None else None
        ),
        draft_n_max=(
            int(record["draft_n_max"]) if record.get("draft_n_max") is not None else None
        ),
    )


def _read_policy_payload(path: Path | None) -> dict:
    policy_path = path or DEFAULT_NP_CEILING_POLICY_PATH
    return _require_mapping(
        yaml.safe_load(policy_path.read_text(encoding="utf-8")), "document"
    )


def _parse_policy_payload(payload: dict) -> NpCeilingPolicy:
    """Parse + validate a policy document. NO feature gate.

    Split out of ``load_np_ceiling_policy`` so ``load_serving_shape`` can reach
    the tenant rows without the flag (P2-3d). The GATE belongs on the public
    consumer entry point, not on the parsing — otherwise the ungated caller
    would need its own second parser, which is exactly how the two blocks in
    this file drifted apart in the first place.
    """
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
    return _parse_policy_payload(_read_policy_payload(path))


def shape_admissibility(
    policy: NpCeilingPolicy, *, np_slots: int, slot_context_tokens: int
) -> tuple[list[str], list[str]]:
    """Where a serving shape is validated, and where it is not.

    Returns ``(admitting, refusing)`` as ``"tenant/mode/profile"`` labels over
    every tenant x mode x budget row in the table. A shape admitted NOWHERE is
    unusable by construction: no tenant could serve it in any mode under any
    budget, so it cannot be a sane default for whichever tenant becomes
    resident.
    """
    admitting: list[str] = []
    refusing: list[str] = []
    for tenant_id, tenant in policy.tenants.items():
        for mode, mode_policy in tenant.modes.items():
            for budget in mode_policy.budgets:
                label = f"{tenant_id}/{mode}/{budget.name}"
                ceiling = np_ceiling(
                    policy,
                    tenant_id,
                    dynamic_budget_gib=budget.dynamic_budget_gib,
                    slot_context_tokens=slot_context_tokens,
                    mode=mode,
                )
                if ceiling is not None and np_slots <= ceiling:
                    admitting.append(label)
                else:
                    refusing.append(label)
    return admitting, refusing


def load_serving_shape(path: Path | None = None) -> dict[str, int]:
    """Load the lane's default serving shape (POLICY AS DATA; P0-1c).

    Reads ONLY the ``serving_shape`` block of the np_ceiling policy file and
    returns ``{"np_slots", "slot_context_tokens", "context_tokens"}`` (the
    last = np_slots * slot_context_tokens, the total ``-c`` value).

    Deliberately NOT feature-flag-gated (unlike ``load_np_ceiling_policy``):
    the stack-priors compiler must resolve the shape for a ``gpu_shadow_lane``
    launcher-tenant record during the activation Step-2 pipeline gates, which
    run before the flag is flipped (same rationale as the preflight probe's
    explicit Features override). It is still inert-by-construction: the only
    caller is the mode-gated ``gpu_shadow_lane`` branch of the priors
    compiler, and no launch-meta entry carries that mode today.

    Raises ValueError when the block is missing or invalid — callers must
    refuse (surface a gap), never fall back to CPU-mode serving defaults.

    P2-3d — THE SHAPE IS CHECKED AGAINST THE CEILING ROWS IN THE SAME FILE.
    Until 2026-07-28 this function validated only that ``np_slots`` was a
    measured np LEVEL and that the context was positive, and never consulted
    the ceiling table sitting a few lines below it in the same document. So
    ``np_slots: 32, slot_context_tokens: 32768`` was accepted and would have
    compiled straight into the builder's real ``-np``/``-c`` while every
    tenant's ceiling refused it. Two blocks in one file with no relation
    enforced between them is a drift hazard whose failure mode is a launch that
    LOOKS authorised and is not.

    Two conditions are enforced here, both chosen to be universally necessary —
    the loader cannot know which tenant will actually be resident (that is the
    registry's business at activation), so it must not assume one:

    1. ``np_slots`` may not exceed ANY tenant's measured throughput-saturation
       point. Past saturation a launch is slower AND more VRAM-hungry, so this
       is never right for any tenant.
    2. The (np, per-slot context) cell must be admissible for at least ONE
       tenant/mode/budget row. A shape admitted nowhere is unusable by
       construction and cannot be a sane default for whichever tenant lands.

    The tighter, program-specific check — that the shape works for every
    Phase-3 bake-off arm specifically — stays in Stage-0 smoke
    (``gpu_shadow_lane_stage0.py::smoke_checks``), which knows the tenancy
    table. The two layers are complementary: this one cannot be skipped because
    it is on the resolution path itself; that one is stricter but only runs
    when invoked.
    """
    payload = _read_policy_payload(path)
    lane = str(payload.get("lane", ""))
    if lane != LANE_NAME:
        raise ValueError(f"np_ceiling policy: lane {lane!r} != {LANE_NAME!r}")
    shape = _require_mapping(payload.get("serving_shape"), "serving_shape")
    np_slots = int(shape["np_slots"])
    slot_context = int(shape["slot_context_tokens"])
    if np_slots not in _VALID_NP_LEVELS:
        raise ValueError(
            f"serving_shape.np_slots {np_slots} is not a measured np level {_VALID_NP_LEVELS}"
        )
    if slot_context <= 0:
        raise ValueError("serving_shape.slot_context_tokens must be positive")

    # Parsed WITHOUT the feature gate on purpose: this function runs during the
    # Step-2 pipeline gates, before the flag is flipped. Parsing is not
    # consuming — the gate stays on load_np_ceiling_policy.
    policy = _parse_policy_payload(payload)

    oversaturated = sorted(
        f"{tenant_id} (saturation {tenant.np_throughput_saturation})"
        for tenant_id, tenant in policy.tenants.items()
        if np_slots > tenant.np_throughput_saturation
    )
    if oversaturated:
        raise ValueError(
            f"serving_shape.np_slots {np_slots} exceeds the measured throughput "
            f"saturation of: {', '.join(oversaturated)}. Past saturation a launch is "
            "slower AND uses more VRAM — refuse, never round up."
        )

    admitting, refusing = shape_admissibility(
        policy, np_slots=np_slots, slot_context_tokens=slot_context
    )
    if not admitting:
        raise ValueError(
            f"serving_shape -np {np_slots} x {slot_context} has no validated operating "
            f"point in this policy: refused by every tenant/mode/budget row "
            f"({', '.join(refusing)}). Refuse, never extrapolate — pick a shape the "
            "grids actually measured, or measure the cell first."
        )

    return {
        "np_slots": np_slots,
        "slot_context_tokens": slot_context,
        "context_tokens": np_slots * slot_context,
    }


def np_ceiling(
    policy: NpCeilingPolicy,
    tenant_id: str,
    *,
    dynamic_budget_gib: float,
    slot_context_tokens: int,
    mode: str = MODE_MTP_OFF,
) -> int | None:
    """Return the validated -np ceiling for a tenant+mode, or None (= refuse).

    Mode selection first: MTP on/off are different capacity frontiers (D6), so a
    mode the tenant has no rows for returns None. It NEVER falls back to the
    other mode — that fallback is exactly how an np16 x 32k launch validated
    only under MTP-off would get authorised under MTP-on.

    Row selection: the most permissive budget row whose dynamic_budget_gib does
    not exceed the budget actually available (rows are evidence "safe at >=
    this budget"). Bucket selection: the smallest measured context bucket that
    covers the requested per-slot context (conservative). None means "no
    validated operating point" — callers must refuse, never extrapolate.
    """
    if slot_context_tokens <= 0:
        raise ValueError("slot_context_tokens must be positive")
    if mode not in VALID_MODES:
        raise ValueError(f"unknown mode {mode!r} (valid: {VALID_MODES})")
    tenant = policy.tenants.get(tenant_id)
    if tenant is None:
        raise KeyError(f"unknown tenant {tenant_id!r}")
    mode_policy = tenant.mode_policy(mode)
    if mode_policy is None:
        return None
    eligible = [
        row for row in mode_policy.budgets if row.dynamic_budget_gib <= dynamic_budget_gib
    ]
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
    host_cpuset: str = LANE_HOST_CPUSET,
    host_threads: int = LANE_HOST_THREADS,
    device: str = LANE_DEVICE,
    binary: Path = LANE_BINARY,
    mtp: bool = False,
    draft_n_max: int | None = None,
    reasoning: bool = False,
) -> list[str]:
    """Informational argv for the lane, mirroring the measured grid shape.

    Matches the np_context_study_v8 server argv (production HIP binary,
    taskset 184-191, -t/-tb 8, -fa on, f16 KV, reasoning off). Every lane and
    tenant property is a PARAMETER, not a literal: P2-1's contract is that
    swapping tenant, mode, port or host slice is a data edit, and an argv
    builder with those values baked in would quietly break that contract.

    ``mtp`` is launch-bound (program decision D6 — ``params.speculative`` is
    global in v8, with no per-request override), so it appears here and
    nowhere else; toggling it is a drained-lane relaunch, never a hot mutation.
    The self-draft spelling is ``--spec-type draft-mtp --spec-draft-n-max N``,
    verbatim from the study driver that produced the grids
    (np_context_study_v8_20260727/driver/run_model_block.sh). ``draft_n_max``
    is REQUIRED when mtp is on: the depth is part of the measured identity
    (the FF MTP arm ran n_max=1, the A4 bridge n_max=4), so defaulting it would
    silently launch a shape no grid cell covers.

    Nothing in this repo executes this argv. It feeds the preflight report and
    the activation PROPOSAL; the operator runs the choreography.
    """
    if np_slots <= 0:
        raise ValueError("np_slots must be positive")
    if slot_context_tokens <= 0:
        raise ValueError("slot_context_tokens must be positive")
    if mtp and (draft_n_max is None or draft_n_max <= 0):
        raise ValueError("draft_n_max must be a positive int when mtp is enabled")
    total_ctx = np_slots * slot_context_tokens
    argv = [
        "taskset",
        "-c",
        host_cpuset,
        str(binary),
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
        device,
        "-ngl",
        "all",
        "-fa",
        "on",
        "-np",
        str(np_slots),
        "-c",
        str(total_ctx),
        "-t",
        str(host_threads),
        "-tb",
        str(host_threads),
        "-b",
        "2048",
        "-ub",
        "2048",
        "-ctk",
        "f16",
        "-ctv",
        "f16",
        "--reasoning",
        "on" if reasoning else "off",
    ]
    if mtp:
        # v8 self-draft (NEXTN/MTP), spelled exactly as the study driver did.
        # Deliberately the LAST argv group so a mode diff is visually obvious
        # in the proposal.
        argv.extend(["--spec-type", "draft-mtp", "--spec-draft-n-max", str(draft_n_max)])
    return argv
