#!/usr/bin/env python3
"""Reviewer control-plane pool generator (RM-1).

Reads the FULL research model registry
(`epyc-inference-research/orchestration/model_registry.yaml`) and emits the
candidate (architect x reviewer x grader) pairing list for the H5
model-role ablation tournament (`handoffs/active/reviewer-model-ablations.md`).

Design contract (RM-1):
  * Position pools are drawn by capability tier + candidate_roles, with
    configurable pruning rules (RAM fit, VRAM fit for GPU-resident candidates,
    t/s floor, quality floor, cross-family preference flag).
  * A floor only DROPS a model when the registry actually carries the relevant
    measured/baseline datum. Absent data -> the model is KEPT and flagged
    `*_measured: false` (never silently pruned on missing evidence).
  * Staged candidates and the production trio are force-included regardless of
    tier/floor pruning (they are the reason the tournament exists).
  * Anchor arms A0/A1/A3/A4 are ALWAYS present in the output, regardless of any
    pruning outcome (they are the guaranteed confirmation-tier baselines).
  * Output is deterministic: same registry bytes + same prune config => byte
    identical JSON. No wall-clock timestamps are embedded.
  * Provenance: registry path + registry file sha256 + echoed prune config +
    prune-config sha256.

This script performs NO inference and reads the registry read-only.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

SCHEMA_VERSION = "1"

# --------------------------------------------------------------------------- #
# Defaults grounded in the live stack (2026-07-17).
# --------------------------------------------------------------------------- #
DEFAULT_REGISTRY = (
    "/mnt/raid0/llm/epyc-inference-research/orchestration/model_registry.yaml"
)

# Host facts (see CLAUDE.md / user_hardware memory): EPYC 9655, 1.1TB RAM,
# single MI210 = 64GB HBM.
DEFAULT_HOST_RAM_GB = 1100.0
DEFAULT_VRAM_GB = 64.0
DEFAULT_CORESIDENCY_FRACTION = 0.85  # fraction of host RAM usable for weights

DEFAULT_TS_FLOOR = 3.0       # t/s; below this a MEASURED model is dropped
DEFAULT_QUALITY_FLOOR = 0.60  # fraction 0-1; below this a MEASURED model is dropped
DEFAULT_GRADER_MAX_SIZE_GB = 45.0  # rubric grading is cheap-model work

# Staged candidates named in RM-1 / RM-2 (registry keys) + the production trio.
DEFAULT_STAGED_KEYS = [
    "glm_52_ud_iq2m",                    # GLM-5.2 UD-IQ2_M
    "hy3_angelslim_iq1m_mtp",            # Hy3 (Hunyuan-v3) IQ1_M + MTP
    "minimax_m27_q8",                    # MiniMax-M2.7 Q8_0
    "deepseek_v4_flash_local_q4kexperts",  # DeepSeek-V4-Flash
    "ingest_long_context",               # Qwen3-Next-80B-A3B-Instruct
    "gemma4_31b_q4km_mtp",               # gemma-4-31B-it
    "qwen36_27b_q8",                     # Qwen3.6-27B
]
DEFAULT_PRODUCTION_TRIO = ["frontdoor", "worker_general", "toolrunner"]

DEFAULT_PRODUCTION_ARCHITECT = "architect_general"  # Qwen3.5-122B-A10B (warm)
DEFAULT_GRADER = "toolrunner"                         # cheap rubric grader

# Position eligibility by candidate_roles tag.
ARCHITECT_ROLES = {
    "architect", "architect_general", "frontdoor", "general",
    "reasoning", "thinking", "standalone_reasoner", "cheap_thinking",
}
REVIEWER_ROLES = ARCHITECT_ROLES | {
    "worker", "coder", "toolrunner", "q_scorer", "verifier_selector", "math",
}
GRADER_ROLES = {
    "worker", "general", "toolrunner", "coder", "q_scorer",
    "cheap_thinking", "try_cheap_first", "verifier_selector",
}

# candidate_roles / architectures that mark an entry as NOT a general-purpose
# reasoning candidate (excluded wholesale: embedders / vision / draft-only).
EMBEDDER_ROLES = {"embedder"}
VISION_ROLES = {
    "vision", "multimodal", "audio", "tts", "agentic_vision",
    "figure_analysis", "vision_escalation",
}
VISION_ARCHES = {"qwen3vl", "qwen3vlmoe", "qwen3_vl", "minicpm_o", "bert"}
EMBEDDER_ARCHES = {"bert"}

# Architecture -> model family (collusion axis). None => fall through to name.
ARCH_FAMILY = {
    "qwen35": "qwen", "qwen35moe": "qwen", "qwen3": "qwen",
    "qwen25_coder": "qwen", "ssm_hybrid": "qwen", "ssm_moe_hybrid": "qwen",
    "qwen3vl": "qwen", "qwen3vlmoe": "qwen", "qwen3_vl": "qwen",
    "gemma4": "gemma", "gemma4-assistant": "gemma",
    "glm4moe": "glm", "glm_moe_dsa": "glm",
    "deepseek2": "deepseek", "deepseek_v4_flash": "deepseek",
    "minimax-m2": "minimax",
    "hy_v3": "hunyuan",
    "nemotron_h": "nemotron", "nemotron_h_moe": "nemotron",
    "nemotron_labs_diffusion": "nemotron",
    "bailingmoe-linear": "bailing",
    "minicpm_o": "minicpm",
    "bert": "bert",
}
# Ordered name-keyword fallback (first match wins).
NAME_FAMILY = [
    ("glm", "glm"), ("deepseek", "deepseek"), ("minimax", "minimax"),
    ("hunyuan", "hunyuan"), ("hy3", "hunyuan"),
    ("bonsai", "qwen"),  # Bonsai/Ternary-Bonsai are Qwen3.6-27B derivatives
    ("qwen", "qwen"), ("gemma", "gemma"), ("hermes", "hermes"),
    ("llama", "llama"), ("nemotron", "nemotron"),
    ("ring", "bailing"), ("ling", "bailing"),
    ("mistral", "mistral"), ("phi", "phi"), ("minicpm", "minicpm"),
]
AMBIGUOUS_ARCHES = {"dense", "moe", "moe_hybrid", "none", "unknown",
                    "external_api", None}


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class PruneConfig:
    host_ram_gb: float = DEFAULT_HOST_RAM_GB
    vram_gb: float = DEFAULT_VRAM_GB
    coresidency_fraction: float = DEFAULT_CORESIDENCY_FRACTION
    ts_floor: float = DEFAULT_TS_FLOOR
    quality_floor: float = DEFAULT_QUALITY_FLOOR
    grader_max_size_gb: float = DEFAULT_GRADER_MAX_SIZE_GB
    sequential_swap: bool = False       # co-resident (sum) vs sequential (max)
    require_cross_family: bool = False  # if True, same-family pairs are dropped
    grader_sweep: bool = False          # if True, sweep the full grader pool
    max_pairings: int = 0               # 0 = unlimited
    production_architect: str = DEFAULT_PRODUCTION_ARCHITECT
    default_grader: str = DEFAULT_GRADER
    staged_keys: Tuple[str, ...] = tuple(DEFAULT_STAGED_KEYS)
    production_trio: Tuple[str, ...] = tuple(DEFAULT_PRODUCTION_TRIO)
    extra_include: Tuple[str, ...] = ()

    def force_include(self) -> List[str]:
        seen: Dict[str, None] = {}
        for k in (list(self.staged_keys) + list(self.production_trio)
                  + list(self.extra_include)):
            seen[k] = None
        return sorted(seen)

    def canonical(self) -> Dict[str, Any]:
        """Decision-affecting knobs only (I/O paths excluded), sorted."""
        d = dataclasses.asdict(self)
        d["staged_keys"] = sorted(self.staged_keys)
        d["production_trio"] = sorted(self.production_trio)
        d["extra_include"] = sorted(self.extra_include)
        d["_anchor_arms"] = anchor_arm_specs(self)
        return d

    def sha256(self) -> str:
        blob = json.dumps(self.canonical(), sort_keys=True,
                          separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()


def anchor_arm_specs(cfg: PruneConfig) -> List[Dict[str, Any]]:
    """Guaranteed confirmation-tier anchor arms (RM-2). Always emitted."""
    arch = cfg.production_architect
    grader = cfg.default_grader
    return [
        {"arm_id": "A0",
         "label": "gates-only (objective-verifier floor)",
         "architect": arch, "reviewer": None, "grader": None},
        {"arm_id": "A1",
         "label": "self-review (status-quo alias)",
         "architect": arch, "reviewer": arch, "grader": grader},
        {"arm_id": "A3",
         "label": "same-family GPU heavyweight (122B, grammar mandatory)",
         "architect": arch, "reviewer": "qwen35_122b_q4km", "grader": grader,
         "gpu_resident": True, "grammar": "mandatory"},
        {"arm_id": "A4",
         "label": "cross-family GLM-5.2-IQ2 CPU (target)",
         "architect": arch, "reviewer": "glm_52_ud_iq2m", "grader": grader,
         "cpu_resident": True},
    ]


# --------------------------------------------------------------------------- #
# Registry parsing helpers
# --------------------------------------------------------------------------- #
def load_registry(path: str) -> Tuple[Dict[str, Any], str]:
    raw = Path(path).read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    data = yaml.safe_load(raw)
    return data, sha


def registry_roles(data: Dict[str, Any]) -> Dict[str, Any]:
    roles = data.get("roles") or {}
    return {k: v for k, v in roles.items() if isinstance(v, dict)}


def _model(entry: Dict[str, Any]) -> Dict[str, Any]:
    return entry.get("model") or {}


def infer_family(entry: Dict[str, Any]) -> str:
    m = _model(entry)
    arch = m.get("architecture")
    fam = ARCH_FAMILY.get(arch)
    if fam:
        return fam
    name = str(m.get("name") or "").lower()
    for sub, f in NAME_FAMILY:
        if sub in name:
            return f
    if arch and arch not in AMBIGUOUS_ARCHES:
        return str(arch)
    return "other"


def get_size_gb(entry: Dict[str, Any]) -> Optional[float]:
    v = _model(entry).get("size_gb")
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def get_tps(entry: Dict[str, Any]) -> Optional[float]:
    """Best measured decode t/s across known perf keys, else None."""
    perf = entry.get("performance") or {}
    best: Optional[float] = None
    for key in ("optimized_tps", "baseline_tps", "mtp_tps",
                "full_spec_tps", "dual_aggregate_tps"):
        v = perf.get(key)
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if best is None or f > best:
            best = f
    return best


_PCT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_FRAC_RE = re.compile(r"(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)")


def get_quality_fraction(entry: Dict[str, Any]) -> Optional[float]:
    """Quality as a 0-1 fraction from measured bench fields, else None."""
    perf = entry.get("performance") or {}
    qp = perf.get("quality_pct")
    try:
        if qp is not None:
            return max(0.0, min(1.0, float(qp) / 100.0))
    except (TypeError, ValueError):
        pass
    qs = perf.get("quality_score")
    if isinstance(qs, (int, float)):
        f = float(qs)
        if f <= 1.0:
            return f
        if f <= 3.0:      # e.g. 2.57 (out of 3)
            return f / 3.0
        return f / 100.0  # assume percentage
    if isinstance(qs, str):
        mp = _PCT_RE.search(qs)
        if mp:
            return max(0.0, min(1.0, float(mp.group(1)) / 100.0))
        mf = _FRAC_RE.search(qs)
        if mf and float(mf.group(2)) > 0:
            return max(0.0, min(1.0, float(mf.group(1)) / float(mf.group(2))))
    return None


def is_deprecated(entry: Dict[str, Any]) -> bool:
    return bool(entry.get("deprecated")) or entry.get("tier") == "X"


def is_external(entry: Dict[str, Any]) -> bool:
    arch = _model(entry).get("architecture")
    backend = (entry.get("backend") or {}).get("type")
    return arch == "external_api" or backend == "external_api"


def exclusion_reason(key: str, entry: Dict[str, Any]) -> Optional[str]:
    """Return a reason string if this entry is a hard-exclude
    (embedder / vision / draft-only), else None."""
    m = _model(entry)
    arch = str(m.get("architecture") or "")
    name = str(m.get("name") or "").lower()
    croles = set(entry.get("candidate_roles") or [])

    if arch in EMBEDDER_ARCHES or croles & EMBEDDER_ROLES or "embed" in key:
        return "embedder"
    if (arch in VISION_ARCHES or arch.endswith("-assistant")
            or croles & VISION_ROLES or "multimodal" in name):
        return "vision_or_multimodal"
    # draft-only: key prefix, assistant arch, external-draft accel, or the
    # entry's ONLY meaningful role is drafting.
    accel_type = (entry.get("acceleration") or {}).get("type")
    non_draft_roles = croles - {"draft", "benchmark"}
    if (key.startswith("draft")
            or accel_type == "external_draft"
            or arch.endswith("assistant")
            or ("draft" in croles and not non_draft_roles)):
        return "draft_only"
    return None


# --------------------------------------------------------------------------- #
# Model card + per-model pruning
# --------------------------------------------------------------------------- #
def build_card(key: str, entry: Dict[str, Any], cfg: PruneConfig,
               forced: bool) -> Dict[str, Any]:
    m = _model(entry)
    size = get_size_gb(entry)
    tps = get_tps(entry)
    quality = get_quality_fraction(entry)
    external = is_external(entry)
    mem = entry.get("memory") or {}

    gpu_ok: Optional[bool] = None
    if size is not None and not external:
        gpu_ok = size <= cfg.vram_gb

    card = {
        "key": key,
        "name": m.get("name"),
        "family": infer_family(entry),
        "arch": m.get("architecture"),
        "tier": entry.get("tier"),
        "quant": m.get("quant"),
        "size_gb": size,
        "tps": tps,
        "tps_measured": tps is not None,
        "quality": round(quality, 4) if quality is not None else None,
        "quality_measured": quality is not None,
        "gpu_resident_ok": gpu_ok,
        "residency": mem.get("residency"),
        "external": external,
        "staged": key in cfg.staged_keys,
        "production": key in cfg.production_trio,
        "forced": forced,
    }
    return card


def floor_prune(card: Dict[str, Any], cfg: PruneConfig) -> List[str]:
    """Return the list of floor-drop reasons (empty => kept). Never applied to
    forced (staged / production) entries."""
    reasons: List[str] = []
    size = card["size_gb"]
    if size is not None and not card["external"]:
        if size > cfg.host_ram_gb:
            reasons.append(f"ram_exceeds_host({size}>{cfg.host_ram_gb})")
        # GPU-exclusive candidates that overflow HBM cannot be GPU-resident.
        if card["gpu_resident_ok"] is False and card["residency"] == "gpu_only":
            reasons.append(f"vram_exceeds({size}>{cfg.vram_gb})")
    if card["tps_measured"] and card["tps"] < cfg.ts_floor:
        reasons.append(f"below_ts_floor({card['tps']}<{cfg.ts_floor})")
    if card["quality_measured"] and card["quality"] < cfg.quality_floor:
        reasons.append(
            f"below_quality_floor({card['quality']}<{cfg.quality_floor})")
    return reasons


# --------------------------------------------------------------------------- #
# Pool construction
# --------------------------------------------------------------------------- #
def eligible_positions(entry: Dict[str, Any], cfg: PruneConfig) -> List[str]:
    croles = set(entry.get("candidate_roles") or [])
    size = get_size_gb(entry)
    positions: List[str] = []
    if croles & ARCHITECT_ROLES:
        positions.append("architect")
    if croles & REVIEWER_ROLES:
        positions.append("reviewer")
    if croles & GRADER_ROLES and (
            size is None or size <= cfg.grader_max_size_gb):
        positions.append("grader")
    return positions


def build_pools(roles: Dict[str, Any], cfg: PruneConfig) -> Dict[str, Any]:
    force = set(cfg.force_include())
    pools: Dict[str, List[Dict[str, Any]]] = {
        "architect": [], "reviewer": [], "grader": []}
    dropped: List[Dict[str, Any]] = []
    excluded_counts: Dict[str, int] = {}

    for key in sorted(roles):
        entry = roles[key]
        forced = key in force

        excl = exclusion_reason(key, entry)
        if excl and not forced:
            excluded_counts[excl] = excluded_counts.get(excl, 0) + 1
            continue

        if is_deprecated(entry) and not forced:
            excluded_counts["deprecated"] = (
                excluded_counts.get("deprecated", 0) + 1)
            continue

        positions = eligible_positions(entry, cfg)
        # Forced entries are guaranteed a slot even if candidate_roles are thin:
        # staged architect/reviewer candidates always join both reasoning pools.
        if forced and not positions:
            positions = ["architect", "reviewer"]
        if not positions:
            continue

        card = build_card(key, entry, cfg, forced)
        reasons = [] if forced else floor_prune(card, cfg)
        if reasons:
            dropped.append({"key": key, "positions": positions,
                            "reasons": reasons})
            continue

        # Forced staged candidates are architect+reviewer candidates by intent;
        # ensure they land in those pools even if their thin candidate_roles
        # only tagged one position.
        if forced:
            for p in ("architect", "reviewer"):
                if p not in positions:
                    positions.append(p)

        card_positions = sorted(set(positions))
        card = dict(card, positions=card_positions)
        for p in card_positions:
            pools[p].append(card)

    for p in pools:
        pools[p].sort(key=lambda c: c["key"])
    return {"pools": pools, "dropped": sorted(dropped, key=lambda d: d["key"]),
            "excluded_counts": excluded_counts}


# --------------------------------------------------------------------------- #
# Anchor resolution
# --------------------------------------------------------------------------- #
def resolve_anchor(role_key: Optional[str], roles: Dict[str, Any],
                   cfg: PruneConfig) -> Optional[Dict[str, Any]]:
    if role_key is None:
        return None
    entry = roles.get(role_key)
    if entry is None:
        return {"key": role_key, "resolved": False}
    card = build_card(role_key, entry, cfg, forced=True)
    card["resolved"] = True
    return card


def build_anchors(roles: Dict[str, Any], cfg: PruneConfig) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for spec in anchor_arm_specs(cfg):
        arch_card = resolve_anchor(spec["architect"], roles, cfg)
        rev_card = resolve_anchor(spec.get("reviewer"), roles, cfg)
        grd_card = resolve_anchor(spec.get("grader"), roles, cfg)
        arm = {
            "arm_id": spec["arm_id"],
            "label": spec["label"],
            "architect": arch_card,
            "reviewer": rev_card,
            "grader": grd_card,
            "anchor": True,
        }
        for extra in ("gpu_resident", "grammar", "cpu_resident"):
            if extra in spec:
                arm[extra] = spec[extra]
        cross = None
        if arch_card and rev_card and rev_card.get("resolved"):
            cross = arch_card.get("family") != rev_card.get("family")
        arm["cross_family_preferred"] = cross
        out.append(arm)
    return out


# --------------------------------------------------------------------------- #
# Pairing generation
# --------------------------------------------------------------------------- #
def _coresidency(a: Dict[str, Any], r: Optional[Dict[str, Any]],
                 g: Optional[Dict[str, Any]], cfg: PruneConfig) -> Dict[str, Any]:
    sizes = [c["size_gb"] for c in (a, r, g)
             if c and c.get("size_gb") is not None and not c.get("external")]
    budget = cfg.host_ram_gb * cfg.coresidency_fraction
    if cfg.sequential_swap:
        footprint = max(sizes) if sizes else 0.0
        mode = "sequential"
    else:
        footprint = sum(sizes) if sizes else 0.0
        mode = "coresident"
    return {
        "mode": mode,
        "footprint_gb": round(footprint, 2),
        "budget_gb": round(budget, 2),
        "fits": footprint <= budget,
    }


def build_pairings(pools: Dict[str, List[Dict[str, Any]]],
                   roles: Dict[str, Any], cfg: PruneConfig) -> Dict[str, Any]:
    architects = pools["architect"]
    reviewers = pools["reviewer"]
    if cfg.grader_sweep:
        graders: List[Optional[Dict[str, Any]]] = list(pools["grader"])
    else:
        gentry = roles.get(cfg.default_grader)
        graders = [build_card(cfg.default_grader, gentry, cfg, forced=True)
                   if gentry is not None
                   else {"key": cfg.default_grader, "resolved": False,
                         "size_gb": None, "family": None, "external": False}]

    # Map generated triples that coincide with an anchor arm.
    anchor_lookup: Dict[Tuple[Any, Any, Any], str] = {}
    for spec in anchor_arm_specs(cfg):
        anchor_lookup[(spec["architect"], spec.get("reviewer"),
                       spec.get("grader"))] = spec["arm_id"]

    pairings: List[Dict[str, Any]] = []
    dropped = 0
    for a in architects:
        for r in reviewers:
            same_family = a["family"] == r["family"]
            if cfg.require_cross_family and same_family:
                dropped += 1
                continue
            for g in graders:
                cores = _coresidency(a, r, g, cfg)
                if not cores["fits"] and not cfg.sequential_swap:
                    dropped += 1
                    continue
                gk = g["key"] if g else None
                pid = f"{a['key']}__{r['key']}__{gk}"
                pairings.append({
                    "pairing_id": pid,
                    "architect": a["key"],
                    "reviewer": r["key"],
                    "grader": gk,
                    "architect_family": a["family"],
                    "reviewer_family": r["family"],
                    "cross_family_preferred": not same_family,
                    "self_review": a["key"] == r["key"],
                    "coresidency": cores,
                    "staged_involved": bool(
                        a.get("staged") or r.get("staged")
                        or (g and g.get("staged"))),
                    "anchor_arm": anchor_lookup.get(
                        (a["key"], r["key"], gk)),
                })

    pairings.sort(key=lambda p: p["pairing_id"])
    truncated = False
    if cfg.max_pairings and len(pairings) > cfg.max_pairings:
        pairings = pairings[:cfg.max_pairings]
        truncated = True
    return {"pairings": pairings, "dropped_pairings": dropped,
            "truncated": truncated}


# --------------------------------------------------------------------------- #
# Top-level assembly
# --------------------------------------------------------------------------- #
def build_output(data: Dict[str, Any], cfg: PruneConfig, registry_path: str,
                 registry_sha256: str, pools_only: bool = False) -> Dict[str, Any]:
    roles = registry_roles(data)
    pool_result = build_pools(roles, cfg)
    pools = pool_result["pools"]
    anchors = build_anchors(roles, cfg)

    out: Dict[str, Any] = {
        "provenance": {
            "generator": "reviewer_pool_gen.py",
            "schema_version": SCHEMA_VERSION,
            "registry_path": registry_path,
            "registry_sha256": registry_sha256,
            "prune_config": cfg.canonical(),
            "prune_config_sha256": cfg.sha256(),
            "n_roles_scanned": len(roles),
            "excluded_counts": pool_result["excluded_counts"],
            "pool_sizes": {p: len(pools[p]) for p in pools},
            "n_floor_dropped": len(pool_result["dropped"]),
        },
        "pools": pools,
        "floor_dropped": pool_result["dropped"],
        "anchor_arms": anchors,
    }
    if not pools_only:
        pair_result = build_pairings(pools, roles, cfg)
        out["pairings"] = pair_result["pairings"]
        out["provenance"]["n_pairings"] = len(pair_result["pairings"])
        out["provenance"]["n_pairings_dropped"] = pair_result["dropped_pairings"]
        out["provenance"]["pairings_truncated"] = pair_result["truncated"]
    out["provenance"]["n_anchor_arms"] = len(anchors)
    return out


def dumps(out: Dict[str, Any]) -> str:
    """Deterministic serialization (sorted keys, no timestamps)."""
    return json.dumps(out, sort_keys=True, indent=2)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--registry", default=DEFAULT_REGISTRY)
    p.add_argument("--output", default=None, help="output path (default stdout)")
    p.add_argument("--pools-only", action="store_true",
                   help="emit pools + anchors only (skip pairing cross-product)")
    p.add_argument("--host-ram-gb", type=float, default=DEFAULT_HOST_RAM_GB)
    p.add_argument("--vram-gb", type=float, default=DEFAULT_VRAM_GB)
    p.add_argument("--coresidency-fraction", type=float,
                   default=DEFAULT_CORESIDENCY_FRACTION)
    p.add_argument("--ts-floor", type=float, default=DEFAULT_TS_FLOOR)
    p.add_argument("--quality-floor", type=float, default=DEFAULT_QUALITY_FLOOR)
    p.add_argument("--grader-max-size-gb", type=float,
                   default=DEFAULT_GRADER_MAX_SIZE_GB)
    p.add_argument("--sequential-swap", action="store_true",
                   help="model footprint = max (swap) instead of sum (co-resident)")
    p.add_argument("--require-cross-family", action="store_true",
                   help="drop same-family architect/reviewer pairings")
    p.add_argument("--grader-sweep", action="store_true",
                   help="sweep the full grader pool (default: single grader)")
    p.add_argument("--max-pairings", type=int, default=0,
                   help="cap emitted pairings (0 = unlimited)")
    p.add_argument("--production-architect", default=DEFAULT_PRODUCTION_ARCHITECT)
    p.add_argument("--default-grader", default=DEFAULT_GRADER)
    p.add_argument("--include-key", action="append", default=[],
                   help="extra force-include registry key (repeatable)")
    return p.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> PruneConfig:
    return PruneConfig(
        host_ram_gb=args.host_ram_gb,
        vram_gb=args.vram_gb,
        coresidency_fraction=args.coresidency_fraction,
        ts_floor=args.ts_floor,
        quality_floor=args.quality_floor,
        grader_max_size_gb=args.grader_max_size_gb,
        sequential_swap=args.sequential_swap,
        require_cross_family=args.require_cross_family,
        grader_sweep=args.grader_sweep,
        max_pairings=args.max_pairings,
        production_architect=args.production_architect,
        default_grader=args.default_grader,
        extra_include=tuple(args.include_key),
    )


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    cfg = config_from_args(args)
    data, sha = load_registry(args.registry)
    out = build_output(data, cfg, args.registry, sha, pools_only=args.pools_only)
    text = dumps(out)
    if args.output:
        Path(args.output).write_text(text + "\n")
    else:
        sys.stdout.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
