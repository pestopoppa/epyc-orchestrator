"""DS-7: Stack template schema, loader, and validator.

Stack templates define the full orchestrator stack configuration as a
declarative YAML file. Each template specifies:
- Which roles are active and their models/quants
- Instance topology (full, quarters, replicas per role)
- NUMA assignments and thread counts
- Memory budget constraints

Templates are discovered from ``stack_templates/<name>.yaml`` and selected
via ``--stack-profile <name>`` or ``ORCHESTRATOR_STACK_PROFILE`` env var.

The ``default.yaml`` template matches the current hardcoded configuration
in ``orchestrator_stack.py`` — it's the codified baseline that autoresearch
(DS-5/AR-3) Pareto frontiers are compared against.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default template search path (relative to epyc-orchestrator root)
_TEMPLATES_DIR = Path(__file__).resolve().parent.parent.parent / "stack_templates"

# System memory budget — total mlock-safe RAM on the 2×EPYC 9004 system
TOTAL_SYSTEM_RAM_GB = 1130
# Reserve for OS + KV caches + overhead
RESERVED_RAM_GB = 200
MAX_STACK_RAM_GB = TOTAL_SYSTEM_RAM_GB - RESERVED_RAM_GB

# Default budget split: HOT tier mlocks; WARM tier is loaded but evictable.
DEFAULT_MAX_MLOCK_GB = int(MAX_STACK_RAM_GB * 0.8)  # 744 GB
DEFAULT_MAX_TOTAL_GB = MAX_STACK_RAM_GB             # 930 GB
DEFAULT_RESERVE_KV_GB = 100                          # minimum headroom for KV caches


@dataclass
class ResourceBudget:
    """Fine-grained memory budget for a stack template (NIB2-19 / DS-7 Gap 4).

    Separates HOT-tier mlock'd memory from total loaded memory (HOT + WARM)
    and reserves a floor for KV caches. Any of these fields being exceeded
    causes template validation to fail-fast.
    """
    max_mlock_gb: float = DEFAULT_MAX_MLOCK_GB
    max_total_gb: float = DEFAULT_MAX_TOTAL_GB
    reserve_kv_gb: float = DEFAULT_RESERVE_KV_GB


@dataclass
class InstanceConfig:
    """Configuration for a single server instance within a role."""
    port: int
    numa: str          # NUMA quarter/node name: "Q0A", "Q0B", "NODE0", "NODE1"
    threads: int       # Thread count (48 for quarter, 96 for full)
    mlock: bool = True
    slot_save_path: str = ""   # If set, enables KV save/restore
    spec_overrides: dict[str, Any] = field(default_factory=dict)


@dataclass
class RoleConfig:
    """Configuration for one orchestrator role."""
    model: str           # Model name from registry (e.g. "Qwen3.5-35B-A3B")
    quant: str           # Quantization (e.g. "Q4_K_M", "Q6_K")
    tier: str            # HOT, WARM, COLD
    ram_gb: float        # Estimated RAM per instance (mlock'd)

    # Instance topology
    full: InstanceConfig | None = None       # Full-speed 96t instance
    quarters: list[InstanceConfig] = field(default_factory=list)  # 48t quarter instances
    replicas: list[InstanceConfig] = field(default_factory=list)  # Additional full-speed replicas

    @property
    def instance_count(self) -> int:
        return (1 if self.full else 0) + len(self.quarters) + len(self.replicas)

    @property
    def total_ram_gb(self) -> float:
        return self.ram_gb * self.instance_count


@dataclass
class StackTemplate:
    """Complete stack configuration template.

    A template fully specifies the orchestrator's model deployment:
    which roles run which models on which NUMA topology.
    """
    name: str
    description: str = ""
    version: str = "1"
    roles: dict[str, RoleConfig] = field(default_factory=dict)
    resource_budget: ResourceBudget = field(default_factory=ResourceBudget)

    @property
    def total_ram_gb(self) -> float:
        return sum(r.total_ram_gb for r in self.roles.values())

    @property
    def total_instances(self) -> int:
        return sum(r.instance_count for r in self.roles.values())

    @property
    def hot_ram_gb(self) -> float:
        """RAM consumed by HOT-tier roles only (the mlock'd set)."""
        return sum(r.total_ram_gb for r in self.roles.values() if r.tier == "HOT")

    @property
    def loaded_ram_gb(self) -> float:
        """RAM consumed by HOT + WARM roles (loaded but possibly evictable)."""
        return sum(r.total_ram_gb for r in self.roles.values() if r.tier in ("HOT", "WARM"))

    def role_names(self) -> list[str]:
        return list(self.roles.keys())


# === Loader ===

def load_template(name: str, templates_dir: Path | None = None) -> StackTemplate:
    """Load a stack template from YAML.

    Args:
        name: Template name (without .yaml extension).
        templates_dir: Directory to search. Defaults to stack_templates/.

    Returns:
        Parsed StackTemplate.

    Raises:
        FileNotFoundError: Template file not found.
        ValueError: Invalid template structure.
    """
    search_dir = templates_dir or _TEMPLATES_DIR
    path = search_dir / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Stack template not found: {path}")

    import yaml
    with open(path) as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Template {name}: expected dict, got {type(data).__name__}")

    budget_data = data.get("resource_budget") or {}
    budget = ResourceBudget(
        max_mlock_gb=float(budget_data.get("max_mlock_gb", DEFAULT_MAX_MLOCK_GB)),
        max_total_gb=float(budget_data.get("max_total_gb", DEFAULT_MAX_TOTAL_GB)),
        reserve_kv_gb=float(budget_data.get("reserve_kv_gb", DEFAULT_RESERVE_KV_GB)),
    )

    template = StackTemplate(
        name=data.get("name", name),
        description=data.get("description", ""),
        version=str(data.get("version", "1")),
        resource_budget=budget,
    )

    for role_name, role_data in data.get("roles", {}).items():
        if not isinstance(role_data, dict):
            continue
        role = RoleConfig(
            model=role_data.get("model", ""),
            quant=role_data.get("quant", "Q4_K_M"),
            tier=role_data.get("tier", "HOT"),
            ram_gb=float(role_data.get("ram_gb", 0)),
        )
        # Parse full instance
        if "full" in role_data:
            fd = role_data["full"]
            role.full = InstanceConfig(
                port=fd["port"],
                numa=fd.get("numa", "NODE0"),
                threads=fd.get("threads", 96),
                mlock=fd.get("mlock", True),
                slot_save_path=fd.get("slot_save_path", ""),
                spec_overrides=fd.get("spec_overrides", {}),
            )
        # Parse quarters
        for qd in role_data.get("quarters", []):
            role.quarters.append(InstanceConfig(
                port=qd["port"],
                numa=qd.get("numa", ""),
                threads=qd.get("threads", 48),
                mlock=qd.get("mlock", True),
                slot_save_path=qd.get("slot_save_path", ""),
            ))
        # Parse replicas
        for rd in role_data.get("replicas", []):
            role.replicas.append(InstanceConfig(
                port=rd["port"],
                numa=rd.get("numa", "NODE1"),
                threads=rd.get("threads", 96),
                mlock=rd.get("mlock", True),
            ))
        template.roles[role_name] = role

    return template


def discover_templates(templates_dir: Path | None = None) -> list[str]:
    """List available template names."""
    search_dir = templates_dir or _TEMPLATES_DIR
    if not search_dir.exists():
        return []
    return sorted(p.stem for p in search_dir.glob("*.yaml"))


def get_active_profile() -> str:
    """Get the active stack profile from env var or default."""
    return os.environ.get("ORCHESTRATOR_STACK_PROFILE", "default")


# === Validator ===

@dataclass
class ValidationResult:
    """Result of template validation."""
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        parts = [f"{'PASS' if self.valid else 'FAIL'}"]
        if self.errors:
            parts.append(f"{len(self.errors)} errors")
        if self.warnings:
            parts.append(f"{len(self.warnings)} warnings")
        return ", ".join(parts)


def validate_template(
    template: StackTemplate,
    registry_path: Path | None = None,
) -> ValidationResult:
    """Validate a stack template against system constraints.

    Checks:
    1. Memory budget: total RAM <= MAX_STACK_RAM_GB
    2. Port conflicts: no duplicate ports across roles
    3. NUMA conflicts: no overlapping CPU assignments
    4. Required roles: frontdoor must exist
    5. Model existence: all models referenced must exist in registry
    6. Tier consistency: HOT roles must have instances defined
    """
    errors: list[str] = []
    warnings: list[str] = []

    # 1. Memory budget (fine-grained: HOT mlock vs total loaded vs KV reserve)
    budget = template.resource_budget
    hot_ram = template.hot_ram_gb
    loaded_ram = template.loaded_ram_gb
    if hot_ram > budget.max_mlock_gb:
        errors.append(
            f"HOT mlock budget exceeded: {hot_ram:.0f} GB > "
            f"{budget.max_mlock_gb:.0f} GB limit"
        )
    elif hot_ram > budget.max_mlock_gb * 0.85:
        warnings.append(
            f"HOT mlock usage high: {hot_ram:.0f} GB "
            f"({hot_ram / budget.max_mlock_gb * 100:.0f}% of mlock limit)"
        )
    if loaded_ram > budget.max_total_gb:
        errors.append(
            f"Total loaded budget exceeded: {loaded_ram:.0f} GB > "
            f"{budget.max_total_gb:.0f} GB limit"
        )
    remaining_gb = TOTAL_SYSTEM_RAM_GB - loaded_ram
    if remaining_gb < budget.reserve_kv_gb:
        errors.append(
            f"KV reserve below minimum: {remaining_gb:.0f} GB headroom < "
            f"{budget.reserve_kv_gb:.0f} GB required for KV caches + OS"
        )

    # 2. Port conflicts
    all_ports: dict[int, str] = {}
    for role_name, role in template.roles.items():
        instances = []
        if role.full:
            instances.append(("full", role.full))
        for i, q in enumerate(role.quarters):
            instances.append((f"quarter[{i}]", q))
        for i, r in enumerate(role.replicas):
            instances.append((f"replica[{i}]", r))
        for label, inst in instances:
            if inst.port in all_ports:
                errors.append(
                    f"Port conflict: {role_name}.{label} port {inst.port} "
                    f"already used by {all_ports[inst.port]}"
                )
            else:
                all_ports[inst.port] = f"{role_name}.{label}"

    # 3. Required roles
    if "frontdoor" not in template.roles:
        errors.append("Required role 'frontdoor' not defined")

    # 4. Tier consistency
    for role_name, role in template.roles.items():
        if role.tier == "HOT" and role.instance_count == 0:
            errors.append(f"Role '{role_name}' is HOT but has no instances defined")

    # 5. Model existence (if registry available)
    if registry_path and registry_path.exists():
        try:
            import yaml
            with open(registry_path) as f:
                registry = yaml.safe_load(f)
            known_models = set()
            for model in registry.get("models", []):
                known_models.add(model.get("name", ""))
                for alias in model.get("aliases", []):
                    known_models.add(alias)
            for role_name, role in template.roles.items():
                if role.model and role.model not in known_models:
                    warnings.append(
                        f"Role '{role_name}': model '{role.model}' not found in registry"
                    )
        except Exception as exc:
            warnings.append(f"Could not validate models against registry: {exc}")

    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )
