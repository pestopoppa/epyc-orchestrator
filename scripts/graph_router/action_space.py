"""GraphRouter action-space helpers.

Live routing targets come from the generated stack-priors contract. Legacy
episodic labels are normalized here so old replay data can be reused without
leaking retired roles into newly generated classifier/verifier artifacts.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STACK_PRIORS_PATH = PROJECT_ROOT / "orchestration/derived/stack_priors.yaml"

logger = logging.getLogger(__name__)

LEGACY_ARCHITECT_CODING = "architect" + "_coding"
LEGACY_WORKER_EXPLORE = "worker" + "_explore"

# Stable display/training order for known live roles. The role set is still
# loaded from stack_priors.yaml; this only prevents harmless YAML ordering
# changes from shifting historical frontdoor conventions.
PREFERRED_ACTION_ORDER = [
    "frontdoor",
    "architect_general",
    "coder_escalation",
    "worker_general",
    "worker_math",
    "worker_vision",
    "ingest_long_context",
    "worker_summarize",
    "toolrunner",
    "vision_escalation",
]

DEGRADED_CANONICAL_ACTIONS = PREFERRED_ACTION_ORDER.copy()

SEED_FRONTDOOR_ACTIONS = {
    "frontdoor:repl",
    "frontdoor:direct",
    "frontdoor:react",
}

ACTION_EXCLUDE: set[str] = {
    "",
    *SEED_FRONTDOOR_ACTIONS,
}
ACTION_EXCLUDE_PREFIXES = ("persona:",)

RAW_TO_LIVE_ACTION: dict[str, str] = {
    "frontdoor": "frontdoor",
    "architect_general": "architect_general",
    LEGACY_ARCHITECT_CODING: "architect_general",
    "coder_escalation": "coder_escalation",
    "ingest_long_context": "ingest_long_context",
    LEGACY_WORKER_EXPLORE: "worker_general",
    "worker_general": "worker_general",
    "worker_math": "worker_math",
    "worker_vision": "worker_vision",
    "worker_summarize": "worker_summarize",
    "toolrunner": "toolrunner",
    "vision_escalation": "vision_escalation",
    "coder": "coder_escalation",
    "escalate:frontdoor->coder_escalation": "coder_escalation",
    "escalate:worker_general->coder_escalation": "coder_escalation",
    f"escalate:coder_escalation->{LEGACY_ARCHITECT_CODING}": "architect_general",
    "SELF": "frontdoor",
    "SELF:direct": "frontdoor",
    "SELF:repl": "frontdoor",
    "ARCHITECT": "architect_general",
    "WORKER": "worker_general",
}


def _ordered_live_roles(roles: Mapping[str, Any]) -> list[str]:
    live: list[str] = []
    for role_id, record in roles.items():
        if not isinstance(record, Mapping):
            continue
        if record.get("deployment_status") != "live_stack":
            continue
        role = str(record.get("role") or role_id)
        live.append(role)

    if not live:
        return []

    live_set = set(live)
    ordered = [role for role in PREFERRED_ACTION_ORDER if role in live_set]
    ordered.extend(role for role in live if role not in set(ordered))
    return ordered


def load_live_canonical_actions(
    stack_priors_path: Path = DEFAULT_STACK_PRIORS_PATH,
) -> list[str]:
    """Load live routing action labels from stack_priors.yaml.

    Falls back only as an explicit degraded/offline mode when the generated
    contract cannot be read or contains no live roles.
    """
    try:
        import yaml

        data = yaml.safe_load(stack_priors_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        logger.warning("Using degraded GraphRouter actions; stack priors unavailable: %s", exc)
        return DEGRADED_CANONICAL_ACTIONS.copy()

    roles = data.get("roles", {})
    if not isinstance(roles, Mapping):
        logger.warning("Using degraded GraphRouter actions; stack priors roles field is invalid")
        return DEGRADED_CANONICAL_ACTIONS.copy()

    actions = _ordered_live_roles(roles)
    if not actions:
        logger.warning("Using degraded GraphRouter actions; no live stack roles found")
        return DEGRADED_CANONICAL_ACTIONS.copy()
    return actions


def normalize_action(
    raw_action: str,
    *,
    include_seeded_frontdoor: bool = False,
) -> str | None:
    """Map a raw episodic-memory action label to a current live routing target."""
    if raw_action in SEED_FRONTDOOR_ACTIONS and include_seeded_frontdoor:
        return "frontdoor"
    if raw_action in ACTION_EXCLUDE:
        return None
    if any(raw_action.startswith(prefix) for prefix in ACTION_EXCLUDE_PREFIXES):
        return None
    if "\n" in raw_action or "FINAL(" in raw_action:
        return None

    canonical = RAW_TO_LIVE_ACTION.get(raw_action)
    if canonical is None:
        logger.warning("Unknown action '%s' - excluding from training", raw_action[:80])
    return canonical


def canonical_actions_from_label_map(label_map_raw: Any) -> list[str]:
    """Recover canonical action order from a saved classifier label_map array."""
    entries: list[tuple[int, str]] = []
    for row in label_map_raw:
        entries.append((int(row[0]), str(row[1])))
    if not entries:
        return []
    max_idx = max(idx for idx, _ in entries)
    actions = ["" for _ in range(max_idx + 1)]
    for idx, action in entries:
        actions[idx] = action
    return actions


def canonical_actions_from_npz(src: Any) -> list[str]:
    """Recover action order from classifier data saved by extract_training_data."""
    files = set(getattr(src, "files", []))
    if "label_map" in files:
        actions = canonical_actions_from_label_map(src["label_map"])
        if actions:
            return actions
    if "canonical_actions" in files:
        return [str(action) for action in src["canonical_actions"].tolist()]
    return []


def infer_n_actions(src: Any, y: Any) -> int:
    """Infer one-hot action width from classifier artifacts, falling back to y."""
    actions = canonical_actions_from_npz(src)
    if actions:
        return len(actions)
    return int(y.max()) + 1 if len(y) else 0
