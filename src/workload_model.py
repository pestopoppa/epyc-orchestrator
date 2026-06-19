"""Workload traffic-class interface.

This module is intentionally read-only: it exposes the workload model declared
in ``orchestration/workload_model.yaml`` without changing request routing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKLOAD_MODEL = REPO_ROOT / "orchestration" / "workload_model.yaml"
VALID_TRAFFIC_CLASSES = {"interactive", "eval_batch", "campaign"}


@dataclass(frozen=True)
class TrafficClass:
    id: str
    display_name: str
    serving_class: str
    contention_priority: str
    queue_budget_ms_default: int | None


def load_workload_model(path: Path = DEFAULT_WORKLOAD_MODEL) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"workload model must be a mapping: {path}")
    return data


def load_traffic_classes(path: Path = DEFAULT_WORKLOAD_MODEL) -> dict[str, TrafficClass]:
    data = load_workload_model(path)
    rows = data.get("traffic_classes", [])
    if not isinstance(rows, list):
        raise ValueError("workload model traffic_classes must be a list")
    classes: dict[str, TrafficClass] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("traffic class rows must be mappings")
        class_id = str(row.get("id") or "").strip()
        if class_id not in VALID_TRAFFIC_CLASSES:
            raise ValueError(f"invalid traffic class id: {class_id!r}")
        latency_slo = row.get("latency_slo") if isinstance(row.get("latency_slo"), dict) else {}
        classes[class_id] = TrafficClass(
            id=class_id,
            display_name=str(row.get("display_name") or class_id),
            serving_class=str(row.get("serving_class") or ""),
            contention_priority=str(row.get("contention_priority") or ""),
            queue_budget_ms_default=latency_slo.get("queue_budget_ms_default"),
        )
    missing = VALID_TRAFFIC_CLASSES - set(classes)
    if missing:
        raise ValueError(f"missing traffic classes: {sorted(missing)}")
    return classes


def infer_workload_class(
    *,
    explicit: str | None = None,
    priority: str | None = None,
    source: str | None = None,
    batch_id: str | None = None,
    concurrency_batch_id: str | None = None,
    eval_batch_id: str | None = None,
    campaign_id: str | None = None,
) -> str:
    """Infer a workload traffic class from existing request metadata."""
    explicit_norm = str(explicit or "").strip().lower()
    if explicit_norm:
        if explicit_norm not in VALID_TRAFFIC_CLASSES:
            raise ValueError(f"invalid explicit workload_class: {explicit!r}")
        return explicit_norm

    source_norm = str(source or "").strip().lower()
    priority_norm = str(priority or "").strip().lower()
    if campaign_id or source_norm in {
        "autopilot_campaign",
        "kbrag",
        "nightshift",
        "research_campaign",
    }:
        return "campaign"
    if eval_batch_id or concurrency_batch_id or batch_id or priority_norm in {"batch", "background"}:
        return "eval_batch"
    if any(marker in source_norm for marker in ("eval", "seeding", "benchmark")):
        return "eval_batch"
    return "interactive"


def capture_workload_class(task_ir: Mapping[str, Any] | None) -> str:
    """Return the explicit workload class when present, else infer from legacy fields."""
    src = task_ir or {}
    priority = src.get("priority") if isinstance(src.get("priority"), str) else None
    source = src.get("source") or src.get("task_type")
    batch_id = src.get("batch_id") if isinstance(src.get("batch_id"), str) else None
    concurrency_batch_id = (
        src.get("concurrency_batch_id") if isinstance(src.get("concurrency_batch_id"), str) else None
    )
    eval_batch_id = src.get("eval_batch_id") if isinstance(src.get("eval_batch_id"), str) else None
    campaign_id = src.get("campaign_id") if isinstance(src.get("campaign_id"), str) else None
    return infer_workload_class(
        explicit=src.get("workload_class"),
        priority=priority,
        source=str(source) if source else None,
        batch_id=batch_id,
        concurrency_batch_id=concurrency_batch_id,
        eval_batch_id=eval_batch_id,
        campaign_id=campaign_id,
    )
