"""One attribution case: a trajectory plus its gold (a consensus map) and what is known about it."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from src.trace.attribution.schema import AgentTrajectory


@dataclass
class AttributionCase:
    case_id: str
    trajectory: AgentTrajectory
    relevance: dict[int, float] = field(
        default_factory=dict
    )  # step_index -> consensus rate; {} = unlabelled
    gold_agents: set[str] = field(default_factory=set)
    reference: str | None = None  # reference answer; rendered only if the harness asks
    meta: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "trajectory": self.trajectory.to_dict(),
            "relevance": {str(k): v for k, v in sorted(self.relevance.items())},
            "gold_agents": sorted(self.gold_agents),
            "reference": self.reference,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AttributionCase":
        return cls(
            d["case_id"],
            AgentTrajectory.from_dict(d["trajectory"]),
            {int(k): float(v) for k, v in (d.get("relevance") or {}).items()},
            set(d.get("gold_agents") or ()),
            d.get("reference"),
            dict(d.get("meta") or {}),
        )


def consensus_from_labels(labels: list[list[int]]) -> dict[int, float]:
    """HIL-12 item (6): per-annotator flagged-step lists -> consensus rate per step (only flagged steps)."""
    if not labels:
        return {}
    out: dict[int, float] = {}
    for flagged in labels:
        for s in set(flagged):
            out[s] = out.get(s, 0.0) + 1.0
    return {s: v / len(labels) for s, v in out.items()}


def load_cases_jsonl(path: Path | str) -> Iterator[AttributionCase]:
    """Cases written by ``AttributionCase.as_dict`` (e.g. orchestrator traces + the human overlap labels)."""
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield AttributionCase.from_dict(json.loads(line))
