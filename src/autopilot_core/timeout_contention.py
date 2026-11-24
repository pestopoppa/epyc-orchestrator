"""HF-TIMEOUT-CLASS: split typed eval timeouts into contention vs model_slow.

Label only. This module never decides which rows leave a quality denominator;
that is `measurement_guards.is_quality_scorable` and a measurement-policy
decision. Callers gate on ``features().timeout_contention_class``.

Inputs are the host covariates the eval tower already records per question
(`host_health.host_timing_covariates`, compacted in eval_tower):

* ``host_inflight`` - decoding instances holding a CPU-region lock when the row
  finished. The timed-out request's own decode counts as at most 1, so a value of
  2 or more means another decode was competing for the host.
* ``loadavg_per_core`` - 1-minute loadavg divided by online cores.

Thresholds (documented, deliberately conservative; middle values stay unknown):

* ``contention``: host_inflight >= 2, or loadavg_per_core >= 1.0
* ``model_slow``: host_inflight known and <= 1, AND loadavg_per_core known and < 0.5
  (the host looked idle, so the model itself was slow)
* ``unknown``: anything else, including missing covariates
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

CLASS_CONTENTION = "contention"
CLASS_MODEL_SLOW = "model_slow"
CLASS_UNKNOWN = "unknown"
CONTENTION_CLASSES = (CLASS_CONTENTION, CLASS_MODEL_SLOW, CLASS_UNKNOWN)

INFLIGHT_CONTENTION_MIN = 2
LOADAVG_PER_CORE_CONTENTION_MIN = 1.0
INFLIGHT_IDLE_MAX = 1
LOADAVG_PER_CORE_IDLE_MAX = 0.5  # strictly below

#: `failure_provenance["class"]` values that are timeouts.
TIMEOUT_PROVENANCE_CLASSES = frozenset(
    {"client_transport_timeout", "admission_timeout", "backend_timeout", "slot_erase_timeout"}
)


def _num(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def classify_contention(covariates: Mapping[str, Any] | None) -> str:
    cov = covariates or {}
    inflight = _num(cov.get("host_inflight"))
    load = _num(cov.get("loadavg_per_core"))
    if (inflight is not None and inflight >= INFLIGHT_CONTENTION_MIN) or (
        load is not None and load >= LOADAVG_PER_CORE_CONTENTION_MIN
    ):
        return CLASS_CONTENTION
    if (
        inflight is not None
        and inflight <= INFLIGHT_IDLE_MAX
        and load is not None
        and load < LOADAVG_PER_CORE_IDLE_MAX
    ):
        return CLASS_MODEL_SLOW
    return CLASS_UNKNOWN


def is_timeout_provenance(provenance: Mapping[str, Any] | None) -> bool:
    return isinstance(provenance, Mapping) and provenance.get("class") in TIMEOUT_PROVENANCE_CLASSES


def label_timeout_provenance(
    provenance: dict[str, Any] | None, covariates: Mapping[str, Any] | None
) -> dict[str, Any] | None:
    """Add ``contention_class`` (+ the inputs it was derived from) to a timeout row, in place."""
    if not is_timeout_provenance(provenance):
        return provenance
    assert provenance is not None
    provenance["contention_class"] = classify_contention(covariates)
    provenance["contention_inputs"] = {
        k: (covariates or {}).get(k) for k in ("host_inflight", "loadavg_per_core") if (covariates or {}).get(k) is not None
    }
    return provenance


def timeout_rate_by_class(rows: Iterable[tuple[Mapping[str, Any] | None, Any]]) -> dict[str, Any]:
    """Aggregate over ``(failure_provenance, _)`` pairs: counts and rate of ALL rows per class."""
    counts = {c: 0 for c in CONTENTION_CLASSES}
    total = 0
    for provenance, _ in rows:
        total += 1
        if is_timeout_provenance(provenance):
            label = provenance.get("contention_class")
            counts[label if label in counts else CLASS_UNKNOWN] += 1
    return {
        "total_rows": total,
        "timeout_rows": sum(counts.values()),
        "counts": counts,
        "rates": {c: (counts[c] / total if total else 0.0) for c in CONTENTION_CLASSES},
    }
