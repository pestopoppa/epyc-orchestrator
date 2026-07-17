"""Prompt-injection / authority-laundering attack corpus for CandidatePackage review.

Spec: control-plane spec §13 (CandidatePackage security) + §20.5 (security tests).

Each ``*.json`` file in this directory is ONE probe describing an adversarial
candidate and the property the reviewer plane is supposed to preserve. The
corpus is consumed by ``tests/test_injection_probes.py`` (corpus runner) and
individual probes are referenced by ``tests/test_candidate_security.py``.

Probe schema
------------
```
{
  "id":               unique probe id,
  "category":         §13.1 threat class,
  "threat":           short spec pointer,
  "description":      what the attack tries,
  "vector":           where the payload lives: outputs | objective |
                      acceptance_checks | author_field | referenced_file,
  "payload":          the adversarial text / value,
  "full_package":     a full CandidatePackage (pre-sanitization) carrying the attack,
  "desired_property": the security property that SHOULD hold,
  "landed_defense":   which landed control (if any) addresses it,
  "expected":         "neutralized" (a landed control defeats it) |
                      "gap" (landed sanitizer does NOT neutralize it — xfail desired)
}
```

The ``sanitize_candidate_package`` reference below models the *landed* Wave-1
sanitization contract expressed by ``orchestration/candidate_package.schema.json``
(strip author framing FIELDS into ``sanitized_view``; project the rest verbatim).
It is deliberately faithful — including the fact that it does NOT scrub in-content
injection — so the runner can distinguish neutralized attacks from real gaps.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent

#: Framing fields the schema forbids inside ``sanitized_view`` (intake-837/838).
BANNED_FRAMING_FIELDS = (
    "author_self_assessment",
    "author_confidence_assertion",
    "quality_labels",
)

#: Sanitization policy version stamped by this reference contract.
REFERENCE_POLICY_VERSION = "wave1-sanitizer"


def load_probes() -> list[dict[str, Any]]:
    """Load every probe JSON in this directory, sorted by id."""
    probes: list[dict[str, Any]] = []
    for path in sorted(_HERE.glob("*.json")):
        probes.append(json.loads(path.read_text(encoding="utf-8")))
    return probes


def sanitize_candidate_package(full_pkg: dict[str, Any]) -> dict[str, Any]:
    """Reference implementation of the landed Wave-1 sanitization contract.

    Mirrors ``candidate_package.schema.json``: the reviewer-visible projection
    strips author self-assessment / confidence / quality labels, and projects the
    neutral fields verbatim. Content of ``outputs`` / ``objective`` is NOT scrubbed
    (that is the current contract — and the source of the documented gaps).
    """
    removed = [f for f in BANNED_FRAMING_FIELDS if f in full_pkg]
    sanitized_view: dict[str, Any] = {
        "task_ref": full_pkg["task_ref"],
        "outputs": list(full_pkg.get("outputs", [])),
        "sanitization": {
            "applied": True,
            "removed_fields": removed,
            "policy_version": REFERENCE_POLICY_VERSION,
        },
    }
    if "objective" in full_pkg:
        sanitized_view["objective"] = full_pkg["objective"]
    if "acceptance_checks" in full_pkg:
        sanitized_view["acceptance_checks"] = list(full_pkg["acceptance_checks"])

    out = {k: v for k, v in full_pkg.items() if k != "sanitized_view"}
    out["sanitized_view"] = sanitized_view
    return out


def sanitized_view_text(sanitized_view: dict[str, Any]) -> str:
    """Flatten the reviewer-visible content the way the reviewer prompt would see it."""
    parts: list[str] = [str(sanitized_view.get("objective", ""))]
    for o in sanitized_view.get("outputs", []) or []:
        parts.append(str(o.get("ref", "")))
        parts.append(str(o.get("label", "")))
    for c in sanitized_view.get("acceptance_checks", []) or []:
        parts.append(str(c.get("statement", "")))
    return "\n".join(parts)
