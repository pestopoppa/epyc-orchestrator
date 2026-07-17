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

The corpus exercises the PRODUCTION assembly sanitizer directly: ``sanitize_candidate_package``
is re-exported from ``src.proactive_delegation.candidate_sanitizer`` (the standalone,
freeze-safe assembly-time sanitizer that BUILDS the reviewer-visible ``sanitized_view``).
The runner distinguishes neutralized attacks from real gaps by running that production
sanitizer and, where a gap is render-layer (review_service's prompt renderer, FROZEN),
observing that the payload still reaches the captured reviewer prompt.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Re-export the PRODUCTION assembly sanitizer so the security corpus tests the real
# assembly-time contract, not a divergent test copy.
from src.proactive_delegation.candidate_sanitizer import (  # noqa: F401
    BANNED_FRAMING_FIELDS,
    SANITIZER_POLICY_VERSION,
    sanitize_candidate_package,
    sanitized_view_text,
)

_HERE = Path(__file__).resolve().parent

#: Back-compat alias for the sanitization policy version stamped by the contract.
REFERENCE_POLICY_VERSION = SANITIZER_POLICY_VERSION


def load_probes() -> list[dict[str, Any]]:
    """Load every probe JSON in this directory, sorted by id."""
    probes: list[dict[str, Any]] = []
    for path in sorted(_HERE.glob("*.json")):
        probes.append(json.loads(path.read_text(encoding="utf-8")))
    return probes
