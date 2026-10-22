"""HIL-6 sibling harness-improvement loop: build-only skeleton (option B, operator 2026-10-22).

The operator chose HIL-4 option **B, a sibling loop on the AutoPilot shared substrate**
(design: ``docs/design/hil6-sibling-loop-implementation.md`` in epyc-root). This package is the
sibling's side of that seam:

* :mod:`.substrate` is the narrow, package-qualified facade over the substrate the loop SHARES
  with AutoPilot (journal, AP-53 rejected-mutation ledger, MHS guards, AP-55 infra fingerprint,
  AP-54 eval fence). One implementation; the sibling never forks logic and never opens
  AutoPilot's live journal.
* :mod:`.interfaces` declares the sibling's OWN parts: proposer, evaluator, judge, decision rule,
  and the index-style proposer view (HIL-2 item 7).
* :mod:`.attempt_ledger` is the HIL-2 item 21 append-only attempt ledger.
* :mod:`.pilot` encodes the HIL-6 pilot requirements and the HIL-2 checklist (items 1-22) as a
  validator over a declared design.
* :mod:`.driver` is the cycle driver skeleton; :mod:`.finalize` is the VB-HIL-WRITE producer call.

**Nothing here runs a loop.** The 2026-09-16 deferment on RUNNING a harness-improvement loop is
in force. There is no ``__main__``, no CLI, no scheduler/autopilot/cron hook, and no concrete
proposer or evaluator that could reach a model: every inference-capable part is an interface that
a later, separately authorized change must implement. A unit test asserts that no non-test module
imports this package.
"""

from __future__ import annotations

__all__ = [
    "attempt_ledger",
    "driver",
    "finalize",
    "interfaces",
    "pilot",
    "substrate",
]
