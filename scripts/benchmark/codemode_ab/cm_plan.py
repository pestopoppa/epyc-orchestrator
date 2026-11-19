"""Pair plans of the CM-2 / CM-4 / CM-5 probe (stdlib only: the runner imports it, and the sealed analyzer loads it by path).

A PAIR is (task_id, kind). Each feature's arm is paired with C3 on the same task_id; C3 is run once per task and shared by every feature that lists it.

kind  meaning
  H   a base stage-1 task (the sealed 30-task plan of the earlier rows: 10 task bases x seeds 0,1,2): the HARM / equal-accuracy set
  L   C-LABEL (CM-4 classify benefit set)       K   C-RISK / C-ROUTE (CM-4 typed-classifier capability set)       P   H-PRICE (CM-5 benefit set)
"""

from __future__ import annotations

BASES = ["N-CSV-n20", "N-CSV-n5", "N-LEDGER-n20", "N-LEDGER-n5", "N-QUOTE", "N-STAT", "S-AGG-n20", "S-AGG-n5", "S-FILT-n20", "S-FILT-n5"]
BASE30 = [t for s in (0, 1, 2) for t in sorted(f"{b}-s{s}" for b in BASES)]
CM_SEEDS = (0, 1, 2, 3, 4)
NEW_CM4 = [t for s in CM_SEEDS for t in (f"C-LABEL-n20-s{s}", f"C-LABEL-n100-s{s}", f"C-RISK-n20-s{s}", f"C-ROUTE-n20-s{s}")]
NEW_CM5 = [t for s in CM_SEEDS for t in (f"H-PRICE-n20-s{s}", f"H-PRICE-n60-s{s}")]
FEATURES = ("cm4", "cm5", "cm2")  # run order: the strongest hypothesis first
ARM_OF = {"cm2": "C3_cm2", "cm4": "C3_cm4", "cm5": "C3_cm5"}
LOOKS = (10, 20, 30, 40, 50)


def kind(task_id: str) -> str:
    if task_id.startswith("C-LABEL"):
        return "L"
    if task_id.startswith(("C-RISK", "C-ROUTE")):
        return "K"
    if task_id.startswith("H-PRICE"):
        return "P"
    return "H"


def _interleave(h: list[str], b: list[str], pattern: str) -> list[str]:
    hi, bi, out = iter(h), iter(b), []
    for ch in pattern:
        out.append(next(hi if ch == "H" else bi))
    return out


def pairs(feature: str) -> list[tuple[str, str]]:
    """The planned pairs of ``feature`` in run order. cm4 = 30 H + 20 benefit (pattern H B H B H), cm5 = 30 H + 10 benefit (H H H B), cm2 = the 30 H."""
    if feature == "cm2":
        ids = list(BASE30)
    elif feature == "cm4":
        ids = _interleave(BASE30, NEW_CM4, "HBHBH" * 10)
    elif feature == "cm5":
        ids = _interleave(BASE30, NEW_CM5, "HHHB" * 10)
    else:
        raise KeyError(feature)
    return [(t, kind(t)) for t in ids]


def looks(feature: str) -> list[int]:
    n = len(pairs(feature))
    return [x for x in LOOKS if x < n] + [n]


def all_units() -> list[tuple[str, str]]:
    """Every (task_id, arm) the whole probe can run: C3 once per task plus each feature's arm."""
    seen: dict[tuple[str, str], None] = {}
    for f in FEATURES:
        for t, _k in pairs(f):
            seen[(t, "C3")] = None
            seen[(t, ARM_OF[f])] = None
    return list(seen)
