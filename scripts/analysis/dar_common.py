#!/usr/bin/env python3
"""Shared loader + helpers for the DAR offline analyses (dar_*.py).

Reads a READ-ONLY SNAPSHOT of the routing/reward store (never the live DB) and
exposes the routing-memory rows plus the derived quantities every DAR analysis
needs: recovered reward, the within-objective matched set, tier proxies, and
BGE embedding access.

MEASUREMENT NOTE (MEASUREMENT.md)
--------------------------------
Every number produced by these scripts is an OBSERVATION, never a
decision-gating measurement. The stored `q_value`s are PRE-fix (the
`role`/`producer_role` reward bug fixed 2026-07-21, epyc-orchestrator
3d452476) — a distinct instrument era from any post-fix reward. Do not
compare or co-train across that boundary. Nothing here carries a protocol-id.

Reward recovery
---------------
Write-time invariant `initial_q = 0.5 + reward*0.5` (q_scorer.py) inverts to
`reward = 2*q - 1`, valid ONLY for `update_count = 0` rows (99.7% of the store;
the TD update path almost never runs — see dar_write_path_audit.py). Rows with
`update_count > 0` have had `q` moved by TD and are excluded from reward
recovery.

Snapshot default: /mnt/raid0/llm/tmp/dar_snapshot/episodic.db
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

SNAPSHOT_DIR = Path(os.environ.get("DAR_SNAPSHOT_DIR", "/mnt/raid0/llm/tmp/dar_snapshot"))
EPISODIC_DB = SNAPSHOT_DIR / "episodic.db"
FAISS_PATH = SNAPSHOT_DIR / "embeddings.faiss"

# ---------------------------------------------------------------------------
# Tier proxy. The eval-tower T0-T3 tier is a property of an eval BATCH's
# question set (src/autopilot_core/tier_specs.py), not stored per routing
# memory, and the within-objective matched set is 100% `chat`/no-question_id.
# So tier is derived here as an explicit PROXY from stored context, two ways:
#   (1) task-class band from task_type + light objective-text pattern detection
#   (2) data-driven empirical-difficulty quartile (1 - mean success across roles)
# Both are labelled proxies; neither is the canonical eval-tower tier.
# ---------------------------------------------------------------------------
_EASY_TT = {"general", "hotpotqa", "simpleqa", "hellaswag", "arc", "mmlu"}
_MID_TT = {"math", "coder", "instruction_precision", "thinking", "gsm8k",
           "mbpp", "humaneval", "ifeval", "bfcl", "math500"}
_HARD_TT = {"gpqa", "debugbench", "bigcodebench", "livecodebench", "agentic",
            "mode_advantage_hard", "mode_advantage", "leetcode", "usaco",
            "cruxeval", "bcb", "lc"}


def task_class_band(task_type: str, objective: str) -> str:
    """Coarse difficulty band (proxy tier) from task_type, with a chat fallback
    that reads the objective text for suite-like patterns."""
    tt = (task_type or "").lower()
    if tt in _HARD_TT:
        return "T3_hard"
    if tt in _MID_TT:
        return "T2_mid"
    if tt in _EASY_TT:
        return "T0_T1_easy"
    if tt != "chat":
        return "other"
    # chat: infer from objective text
    o = (objective or "")
    ol = o.lower()
    import re
    if re.search(r"\b[A-D]\)\s", o) or "\nA)" in o or "\n(a)" in ol:
        return "T0_T1_easy"          # multiple-choice (mmlu/arc-like)
    if "<answer>" in ol or "short, precise answer" in ol or "give a short" in ol:
        return "T0_T1_easy"          # short-factual (simpleqa-like)
    if any(k in ol for k in ("def ", "function", "python", "algorithm", "code",
                             "implement", "return just")):
        return "T2_mid"              # code
    if any(k in ol for k in ("prove", "integral", "theorem", "gpqa",
                             "olympiad", "derive")):
        return "T3_hard"
    return "chat_unknown"


@dataclass
class Row:
    __slots__ = ("id", "emb_idx", "role", "action_type", "outcome",
                 "q_value", "update_count", "objective", "task_type",
                 "question_id", "source", "priority")
    id: str
    emb_idx: int
    role: str
    action_type: str
    outcome: str
    q_value: float
    update_count: int
    objective: str
    task_type: str
    question_id: str
    source: str
    priority: str

    @property
    def reward(self) -> float | None:
        """Write-time reward, recoverable only for never-TD-updated rows."""
        if self.update_count != 0 or self.q_value is None:
            return None
        return 2.0 * self.q_value - 1.0

    @property
    def success(self) -> int:
        return 1 if self.outcome == "success" else 0


def load_rows(action_type: str | None = "routing", db: Path = EPISODIC_DB) -> list[Row]:
    """Load rows from the snapshot, extracting context JSON fields in SQL."""
    if not db.exists():
        raise SystemExit(f"snapshot DB not found: {db} (run the snapshot step first)")
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    where = "" if action_type is None else f"WHERE action_type = '{action_type}'"
    cur = con.execute(f"""
        SELECT id, embedding_idx, action, action_type, outcome, q_value,
               update_count,
               json_extract(context,'$.objective'),
               json_extract(context,'$.task_type'),
               json_extract(context,'$.question_id'),
               json_extract(context,'$.source'),
               json_extract(context,'$.priority')
        FROM memories {where}
    """)
    rows = [Row(*r) for r in cur]
    con.close()
    return rows


def suite_of(question_id: str | None) -> str | None:
    if not question_id:
        return None
    return question_id.split("_", 1)[0]


def matched_set(rows: Iterable[Row], min_obs: int = 5, min_roles: int = 2):
    """Within-objective matched set: objectives routed to >= min_roles distinct
    roles, each role with >= min_obs observations.

    Returns (obj_to_role_stats, matched_objectives) where obj_to_role_stats maps
    objective -> {role: (n, n_success)} restricted to eligible roles.
    """
    by_obj: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for r in rows:
        if r.action_type != "routing" or r.objective is None:
            continue
        cell = by_obj[r.objective][r.role]
        cell[0] += 1
        cell[1] += r.success
    matched: dict[str, dict[str, tuple[int, int]]] = {}
    for obj, roles in by_obj.items():
        elig = {role: (n, s) for role, (n, s) in roles.items() if n >= min_obs}
        if len(elig) >= min_roles:
            matched[obj] = elig
    return by_obj, matched


def objective_role_gap(role_stats: dict[str, tuple[int, int]]) -> float:
    """Best-worst role success-rate gap (in fraction, 0..1) for one objective."""
    rates = [s / n for (n, s) in role_stats.values() if n > 0]
    if len(rates) < 2:
        return 0.0
    return max(rates) - min(rates)


def entropy(values, bin_width: float = 0.1) -> float:
    counts = Counter(round(v / bin_width) for v in values)
    total = len(values)
    if not total:
        return 0.0
    return -sum((n / total) * math.log2(n / total) for n in counts.values() if n)


class Embeddings:
    """Lazy BGE embedding access via the snapshot faiss index (reconstruct by
    embedding_idx). Vectors are 1024-d, L2-normalized (IndexFlatIP)."""

    def __init__(self, path: Path = FAISS_PATH):
        import faiss
        self._idx = faiss.read_index(str(path))
        self.ntotal = self._idx.ntotal
        self.dim = self._idx.d

    def get(self, emb_idx: int) -> np.ndarray | None:
        if emb_idx is None or emb_idx < 0 or emb_idx >= self.ntotal:
            return None
        try:
            return self._idx.reconstruct(int(emb_idx)).astype(np.float32)
        except Exception:
            return None

    def matrix(self, emb_idxs: list[int]) -> tuple[np.ndarray, np.ndarray]:
        """Return (X, mask) where X[k] is the embedding for emb_idxs[k] (zeros if
        missing) and mask[k] is True where present."""
        X = np.zeros((len(emb_idxs), self.dim), dtype=np.float32)
        mask = np.zeros(len(emb_idxs), dtype=bool)
        for k, ei in enumerate(emb_idxs):
            v = self.get(ei)
            if v is not None:
                X[k] = v
                mask[k] = True
        return X, mask


def snapshot_meta() -> dict:
    ts = (SNAPSHOT_DIR / "SNAPSHOT_TIMESTAMP.txt")
    return {
        "snapshot_dir": str(SNAPSHOT_DIR),
        "snapshot_ts_utc": ts.read_text().strip() if ts.exists() else "unknown",
        "episodic_db": str(EPISODIC_DB),
    }


if __name__ == "__main__":
    rows = load_rows("routing")
    uc0 = [r for r in rows if r.update_count == 0]
    print("snapshot:", snapshot_meta())
    print(f"routing rows: {len(rows):,}   update_count=0: {len(uc0):,} "
          f"({100*len(uc0)/len(rows):.2f}%)")
    _, matched = matched_set(rows)
    print(f"matched objectives (>=2 roles, >=5 obs): {len(matched):,}")
