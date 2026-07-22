#!/usr/bin/env python3
"""Consolidate the append-only routing store into TD-equivalent Q-values.

DAR-L491 write-path defect (see scripts/analysis/dar_write_path_audit.py):
the production routing scorer blind-appended a fresh row per observation instead
of TD-updating in place, so ~99.7% of routing rows have update_count=0 and each
(objective, action) pair is duplicated ~177x. This script replays those
append-only observations chronologically through the SAME TD math the live path
uses (episodic_store.apply_td_update), producing ONE consolidated Q-value per
(objective, action) pair — i.e. the store the live path WOULD have produced had
it TD-updated in place.

Design invariants
-----------------
* NON-DESTRUCTIVE. The original ``memories`` table is never modified. Results
  are written to a NEW table ``memories_consolidated`` (drop-in schema) plus
  provenance/meta side tables. A future operator step can atomically swap tables.
* RUN AGAINST A COPY. Refuses to WRITE to the live sessions store; copy the
  store dir to /mnt/raid0/llm/tmp/ first. ``--dry-run`` opens read-only and may
  point at the live DB (read-only queries are allowed).
* IDEMPOTENT. A real run drops+rebuilds the consolidated tables deterministically
  from ``memories`` content, so re-running yields byte-identical results.
* FAISS-SAFE (no new desync class). The FAISS index / id_map are NOT touched.
  Each consolidated row keeps its group's representative row id + embedding_idx,
  so its FAISS vector and id_map entry stay valid. Collapsed duplicates' vectors
  become orphaned-but-benign: retrieve_by_similarity over-fetches and filters by
  SQLite membership (episodic_store.retrieve_by_similarity Phase 2), so an
  orphaned vector simply resolves to no row and is dropped. A later compaction
  can rebuild a compact FAISS index from the consolidated id set.

Reward recovery
---------------
Write-time invariant ``initial_q = 0.5 + reward*0.5`` inverts to
``reward = 2*q - 1`` for update_count=0 rows (the append-only class). Each
appended row therefore encodes its own observation's reward, which is what the
chronological TD replay consumes.

Row classes
-----------
* update_count == 0 AND objective present  -> consolidated by (objective, action)
* update_count  > 0                          -> PASSTHROUGH (already TD-updated,
  e.g. the external/MemRL path). A distinct, legitimately-episodic class; copied
  verbatim, never merged.
* update_count == 0 AND objective is NULL    -> PASSTHROUGH (not keyable by the
  (objective, action) identity; e.g. external rows storing task_description).

Poisoned rows (in-band [ERROR:] answers, seeding fix 3bfe2584)
-------------------------------------------------------------
The ``[ERROR:`` marker lives in the ANSWER text, which is persisted in seed-run
report artifacts, NOT in episodic.db (routing rows store only task_type /
objective / priority). Poisoned rows are therefore NOT reliably identifiable
from store data — a 0.0 reward (q==0.5) is indistinguishable from a legitimately
wrong answer. This script makes no heuristic guess. It prints the store-side
0.0-reward exposure for operator era-triage and offers ``--exclude-memory-ids
FILE`` so an operator can feed a list derived offline from the artifacts.

Usage
-----
    # verify against a copy (writes memories_consolidated)
    cp -r /mnt/raid0/llm/epyc-orchestrator/orchestration/repl_memory/sessions \
          /mnt/raid0/llm/tmp/q_consolidate_test
    python scripts/maintenance/consolidate_q_append_only.py \
        --db /mnt/raid0/llm/tmp/q_consolidate_test/episodic.db

    # dry-run report (read-only; safe against the live DB)
    python scripts/maintenance/consolidate_q_append_only.py --db <episodic.db> --dry-run
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from orchestration.repl_memory.episodic_store import apply_td_update  # noqa: E402

LIVE_SESSIONS_DIR = Path(
    "/mnt/raid0/llm/epyc-orchestrator/orchestration/repl_memory/sessions"
)

CONSOLIDATED_TABLE = "memories_consolidated"
PROVENANCE_TABLE = "_q_consolidation_provenance"
META_TABLE = "_q_consolidation_meta"

DEFAULT_LEARNING_RATE = 0.1
DEFAULT_TEMPORAL_DECAY_RATE = 0.99


def _parse_ts(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None


def _is_live_path(db_path: Path) -> bool:
    """True if db_path is inside the live sessions store."""
    try:
        resolved = db_path.resolve()
        live = LIVE_SESSIONS_DIR.resolve()
    except OSError:
        return False
    return resolved == live or live == resolved.parent or live in resolved.parents


def _load_memory_columns(con: sqlite3.Connection) -> list[str]:
    return [row[1] for row in con.execute("PRAGMA table_info(memories)")]


def replay_group(
    rows: list[dict[str, Any]],
    *,
    learning_rate: float,
    temporal_decay_rate: float | None,
) -> tuple[float, int]:
    """Chronologically replay an append-only (objective, action) group.

    ``rows`` must be sorted by created_at ascending. The first row's stored
    q_value IS its initial_q (first observation, no TD step); each subsequent row
    contributes reward = 2*q - 1 through apply_td_update, decaying over the
    wall-clock gap between consecutive created_at timestamps.

    Returns (final_q, update_count) where update_count == len(rows) - 1 — matching
    the live path (first obs = store(); each later obs = update_q_value()).
    """
    q = float(rows[0]["q_value"])
    update_count = 0
    last_ts = _parse_ts(rows[0]["created_at"])
    for r in rows[1:]:
        reward = 2.0 * float(r["q_value"]) - 1.0
        ts = _parse_ts(r["created_at"])
        days = 0.0
        if last_ts is not None and ts is not None:
            days = max(0.0, (ts - last_ts).total_seconds() / 86400.0)
        q = apply_td_update(
            q, reward, learning_rate,
            days_elapsed=days, temporal_decay_rate=temporal_decay_rate,
        )
        update_count += 1
        if ts is not None:
            last_ts = ts
    return q, update_count


def _obj_of(context_json: str | None) -> str | None:
    if not context_json:
        return None
    try:
        return json.loads(context_json).get("objective")
    except (json.JSONDecodeError, TypeError):
        return None


def plan_consolidation(
    con: sqlite3.Connection,
    *,
    action_type: str,
    learning_rate: float,
    temporal_decay_rate: float | None,
    exclude_ids: set[str],
) -> dict[str, Any]:
    """Compute the consolidation plan (pure read). Returns a dict describing the
    consolidated rows, passthrough rows, and summary stats. Writes nothing."""
    cols = _load_memory_columns(con)
    col_list = ", ".join(cols)
    rows = con.execute(
        f"SELECT {col_list} FROM memories WHERE action_type = ?", (action_type,)
    ).fetchall()

    ci = {name: i for i, name in enumerate(cols)}

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    passthrough: list[tuple] = []
    excluded = 0
    zero_reward_uc0 = 0  # store-side 0.0-reward exposure (q==0.5, uc==0)

    for row in rows:
        rid = row[ci["id"]]
        if rid in exclude_ids:
            excluded += 1
            continue
        uc = row[ci["update_count"]] or 0
        obj = _obj_of(row[ci["context"]])
        q = row[ci["q_value"]]
        if uc == 0 and q is not None and abs(float(q) - 0.5) < 1e-9:
            zero_reward_uc0 += 1
        if uc == 0 and obj is not None:
            action = row[ci["action"]]
            groups[(obj, action)].append({c: row[ci[c]] for c in cols})
        else:
            passthrough.append(row)

    # Deterministic ordering: sort each group by created_at, and the groups by key
    consolidated: list[dict[str, Any]] = []
    source_rows_collapsed = 0
    for key in sorted(groups.keys()):
        grp = sorted(
            groups[key],
            key=lambda r: (str(r["created_at"] or ""), str(r["id"])),
        )
        final_q, update_count = replay_group(
            grp, learning_rate=learning_rate, temporal_decay_rate=temporal_decay_rate,
        )
        rep = grp[0]  # representative keeps id + embedding_idx (FAISS-safe)
        last = grp[-1]
        consolidated_row = dict(rep)
        consolidated_row["q_value"] = final_q
        consolidated_row["update_count"] = update_count
        consolidated_row["updated_at"] = last["created_at"]
        consolidated.append(
            {
                "row": consolidated_row,
                "source_ids": [g["id"] for g in grp],
                "source_row_count": len(grp),
            }
        )
        source_rows_collapsed += len(grp)

    return {
        "columns": cols,
        "column_index": ci,
        "consolidated": consolidated,
        "passthrough": passthrough,
        "action_type": action_type,
        "n_source_rows": len(rows),
        "n_groups": len(groups),
        "n_source_rows_collapsed": source_rows_collapsed,
        "n_passthrough": len(passthrough),
        "n_excluded": excluded,
        "zero_reward_uc0_exposure": zero_reward_uc0,
        "learning_rate": learning_rate,
        "temporal_decay_rate": temporal_decay_rate,
    }


def _write_consolidation(con: sqlite3.Connection, plan: dict[str, Any]) -> None:
    """Rebuild the consolidated + provenance + meta tables (idempotent)."""
    cols = plan["columns"]
    col_list = ", ".join(cols)
    placeholders = ", ".join("?" for _ in cols)

    con.execute(f"DROP TABLE IF EXISTS {CONSOLIDATED_TABLE}")
    con.execute(f"DROP TABLE IF EXISTS {PROVENANCE_TABLE}")
    con.execute(f"DROP TABLE IF EXISTS {META_TABLE}")
    # Same schema as memories (drop-in), created empty.
    con.execute(f"CREATE TABLE {CONSOLIDATED_TABLE} AS SELECT {col_list} FROM memories WHERE 0")
    con.execute(
        f"CREATE TABLE {PROVENANCE_TABLE} ("
        "consolidated_id TEXT, source_row_count INTEGER, method TEXT, "
        "source_ids_json TEXT)"
    )
    con.execute(
        f"CREATE TABLE {META_TABLE} ("
        "generated_at TEXT, action_type TEXT, n_source_rows INTEGER, "
        "n_groups INTEGER, n_source_rows_collapsed INTEGER, n_passthrough INTEGER, "
        "n_excluded INTEGER, zero_reward_uc0_exposure INTEGER, "
        "learning_rate REAL, temporal_decay_rate REAL, params_json TEXT)"
    )

    # Consolidated (td_replay) rows.
    for item in plan["consolidated"]:
        row = item["row"]
        con.execute(
            f"INSERT INTO {CONSOLIDATED_TABLE} ({col_list}) VALUES ({placeholders})",
            tuple(row[c] for c in cols),
        )
        con.execute(
            f"INSERT INTO {PROVENANCE_TABLE} VALUES (?, ?, ?, ?)",
            (row["id"], item["source_row_count"], "td_replay",
             json.dumps(item["source_ids"])),
        )

    # Passthrough rows copied verbatim.
    ci = plan["column_index"]
    for row in plan["passthrough"]:
        con.execute(
            f"INSERT INTO {CONSOLIDATED_TABLE} ({col_list}) VALUES ({placeholders})",
            tuple(row[ci[c]] for c in cols),
        )
        con.execute(
            f"INSERT INTO {PROVENANCE_TABLE} VALUES (?, ?, ?, ?)",
            (row[ci["id"]], 1, "passthrough", None),
        )

    con.execute(
        f"INSERT INTO {META_TABLE} VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            datetime.now(timezone.utc).isoformat(),
            plan["action_type"],
            plan["n_source_rows"],
            plan["n_groups"],
            plan["n_source_rows_collapsed"],
            plan["n_passthrough"],
            plan["n_excluded"],
            plan["zero_reward_uc0_exposure"],
            plan["learning_rate"],
            plan["temporal_decay_rate"] if plan["temporal_decay_rate"] is not None else -1.0,
            json.dumps({
                "learning_rate": plan["learning_rate"],
                "temporal_decay_rate": plan["temporal_decay_rate"],
            }),
        ),
    )
    con.commit()


def run(
    db_path: Path,
    *,
    dry_run: bool,
    action_type: str = "routing",
    learning_rate: float = DEFAULT_LEARNING_RATE,
    temporal_decay_rate: float | None = DEFAULT_TEMPORAL_DECAY_RATE,
    exclude_ids: set[str] | None = None,
) -> dict[str, Any]:
    exclude_ids = exclude_ids or set()
    if not db_path.exists():
        raise SystemExit(f"episodic DB not found: {db_path}")

    if not dry_run and _is_live_path(db_path):
        raise SystemExit(
            "refusing to WRITE to the live sessions store. Copy it to "
            "/mnt/raid0/llm/tmp/ first and point --db at the copy, or use "
            "--dry-run (read-only)."
        )

    if dry_run:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    else:
        con = sqlite3.connect(db_path)

    try:
        plan = plan_consolidation(
            con,
            action_type=action_type,
            learning_rate=learning_rate,
            temporal_decay_rate=temporal_decay_rate,
            exclude_ids=exclude_ids,
        )
        if not dry_run:
            _write_consolidation(con, plan)
    finally:
        con.close()

    n_consolidated = len(plan["consolidated"])
    before = plan["n_source_rows"]
    after = n_consolidated + plan["n_passthrough"]
    print(f"[q-consolidate] action_type={action_type}  db={db_path}")
    print(f"  mode:                    {'DRY-RUN (no write)' if dry_run else 'WRITE ' + CONSOLIDATED_TABLE}")
    print(f"  source rows:             {before:,}")
    print(f"  excluded (operator list):{plan['n_excluded']:,}")
    print(f"  (objective,action) grps: {plan['n_groups']:,}")
    print(f"  rows collapsed by TD:    {plan['n_source_rows_collapsed']:,}  -> {n_consolidated:,} rows")
    print(f"  passthrough rows:        {plan['n_passthrough']:,}  (uc>0 or objective NULL)")
    print(f"  consolidated total rows: {after:,}  ({before:,} -> {after:,})")
    print(f"  0.0-reward exposure:     {plan['zero_reward_uc0_exposure']:,}  "
          f"(uc=0 & q==0.5; NOT poison-identifiable from store — see --help)")
    return {
        "before_rows": before,
        "after_rows": after,
        "n_groups": plan["n_groups"],
        "n_consolidated": n_consolidated,
        "n_passthrough": plan["n_passthrough"],
        "n_excluded": plan["n_excluded"],
        "zero_reward_uc0_exposure": plan["zero_reward_uc0_exposure"],
        "dry_run": dry_run,
    }


def _load_exclude_ids(path: Path | None) -> set[str]:
    if path is None:
        return set()
    if not path.exists():
        raise SystemExit(f"--exclude-memory-ids file not found: {path}")
    ids = {ln.strip() for ln in path.read_text().splitlines() if ln.strip()}
    return {i for i in ids if not i.startswith("#")}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--db", type=Path, required=True,
                    help="Path to episodic.db (a COPY under /mnt/raid0/llm/tmp for writes).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Compute the plan read-only and print the summary; write nothing.")
    ap.add_argument("--action-type", default="routing")
    ap.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    ap.add_argument("--temporal-decay-rate", type=float, default=DEFAULT_TEMPORAL_DECAY_RATE,
                    help="Per-day decay toward 0.5; pass a negative value to disable decay.")
    ap.add_argument("--exclude-memory-ids", type=Path, default=None,
                    help="File of memory ids (one per line) to drop before replay. "
                         "Use for operator-derived poisoned-row lists (in-band errors "
                         "are NOT store-identifiable; derive offline from seed artifacts).")
    args = ap.parse_args()

    decay = args.temporal_decay_rate
    if decay is not None and decay < 0:
        decay = None

    run(
        args.db,
        dry_run=args.dry_run,
        action_type=args.action_type,
        learning_rate=args.learning_rate,
        temporal_decay_rate=decay,
        exclude_ids=_load_exclude_ids(args.exclude_memory_ids),
    )


if __name__ == "__main__":
    main()
