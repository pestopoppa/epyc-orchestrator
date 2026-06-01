#!/usr/bin/env python3
"""One-shot permanent scrub of the legacy gate-lock / stale-baseline / 2.900 narrative.

Runs ONLY with the autopilot stopped (else in-memory stores re-clobber disk).
Scrubs every planner read-source the narrative re-injects from:
  1. autopilot_journal.jsonl  — `falsifier` field (22 trials)  -> neutral generic falsifier
  2. short_term_memory.md     — "baseline 2.900" gate-lock failure-pattern lines
  3. strategies.db            — 16 narrative strategy rows (+ FTS rebuild + faiss/id_map rebuild)
Episodic memory is intentionally UNTOUCHED (its 26 matches were timestamp false-positives).
Everything is backed up first.
"""
from __future__ import annotations
import json, re, shutil, sqlite3, time
from pathlib import Path
import numpy as np
import faiss

ORCH = Path("orchestration")
STRAT_DIR = ORCH / "repl_memory" / "strategies"
JOURNAL = ORCH / "autopilot_journal.jsonl"
STM = Path("scripts/autopilot/short_term_memory.md")
TS = time.strftime("%Y%m%d_%H%M%S")
BK = ORCH / f"scrub_gatelock_backup_{TS}"
BK.mkdir(parents=True, exist_ok=True)

# Narrative for free-text reasoning (falsifier) — no timestamps live here.
NARR = re.compile(
    r"gate.?lock|stale[ -]?baseline|contaminated.{0,8}baseline|baseline.{0,12}(contaminat|stale|miscalibrat)"
    r"|no[ -]?op spiral|re[ -]?baseline|miscalibrat|2\.900",
    re.I,
)
NEUTRAL_FALSIFIER = (
    "Falsifier: this action is refuted if the next eval's per-tier quality delta "
    "moves opposite to the predicted direction."
)

def backup(p: Path):
    if p.exists():
        shutil.copy2(p, BK / p.name)

print(f"== backups -> {BK} ==")
for p in [JOURNAL, STM, STRAT_DIR / "strategies.db",
          STRAT_DIR / "strategy_embeddings.faiss", STRAT_DIR / "strategy_id_map.npy"]:
    backup(p)

# ---------------------------------------------------------------- 1. JOURNAL
lines_out, scrubbed_trials = [], []
for line in JOURNAL.read_text().splitlines():
    s = line.strip()
    if not s:
        lines_out.append(line); continue
    try:
        r = json.loads(s)
    except Exception:
        lines_out.append(line); continue
    f = r.get("falsifier")
    if isinstance(f, str) and NARR.search(f):
        r["falsifier"] = NEUTRAL_FALSIFIER
        scrubbed_trials.append(r.get("trial_id"))
        lines_out.append(json.dumps(r))
    else:
        lines_out.append(line)
JOURNAL.write_text("\n".join(lines_out) + "\n")
print(f"[journal] scrubbed falsifier on {len(scrubbed_trials)} trials: {sorted(t for t in scrubbed_trials if t is not None)}")

# ------------------------------------------------------------ 2. SHORT-TERM MEM
stm_lines = STM.read_text().splitlines()
STM_NARR = re.compile(r"baseline 2\.900|gate.?lock|stale[ -]?baseline|no[ -]?op spiral|re[ -]?baseline", re.I)
kept, dropped = [], 0
for ln in stm_lines:
    if STM_NARR.search(ln):
        dropped += 1
        continue
    kept.append(ln)
# also refresh the stale "Best quality: 2.40" working-context line (T0-saturation artifact)
kept = [re.sub(r"^- Best quality: 2\.40\b.*$", "- Best quality: (recomputed live from per-tier frontier)", ln) for ln in kept]
STM.write_text("\n".join(kept) + "\n")
print(f"[stm] removed {dropped} gate-lock failure-pattern lines")

# ---------------------------------------------------------------- 3. STRATEGIES
con = sqlite3.connect(STRAT_DIR / "strategies.db")
cur = con.cursor()
SNARR = re.compile(
    r"gate.?lock|stale[ -]?baseline|contaminated.{0,8}baseline|baseline.{0,12}(contaminat|stale|miscalibrat)"
    r"|no[ -]?op spiral|re[ -]?baseline|miscalibrat|2\.900", re.I)
rows = list(cur.execute("SELECT id, coalesce(description,''), coalesce(insight,'') FROM strategies"))
del_ids = [r[0] for r in rows if SNARR.search(r[1] + " " + r[2])]
print(f"[strategies] deleting {len(del_ids)} narrative rows")
qmarks = ",".join("?" * len(del_ids))
cur.execute(f"DELETE FROM strategies WHERE id IN ({qmarks})", del_ids)
# Rebuild the standalone FTS mirror from the cleaned base table.
cur.execute("DELETE FROM strategies_fts")
cur.execute("INSERT INTO strategies_fts(rowid, id, description, insight, species) "
            "SELECT rowid, id, description, insight, species FROM strategies")
con.commit()
n_left = cur.execute("SELECT COUNT(*) FROM strategies").fetchone()[0]
n_fts = cur.execute("SELECT COUNT(*) FROM strategies_fts").fetchone()[0]
con.close()
print(f"[strategies] base={n_left} fts={n_fts} (must match)")

# Rebuild faiss + id_map keeping only surviving uuids (IndexFlatIP -> reconstructable, no re-embed).
idx = faiss.read_index(str(STRAT_DIR / "strategy_embeddings.faiss"))
id_map = np.load(STRAT_DIR / "strategy_id_map.npy", allow_pickle=True)
del_set = set(del_ids)
keep_pos = [i for i, uid in enumerate(id_map) if uid not in del_set]
vecs = idx.reconstruct_n(0, idx.ntotal)  # (ntotal, d)
new_idx = faiss.IndexFlatIP(idx.d)
new_idx.add(np.ascontiguousarray(vecs[keep_pos]))
new_map = np.array([id_map[i] for i in keep_pos], dtype=object)
faiss.write_index(new_idx, str(STRAT_DIR / "strategy_embeddings.faiss"))
np.save(STRAT_DIR / "strategy_id_map.npy", new_map)
print(f"[strategies] faiss {idx.ntotal} -> {new_idx.ntotal}, id_map {len(id_map)} -> {len(new_map)}")

print("\n== DONE ==")
