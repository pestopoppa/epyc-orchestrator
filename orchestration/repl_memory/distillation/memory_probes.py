"""M-27 read-only probes and reports over the skill backup, episodic store and SkillLearnBench.

Everything here is ADVISORY and REPORT-ONLY: nothing is distilled, stored, gated or scored.
No embedder or LLM is run; stored vectors are read with faiss.reconstruct. Databases are
opened ``mode=ro``.

  (a) negative-capability claims + skills whose source trajectory has no labelled success
  (b) PII-seed classes (see ``pii_seed``)
  (c) pair census: same task type, stored-embedding cosine in [0.6, 0.85), literal sets differ
  (d) retrieval_count floors, source role per skill, source-class composition
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Iterable

from orchestration.repl_memory.distillation.literal_lint import (
    extract_literals,
    extract_negative_capability,
    strip_frontmatter,
    unlabelled_success,
)
from orchestration.repl_memory.distillation.pii_seed import extract_pii_seed

PAIR_LO, PAIR_HI = 0.6, 0.85
FLOORS = (5, 20)
EFFECTIVENESS_CUT = 0.3

# Episodic context.source labels -> composition class. The store has NO interactive/cron label:
# unknown labels fall in "unlabelled" and "interactive" is reported as unobservable.
SOURCE_CLASS = {
    "progress_log": "autopilot",  # autopilot progress JSONL writer
    "external": "eval_or_external",  # external-harness writes
    "seed": "seed",
    "legacy": "pre_label_legacy",
}


def classify_source(label: str | None) -> str:
    return SOURCE_CLASS.get((label or "").lower(), "unlabelled")


def source_composition(labels: Iterable[str | None]) -> dict:
    c = Counter(classify_source(x) for x in labels)
    n = sum(c.values())
    return {"n": n, "counts": dict(c), "interactive_label_available": False}


def _ro(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def _skill_rows(db: Path) -> list[dict]:
    con = _ro(db)
    con.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in con.execute("SELECT * FROM skills ORDER BY created_at, id")]
    finally:
        con.close()


def skill_text(r: dict) -> str:
    return f"{r['title']}\n{r['principle']}\n{r['when_to_apply']}"


# ── (a)(b) lint extensions on the skill backup ──────────────────────────────
def lint_extensions_skills_db(db: Path) -> dict:
    rows = _skill_rows(db)
    neg = [r for r in rows if extract_negative_capability(skill_text(r))]
    unl = [r for r in rows if unlabelled_success(r["source_outcome"])]
    pii: Counter = Counter()
    for r in rows:
        for cls in extract_pii_seed(skill_text(r)):
            pii[cls] += 1
    return {
        "n": len(rows),
        "negative_capability_flagged": len(neg),
        "negative_capability_examples": [extract_negative_capability(skill_text(r))[0] for r in neg[:5]],
        "unlabelled_success_flagged": len(unl),
        "source_outcome_counts": dict(Counter(r["source_outcome"] for r in rows)),
        "pii_seed_flagged_by_class": dict(pii),
    }


def lint_extensions_skilllearnbench(repo: Path) -> dict:
    """Negative-capability and PII-seed flag rate per generator config. SkillLearnBench carries no
    per-skill source-trajectory outcome label, so the unlabelled-success class is not computable there."""
    per: dict[str, Counter] = defaultdict(Counter)
    for sk in sorted((repo / "skills").glob("*/*/*/SKILL.md")):
        cfg = sk.parts[-4]
        body = strip_frontmatter(sk.read_text(errors="ignore"))
        per[cfg]["n"] += 1
        per[cfg]["negcap"] += bool(extract_negative_capability(body))
        pii = extract_pii_seed(body)
        per[cfg]["pii_any"] += bool(pii)
        for cls in pii:
            per[cfg][cls] += 1
    return {"per_config": {k: dict(v) for k, v in sorted(per.items())},
            "unlabelled_success": "not computable: corpus has no source-trajectory outcome label"}


# ── (d) retrieval_count floors ──────────────────────────────────────────────
def retrieval_floor_report(db: Path) -> dict:
    rows = _skill_rows(db)
    dist = Counter(int(r["retrieval_count"] or 0) for r in rows)
    out = {"n": len(rows), "retrieval_count_distribution": {str(k): v for k, v in sorted(dist.items())},
           "max": max(dist) if dist else 0, "floors": {}}
    for f in FLOORS:
        at = [r for r in rows if int(r["retrieval_count"] or 0) >= f]
        out["floors"][str(f)] = {
            "at_or_above_floor": len(at),
            f"below_effectiveness_{EFFECTIVENESS_CUT}": sum(1 for r in at if (r["effectiveness_score"] or 0) < EFFECTIVENESS_CUT),
        }
    out["effectiveness_below_cut_all_skills"] = sum(1 for r in rows if (r["effectiveness_score"] or 0) < EFFECTIVENESS_CUT)
    return out


# ── (d) source role per skill ───────────────────────────────────────────────
def source_role_per_skill(skills_db: Path, episodic_dbs: list[Path]) -> dict:
    """Resolve source_trajectory_ids against episodic stores (current first, then older backups);
    role = the ``frontdoor``-style prefix of the memory ``action``. Unresolved ids are counted."""
    rows = _skill_rows(skills_db)
    want: dict[str, list[str]] = {}
    for r in rows:
        try:
            want[r["id"]] = list(json.loads(r["source_trajectory_ids"]))
        except Exception:  # noqa: BLE001
            want[r["id"]] = []
    all_ids = {i for v in want.values() for i in v}
    found: dict[str, str] = {}
    searched = []
    for p in episodic_dbs:
        if not p.exists() or not all_ids - set(found):
            continue
        con = _ro(p)
        try:
            remaining = sorted(all_ids - set(found))
            for k in range(0, len(remaining), 500):
                chunk = remaining[k : k + 500]
                q = ",".join("?" * len(chunk))
                for mid, action in con.execute(f"SELECT id, action FROM memories WHERE id IN ({q})", chunk):
                    found[mid] = (action or "").split(":", 1)[0]
        finally:
            con.close()
        searched.append(p.name)
    per_skill, role_counts = {}, Counter()
    for sid, ids in want.items():
        roles = Counter(found[i] for i in ids if i in found)
        top = roles.most_common(1)[0][0] if roles else "unresolved"
        per_skill[sid] = {"top_role": top, "resolved": sum(roles.values()), "of": len(ids)}
        role_counts[top] += 1
    return {"searched": searched, "ids_total": len(all_ids), "ids_resolved": len(found),
            "skills_by_top_role": dict(role_counts), "per_skill": per_skill}


# ── (c) pair census ─────────────────────────────────────────────────────────
def _task_types(r: dict) -> set[str]:
    try:
        return {t for t in json.loads(r["task_types"]) if t != "unknown"}
    except Exception:  # noqa: BLE001
        return set()


def _literal_set(text: str) -> frozenset[str]:
    lits = extract_literals(strip_frontmatter(text))
    return frozenset(f"{c}:{v}" for c, vs in lits.items() for v in vs)


def cosine(a, b) -> float:
    da = math.sqrt(sum(x * x for x in a))
    db = math.sqrt(sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b)) / (da * db) if da and db else 0.0


def pair_census(items: list[dict], lo: float = PAIR_LO, hi: float = PAIR_HI, n_examples: int = 10) -> dict:
    """items: {id, task_types:set, text, vec}. A pair counts when it shares a (non-unknown) task type,
    cosine in [lo, hi) and the two literal sets differ."""
    lits = {i["id"]: _literal_set(i["text"]) for i in items}
    n_pairs = n_same_type = n_band = n_differ = 0
    examples = []
    for a in range(len(items)):
        for b in range(a + 1, len(items)):
            x, y = items[a], items[b]
            n_pairs += 1
            if not (x["task_types"] & y["task_types"]):
                continue
            n_same_type += 1
            c = cosine(x["vec"], y["vec"])
            if lo <= c < hi:
                n_band += 1
                if lits[x["id"]] != lits[y["id"]]:
                    n_differ += 1
                if lits[x["id"]] != lits[y["id"]] and len(examples) < n_examples:
                    examples.append({"a": x["id"], "b": y["id"], "cosine": round(c, 4),
                                     "only_a": sorted(lits[x["id"]] - lits[y["id"]])[:5],
                                     "only_b": sorted(lits[y["id"]] - lits[x["id"]])[:5]})
    counted = n_differ
    return {"band": [lo, hi], "pairs_total": n_pairs, "pairs_same_task_type": n_same_type,
            "pairs_in_band": n_band, "pairs_in_band_literals_differ": counted, "examples": examples}


def load_backup_vectors(skills_db: Path, faiss_path: Path, idmap_path: Path) -> tuple[list[dict], dict]:
    """Read stored skill vectors (no embedder). Returns (items, readability report)."""
    import faiss  # imported late: only needed here
    import numpy as np

    idx = faiss.read_index(str(faiss_path))
    idmap = np.load(str(idmap_path), allow_pickle=True)
    rows = _skill_rows(skills_db)
    items, missing = [], 0
    for r in rows:
        ei = r["embedding_idx"]
        if ei is None or not (0 <= ei < idx.ntotal):
            missing += 1
            continue
        items.append({"id": r["id"], "task_types": _task_types(r), "text": skill_text(r),
                      "vec": [float(v) for v in idx.reconstruct(int(ei))]})
    return items, {"index_ntotal": int(idx.ntotal), "dim": int(idx.d), "id_map_len": int(len(idmap)),
                   "skills": len(rows), "vectors_read": len(items), "missing_embedding": missing}


# ── (d) top-k source-class composition for a FIXED recall-query set ─────────
# Needs query embeddings (the embedder = a model call), so the live run is an inference-window step;
# the logic below is pure and fixture-tested with an injected retriever.
FIXED_RECALL_QUERIES = (
    "route a code generation task",
    "escalate after repeated failure",
    "which role handles long-context summarisation",
    "debug a failing unit test",
    "answer a factual question quickly",
    "plan a multi-step refactor",
    "review a pull request for security issues",
    "extract fields from a csv file",
)


def recall_composition(
    retrieve: Callable[[str, int], list[str | None]],
    queries: Iterable[str] = FIXED_RECALL_QUERIES,
    k: int = 10,
) -> dict:
    """``retrieve(query, k)`` -> the ``context.source`` label of each of the top-k hits."""
    per_query, total = {}, Counter()
    for q in queries:
        labels = retrieve(q, k)
        comp = source_composition(labels)
        per_query[q] = comp["counts"]
        total.update(comp["counts"])
    return {"k": k, "per_query": per_query, "pooled": dict(total), "interactive_label_available": False}


def live_recall_composition(k: int = 10) -> dict:  # pragma: no cover - needs the embedder
    import os

    if os.environ.get("WINDOW_CONFIRMED") != "1":
        raise SystemExit("refusing: query embedding calls the embedder; set WINDOW_CONFIRMED=1 in a granted window")
    from orchestration.repl_memory.embedder import TaskEmbedder
    from orchestration.repl_memory.episodic_store import EpisodicStore

    emb, store = TaskEmbedder(), EpisodicStore()

    def retrieve(q: str, kk: int):
        hits = store.retrieve_by_similarity(emb.embed_text(q, side="query"), k=kk)
        return [(m.context or {}).get("source") for m in hits]

    return recall_composition(retrieve, k=k)


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sessions-dir", type=Path, required=True)
    ap.add_argument("--skills-db", default="skills.db.pre-purge-20260727T221856Z")
    ap.add_argument("--faiss", default="skill_embeddings.faiss.pre-purge")
    ap.add_argument("--id-map", default="skill_id_map.npy.pre-purge")
    ap.add_argument("--episodic", nargs="*", default=["episodic.db", "episodic.db.pre-repair-20260727T210348Z",
                                                      "episodic.db.pre-reseed-20260727T220715Z"])
    ap.add_argument("--skilllearnbench", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--live-recall", action="store_true", help="needs the embedder; requires WINDOW_CONFIRMED=1")
    a = ap.parse_args(argv)
    if a.live_recall:
        print(json.dumps(live_recall_composition(), indent=1, sort_keys=True))
        return 0
    sd = a.sessions_dir
    res: dict = {"advisory_only": True, "nothing_distilled": True}
    res["skills_db_lint_ext"] = lint_extensions_skills_db(sd / a.skills_db)
    res["retrieval_floors"] = retrieval_floor_report(sd / a.skills_db)
    res["source_role"] = source_role_per_skill(sd / a.skills_db, [sd / e for e in a.episodic])
    items, readable = load_backup_vectors(sd / a.skills_db, sd / a.faiss, sd / a.id_map)
    res["embeddings_readable"] = readable
    res["pair_census_backup"] = pair_census(items)
    if a.skilllearnbench:
        res["skilllearnbench_lint_ext"] = lint_extensions_skilllearnbench(a.skilllearnbench)
        res["skilllearnbench_pair_census"] = "not computable: no stored embeddings in the corpus (needs an embedder run)"
    con = _ro(sd / "episodic.db")
    try:
        labels = [r[0] for r in con.execute("SELECT json_extract(context,'$.source') FROM memories")]
    finally:
        con.close()
    res["episodic_source_composition_all_rows"] = source_composition(labels)
    text = json.dumps(res, indent=1, sort_keys=True)
    (a.out.write_text(text) if a.out else print(text))
    return 0


if __name__ == "__main__":
    sys.exit(main())
