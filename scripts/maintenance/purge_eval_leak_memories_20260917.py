#!/usr/bin/env python3
"""purge_eval_leak_memories_20260917.py — purge two eval-pool leaks from memory and state backups.

OPERATOR DECISION 2026-09-17: "clean both".

THE LEAKS
---------
1. HumanEval/55 (pool rows ``sv_HE-R_HumanEval/55::sol0-4`` and ``sv_HE-R+_HumanEval/55::sol0-4``).
   Four routing memories quote the task text (``def fib(n: int): Return n-th Fibonacci
   number. >>> fib(10) 55 ...``). They sit in every ``episodic.db``: the live store, every
   AutoPilot checkpoint, and the backup copies.
2. simpleqa_general_00912 ("At which university did Jurgen Aschoff study medicine?"; the second
   Aschoff row, simpleqa_general_00795, is included as well). The retired rules.md few-shot
   example and the DEFAULT_ROOT_LM_RULES query stem (fixed by 09bdb998) are quoted in the
   ``last_traces`` text of AutoPilot state backups.

DETECTION (a re-implementation of the MHS-3 content matcher; reading only)
-------------------------------------------------------------------------
Tokens are casefolded runs of [A-Za-z0-9_] and non-ASCII bytes, the same rule as
``prompt_forge._hash_token_buffer`` (branch sub/mhs3-complete-20260916). Tokens are kept as exact
token tuples here instead of 64-bit hashes, so there are no hash collisions. The signature is
built from the pool rows above (``prompt`` and ``expected`` texts):

* shingles: every 8-token window. A shingle found in >= 32 distinct rows of the pool (the
  research question pool plus the orchestrator ``core_*.jsonl``) is TEMPLATE text and is dropped.
* a hit counts when its 8 tokens include an ANCHOR token (>= 4 bytes, not all digits, pool
  frequency <= 200), or when it lies in a run of >= 8 consecutive hits (a >= 15-token verbatim
  copy). These are the MHS-3 rules.
* keyword: the token ``aschoff`` is always a hit (the 00912 answer is too short for a shingle).

Every text column of every table is scanned, one column at a time.

WHAT --apply DOES, PER STORE (``episodic.db`` plus its ``embeddings.faiss``/``id_map.npy`` pair)
------------------------------------------------------------------------------------------------
1. Durable backup of the originals to ``--backup-root/<store-key>/`` (sha256 verified).
2. Staged copy under ``--backup-root/stage/<store-key>/`` with the canonical names, then:
   * ``PRAGMA secure_delete=ON`` and DELETE of every flagged row, in every table. The deleted
     text is zeroed on disk, not left in the freelist. ``_q_consolidation_provenance`` rows for
     deleted consolidated ids go too.
   * FAISS: the vectors of the deleted ids are removed from the IndexFlatIP, and their ids from
     ``id_map`` (the same relative order in both). ``memories.embedding_idx`` is shifted down by the
     number of removed positions below it. Nothing else is repaired: a pre-existing desync in an
     historical checkpoint (20260716_062336: index +32) is preserved exactly. A row that did not
     match but pointed at a removed position (only possible with a pre-existing wrong pointer) gets
     ``embedding_idx = NULL``, the fail-closed choice ``repair_faiss_id_map.py`` also makes.
3. Verification on the stage, all of which must pass:
   * ``repair_faiss_id_map.diagnose()`` (the existing consistency oracle) before and after. The
     counts must move by exactly the removed rows: desync unchanged, correct/wrong/missing reduced
     only by the removed rows.
   * the kept vectors are byte-identical to the originals in their original order;
   * ``check_episodic_integrity.py --json`` (the episodic FAISS health check). It must PASS for
     the live store. For historical stores it must be no worse than before.
   * ``PRAGMA integrity_check`` is ok; a rescan finds zero leaked rows; the raw bytes of the DB
     file contain no ``aschoff`` and no ``Return n-th Fibonacci number.``.
4. Publish: the stage files are copied next to their targets and os.replace()d in the order
   id_map, then index, then db, under ``.episodic_faiss.lock``. Before that, the target is checked
   for being unchanged since the backup. Any failure restores every original from the backup.

``repair_faiss_id_map.repair()`` is deliberately NOT called. It would also truncate the
pre-existing desync of historical checkpoints and write 1-3 GB ``pre-repair`` copies into each
store directory. Its ``diagnose()`` is the oracle instead.

STATE BACKUPS (``autopilot_state*.bak*`` and the other autopilot_state backup copies)
------------------------------------------------------------------------------------
In-place redaction of the raw file text, with a backup first. The leaked strings are replaced with
the 09bdb998 synthetic entity ("Halvard Oskeberg" / chemistry), so the JSON stays byte-valid and
keeps its formatting. The file must still parse and rescan clean, or it is restored.

SAFETY / SCOPE
--------------
* Dry-run by default: it scans and reports, and writes nothing except the signature cache.
* The LIVE store (``orchestration/repl_memory/sessions``) is touched only with ``--include-live``,
  and never while AutoPilot runs (singleton lock), the API runs (a ``src.api:app`` process or a
  listener on :8000), or any process holds the store files open. It then prints the operator
  procedure and exits 3.
* Checkpoints whose ``checkpoint_meta.json`` pins the episodic files (``multitier_v10_20260810`` =
  production_best) are touched only with ``--include-pinned``. That also requires the ckpt-leak
  ratification (RATIFY-CKPT-PROMPT-LEAK-20260916) to be applied first, because its manifest check
  would otherwise refuse. The new hashes are written to ``pinned_repin_proposal.json`` for an
  operator ratification amendment. This script never edits a checkpoint meta or a ratified receipt.
* Checkpoint ``autopilot_state.json`` files are scanned and reported, never edited.
* Idempotent: a store or file that rescans clean is reported CLEAN and skipped. Every run appends
  to ``<backup-root>/receipt.json``.

Usage:
    purge_eval_leak_memories_20260917.py                     # dry-run inventory
    purge_eval_leak_memories_20260917.py --apply             # offline stores + state backups
    purge_eval_leak_memories_20260917.py --apply --include-live      # API stopped!
    purge_eval_leak_memories_20260917.py --apply --include-pinned    # after the ckpt-leak ratify
    purge_eval_leak_memories_20260917.py --verify [--include-live] [--include-pinned]
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import mmap
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "epyc.eval_leak_memory_purge.v1"
GATE_ID = "PURGE-EVAL-LEAK-MEMORIES-20260917"
DEFAULT_ORCH = Path("/mnt/raid0/llm/epyc-orchestrator")
DEFAULT_BACKUP_ROOT = Path("/mnt/raid0/llm/backups/episodic-leak-20260917")
DEFAULT_POOL = Path("/mnt/raid0/llm/epyc-inference-research/benchmarks/prompts/question_pool.jsonl")
DEFAULT_WORKTREES = Path("/mnt/raid0/llm/worktrees")

LEAK_IDS = {"simpleqa_general_00912", "simpleqa_general_00795"}
LEAK_ID_RE = re.compile(r"(?:^|[_/])HumanEval/55(?:::|$)")
KEYWORD_TOKENS = {b"aschoff"}
RAW_MARKERS = [re.compile(rb"(?i)aschoff"), re.compile(rb"Return n-th Fibonacci number\.")]

NGRAM = 8
TEMPLATE_DF = 32
ANCHOR_MIN_BYTES = 4
RARE_TF = 200
MIN_UNANCHORED_RUN = 8
TOKEN_RE = re.compile(rb"[0-9a-z_\x80-\xff]+")

# Ordered: the longest strings first. The targets are the 09bdb998 replacements.
STATE_REPLACEMENTS = [
    ("How many months after his wife, Hilde, died did Jurgen Aschoff also pass away?",
     "How many months after his wife died did Halvard Oskeberg also pass away?"),
    ("At which university did Jurgen Aschoff study medicine?",
     "At which university did Halvard Oskeberg study chemistry?"),
    ("Jurgen Aschoff study medicine university", "Halvard Oskeberg study chemistry university"),
    ("Jurgen Aschoff university", "Halvard Oskeberg university"),
    ("Jurgen Aschoff", "Halvard Oskeberg"),
    ("Aschoff", "Oskeberg"),
    ("aschoff", "oskeberg"),
    ("ASCHOFF", "OSKEBERG"),
]

RULES_POST_SHA = "18f8ea018234605002459cd98e59a9aaeddec235d8e022ff59f6ebd67766d607"
STORE_FILES = ("id_map.npy", "embeddings.faiss", "episodic.db")  # publish order


class Refuse(RuntimeError):
    pass


def log(msg: str = "") -> None:
    print(msg, flush=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(16 << 20), b""):
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------------- signature


def tokens(text) -> list[bytes]:
    if text is None:
        return []
    if isinstance(text, str):
        data = text.encode("utf-8", errors="replace")
    elif isinstance(text, (bytes, bytearray, memoryview)):
        data = bytes(text)
    else:
        data = str(text).encode("utf-8", errors="replace")
    return TOKEN_RE.findall(data.lower())


def _iter_jsonl(path: Path):
    with path.open("rb") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict) and not any(str(k).startswith("__") for k in row):
                yield row


def _row_texts(row: dict):
    expected = row.get("expected")
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        expected = str(expected)
    for text in (row.get("prompt"), expected):
        if isinstance(text, str) and text:
            yield text


def _is_leak_row(row: dict) -> bool:
    rid = str(row.get("id") or "")
    return rid in LEAK_IDS or bool(LEAK_ID_RE.search(rid))


@dataclass
class Signature:
    shingles: dict  # tuple[bytes,...] -> sorted list of source ids
    anchors: set
    vocab: set
    sources: list
    dropped_template: int
    digest: str

    @classmethod
    def build(cls, pool_paths: list[Path]) -> "Signature":
        cand: dict[tuple, set] = defaultdict(set)
        sources = []
        for path in pool_paths:
            for row in _iter_jsonl(path):
                if not _is_leak_row(row):
                    continue
                sources.append(str(row["id"]))
                for text in _row_texts(row):
                    t = tokens(text)
                    for i in range(len(t) - NGRAM + 1):
                        cand[tuple(t[i : i + NGRAM])].add(str(row["id"]))
        if not cand:
            raise Refuse(f"no leak rows found in the pool {pool_paths}; refusing to run blind")
        vocab = {tok for sh in cand for tok in sh}
        df: dict[tuple, int] = Counter()
        tf: Counter = Counter()
        row_no = 0
        for path in pool_paths:
            for row in _iter_jsonl(path):
                row_no += 1
                seen = set()
                for text in _row_texts(row):
                    t = tokens(text)
                    run = 0
                    for i, tok in enumerate(t):
                        if tok in vocab:
                            tf[tok] += 1
                            run += 1
                            if run >= NGRAM:
                                sh = tuple(t[i - NGRAM + 1 : i + 1])
                                if sh in cand:
                                    seen.add(sh)
                        else:
                            run = 0
                for sh in seen:
                    df[sh] += 1
        kept = {sh: sorted(ids) for sh, ids in cand.items() if df[sh] < TEMPLATE_DF}
        anchors = {
            tok
            for tok in vocab
            if len(tok) >= ANCHOR_MIN_BYTES and not tok.isdigit() and tf[tok] <= RARE_TF
        }
        digest = hashlib.sha256(
            json.dumps(sorted(b" ".join(sh).decode("utf-8", "replace") for sh in kept)).encode()
        ).hexdigest()
        return cls(
            shingles=kept,
            anchors=anchors,
            vocab={tok for sh in kept for tok in sh},
            sources=sorted(set(sources)),
            dropped_template=len(cand) - len(kept),
            digest=digest,
        )

    # cache -------------------------------------------------------------
    def to_json(self) -> dict:
        dec = lambda b: b.decode("latin-1")  # noqa: E731 - byte-exact round trip
        return {
            "shingles": [[[dec(t) for t in sh], ids] for sh, ids in sorted(self.shingles.items())],
            "anchors": sorted(dec(t) for t in self.anchors),
            "sources": self.sources,
            "dropped_template": self.dropped_template,
            "digest": self.digest,
        }

    @classmethod
    def from_json(cls, d: dict) -> "Signature":
        enc = lambda s: s.encode("latin-1")  # noqa: E731
        shingles = {tuple(enc(t) for t in sh): ids for sh, ids in d["shingles"]}
        return cls(
            shingles=shingles,
            anchors={enc(t) for t in d["anchors"]},
            vocab={tok for sh in shingles for tok in sh},
            sources=d["sources"],
            dropped_template=d["dropped_template"],
            digest=d["digest"],
        )

    # matching ----------------------------------------------------------
    def match(self, text) -> tuple[int, set]:
        """(counted hits, matched source ids) for one text. Keyword hits count as one."""
        t = tokens(text)
        hits = 0
        srcs: set = set()
        run_hits: list[tuple] = []
        run = 0
        prev_hit = -2

        def close_run():
            nonlocal hits
            if not run_hits:
                return
            long_run = len(run_hits) >= MIN_UNANCHORED_RUN
            for sh in run_hits:
                if long_run or any(tok in self.anchors for tok in sh):
                    hits += 1
                    srcs.update(self.shingles[sh])
            run_hits.clear()

        for i, tok in enumerate(t):
            if tok in KEYWORD_TOKENS:
                hits += 1
                srcs.add("keyword:" + tok.decode())
            if tok in self.vocab:
                run += 1
                if run >= NGRAM:
                    sh = tuple(t[i - NGRAM + 1 : i + 1])
                    if sh in self.shingles:
                        if prev_hit != i - 1:
                            close_run()
                        run_hits.append(sh)
                        prev_hit = i
            else:
                run = 0
        close_run()
        return hits, srcs


def load_signature(pool_paths: list[Path], cache: Path | None) -> Signature:
    key = [[str(p), p.stat().st_size, int(p.stat().st_mtime)] for p in pool_paths]
    if cache and cache.exists():
        try:
            d = json.loads(cache.read_text())
            if d.get("key") == key:
                return Signature.from_json(d["signature"])
        except Exception:
            pass
    log(f"[signature] building from {len(pool_paths)} pool file(s); this reads the pool twice ...")
    t0 = time.time()
    sig = Signature.build(pool_paths)
    log(f"[signature] built in {time.time() - t0:.0f}s")
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(".tmp")
        tmp.write_text(json.dumps({"key": key, "signature": sig.to_json()}))
        tmp.replace(cache)
    return sig


# --------------------------------------------------------------------------- scanning


_SIG: Signature | None = None


def _init_worker(sig_json: dict) -> None:
    global _SIG
    _SIG = Signature.from_json(sig_json)


def scan_db(db: Path, sig: Signature | None = None) -> dict:
    """Flagged rows per table: {table: [{"rowid", "id", "columns", "sources"}]}."""
    sig = sig or _SIG
    assert sig is not None
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.text_factory = bytes
    out: dict = {"db": str(db), "tables": {}, "row_counts": {}}
    try:
        tables = [
            r[0].decode()
            for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        ]
        for table in tables:
            cols = [
                (r[1].decode(), (r[2] or b"").decode().upper())
                for r in con.execute(f'PRAGMA table_info("{table}")')
            ]
            names = [c for c, _ in cols]
            text_cols = [c for c, typ in cols if typ in ("", "TEXT") or "CHAR" in typ or "CLOB" in typ]
            if not text_cols:
                continue
            has_id = "id" in names
            sel = ", ".join(f'"{c}"' for c in text_cols)
            flagged = []
            n = 0
            for row in con.execute(
                f'SELECT rowid, {"id" if has_id else "NULL"}, {sel} FROM "{table}"'
            ):
                n += 1
                hit_cols, srcs = [], set()
                for col, val in zip(text_cols, row[2:]):
                    if val is None or not isinstance(val, (bytes, str)):
                        continue
                    h, s = match_value(sig, val)
                    if h:
                        hit_cols.append(col)
                        srcs |= s
                if hit_cols:
                    rid = row[1]
                    flagged.append(
                        {
                            "rowid": row[0],
                            "id": rid.decode("utf-8", "replace") if isinstance(rid, bytes) else rid,
                            "columns": hit_cols,
                            "sources": sorted(srcs),
                        }
                    )
            out["row_counts"][table] = n
            if flagged:
                out["tables"][table] = flagged
    finally:
        con.close()
    return out


def raw_marker_count(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0
    with path.open("rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        return sum(len(rx.findall(mm)) for rx in RAW_MARKERS)


# --------------------------------------------------------------------------- discovery


@dataclass
class Store:
    db: Path
    faiss: Path | None
    id_map: Path | None
    kind: str  # live | pinned | checkpoint | archive | worktree
    key: str
    notes: list = field(default_factory=list)


def _pair(db: Path) -> tuple[Path | None, Path | None]:
    suffix = db.name[len("episodic.db") :]
    faiss = db.parent / f"embeddings.faiss{suffix}"
    id_map = db.parent / f"id_map.npy{suffix}"
    return (faiss if faiss.exists() else None, id_map if id_map.exists() else None)


def _is_sqlite(path: Path) -> bool:
    try:
        with path.open("rb") as fh:
            return fh.read(16) == b"SQLite format 3\x00"
    except OSError:
        return False


def discover_stores(orch: Path, worktrees: Path | None, backup_root: Path) -> list[Store]:
    live_dir = (orch / "orchestration/repl_memory/sessions").resolve()
    ckpt_root = (orch / "orchestration/autopilot_checkpoints").resolve()
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(orch / "orchestration"):
        if Path(dirpath).resolve() == backup_root.resolve():
            dirnames[:] = []
            continue
        for fn in filenames:
            if fn.startswith("episodic.db") and not fn.endswith(("-wal", "-shm", "-journal")):
                found.append(Path(dirpath) / fn)
    if worktrees and worktrees.is_dir():
        for wt in sorted(worktrees.iterdir()):
            p = wt / "orchestration/repl_memory/sessions/episodic.db"
            if p.is_file():
                found.append(p)
    stores = []
    seen_real: set = set()
    for db in sorted(set(found), key=lambda p: (len(p.parts), str(p))):
        if db.resolve() in seen_real:
            continue
        seen_real.add(db.resolve())
        if not _is_sqlite(db):
            continue
        faiss, id_map = _pair(db)
        rdb = db.resolve()
        if rdb == (live_dir / "episodic.db"):
            kind = "live"
        elif ckpt_root in rdb.parents:
            meta = db.parent / "checkpoint_meta.json"
            pinned = False
            if meta.exists():
                try:
                    pinned = "episodic.db" in (json.loads(meta.read_text()).get("file_sha256") or {})
                except Exception:
                    pinned = True  # unreadable meta: treat as pinned (fail closed)
            kind = "pinned" if pinned else "checkpoint"
        elif worktrees and worktrees.resolve() in rdb.parents:
            kind = "worktree"
        else:
            kind = "archive"
        rel = str(rdb).lstrip("/").replace("/", "__")
        stores.append(Store(db=db, faiss=faiss, id_map=id_map, kind=kind, key=rel))
    return stores


def discover_state_files(orch: Path) -> tuple[list[Path], list[Path]]:
    """(editable backups, read-only checkpoint states)."""
    od = orch / "orchestration"
    editable = set()
    for pattern in ("autopilot_state*.bak*", "autopilot_state*.pre*", "autopilot_state.json.run3-poisoned"):
        editable.update(od.glob(pattern))
    editable.update((od / "archived_backups").glob("autopilot_state*"))
    editable = {
        p for p in editable
        if p.is_file() and p.name not in ("autopilot_state.json", "autopilot_state.json.lock")
    }
    readonly = sorted((od / "autopilot_checkpoints").glob("*/autopilot_state.json"))
    readonly.append(od / "autopilot_state.json")
    return sorted(editable), [p for p in readonly if p.is_file()]


def match_value(sig: Signature, raw) -> tuple[int, set]:
    """Match a stored value. JSON is decoded first and its string leaves are matched one by one
    (escaped newlines would otherwise glue tokens together); the raw text is matched too."""
    if isinstance(raw, (bytes, bytearray, memoryview)):
        raw = bytes(raw).decode("utf-8", errors="replace")
    hits, srcs = sig.match(raw)
    stripped = raw.lstrip()[:1]
    if stripped not in ("{", "[", '"'):
        return hits, srcs
    try:
        data = json.loads(raw)
    except Exception:
        return hits, srcs
    leaf_hits = 0
    stack = [data]
    while stack:
        o = stack.pop()
        if isinstance(o, dict):
            stack.extend(o.keys())
            stack.extend(o.values())
        elif isinstance(o, list):
            stack.extend(o)
        elif isinstance(o, str):
            h, s = sig.match(o)
            leaf_hits += h
            srcs |= s
    return max(hits, leaf_hits), srcs


def scan_state_text(sig: Signature, raw: str) -> tuple[int, set]:
    return match_value(sig, raw)


# --------------------------------------------------------------------------- liveness guards


def _proc_cmdlines():
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            yield int(p.name), (p / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
        except OSError:
            continue


def _listening_ports() -> set[int]:
    ports = set()
    for f in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            lines = Path(f).read_text().splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            parts = line.split()
            if len(parts) > 3 and parts[3] == "0A":
                ports.add(int(parts[1].rsplit(":", 1)[1], 16))
    return ports


def open_handles(paths: list[Path]) -> list[str]:
    targets = {str(p.resolve()) for p in paths}
    targets |= {t + s for t in list(targets) for s in ("-wal", "-shm", "-journal")}
    holders = []
    for p in Path("/proc").iterdir():
        if not p.name.isdigit() or int(p.name) == os.getpid():
            continue
        try:
            for fd in (p / "fd").iterdir():
                try:
                    if os.readlink(fd) in targets:
                        holders.append(f"pid {p.name} holds {os.readlink(fd)}")
                except OSError:
                    continue
        except OSError:
            continue
    return holders


@contextmanager
def autopilot_lock(orch: Path):
    lock = orch / "orchestration/.autopilot.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    with open(lock, "a+") as fh:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise Refuse(f"AutoPilot is running (singleton lock {lock} is held)")
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


@contextmanager
def faiss_writer_lock(sessions: Path):
    path = sessions / ".episodic_faiss.lock"
    with open(path, "a+") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def _cwd(pid: int) -> str | None:
    try:
        return os.path.realpath(f"/proc/{pid}/cwd")
    except OSError:
        return None


def live_guard(store: Store, api_port: int, orch: Path) -> list[str]:
    """Reasons the live store must not be touched. An API/AutoPilot process counts when its cwd is
    this orchestrator tree, or its cwd is unreadable (fail closed)."""
    problems = []
    root = str(orch.resolve())
    for pid, cmd in _proc_cmdlines():
        if pid == os.getpid():
            continue
        is_api = "src.api:app" in cmd
        is_ap = bool(re.search(r"autopilot\.py\s+(start|run|resume)", cmd))
        if not (is_api or is_ap):
            continue
        cwd = _cwd(pid)
        if cwd is None or cwd == root or cwd.startswith(root + os.sep) or root in cmd:
            what = "orchestrator API" if is_api else "AutoPilot"
            problems.append(f"{what} process running: pid {pid} (cwd {cwd}): {cmd[:100]}")
    if api_port in _listening_ports():
        problems.append(f"a process is listening on :{api_port} (the orchestrator API)")
    files = [p for p in (store.db, store.faiss, store.id_map) if p]
    problems += open_handles(files)
    return problems


def print_live_procedure(script: Path, problems: list[str], orch: Path) -> None:
    wrapper = script.resolve().with_suffix(".sh")
    log("\n  LIVE STORE REFUSED:")
    for p in problems:
        log(f"    - {p}")
    log(
        f"""
  Run the live purge in a window with the API stopped (operator-controlled; AutoPilot must stay
  stopped the whole time):
    1. python3 {orch}/scripts/server/orchestrator_stack.py stop orchestrator
    2. curl -sf -m 3 http://127.0.0.1:8000/health && echo 'API STILL UP - stop' || echo 'API down'
    3. bash {wrapper} --apply --include-live
    4. bash {wrapper} --verify --include-live
    5. python3 {orch}/scripts/server/orchestrator_stack.py reload orchestrator
"""
    )


# --------------------------------------------------------------------------- store purge


def _load_repair_module():
    path = REPO_ROOT / "scripts/maintenance/repair_faiss_id_map.py"
    spec = importlib.util.spec_from_file_location("repair_faiss_id_map", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def integrity_check(stage: Path) -> dict | None:
    script = REPO_ROOT / "scripts/maintenance/check_episodic_integrity.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--sessions-dir", str(stage), "--json"],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    try:
        out = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {"ok": False, "error": (proc.stderr or proc.stdout)[-800:], "checks": []}
    out["exit"] = proc.returncode
    return out


def _check_status(report: dict | None) -> dict:
    if not report:
        return {}
    return {c["check"]: c.get("pass") for c in report.get("checks", [])}


def _copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".partial")
    shutil.copy2(src, tmp)
    with tmp.open("rb") as fh:
        os.fsync(fh.fileno())
    tmp.replace(dst)
    return sha256_file(dst)


def purge_store(store: Store, scan: dict, backup_root: Path, run_stamp: str) -> dict:
    """Backup, stage, purge, verify, publish. Returns the per-store receipt."""
    import faiss
    import numpy as np

    rec: dict = {"store": str(store.db), "kind": store.kind, "key": store.key, "notes": store.notes}
    for p in (store.db,):
        wal = p.with_name(p.name + "-wal")
        if wal.exists() and wal.stat().st_size:
            raise Refuse(f"{wal} is non-empty; checkpoint the WAL (with its owner stopped) first")

    files = {"episodic.db": store.db}
    has_pair = store.faiss is not None and store.id_map is not None
    if has_pair:
        files["embeddings.faiss"] = store.faiss
        files["id_map.npy"] = store.id_map
    elif store.faiss is not None or store.id_map is not None:
        rec["notes"].append(
            "incomplete FAISS pair (index without id_map or the reverse); vectors are not "
            "addressable by id, so only the DB rows are purged; the index file is left as is"
        )

    # 1. backup
    bdir = backup_root / store.key
    pre = {name: sha256_file(p) for name, p in files.items()}
    if bdir.exists():
        manifest = json.loads((bdir / "MANIFEST.json").read_text()) if (bdir / "MANIFEST.json").exists() else {}
        if manifest.get("preimage_sha256") != pre:
            bdir = backup_root / f"{store.key}.{pre['episodic.db'][:8]}"
    bdir.mkdir(parents=True, exist_ok=True)
    for name, p in files.items():
        dst = bdir / name
        if not (dst.exists() and sha256_file(dst) == pre[name]):
            if _copy(p, dst) != pre[name]:
                raise Refuse(f"backup of {p} does not match its source (file changed during copy?)")
    (bdir / "MANIFEST.json").write_text(
        json.dumps(
            {"source": {n: str(p) for n, p in files.items()}, "preimage_sha256": pre, "at": utc_now()},
            indent=2,
        )
    )
    rec["backup_dir"] = str(bdir)
    rec["preimage_sha256"] = pre
    log(f"    [backup] {bdir}")

    # 2. stage
    stage = backup_root / "stage" / store.key
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    for name in files:
        shutil.copy2(bdir / name, stage / name)

    repair = _load_repair_module() if has_pair else None
    before_diag = repair.diagnose(stage) if repair else None
    before_health = integrity_check(stage) if has_pair else None

    flagged = scan["tables"]
    del_ids = {r["id"] for rows in flagged.values() for r in rows if r["id"] is not None}
    con = sqlite3.connect(stage / "episodic.db")
    try:
        con.execute("PRAGMA secure_delete=ON")
        mem_ids_flagged = {r["id"] for r in flagged.get("memories", [])}
        # a removal id must not survive in memories under a non-flagged row
        if del_ids:
            q = ",".join("?" * len(del_ids))
            try:
                survivors = con.execute(
                    f"SELECT rowid, id FROM memories WHERE id IN ({q})", sorted(del_ids)
                ).fetchall()
            except sqlite3.OperationalError:
                survivors = []
            flagged_rowids = {r["rowid"] for r in flagged.get("memories", [])}
            bad = [s for s in survivors if s[0] not in flagged_rowids]
            if bad:
                raise Refuse(f"ids {sorted({b[1] for b in bad})} also label non-leaked memories rows")

        removed_positions: list[int] = []
        nulled = 0
        diag_removed = {"correct": 0, "wrong": 0, "missing": 0}
        if has_pair:
            index = faiss.read_index(str(stage / "embeddings.faiss"))
            if not isinstance(index, faiss.IndexFlatIP):
                raise Refuse(f"unexpected index type {type(index).__name__} (IndexFlatIP expected)")
            id_map = np.load(stage / "id_map.npy", allow_pickle=True).tolist()
            pos_of = {}
            for i, mid in enumerate(id_map):
                pos_of.setdefault(str(mid), []).append(i)
            removed_positions = sorted(p for mid in del_ids for p in pos_of.get(str(mid), []))
            rset = set(removed_positions)
            # classify removed memories rows the way diagnose() does (last occurrence wins)
            last_pos = {k: v[-1] for k, v in pos_of.items()}
            for r in flagged.get("memories", []):
                (ei,) = con.execute("SELECT embedding_idx FROM memories WHERE rowid=?", (r["rowid"],)).fetchone()
                tp = last_pos.get(str(r["id"]))
                diag_removed["missing" if tp is None else ("correct" if tp == ei else "wrong")] += 1

        for table, rows in flagged.items():
            rowids = [r["rowid"] for r in rows]
            for i in range(0, len(rowids), 500):
                chunk = rowids[i : i + 500]
                con.execute(f'DELETE FROM "{table}" WHERE rowid IN ({",".join("?" * len(chunk))})', chunk)
        prov_deleted = 0
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "_q_consolidation_provenance" in tables and mem_ids_flagged:
            q = ",".join("?" * len(mem_ids_flagged))
            prov_deleted = con.execute(
                f"DELETE FROM _q_consolidation_provenance WHERE consolidated_id IN ({q})",
                sorted(mem_ids_flagged),
            ).rowcount

        if removed_positions and "memories" in tables:
            nulled = con.execute(
                f"UPDATE memories SET embedding_idx = NULL WHERE embedding_idx IN "
                f"({','.join('?' * len(removed_positions))})",
                removed_positions,
            ).rowcount
            # shift = number of removed positions strictly below embedding_idx
            con.execute("CREATE TEMP TABLE _leak_removed(p INTEGER PRIMARY KEY)")
            con.executemany("INSERT INTO _leak_removed(p) VALUES (?)", [(p,) for p in removed_positions])
            con.execute(
                "UPDATE memories SET embedding_idx = embedding_idx - "
                "(SELECT count(*) FROM _leak_removed WHERE p < memories.embedding_idx) "
                "WHERE embedding_idx > ?",
                (removed_positions[0],),
            )
        con.commit()
        rec["deleted_rows"] = {t: len(r) for t, r in flagged.items()}
        rec["deleted_ids"] = sorted(i for i in del_ids if i is not None)
        rec["provenance_rows_deleted"] = prov_deleted
        rec["embedding_idx_nulled"] = nulled
    finally:
        con.close()

    if has_pair and removed_positions:
        ntotal = index.ntotal
        in_index = [p for p in removed_positions if p < ntotal]
        xb = faiss.rev_swig_ptr(index.get_xb(), ntotal * index.d).reshape(ntotal, index.d)
        keep_mask = np.ones(ntotal, dtype=bool)
        keep_mask[in_index] = False
        kept = np.ascontiguousarray(xb[keep_mask])
        new_index = faiss.IndexFlatIP(index.d)
        new_index.add(kept)
        new_id_map = [m for i, m in enumerate(id_map) if i not in rset]
        np.save(str(stage / "id_map.npy.tmp"), np.array(new_id_map, dtype=object), allow_pickle=True)
        os.replace(stage / "id_map.npy.tmp.npy", stage / "id_map.npy")
        faiss.write_index(new_index, str(stage / "embeddings.faiss"))
        # kept vectors must be byte-identical, in order
        check = faiss.read_index(str(stage / "embeddings.faiss"))
        cx = faiss.rev_swig_ptr(check.get_xb(), check.ntotal * check.d).reshape(check.ntotal, check.d)
        if not np.array_equal(cx, kept):
            raise Refuse("compacted index vectors differ from the kept originals")
        del xb, kept, cx
        rec["faiss"] = {
            "removed_positions": removed_positions,
            "ntotal_before": ntotal,
            "ntotal_after": check.ntotal,
            "id_map_before": len(id_map),
            "id_map_after": len(new_id_map),
        }
    elif has_pair:
        rec["faiss"] = {"removed_positions": [], "note": "no flagged id is in id_map"}

    # 3. verify on the stage
    ver: dict = {}
    c2 = sqlite3.connect(f"file:{stage / 'episodic.db'}?mode=ro", uri=True)
    ver["sqlite_integrity"] = c2.execute("PRAGMA integrity_check").fetchone()[0]
    c2.close()
    if ver["sqlite_integrity"] != "ok":
        raise Refuse(f"PRAGMA integrity_check: {ver['sqlite_integrity']}")
    rescan = scan_db(stage / "episodic.db")
    ver["rescan_flagged_rows"] = sum(len(v) for v in rescan["tables"].values())
    ver["raw_marker_hits"] = raw_marker_count(stage / "episodic.db")
    if ver["rescan_flagged_rows"] or ver["raw_marker_hits"]:
        raise Refuse(f"leak still present on stage: {ver}")
    if has_pair:
        after_diag = repair.diagnose(stage)
        k = len(removed_positions)
        k_idx = len([p for p in removed_positions if p < before_diag["ntotal"]])
        expect = {
            "ntotal": before_diag["ntotal"] - k_idx,
            "id_map_len": before_diag["id_map_len"] - k,
            "desync": before_diag["desync"] - k_idx + k,
            "db_rows": before_diag["db_rows"] - len(flagged.get("memories", [])),
            "embedding_idx_correct": before_diag["embedding_idx_correct"] - diag_removed["correct"],
            "rows_missing_from_id_map": before_diag["rows_missing_from_id_map"] - diag_removed["missing"],
        }
        diffs = {kk: (after_diag[kk], vv) for kk, vv in expect.items() if after_diag[kk] != vv}
        # wrong pointers: removed wrong rows go; nulled rows leave the "wrong" bucket too
        wrong_expect = before_diag["embedding_idx_wrong"] - diag_removed["wrong"]
        if not (wrong_expect - nulled <= after_diag["embedding_idx_wrong"] <= wrong_expect):
            diffs["embedding_idx_wrong"] = (after_diag["embedding_idx_wrong"], wrong_expect)
        ver["diagnose_before"] = before_diag
        ver["diagnose_after"] = after_diag
        if diffs:
            raise Refuse(f"FAISS/id_map consistency moved beyond the removed rows: {diffs}")
        after_health = integrity_check(stage)
        ver["health_before"] = _check_status(before_health)
        ver["health_after"] = _check_status(after_health)
        ver["health_after_ok"] = bool(after_health and after_health.get("ok"))
        if store.kind == "live" and not ver["health_after_ok"]:
            raise Refuse(f"integrity check fails on the purged live store: {after_health}")
        worse = [
            c for c, ok in ver["health_after"].items()
            if ok is False and ver["health_before"].get(c) is not False
        ]
        if worse:
            raise Refuse(f"integrity checks newly failing after purge: {worse}")
    rec["verify"] = ver

    # 4. publish
    post = {name: sha256_file(stage / name) for name in files}
    rec["postimage_sha256"] = post
    ctx = faiss_writer_lock(store.db.parent) if store.kind == "live" else _null_ctx()
    with ctx:
        for name, p in files.items():
            if sha256_file(p) != pre[name]:
                raise Refuse(f"{p} changed since the backup; nothing published")
        done = []
        try:
            for name in STORE_FILES:
                if name not in files:
                    continue
                target = files[name]
                tmp = target.with_name(f".{target.name}.leakpurge-{run_stamp}.tmp")
                shutil.copy2(stage / name, tmp)
                with tmp.open("rb") as fh:
                    os.fsync(fh.fileno())
                os.replace(tmp, target)
                done.append(name)
            for name, p in files.items():
                if sha256_file(p) != post[name]:
                    raise Refuse(f"published {p} does not match the verified stage")
        except BaseException:
            for name in files:
                t = files[name]
                t.with_name(f".{t.name}.leakpurge-{run_stamp}.tmp").unlink(missing_ok=True)
            for name in done:
                target = files[name]
                tmp = target.with_name(f".{target.name}.leakpurge-restore.tmp")
                shutil.copy2(bdir / name, tmp)
                os.replace(tmp, target)
            rec["rolled_back"] = done
            raise
    shutil.rmtree(stage, ignore_errors=True)
    try:
        stage.parent.rmdir()  # only when empty
    except OSError:
        pass
    rec["status"] = "PURGED"
    return rec


@contextmanager
def _null_ctx():
    yield


# --------------------------------------------------------------------------- state redaction


def redact_state(path: Path, sig: Signature, backup_root: Path) -> dict:
    raw = path.read_text(encoding="utf-8")
    pre_sha = sha256_file(path)
    new = raw
    counts = {}
    for old, rep in STATE_REPLACEMENTS:
        n = new.count(old)
        if n:
            counts[old] = n
            new = new.replace(old, rep)
    json.loads(new)  # must stay valid
    hits, srcs = scan_state_text(sig, new)
    if hits or re.search(r"(?i)aschoff", new):
        raise Refuse(f"{path}: redaction leaves {hits} hit(s) {sorted(srcs)}; needs manual review")
    bdir = backup_root / "state" / str(path.resolve()).lstrip("/").replace("/", "__")
    dst = bdir / f"{path.name}.{pre_sha[:12]}"
    if not (dst.exists() and sha256_file(dst) == pre_sha):
        if _copy(path, dst) != pre_sha:
            raise Refuse(f"backup of {path} does not match its source")
    tmp = path.with_name(f".{path.name}.leakpurge.tmp")
    tmp.write_text(new, encoding="utf-8")
    shutil.copymode(path, tmp)
    os.utime(tmp, (path.stat().st_atime, path.stat().st_mtime))  # keep the backup's timestamp
    if sha256_file(path) != pre_sha:
        tmp.unlink()
        raise Refuse(f"{path} changed during redaction")
    os.replace(tmp, path)
    return {
        "file": str(path),
        "status": "REDACTED",
        "replacements": counts,
        "backup": str(dst),
        "preimage_sha256": pre_sha,
        "postimage_sha256": sha256_file(path),
    }


# --------------------------------------------------------------------------- main


def append_receipt(backup_root: Path, run: dict) -> Path:
    backup_root.mkdir(parents=True, exist_ok=True)
    path = backup_root / "receipt.json"
    data = {"schema": SCHEMA, "gate_id": GATE_ID, "runs": []}
    if path.exists():
        data = json.loads(path.read_text())
    data["runs"].append(run)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, default=str))
    tmp.replace(path)
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true", help="purge (default is a dry-run)")
    mode.add_argument("--verify", action="store_true", help="read-only post-state check; exit 1 on any leak")
    ap.add_argument("--include-live", action="store_true", help="also the live store (API + AutoPilot stopped)")
    ap.add_argument("--include-pinned", action="store_true", help="also manifest-pinned checkpoints (v10)")
    ap.add_argument("--orch-root", type=Path, default=Path(os.environ.get("ORCH", DEFAULT_ORCH)))
    ap.add_argument("--worktrees", type=Path, default=DEFAULT_WORKTREES, help="'' to skip")
    ap.add_argument("--backup-root", type=Path, default=Path(os.environ.get("BACKUP_ROOT", DEFAULT_BACKUP_ROOT)))
    ap.add_argument("--pool", type=Path, action="append", help="eval pool jsonl (repeatable)")
    ap.add_argument("--api-port", type=int, default=8000)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--json", type=Path, help="also write the full scan report here")
    args = ap.parse_args()

    orch = args.orch_root.resolve()
    backup_root = args.backup_root
    worktrees = args.worktrees if str(args.worktrees) not in ("", ".") else None
    pools = args.pool or [DEFAULT_POOL, *sorted((orch / "benchmarks/prompts").glob("core_*.jsonl"))]
    mode_name = "apply" if args.apply else ("verify" if args.verify else "dry-run")
    run_stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    log(f"=== {GATE_ID}  mode={mode_name}  orch={orch}  backups={backup_root}")

    cache = backup_root / "signature-cache.json"
    sig = load_signature([p for p in pools if p.exists()], cache)
    global _SIG
    _SIG = sig
    log(
        f"[signature] {len(sig.shingles)} leak shingles from {len(sig.sources)} pool rows "
        f"({sig.dropped_template} template shingles dropped at DF>={TEMPLATE_DF}), "
        f"{len(sig.anchors)} anchor tokens, digest {sig.digest[:16]}"
    )

    stores = discover_stores(orch, worktrees, backup_root)
    state_edit, state_ro = discover_state_files(orch)
    log(f"[discover] {len(stores)} episodic.db file(s), {len(state_edit)} state backup(s), "
        f"{len(state_ro)} read-only state file(s)")

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=max(1, args.jobs), initializer=_init_worker,
                             initargs=(sig.to_json(),)) as ex:
        scans = dict(zip([s.key for s in stores], ex.map(scan_db, [s.db for s in stores])))
    log(f"[scan] episodic stores scanned in {time.time() - t0:.0f}s\n")

    run: dict = {"at": utc_now(), "mode": mode_name, "signature_digest": sig.digest,
                 "signature_sources": sig.sources, "stores": [], "state_files": [],
                 "include_live": args.include_live, "include_pinned": args.include_pinned}
    rc = 0
    dirty_in_scope = 0

    pinned_ok = True
    if args.include_pinned:
        for s in stores:
            if s.kind != "pinned":
                continue
            meta = json.loads((s.db.parent / "checkpoint_meta.json").read_text())
            if meta.get("file_sha256", {}).get("prompts/rules.md") not in (None, RULES_POST_SHA):
                pinned_ok = False
                s.notes.append("RATIFY-CKPT-PROMPT-LEAK-20260916 not applied yet (rules.md still pre-state)")

    proposal = {}
    log(f"{'KIND':<10} {'STATUS':<9} {'ROWS':>5}  STORE")
    for s in stores:
        sc = scans[s.key]
        n = sum(len(v) for v in sc["tables"].values())
        pair = "pair" if (s.faiss and s.id_map) else ("index-only" if s.faiss else "db-only")
        in_scope = (
            s.kind in ("checkpoint", "archive", "worktree")
            or (s.kind == "live" and args.include_live)
            or (s.kind == "pinned" and args.include_pinned)
        )
        status = "CLEAN" if n == 0 else ("DIRTY" if in_scope else "OUT")
        log(f"{s.kind:<10} {status:<9} {n:>5}  {s.db}  [{pair}]")
        for table, rows in sc["tables"].items():
            for r in rows:
                log(f"             {table} rowid={r['rowid']} id={r['id']} cols={r['columns']} "
                    f"src={','.join(r['sources'])[:90]}")
        entry = {"store": str(s.db), "kind": s.kind, "pair": pair, "flagged": sc["tables"],
                 "row_counts": sc["row_counts"], "status": status}
        if n and in_scope:
            dirty_in_scope += 1
            if args.apply:
                try:
                    if s.kind == "live":
                        problems = live_guard(s, args.api_port, orch)
                        if problems:
                            print_live_procedure(Path(__file__), problems, orch)
                            entry["status"] = "REFUSED-LIVE"
                            entry["problems"] = problems
                            rc = max(rc, 3)
                            run["stores"].append(entry)
                            continue
                        with autopilot_lock(orch):
                            problems = live_guard(s, args.api_port, orch)
                            if problems:
                                raise Refuse(f"live guard failed under the AutoPilot lock: {problems}")
                            entry.update(purge_store(s, sc, backup_root, run_stamp))
                    elif s.kind == "pinned":
                        if not pinned_ok:
                            raise Refuse("; ".join(s.notes))
                        with autopilot_lock(orch):
                            entry.update(purge_store(s, sc, backup_root, run_stamp))
                        proposal[str(s.db.parent)] = {
                            "file_sha256_before": {k: v for k, v in entry["preimage_sha256"].items()},
                            "file_sha256_after": {k: v for k, v in entry["postimage_sha256"].items()},
                            "reason": f"{GATE_ID}: HumanEval/55 / Aschoff memories removed",
                        }
                    else:
                        with autopilot_lock(orch):
                            entry.update(purge_store(s, sc, backup_root, run_stamp))
                    log(f"             -> PURGED {entry.get('deleted_rows')} "
                        f"faiss_removed={len(entry.get('faiss', {}).get('removed_positions', []))} "
                        f"health_after_ok={entry['verify'].get('health_after_ok')}")
                except Refuse as exc:
                    entry["status"] = "FAILED"
                    entry["error"] = str(exc)
                    log(f"             -> FAILED: {exc}")
                    rc = max(rc, 1)
        elif n and args.verify and s.kind in ("live", "pinned") and not in_scope:
            pass  # out of the requested verify scope; reported as OUT above
        entry.setdefault("notes", s.notes)
        run["stores"].append(entry)

    log(f"\n{'STATE':<10} {'STATUS':<9} {'HITS':>5}  FILE")
    for path in state_edit:
        hits, srcs = scan_state_text(sig, path.read_text(encoding="utf-8", errors="replace"))
        entry = {"file": str(path), "hits": hits, "sources": sorted(srcs), "status": "CLEAN" if not hits else "DIRTY"}
        log(f"{'backup':<10} {entry['status']:<9} {hits:>5}  {path}")
        if hits:
            dirty_in_scope += 1
            if args.apply:
                try:
                    entry.update(redact_state(path, sig, backup_root))
                    log(f"             -> REDACTED {entry['replacements']}")
                except Refuse as exc:
                    entry["status"] = "FAILED"
                    entry["error"] = str(exc)
                    log(f"             -> FAILED: {exc}")
                    rc = max(rc, 1)
        run["state_files"].append(entry)
    for path in state_ro:
        hits, srcs = scan_state_text(sig, path.read_text(encoding="utf-8", errors="replace"))
        log(f"{'read-only':<10} {'CLEAN' if not hits else 'REPORT':<9} {hits:>5}  {path}")
        run["state_files"].append({"file": str(path), "hits": hits, "sources": sorted(srcs),
                                   "status": "CLEAN" if not hits else "REPORT-ONLY"})

    if proposal:
        pp = backup_root / "pinned_repin_proposal.json"
        pp.write_text(json.dumps(proposal, indent=2, sort_keys=True))
        log(f"\n[pinned] new checkpoint file hashes for the operator re-pin amendment: {pp}")
        run["pinned_repin_proposal"] = str(pp)

    if args.verify:
        remaining = [e for e in run["stores"] if e["status"] == "DIRTY"]
        remaining += [e for e in run["state_files"] if e["status"] == "DIRTY"]
        health = []
        for s in stores:
            if s.kind == "live" and args.include_live and s.faiss and s.id_map:
                rep = integrity_check(s.db.parent)
                health.append(rep)
                log(f"\n[verify] live integrity check: {'HEALTHY' if rep.get('ok') else 'DEGRADED'}")
                for c in rep.get("checks", []):
                    log(f"           {c['check']}: {c.get('pass')}  {c['detail'][:110]}")
                if not rep.get("ok"):
                    rc = 1
            if s.kind == "live" and args.include_live:
                raw = raw_marker_count(s.db)
                log(f"[verify] live raw marker hits: {raw}")
                if raw:
                    rc = 1
        run["verify_health"] = health
        if remaining:
            rc = 1
        log(f"\n[verify] {'PASS' if rc == 0 else 'FAIL'}: {len(remaining)} in-scope item(s) still carry the leak")
    elif not args.apply:
        log(f"\nDRY RUN: {dirty_in_scope} in-scope item(s) would be purged/redacted. Re-run with --apply.")

    if args.json:
        args.json.write_text(json.dumps(run, indent=2, default=str))
    if args.apply or args.verify:
        run["exit"] = rc
        rp = append_receipt(backup_root, run)
        log(f"[receipt] {rp}")
    return rc


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Refuse as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        sys.exit(2)
