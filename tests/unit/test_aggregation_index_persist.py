"""AGG-1.5 r3 follow-up: the ``l1_passage_bm25`` index persists next to the store and a fresh process LOADS it.

Zero model, zero server. Keyed by (corpus, content hash, builder version); atomic write; integrity-checked; a hash
change invalidates; ``l1_bm25`` is unchanged; the manifest records ``built`` vs ``loaded``.
"""

from __future__ import annotations

import gzip
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from src.aggregation import FLAG_ENV
from src.aggregation import registry as R
from src.aggregation.registry import IndexRegistry, PassageHandle
from src.aggregation.store import CorpusStore
from src.models.document import DocumentPreprocessResult, Section

REPO_ROOT = Path(__file__).resolve().parents[2]
QUERIES = ("Total Current Assets", "pension trusts contributions", "Cash and cash equivalents",
           "signatures exhibits", "no_such_token_zzz")


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setenv(FLAG_ENV, "1")


def _doc(extra: str = "") -> DocumentPreprocessResult:
    head = "Funding. We make contributions to pension trusts.\n" * 40
    body = (head + "Cash and cash equivalents 4,954 5,596\nTotal Current Assets 21,539 20,000\n"
            "Total Assets 92,187 92,187\n" + head + extra)
    secs = [Section(id="s1", title="Overview", level=1, content="Intro text about the business.\n",
                    page_start=1, page_end=1),
            Section(id="s2", title="Funding", level=1, content=body, page_start=2, page_end=2),
            Section(id="s3", title="Tail", level=1, content="Signatures and exhibits.\n", page_start=3, page_end=3)]
    return DocumentPreprocessResult(original_path="/x/b.pdf", sections=secs, figures=[], total_pages=3, tables=[])


def _store(tmp_path: Path, extra: str = "") -> CorpusStore:
    st = CorpusStore(tmp_path / "store.sqlite")
    st.ingest(_doc(extra), corpus_id="c", doc_id="LONG_2022_10K", entity="LONG")
    return st


def _hits(h: PassageHandle) -> list:
    return [[(x["chunk_id"], x["start"], x["end"], x["line"], x["score"]) for x in h.search(q, 8)] for q in QUERIES]


def _file(reg: IndexRegistry, st: CorpusStore) -> Path:
    p = reg.index_path("c", "l1_passage_bm25", st.corpus_content_hash("c"))
    assert p is not None
    return p


def test_second_registry_loads_and_matches_exactly(on, tmp_path):
    st = _store(tmp_path)
    r1 = IndexRegistry(st)
    h1 = r1.get_or_build("c", "l1_passage_bm25")
    assert (r1.built, r1.loaded, h1.origin) == (1, 0, "built")
    p = _file(r1, st)
    assert p.exists() and p.parent == Path(st.path).with_name("store.sqlite.indexes")
    assert not list(p.parent.glob("*.tmp-*"))  # atomic write leaves no temp file

    r2 = IndexRegistry(st)  # a new registry over the same store: nothing in memory
    h2 = r2.get_or_build("c", "l1_passage_bm25")
    assert (r2.built, r2.loaded, h2.origin) == (0, 1, "loaded")
    assert h2.text == h1.text and h2.chunk_ids == h1.chunk_ids and h2.chunk_starts == h1.chunk_starts
    assert _hits(h2) == _hits(h1)  # same chunk, span, line AND bit-identical score
    for x in h2.search("Total Current Assets", 3):
        assert h2.span(x["start"], x["end"]) == h1.span(x["start"], x["end"])
    assert r2.get_or_build("c", "l1_passage_bm25") is h2  # in-memory hit counts as neither
    assert (r2.built, r2.loaded) == (0, 1) and len(r2.events) == 1


def test_manifest_records_built_vs_loaded(on, tmp_path):
    st = _store(tmp_path)
    IndexRegistry(st).get_or_build("c", "l1_passage_bm25")
    (m,) = [x for x in st.manifest("c") if x["builder"] == "l1_passage_bm25"]
    assert (m["origin"], m["n_built"], m["n_loaded"], m["layer"]) == ("built", 1, 0, "L1")
    IndexRegistry(st).get_or_build("c", "l1_passage_bm25")
    IndexRegistry(st).get_or_build("c", "l1_passage_bm25")
    (m,) = [x for x in st.manifest("c") if x["builder"] == "l1_passage_bm25"]
    assert (m["origin"], m["n_built"], m["n_loaded"]) == ("loaded", 1, 2)
    assert st.built_layers("c") == {"L1"}


def test_legacy_manifest_without_counters_is_migrated(on, tmp_path):
    import sqlite3

    path = tmp_path / "old.sqlite"
    c = sqlite3.connect(path)
    c.execute("""CREATE TABLE index_manifest (corpus_id TEXT NOT NULL, content_hash TEXT NOT NULL,
        builder TEXT NOT NULL, model_id TEXT NOT NULL, layer TEXT, n_items INTEGER, built_at DOUBLE,
        PRIMARY KEY (corpus_id, content_hash, builder, model_id))""")
    c.execute("INSERT INTO index_manifest VALUES ('c','h','l1_bm25','','L1',3,1.0)")
    c.commit()
    c.close()
    st = CorpusStore(path)
    (m,) = st.manifest("c")
    assert (m["origin"], m["n_built"], m["n_loaded"]) == ("built", 1, 0)
    st.record_index("c", "h", "l1_bm25", "", "L1", 3, origin="loaded")
    (m,) = st.manifest("c")
    assert (m["origin"], m["n_built"], m["n_loaded"]) == ("loaded", 1, 1)
    with pytest.raises(ValueError):
        st.record_index("c", "h", "l1_bm25", "", "L1", 3, origin="cached")


def test_content_hash_change_invalidates_and_prunes(on, tmp_path):
    st = _store(tmp_path)
    r1 = IndexRegistry(st)
    h1 = r1.get_or_build("c", "l1_passage_bm25")
    old = _file(r1, st)
    # a changed document: same doc_id, new content -> new corpus content hash
    st.ingest(_doc(extra="Brand new Quarterly Dividend Declared 0.77\n"), corpus_id="c", doc_id="LONG_2022_10K",
              entity="LONG")
    r2 = IndexRegistry(st)
    h2 = r2.get_or_build("c", "l1_passage_bm25")
    new = _file(r2, st)
    assert new != old and (r2.built, r2.loaded) == (1, 0) and h2.origin == "built"
    assert not old.exists() and new.exists()  # the stale file is pruned
    assert h2.text != h1.text and "Quarterly Dividend" in h2.text
    assert h2.search("Quarterly Dividend Declared", 1)[0]["score"] > 0
    assert IndexRegistry(st).get_or_build("c", "l1_passage_bm25").origin == "loaded"  # and the new file serves


def _tamper(path: Path, fn) -> None:
    head, _, payload = gzip.decompress(path.read_bytes()).partition(b"\n")
    h, payload = fn(json.loads(head), payload)
    path.write_bytes(gzip.compress(json.dumps(h, sort_keys=True, separators=(",", ":")).encode() + b"\n" + payload))


def _flip_payload(h, payload):
    body = json.loads(payload)
    body["tf"][0]["injected_term"] = 9  # valid JSON, valid gzip, but not what was hashed
    return h, json.dumps(body, sort_keys=True, separators=(",", ":")).encode()


@pytest.mark.parametrize("how,reason", [
    ("garbage", "unreadable"),
    ("truncated", "unreadable"),
    ("payload", "payload_integrity"),
    ("version", "builder_version_mismatch"),
    ("engine", "engine_fp_mismatch"),
    ("hash", "content_hash_mismatch"),
    ("text", "text_sha256_mismatch"),
])
def test_corrupt_or_foreign_file_is_refused_and_rebuilt(on, tmp_path, how, reason):
    st = _store(tmp_path)
    r1 = IndexRegistry(st)
    good = r1.get_or_build("c", "l1_passage_bm25")
    p = _file(r1, st)
    if how == "garbage":
        p.write_bytes(b"not a gzip file")
    elif how == "truncated":
        p.write_bytes(p.read_bytes()[:40])
    elif how == "payload":
        _tamper(p, _flip_payload)
    else:
        key = {"version": "builder_version", "engine": "engine_fp", "hash": "content_hash",
               "text": "text_sha256"}[how]
        _tamper(p, lambda h, pl: ({**h, key: "x" * 8}, pl))
    r2 = IndexRegistry(st)
    h = r2.get_or_build("c", "l1_passage_bm25")
    assert (r2.built, r2.loaded) == (1, 0) and h.origin == "built"
    assert len(r2.invalid) == 1 and r2.invalid[0]["reason"].startswith(reason)
    assert _hits(h) == _hits(good)  # the rebuild is the correct index, not the bad file
    r3 = IndexRegistry(st)  # and the rebuild repaired the file
    assert r3.get_or_build("c", "l1_passage_bm25").origin == "loaded" and not r3.invalid


def test_failed_write_never_leaves_a_partial_file_and_does_not_fail_the_build(on, tmp_path, monkeypatch):
    st = _store(tmp_path)
    real = os.replace

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(R.os, "replace", boom)
    r1 = IndexRegistry(st)
    h = r1.get_or_build("c", "l1_passage_bm25")
    monkeypatch.setattr(R.os, "replace", real)
    assert h.origin == "built" and len(r1.persist_errors) == 1
    d = r1.index_dir()
    assert not _file(r1, st).exists()
    assert not d.exists() or not list(d.glob("*"))  # no final file, no temp file
    assert IndexRegistry(st).get_or_build("c", "l1_passage_bm25").origin == "built"  # next process builds, then persists
    assert IndexRegistry(st).get_or_build("c", "l1_passage_bm25").origin == "loaded"


def test_memory_store_and_persist_off_write_nothing(on, tmp_path):
    mem = CorpusStore()
    mem.ingest(_doc(), corpus_id="c", doc_id="LONG_2022_10K", entity="LONG")
    r = IndexRegistry(mem)
    assert r.persist is False and r.index_dir() is None
    r.get_or_build("c", "l1_passage_bm25")
    assert (r.built, r.loaded) == (1, 0)
    st = _store(tmp_path)
    off = IndexRegistry(st, persist=False)
    off.get_or_build("c", "l1_passage_bm25")
    assert not Path(str(st.path) + R.INDEX_DIR_SUFFIX).exists()


def test_l1_bm25_unchanged_never_persists(on, tmp_path):
    st = _store(tmp_path)
    r1 = IndexRegistry(st)
    h = r1.get_or_build("c", "l1_bm25")
    assert (r1.built, r1.loaded) == (1, 0)
    assert not Path(str(st.path) + R.INDEX_DIR_SUFFIX).exists()
    r2 = IndexRegistry(st)
    h2 = r2.get_or_build("c", "l1_bm25")
    assert (r2.built, r2.loaded) == (1, 0)  # rebuilt in the new process, exactly as before
    assert h.search("Total Current Assets", 3) == h2.search("Total Current Assets", 3)
    assert [m["origin"] for m in st.manifest("c")] == ["built"]


def test_engine_selfcheck_reproduces_a_live_index():
    assert R._engine_fingerprint() is not None  # restored state == live state on the probe, else persistence is off


_CHILD = r"""
import json, os, sys
sys.path.insert(0, {root!r})
from src.aggregation import registry as R
from src.aggregation.registry import IndexRegistry
from src.aggregation.store import CorpusStore
from src.repl_environment import context_search as cs

store_path, expect = sys.argv[1], sys.argv[2]
if expect == "load":
    def _no_build(self, *a, **k):
        raise AssertionError("ContextSearchIndex was constructed from text in the loading process")
    # the restore path builds an EMPTY index and fills it; a text build has sections -> forbid it
    real = cs.ContextSearchIndex.__init__
    def guarded(self, sections, *a, **k):
        if any(s.name != "probe" for s in sections):  # the 3 KB engine self-check probe is not a corpus build
            _no_build(self)
        return real(self, sections, *a, **k)
    cs.ContextSearchIndex.__init__ = guarded
st = CorpusStore(store_path)
reg = IndexRegistry(st)
h = reg.get_or_build("c", "l1_passage_bm25")
queries = json.loads(sys.argv[3])
out = {{"pid": os.getpid(), "built": reg.built, "loaded": reg.loaded, "origin": h.origin,
        "invalid": reg.invalid,
        "hits": [[[x["chunk_id"], x["start"], x["end"], x["line"], x["score"]] for x in h.search(q, 8)]
                 for q in queries],
        "manifest": [m for m in st.manifest("c") if m["builder"] == "l1_passage_bm25"]}}
print(json.dumps(out))
"""


def _run_child(store: Path, expect: str) -> dict:
    env = {**os.environ, FLAG_ENV: "1", "PYTHONPATH": str(REPO_ROOT)}
    cp = subprocess.run([sys.executable, "-c", _CHILD.format(root=str(REPO_ROOT)), str(store), expect,
                         json.dumps(QUERIES)], capture_output=True, text=True, env=env, cwd=str(REPO_ROOT),
                        timeout=300)
    assert cp.returncode == 0, cp.stderr[-2000:]
    return json.loads(cp.stdout.strip().splitlines()[-1])


def test_two_processes_process2_builds_zero_and_returns_identical_results(tmp_path):
    st = _store(tmp_path)
    st.close()
    store = tmp_path / "store.sqlite"
    p1 = _run_child(store, "build")
    p2 = _run_child(store, "load")
    assert p1["pid"] != p2["pid"]
    assert (p1["built"], p1["loaded"], p1["origin"]) == (1, 0, "built")
    assert (p2["built"], p2["loaded"], p2["origin"]) == (0, 1, "loaded") and p2["invalid"] == []
    assert p2["hits"] == p1["hits"] and any(p1["hits"])  # identical chunk, span, line and score, in order
    (m,) = p2["manifest"]
    assert (m["origin"], m["n_built"], m["n_loaded"]) == ("loaded", 1, 1)
    # and a third process sees the same file and still builds nothing
    p3 = _run_child(store, "load")
    assert (p3["built"], p3["loaded"]) == (0, 1) and p3["hits"] == p1["hits"]
