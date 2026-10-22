"""opencode mining readers use the operator's READ-ONLY snapshots; live stores are refused (2026-10-22 ruling)."""
import importlib.util
import json
import re
import sqlite3
import sys
from pathlib import Path

import pytest

ANALYSIS = Path(__file__).resolve().parents[2] / "scripts/analysis"
sys.path.insert(0, str(ANALYSIS))
import opencode_ro  # noqa: E402


def _load(name: str, fname: str):
    spec = importlib.util.spec_from_file_location(name, ANALYSIS / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


kpf = _load("kpf_lb1_ro", "kpf_lb1_cache_by_turn.py")
ufh = _load("ufh14_ro", "ufh14_lb9_retokenize_gap.py")
toc = _load("toc_sp_ro", "toc_sp_lb1_repeat_calls.py")

LIVE = ["~/.local/share/opencode/opencode.db", "/home/node/.local/share/opencode/opencode.db",
        "/mnt/raid0/llm/opencode/epyc/opencode.db"]


def _mkdb(path: Path, sid: str, provider: str = "qwen-gpu") -> str:
    c = sqlite3.connect(path)
    c.executescript("""
      CREATE TABLE session (id TEXT PRIMARY KEY, model TEXT, time_updated INTEGER);
      CREATE TABLE message (id TEXT PRIMARY KEY, session_id TEXT, time_created INTEGER, data TEXT);
      CREATE TABLE part (id TEXT PRIMARY KEY, message_id TEXT, session_id TEXT, time_created INTEGER, data TEXT);
      CREATE TABLE event (id TEXT PRIMARY KEY, data TEXT);
    """)
    c.execute("INSERT INTO session VALUES (?, ?, 1)", (sid, json.dumps({"providerID": provider, "id": "m"}, separators=(",", ":"))))
    c.execute("INSERT INTO message VALUES ('m1', ?, 1, ?)",
              (sid, json.dumps({"role": "assistant", "tokens": {"input": 10, "cache": {"read": 90}}})))
    c.execute("INSERT INTO part VALUES ('p1', 'm1', ?, 1, ?)",
              (sid, json.dumps({"type": "tool", "tool": "bash", "state": {"input": {"c": 1}, "output": "o", "status": "completed"}})))
    c.commit()
    c.close()
    return str(path)


def test_defaults_are_both_snapshots():
    assert opencode_ro.OPENCODE_RO_DBS == {"epyc": "/mnt/raid0/llm/opencode-db/epyc.db",
                                           "host": "/mnt/raid0/llm/opencode-db/host-daniele.db"}
    assert kpf.DBS is opencode_ro.OPENCODE_RO_DBS and ufh.DEFAULT_DBS is opencode_ro.OPENCODE_RO_DBS
    assert [s for s, _ in opencode_ro.resolve_dbs(None)] in (["epyc", "host"], ["epyc"], ["host"], [])
    for p in opencode_ro.OPENCODE_RO_DBS.values():
        assert not opencode_ro.is_live_opencode_path(p)


@pytest.mark.parametrize("live", LIVE)
def test_live_paths_refused_everywhere(live):
    assert opencode_ro.is_live_opencode_path(live)
    with pytest.raises(opencode_ro.LiveOpencodeStoreError, match="LIVE opencode store"):
        opencode_ro.open_ro(live)
    with pytest.raises(opencode_ro.LiveOpencodeStoreError):
        opencode_ro.resolve_dbs([live])
    with pytest.raises(opencode_ro.LiveOpencodeStoreError):
        kpf.opencode_table(live, ["qwen-gpu"], 1, 1)
    with pytest.raises(opencode_ro.LiveOpencodeStoreError):
        list(toc.opencode_sessions(live, ["qwen-gpu"], 1, 1))
    with pytest.raises(opencode_ro.LiveOpencodeStoreError):
        ufh.main(["--db", live])


def test_symlink_into_live_refused(tmp_path):
    link = tmp_path / "x.db"
    link.symlink_to("/mnt/raid0/llm/opencode/epyc/opencode.db")
    with pytest.raises(opencode_ro.LiveOpencodeStoreError):
        opencode_ro.open_ro(link)


def test_open_ro_query_only_and_retry(tmp_path):
    db = _mkdb(tmp_path / "a.db", "s1")
    conn = opencode_ro.open_ro(db)
    with pytest.raises(sqlite3.OperationalError):
        conn.execute("DELETE FROM session")
    conn.close()
    calls = []

    def flaky(target, **kw):
        calls.append(target)
        if len(calls) == 1:
            raise sqlite3.OperationalError("unable to open database file")
        return sqlite3.connect(target, **kw)

    c2 = opencode_ro.open_ro(db, retry_delay=0, _connect=flaky)
    assert len(calls) == 2 and calls[0].endswith("?mode=ro")
    c2.close()

    calls.clear()

    def dead(target, **kw):
        calls.append(target)
        raise sqlite3.OperationalError("unable to open database file")

    with pytest.raises(sqlite3.OperationalError):
        opencode_ro.open_ro(db, retry_delay=0, _connect=dead)
    assert len(calls) == 2


def test_readers_merge_both_dbs(tmp_path):
    a, b = _mkdb(tmp_path / "e.db", "s_e"), _mkdb(tmp_path / "h.db", "s_h")
    t = kpf.opencode_table([a, b], ["qwen-gpu"], 10, 10)
    assert t["qwen-gpu"]["1"]["n"] == 2
    seqs = list(toc.opencode_sessions([a, b], ["qwen-gpu"], 10, 10))
    assert len(seqs) == 2 and all(p == "qwen-gpu" for p, _ in seqs)
    assert opencode_ro.source_for(a) == "unknown"


def test_readers_never_touch_event_table():
    for f in ("opencode_ro.py", "kpf_lb1_cache_by_turn.py", "ufh14_lb9_retokenize_gap.py", "toc_sp_lb1_repeat_calls.py"):
        assert not re.search(r"(?i)\b(from|join)\s+event\b", (ANALYSIS / f).read_text()), f


@pytest.mark.parametrize("tag", ["epyc", "host"])
def test_smoke_real_snapshot(tag):
    db = Path(opencode_ro.OPENCODE_RO_DBS[tag])
    if not db.exists():
        pytest.skip(f"{db} absent")
    conn = opencode_ro.open_ro(db)
    try:
        n = conn.execute("select count(*) from session").fetchone()[0]
    finally:
        conn.close()
    print(f"SMOKE {tag}: {n} sessions")
    assert n > 0
