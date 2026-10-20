"""scripts/migration/reembed_stores_eg2.py -- synthetic stores only, a fake EG2 endpoint on localhost.

Nothing here touches a live store: every store is built in ``tmp_path`` with the real store classes
(EpisodicStore, SkillBank, SQLiteSessionStore) and the "EG2 server" is a ``http.server`` thread on an
ephemeral 127.0.0.1 port returning deterministic 768-d vectors. FAISS is installed in the
orchestrator venv, so the FAISS tests run; they are gated by ``require_faiss`` (importorskip) for
environments without it, and the sqlite/npy session-store tests do not need it.
"""

from __future__ import annotations

import fcntl
import hashlib
import http.server
import importlib.util
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

TOOL = Path(__file__).resolve().parents[2] / "scripts" / "migration" / "reembed_stores_eg2.py"
_spec = importlib.util.spec_from_file_location("reembed_stores_eg2_under_test", TOOL)
assert _spec and _spec.loader
tool = importlib.util.module_from_spec(_spec)
sys.modules["reembed_stores_eg2_under_test"] = tool
_spec.loader.exec_module(tool)

DOC = "title: none | text: "


@pytest.fixture
def require_faiss():
    return pytest.importorskip("faiss")


# ───────────────────────────────────────────────────────────────────────────── fake EG2 endpoint


class FakeEG2:
    """Deterministic EG2 stand-in. ``mode`` misbehaves once ``bad_from`` texts have been served."""

    def __init__(self) -> None:
        self.mode = "ok"
        self.bad_from = 0
        self.fail_next = 0  # next N embedding POSTs answer 503
        self.status_next = 0  # next embedding POST answers this status once
        self.models = ["embeddinggemma-2-BF16.gguf"]
        self.requests: list[list[str]] = []
        self.served = 0
        self.seen: dict[str, int] = {}
        self.lock = threading.Lock()
        self.url = ""

    @staticmethod
    def base_vec(text: str) -> np.ndarray:
        seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
        v = np.random.default_rng(seed).standard_normal(768).astype(np.float32)
        return v / np.linalg.norm(v)

    def respond(self, inputs: list[str]) -> tuple[int, dict]:
        with self.lock:
            self.requests.append(list(inputs))
            if self.status_next:
                code, self.status_next = self.status_next, 0
                return code, {"error": "scripted"}
            if self.fail_next > 0:
                self.fail_next -= 1
                return 503, {"error": "busy"}
            effective = self.mode if self.served >= self.bad_from else "ok"
            rows = []
            for i, text in enumerate(inputs):
                self.served += 1
                count = self.seen[text] = self.seen.get(text, 0) + 1
                drift = effective == "drift" and count > 1 and "probe" not in text
                v = self.base_vec(text + "#drift" if drift else text)
                if effective == "nan" and i == 0:
                    v = v.copy()
                    v[3] = np.nan
                elif effective == "wrongdim":
                    v = np.concatenate([v, v[:256]])
                elif effective == "zero":
                    v = np.zeros(768, dtype=np.float32)
                rows.append({"object": "embedding", "index": i, "embedding": v.tolist()})
            if effective == "short":
                rows = rows[:-1]
            return 200, {"object": "list", "data": rows}

    def texts(self) -> list[str]:
        return [t for batch in self.requests for t in batch]


def expected_vec(text: str) -> np.ndarray:
    raw = FakeEG2.base_vec(DOC + text)[None]
    return tool._arms().truncate_renormalise(raw, 256)[0]


@pytest.fixture
def eg2():
    fake = FakeEG2()

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *_a):  # silence
            pass

        def _send(self, code: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/v1/models":
                self._send(200, {"data": [{"id": m} for m in fake.models]})
            else:
                self._send(404, {"error": "nope"})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            code, payload = fake.respond(body["input"])
            self._send(code, payload)

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    fake.url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        yield fake
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


# ───────────────────────────────────────────────────────────────────────────── synthetic stores


def make_episodic(d: Path, n: int = 12, dim: int = 1024) -> dict[str, str]:
    from orchestration.repl_memory.episodic_store import EpisodicStore
    from orchestration.repl_memory.memory_record import build_memory_record

    store = EpisodicStore(db_path=d, embedding_dim=dim)
    rng = np.random.default_rng(7)
    texts = {}
    for i in range(n):
        objective = f"episodic task number {i}: solve thing {i}"
        task_type = "code" if i % 2 else "chat"
        ctx = build_memory_record(
            objective=objective, task_type=task_type, priority="normal", source="test"
        ).to_context()
        mid = store.store(
            rng.standard_normal(dim).astype(np.float32),
            "frontdoor:direct",
            "routing",
            ctx,
            outcome="success",
        )
        texts[mid] = f"type:{task_type} | objective:{objective} | priority:normal"
    store.close()
    return texts


def make_skills(d: Path, n: int = 5, dim: int = 1024) -> dict[str, str]:
    from orchestration.repl_memory.skill_bank import Skill, SkillBank

    bank = SkillBank(db_path=d / "skills.db", faiss_path=d, embedding_dim=dim)
    rng = np.random.default_rng(11)
    texts = {}
    for i in range(n):
        skill = Skill(
            id=f"route_{i:03d}",
            title=f"Skill title {i}",
            skill_type="routing",
            principle=f"principle {i}",
            when_to_apply=f"when situation {i}",
            task_types=["code", "chat"],
            teacher_model="test",
        )
        bank.store(skill, embedding=rng.standard_normal(dim).astype(np.float32))
        texts[skill.id] = f"skill:Skill title {i} | when:when situation {i} | task_types:code,chat"
    bank.close()
    return texts


def make_sessions(d: Path, embeddings: int = 0, dim: int = 1024) -> None:
    from src.session import Session, SQLiteSessionStore

    d.mkdir(parents=True, exist_ok=True)
    store = SQLiteSessionStore(
        db_path=d / "sessions.db", embeddings_path=d / "session_embeddings.npy", embedding_dim=dim
    )
    store.create_session(Session(id="sess-1", name="one"))
    rng = np.random.default_rng(3)
    for _ in range(embeddings):
        store.store_embedding("sess-1", rng.standard_normal(dim).astype(np.float32))
    store.close()


def tree_hash(root: Path) -> dict[str, tuple]:
    out: dict[str, tuple] = {}
    for dirpath, dirs, files in os.walk(root, followlinks=False):
        for name in sorted(dirs + files):
            p = Path(dirpath) / name
            rel = p.relative_to(root).as_posix()
            if p.is_symlink():
                out[rel] = ("l", os.readlink(p))
            elif p.is_dir():
                out[rel] = ("d",)
            else:
                out[rel] = ("f", hashlib.sha256(p.read_bytes()).hexdigest())
    return out


@pytest.fixture(scope="module")
def template(tmp_path_factory):
    """Built once with the real store classes; each test copies it."""
    root = tmp_path_factory.mktemp("template")
    info = {
        "ep_texts": make_episodic(root / "episodic"),
        "sk_texts": make_skills(root / "skills"),
    }
    make_sessions(root / "sessions")
    (root / "episodic" / "notes.txt").write_text("operator notes\n")
    (root / "episodic" / "backups").mkdir()
    (root / "episodic" / "backups" / "old.bak").write_bytes(b"\x00old backup\x00" * 100)
    info["root"] = root
    return info


class Layout:
    def __init__(self, root: Path, info: dict) -> None:
        self.live = root / "live"
        shutil.copytree(info["root"], self.live)
        self.ep, self.sk, self.se = (self.live / n for n in ("episodic", "skills", "sessions"))
        self.out = root / "out"
        self.ep_texts, self.sk_texts = info["ep_texts"], info["sk_texts"]

    @property
    def stores(self) -> list[str]:
        return [
            "--episodic-dir",
            str(self.ep),
            "--skill-dir",
            str(self.sk),
            "--session-dir",
            str(self.se),
        ]

    def snapshot(self) -> dict[str, tuple]:
        return tree_hash(self.live)


@pytest.fixture
def layout(tmp_path, template, require_faiss):
    return Layout(tmp_path, template)


def cli(capsys, *argv: str) -> tuple[int, str, str]:
    capsys.readouterr()
    rc = tool.main(list(argv))
    cap = capsys.readouterr()
    return rc, cap.out, cap.err


def apply_args(lay: Layout, fake: FakeEG2, run_id: str = "t1", *extra: str) -> list[str]:
    return [
        "run", "--apply", *lay.stores, "--out-dir", str(lay.out), "--embed-url", fake.url,
        "--assume-stopped", "--run-id", run_id, "--retry-backoff", "0", "--batch-size", "4",
        *extra,
    ]  # fmt: skip


# ───────────────────────────────────────────────────────────────────────────── plan


def test_plan_reports_counts_dims_bytes_and_text_source(layout, capsys):
    rc, out, _ = cli(capsys, "plan", *layout.stores, "--json")
    assert rc == 0
    plan = json.loads(out)
    assert plan["blocked"] is False
    by_dir = {Path(u["path"]).name: u for u in plan["units"]}
    assert set(by_dir) == {"episodic", "skills", "sessions"}
    ep = by_dir["episodic"]["components"][0]
    assert (ep["kind"], ep["status"], ep["vectors"], ep["dim"], ep["rows"]) == (
        "episodic",
        "READY",
        12,
        1024,
        12,
    )
    assert (
        ep["id_map_len"] == 12
        and ep["bytes"]["embeddings.faiss"] > 0
        and ep["bytes"]["episodic.db"] > 0
    )
    assert "embedding_text" in ep["text_source"] and "memories.context" in ep["text_source"]
    sk = by_dir["skills"]["components"][0]
    assert (sk["kind"], sk["status"], sk["vectors"], sk["dim"]) == ("skills", "READY", 5, 1024)
    assert "skill_embedding_text" in sk["text_source"]
    se = by_dir["sessions"]["components"][0]
    assert (se["kind"], se["status"], se["dim"]) == ("sessions", "EMPTY", 1024)
    assert "NONE RETAINED" in se["text_source"]
    assert plan["target"]["dim"] == 256 and plan["target"]["dtype"] == "bf16"
    assert plan["target"]["doc_prompt"] == DOC
    rc, out, _ = cli(capsys, "plan", *layout.stores)
    assert rc == 0 and "READY" in out and "text source" in out


def test_plan_target_identity_is_the_arm_spec(layout, capsys):
    arms = tool._arms()
    spec = {s.arm_id: s for s in arms.arm_specs("n/a", tool.DEFAULT_RUNTIME, "n/a")}[
        "eg2-gguf-bf16-256"
    ]
    ident = spec.index_identity()
    rc, out, _ = cli(capsys, "plan", *layout.stores, "--json")
    target = json.loads(out)["target"]
    assert rc == 0
    for key in ("model", "revision", "dim", "dtype", "prompt_hash", "digest"):
        assert target[key] == ident[key]
    assert target["prompt_hash"] == spec.prompt_hash()


def test_plan_writes_nothing(layout, capsys):
    before, sig = layout.snapshot(), tool.tree_signature(layout.live)
    rc, _, _ = cli(capsys, "plan", *layout.stores)
    assert rc == 0
    assert layout.snapshot() == before
    assert tool.tree_signature(layout.live) == sig  # names, sizes AND mtimes


def test_plan_stores_dir_discovery(layout, capsys):
    rc, out, _ = cli(capsys, "plan", "--stores-dir", str(layout.live), "--json")
    assert rc == 0
    assert {Path(u["path"]).name for u in json.loads(out)["units"]} == {
        "episodic",
        "skills",
        "sessions",
    }


def test_plan_without_any_store_is_refused(capsys):
    rc, _, err = cli(capsys, "plan")
    assert rc == 2 and "no store given" in err


@pytest.mark.parametrize("damage", ["delete_rows", "blank_context", "delete_db"])
def test_vectors_only_episodic_store_is_blocked(layout, capsys, eg2, damage):
    if damage == "delete_db":
        (layout.ep / "episodic.db").unlink()
    else:
        con = sqlite3.connect(layout.ep / "episodic.db")
        con.execute(
            "DELETE FROM memories"
            if damage == "delete_rows"
            else "UPDATE memories SET context='{}'"
        )
        con.commit()
        con.close()
    before = layout.snapshot()
    rc, out, _ = cli(capsys, "plan", *layout.stores, "--json")
    assert rc == 3
    unit = next(u for u in json.loads(out)["units"] if Path(u["path"]).name == "episodic")
    comp = unit["components"][0]
    assert comp["status"] == "BLOCKED"
    if damage == "delete_db":
        assert "vectors only" in comp["blocked"][0] and "not retained" in comp["blocked"][0]
    else:
        assert "no retained source text" in comp["blocked"][0]
        assert "not invented" in comp["blocked"][0]
        reason = "no_db_row" if damage == "delete_rows" else "no_task_text"
        assert comp["detail"]["unresolved"] == {reason: 12}
    # an --apply on a BLOCKED store does nothing at all
    rc, _, _ = cli(capsys, *apply_args(layout, eg2))
    assert rc == 3 and eg2.requests == []
    assert layout.snapshot() == before and not layout.out.exists()


def test_session_store_with_vectors_is_blocked_empty_is_not(tmp_path, capsys):
    make_sessions(tmp_path / "empty")
    make_sessions(tmp_path / "full", embeddings=2)
    rc, out, _ = cli(capsys, "plan", "--session-dir", str(tmp_path / "empty"), "--json")
    assert rc == 0 and json.loads(out)["units"][0]["components"][0]["status"] == "EMPTY"
    rc, out, _ = cli(capsys, "plan", "--session-dir", str(tmp_path / "full"), "--json")
    assert rc == 3
    comp = json.loads(out)["units"][0]["components"][0]
    assert comp["status"] == "BLOCKED" and "NO retained source text" in comp["blocked"][0]
    assert comp["detail"]["metadata_rows"] == {"session": 2}


def test_skills_without_a_vector_index_is_nothing_to_do(layout, capsys, eg2):
    (layout.sk / "skill_embeddings.faiss").unlink()
    (layout.sk / "skill_id_map.npy").unlink()
    rc, out, _ = cli(capsys, "plan", "--skill-dir", str(layout.sk), "--json")
    unit = json.loads(out)["units"][0]
    assert rc == 0 and unit["status"] == "NOTHING_TO_DO"
    assert unit["components"][0]["status"] == "NO_VECTORS"
    before = layout.snapshot()
    rc, _, _ = cli(capsys, "run", "--apply", "--skill-dir", str(layout.sk), "--out-dir", str(layout.out),
                   "--embed-url", eg2.url, "--assume-stopped")  # fmt: skip
    assert rc == 0 and layout.snapshot() == before and not layout.out.exists()


def test_co_located_foreign_vector_index_blocks_the_unit(layout, capsys):
    (layout.ep / "strategy_embeddings.faiss").write_bytes(b"x")
    (layout.ep / "embeddings.faiss.bak").write_bytes(b"x")  # a backup suffix is not a store
    rc, out, _ = cli(capsys, "plan", "--episodic-dir", str(layout.ep), "--json")
    assert rc == 3
    reasons = json.loads(out)["units"][0]["reasons"]
    assert any("strategy_embeddings.faiss" in r for r in reasons)
    assert not any("embeddings.faiss.bak" in r for r in reasons)


def test_legacy_numpy_episodic_layout_is_blocked(tmp_path, capsys):
    d = tmp_path / "legacy"
    d.mkdir()
    con = sqlite3.connect(d / "episodic.db")
    con.execute("CREATE TABLE memories (id TEXT PRIMARY KEY, context TEXT, embedding_idx INTEGER)")
    con.commit()
    con.close()
    np.save(d / "embeddings.npy", np.zeros((4, 1024), dtype=np.float32))
    rc, out, _ = cli(capsys, "plan", "--episodic-dir", str(d), "--json")
    assert rc == 3 and "legacy NumpyEmbeddingStore" in json.dumps(json.loads(out))


def test_desynced_index_and_id_map_is_blocked(layout, capsys):
    arr = np.load(layout.ep / "id_map.npy", allow_pickle=True)
    np.save(layout.ep / "id_map.npy", arr[:-2], allow_pickle=True)
    rc, out, _ = cli(capsys, "plan", "--episodic-dir", str(layout.ep), "--json")
    assert rc == 3 and "desync" in json.dumps(json.loads(out))


# ───────────────────────────────────────────────────────────────────────────── dry run


def test_dry_run_writes_nothing_and_embeds_nothing(layout, capsys, eg2):
    before, sig = layout.snapshot(), tool.tree_signature(layout.live)
    parent_before = sorted(os.listdir(layout.live))
    rc, out, _ = cli(capsys, "run", *layout.stores, "--out-dir", str(layout.out), "--embed-url", eg2.url,
                     "--run-id", "dry1", "--json")  # fmt: skip
    assert rc == 0
    payload = json.loads(out)
    assert payload["mode"] == "dry-run" and payload["would_refuse"] == []
    assert eg2.requests == []  # no request of any kind reaches the endpoint
    assert layout.snapshot() == before
    assert tool.tree_signature(layout.live) == sig
    assert sorted(os.listdir(layout.live)) == parent_before
    assert not layout.out.exists()
    ops = payload["operations"]
    swaps = [o for o in ops if o["action"] == "swap"]
    assert {Path(o["rename_new_to"]).name for o in swaps} == {"episodic", "skills", "sessions"}
    assert all(o["rename_old_to"].endswith(".pre-eg2-dry1") and o["old_kept_intact"] for o in swaps)
    written = [
        o["path"] for o in ops if o["action"] in ("write", "write_new", "copy", "create_dir")
    ]
    assert any(p.endswith("episodic.migrating-dry1") for p in written)
    assert any(p.endswith("embeddings.faiss") for p in written)
    assert any(p.endswith("embedding_model.json") for p in written)
    assert any(p.endswith("manifest-dry1.json") for p in written)
    assert any(p.endswith("rollback-dry1.sh") for p in written)


def test_dry_run_is_the_default_and_reports_what_apply_would_refuse(layout, capsys):
    (layout.live / "episodic.pre-eg2-old").mkdir()
    rc, out, _ = cli(capsys, "run", *layout.stores)
    assert (
        rc == 2 and "DRY RUN" in out and "an --apply would REFUSE" in out and "pre-eg2-old" in out
    )


# ───────────────────────────────────────────────────────────────────────────── apply + swap


def load_ok(lay: Layout, dim: int = 256):
    from orchestration.repl_memory.episodic_store import EpisodicStore
    from orchestration.repl_memory.skill_bank import SkillBank
    from src.session import SQLiteSessionStore

    ep = EpisodicStore(db_path=lay.ep, embedding_dim=dim)
    sk = SkillBank(db_path=lay.sk / "skills.db", faiss_path=lay.sk, embedding_dim=dim)
    se = SQLiteSessionStore(
        db_path=lay.se / "sessions.db",
        embeddings_path=lay.se / "session_embeddings.npy",
        embedding_dim=dim,
    )
    return ep, sk, se


def test_apply_swaps_and_the_real_stores_load_at_256(layout, capsys, eg2, require_faiss):
    faiss = require_faiss
    original = layout.snapshot()
    rc, out, err = cli(capsys, *apply_args(layout, eg2))
    assert rc == 0, err
    # the old directories are kept intact, byte for byte
    pre = {n: layout.live / f"{n}.pre-eg2-t1" for n in ("episodic", "skills", "sessions")}
    assert all(p.is_dir() for p in pre.values())
    for name, path in pre.items():
        assert tree_hash(path) == tree_hash_of(original, name)
    assert not list(layout.live.glob("*.migrating-*"))

    # nothing but document-prefixed texts reached the endpoint
    texts = eg2.texts()
    assert texts and all(t.startswith(DOC) for t in texts)
    stored_texts = set(layout.ep_texts.values()) | set(layout.sk_texts.values())
    assert {t[len(DOC) :] for t in texts if "probe" not in t} == stored_texts

    # the real store classes load the migrated directories at 256-d and retrieve correctly
    ep, sk, se = load_ok(layout)
    try:
        assert ep.count() == 12
        mid, text = sorted(layout.ep_texts.items())[3]
        hit = ep.retrieve_by_similarity(expected_vec(text), k=1)[0]
        assert hit.id == mid and hit.similarity_score == pytest.approx(1.0, abs=1e-4)
        sid, stext = sorted(layout.sk_texts.items())[2]
        skill, score = sk.search_by_embedding(expected_vec(stext), k=1, min_similarity=0.5)[0]
        assert skill.id == sid and score == pytest.approx(1.0, abs=1e-4)
        assert se.embeddings_available
        # ids and positions are untouched: vector i is still the embedding of id_map[i]'s text
        index = faiss.read_index(str(layout.ep / "embeddings.faiss"))
        id_map = np.load(layout.ep / "id_map.npy", allow_pickle=True).tolist()
        assert index.d == 256 and index.ntotal == 12 == len(id_map)
        for i, memory_id in enumerate(id_map):
            assert np.allclose(
                index.reconstruct(i), expected_vec(layout.ep_texts[memory_id]), atol=1e-5
            )
        assert (
            sqlite3.connect(layout.ep / "episodic.db")
            .execute("SELECT COUNT(*) FROM memories")
            .fetchone()[0]
            == 12
        )
    finally:
        ep.close()
        sk.close()
        se.close()
    # a consumer still configured for BGE-1024 is refused by the migrated session store
    from src.session import SQLiteSessionStore

    refused = SQLiteSessionStore(
        db_path=layout.se / "sessions.db",
        embeddings_path=layout.se / "session_embeddings.npy",
        embedding_dim=1024,
    )
    assert not refused.embeddings_available
    refused.close()

    # the dimension marker
    arms = tool._arms()
    spec = {s.arm_id: s for s in arms.arm_specs("n/a", tool.DEFAULT_RUNTIME, "n/a")}[
        "eg2-gguf-bf16-256"
    ]
    for d in (layout.ep, layout.sk, layout.se):
        marker = json.loads((d / "embedding_model.json").read_text())
        assert (marker["model"], marker["revision"], marker["dim"], marker["dtype"]) == (
            spec.model_repo, spec.revision, 256, "bf16"
        )  # fmt: skip
        assert marker["prompt_hash"] == spec.prompt_hash()
        assert tool.verify_store_marker(d, expected_dim=256)["dim"] == 256
        with pytest.raises(tool.StoreMarkerError):
            tool.verify_store_marker(d, expected_dim=1024)
    assert cli(capsys, "check-marker", str(layout.ep))[0] == 0
    assert cli(capsys, "check-marker", str(layout.ep), "--expect-dim", "1024")[0] == 2

    # unmanaged files ride along as hard links, so no 22 GB duplicate and no second secret
    for rel in ("notes.txt", "backups/old.bak"):
        assert os.stat(layout.ep / rel).st_ino == os.stat(pre["episodic"] / rel).st_ino
        assert (layout.ep / rel).read_bytes() == (pre["episodic"] / rel).read_bytes()

    # manifest: sealed, hashes of every old and new file, identity, rollback script
    manifest = json.loads((layout.out / "manifest-t1.json").read_text())
    assert tool.verify_seal(manifest)
    assert manifest["target"]["dim"] == 256
    ep_unit = next(u for u in manifest["units"] if u["live"].endswith("episodic"))
    for name in ("episodic.db", "embeddings.faiss", "id_map.npy"):
        assert len(ep_unit["old_files"][name]["sha256"]) == 64
        assert len(ep_unit["new_files"][name]["sha256"]) == 64
    assert (
        ep_unit["old_files"]["embeddings.faiss"]["sha256"]
        != ep_unit["new_files"]["embeddings.faiss"]["sha256"]
    )
    assert (
        ep_unit["old_files"]["episodic.db"]["sha256"]
        == ep_unit["new_files"]["episodic.db"]["sha256"]
    )
    assert (
        ep_unit["validation"]["components"]["episodic"]["semantic_self_check"]["min_cosine"] > 0.999
    )
    assert os.access(layout.out / "rollback-t1.sh", os.X_OK)
    assert (layout.out / "swap-receipt-t1.json").exists()
    assert (layout.out / "run-t1.log.jsonl").read_text().strip()
    # the tool wrote nothing outside the stores' siblings and --out-dir
    assert sorted(os.listdir(layout.live)) == sorted(
        ["episodic", "skills", "sessions"]
        + [f"{n}.pre-eg2-t1" for n in ("episodic", "skills", "sessions")]
    )


def tree_hash_of(snapshot: dict[str, tuple], name: str) -> dict[str, tuple]:
    pref = name + "/"
    return {k[len(pref) :]: v for k, v in snapshot.items() if k.startswith(pref)}


def test_apply_is_idempotent_once_migrated(layout, capsys, eg2):
    assert cli(capsys, *apply_args(layout, eg2))[0] == 0
    after = layout.snapshot()
    n_requests = len(eg2.requests)
    rc, out, _ = cli(capsys, *apply_args(layout, eg2, "t2"))
    assert rc == 0 and len(eg2.requests) == n_requests
    assert layout.snapshot() == after and not list(layout.live.glob("*.pre-eg2-t2"))


def test_co_located_stores_are_one_unit_with_one_swap(
    tmp_path, template, capsys, eg2, require_faiss
):
    one = tmp_path / "sessions"
    one.mkdir()
    for name in ("episodic", "skills", "sessions"):
        for f in (template["root"] / name).iterdir():
            if f.is_file():
                shutil.copy2(f, one / f.name)
    (one / "session_pickle_hmac.key").write_text("secret")
    original = tree_hash(one)
    stores = ["--episodic-dir", str(one), "--skill-dir", str(one), "--session-dir", str(one)]
    rc, out, _ = cli(capsys, "plan", *stores, "--json")
    plan = json.loads(out)
    assert rc == 0 and len(plan["units"]) == 1
    assert [c["kind"] for c in plan["units"][0]["components"]] == ["episodic", "skills", "sessions"]
    rc, _, err = cli(
        capsys, "run", "--apply", *stores, "--out-dir", str(tmp_path / "out"), "--embed-url", eg2.url,
        "--assume-stopped", "--run-id", "co", "--retry-backoff", "0",
    )  # fmt: skip
    assert rc == 0, err
    pre = tmp_path / "sessions.pre-eg2-co"
    assert (one / "embedding_model.json").exists()
    assert tree_hash(pre) == original
    assert (
        os.stat(one / "session_pickle_hmac.key").st_ino
        == os.stat(pre / "session_pickle_hmac.key").st_ino
    )
    assert sorted(p.name for p in tmp_path.glob("sessions*")) == ["sessions", "sessions.pre-eg2-co"]


# ───────────────────────────────────────────────────────────────────────────── validation failure


@pytest.mark.parametrize("mode", ["nan", "wrongdim", "zero", "drift", "short"])
def test_validation_failure_leaves_the_live_stores_untouched(layout, capsys, eg2, mode):
    before = layout.snapshot()
    eg2.mode, eg2.bad_from = mode, 6  # the probe (2 texts) and the first batch succeed
    rc, _, err = cli(capsys, *apply_args(layout, eg2, "bad", "--retries", "1"))
    assert rc == 4, err
    assert layout.snapshot() == before
    assert not list(layout.live.glob("*.pre-eg2-*"))
    assert not list(layout.live.glob("*.migrating-*"))  # the tool removes only its own copy
    assert not (layout.out / "manifest-bad.json").exists()


def test_semantic_self_check_catches_misattributed_vectors(layout, capsys, eg2):
    before = layout.snapshot()
    eg2.mode, eg2.bad_from = "drift", 0
    rc, _, err = cli(capsys, *apply_args(layout, eg2, "drift"))
    # drift starts at the probe's exemption boundary: the second sight of a text differs
    assert rc == 4 and "semantic self-check failed" in err
    assert layout.snapshot() == before


def test_keep_failed_keeps_the_migrating_copy_for_forensics(layout, capsys, eg2):
    eg2.mode, eg2.bad_from = "nan", 6
    rc, _, _ = cli(capsys, *apply_args(layout, eg2, "kf", "--retries", "0", "--keep-failed"))
    assert rc == 4 and list(layout.live.glob("*.migrating-kf"))
    assert not list(layout.live.glob("*.pre-eg2-*"))


def test_a_writer_that_woke_up_mid_copy_aborts_before_the_swap(layout, capsys, eg2, monkeypatch):
    before = layout.snapshot()
    real = tool.validate_unit

    def wake_up(*args, **kwargs):
        result = real(*args, **kwargs)
        (layout.ep / "late_write.tmp").write_text(
            "a writer is alive"
        )  # after validation, before swap
        return result

    monkeypatch.setattr(tool, "validate_unit", wake_up)
    rc, _, err = cli(capsys, *apply_args(layout, eg2, "wake"))
    assert rc == 4 and "changed while it was being migrated" in err
    (layout.ep / "late_write.tmp").unlink()
    assert layout.snapshot() == before
    assert not list(layout.live.glob("*.pre-eg2-*")) and not list(layout.live.glob("*.migrating-*"))


# ───────────────────────────────────────────────────────────────────────────── swap primitives


def _two_dirs(tmp_path: Path):
    live, new, pre = tmp_path / "live", tmp_path / "live.migrating-x", tmp_path / "live.pre-eg2-x"
    live.mkdir()
    new.mkdir()
    (live / "f").write_text("old")
    (new / "f").write_text("new")
    return live, new, pre


def test_swap_prefers_atomic_exchange_when_the_kernel_supports_it(tmp_path):
    live, new, pre = _two_dirs(tmp_path)
    probe_a, probe_b = tmp_path / "pa", tmp_path / "pb"
    probe_a.mkdir()
    probe_b.mkdir()
    if not tool._try_exchange(probe_a, probe_b):
        pytest.skip("renameat2(RENAME_EXCHANGE) is not supported on this filesystem")
    assert tool.atomic_swap(live, new, pre) == "exchange"
    assert (
        (live / "f").read_text() == "new" and (pre / "f").read_text() == "old" and not new.exists()
    )


def test_swap_falls_back_to_two_renames(tmp_path, monkeypatch):
    live, new, pre = _two_dirs(tmp_path)
    monkeypatch.setattr(tool, "_try_exchange", lambda a, b: False)
    assert tool.atomic_swap(live, new, pre) == "rename"
    assert (live / "f").read_text() == "new" and (pre / "f").read_text() == "old"
    tool.revert_swap(live, new, pre)
    assert (
        (live / "f").read_text() == "old" and (new / "f").read_text() == "new" and not pre.exists()
    )


def test_swap_failure_between_the_renames_restores_the_live_name(tmp_path, monkeypatch):
    live, new, pre = _two_dirs(tmp_path)
    monkeypatch.setattr(tool, "_try_exchange", lambda a, b: False)
    real, calls = tool._rename, []

    def flaky(src, dst):
        calls.append((src, dst))
        if len(calls) == 2:
            raise OSError("disk says no")
        real(src, dst)

    monkeypatch.setattr(tool, "_rename", flaky)
    with pytest.raises(OSError, match="disk says no"):
        tool.atomic_swap(live, new, pre)
    assert (
        (live / "f").read_text() == "old" and (new / "f").read_text() == "new" and not pre.exists()
    )


def test_swap_never_overwrites_an_existing_pre_dir(tmp_path):
    live, new, pre = _two_dirs(tmp_path)
    pre.mkdir()
    (pre / "keep").write_text("previous old store")
    with pytest.raises(FileExistsError):
        tool.atomic_swap(live, new, pre)
    assert (pre / "keep").read_text() == "previous old store" and (live / "f").read_text() == "old"


def test_second_swap_failing_reverts_the_first(layout, capsys, eg2, monkeypatch):
    before = layout.snapshot()
    monkeypatch.setattr(tool, "_try_exchange", lambda a, b: False)
    real, state = tool._rename, {"n": 0}

    def flaky(src, dst):
        state["n"] += 1
        # swaps are 2 renames each; fail the third unit's second rename (new -> live)
        if state["n"] == 6:
            raise OSError("injected rename failure")
        real(src, dst)

    monkeypatch.setattr(tool, "_rename", flaky)
    rc, _, err = cli(capsys, *apply_args(layout, eg2, "sw"))
    assert rc == 5 and "reverted" in err
    assert layout.snapshot() == before
    # the manifest and rollback script of a swap that did not happen are set aside, not left live
    assert not (layout.out / "manifest-sw.json").exists()
    assert (layout.out / "manifest-sw.json.swap-failed").exists()
    assert not (layout.out / "rollback-sw.sh").exists()
    assert not list(layout.live.glob("*.pre-eg2-*")) and not list(layout.live.glob("*.migrating-*"))


# ───────────────────────────────────────────────────────────────────────────── rollback


def test_rollback_restores_byte_identical_and_keeps_the_eg2_dir(layout, capsys, eg2):
    original = layout.snapshot()
    assert cli(capsys, *apply_args(layout, eg2))[0] == 0
    manifest = layout.out / "manifest-t1.json"
    rc, _, err = cli(capsys, "rollback", "--manifest", str(manifest))
    assert rc == 6 and "--assume-stopped" in err  # no attestation, nothing moves
    assert (layout.live / "episodic.pre-eg2-t1").is_dir()
    rc, out, err = cli(capsys, "rollback", "--manifest", str(manifest), "--assume-stopped")
    assert rc == 0, err
    restored = layout.snapshot()
    for name in ("episodic", "skills", "sessions"):
        assert {k: v for k, v in restored.items() if k.split("/")[0] == name} == {
            k: v for k, v in original.items() if k.split("/")[0] == name
        }
    for name in ("episodic", "skills", "sessions"):
        assert not (layout.live / f"{name}.pre-eg2-t1").exists()
        kept = layout.live / f"{name}.eg2-rolledback-t1"
        assert (kept / "embedding_model.json").exists()  # the EG2 store is kept, never deleted
    assert (layout.out / "rollback-receipt-t1.json").exists()
    assert not (layout.ep / "embedding_model.json").exists()
    assert (
        cli(capsys, "plan", *layout.stores)[0] == 0
    )  # the restored stores plan as migratable again


def test_rollback_verifies_old_hashes_before_restoring(layout, capsys, eg2):
    assert cli(capsys, *apply_args(layout, eg2))[0] == 0
    with open(layout.live / "episodic.pre-eg2-t1" / "id_map.npy", "ab") as fh:
        fh.write(b"tamper")
    state = layout.snapshot()
    rc, _, err = cli(
        capsys, "rollback", "--manifest", str(layout.out / "manifest-t1.json"), "--assume-stopped"
    )
    assert rc == 6 and "does not match the manifest hash" in err
    assert layout.snapshot() == state  # nothing was renamed


def test_rollback_refuses_diverged_eg2_store_unless_allowed(layout, capsys, eg2):
    original = layout.snapshot()
    assert cli(capsys, *apply_args(layout, eg2))[0] == 0
    with open(layout.ep / "embeddings.faiss", "ab") as fh:
        fh.write(b"post-swap write")
    manifest = str(layout.out / "manifest-t1.json")
    rc, _, err = cli(capsys, "rollback", "--manifest", manifest, "--assume-stopped")
    assert rc == 6 and "changed since the swap" in err
    rc, _, err = cli(
        capsys, "rollback", "--manifest", manifest, "--assume-stopped", "--allow-diverged"
    )
    assert rc == 0, err
    assert tree_hash_of(layout.snapshot(), "episodic") == tree_hash_of(original, "episodic")
    assert (layout.live / "episodic.eg2-rolledback-t1" / "embeddings.faiss").exists()


def test_rollback_refuses_a_tampered_manifest(layout, capsys, eg2):
    assert cli(capsys, *apply_args(layout, eg2))[0] == 0
    path = layout.out / "manifest-t1.json"
    data = json.loads(path.read_text())
    data["units"][0]["pre"] = str(layout.live)  # try to aim the rollback somewhere else
    path.write_text(json.dumps(data))
    rc, _, err = cli(capsys, "rollback", "--manifest", str(path), "--assume-stopped")
    assert rc == 6 and "seal mismatch" in err


def test_recorded_rollback_script_restores_with_plain_shell(layout, capsys, eg2):
    original = layout.snapshot()
    assert cli(capsys, *apply_args(layout, eg2))[0] == 0
    script = layout.out / "rollback-t1.sh"
    assert subprocess.run(["bash", "-n", str(script)]).returncode == 0
    no_flag = subprocess.run(["bash", str(script)], capture_output=True, text=True)
    assert no_flag.returncode == 2 and (layout.live / "episodic.pre-eg2-t1").is_dir()
    with open(layout.live / "skills.pre-eg2-t1" / "skills.db", "ab") as fh:
        fh.write(b"x")
    bad = subprocess.run(["bash", str(script), "--assume-stopped"], capture_output=True, text=True)
    assert bad.returncode != 0 and (layout.live / "episodic.pre-eg2-t1").is_dir()
    assert (layout.ep / "embedding_model.json").exists()  # still the EG2 store: nothing moved
    with open(layout.live / "skills.pre-eg2-t1" / "skills.db", "rb+") as fh:
        fh.truncate(fh.seek(0, 2) - 1)  # undo the tamper
    ok = subprocess.run(["bash", str(script), "--assume-stopped"], capture_output=True, text=True)
    assert ok.returncode == 0, ok.stderr
    for name in ("episodic", "skills", "sessions"):
        assert tree_hash_of(layout.snapshot(), name) == tree_hash_of(original, name)
        assert (layout.live / f"{name}.eg2-rolledback-t1").is_dir()


# ───────────────────────────────────────────────────────────────────────────── refusals


def refused(layout: Layout, capsys, eg2: FakeEG2, *extra: str, run_id: str = "r1") -> str:
    before = layout.snapshot()
    rc, _, err = cli(capsys, *apply_args(layout, eg2, run_id, *extra))
    assert rc == 2, err
    assert layout.snapshot() == before and not layout.out.exists()
    assert eg2.requests == []
    return err


def test_refuses_without_the_stopped_attestation(layout, capsys, eg2):
    args = [a for a in apply_args(layout, eg2) if a != "--assume-stopped"]
    before = layout.snapshot()
    rc, _, err = cli(capsys, *args)
    assert rc == 2 and "--assume-stopped" in err
    assert layout.snapshot() == before and not layout.out.exists()


def test_refuses_to_overwrite_an_existing_pre_eg2_dir(layout, capsys, eg2):
    keep = layout.live / "episodic.pre-eg2-20260101T000000Z"
    keep.mkdir()
    (keep / "precious").write_text("an earlier migration's old store")
    err = refused(layout, capsys, eg2)
    assert "pre-eg2-20260101T000000Z" in err and "never deletes" in err
    assert (keep / "precious").read_text() == "an earlier migration's old store"


def test_refuses_with_a_stale_migrating_copy(layout, capsys, eg2):
    (layout.live / "skills.migrating-old").mkdir()
    assert "migrating-old" in refused(layout, capsys, eg2)


def test_refuses_when_a_sqlite_wal_is_open(layout, capsys, eg2):
    con = sqlite3.connect(layout.se / "sessions.db")
    try:
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("CREATE TABLE IF NOT EXISTS zz_probe (x)")
        con.execute("INSERT INTO zz_probe VALUES (1)")
        con.commit()
        assert (layout.se / "sessions.db-wal").exists()
        assert "sqlite sidecar" in refused(layout, capsys, eg2)
    finally:
        con.close()


def test_refuses_when_a_lock_file_is_held(layout, capsys, eg2):
    fd = os.open(layout.ep / ".episodic_faiss.lock", os.O_RDWR | os.O_CREAT)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        assert "flock held on .episodic_faiss.lock" in refused(layout, capsys, eg2)
    finally:
        os.close(fd)


def test_refuses_when_another_process_has_a_store_file_open(layout, tmp_path, capsys, eg2):
    fake_proc = tmp_path / "proc"
    (fake_proc / "4242" / "fd").mkdir(parents=True)
    os.symlink(layout.ep / "embeddings.faiss", fake_proc / "4242" / "fd" / "3")
    (fake_proc / "4343").mkdir()
    (fake_proc / "4343" / "maps").write_text(
        f"7f00-7f01 r--s 0 00:00 0 {layout.se}/session_embeddings.npy\n"
    )
    err = refused(layout, capsys, eg2, "--proc-root", str(fake_proc))
    assert "pid 4242 holds fd" in err and "pid 4343 holds mmap" in err


def test_refuses_an_out_dir_inside_a_store(layout, capsys, eg2):
    args = apply_args(layout, eg2)
    args[args.index("--out-dir") + 1] = str(layout.ep / "out")
    before = layout.snapshot()
    rc, _, err = cli(capsys, *args)
    assert rc == 2 and "overlaps the store directory" in err and layout.snapshot() == before


def test_refuses_without_embed_url_or_out_dir(layout, capsys):
    rc, _, err = cli(capsys, "run", "--apply", *layout.stores, "--assume-stopped")
    assert rc == 2 and "--embed-url and --out-dir" in err


def test_refuses_a_server_that_is_not_embeddinggemma(layout, capsys, eg2):
    eg2.models = ["bge-large-en-v1.5-f16.gguf"]
    err = refused(layout, capsys, eg2)
    assert "none looks like EmbeddingGemma 2" in err or "EmbeddingGemma" in err


def test_refuses_a_server_returning_the_wrong_dimension(layout, capsys, eg2):
    eg2.mode = "wrongdim"
    before = layout.snapshot()
    rc, _, err = cli(capsys, *apply_args(layout, eg2))
    assert rc == 2 and "expected (*, 768)" in err
    assert layout.snapshot() == before


def test_nested_store_directories_are_refused(tmp_path):
    from argparse import Namespace

    make_sessions(tmp_path / "outer")
    make_sessions(tmp_path / "outer" / "inner")
    plan = tool.build_plan(
        Namespace(
            episodic_dir=None, skill_dir=None, session_dir=None, stores_dir=str(tmp_path / "outer")
        ),
        tool.target_identity(tool.target_spec()),
    )
    assert any("nested" in r for r in plan["plan_refusals"])


# ───────────────────────────────────────────────────────────────────────────── marker API


def test_marker_contract_for_consumers(tmp_path):
    d = tmp_path / "store"
    d.mkdir()
    ident = tool.target_identity(tool.target_spec())
    # no marker == legacy BGE-1024
    assert tool.verify_store_marker(d, expected_dim=1024)["legacy"] is True
    with pytest.raises(tool.StoreMarkerError, match="legacy 1024-d"):
        tool.verify_store_marker(d, expected_dim=256)
    (d / tool.MARKER_NAME).write_text(json.dumps(tool.build_marker(ident, "r", {})))
    assert tool.verify_store_marker(d, expected_dim=256, expected_identity=ident)["dim"] == 256
    with pytest.raises(tool.StoreMarkerError, match="mixed store"):
        tool.verify_store_marker(d, expected_dim=1024)
    other = dict(ident, revision="deadbeef")
    with pytest.raises(tool.StoreMarkerError, match="revision"):
        tool.verify_store_marker(d, expected_dim=256, expected_identity=other)
    (d / tool.MARKER_NAME).write_text("{not json")
    with pytest.raises(tool.StoreMarkerError, match="unreadable"):
        tool.verify_store_marker(d, expected_dim=256)


def test_plan_blocks_foreign_and_mixed_markers(layout, capsys):
    ident = tool.target_identity(tool.target_spec())
    marker = tool.build_marker(ident, "r", {})
    (layout.ep / tool.MARKER_NAME).write_text(json.dumps(dict(marker, model="someone/else")))
    rc, out, _ = cli(capsys, "plan", "--episodic-dir", str(layout.ep), "--json")
    assert rc == 3 and "different embedding identity" in json.dumps(json.loads(out))
    (layout.ep / tool.MARKER_NAME).write_text(json.dumps(marker))  # marked 256 but 1024-d vectors
    rc, out, _ = cli(capsys, "plan", "--episodic-dir", str(layout.ep), "--json")
    assert rc == 3 and "mixed store" in json.dumps(json.loads(out))
    # 256-d vectors without a marker: unknown provenance
    make_episodic(layout.live / "e256", n=3, dim=256)
    rc, out, _ = cli(capsys, "plan", "--episodic-dir", str(layout.live / "e256"), "--json")
    assert rc == 3 and "without embedding_model.json" in json.dumps(json.loads(out))


# ───────────────────────────────────────────────────────────────────────────── client


def _client(fake: FakeEG2, **kw) -> object:
    spec = tool.target_spec()
    return tool.EmbedClient(fake.url, spec, sleep=lambda _s: None, **kw)


def test_client_batches_prefixes_truncates_and_renormalises(eg2):
    client = _client(eg2, batch_size=4)
    texts = [f"text number {i}" for i in range(10)]
    vecs = client.embed_documents(texts)
    assert vecs.shape == (10, 256) and vecs.dtype == np.float32
    assert [len(b) for b in eg2.requests] == [4, 4, 2] and client.requests == 3
    assert all(t.startswith(DOC) for t in eg2.texts())
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-5)
    for t, v in zip(texts, vecs):
        assert np.allclose(v, expected_vec(t), atol=1e-6)
    # truncation is the leading 256 columns, re-normalised -- not a re-projection
    raw = FakeEG2.base_vec(DOC + texts[0])
    assert np.allclose(vecs[0], raw[:256] / np.linalg.norm(raw[:256]), atol=1e-6)


def test_client_retries_transient_failures_with_backoff(eg2):
    sleeps: list[float] = []
    client = tool.EmbedClient(
        eg2.url, tool.target_spec(), retries=3, backoff_s=0.5, sleep=sleeps.append
    )
    eg2.fail_next = 2
    assert client.embed_documents(["a"]).shape == (1, 256)
    assert client.requests == 3 and client.retried == 2 and sleeps == [0.5, 1.0]


def test_client_gives_up_after_the_retry_budget(eg2):
    client = _client(eg2, retries=2)
    eg2.fail_next = 99
    with pytest.raises(tool.EmbedError, match="giving up after 3 attempts"):
        client.embed_documents(["a"])
    assert client.requests == 3


def test_client_does_not_retry_a_client_error_or_a_wrong_dimension(eg2):
    client = _client(eg2, retries=3)
    eg2.status_next = 400
    with pytest.raises(tool.EmbedError, match="HTTP 400"):
        client.embed_documents(["a"])
    assert client.requests == 1
    eg2.mode = "wrongdim"
    client = _client(eg2, retries=3)
    with pytest.raises(tool.WrongDimensionError):
        client.embed_documents(["a"])
    assert client.requests == 1


@pytest.mark.parametrize(
    ("mode", "msg"), [("nan", "non-finite"), ("zero", "zero-norm"), ("short", "rows for")]
)
def test_client_refuses_degenerate_vectors(eg2, mode, msg):
    eg2.mode = mode
    with pytest.raises(tool.EmbedError, match=msg):
        _client(eg2, retries=1).embed_documents(["a", "b"])


def test_client_only_speaks_http():
    with pytest.raises(tool.Refused, match="http"):
        tool.EmbedClient("file:///etc/passwd", tool.target_spec())


def test_client_probe_requires_a_deterministic_eg2_server(eg2):
    info = _client(eg2).probe()
    assert info["models"] == ["embeddinggemma-2-BF16.gguf"] and info["probe_cosine_repeat"] > 0.9999
    eg2.models = ["something-else"]
    with pytest.raises(tool.Refused, match="EmbeddingGemma 2"):
        _client(eg2).probe()
    assert _client(eg2).probe(check_model=False)["model_check"].startswith("skipped")


# ───────────────────────────────────────────────────────────────────────────── misc


def test_declared_store_kind_missing_from_its_directory_is_blocked(layout, capsys):
    rc, out, _ = cli(capsys, "plan", "--skill-dir", str(layout.ep), "--json")
    assert rc == 3
    assert "declared as the skills store but no skills files were found" in json.dumps(
        json.loads(out)
    )


def test_command_line_entry_point_and_help(layout):
    for argv in (["--help"], ["plan", "--help"], ["run", "--help"], ["rollback", "--help"]):
        done = subprocess.run([sys.executable, str(TOOL), *argv], capture_output=True, text=True)
        assert done.returncode == 0, done.stderr
    done = subprocess.run(
        [sys.executable, str(TOOL), "plan", "--stores-dir", str(layout.live), "--json"],
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0 and len(json.loads(done.stdout)["units"]) == 3


def test_module_docstring_carries_the_runbook_and_the_retraining_flags():
    doc = tool.__doc__
    assert "@@" not in doc
    order = [
        "STOP THE WRITERS",
        " 1. ``plan``",
        " 2. ``run`` (dry)",
        " 3. ``run --apply",
        "RESTART CONSUMERS WITH THE EG2 POOL",
        "KEEP ``<dir>.pre-eg2-<run_id>/``",
    ]
    positions = [doc.index(marker) for marker in order]
    assert positions == sorted(positions)
    assert "RETRAINING / RE-CALIBRATION FLAGS" in doc and "routing_classifier_weights.npz" in doc
    assert "CONSUMERS THAT MUST READ" in doc
