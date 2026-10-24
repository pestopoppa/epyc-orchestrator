"""SEC-ART-1 backfill planner: mocked HF tree API, no network, no GGUF reads."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import yaml

_P = Path(__file__).resolve().parents[2] / "scripts/validate/backfill_registry_artifact_pins.py"
_S = importlib.util.spec_from_file_location("backfill_pins", _P)
bf = importlib.util.module_from_spec(_S)
_S.loader.exec_module(bf)

COMMIT = "c" * 40
OID = "d" * 64
M = "/mnt/raid0/llm/models"
TREE = [
    {"type": "file", "path": "m-Q8_0.gguf", "size": 100, "lfs": {"oid": OID, "size": 100}},
    {"type": "file", "path": "README.md", "size": 5},
    {"type": "directory", "path": "sub"},
]
REG = {"roles": {
    "a": {"model": {"path": f"{M}/org/repo/m-Q8_0.gguf"}},
    "b": {"model": {"path": f"{M}/org/repo/patched.gguf"}},
    "c": {"model": {"path": f"{M}/flat.gguf"}},
    "d": {"model": {"path": f"{M}/org/repo/m-Q8_0.gguf", "hf_repo": "o/r", "hf_revision": COMMIT, "sha256": OID}},
    "e": {"model": {"path": f"{M}/org/gone/x.gguf"}},
}}


def fake_get(calls):
    def get(url):
        calls.append(url)
        if "/revision/" in url:
            return {"sha": COMMIT}
        assert url.endswith(f"/tree/{COMMIT}?recursive=1&expand=1")
        return TREE
    return get


def sizes(p):
    return {"m-Q8_0.gguf": 100, "patched.gguf": 77, "flat.gguf": 9}.get(p.rsplit("/", 1)[-1])


def test_plan_classifies_each_role_and_resolves_ref_once():
    calls: list[str] = []
    rows = {r["role"]: r for r in bf.plan(REG, {}, fake_get(calls), stat_size=sizes)}
    assert "d" not in rows                                  # already fully pinned
    assert rows["a"]["status"] == "pin" and rows["a"]["sha256"] == OID and rows["a"]["hf_revision"] == COMMIT
    assert rows["b"]["status"] == "needs_local_hash" and "not in upstream" in rows["b"]["reason"]
    assert rows["c"]["status"] == "needs_repo_mapping"
    assert rows["e"]["status"] == "missing_local_file"
    assert sum("/revision/" in u for u in calls) == 1        # org/repo cached across roles a,b
    assert all(r.get("sha256") is None for k, r in rows.items() if k != "a")


def test_size_mismatch_needs_local_hash():
    rows = bf.plan({"roles": {"a": {"model": {"path": f"{M}/org/repo/m-Q8_0.gguf"}}}}, {}, fake_get([]),
                   stat_size=lambda p: 99)
    assert rows[0]["status"] == "needs_local_hash" and "size differs" in rows[0]["reason"]


def test_repo_map_resolves_flat_paths_and_tree_error_is_reported():
    rmap = {"flat.gguf": {"hf_repo": "org/repo", "hf_filename": "m-Q8_0.gguf"}}
    reg = {"roles": {"c": {"model": {"path": f"{M}/flat.gguf"}}}}
    rows = bf.plan(reg, rmap, fake_get([]), stat_size=lambda p: 100)
    assert rows[0]["status"] == "pin"

    def boom(url):
        raise OSError("403")
    rows = bf.plan(reg, rmap, boom, stat_size=lambda p: 100)
    assert rows[0]["status"] == "tree_error" and "403" in rows[0]["error"]


def test_cli_offline_tree_dir_and_network_refusal(tmp_path, capsys):
    reg = tmp_path / "r.yaml"
    reg.write_text(yaml.safe_dump({"roles": {"a": {"model": {"path": str(tmp_path / "org/repo/m-Q8_0.gguf")}}}}))
    assert bf.main(["--registry", str(reg)]) == 2           # no source => refuse, never silently networks
    (tmp_path / "org/repo").mkdir(parents=True)
    (tmp_path / "org/repo/m-Q8_0.gguf").write_bytes(b"x" * 100)
    td = tmp_path / "trees"
    td.mkdir()
    (td / "org__repo.json").write_text(json.dumps({"sha": COMMIT, "tree": TREE}))
    rm = tmp_path / "map.json"
    rm.write_text(json.dumps({"m-Q8_0.gguf": {"hf_repo": "org/repo"}}))
    assert bf.main(["--registry", str(reg), "--tree-dir", str(td), "--repo-map", str(rm)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["counts"] == {"pin": 1} and OID in out["yaml_fragment"]


def _split_tree(sizes=(100, 200, 300)):
    return [{"type": "file", "path": f"Q/m-0000{i}-of-00003.gguf", "size": s, "lfs": {"oid": str(i) * 63 + "a", "size": s}}
            for i, s in enumerate(sizes, 1)]


def test_split_gguf_pins_every_shard_by_basename():
    tree = _split_tree()

    def get(url):
        return {"sha": COMMIT} if "/revision/" in url else tree

    reg = {"roles": {"g": {"model": {"path": f"{M}/org/repo/Q/m-00001-of-00003.gguf"}}}}
    sizes = {"m-00001-of-00003.gguf": 100, "m-00002-of-00003.gguf": 200, "m-00003-of-00003.gguf": 300}
    rows = bf.plan(reg, {}, get, stat_size=lambda p: sizes.get(p.rsplit("/", 1)[-1]))
    assert rows[0]["status"] == "pin" and rows[0]["sha256"] == "1" * 63 + "a"
    assert rows[0]["shard_sha256"] == {"m-00002-of-00003.gguf": "2" * 63 + "a", "m-00003-of-00003.gguf": "3" * 63 + "a"}
    assert "shard_sha256" in bf.yaml_fragment(rows)


def test_split_gguf_with_one_bad_shard_pins_nothing():
    tree = _split_tree()

    def get(url):
        return {"sha": COMMIT} if "/revision/" in url else tree

    reg = {"roles": {"g": {"model": {"path": f"{M}/org/repo/Q/m-00001-of-00003.gguf"}}}}
    sizes = {"m-00001-of-00003.gguf": 100, "m-00002-of-00003.gguf": 200, "m-00003-of-00003.gguf": 301}
    rows = bf.plan(reg, {}, get, stat_size=lambda p: sizes.get(p.rsplit("/", 1)[-1]))
    assert rows[0]["status"] == "needs_local_hash" and "00003" in rows[0]["reason"]


def test_relative_path_resolves_against_model_base_path():
    """SEC-ART-1.1: `QuantFactory/<repo>/<file>.gguf` (relative, as draft_qwen25 is declared) derives its repo
    from the base-resolved path, exactly as the launcher's pin_index resolves it."""
    reg = {"runtime_defaults": {"model_base_path": M},
           "roles": {"a": {"model": {"path": "org/repo/m-Q8_0.gguf"}}}}
    seen: list[str] = []

    def stat(p):
        seen.append(p)
        return sizes(p)
    rows = bf.plan(reg, {}, fake_get([]), stat_size=stat)
    assert rows[0]["status"] == "pin" and rows[0]["hf_repo"] == "org/repo" and rows[0]["path"] == f"{M}/org/repo/m-Q8_0.gguf"
    assert seen == [f"{M}/org/repo/m-Q8_0.gguf"]


def test_throttled_get_is_sequential_spaced_and_logged():
    now = [0.0]
    slept: list[float] = []
    log: list[str] = []

    def sleep(s):
        slept.append(s)
        now[0] += s
    get = bf.throttled(lambda u: u.upper(), 1.5, clock=lambda: now[0], sleep=sleep, log=log)
    assert get("a") == "A"                       # first call never waits
    now[0] += 0.5
    assert get("b") == "B"
    assert slept == [1.0] and log == ["a", "b"]
