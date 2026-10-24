#!/usr/bin/env python3
"""SEC-ART-1 backfill planner: derive hf_repo / hf_revision / sha256 for lineup GGUFs WITHOUT hashing them.

For every `roles.<role>.model.path` ending `.gguf` that lacks pins, find the upstream HF repo, read the
repo tree at one resolved commit sha (HF tree API, `?recursive=1&expand=1`), and compare the local file
SIZE with the upstream entry:

  pin                  size equals the upstream LFS size -> sha256 = upstream LFS oid (byte equality is
                       INFERRED from size; SEC-ART-2's full hash at registration is the confirmation)
  needs_local_hash     file not in the repo, or size differs (re-quantised / patched / merged locally)
                       -> listed for a later full hash under the CPU scheduling policy
  needs_repo_mapping   no hf_repo derivable (flat path, not in --repo-map)
  missing_local_file / tree_error                          -> listed, nothing pinned

It never reads GGUF bytes, never edits the registry (the master registry lives in epyc-inference-research
and changes via stack-change); it prints a JSON plan and the YAML fragment to apply.

repo derivation: --repo-map JSON {"<gguf basename or role>": {"hf_repo": "org/repo", "hf_filename": "path/in/repo.gguf",
"hf_revision": "<sha or ref>"}} wins; else a path /<models root>/<org>/<repo>/[sub/]file.gguf gives org/repo.

A relative `model.path` resolves against `runtime_defaults.model_base_path` (as the launcher's pin_index does),
so a role such as `QuantFactory/Qwen2.5-0.5B-GGUF/x.gguf` derives its repo from the path.

Network: OFF unless --allow-network. Requests are strictly sequential (one at a time, cached per repo+ref) and
spaced by --request-interval seconds (default 1.0); only the HF revision and tree METADATA endpoints are read,
never a file. Offline use: --tree-dir DIR holding `<org>__<repo>.json` (a saved tree
API response). Tests: tests/unit/test_backfill_registry_artifact_pins.py
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.registry.artifact_integrity import _abs_model_path, shard_paths  # noqa: E402

MODELS_ROOT = "/mnt/raid0/llm/models"
HF = "https://huggingface.co"
SHA40 = re.compile(r"^[0-9a-f]{40}$")


def throttled(get: Callable[[str], Any], interval: float,
              clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep,
              log: list[str] | None = None) -> Callable[[str], Any]:
    """Wrap ``get`` so consecutive calls are at least ``interval`` s apart; ``log`` collects every URL requested."""
    last: list[float] = []

    def wrapped(url: str) -> Any:
        if last:
            wait = interval - (clock() - last[0])
            if wait > 0:
                sleep(wait)
        if log is not None:
            log.append(url)
        try:
            return get(url)
        finally:
            last[:] = [clock()]
    return wrapped


def http_get_json(url: str) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": "epyc-backfill/1"})
    tok = os.environ.get("HF_TOKEN")
    if tok:
        req.add_header("Authorization", f"Bearer {tok}")
    with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310 - fixed https host
        return json.load(r)


def fetch_repo_state(repo: str, revision: str, get: Callable[[str], Any]) -> tuple[str, dict[str, dict]]:
    """(resolved commit sha, {path: entry}) -- resolves the ref first so the tree read is of one commit."""
    q = urllib.parse.quote
    sha = revision
    if not SHA40.match(revision):
        sha = get(f"{HF}/api/models/{repo}/revision/{q(revision, safe='')}")["sha"]
    entries: dict[str, dict] = {}
    url = f"{HF}/api/models/{repo}/tree/{sha}?recursive=1&expand=1"
    tree = get(url)
    for e in tree:
        if e.get("type") == "file":
            entries[e["path"]] = e
    return sha, entries


def lineup_ggufs(registry: dict) -> list[tuple[str, dict]]:
    """(role, model) for each `roles.<role>.model` with a .gguf path; `_abs_path` is the base-resolved path."""
    out = []
    for role, cfg in sorted((registry.get("roles") or {}).items()):
        m = cfg.get("model") if isinstance(cfg, dict) else None
        if isinstance(m, dict) and str(m.get("path", "")).endswith(".gguf"):
            out.append((role, {**m, "_abs_path": _abs_model_path(str(m["path"]), registry)}))
    return out


def derive_repo(path: str, role: str, repo_map: dict, models_root: str = MODELS_ROOT) -> dict | None:
    base = path.rsplit("/", 1)[-1]
    hit = repo_map.get(path) or repo_map.get(base) or repo_map.get(role)
    if hit:
        return dict(hit)
    if path.startswith(models_root + "/"):
        parts = path[len(models_root) + 1:].split("/")
        if len(parts) >= 3:
            return {"hf_repo": f"{parts[0]}/{parts[1]}", "hf_filename": "/".join(parts[2:])}
    return None


def split_shards(path: str, hf_path: str, entries: dict[str, dict],
                 stat_size: Callable[[str], int | None]) -> tuple[dict[str, str], str | None]:
    """For a split GGUF (`<stem>-00001-of-0000N.gguf`): {basename: oid} of shards 2..N, or (.., reason).

    Every sibling must exist locally AND size-match its upstream LFS entry in the same repo directory;
    one miss sends the whole role to `needs_local_hash` (a half-pinned split set pins nothing).
    """
    local = shard_paths(path)
    if len(local) == 1:
        return {}, None
    upstream = shard_paths(hf_path)
    out: dict[str, str] = {}
    for lp, up in list(zip(local, upstream))[1:]:
        ent = entries.get(up) or {}
        lfs = ent.get("lfs") or {}
        size = stat_size(lp)
        if not lfs.get("oid") or size is None or lfs.get("size") != size:
            return {}, f"shard {os.path.basename(lp)} missing or size differs (upstream {lfs.get('size')})"
        out[os.path.basename(lp)] = lfs["oid"]
    return out, None


def plan(registry: dict, repo_map: dict, get: Callable[[str], Any] | None,
         stat_size: Callable[[str], int | None] = lambda p: os.path.getsize(p) if os.path.exists(p) else None,
         tree_loader: Callable[[str, str], tuple[str, dict]] | None = None) -> list[dict]:
    cache: dict[tuple[str, str], Any] = {}
    rows = []
    for role, m in lineup_ggufs(registry):
        if all(m.get(f) for f in ("hf_repo", "hf_revision", "sha256")):
            continue
        path = m["_abs_path"]
        row: dict[str, Any] = {"role": role, "path": path}
        spec = derive_repo(path, role, repo_map)
        if not spec:
            rows.append({**row, "status": "needs_repo_mapping"})
            continue
        repo, want = spec["hf_repo"], spec.get("hf_revision") or m.get("hf_revision") or "main"
        row["hf_repo"] = repo
        size = stat_size(path)
        if size is None:
            rows.append({**row, "status": "missing_local_file"})
            continue
        key = (repo, want)
        if key not in cache:
            try:
                cache[key] = (tree_loader(repo, want) if tree_loader else fetch_repo_state(repo, want, get))
            except Exception as exc:  # noqa: BLE001 - reported per row
                cache[key] = exc
        state = cache[key]
        if isinstance(state, Exception):
            rows.append({**row, "status": "tree_error", "error": f"{type(state).__name__}: {state}"})
            continue
        sha, entries = state
        fname = spec.get("hf_filename") or path.rsplit("/", 1)[-1]
        e = entries.get(fname) or next((v for k, v in entries.items() if k.rsplit("/", 1)[-1] == fname.rsplit("/", 1)[-1]), None)
        lfs = (e or {}).get("lfs") or {}
        row.update(hf_revision=sha, local_size=size)
        if e and lfs.get("oid") and lfs.get("size") == size:
            shard_sha, bad = split_shards(path, e["path"], entries, stat_size)
            if bad:
                rows.append({**row, "status": "needs_local_hash", "reason": bad})
                continue
            rows.append({**row, "status": "pin", "hf_filename": e["path"], "sha256": lfs["oid"],
                         **({"shard_sha256": shard_sha} if shard_sha else {}),
                         "basis": "size-match to upstream LFS size; full hash confirms in SEC-ART-2"})
        else:
            why = "not in upstream repo" if not e else f"size differs (upstream {lfs.get('size')})"
            rows.append({**row, "status": "needs_local_hash", "reason": why})
    return rows


def yaml_fragment(rows: list[dict]) -> str:
    frag = {r["role"]: {"model": {"hf_repo": r["hf_repo"], "hf_revision": r["hf_revision"], "sha256": r["sha256"],
                                  **({"shard_sha256": r["shard_sha256"]} if r.get("shard_sha256") else {})}}
            for r in rows if r["status"] == "pin"}
    return yaml.safe_dump({"roles": frag}, sort_keys=True) if frag else ""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--registry", default=str(Path(__file__).resolve().parents[2] / "orchestration/model_registry.yaml"))
    ap.add_argument("--repo-map", help="JSON file: basename|role|path -> {hf_repo, hf_filename?, hf_revision?}")
    ap.add_argument("--tree-dir", help="offline: DIR/<org>__<repo>.json tree-API responses")
    ap.add_argument("--allow-network", action="store_true", help="query huggingface.co (off by default)")
    ap.add_argument("--only-role", action="append", help="limit to these roles (repeatable)")
    ap.add_argument("--request-interval", type=float, default=1.0,
                    help="network mode: minimum seconds between consecutive HF API requests (sequential always)")
    ap.add_argument("--out", help="write the JSON plan here (default stdout)")
    a = ap.parse_args(argv)
    reg = yaml.safe_load(Path(a.registry).read_text())
    if a.only_role:
        reg = {**reg, "roles": {k: v for k, v in reg["roles"].items() if k in a.only_role}}
    rmap = json.loads(Path(a.repo_map).read_text()) if a.repo_map else {}
    loader = get = None
    requested: list[str] = []
    if a.tree_dir:
        def loader(repo: str, rev: str) -> tuple[str, dict]:  # noqa: F811
            data = json.loads((Path(a.tree_dir) / (repo.replace("/", "__") + ".json")).read_text())
            sha = data["sha"] if isinstance(data, dict) else rev
            tree = data["tree"] if isinstance(data, dict) else data
            return sha, {e["path"]: e for e in tree if e.get("type") == "file"}
    elif a.allow_network:
        get = throttled(http_get_json, a.request_interval, log=requested)
    else:
        print("refusing: pass --tree-dir (offline) or --allow-network", file=sys.stderr)
        return 2
    rows = plan(reg, rmap, get, tree_loader=loader)
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    doc = json.dumps({"counts": counts, "rows": rows, "yaml_fragment": yaml_fragment(rows),
                      "requests": requested}, indent=2)
    if a.out:
        Path(a.out).write_text(doc + "\n")
    else:
        print(doc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
