#!/usr/bin/env python3
"""Champion serving-config artifact -> master-registry diff -> scratch dry-run preflight.

PROMOTION-REGISTRY-SERVING-1 (operator 2026-11-24). The kernel-promotion package carries the
champion serving-config artifact (``docs/reference/kernel-manifests/serving-configs/
<champion>.serving-config.json`` in epyc-root, schema ``epyc.champion.serving_config.v1``). This
module is the registry half of that promotion:

``diff``     artifact + master registry -> a registry diff (``epyc.registry_serving_diff.v1``)
``apply``    diff + master registry -> a patched COPY (never an in-place write)
``dryrun``   apply the diff to a COPY of the master inside a scratch dir, rebuild a scratch
             copy of this repo against it, and run ``stack_change_pipeline.py check`` there
``section``  the three package sections (serving_config, registry_diff, registry_dry_run) that
             ``kernel-promotion/scripts/package.sh`` folds into the one ratification package

Nothing here writes the live registry, the shared research clone or the live derived layer: every
write lands under ``--scratch`` / ``--out``, and the dry-run proves that by comparing the live
lean registry's digest before and after.

Serving-config shape consumed (schema ``epyc.champion.serving_config.v1``; the authoritative
parser is ``scripts/champion/serving_config/validate_serving_config.py`` in epyc-root, which this
module loads via ``--validator`` rather than re-parsing)::

    {"schema": ..., "champion": "C22",
     "servers": [{"server_mode": "architect_critic", ...,
         "registry_serving_shape": {...},
         "optimal":   {"status": "measured"|"pending", "config": {n_ctx, np, batch, ubatch, kv_k, kv_v,
                       kv_unified, ctx_checkpoints, cache_ram_mib, drafter, sampling},
                       "argv_delta": [{flag, value?, reason}], "provenance": {...}},
         "alternates": [{"name", "status", "argv_delta": [...], "provenance": {...}}],
         "tradeoffs": [...]}]}

Only MEASURED entries are written. ``optimal.config`` maps onto the registry knobs (``np`` ->
``slots``, ``cache_ram_mib`` -> ``cache_ram``, ``ctx_checkpoints``, and ``serving_shape.{n_ctx,batch,
ubatch,kv_unified,kv_quant}``); a value equal to the registry's is not an op, so a measured optimum
that IS the served recipe yields an empty diff. ``drafter`` and ``sampling`` are not registry
serving_shape and are not written. A measured alternate becomes ``serving_shape.alternates.<name>``
(``argv_delta`` as ``{flag: value}``, against the registry serving_shape). A PENDING optimal skips its
whole server and a PENDING alternate is skipped (an existing master alternate of that name is left
alone); both are listed in the diff's ``skipped_pending`` so the package shows what the signature
does not cover. Measured alternates the artifact no longer lists are removed.

The diff records the value it replaces (``old``/``had_old``). ``apply`` refuses a master that has
drifted from those olds: the operator signs a diff against a known master, not "whatever is there".
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.registry.serving_alternates import (  # noqa: E402
    ALTERNATES_KEY,
    ServingAlternatesError,
    parse_alternates,
    validate_registry_alternates,
)

CONFIG_SCHEMA = "epyc.champion.serving_config.v1"
DIFF_SCHEMA = "epyc.registry_serving_diff.v1"
#: Env seam for tests: when set, replaces the update+check pair run in the scratch repo.
PREFLIGHT_CMD_ENV = "SERVING_CONFIG_PREFLIGHT_CMD"
#: Directories left out of the scratch repo copy (bulky, never read by the pipeline check).
_SCRATCH_EXCLUDES = ("artifacts", "benchmarks", "docs", "handoffs", "progress", "orchestration/reports")
#: Pipeline steps that read the LIVE HOST (listeners on declared ports, running-process env), not
#: the registry under test. A failure there is host state -- another session's server on a stack
#: port -- and says nothing about a registry diff, so it is recorded (`host_state_failures`) but
#: does not fail the dry-run. Every other failed step does.
HOST_STATE_STEPS = frozenset({"runtime_attestation", "declared_env_attestation"})
#: Rows of the printed report that fold other rows rather than being a step of their own.
_DERIVED_ROWS = frozenset({"summary", "acceptance", "promotion_gate"})


class ServingConfigError(ValueError):
    """The serving-config artifact or the registry diff is unusable."""


# --------------------------------------------------------------------------- artifact


def _load_validator(path: Path | None):
    """Import the root repo's ``validate_serving_config.py`` (the one parser of this artifact)."""
    if path is None:
        return None
    import importlib.util

    spec = importlib.util.spec_from_file_location("_serving_config_validator", str(path))
    if spec is None or spec.loader is None:
        raise ServingConfigError(f"cannot load serving-config validator {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_serving_config(
    path: Path, *, validator: Path | None = None, registry: dict[str, Any] | None = None,
    strict_files: bool = True,
) -> dict[str, Any]:
    """Load the champion serving-config artifact; validate it with the root validator when given.

    ``registry`` (the master doc) turns on the validator's ``registry_serving_shape`` / ``registry_runtime``
    drift check. ``strict_files`` (default True, defect 5 of the 2026-11-24 wiring audit) makes a RESULT.json
    that is absent from this host, or whose sha256 differs, an error: the signed diff must not rest on
    provenance nobody could open.
    """
    path = Path(path)
    if not path.is_file():
        raise ServingConfigError(f"serving-config artifact not found: {path}")
    try:
        cfg = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ServingConfigError(f"{path} is not valid JSON: {exc}") from exc
    mod = _load_validator(validator)
    if mod is not None:
        rep = mod.validate(cfg, strict_files, registry)
        if rep.errors:
            raise ServingConfigError(
                f"{path} fails validate_serving_config.py:\n  " + "\n  ".join(rep.errors)
            )
    return check_shape(cfg, source=str(path))


def check_shape(cfg: Any, *, source: str = "serving-config") -> dict[str, Any]:
    """The minimum this module relies on (full validation is the root validator's job)."""
    if not isinstance(cfg, dict) or cfg.get("schema") != CONFIG_SCHEMA:
        raise ServingConfigError(f"{source}: schema must be {CONFIG_SCHEMA!r}")
    if not cfg.get("champion") or not isinstance(cfg["champion"], str):
        raise ServingConfigError(f"{source}: `champion` (string) is required")
    servers = cfg.get("servers")
    if not isinstance(servers, list) or not servers:
        raise ServingConfigError(f"{source}: `servers` must be a non-empty list")
    for i, srv in enumerate(servers):
        if not isinstance(srv, dict) or not isinstance(srv.get("server_mode"), str):
            raise ServingConfigError(f"{source}: servers[{i}].server_mode required")
        for where, entry in [("optimal", srv.get("optimal"))] + [
            (f"alternates[{j}]", a) for j, a in enumerate(srv.get("alternates") or [])
        ]:
            if not isinstance(entry, dict) or entry.get("status") not in ("measured", "pending"):
                raise ServingConfigError(
                    f"{source}: servers[{i}]({srv['server_mode']}).{where}.status must be measured|pending"
                )
    return cfg


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- diff / apply

_ABSENT = object()

#: optimal.config key -> registry path under server_mode.<role> (drafter and sampling are not
#: serving_shape: the drafter is topology-selected and sampling is a per-request default).
_CONFIG_TO_REGISTRY: dict[str, tuple[str, ...]] = {
    "np": ("slots",),
    "ctx_checkpoints": ("ctx_checkpoints",),
    "cache_ram_mib": ("cache_ram",),
    "n_ctx": ("serving_shape", "n_ctx"),
    "batch": ("serving_shape", "batch"),
    "ubatch": ("serving_shape", "ubatch"),
    "kv_unified": ("serving_shape", "kv_unified"),
    "kv_k": ("serving_shape", "kv_quant", "k"),
    "kv_v": ("serving_shape", "kv_quant", "v"),
}


def _get(doc: Any, path: list[str]) -> Any:
    cur = doc
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return _ABSENT
        cur = cur[key]
    return cur


def argv_delta_mapping(delta: list[dict[str, Any]]) -> dict[str, Any]:
    """The artifact's ``[{flag, value?, reason}]`` as the registry slot's ``{flag: value}``."""
    return {d["flag"]: d.get("value", True) for d in delta}


def build_registry_diff(
    cfg: dict[str, Any], master: dict[str, Any], *, config_sha256: str | None = None
) -> dict[str, Any]:
    """The registry edits that make ``master`` serve ``cfg``. Pure; reads nothing else.

    Only MEASURED entries are written. A pending optimal skips its whole server; a pending alternate
    is skipped (and an existing master alternate of the same name is left alone). Skips are listed in
    ``skipped_pending`` so the package shows what the signature does NOT cover.
    """
    cfg = check_shape(cfg)
    server_mode = master.get("server_mode")
    if not isinstance(server_mode, dict):
        raise ServingConfigError("master registry has no server_mode section")
    ops: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    def put(path: list[str], new: Any, provenance: Any) -> None:
        old = _get(master, path)
        if old is not _ABSENT and old == new:
            return
        ops.append({
            "op": "set",
            "path": path,
            "had_old": old is not _ABSENT,
            "old": None if old is _ABSENT else old,
            "new": new,
            "provenance": provenance,
        })

    for srv in cfg["servers"]:
        role = srv["server_mode"]
        optimal = srv["optimal"]
        if optimal["status"] != "measured":
            skipped.append({"server_mode": role, "entry": "optimal", "row_id": optimal.get("row_id"),
                            "reason": optimal.get("pending_reason")})
            for alt in srv.get("alternates") or []:
                skipped.append({"server_mode": role, "entry": alt.get("name"), "row_id": alt.get("row_id"),
                                "reason": "server optimal is pending"})
            continue
        if role not in server_mode or not isinstance(server_mode[role], dict):
            raise ServingConfigError(
                f"serving config names role {role!r}, which is not a server_mode record in the master registry"
            )
        base = ["server_mode", role]
        config = optimal["config"]
        for key, rel in _CONFIG_TO_REGISTRY.items():
            if key in config:
                put(base + list(rel), config[key], optimal["provenance"])

        declared: dict[str, dict[str, Any]] = {}
        pending_names: set[str] = set()
        for alt in srv.get("alternates") or []:
            name = alt["name"]
            if alt["status"] != "measured":
                pending_names.add(name)
                skipped.append({"server_mode": role, "entry": name, "row_id": alt.get("row_id"),
                                "reason": alt.get("pending_reason")})
                continue
            declared[name] = {"argv_delta": argv_delta_mapping(alt["argv_delta"]),
                              "provenance": {k: alt["provenance"][k] for k in
                                             ("row_id", "result_json", "result_sha256", "store")
                                             if k in alt["provenance"]}}
        # validate the slot's own schema before it can reach a signed diff
        parse_alternates(declared, source=f"servers({role}).alternates")
        current = _get(master, base + ["serving_shape", ALTERNATES_KEY])
        current = {} if current is _ABSENT or current is None else current
        for name, alt in declared.items():
            put(base + ["serving_shape", ALTERNATES_KEY, name], alt, alt["provenance"])
        for name in sorted(set(current) - set(declared) - pending_names):
            ops.append({
                "op": "remove",
                "path": base + ["serving_shape", ALTERNATES_KEY, name],
                "had_old": True,
                "old": current[name],
                "new": None,
                "provenance": "absent from the champion serving-config artifact",
            })

    return {
        "schema": DIFF_SCHEMA,
        "champion": cfg["champion"],
        "target": "master registry (research orchestration/model_registry.yaml)",
        "serving_config_sha256": config_sha256,
        "ops": ops,
        "skipped_pending": skipped,
        "summary": summarize_diff(ops, skipped),
    }


def summarize_diff(ops: list[dict[str, Any]], skipped: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    roles: dict[str, dict[str, int]] = {}
    for op in ops:
        role = op["path"][1]
        kind = "alternates" if ALTERNATES_KEY in op["path"] else "knobs"
        roles.setdefault(role, {"knobs": 0, "alternates": 0})[kind] += 1
    return {"ops": len(ops), "roles": roles, "skipped_pending": len(skipped or [])}


def render_diff(diff: dict[str, Any]) -> list[str]:
    """Human-readable lines, one per op: the thing the operator actually reads and signs."""
    lines = []
    for op in diff["ops"]:
        path = ".".join(op["path"])
        old = json.dumps(
            op["old"].get("argv_delta") if ALTERNATES_KEY in op["path"] and isinstance(op["old"], dict) else op["old"]
        ) if op["had_old"] else "<absent>"
        new = "<removed>" if op["op"] == "remove" else json.dumps(
            op["new"].get("argv_delta") if ALTERNATES_KEY in op["path"] and isinstance(op["new"], dict) else op["new"]
        )
        lines.append(f"{path}: {old} -> {new}")
    for sk in diff.get("skipped_pending") or []:
        lines.append(f"SKIPPED (pending, never written) {sk['server_mode']}.{sk['entry']} row={sk.get('row_id')}")
    return lines


def apply_registry_diff(master: dict[str, Any], diff: dict[str, Any]) -> dict[str, Any]:
    """Return a patched deep COPY of ``master``. Refuses a master that drifted from the diff's olds."""
    if diff.get("schema") != DIFF_SCHEMA:
        raise ServingConfigError(f"diff schema must be {DIFF_SCHEMA!r}")
    out = copy.deepcopy(master)
    for op in diff["ops"]:
        path = op["path"]
        current = _get(out, path)
        if op["had_old"]:
            if current is _ABSENT or current != op["old"]:
                raise ServingConfigError(
                    f"master drifted from the signed diff at {'.'.join(path)}: diff expects "
                    f"{json.dumps(op['old'])}, master has "
                    f"{'<absent>' if current is _ABSENT else json.dumps(current)}"
                )
        elif current is not _ABSENT:
            raise ServingConfigError(
                f"master drifted from the signed diff at {'.'.join(path)}: diff expects it absent"
            )
        parent = _get(out, path[:-1])
        if not isinstance(parent, dict):
            # Create intermediate mappings only below an existing server_mode record.
            parent = out
            for key in path[:-1]:
                parent = parent.setdefault(key, {})
        if op["op"] == "remove":
            parent.pop(path[-1], None)
        else:
            parent[path[-1]] = copy.deepcopy(op["new"])
    validate_registry_alternates(out)
    return out


# --------------------------------------------------------------------------- dry run


def _git_archive_repo(repo_root: Path, dest: Path) -> str:
    """Copy the committed tree of ``repo_root`` (minus bulky dirs) into ``dest``; return the commit."""
    sha = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    specs = [".", *[f":(exclude){p}" for p in _SCRATCH_EXCLUDES]]
    archive = subprocess.Popen(
        ["git", "-C", str(repo_root), "archive", "--format=tar", sha, "--", *specs],
        stdout=subprocess.PIPE,
    )
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(["tar", "-x", "-C", str(dest)], stdin=archive.stdout, check=True)
    archive.stdout.close()
    if archive.wait() != 0:
        raise ServingConfigError(f"git archive of {repo_root} failed")
    return sha


def _digest_or_none(path: Path) -> str | None:
    try:
        return _sha256(path)
    except OSError:
        return None


def failed_steps(report_text: str) -> list[str]:
    """Names of the steps a printed ``stack_change_pipeline`` report marks ``failed``."""
    out = []
    for line in report_text.splitlines():
        name, sep, status = line.partition(": ")
        if sep and not line.startswith(" ") and status.strip() == "failed" and name not in _DERIVED_ROWS:
            out.append(name)
    return out


def _tail(text: str, n: int = 12) -> list[str]:
    return [ln for ln in text.splitlines() if ln.strip()][-n:]


def dry_run(
    diff: dict[str, Any],
    master_path: Path,
    *,
    repo_root: Path = REPO_ROOT,
    scratch: Path | None = None,
    python: str | None = None,
    timeout_s: int = 1800,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> dict[str, Any]:
    """Apply ``diff`` to a COPY of the master in ``scratch`` and run the stack-change preflight on it.

    Preflight = ``stack_change_pipeline.py update`` (regenerates the scratch copy's lean/descriptor/
    priors layer from the patched master -- `check` compares those to the master, so a diff that
    changes the master can only pass against a regenerated layer) then ``check``, both inside a
    scratch copy of the repo so every default output path is the scratch's. ``SERVING_CONFIG_PREFLIGHT_CMD``
    replaces that pair (test seam). Returns a record with ``status`` pass|fail.
    """
    repo_root = Path(repo_root)
    live_lean = repo_root / "orchestration" / "model_registry.yaml"
    live_before = _digest_or_none(live_lean)
    own_tmp = scratch is None
    scratch = Path(tempfile.mkdtemp(prefix="serving-config-dryrun-")) if own_tmp else Path(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {"scratch": str(scratch), "steps": []}
    try:
        master = yaml.safe_load(Path(master_path).read_text())
        patched = apply_registry_diff(master, diff)
        scratch_master = scratch / "research" / "orchestration" / "model_registry.yaml"
        scratch_master.parent.mkdir(parents=True, exist_ok=True)
        scratch_master.write_text(yaml.safe_dump(patched, sort_keys=False, width=1000))
        record["patched_master"] = str(scratch_master)
        record["patched_master_sha256"] = _sha256(scratch_master)

        scratch_repo = scratch / "repo"
        record["repo_commit"] = _git_archive_repo(repo_root, scratch_repo)
        py = python or sys.executable
        env = {**os.environ, "PYTHONPATH": str(scratch_repo), "PYTHONDONTWRITEBYTECODE": "1"}
        env.pop("ORCHESTRATOR_REGISTRY_NO_COMPILE", None)
        override = os.environ.get(PREFLIGHT_CMD_ENV)
        pipeline = [
            py, str(scratch_repo / "scripts/registry/stack_change_pipeline.py"),
        ]
        flags = ["--research-registry", str(scratch_master), "--allow-stale-research-registry"]
        cmds = (
            [shlex.split(override)]
            if override
            else [[*pipeline, "update", *flags], [*pipeline, "check", *flags]]
        )
        passed = True
        host_state: list[str] = []
        for cmd in cmds:
            res = runner(cmd, cwd=scratch_repo, env=env, capture_output=True, text=True, timeout=timeout_s)
            out_text = (res.stdout or "") + "\n" + (res.stderr or "")
            failed = failed_steps(out_text) if not override else []
            step = {
                "cmd": " ".join(shlex.quote(c) for c in cmd),
                "exit": res.returncode,
                "failed_steps": failed,
                "tail": _tail(out_text),
            }
            record["steps"].append(step)
            if res.returncode == 0:
                continue
            blocking = [n for n in failed if n not in HOST_STATE_STEPS]
            if failed and not blocking:
                host_state.extend(n for n in failed if n not in host_state)
                continue
            passed = False
            break
        record["host_state_failures"] = host_state
        record["host_state_carve_out"] = {
            "excluded_steps": sorted(HOST_STATE_STEPS),
            "failures_excluded": host_state,
            "rationale": (
                "these pipeline steps read the live host (unmanaged listeners on declared stack "
                "ports, running-process env), not the registry under test; a failure there is "
                "another session's server and says nothing about the diff. Every other failed "
                "step fails the dry-run."
            ),
        }
        live_after = _digest_or_none(live_lean)
        record["live_registry_untouched"] = live_before == live_after
        record["status"] = "pass" if passed and record["live_registry_untouched"] else "fail"
        record["preflight_exit"] = record["steps"][-1]["exit"] if record["steps"] else None
        record["preflight_note"] = (
            "host-state steps failed (live listeners on declared ports) and are excluded: "
            f"{host_state}" if host_state else None
        )
    except (ServingConfigError, ServingAlternatesError, subprocess.SubprocessError, OSError) as exc:
        record["status"] = "fail"
        record["error"] = f"{type(exc).__name__}: {exc}"
    record["diff_summary"] = diff.get("summary")
    return record


# --------------------------------------------------------------------------- CLI


def _load_master(path: str | None) -> tuple[Path, Callable[[], None]]:
    if path:
        return Path(path), lambda: None
    from src.registry.research_origin import materialize_research_registry

    mat = materialize_research_registry()
    return mat.path, mat.cleanup


def build_package_sections(
    config_path: Path, master_path: Path, *, run_dry: bool = True, repo_root: Path = REPO_ROOT,
    scratch: Path | None = None, python: str | None = None, validator: Path | None = None,
) -> dict[str, Any]:
    master = yaml.safe_load(Path(master_path).read_text())
    cfg = load_serving_config(config_path, validator=validator, registry=master)
    sha = _sha256(config_path)
    diff = build_registry_diff(cfg, master, config_sha256=sha)
    sections: dict[str, Any] = {
        "serving_config": {"path": str(config_path), "sha256": sha, "schema": CONFIG_SCHEMA,
                           "champion": cfg["champion"]},
        "registry_diff": diff,
    }
    if run_dry:
        sections["registry_dry_run"] = dry_run(
            diff, master_path, repo_root=repo_root, scratch=scratch, python=python
        )
    else:
        sections["registry_dry_run"] = {"status": "skipped", "reason": "--no-dry-run"}
    return sections


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("diff", "dryrun", "section"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True, type=Path)
        p.add_argument("--master", help="master registry file (default: research origin/main)")
        p.add_argument("--out", type=Path)
        p.add_argument("--validator", type=Path, default=None,
                       help="epyc-root scripts/champion/serving_config/validate_serving_config.py")
        p.add_argument("--no-validate", action="store_true",
                       help="explicitly skip the root validator (default: refuse when --validator is not given)")
        if name != "diff":
            p.add_argument("--scratch", type=Path)
            p.add_argument("--repo-root", type=Path, default=REPO_ROOT)
            p.add_argument("--python", default=None)
        if name == "section":
            p.add_argument("--no-dry-run", action="store_true")
    p = sub.add_parser("apply")
    p.add_argument("--diff", required=True, type=Path)
    p.add_argument("--master", help="master registry file (default: research origin/main)")
    p.add_argument("--out", required=True, type=Path, help="patched COPY (comments are not preserved)")
    args = ap.parse_args(argv)

    if args.cmd in ("diff", "dryrun", "section"):
        # N6 (final-gate audit): no validator used to mean NO validation, silently. Say it or refuse.
        if args.validator is not None and args.no_validate:
            print("FAIL: --validator and --no-validate are mutually exclusive", file=sys.stderr)
            return 2
        if args.validator is None and not args.no_validate:
            print(f"FAIL: `{args.cmd}` needs --validator <validate_serving_config.py>; "
                  "pass --no-validate to skip validation explicitly", file=sys.stderr)
            return 2
        if args.no_validate:
            print("WARNING: --no-validate: the serving-config artifact is NOT validated", file=sys.stderr)

    master_path, cleanup = _load_master(args.master)
    try:
        if args.cmd == "apply":
            diff = json.loads(args.diff.read_text())
            patched = apply_registry_diff(yaml.safe_load(master_path.read_text()), diff)
            args.out.write_text(yaml.safe_dump(patched, sort_keys=False, width=1000))
            print(f"patched copy written to {args.out}")
            return 0
        if args.cmd in ("diff", "dryrun"):
            master_doc = yaml.safe_load(master_path.read_text())
            cfg = load_serving_config(args.config, validator=args.validator, registry=master_doc)
            diff = build_registry_diff(cfg, master_doc, config_sha256=_sha256(args.config))
        if args.cmd == "diff":
            payload: Any = diff
            print("\n".join(render_diff(diff)), file=sys.stderr)
        else:
            if args.cmd == "dryrun":
                payload = dry_run(diff, master_path, repo_root=args.repo_root,
                                  scratch=args.scratch, python=args.python)
            else:
                payload = build_package_sections(
                    args.config, master_path, run_dry=not args.no_dry_run,
                    repo_root=args.repo_root, scratch=args.scratch, python=args.python,
                    validator=args.validator,
                )
        text = json.dumps(payload, indent=2) + "\n"
        if args.out:
            args.out.write_text(text)
        else:
            print(text, end="")
        status = payload.get("status") if args.cmd == "dryrun" else None
        return 0 if status in (None, "pass") else 1
    except (ServingConfigError, ServingAlternatesError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2
    finally:
        cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
