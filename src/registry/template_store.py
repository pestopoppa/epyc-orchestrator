"""SEC-ART-6: content-addressed, read-only chat-template store.

handoffs/active/model-artifact-integrity.md (SEC-ART-4 design, "TOCTOU"). A chat template is code that runs on
every prompt (intake-2200#record: backdoors can live in a template with no weight change). Today the override
files under /mnt/raid0/llm/models/chat-templates/ are plain files owned by the service user, so any agent session
can rewrite one in place and the ONLY barrier is the launch-time hash check (SEC-ART-4).

This module makes the file path itself the pin:

    <STORE_ROOT>/<64-hex sha256 of the bytes>.jinja      file 0444, store directory 0555

Rewriting a template therefore means publishing NEW bytes under a NEW name and re-pointing the registry
(`server_mode.<role>.chat_template_file`), which is a reviewed registry change and moves the pinned digest with it. A
file whose bytes no longer hash to its own name is a tamper signal that every consumer detects with one read.

Honest limit: agent sessions run as the same uid as the service, so mode bits are a speed bump, not a wall
(`chmod u+w` works for the owner). The barrier that does not depend on the mode bits is the name==hash invariant
plus the registry pin. Making the store truly unwritable needs a different owner (`sudo chown -R root:root` +
`chattr +i`), an operator action listed in the SEC-ART-6 decision package; `verify_store` reports the owner so
the state is visible either way.

Pure stdlib, no registry loader import. Consumers:
  * the SEC-ART-4 launcher check: `check_template_path(path, expected_sha=<pin>)` before `--chat-template-file`
    is emitted (returns the digest, raises TemplateStoreError otherwise);
  * the registry validator / a CI gate: `check_registry_template_store(registry, strict=...)`;
  * operators: `python -m src.registry.template_store publish|verify|path ...`.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any

DEFAULT_STORE_ROOT = "/mnt/raid0/llm/models/chat-templates/sha256"
STORE_ROOT_ENV = "EPYC_TEMPLATE_STORE"
STRICT_ENV = "REGISTRY_TEMPLATE_STORE_STRICT"
SUFFIX = ".jinja"
_NAME_RE = re.compile(r"^([0-9a-f]{64})\.jinja$")
_SHA64_RE = re.compile(r"^[0-9a-f]{64}$")
_SHA12_RE = re.compile(r"^[0-9a-f]{12}$")


class TemplateStoreError(ValueError):
    """A template path or store entry violates the content-addressed, read-only contract."""


def store_root(root: str | os.PathLike[str] | None = None) -> Path:
    return Path(root or os.environ.get(STORE_ROOT_ENV) or DEFAULT_STORE_ROOT)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def store_path(sha: str, root: str | os.PathLike[str] | None = None) -> Path:
    if not _SHA64_RE.match(sha):
        raise TemplateStoreError(f"not a 64-hex lowercase sha256: {sha!r}")
    return store_root(root) / f"{sha}{SUFFIX}"


def publish(src: str | os.PathLike[str], root: str | os.PathLike[str] | None = None) -> tuple[str, Path]:
    """Copy `src` into the store under its own digest; returns (sha256, store path). Idempotent.

    Never overwrites: if the target exists its bytes must already hash to its name (else TemplateStoreError,
    a tampered entry must be investigated, not healed silently). The store directory is created 0555 and is
    made writable only for the duration of the write.
    """
    data = Path(src).read_bytes()
    if not data:
        raise TemplateStoreError(f"refusing to publish an empty template: {src}")
    sha = sha256_bytes(data)
    base = store_root(root)
    target = base / f"{sha}{SUFFIX}"
    base.parent.mkdir(parents=True, exist_ok=True)
    if not base.exists():
        base.mkdir()
    if target.exists():
        if sha256_bytes(target.read_bytes()) != sha:
            raise TemplateStoreError(f"store entry {target} does not hash to its own name: tampered, not overwriting")
        _seal(target, base)
        return sha, target
    mode = stat.S_IMODE(base.stat().st_mode)
    os.chmod(base, mode | stat.S_IWUSR)
    try:
        fd, tmp = tempfile.mkstemp(prefix=".publish-", dir=base)
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
            os.chmod(tmp, 0o444)
            os.link(tmp, target)          # fails if target appeared meanwhile: no overwrite race
        except FileExistsError:
            if sha256_bytes(target.read_bytes()) != sha:
                raise TemplateStoreError(f"store entry {target} appeared with different bytes") from None
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass
    finally:
        os.chmod(base, 0o555)
    return sha, target


def _seal(target: Path, base: Path) -> None:
    """Re-apply 0444 / 0555 (idempotent publish repairs a loosened mode, never a changed content)."""
    try:
        if stat.S_IMODE(target.stat().st_mode) != 0o444:
            os.chmod(target, 0o444)
        if stat.S_IMODE(base.stat().st_mode) != 0o555:
            os.chmod(base, 0o555)
    except OSError:
        pass


def check_template_path(
    path: str | os.PathLike[str],
    *,
    expected_sha: str | None = None,
    root: str | os.PathLike[str] | None = None,
    require_readonly: bool = True,
) -> str:
    """Prove `path` is a store entry whose bytes are what its name says. Returns the sha256.

    Raises TemplateStoreError when: the resolved path is outside the store (symlinks are resolved first, so a
    link planted in the store cannot point out of it); the name is not `<64hex>.jinja`; the bytes do not hash to
    the name; `expected_sha` (a 64-hex pin, or the 12-hex legacy prefix) differs; or the file is group/other
    writable (require_readonly). The owner-write bit is NOT checked (see the module docstring: same uid).
    """
    base = store_root(root).resolve()
    real = Path(os.path.realpath(path))
    if real.parent != base:
        raise TemplateStoreError(f"{path} resolves to {real}, outside the template store {base}")
    m = _NAME_RE.match(real.name)
    if not m:
        raise TemplateStoreError(f"{real.name} is not <64hex>{SUFFIX}")
    named = m.group(1)
    try:
        actual = sha256_bytes(real.read_bytes())
    except OSError as exc:
        raise TemplateStoreError(f"cannot read {real}: {exc}") from exc
    if actual != named:
        raise TemplateStoreError(f"{real} hashes to {actual}, not its name {named}: template was modified in place")
    if expected_sha:
        if _SHA64_RE.match(expected_sha):
            ok = expected_sha == named
        elif _SHA12_RE.match(expected_sha):
            ok = named.startswith(expected_sha)
        else:
            raise TemplateStoreError(f"expected_sha {expected_sha!r} is neither 64-hex nor 12-hex")
        if not ok:
            raise TemplateStoreError(f"{real.name}: registry pin {expected_sha} does not match the stored digest")
    if require_readonly and real.stat().st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise TemplateStoreError(f"{real} is group/other-writable")
    return named


def verify_store(root: str | os.PathLike[str] | None = None) -> list[str]:
    """Walk the whole store; returns problem strings (empty = sound). A missing store is one problem."""
    base = store_root(root)
    if not base.is_dir():
        return [f"template store {base} does not exist"]
    problems: list[str] = []
    dmode = stat.S_IMODE(base.stat().st_mode)
    if dmode & (stat.S_IWGRP | stat.S_IWOTH):
        problems.append(f"store dir {base} is group/other-writable ({oct(dmode)})")
    if dmode & stat.S_IWUSR:
        problems.append(f"store dir {base} is owner-writable ({oct(dmode)}); expected 0555 (run publish or chmod 0555)")
    for entry in sorted(base.iterdir()):
        if entry.name.startswith(".publish-"):
            problems.append(f"leftover temp file {entry.name}")
            continue
        if entry.is_symlink() or not entry.is_file():
            problems.append(f"{entry.name}: not a regular file")
            continue
        try:
            check_template_path(entry, root=base)
        except TemplateStoreError as exc:
            problems.append(str(exc))
        if stat.S_IMODE(entry.stat().st_mode) & stat.S_IWUSR:
            problems.append(f"{entry.name}: owner-writable (expected 0444)")
    return problems


def check_registry_template_store(
    registry: dict[str, Any],
    *,
    strict: bool | None = None,
    root: str | os.PathLike[str] | None = None,
) -> tuple[list[str], list[str]]:
    """SEC-ART-6 over `server_mode.<role>.chat_template_file`. Returns (errors, warnings).

    A template path INSIDE the store must pass `check_template_path` against the role's pin
    (`chat_template.served_template_sha256`, else `served_template_sha256_12`; the `chat_template` pin block sits
    beside the path under `server_mode.<role>`). A path OUTSIDE the store is a
    warning until the registry is migrated, an error under strict (`REGISTRY_TEMPLATE_STORE_STRICT=1`).
    """
    if strict is None:
        strict = os.environ.get(STRICT_ENV, "").strip().lower() in ("1", "true", "yes", "on")
    base = store_root(root)
    errors: list[str] = []
    warnings: list[str] = []
    # The live registry declares the file under `server_mode.<role>` (stack_priors reads it there);
    # `roles.<role>` is accepted too so a future move does not silently blind this check.
    for section in ("server_mode", "roles"):
        for role, cfg in sorted((registry.get(section) or {}).items()):
            if not isinstance(cfg, dict) or not cfg.get("chat_template_file"):
                continue
            where = f"{section}.{role}.chat_template_file"
            path = str(cfg["chat_template_file"])
            pins = cfg.get("chat_template") if isinstance(cfg.get("chat_template"), dict) else {}
            pin = pins.get("served_template_sha256") or pins.get("served_template_sha256_12")
            try:
                inside = Path(os.path.realpath(path)).parent == base.resolve()
            except OSError:
                inside = False
            if not inside:
                (errors if strict else warnings).append(
                    f"{where}={path} is outside the read-only template store {base}"
                )
                continue
            try:
                check_template_path(path, expected_sha=str(pin) if pin else None, root=base)
            except TemplateStoreError as exc:
                errors.append(f"{where}: {exc}")
    return errors, warnings


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="content-addressed read-only chat-template store (SEC-ART-6)")
    ap.add_argument("--root", help=f"store directory (default ${STORE_ROOT_ENV} or {DEFAULT_STORE_ROOT})")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("publish", help="copy template files into the store under their sha256")
    p.add_argument("files", nargs="+")
    sub.add_parser("verify", help="check every store entry (name==hash, modes); exit 3 on any problem")
    q = sub.add_parser("path", help="print the store path for a sha256")
    q.add_argument("sha")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "publish":
            for f in a.files:
                sha, target = publish(f, a.root)
                print(f"{sha}  {target}  <- {f}")
            return 0
        if a.cmd == "path":
            print(store_path(a.sha, a.root))
            return 0
        problems = verify_store(a.root)
        for line in problems:
            print(f"PROBLEM: {line}", file=sys.stderr)
        n = len([e for e in store_root(a.root).glob(f"*{SUFFIX}")]) if store_root(a.root).is_dir() else 0
        print(f"template store {store_root(a.root)}: {n} entries, {len(problems)} problem(s)")
        return 3 if problems else 0
    except TemplateStoreError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
