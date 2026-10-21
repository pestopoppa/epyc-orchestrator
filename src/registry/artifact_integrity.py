"""Launch-time artifact integrity: GGUF fingerprint pins (SEC-ART-2) and served-template pins (SEC-ART-4).

Owner handoff: handoffs/active/model-artifact-integrity.md (epyc-root).

Pure helpers, no llama-server import and no network. Two independent checks run from
``verify_launch_argv`` on the FINAL argv of a llama-server launch, before any ``Popen``:

SEC-ART-2 -- GGUF bytes
    The registry pins ``roles.<role>.model.sha256`` (and ``shard_sha256`` for the non-first shards of a
    split GGUF). ``register_gguf`` hashes a file in full once and records a stat fingerprint
    (size, mtime_ns, inode) plus that digest in a LEDGER (host-local, never committed). At launch the
    fingerprint is compared; only a changed file is re-hashed, and a file whose digest differs from its
    pin is REFUSED. A never-registered file is a WARNING (REFUSAL when strict).

SEC-ART-4 -- chat template bytes
    The effective template is the ``--chat-template-file`` bytes if that flag is on the argv, else the
    GGUF's embedded ``tokenizer.chat_template`` (read from the GGUF header only, never tensor data). It is
    sha256'd and compared with ``served_template_sha256`` (64 lowercase hex). An inline ``--chat-template``
    cannot be pinned by path and is refused.

Strict switches (missing pin / missing ledger entry becomes a refusal) are resolved by
``strict_enabled`` from an env var first, then ``runtime_defaults.artifact_integrity.<key>`` in the
registry, so the flip is one reviewed registry edit (via stack-change) or one env var, never a code change.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import re
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Callable

import yaml

log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]

SHA64_RE = re.compile(r"^[0-9a-f]{64}$")
_SPLIT_RE = re.compile(r"^(?P<stem>.+)-(?P<idx>\d{5})-of-(?P<total>\d{5})\.gguf$")

# Strict switches: (env var, runtime_defaults.artifact_integrity key).
PINS_STRICT = ("REGISTRY_ARTIFACT_PINS_STRICT", "pins_strict")
TEMPLATE_PINS_STRICT = ("REGISTRY_TEMPLATE_PINS_STRICT", "template_pins_strict")

LEDGER_ENV = "ARTIFACT_FINGERPRINT_LEDGER"
DEFAULT_LEDGER = Path("/mnt/raid0/llm/cache/artifact_fingerprints.json")
REGISTRY_ENV = "ARTIFACT_INTEGRITY_REGISTRY"
DEFAULT_REGISTRY = _REPO_ROOT / "orchestration" / "model_registry.yaml"


class ArtifactIntegrityError(ValueError):
    """A launch must be refused. ValueError so existing launch-build refusal handling applies."""


class GgufPinError(ArtifactIntegrityError):
    def __init__(self, path: str, expected: str, actual: str, why: str = "sha256 mismatch") -> None:
        self.path, self.expected, self.actual = path, expected, actual
        super().__init__(f"GGUF {why}: {path} sha256 {actual} != pinned {expected}")


class TemplatePinError(ArtifactIntegrityError):
    def __init__(self, role: str, source: str, expected: str, actual: str, why: str = "") -> None:
        self.role, self.source, self.expected, self.actual = role, source, expected, actual
        super().__init__(
            f"template sha256 {actual} != pinned {expected} for role {role!r} ({source}){why}"
        )


# ---------------------------------------------------------------- strict switches / registry access

def strict_enabled(switch: tuple[str, str], registry: dict[str, Any] | None = None) -> bool:
    """Env var wins (``1/true/yes/on`` = strict, ``0/false/no/off`` = lax); else the registry flag."""
    env, key = switch
    raw = os.environ.get(env, "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    block = ((registry or {}).get("runtime_defaults") or {}).get("artifact_integrity")
    return bool(isinstance(block, dict) and block.get(key) is True)


_REG_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}


def load_registry(path: str | Path | None = None) -> dict[str, Any]:
    """The lean runtime registry (mtime-cached). Missing/unreadable -> {} (callers treat as unpinned)."""
    p = Path(path or os.environ.get(REGISTRY_ENV) or DEFAULT_REGISTRY)
    try:
        mtime = p.stat().st_mtime
    except OSError:
        return {}
    hit = _REG_CACHE.get(str(p))
    if hit and hit[0] == mtime:
        return hit[1]
    try:
        data = yaml.safe_load(p.read_text()) or {}
    except (OSError, yaml.YAMLError):
        return {}
    _REG_CACHE[str(p)] = (mtime, data)
    return data


@dataclass(frozen=True)
class GgufPin:
    role: str
    path: str                      # absolute path named by the registry (first shard for a split GGUF)
    sha256: str | None
    shard_sha256: dict[str, str] = field(default_factory=dict)   # basename -> sha256 for non-first shards
    hf_repo: str | None = None
    hf_revision: str | None = None


def _abs_model_path(path: str, registry: dict[str, Any]) -> str:
    if path.startswith("/"):
        return path
    base = ((registry.get("runtime_defaults") or {}).get("model_base_path")) or "/mnt/raid0/llm/models"
    return os.path.join(base, path)


def pin_index(registry: dict[str, Any]) -> dict[str, GgufPin]:
    """realpath-keyed pins from ``roles.<role>.model`` (lineup roles with a ``.gguf`` path).

    Several roles may share one GGUF; the first with a ``sha256`` wins (the validator reports a conflict).
    """
    out: dict[str, GgufPin] = {}
    for role, cfg in sorted((registry.get("roles") or {}).items()):
        model = cfg.get("model") if isinstance(cfg, dict) else None
        if not isinstance(model, dict):
            continue
        raw = model.get("path") or model.get("full_path")
        if not (isinstance(raw, str) and raw.endswith(".gguf")):
            continue
        path = _abs_model_path(raw, registry)
        key = os.path.realpath(path)
        shards = model.get("shard_sha256")
        pin = GgufPin(
            role=role,
            path=path,
            sha256=model.get("sha256") if isinstance(model.get("sha256"), str) else None,
            shard_sha256=dict(shards) if isinstance(shards, dict) else {},
            hf_repo=model.get("hf_repo"),
            hf_revision=model.get("hf_revision"),
        )
        prior = out.get(key)
        if prior is None or (prior.sha256 is None and pin.sha256 is not None):
            out[key] = pin
    return out


# ---------------------------------------------------------------- hashing, fingerprints, ledger

def sha256_file(path: str | Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def shard_paths(path: str | Path) -> list[str]:
    """``[path]`` for a single GGUF; every sibling shard (index order) for ``<stem>-00001-of-0000N.gguf``."""
    p = Path(path)
    m = _SPLIT_RE.match(p.name)
    if not m:
        return [str(p)]
    total = int(m.group("total"))
    return [str(p.with_name(f"{m.group('stem')}-{i:05d}-of-{total:05d}.gguf")) for i in range(1, total + 1)]


def stat_fingerprint(path: str | Path) -> dict[str, int]:
    st = os.stat(path)
    return {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "ino": st.st_ino}


def ledger_path() -> Path:
    return Path(os.environ.get(LEDGER_ENV) or DEFAULT_LEDGER)


def load_ledger(path: Path | None = None) -> dict[str, dict[str, Any]]:
    p = path or ledger_path()
    try:
        data = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    entries = data.get("entries") if isinstance(data, dict) else None
    return entries if isinstance(entries, dict) else {}


def update_ledger(updates: dict[str, dict[str, Any]], path: Path | None = None) -> None:
    """Merge ``updates`` into the ledger under an exclusive lock (atomic replace)."""
    p = path or ledger_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    lock = p.with_name(p.name + ".lock")
    with open(lock, "a+") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            entries = load_ledger(p)
            entries.update(updates)
            tmp = p.with_name(p.name + f".tmp{os.getpid()}")
            tmp.write_text(json.dumps({"schema": 1, "entries": entries}, indent=1, sort_keys=True) + "\n")
            os.replace(tmp, p)
        finally:
            fcntl.flock(lk, fcntl.LOCK_UN)


def register_gguf(
    path: str | Path,
    *,
    pin_sha256: str | None = None,
    ledger: Path | None = None,
    hasher: Callable[[str | Path], str] = sha256_file,
) -> dict[str, Any]:
    """Hash ``path`` in full and record digest + stat fingerprint in the ledger.

    With ``pin_sha256``: a digest that differs raises ``GgufPinError`` and NOTHING is recorded (a ledger
    entry must never launder a bad file into a trusted one). Without a pin the entry is recorded with
    ``pin_sha256: None`` and the digest is the proposal to review into the registry.
    """
    real = os.path.realpath(path)
    t0 = time.monotonic()
    fp = stat_fingerprint(real)
    digest = hasher(real)
    if pin_sha256 and digest != pin_sha256:
        raise GgufPinError(real, pin_sha256, digest, "registration refused: sha256 mismatch")
    # The file must not have changed while it was being hashed.
    if stat_fingerprint(real) != fp:
        raise ArtifactIntegrityError(f"{real} changed while being hashed; re-run registration")
    entry = {
        **fp,
        "sha256": digest,
        "pin_sha256": pin_sha256,
        "hashed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hash_seconds": round(time.monotonic() - t0, 1),
    }
    update_ledger({real: entry}, ledger)
    return entry


def verify_gguf_for_launch(
    path: str | Path,
    pin: GgufPin | None,
    *,
    strict: bool,
    ledger: Path | None = None,
    hasher: Callable[[str | Path], str] = sha256_file,
) -> list[str]:
    """Check a GGUF (every shard of a split set) against its pin. Returns warnings; raises on refusal.

    Per file: expected digest = ``pin.sha256`` for the registry's own path, ``pin.shard_sha256[basename]``
    for another shard. Ledger entry whose stat fingerprint still matches and whose digest equals the pin ->
    accepted without reading the file. Changed stat -> re-hash; equal to the pin -> ledger refreshed and
    accepted; different -> ``GgufPinError``. No pin / no ledger entry -> warning, or refusal when strict.
    """
    warnings: list[str] = []
    p = str(path)
    if pin is None:
        msg = f"no registry pin for {p}"
        if strict:
            raise ArtifactIntegrityError(f"{msg} (strict: every served GGUF must be pinned)")
        return [msg]
    ledger_entries = load_ledger(ledger)
    refresh: dict[str, dict[str, Any]] = {}
    own = os.path.realpath(pin.path)
    for shard in shard_paths(p):
        real = os.path.realpath(shard)
        expected = pin.sha256 if real == own else pin.shard_sha256.get(os.path.basename(shard))
        if not expected:
            msg = f"{shard}: role {pin.role!r} carries no sha256 pin for this file"
            if strict:
                raise ArtifactIntegrityError(f"{msg} (strict)")
            warnings.append(msg)
            continue
        try:
            fp = stat_fingerprint(real)
        except OSError as exc:
            raise ArtifactIntegrityError(f"{shard}: cannot stat pinned GGUF ({exc})") from exc
        entry = ledger_entries.get(real)
        if entry is None:
            msg = f"{shard}: pinned but never registered (no full-hash ledger entry; run register_gguf_fingerprints.py)"
            if strict:
                raise ArtifactIntegrityError(f"{msg} (strict)")
            warnings.append(msg)
            continue
        same_stat = all(entry.get(k) == fp[k] for k in ("size", "mtime_ns", "ino"))
        if same_stat and entry.get("sha256") == expected:
            continue
        digest = hasher(real)
        if digest != expected:
            raise GgufPinError(real, expected, digest)
        if stat_fingerprint(real) != fp:
            raise ArtifactIntegrityError(f"{real} changed while being re-hashed; refusing")
        refresh[real] = {**entry, **fp, "sha256": digest, "pin_sha256": expected,
                         "rehashed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if refresh:
        try:
            update_ledger(refresh, ledger)
        except OSError as exc:  # a read-only ledger must not turn a verified file into a refusal
            warnings.append(f"ledger refresh failed ({exc}); file verified by re-hash")
    return warnings


# ---------------------------------------------------------------- GGUF header: embedded chat template

_GGUF_FIXED = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
_T_STRING, _T_ARRAY = 8, 9
CHAT_TEMPLATE_KEY = "tokenizer.chat_template"
_open = open  # test hook: a counting wrapper proves the tensor data is never read


def _read_exact(fh: BinaryIO, n: int) -> bytes:
    buf = fh.read(n)
    if len(buf) != n:
        raise TemplatePinError("?", "gguf", "-", "-", " (truncated GGUF header)")
    return buf


def _skip_value(fh: BinaryIO, vtype: int) -> None:
    if vtype in _GGUF_FIXED:
        fh.seek(_GGUF_FIXED[vtype], os.SEEK_CUR)
    elif vtype == _T_STRING:
        (n,) = struct.unpack("<Q", _read_exact(fh, 8))
        fh.seek(n, os.SEEK_CUR)
    elif vtype == _T_ARRAY:
        etype, count = struct.unpack("<IQ", _read_exact(fh, 12))
        if etype in _GGUF_FIXED:
            fh.seek(_GGUF_FIXED[etype] * count, os.SEEK_CUR)
        else:
            for _ in range(count):
                _skip_value(fh, etype)
    else:
        raise TemplatePinError("?", "gguf", "-", "-", f" (unknown GGUF value type {vtype})")


def read_gguf_chat_template(gguf_path: str | Path) -> bytes | None:
    """``tokenizer.chat_template`` bytes from the GGUF header, or None if the key is absent.

    Reads the KV section only (never tensor data). GGUF v2/v3; anything else raises ``TemplatePinError``.
    """
    with _open(gguf_path, "rb") as fh:
        if _read_exact(fh, 4) != b"GGUF":
            raise TemplatePinError("?", str(gguf_path), "-", "-", " (not a GGUF: bad magic)")
        (version,) = struct.unpack("<I", _read_exact(fh, 4))
        if version not in (2, 3):
            raise TemplatePinError("?", str(gguf_path), "-", "-", f" (unsupported GGUF version {version})")
        _tensors, n_kv = struct.unpack("<QQ", _read_exact(fh, 16))
        for _ in range(n_kv):
            (klen,) = struct.unpack("<Q", _read_exact(fh, 8))
            key = _read_exact(fh, klen)
            (vtype,) = struct.unpack("<I", _read_exact(fh, 4))
            if key == CHAT_TEMPLATE_KEY.encode():
                if vtype != _T_STRING:
                    raise TemplatePinError("?", str(gguf_path), "-", "-", " (chat_template is not a string)")
                (vlen,) = struct.unpack("<Q", _read_exact(fh, 8))
                return _read_exact(fh, vlen)
            _skip_value(fh, vtype)
    return None


def template_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def argv_value(argv: list[str], *flags: str) -> str | None:
    """Last value of any of ``flags`` on ``argv`` (handles ``--flag value`` and ``--flag=value``)."""
    found: str | None = None
    for i, tok in enumerate(argv):
        for f in flags:
            if tok == f and i + 1 < len(argv):
                found = argv[i + 1]
            elif tok.startswith(f + "="):
                found = tok[len(f) + 1:]
    return found


def effective_template_bytes(argv: list[str]) -> tuple[bytes | None, str]:
    """(bytes, source label) of the template llama-server will serve for this argv; mirrors its precedence.

    ``--chat-template-file`` bytes win; else the GGUF-embedded template (None if the GGUF has none).
    An inline ``--chat-template`` is refused (cannot be pinned by path).
    """
    if argv_value(argv, "--chat-template") is not None:
        raise ArtifactIntegrityError(
            "inline --chat-template cannot be pinned; serve it as --chat-template-file (SEC-ART-4)"
        )
    tfile = argv_value(argv, "--chat-template-file")
    if tfile:
        try:
            return Path(tfile).read_bytes(), f"file:{tfile}"
        except OSError as exc:
            raise ArtifactIntegrityError(f"cannot read --chat-template-file {tfile}: {exc}") from exc
    model = argv_value(argv, "-m", "--model")
    if not model:
        return None, "no-model"
    try:
        return read_gguf_chat_template(model), f"gguf:{model}"
    except OSError as exc:
        raise ArtifactIntegrityError(f"cannot read GGUF header {model}: {exc}") from exc


def check_template_pin(role: str, data: bytes | None, source: str, pinned: str | None, *, strict: bool) -> list[str]:
    """Compare ``data`` with the 64-hex pin. Returns warnings; raises ``TemplatePinError`` on mismatch."""
    if not pinned:
        msg = f"role {role!r} has no served_template_sha256 pin ({source})"
        if strict:
            raise TemplatePinError(role, source, "<none>", template_sha256(data or b""), " (strict: pin required)")
        return [msg]
    actual = template_sha256(data) if data is not None else "<no-template>"
    if actual != pinned:
        raise TemplatePinError(role, source, pinned, actual)
    return []


# ---------------------------------------------------------------- the launch funnel

_MODEL_FLAGS = ("-m", "--model", "-md", "--model-draft", "--mmproj", "-mm")


def verify_launch_argv(
    argv: list[str],
    *,
    role: str | None,
    template_pin: str | None,
    check_template: bool,
    registry: dict[str, Any] | None = None,
    ledger: Path | None = None,
    hasher: Callable[[str | Path], str] = sha256_file,
) -> list[str]:
    """Verify the artifacts a llama-server argv is about to load. Raises ``ArtifactIntegrityError``.

    ``check_template`` is False for template-less launches (embedders, ``worker_fast`` ...); then only the
    GGUF pins run. Returns the warnings (unpinned / unregistered files) for the caller to log.
    """
    reg = registry if registry is not None else load_registry()
    index = pin_index(reg)
    pins_strict = strict_enabled(PINS_STRICT, reg)
    warnings: list[str] = []
    seen: set[str] = set()
    for flag in _MODEL_FLAGS:
        for i, tok in enumerate(argv):
            value = argv[i + 1] if tok == flag and i + 1 < len(argv) else (
                tok[len(flag) + 1:] if tok.startswith(flag + "=") else None)
            if not value or not value.endswith(".gguf"):
                continue
            real = os.path.realpath(value)
            if real in seen:
                continue
            seen.add(real)
            # Only the lineup's own GGUFs are in the index; a path outside it is not ours to pin, but in
            # strict mode an unindexed -m model is a refusal (nothing served may be unpinned).
            pin = index.get(real)
            if pin is None and flag not in ("-m", "--model"):
                continue
            warnings += verify_gguf_for_launch(real, pin, strict=pins_strict, ledger=ledger, hasher=hasher)
    if check_template and role:
        data, source = effective_template_bytes(argv)
        warnings += check_template_pin(
            role, data, source, template_pin, strict=strict_enabled(TEMPLATE_PINS_STRICT, reg)
        )
    for w in warnings:
        log.warning("artifact integrity: %s", w)
    return warnings


__all__ = [
    "ArtifactIntegrityError", "GgufPinError", "TemplatePinError", "GgufPin", "PINS_STRICT",
    "TEMPLATE_PINS_STRICT", "strict_enabled", "load_registry", "pin_index", "sha256_file", "shard_paths",
    "stat_fingerprint", "load_ledger", "update_ledger", "register_gguf", "verify_gguf_for_launch",
    "read_gguf_chat_template", "template_sha256", "effective_template_bytes", "check_template_pin",
    "verify_launch_argv", "argv_value",
]
