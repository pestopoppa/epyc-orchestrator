"""Hardened pickle boundary for REPL checkpoint state.

Context
-------
REPL checkpoints are JSON-only, so REPL-defined functions/classes, numpy arrays
and other rich objects are dropped at save time. This module is the operator-
approved (2026-07-27) mechanism for carrying a bounded subset of them across a
resume without arming a deserialization RCE.

Threat model
------------
The objects being pickled are authored by *model-written REPL code*. Plain
``pickle.loads`` would therefore execute attacker-influenced ``__reduce__``
payloads in the orchestrator process, outside the REPL sandbox — with real
builtins and no AST security visitor. Note that ``pickle`` is itself listed in
``ASTSecurityVisitor.FORBIDDEN_MODULES``: the sandbox already treats it as
unsafe for model code, and this module must not become a way around that.

Layers
------
1. ``find_class`` allowlist (the load-bearing one). Unpickling can only resolve
   an explicit ``(module, name)`` pair from :data:`ALLOWED_GLOBALS`. A crafted
   ``__reduce__`` returning ``(__import__, ("os",))`` pickles fine but fails
   closed at load: ``builtins.__import__`` is not on the list. Model-defined
   classes live in ``__main__``, which is likewise absent, so they cannot be
   reconstructed at all. No ``operator``/``functools`` gadgets, no
   ``copyreg._reconstructor``.

   **"Allowlisted" is not the same as "inert."** An adversarial review on
   2026-07-27 demonstrated that ``numpy.ndarray`` — nominally a data type —
   is a memory-corruption primitive in its constructor form: ``ndarray(shape,
   dtype('O'), buffer=...)`` reinterprets the buffer bytes as raw ``PyObject*``
   pointers with no validation, yielding an arbitrary-pointer dereference
   (verified end-to-end: a signed checkpoint segfaulted a fresh restore).
   Entries that are types are therefore *individually* assessed for
   constructor-reachable primitives, not trusted for being types. See
   :class:`_NDArrayTypeToken` and :func:`_guarded_numpy_scalar`.
2. Constructor guards on the entries that need them, plus
   :func:`_assert_array_safe` — a metadata-only structural check on every value
   entering and leaving this module. It rejects object-dtype arrays (their
   storage *is* a pointer array) and arrays whose LOGICAL size exceeds
   :data:`MAX_ARRAY_BYTES`, since a stride-0 view is a ~100-byte pickle
   describing a multi-gigabyte array.
3. HMAC over the blob. Proves *we* wrote it, so a tampered SQLite row or state
   file is rejected. This addresses tampering at rest **only** — it does not
   establish that the content is safe, because we are the ones pickling
   model-authored objects. Layers 1-2 are what make the content safe.
4. Size cap on the encoded payload. Bounds storage and unpickle-time work.

A fifth layer lives in ``security.py``: the AST visitor rejects REPL code that
binds a serialization hook. Note it can never be complete — a hook name can
always be COMPUTED (``ns["__redu" + "ce__"]``) and no static check sees that.
What *is* closed are the two routes that install such a dict as a class
namespace (``type(n, b, d)`` and a custom metaclass), which makes computed keys
inert. Treat this layer as depth, never as the guarantee.

See ``handoffs/active/repl-session-memory-maturity.md`` D-a.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import io
import logging
import os
import pickle
import secrets
import stat
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Per-variable cap on the pickled payload, in bytes.
MAX_PICKLED_BYTES = 5_000_000

# Pickle protocol pinned so stored blobs stay readable across interpreter bumps.
PICKLE_PROTOCOL = 4

_HMAC_KEY_ENV = "ORCHESTRATOR_SESSION_HMAC_KEY"
_HMAC_KEY_FILENAME = "session_pickle_hmac.key"

# ---------------------------------------------------------------------------
# Allowlist
# ---------------------------------------------------------------------------
# INVARIANT: every entry must be an inert data type. Adding anything callable —
# or anything whose construction runs user-controlled code — defeats layer 1.
# Deliberately absent: builtins.eval/exec/__import__/getattr, os.*, subprocess.*,
# operator.attrgetter/methodcaller, functools.partial, copyreg._reconstructor,
# and `__main__` (so REPL-defined classes can never be reconstructed).
_BASE_ALLOWED: frozenset[tuple[str, str]] = frozenset(
    {
        ("builtins", "list"),
        ("builtins", "dict"),
        ("builtins", "set"),
        ("builtins", "frozenset"),
        ("builtins", "tuple"),
        ("builtins", "bytes"),
        ("builtins", "bytearray"),
        ("builtins", "str"),
        ("builtins", "int"),
        ("builtins", "float"),
        ("builtins", "bool"),
        ("builtins", "complex"),
        ("collections", "OrderedDict"),
        ("collections", "defaultdict"),
        ("collections", "Counter"),
        ("collections", "deque"),
        ("datetime", "datetime"),
        ("datetime", "date"),
        ("datetime", "time"),
        ("datetime", "timedelta"),
        ("datetime", "timezone"),
        ("decimal", "Decimal"),
        ("fractions", "Fraction"),
    }
)

# numpy reconstruction needs a small number of internal entry points. The module
# path moved between numpy versions, so both spellings are listed.
_NUMPY_ALLOWED: frozenset[tuple[str, str]] = frozenset(
    {
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy.core.multiarray", "scalar"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "scalar"),
    }
)

ALLOWED_GLOBALS: frozenset[tuple[str, str]] = _BASE_ALLOWED | _NUMPY_ALLOWED

# NOTE: pandas is intentionally NOT allowlisted in v1. Unpickling a DataFrame
# pulls in a wide surface of pandas internals (block managers, index
# constructors) that cannot be audited as tightly as the list above. DataFrames
# continue to be reported as unavailable by the restore reconciliation. Revisit
# only with a dedicated review — see D-a follow-ups.


class UnsafePickleError(Exception):
    """Raised when a payload fails an allowlist, HMAC, or size check."""


#: Cap on the LOGICAL size of a reconstructed array. The byte cap bounds the
#: encoded payload; a stride-0 view is a ~100-byte pickle describing a
#: multi-gigabyte array, so the logical size needs its own bound.
MAX_ARRAY_BYTES = 64_000_000


class _NDArrayTypeToken:
    """Stand-in for ``numpy.ndarray`` that cannot be called.

    A legitimate numpy pickle never invokes ``ndarray`` as a constructor — it
    passes the class as the ``subtype`` argument to ``multiarray._reconstruct``
    and fills the data in via ``__setstate__`` (verified against the actual
    opcode stream for int/float/str arrays and numpy scalars).

    The constructor form is a memory-corruption primitive: ``ndarray(shape,
    dtype('O'), buffer=...)`` reinterprets the raw buffer bytes as ``PyObject*``
    pointers with NO validation, so a crafted ``__reduce__`` yields an
    arbitrary-pointer dereference in the orchestrator process. Making the token
    non-callable removes the primitive while keeping the legitimate path intact.
    """

    __slots__ = ()

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        # Reached only if a pickle stream tries to CALL numpy.ndarray. Raise our
        # own error rather than leaning on an incidental TypeError, so the reason
        # is unambiguous in logs and no argument shape can quietly succeed.
        raise UnsafePickleError(
            "direct numpy.ndarray construction is not allowed in a checkpoint "
            "(the buffer form is an arbitrary-pointer primitive)"
        )


def _guarded_numpy_reconstruct(fn: Any) -> Any:
    """Wrap ``multiarray._reconstruct`` to accept only the ndarray token."""

    def _reconstruct(subtype: Any, *args: Any, **kwargs: Any) -> Any:
        import numpy as _np

        if isinstance(subtype, _NDArrayTypeToken) or subtype is _NDArrayTypeToken:
            subtype = _np.ndarray
        elif subtype is not _np.ndarray:
            raise UnsafePickleError(
                f"_reconstruct subtype must be numpy.ndarray, got {subtype!r}"
            )
        return fn(subtype, *args, **kwargs)

    return _reconstruct


def _assert_array_safe(value: Any, _seen: set[int] | None = None) -> None:
    """Reject object-dtype and implausibly large arrays anywhere in a value.

    Reads only array METADATA (``dtype``, ``nbytes``) — never an element — so it
    is safe to run against a structure that may contain a forged pointer.
    Object-dtype arrays are refused outright: their storage IS a pointer array,
    which is the primitive this boundary exists to deny.
    """
    if _seen is None:
        _seen = set()
    if id(value) in _seen:
        return
    _seen.add(id(value))

    cls_name = type(value).__name__
    if cls_name == "ndarray" or hasattr(value, "dtype") and hasattr(value, "nbytes"):
        dtype = getattr(value, "dtype", None)
        if dtype is not None and getattr(dtype, "hasobject", False):
            raise UnsafePickleError("object-dtype arrays are not allowed")
        nbytes = getattr(value, "nbytes", 0)
        if isinstance(nbytes, int) and nbytes > MAX_ARRAY_BYTES:
            raise UnsafePickleError(
                f"array logical size {nbytes} exceeds cap {MAX_ARRAY_BYTES}"
            )
        return

    if isinstance(value, dict):
        for k, v in value.items():
            _assert_array_safe(k, _seen)
            _assert_array_safe(v, _seen)
    elif isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            _assert_array_safe(item, _seen)


def _guarded_numpy_scalar(fn: Any) -> Any:
    """Wrap ``numpy.*.multiarray.scalar`` to refuse object dtypes.

    An object-dtype scalar carries a nested pickle in its data buffer, which
    numpy would deserialize with plain ``pickle.loads`` — escaping this module's
    allowlist entirely. Refuse it here rather than relying on the numpy version.
    """

    def _scalar(dtype: Any, *args: Any, **kwargs: Any) -> Any:
        kind = getattr(dtype, "kind", None)
        if kind == "O" or getattr(dtype, "hasobject", False):
            raise UnsafePickleError("object-dtype numpy scalars are not allowed")
        return fn(dtype, *args, **kwargs)

    return _scalar


class _AllowlistUnpickler(pickle.Unpickler):
    """Unpickler that can only resolve explicitly allowlisted globals."""

    def find_class(self, module: str, name: str) -> Any:  # noqa: D102
        if (module, name) not in ALLOWED_GLOBALS:
            raise UnsafePickleError(
                f"blocked global during unpickle: {module}.{name} is not allowlisted"
            )
        if name == "ndarray" and module == "numpy":
            # Type token only — never callable. See _NDArrayTypeToken.
            return _NDArrayTypeToken
        resolved = super().find_class(module, name)
        if name == "_reconstruct":
            return _guarded_numpy_reconstruct(resolved)
        if name == "scalar":
            # numpy's multiarray.scalar() historically deserialized an
            # object-dtype scalar by calling pickle.loads() on its raw data
            # buffer — i.e. a nested, UNGUARDED unpickle inside our guarded one.
            # Current numpy refuses it ("Cannot unpickle a scalar with object
            # dtype"), but that is a numpy-version-dependent invariant and this
            # allowlist must not silently depend on it. Wrap it ourselves.
            return _guarded_numpy_scalar(resolved)
        return resolved

    def persistent_load(self, pid: Any) -> Any:  # noqa: D102
        # We never emit persistent ids; refuse them rather than inventing a
        # resolution path an attacker could steer.
        raise UnsafePickleError("persistent ids are not supported")


def _resolve_hmac_key() -> bytes:
    """Return the HMAC key, generating and persisting one on first use.

    Precedence: ``ORCHESTRATOR_SESSION_HMAC_KEY`` env var, else a 0600 key file
    beside the session store. A stable key is required or every restart would
    invalidate stored sessions.
    """
    env_key = os.environ.get(_HMAC_KEY_ENV)
    if env_key:
        return env_key.encode("utf-8")

    from src.config import get_config

    key_dir = Path(get_config().paths.sessions_dir)
    key_path = key_dir / _HMAC_KEY_FILENAME

    def _read() -> bytes | None:
        # Hex-encoded on disk: raw random bytes could contain leading/trailing
        # whitespace, and any strip()-style normalization would silently alter
        # the key. An empty/short read means we caught a partially-written file.
        try:
            raw = key_path.read_text(encoding="ascii").strip()
        except (FileNotFoundError, ValueError, UnicodeDecodeError):
            return None
        if len(raw) < 64:
            return None
        try:
            return bytes.fromhex(raw)
        except ValueError:
            return None

    existing = _read()
    if existing is not None:
        return existing

    key_dir.mkdir(parents=True, exist_ok=True)
    key = secrets.token_bytes(32)
    # Write to a private temp file and rename into place. os.open(O_CREAT|O_EXCL)
    # directly on key_path would publish a ZERO-LENGTH file that a concurrent
    # worker could read as an empty HMAC key between create and write; rename is
    # atomic, so a reader sees either no file or the complete key.
    tmp_path = key_path.with_suffix(f".{os.getpid()}.tmp")
    fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, stat.S_IRUSR | stat.S_IWUSR)
    try:
        os.write(fd, key.hex().encode("ascii"))
        os.fsync(fd)
    finally:
        os.close(fd)
    try:
        os.replace(tmp_path, key_path)
        logger.info("Generated a new session pickle HMAC key at %s", key_path)
        return key
    except OSError:
        tmp_path.unlink(missing_ok=True)
        raise

    # NOTE: os.replace clobbers a key written by a racing worker. Both keys are
    # valid-length secrets and the loser's in-flight checkpoints simply fail
    # their HMAC on a later restore (fail-closed, reported as unavailable), which
    # is the correct outcome for a first-start race.


def _sign(blob: bytes) -> str:
    return hmac.new(_resolve_hmac_key(), blob, hashlib.sha256).hexdigest()


def is_safely_picklable(value: Any) -> bool:
    """Cheap pre-check: would :func:`dumps` accept this value?"""
    try:
        dumps(value)
        return True
    except Exception:
        return False


def dumps(value: Any) -> dict[str, Any]:
    """Serialize ``value`` to a signed, size-capped envelope.

    Round-trips through the allowlist unpickler before returning, so a value we
    could never load back is rejected at *save* time rather than silently
    persisting and failing on a future resume.

    Raises:
        UnsafePickleError: if the value exceeds the cap or is not loadable
            under the allowlist.
    """
    try:
        blob = pickle.dumps(value, protocol=PICKLE_PROTOCOL)
    except Exception as e:
        raise UnsafePickleError(f"not picklable: {type(e).__name__}: {e}") from e

    if len(blob) > MAX_PICKLED_BYTES:
        raise UnsafePickleError(
            f"too large to save: {len(blob)} bytes (cap {MAX_PICKLED_BYTES})"
        )

    # Verify loadability under the allowlist NOW. This is what turns a hostile
    # __reduce__ into a save-time rejection instead of a restore-time surprise.
    try:
        _assert_array_safe(value)
        _assert_array_safe(_AllowlistUnpickler(io.BytesIO(blob)).load())
    except UnsafePickleError:
        raise
    except Exception as e:
        raise UnsafePickleError(f"not loadable under allowlist: {type(e).__name__}: {e}") from e

    return {
        "b64": base64.b64encode(blob).decode("ascii"),
        "hmac": _sign(blob),
        "type": type(value).__name__,
        "bytes": len(blob),
    }


def loads(envelope: dict[str, Any]) -> Any:
    """Verify and deserialize an envelope produced by :func:`dumps`.

    Raises:
        UnsafePickleError: on a malformed envelope, HMAC mismatch, oversize
            payload, or a global outside the allowlist.
    """
    if not isinstance(envelope, dict):
        raise UnsafePickleError(f"malformed envelope: expected dict, got {type(envelope).__name__}")

    b64 = envelope.get("b64")
    signature = envelope.get("hmac")
    if not isinstance(b64, str) or not isinstance(signature, str):
        raise UnsafePickleError("malformed envelope: missing b64 or hmac")

    # Bound the DECODED size from the encoded length before allocating, so an
    # oversize field cannot force a large decode first. base64 is 4:3.
    if (len(b64) // 4) * 3 > MAX_PICKLED_BYTES:
        raise UnsafePickleError(f"payload exceeds cap: ~{(len(b64) // 4) * 3} bytes encoded")

    try:
        blob = base64.b64decode(b64, validate=True)
    except Exception as e:
        raise UnsafePickleError(f"malformed envelope: bad base64: {e}") from e

    # Check size before verifying, so an oversize payload is cheap to reject.
    if len(blob) > MAX_PICKLED_BYTES:
        raise UnsafePickleError(f"payload exceeds cap: {len(blob)} bytes")

    if not hmac.compare_digest(_sign(blob), signature):
        raise UnsafePickleError("HMAC mismatch: payload was not written by this orchestrator")

    try:
        loaded = _AllowlistUnpickler(io.BytesIO(blob)).load()
        # Metadata-only structural check BEFORE the value escapes this function,
        # so a forged pointer is never handed to a caller that might deref it.
        _assert_array_safe(loaded)
        return loaded
    except UnsafePickleError:
        raise
    except Exception as e:
        raise UnsafePickleError(f"unpickle failed: {type(e).__name__}: {e}") from e
