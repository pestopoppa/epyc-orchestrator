"""Eval-only opaque-secret tool with RUNTIME-GENERATED secrets.

An earlier version hardcoded the secret VALUES in this module (and repeated them
as `expected:` in tool_sentinels.yaml). That made the "obtainable only via the
tool" claim FALSE: a model with repo file/search tools could grep them out of
source instead of calling get_eval_secret. Now the values are minted at runtime
(`secrets.token_hex`) inside the orchestrator process, held in memory, and
written ONLY to a tmpfs file OUTSIDE the model's read_file allowed roots
(`/mnt/raid0/llm`, `/tmp`) so the EVAL HARNESS can read ground truth while the
model-under-test's read_file cannot. Nothing secret is committed to source.

The secret is minted ONCE (atomic load-or-create) and is STABLE across
orchestrator reloads and concurrent uvicorn workers: every reload/worker loads
the existing tmpfs value rather than re-minting. This matters because the eval
harness reads `expected` from the file at trial start and scores against it
later — re-minting between those points (the old behavior) made tool_use answers
score wrong even when the tool was correctly called.

HONEST residual surface: this is best-effort, not a proof. If some REPL read
helper accepts arbitrary absolute paths it could still reach the tmpfs file.
The AUTHORITATIVE gate signal is therefore the DIRECTLY-MEASURED get_eval_secret
call count (request-local telemetry — see repl_executor/context.py), NOT
answer-correctness. The runtime generation removes the realistic bypass
(grepping the repo); the call-count measurement is what actually guarantees the
model used the tool.
"""
from __future__ import annotations

import fcntl
import json
import os
import secrets as _secrets
import tempfile
from pathlib import Path

SECRET_NAMES: tuple[str, ...] = ("alpha", "bravo", "charlie", "delta", "echo")

# tmpfs (RAM-backed) and OUTSIDE read_file's allowed roots (/mnt/raid0/llm,
# /tmp); repo-scoped grep/code_search can't reach it either. Overridable for
# tests via EVAL_SECRETS_PATH.
SECRETS_PATH = Path(os.environ.get("EVAL_SECRETS_PATH", "/dev/shm/epyc_eval_secrets.json"))

# Populated at runtime by generate_and_persist_secrets(); EMPTY at import so no
# secret value is ever baked into source (the test asserts this).
_SECRETS: dict[str, str] = {}


def _valid_secrets(data: object) -> bool:
    """A complete, non-empty string value for every expected name."""
    return (
        isinstance(data, dict)
        and set(data.keys()) == set(SECRET_NAMES)
        and all(isinstance(v, str) and v for v in data.values())
    )


def _mint() -> dict[str, str]:
    return {name: f"EVS-{_secrets.token_hex(8)}" for name in SECRET_NAMES}


def generate_and_persist_secrets() -> dict[str, str]:
    """Atomic LOAD-OR-CREATE of the eval secrets — stable across orchestrator
    reloads and safe under concurrent multi-worker first-start (uvicorn --workers,
    each worker registers built-ins).

    Earlier this unconditionally re-minted on every call, so each orchestrator
    reload changed the secret out from under the harness mid-trial (the harness
    reads `expected` from the file at trial start, then scores against it later)
    → tool_use answers scored wrong even though the tool was called. Now:

    - Fast path: if the file already holds a complete valid set, LOAD it (never
      overwrite) — a reload keeps the same secret.
    - Miss path: serialize with an flock, RE-CHECK under the lock (a concurrent
      worker may have just created it), then publish atomically (temp file →
      chmod 0600 → os.replace) so readers never see a partial file. All starters
      converge on the single winning set.
    """
    global _SECRETS
    # Fast path — existing valid file (the common reload case); lock-free.
    existing = load_persisted_secrets()
    if _valid_secrets(existing):
        _SECRETS = existing
        return dict(_SECRETS)

    try:
        SECRETS_PATH.parent.mkdir(parents=True, exist_ok=True)
        lock_path = SECRETS_PATH.with_name(SECRETS_PATH.name + ".lock")
        with open(lock_path, "w") as lock_f:
            fcntl.flock(lock_f, fcntl.LOCK_EX)
            # Re-check under the lock: a concurrent worker may have created it.
            existing = load_persisted_secrets()
            if _valid_secrets(existing):
                _SECRETS = existing
                return dict(_SECRETS)
            candidate = _mint()
            fd, tmp = tempfile.mkstemp(
                dir=str(SECRETS_PATH.parent), prefix=".eval_secrets.", suffix=".tmp"
            )
            try:
                os.write(fd, json.dumps(candidate).encode())
                os.fchmod(fd, 0o600)
                os.close(fd)
                fd = -1
                os.replace(tmp, str(SECRETS_PATH))  # atomic publish under lock
                tmp = None
            finally:
                if fd != -1:
                    os.close(fd)
                if tmp is not None:
                    try:
                        os.unlink(tmp)
                    except OSError:
                        pass
            _SECRETS = candidate
            return dict(_SECRETS)
    except OSError:
        # tmpfs/lock unavailable — fall back to an in-memory mint so the tool
        # still returns a value (harness will warn it cannot read ground truth).
        if not _valid_secrets(_SECRETS):
            _SECRETS = _mint()
        return dict(_SECRETS)


def load_persisted_secrets() -> dict[str, str]:
    """Harness-side: read the ground-truth secrets the orchestrator minted.
    Returns {} if unavailable (harness then scores those questions incorrect
    rather than spuriously passing on an empty expected)."""
    try:
        data = json.loads(SECRETS_PATH.read_text())
    except (OSError, ValueError):
        return {}
    return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}


def get_eval_secret(name: str) -> str:
    """Return the in-memory secret for `name` (orchestrator-side). Lazily mints
    the set if registration somehow didn't run, so a call always yields a value.
    """
    key = str(name).strip().lower()
    if not _SECRETS:
        generate_and_persist_secrets()
    if key not in _SECRETS:
        return f"ERROR: unknown secret name {name!r}. Valid names: {sorted(SECRET_NAMES)}"
    return _SECRETS[key]
