"""Eval-only opaque-secret tool with RUNTIME-GENERATED secrets.

An earlier version hardcoded the secret VALUES in this module (and repeated them
as `expected:` in tool_sentinels.yaml). That made the "obtainable only via the
tool" claim FALSE: a model with repo file/search tools could grep them out of
source instead of calling get_eval_secret. Now the values are minted at runtime
(`secrets.token_hex`) inside the orchestrator process, held in memory, and
written ONLY to a tmpfs file OUTSIDE the model's read_file allowed roots
(`/mnt/raid0/llm`, `/tmp`) so the EVAL HARNESS can read ground truth while the
model-under-test's read_file cannot. Nothing secret is committed to source.

HONEST residual surface: this is best-effort, not a proof. If some REPL read
helper accepts arbitrary absolute paths it could still reach the tmpfs file.
The AUTHORITATIVE gate signal is therefore the DIRECTLY-MEASURED get_eval_secret
call count (request-local telemetry — see repl_executor/context.py), NOT
answer-correctness. The runtime generation removes the realistic bypass
(grepping the repo); the call-count measurement is what actually guarantees the
model used the tool.
"""
from __future__ import annotations

import json
import os
import secrets as _secrets
from pathlib import Path

SECRET_NAMES: tuple[str, ...] = ("alpha", "bravo", "charlie", "delta", "echo")

# tmpfs (RAM-backed) and OUTSIDE read_file's allowed roots (/mnt/raid0/llm,
# /tmp); repo-scoped grep/code_search can't reach it either. Overridable for
# tests via EVAL_SECRETS_PATH.
SECRETS_PATH = Path(os.environ.get("EVAL_SECRETS_PATH", "/dev/shm/epyc_eval_secrets.json"))

# Populated at runtime by generate_and_persist_secrets(); EMPTY at import so no
# secret value is ever baked into source (the test asserts this).
_SECRETS: dict[str, str] = {}


def generate_and_persist_secrets() -> dict[str, str]:
    """Orchestrator-side: mint fresh random secrets in memory and persist them to
    the model-unreachable tmpfs file for the eval harness. Called once at tool
    registration. Values are never written to source or to a model-readable path.
    """
    global _SECRETS
    _SECRETS = {name: f"EVS-{_secrets.token_hex(8)}" for name in SECRET_NAMES}
    try:
        SECRETS_PATH.write_text(json.dumps(_SECRETS))
        os.chmod(SECRETS_PATH, 0o600)
    except OSError:
        pass  # harness will warn if it cannot read ground truth
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
