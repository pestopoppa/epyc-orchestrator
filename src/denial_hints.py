"""Tool-denial messages that name what IS allowed (flag ``tool_denial_names_allowed``).

Flag OFF: every helper returns the legacy text unchanged. Flag ON: the denial also names the
allowed set (or says there is none), so a model that was refused can pick a permitted action
instead of guessing. Pure string building; no I/O.
"""

from __future__ import annotations

from typing import Iterable


def denial_names_allowed() -> bool:
    from src.features import features as _get_features

    return bool(_get_features().tool_denial_names_allowed)


# Sanctioned alternatives for always-blocked shell commands. A command absent here has none.
_BLOCKED_ALTERNATIVES = {
    "rm": "no delete tool exists; do not delete, overwrite with file_write_safe(path, content) if you must replace content",
    "mv": "no move/rename tool exists; write the new path with file_write_safe(path, content) instead",
    "cp": "no copy tool exists; read with read_file(path) then file_write_safe(new_path, content)",
}


def blocked_command_message(base_cmd: str, safe_commands: Iterable[str]) -> str:
    legacy = f"[ERROR: Command '{base_cmd}' is blocked for security]"
    if not denial_names_allowed():
        return legacy
    alt = _BLOCKED_ALTERNATIVES.get(base_cmd, "no sanctioned alternative exists for this command")
    return f"{legacy[:-1]}. Allowed commands: {sorted(safe_commands)}. For '{base_cmd}': {alt}]"


def shell_blocked_message(legacy: str) -> str:
    if not denial_names_allowed():
        return legacy
    return (
        f"{legacy}. Blocked: destructive/system patterns (rm -r /, mkfs, dd if=, writes to /dev, "
        "shutdown/reboot, systemctl stop/disable, curl|sh, inline python/perl/ruby -e, eval). "
        "Any other single-line command is allowed"
    )


def path_not_allowed_message(path: str, allowed_paths: Iterable[str]) -> str:
    legacy = f"Path not allowed: {path}"
    if not denial_names_allowed():
        return legacy
    return f"{legacy}. Allowed path prefixes: {list(allowed_paths)}"


def sandbox_hint_fallback(legacy: str, allowed_builtins: Iterable[str], allowed_modules: Iterable[str]) -> str:
    if not denial_names_allowed():
        return legacy
    return (
        " Hint: allowed imports: " + ", ".join(sorted(allowed_modules))
        + "; allowed builtins: " + ", ".join(sorted(allowed_builtins))
        + ". See the SANDBOX CONTRACT in the prompt for the allowed helpers."
    )
