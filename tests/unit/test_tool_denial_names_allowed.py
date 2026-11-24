"""HF-DENIALS: tool denial strings name the allowed set when tool_denial_names_allowed is on."""

import pytest

from src.features import reset_features
from orchestration.tools import code as code_tools
from orchestration.tools import system as system_tools
from src import denial_hints
from src.repl_environment.external_access import _ExternalAccessMixin


@pytest.fixture
def flag(monkeypatch):
    def _set(on):
        monkeypatch.setenv("ORCHESTRATOR_TOOL_DENIAL_NAMES_ALLOWED", "1" if on else "0")
        reset_features()

    yield _set
    monkeypatch.undo()
    reset_features()


class _Shim:
    _audit = staticmethod(lambda *a, **k: None)
    _exploration_calls = 0


def _shell(cmd):
    return _ExternalAccessMixin._run_shell_inner(_Shim(), cmd, 5, {})


@pytest.mark.parametrize("cmd", ["rm", "mv", "cp", "sudo"])
def test_blocked_command_off_is_legacy_on_names_allowed(flag, cmd):
    flag(False)
    assert _shell(f"{cmd} x") == f"[ERROR: Command '{cmd}' is blocked for security]"
    flag(True)
    out = _shell(f"{cmd} x")
    assert out.startswith(f"[ERROR: Command '{cmd}' is blocked for security.")
    assert "Allowed commands: ['" in out and "'ls'" in out
    assert "file_write_safe" in out if cmd in ("rm", "mv", "cp") else "no sanctioned alternative" in out


def test_rm_mv_say_no_delete_move_tool(flag):
    flag(True)
    assert "no delete tool exists" in _shell("rm x")
    assert "no move/rename tool exists" in _shell("mv a b")


def test_run_shell_safety_block(flag):
    flag(False)
    assert code_tools.run_shell("mkfs /dev/x") == {"error": "Command blocked for safety"}
    flag(True)
    err = code_tools.run_shell("mkfs /dev/x")["error"]
    assert err.startswith("Command blocked for safety. Blocked:") and "Any other single-line command is allowed" in err


def test_path_not_allowed_all_three_sites(flag):
    flag(False)
    legacy = "Path not allowed: /etc/passwd"
    assert system_tools.read_file("/etc/passwd") == {"error": legacy}
    assert system_tools.write_file("/etc/x", "y") == {"error": legacy.replace("passwd", "x")}
    assert system_tools.list_directory("/etc") == [{"error": "Path not allowed: /etc"}]
    flag(True)
    for err in (
        system_tools.read_file("/etc/passwd")["error"],
        system_tools.write_file("/etc/x", "y")["error"],
        system_tools.list_directory("/etc")[0]["error"],
    ):
        assert err.startswith("Path not allowed: /etc")
        assert "Allowed path prefixes: ['/mnt/raid0/llm/', '/tmp/']" in err


def test_sandbox_hint_fallback(flag):
    flag(False)
    assert denial_hints.sandbox_hint_fallback("", {"len"}, {"math"}) == ""
    assert denial_hints.sandbox_hint_fallback(" Hint: x", {"len"}, {"math"}) == " Hint: x"
    flag(True)
    out = denial_hints.sandbox_hint_fallback("", {"len", "abs"}, {"math", "re"})
    assert "allowed imports: math, re" in out and "allowed builtins: abs, len" in out
    assert "SANDBOX CONTRACT" in out


def test_repl_security_error_fallback_hint(flag, monkeypatch):
    from src.repl_environment import REPLEnvironment
    from src.repl_environment.types import REPLSecurityError

    def _msg():
        env = REPLEnvironment(context="")
        with pytest.raises(REPLSecurityError) as ei:
            env._validate_code("x = globals()")
        return str(ei.value)

    flag(False)
    off = _msg()
    flag(True)
    on = _msg()
    assert on.startswith(off.rstrip("."))
    assert "allowed imports:" in on and "allowed imports:" not in off
