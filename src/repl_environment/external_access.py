"""External access tools for the REPL environment.

Network access (web_fetch) and sandboxed shell execution (run_shell).
"""

from __future__ import annotations

import logging
import subprocess
import shlex

from src.config import _registry_timeout

logger = logging.getLogger(__name__)

# Default shell command timeout from registry
_SHELL_TIMEOUT = int(_registry_timeout("repl", "shell_command", 30))


def _get_project_root() -> str:
    """Get project root from config with fallback."""
    import os
    from pathlib import Path

    try:
        from src.config import get_config

        root = str(get_config().paths.project_root)
        if Path(root).exists():
            return root
    except Exception:
        pass

    # Fallback: cwd
    return os.getcwd()


class _ExternalAccessMixin:
    """Mixin providing network and shell access tools (_web_fetch, _run_shell).

    Required attributes (provided by REPLEnvironment.__init__):
        _exploration_calls: int — exploration call counter
        _exploration_log: ExplorationLog — exploration event history
    """

    @staticmethod
    def _audit(self, tool, text, cwd, started, aud, result) -> None:
        """SEC-EXEC-4: one durable record per call, refused/timed-out/errored included."""
        from src.repl_environment.exec_containment import audit_tool_call

        outcome = aud.get("outcome")
        refusal = aud.get("refusal")
        if outcome is None:
            if isinstance(result, str) and result.startswith("[ERROR"):
                outcome, refusal = "refused", result[:500]
            else:
                outcome = "ok"
        audit_tool_call(
            tool, text=text, cwd=aud.get("cwd", cwd), started=started, outcome=outcome,
            exit_code=aud.get("exit"), refusal=refusal,
            role=getattr(self, "role", None), request_id=getattr(self, "task_id", None),
        )

    def _web_fetch(self, url: str, max_chars: int = 10000) -> str:
        import time as _t

        started = _t.time()
        aud: dict = {}
        result = ""
        try:
            result = _ExternalAccessMixin._web_fetch_inner(self, url, max_chars, aud)
            return result
        finally:
            _ExternalAccessMixin._audit(self, "web_fetch", url, None, started, aud, result)

    def _web_fetch_inner(self, url: str, max_chars: int, aud: dict) -> str:
        """Fetch content from a URL.

        Args:
            url: URL to fetch (must be http or https).
            max_chars: Maximum characters to return (default 10000).

        Returns:
            Plain text content from the URL, truncated to max_chars.
        """
        self._exploration_calls += 1
        import requests

        # Validate URL
        if not url.startswith(("http://", "https://")):
            return "[ERROR: Only http/https URLs are allowed]"

        try:
            resp = requests.get(
                url,
                timeout=30,
                headers={"User-Agent": "Mozilla/5.0 (compatible; OrchestratorBot/1.0)"},
            )
            resp.raise_for_status()

            content_type = resp.headers.get("Content-Type", "")

            # Handle HTML - extract text
            if "text/html" in content_type:
                try:
                    from bs4 import BeautifulSoup

                    soup = BeautifulSoup(resp.text, "html.parser")
                    # Remove script and style elements
                    for elem in soup(["script", "style", "nav", "footer"]):
                        elem.decompose()
                    text = soup.get_text(separator="\n", strip=True)
                except ImportError:
                    # Fallback: basic tag stripping
                    import re

                    text = re.sub(r"<[^>]+>", "", resp.text)
            else:
                text = resp.text

            result = text[:max_chars]
            if len(text) > max_chars:
                result += f"\n[... truncated at {max_chars} chars, total: {len(text)}]"

            self._exploration_log.add_event("web_fetch", {"url": url}, result)
            return result

        except requests.exceptions.Timeout:
            aud["outcome"] = "timeout"
            return "[ERROR: Request timed out after 30s]"
        except requests.exceptions.RequestException as e:
            aud["outcome"] = "error"
            return f"[ERROR: Request failed: {e}]"
        except Exception as e:
            logger.debug("web_fetch failed", exc_info=True)
            aud["outcome"] = "error"
            return f"[ERROR: {type(e).__name__}: {e}]"

    def _run_shell(self, cmd: str, timeout: int = _SHELL_TIMEOUT) -> str:
        import time as _t

        started = _t.time()
        aud: dict = {}
        result = ""
        try:
            result = _ExternalAccessMixin._run_shell_inner(self, cmd, timeout, aud)
            return result
        finally:
            _ExternalAccessMixin._audit(self, "run_shell", cmd, None, started, aud, result)

    def _run_shell_inner(self, cmd: str, timeout: int, aud: dict) -> str:
        """Run a sandboxed shell command (read-only operations only).

        Args:
            cmd: Shell command to execute.
            timeout: Maximum execution time in seconds (from registry, max 120).

        Returns:
            Command output (stdout + stderr combined).
        """
        self._exploration_calls += 1

        # Parse command
        try:
            parts = shlex.split(cmd)
        except ValueError as e:
            return f"[ERROR: Invalid command syntax: {e}]"

        if not parts:
            return "[ERROR: Empty command]"

        # Allowlist of safe commands
        SAFE_COMMANDS = {
            "ls",
            "find",
            "wc",
            "du",
            "file",
            "head",
            "tail",
            "cat",
            "grep",
            "awk",
            "sed",
            "sort",
            "uniq",
            "tr",
            "cut",
            "git",
            "pwd",
            "whoami",
            "date",
            "echo",
            "printf",
            # python/python3 deliberately absent (SEC-EXEC-2): run_python_code is the
            # sanctioned Python path; a shell interpreter would bypass the exec refusals.
        }

        # Commands that are always blocked
        BLOCKED_COMMANDS = {
            "rm",
            "mv",
            "cp",
            "chmod",
            "chown",
            "chgrp",
            "dd",
            "mkfs",
            "mount",
            "umount",
            "kill",
            "pkill",
            "killall",
            "sudo",
            "su",
            "bash",
            "sh",
            "zsh",
            "csh",
            "wget",
            "curl",  # Blocked to prevent downloads
            "nc",
            "netcat",
            "ncat",  # Network tools
        }

        base_cmd = parts[0].split("/")[-1]  # Handle /usr/bin/ls -> ls

        if base_cmd in BLOCKED_COMMANDS:
            return f"[ERROR: Command '{base_cmd}' is blocked for security]"

        if base_cmd not in SAFE_COMMANDS:
            return f"[ERROR: Command '{base_cmd}' not in allowlist: {sorted(SAFE_COMMANDS)}]"

        # Additional git restrictions
        if base_cmd == "git":
            if len(parts) > 1:
                git_subcmd = parts[1]
                safe_git = {"status", "log", "diff", "branch", "show", "ls-files", "rev-parse"}
                if git_subcmd not in safe_git:
                    return f"[ERROR: git {git_subcmd} not allowed. Safe: {sorted(safe_git)}]"

        # Timeout cap
        timeout = min(timeout, 120)

        try:
            # BEP harness (Phase 1, #5): run model shell/tests from the scratch task-root when
            # ORCHESTRATOR_EDIT_ROOT is active; else project root (default-off parity).
            from src.repl_environment.task_root import get_task_root, task_root_active

            _shell_cwd = str(get_task_root()) if task_root_active() else _get_project_root()
            aud["cwd"] = _shell_cwd
            # AP-54 eval knowledge fence (no-op unless the request carried `eval_fence`).
            from src.repl_environment.knowledge_fence import check_shell

            fence_denial = check_shell(parts, str(_shell_cwd))
            if fence_denial is not None:
                return f"[ERROR: {fence_denial}]"
            # INF-78 OAB-1: task_root-scoped request — no shell writers, reads confined to
            # task_root + read_roots (no-op without a request scope).
            from src.repl_environment.task_root import check_shell_scope

            scope_denial = check_shell_scope(parts, str(_shell_cwd))
            if scope_denial is not None:
                return f"[ERROR: {scope_denial}]"
            # SEC-EXEC-2: interpreter / find-awk-sed exec forms refused unscoped in production.
            from src.repl_environment.task_root import check_shell_exec

            exec_denial = check_shell_exec(parts)
            if exec_denial is not None:
                return f"[ERROR: {exec_denial}]"
            # AP-54: an ARMED eval request runs the command under kernel
            # enforcement (Landlock) when available; otherwise, and in
            # production, the launch is unchanged.
            from src.repl_environment.knowledge_fence import shell_fence_command

            shell_argv, shell_env, shell_scratch = shell_fence_command(shlex.split(cmd))
            if shell_env is None:  # SEC-EXEC-1: unfenced children get the minimal env too
                from src.repl_environment.exec_containment import child_env

                shell_env = child_env(task_id=getattr(self, "task_id", None))
            try:
                result = subprocess.run(
                    shell_argv,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=_shell_cwd,  # task-root under A/B, else project root
                    env=shell_env,
                )
            finally:
                if shell_scratch is not None:
                    import shutil

                    shutil.rmtree(shell_scratch, ignore_errors=True)

            aud["exit"] = result.returncode
            output = result.stdout
            if result.stderr:
                output += "\n[STDERR]\n" + result.stderr

            # Cap output
            if len(output) > 8000:
                output = output[:8000] + "\n[... truncated at 8000 chars]"

            self._exploration_log.add_event("run_shell", {"cmd": cmd}, output)
            return output

        except subprocess.TimeoutExpired:
            aud["outcome"] = "timeout"
            return f"[ERROR: Command timed out after {timeout}s]"
        except Exception as e:
            aud["outcome"] = "error"
            logger.debug("run_shell failed", exc_info=True)
            return f"[ERROR: {type(e).__name__}: {e}]"

    def _run_python_code(
        self, code: str, stdin_data: str = "", timeout: int = 60,
    ) -> str:
        import time as _t

        started = _t.time()
        aud: dict = {}
        result = ""
        try:
            result = _ExternalAccessMixin._run_python_code_inner(self, code, stdin_data, timeout, aud)
            return result
        finally:
            _ExternalAccessMixin._audit(self, "run_python_code", code, None, started, aud, result)

    def _run_python_code_inner(
        self, code: str, stdin_data: str, timeout: int, aud: dict,
    ) -> str:
        """Run a Python code string as a subprocess.

        Convenience wrapper: writes code to a temp file, runs it with python3,
        and returns stdout/stderr.  Use this instead of exec() for testing
        solutions that need stdin/stdout.

        Args:
            code: Python source code to run.
            stdin_data: Optional input to feed via stdin.
            timeout: Max execution time in seconds (capped at 120).

        Returns:
            Combined stdout + stderr output.
        """
        # INF-78 OAB-1: unavailable in a task_root-scoped request (no-op otherwise).
        from src.repl_environment.task_root import scope_refusal as _scope_refusal

        _scope_denial = _scope_refusal('run_python_code')
        if _scope_denial is not None:
            return f"[ERROR: {_scope_denial}]"
        from src.repl_environment.unicode_sanitizer import sanitize_code_unicode

        code = sanitize_code_unicode(code)
        self._exploration_calls += 1
        # AP-54 eval knowledge fence (no-op unless armed on this request).
        from src.repl_environment.knowledge_fence import check_python_source

        fence_denial = check_python_source(code)
        if fence_denial is not None:
            return f"[ERROR: {fence_denial}]"
        timeout = min(timeout, 120)

        import tempfile
        import os

        try:
            from src.config import get_config as _gc
            tmp_dir = str(_gc().paths.tmp_dir)
        except Exception:
            tmp_dir = "/mnt/raid0/llm/tmp"
        os.makedirs(tmp_dir, exist_ok=True)

        fence_launch = None
        # AP-54: a fenced request runs in a private per-call directory (its only
        # fully readable scratch space under kernel enforcement). None in production.
        from src.repl_environment.knowledge_fence import python_fence_run_dir

        fence_run_dir = python_fence_run_dir(tmp_dir)
        script_dir = fence_run_dir or tmp_dir
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", dir=script_dir, delete=False,
            ) as f:
                f.write(code)
                script_path = f.name

            # AP-54: under an eval fence (armed or control arm) the child runs
            # behind the audit-hook bootstrap; production launches exactly as before.
            from src.repl_environment.knowledge_fence import python_fence_launch

            from src.repl_environment.exec_containment import child_env

            aud["cwd"] = tmp_dir if fence_run_dir is None else script_dir
            fence_launch = python_fence_launch(script_path, script_dir)
            if fence_launch is None:
                result = subprocess.run(
                    ["python3", script_path],
                    input=stdin_data or None,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=tmp_dir,
                    env=child_env(task_id=getattr(self, "task_id", None)),  # SEC-EXEC-1: no inherited API secrets
                )
            else:
                result = subprocess.run(
                    fence_launch.argv,
                    input=stdin_data or None,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=script_dir,
                    env=fence_launch.env,
                )

            aud["exit"] = result.returncode
            output = result.stdout
            if result.stderr:
                output += "\n[STDERR]\n" + result.stderr

            if len(output) > 8000:
                output = output[:8000] + "\n[... truncated at 8000 chars]"

            self._exploration_log.add_event(
                "run_python_code",
                {"code_len": len(code), "has_stdin": bool(stdin_data)},
                output,
            )
            return output

        except subprocess.TimeoutExpired:
            aud["outcome"] = "timeout"
            return f"[ERROR: Python code timed out after {timeout}s]"
        except Exception as e:
            aud["outcome"] = "error"
            logger.debug("run_python_code failed", exc_info=True)
            return f"[ERROR: {type(e).__name__}: {e}]"
        finally:
            try:
                os.unlink(script_path)
            except Exception:
                pass
            if fence_launch is not None:
                from src.repl_environment.knowledge_fence import fold_python_touched

                fold_python_touched(fence_launch.touched_file)
            if fence_run_dir is not None:
                import shutil

                shutil.rmtree(fence_run_dir, ignore_errors=True)
