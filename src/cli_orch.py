#!/usr/bin/env python3
"""Unified orchestration CLI.

Routes to various subcommand modules:
  orch sessions ...  -> Session management
  orch run ...       -> Run tasks
  orch stack ...     -> Server stack management
  orch status        -> Quick status

Usage:
    orch sessions list [--status STATUS]
    orch sessions search QUERY
    orch sessions resume SESSION_ID
    orch run "task description"
    orch status
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from scripts.server.stack_manifest import HOT_ROLES, PORT_MAP
from src.registry.stack_priors import (
    live_stack_role_records,
    stack_prior_endpoint_port,
    stack_prior_serving,
)

logger = logging.getLogger(__name__)

DEFAULT_STACK_PRIORS_PATH = Path(__file__).parent.parent / "orchestration" / "derived" / "stack_priors.yaml"
FALLBACK_STATUS_EXCLUDED_ROLES = frozenset({
    "embedder",
})


def _fallback_status_targets() -> list[tuple[str, int]]:
    hot_ports = {
        PORT_MAP[role]
        for role in HOT_ROLES
        if role not in FALLBACK_STATUS_EXCLUDED_ROLES
        and isinstance(PORT_MAP.get(role), int)
    }
    names_by_port: dict[int, list[str]] = {}
    for role, port in sorted(PORT_MAP.items()):
        if role in FALLBACK_STATUS_EXCLUDED_ROLES:
            continue
        if isinstance(port, int) and port in hot_ports:
            names_by_port.setdefault(port, []).append(role)
    return [
        ("/".join(sorted(names)), port)
        for port, names in sorted(names_by_port.items())
    ]


def _stack_status_targets(
    stack_priors_path: Path = DEFAULT_STACK_PRIORS_PATH,
) -> list[tuple[str, int]]:
    roles = live_stack_role_records(stack_priors_path)
    if not roles:
        return _fallback_status_targets()

    names_by_port: dict[int, list[str]] = {}
    for role, record in roles.items():
        serving = stack_prior_serving(record)
        try:
            port = stack_prior_endpoint_port(serving)
        except ValueError:
            continue
        if port is None:
            continue
        names_by_port.setdefault(port, []).append(role)

    if not names_by_port:
        return _fallback_status_targets()
    return [
        ("/".join(sorted(names)), port)
        for port, names in sorted(names_by_port.items())
    ]


def main() -> int:
    """Main entry point for orch CLI."""
    parser = argparse.ArgumentParser(
        prog="orch",
        description="Hierarchical orchestration CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Subcommands:
  sessions    Manage conversation sessions
  run         Execute orchestration tasks
  stack       Manage server stack
  status      Show system status

Examples:
  orch sessions list --status active
  orch sessions resume abc123
  orch run "Write a Python function"
  orch status
        """,
    )

    parser.add_argument(
        "subcommand",
        nargs="?",
        choices=["sessions", "run", "stack", "status", "help"],
        help="Subcommand to run",
    )

    parser.add_argument(
        "args",
        nargs="*",
        help="Arguments for subcommand",
    )

    # Parse just the first argument to determine subcommand
    args, remaining = parser.parse_known_args()

    if not args.subcommand or args.subcommand == "help":
        parser.print_help()
        return 0

    # Combine args.args and remaining (for different parsing behaviors)
    subcommand_args = (args.args or []) + remaining

    # Route to subcommand
    if args.subcommand == "sessions":
        from src.cli_sessions import main as sessions_main

        # Reconstruct sys.argv for the subcommand
        sys.argv = ["orch sessions"] + subcommand_args
        return sessions_main()

    elif args.subcommand == "run":
        from src.cli import main as cli_main

        sys.argv = ["orch run"] + subcommand_args
        return cli_main()

    elif args.subcommand == "stack":
        # Delegate to orchestrator_stack.py
        script_path = Path(__file__).parent.parent / "scripts" / "server" / "orchestrator_stack.py"
        if script_path.exists():
            cmd = [sys.executable, str(script_path)] + subcommand_args
            return subprocess.run(cmd).returncode
        else:
            print(f"Stack script not found: {script_path}", file=sys.stderr)
            return 1

    elif args.subcommand == "status":
        return cmd_status()

    return 1


def cmd_status() -> int:
    """Show quick system status."""
    import urllib.request
    import urllib.error

    print()
    print("=" * 50)
    print("ORCHESTRATOR STATUS")
    print("=" * 50)
    print()

    # Check orchestrator API
    try:
        with urllib.request.urlopen("http://localhost:8000/health", timeout=3) as resp:
            if resp.status == 200:
                print("  Orchestrator API:  \033[92m● Running\033[0m (port 8000)")
            else:
                print("  Orchestrator API:  \033[91m✗ Error\033[0m")
    except (urllib.error.URLError, TimeoutError):
        print("  Orchestrator API:  \033[90m○ Offline\033[0m")

    # Check llama-server ports from generated stack priors.
    for name, port in _stack_status_targets():
        try:
            with urllib.request.urlopen(f"http://localhost:{port}/health", timeout=2) as resp:
                if resp.status == 200:
                    print(f"  {name:<20} \033[92m● Running\033[0m (port {port})")
        except (urllib.error.URLError, TimeoutError):
            pass  # Don't show offline servers

    print()

    # Check session store
    try:
        from src.session import SQLiteSessionStore

        store = SQLiteSessionStore()
        sessions = store.list_sessions(limit=5)
        active = sum(1 for s in sessions if s.status.value == "active")
        total = len(sessions)

        print(f"  Sessions:          {active} active / {total} recent")
    except Exception as e:
        logger.debug("Session store error: %s", e)
        print(f"  Sessions:          \033[91m✗ Error\033[0m ({e})")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
