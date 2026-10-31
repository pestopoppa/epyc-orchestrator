"""Every stack CLI option needs a reachable reader or an explicit refusal.

The static check covers new argparse options without maintaining a hand-written
list.  The dispatch checks run main() with every lifecycle handler trapped, so
no test can accidentally launch a server.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
STACK = ROOT / "scripts/server/orchestrator_stack.py"
COMMANDS = ROOT / "scripts/server/stack_commands.py"
PREWARM = ROOT / "scripts/server/stack_prewarm.py"


def _functions(path: Path) -> dict[str, ast.FunctionDef]:
    return {
        node.name: node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _reads(nodes: list[ast.AST]) -> set[str]:
    found: set[str] = set()
    for root in nodes:
        for node in ast.walk(root):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                if node.value.id == "args" and isinstance(node.ctx, ast.Load):
                    found.add(node.attr)
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            if node.func.id != "getattr" or len(node.args) < 2:
                continue
            obj, key = node.args[:2]
            if isinstance(obj, ast.Name) and obj.id == "args" and isinstance(key, ast.Constant):
                if isinstance(key.value, str):
                    found.add(key.value)
    return found


def _reachable(functions: dict[str, ast.FunctionDef], first: str) -> list[ast.AST]:
    seen: set[str] = set()
    pending = [first]
    while pending:
        name = pending.pop()
        if name in seen or name not in functions:
            continue
        seen.add(name)
        pending.extend(
            node.func.id
            for node in ast.walk(functions[name])
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        )
    return [functions[name] for name in seen]


def _declared_options(main: ast.FunctionDef) -> dict[str, set[str]]:
    # Discover parser objects from add_parser, including newly added commands.
    owners: dict[str, set[str]] = {}
    options: dict[str, set[str]] = {}
    for node in ast.walk(main):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_parser" or not node.args:
            continue
        if not isinstance(node.args[0], ast.Constant) or not isinstance(node.args[0].value, str):
            continue
        name = node.args[0].value
        options[name] = set()
        for statement in ast.walk(main):
            if not isinstance(statement, ast.Assign) or statement.value is not node:
                continue
            for target in statement.targets:
                if isinstance(target, ast.Name):
                    owners[target.id] = {name}
    # The shared lifecycle option is added through a loop variable. Resolve
    # its tuple of parser objects instead of hard-coding those subcommands.
    for node in ast.walk(main):
        if not isinstance(node, ast.For) or not isinstance(node.target, ast.Name):
            continue
        if not isinstance(node.iter, (ast.Tuple, ast.List)):
            continue
        commands = set().union(
            *(owners.get(item.id, set()) for item in node.iter.elts if isinstance(item, ast.Name))
        )
        if commands:
            owners[node.target.id] = commands
    for node in ast.walk(main):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument" or not isinstance(node.func.value, ast.Name):
            continue
        parser_name = node.func.value.id
        assert parser_name in owners, f"untracked parser option: {ast.unparse(node)}"
        for command in owners[parser_name]:
            dest = next(
                (
                    kw.value.value
                    for kw in node.keywords
                    if kw.arg == "dest" and isinstance(kw.value, ast.Constant)
                ),
                None,
            )
            if dest is None and node.args and isinstance(node.args[0], ast.Constant):
                dest = str(node.args[0].value).lstrip("-").replace("-", "_")
            assert dest, ast.unparse(node)
            options[command].add(dest)
    return options


def _command_condition(node: ast.AST) -> set[str] | None:
    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
        parts = [_command_condition(value) for value in node.values]
        known = [part for part in parts if part is not None]
        return set.intersection(*known) if known else None
    if not isinstance(node, ast.Compare) or len(node.ops) != 1:
        return None
    if not isinstance(node.left, ast.Attribute) or node.left.attr != "command":
        return None
    if not isinstance(node.left.value, ast.Name) or node.left.value.id != "args":
        return None
    value = node.comparators[0]
    if isinstance(node.ops[0], ast.Eq) and isinstance(value, ast.Constant):
        return {value.value} if isinstance(value.value, str) else None
    if isinstance(node.ops[0], ast.In) and isinstance(value, (ast.Tuple, ast.List)):
        values = [item.value for item in value.elts if isinstance(item, ast.Constant)]
        return set(values) if all(isinstance(item, str) for item in values) else None
    return None


def _main_reads_for(nodes: list[ast.AST], command: str) -> set[str]:
    class Reads(ast.NodeVisitor):
        def __init__(self) -> None:
            self.found: set[str] = set()

        def visit_If(self, node: ast.If) -> None:
            eligible = _command_condition(node.test)
            if eligible is None or command in eligible:
                self.visit(node.test)
                for item in node.body:
                    self.visit(item)
            for item in node.orelse:
                self.visit(item)

        def visit_Attribute(self, node: ast.Attribute) -> None:
            if isinstance(node.value, ast.Name) and node.value.id == "args":
                if isinstance(node.ctx, ast.Load):
                    self.found.add(node.attr)
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> None:
            if isinstance(node.func, ast.Name) and node.func.id == "getattr":
                if len(node.args) >= 2:
                    obj, key = node.args[:2]
                    if isinstance(obj, ast.Name) and obj.id == "args":
                        if isinstance(key, ast.Constant) and isinstance(key.value, str):
                            self.found.add(key.value)
            self.generic_visit(node)

    visitor = Reads()
    for node in nodes:
        visitor.visit(node)
    return visitor.found


def _missing_options(main: ast.FunctionDef) -> dict[str, set[str]]:
    parse_index = next(
        index
        for index, node in enumerate(main.body)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "args" for target in node.targets)
    )
    stack_functions = _functions(STACK)
    command_functions = _functions(COMMANDS)
    prewarm_functions = _functions(PREWARM)
    declared = _declared_options(main)
    missing_by_command = {}
    for command, dests in declared.items():
        path = []
        if command == "start":
            path.extend(_reachable(stack_functions, "_cmd_validate_only"))
            path.extend(_reachable(prewarm_functions, "prewarm_all"))
        path.extend(_reachable(command_functions, f"cmd_{command}"))
        missing_by_command[command] = (
            dests - _main_reads_for(main.body[parse_index + 1 :], command) - _reads(path)
        )
    return missing_by_command


def test_all_declared_options_have_a_reader_on_their_dispatch_path():
    missing = _missing_options(_functions(STACK)["main"])
    assert set(missing) == {"start", "stop", "reload", "status"}
    assert not any(missing.values()), missing


@pytest.mark.parametrize(
    ("declaration", "command", "option"),
    [
        ('status_parser = subparsers.add_parser("status")\n'
         '    status_parser.add_argument("--unused-status")', "status", "unused_status"),
        ('subparsers.add_parser("status")\n'
         '    extra_parser = subparsers.add_parser("extra")\n'
         '    extra_parser.add_argument("--unused-extra")', "extra", "unused_extra"),
        ('subparsers.add_parser("status")\n'
         '    stop_parser.add_argument("--dev", action="store_true")', "stop", "dev"),
        ('subparsers.add_parser("status")\n'
         '    start_parser.add_argument("--unused-start")', "start", "unused_start"),
    ],
)
def test_structural_guard_detects_inert_mutations(declaration, command, option):
    source = STACK.read_text()
    mutated = source.replace('subparsers.add_parser("status", help="Show status")', declaration, 1)
    assert mutated != source
    tree = ast.parse(mutated)
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    assert option in _missing_options(main)[command]


@pytest.fixture
def stack_module(monkeypatch):
    if str(ROOT) not in sys.path:
        monkeypatch.syspath_prepend(str(ROOT))
    spec = importlib.util.spec_from_file_location("_stack_option_consumption", STACK)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "guard_against_running_bench", lambda *_a, **_kw: True)
    return module


@pytest.mark.parametrize(
    ("argv", "handler"),
    [
        (["start", "--hot-only"], "cmd_start"),
        (["stop", "--all"], "cmd_stop"),
        (["reload", "orchestrator"], "cmd_reload"),
        (["status"], "cmd_status"),
    ],
)
def test_each_subcommand_reaches_only_its_trapped_handler(
    stack_module, monkeypatch, argv, handler
):
    from scripts.server import stack_commands

    reached = []
    for name in ("cmd_start", "cmd_stop", "cmd_reload", "cmd_status"):
        monkeypatch.setattr(
            stack_commands,
            name,
            lambda _args, name=name: reached.append(name) or 0,
        )
    monkeypatch.setattr(sys, "argv", ["orchestrator_stack.py", *argv])
    assert stack_module.main() == 0
    assert reached == [handler]


@pytest.mark.parametrize(
    "other", [["--include-warm", "worker_fast"], ["--only", "worker_fast"]]
)
def test_hot_only_conflicts_refuse_before_any_launch(
    stack_module, monkeypatch, other, capsys
):
    from scripts.server import stack_commands

    monkeypatch.setattr(
        stack_commands,
        "cmd_start",
        lambda _args: pytest.fail("conflicting HOT-only request reached launcher"),
    )
    monkeypatch.setattr(
        sys, "argv", ["orchestrator_stack.py", "start", "--hot-only", *other]
    )
    assert stack_module.main() == 2
    assert "[REFUSED] --hot-only" in capsys.readouterr().out
