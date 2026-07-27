"""AST-based security validation for the REPL sandbox."""

from __future__ import annotations

import ast


class ASTSecurityVisitor(ast.NodeVisitor):
    """AST visitor that checks for dangerous code patterns.

    This is more robust than regex because it analyzes the parsed syntax tree,
    making it immune to string concatenation tricks like:
        getattr(__builtins__, '__im' + 'port__')('os')
    """

    # Forbidden module imports
    FORBIDDEN_MODULES = frozenset(
        {
            "os",
            "sys",
            "subprocess",
            "socket",
            "shutil",
            "pathlib",
            "tempfile",
            "multiprocessing",
            "threading",
            "ctypes",
            "pickle",
            "importlib",
            "builtins",
            "code",
            "codeop",
            "runpy",
            "pkgutil",
        }
    )

    # Forbidden built-in function calls
    FORBIDDEN_CALLS = frozenset(
        {
            "__import__",
            "eval",
            "exec",
            "compile",
            "open",
            "getattr",
            "setattr",
            "delattr",
            "hasattr",
            "globals",
            "locals",
            "vars",
            "dir",
            "input",
            "breakpoint",
            "memoryview",
        }
    )

    # Forbidden attribute accesses (dunder attributes for escaping sandbox)
    FORBIDDEN_ATTRS = frozenset(
        {
            "__class__",
            "__bases__",
            "__subclasses__",
            "__mro__",
            "__dict__",
            "__globals__",
            "__locals__",
            "__code__",
            "__builtins__",
            "__closure__",
            "__func__",
            "__self__",
            "__module__",
            "__qualname__",
            "__annotations__",
            "__reduce__",
            "__reduce_ex__",
            "__getstate__",
            "__setstate__",
        }
    )

    # Dunders that customize how an object is SERIALIZED. Defining one lets a
    # value smuggle a callable + args into a pickle stream, to be invoked at
    # unpickle time outside this sandbox. FORBIDDEN_ATTRS already covers reading
    # them (an ast.Attribute); this covers *defining* them, which is an
    # ast.FunctionDef inside a ClassDef and therefore a different node type.
    # Layer 4 of the checkpoint pickle boundary — see safe_pickle.py.
    FORBIDDEN_METHOD_DEFS = frozenset(
        {
            "__reduce__",
            "__reduce_ex__",
            "__getstate__",
            "__setstate__",
            "__getnewargs__",
            "__getnewargs_ex__",
        }
    )

    def __init__(self):
        self.violations: list[str] = []

    def _check_method_def(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        if node.name in self.FORBIDDEN_METHOD_DEFS:
            self.violations.append(f"def {node.name}(...)")

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Reject definitions of serialization-hook dunders."""
        self._check_method_def(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Reject async definitions of serialization-hook dunders."""
        self._check_method_def(node)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        """Reject `__reduce__ = fn`, which binds a hook without a FunctionDef."""
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in self.FORBIDDEN_METHOD_DEFS:
                self.violations.append(f"{target.id} = ...")
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Reject serialization hooks smuggled in via keyword bases or a dict.

        `type('C', (), {'__reduce__': fn})` is handled by visit_Call below.
        """
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        """Check regular imports: import os"""
        for alias in node.names:
            module = alias.name.split(".")[0]
            if module in self.FORBIDDEN_MODULES:
                self.violations.append(f"import {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Check from imports: from os import path"""
        if node.module:
            module = node.module.split(".")[0]
            if module in self.FORBIDDEN_MODULES:
                self.violations.append(f"from {node.module} import ...")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Check function calls for forbidden functions."""
        # Check direct calls: eval(...)
        if isinstance(node.func, ast.Name):
            if node.func.id in self.FORBIDDEN_CALLS:
                self.violations.append(f"{node.func.id}()")

        # Check attribute calls: obj.__class__()
        elif isinstance(node.func, ast.Attribute):
            if node.func.attr in self.FORBIDDEN_ATTRS:
                self.violations.append(f".{node.func.attr}()")

        # Dynamic class creation can bind a serialization hook without any
        # FunctionDef or Assign: type('C', (), {'__reduce__': fn}).
        if isinstance(node.func, ast.Name) and node.func.id == "type" and len(node.args) == 3:
            for arg in node.args:
                if not isinstance(arg, ast.Dict):
                    continue
                for key in arg.keys:
                    if (
                        isinstance(key, ast.Constant)
                        and isinstance(key.value, str)
                        and key.value in self.FORBIDDEN_METHOD_DEFS
                    ):
                        self.violations.append(f"type(..., {{'{key.value}': ...}})")

        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Check attribute access for forbidden dunder attributes."""
        if node.attr in self.FORBIDDEN_ATTRS:
            self.violations.append(f".{node.attr}")
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        """Check subscript access for string-based dunder bypass attempts.

        Catches patterns like: obj['__class__'] or obj["__globals__"]
        """
        if isinstance(node.slice, ast.Constant):
            if isinstance(node.slice.value, str):
                if node.slice.value in self.FORBIDDEN_ATTRS:
                    self.violations.append(f"['{node.slice.value}']")
        self.generic_visit(node)
