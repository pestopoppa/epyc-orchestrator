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
        """Reject custom metaclasses.

        A metaclass ``__new__`` receives the class namespace as an ordinary dict
        and can install a serialization hook under a COMPUTED key
        (``ns["__redu" + "ce__"] = fn``), which no static key check can see. The
        key computation is not itself the vulnerability — installing the dict as
        a class namespace is, and that has exactly two routes: ``type(n, b, d)``
        (rejected in visit_Call) and a metaclass. Closing both makes computed
        keys inert. Custom metaclasses have no legitimate use in analysis code.
        """
        for keyword in node.keywords:
            if keyword.arg == "metaclass":
                self.violations.append("class ... (metaclass=...)")
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        """Extend dunder-subscript checks to serialization hooks."""
        target = getattr(node, "slice", None)
        if (
            isinstance(target, ast.Constant)
            and isinstance(target.value, str)
            and target.value in self.FORBIDDEN_METHOD_DEFS
        ):
            self.violations.append(f"['{target.value}']")
        return self._visit_subscript_forbidden_attrs(node)

    def visit_Dict(self, node: ast.Dict) -> None:
        """Reject any dict literal keyed by a serialization-hook name.

        Closes the indirection path where the namespace is built first and only
        later used to create a class, so the hook name never appears next to a
        `type(...)` call: `d = {'__reduce__': fn}` … `type('C', (), d)`.
        """
        for key in node.keys:
            if (
                isinstance(key, ast.Constant)
                and isinstance(key.value, str)
                and key.value in self.FORBIDDEN_METHOD_DEFS
            ):
                self.violations.append(f"{{'{key.value}': ...}}")
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

        # Dynamic class creation is a general sandbox-escape primitive and can
        # bind a serialization hook with no FunctionDef and no Assign:
        #   type('C', (), {'__reduce__': fn})        <- literal namespace
        #   d = {...}; type('C', (), d)              <- namespace via variable
        # Matching only the literal form left the variable form open, so the
        # 3-argument form is rejected outright. The 1-argument form (`type(x)`,
        # ubiquitous in analysis code) is untouched.
        if isinstance(node.func, ast.Name) and node.func.id == "type" and len(node.args) == 3:
            self.violations.append("type(name, bases, dict) dynamic class creation")

        # dict(__reduce__=fn) builds a hook-bearing namespace with no ast.Dict
        # node at all, so visit_Dict never sees it.
        if isinstance(node.func, ast.Name) and node.func.id == "dict":
            for keyword in node.keywords:
                if keyword.arg in self.FORBIDDEN_METHOD_DEFS:
                    self.violations.append(f"dict({keyword.arg}=...)")

        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Check attribute access for forbidden dunder attributes."""
        if node.attr in self.FORBIDDEN_ATTRS:
            self.violations.append(f".{node.attr}")
        self.generic_visit(node)

    def _visit_subscript_forbidden_attrs(self, node: ast.Subscript) -> None:
        """Check subscript access for string-based dunder bypass attempts.

        Catches patterns like: obj['__class__'] or obj["__globals__"]
        """
        if isinstance(node.slice, ast.Constant):
            if isinstance(node.slice.value, str):
                if node.slice.value in self.FORBIDDEN_ATTRS:
                    self.violations.append(f"['{node.slice.value}']")
        self.generic_visit(node)
