"""AP-54 fence path rules: standalone, stdlib-only, import-free.

This file is the single source of the path classification rules. Two places use it:
``knowledge_fence`` imports it, and the ``run_python_code`` audit-hook bootstrap
embeds its SOURCE TEXT verbatim into the child interpreter. That second use is why
it must not import anything outside the stdlib and must not use relative imports.
"""

import os

KNOWLEDGE_SUBDIRS = ("wiki", "handoffs", "research", "progress")
KNOWLEDGE_NESTED = (("docs", "chapters"),)
KNOWLEDGE_CHECKOUT_PREFIXES = ("epyc-root", "root-archetype")
GOLD_PAIRS = (("benchmarks", "prompts"), ("benchmarks", "results"), ("data", "kb_rag"))
GOLD_BASENAMES = ("sentinel_questions.yaml", "tool_sentinels.yaml")
PHYSREASON_DIR = "PhysReason_full"
PYTHON_EXECUTABLE_PREFIXES = ("python", "pypy")
RECURSIVE_COMMANDS = ("find", "du", "tree", "rg", "fd")


def is_under(path, root):
    root = root.rstrip("/") or "/"
    if root == "/":
        return True
    return path == root or path.startswith(root + "/")


def component_reason(path):
    """Classify a normalized absolute path by its components alone."""
    parts = [p for p in path.split("/") if p]
    if path.startswith("/workspace/") and len(parts) >= 2:
        if parts[1] in KNOWLEDGE_SUBDIRS or tuple(parts[1:3]) in KNOWLEDGE_NESTED:
            return "knowledge_root"
    for i, part in enumerate(parts[:-1]):
        if part.startswith(KNOWLEDGE_CHECKOUT_PREFIXES):
            if parts[i + 1] in KNOWLEDGE_SUBDIRS or tuple(parts[i + 1 : i + 3]) in KNOWLEDGE_NESTED:
                return "knowledge_root"
        if (part, parts[i + 1]) in GOLD_PAIRS:
            return "knowledge_root" if part == "data" else "eval_gold"
        if part == PHYSREASON_DIR:
            # <PhysReason_full>/<problem>/images/... is the served image; the rest
            # (problem.json) carries the solution.
            if not (len(parts) > i + 2 and parts[i + 2] == "images"):
                return "eval_gold"
    if parts and parts[-1] in GOLD_BASENAMES:
        return "eval_gold"
    return None


def safe_realpath(path):
    """``os.path.realpath`` that never raises; ``None`` when it cannot resolve."""
    try:
        return os.path.realpath(path).rstrip("/") or "/"
    except (OSError, ValueError):
        return None


def candidates(path, cwd):
    """(lexical, real-or-None) spellings of ``path`` resolved against ``cwd``."""
    raw = os.path.expanduser(path)
    if not os.path.isabs(raw):
        raw = os.path.join(cwd, raw)
    lexical = os.path.normpath(raw)
    return lexical, safe_realpath(raw)


def classify(path, cwd, fenced_dirs, explicit_roots, tree_only=(), tree=False):
    """Why ``path`` is fenced, or ``None``.

    ``fenced_dirs`` are the concrete fenced directories (the knowledge folders of
    every discovered checkout, and every gold directory), and ``explicit_roots``
    are extra fenced files and roots. ``tree_only`` are directories whose
    component rule is finer than a prefix (PhysReason keeps its images readable);
    they count only as walk roots. With ``tree=True``, a directory that CONTAINS
    fenced data is refused too (a recursive walk root). A path that cannot be
    resolved is reported as ``unresolvable_path``.
    """
    lexical, real = candidates(path, cwd)
    spellings = [lexical] if real in (None, lexical) else [lexical, real]
    for cand in spellings:
        reason = component_reason(cand)
        if reason:
            return reason
        for root in explicit_roots:
            if is_under(cand, root):
                return "knowledge_root"
        for root in fenced_dirs:
            if is_under(cand, root):
                return "eval_gold" if component_reason(root + "/x") == "eval_gold" else "knowledge_root"
    if tree:
        for cand in spellings:
            if cand == "/":
                return "walk_contains_fenced_root"
            for root in list(fenced_dirs) + list(explicit_roots) + list(tree_only):
                if is_under(root, cand):
                    return "walk_contains_fenced_root"
    if real is None:
        return "unresolvable_path"
    return None
