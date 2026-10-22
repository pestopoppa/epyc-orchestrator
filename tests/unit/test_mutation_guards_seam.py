"""HIL-6 seam S1: the MHS-1/2/4/15 guards moved to ``src.autopilot_core.mutation_guards``.

AutoPilot identity is proven by the existing PromptForge guard tests (unchanged) plus these:
every moved name is the SAME object through PromptForge in both import contexts AutoPilot uses,
and the guard module imports neither PromptForge nor the ``species`` package.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

MOVED = (
    "MutationEffect", "_EFFECT_TOKEN_LIMIT", "_EFFECT_REASON_LIMIT", "_EFFECT_ALIASES",
    "_truncate_reason", "SAFETY_BANNED_CALLS", "SAFETY_RESTRICTED_CALLS", "SAFETY_BANNED_MODULES",
    "SAFETY_IMPORT_ALLOWLIST", "SAFETY_FIRST_PARTY_ROOTS", "SAFETY_DUNDER_ATTR_ALLOWLIST",
    "SAFETY_TOPLEVEL_ALLOWED", "SAFETY_STRICT_NODE_DENYLIST", "StaticSafetyReport", "_dotted_name",
    "_collect_dotted_calls", "_collect_imported_roots", "_top_level_definition_names",
    "screen_static_safety", "classify_mutation_effect", "MUTATION_EFFECT_RISK",
    "MUTATION_RISK_GATE_ENV", "DEFAULT_MUTATION_RISK_GATE", "mutation_effect_risk",
    "mutation_risk_gate", "mutation_risk_gate_reason", "rank_mutations_by_risk",
    "classify_prompt_effect", "INTERFERENCE_WARNING_PREFIX", "INTERFERENCE_EXAMPLE_ADD_THRESHOLD",
    "_INTERFERENCE_MIN_DUP_CHARS", "_INTERFERENCE_POS_RE", "_INTERFERENCE_NEG_RE",
    "_INTERFERENCE_DIRECTIVE_SPLIT_RE", "_INTERFERENCE_WORD_RE", "_INTERFERENCE_STOPWORDS",
    "_INTERFERENCE_PRIORITY_RE", "_INTERFERENCE_EXAMPLE_RE", "_interference_lines",
    "_interference_dup_keys", "_interference_scope_nouns", "_interference_conflicts",
    "_interference_priority_lines", "_interference_example_count", "prompt_interference_lint",
)


def _run(code: str) -> str:
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
                         timeout=120)
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


def test_package_context_reexports_are_the_same_objects():
    from scripts.autopilot.species import prompt_forge as pf
    from src.autopilot_core import mutation_guards as mg

    missing = [n for n in MOVED if getattr(pf, n) is not getattr(mg, n)]
    assert missing == []


def test_bare_autopilot_context_reexports_are_the_same_objects():
    # autopilot.py puts scripts/autopilot then the repo root on sys.path and imports
    # ``species.prompt_forge`` as a bare module.
    names = ",".join(repr(n) for n in MOVED)
    out = _run(
        "import sys; sys.path.insert(0, 'scripts/autopilot'); sys.path.insert(0, '.');"
        "import species.prompt_forge as pf; import src.autopilot_core.mutation_guards as mg;"
        f"print(sum(getattr(pf, n) is not getattr(mg, n) for n in ({names},)))"
    )
    assert out == "0"


def test_guard_module_is_import_safe():
    out = _run(
        "import sys; sys.path.insert(0, '.'); import src.autopilot_core.mutation_guards;"
        "print(sorted(m for m in sys.modules if 'prompt_forge' in m or m.split('.')[0] == 'species'"
        " or m.startswith('scripts.autopilot.species')), 'subprocess' in sys.modules)"
    )
    assert out == "[] False"


def test_guard_module_imports_only_the_standard_library():
    import ast

    tree = ast.parse((ROOT / "src/autopilot_core/mutation_guards.py").read_text())
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    assert roots <= {"__future__", "ast", "logging", "math", "os", "re", "dataclasses", "enum", "typing"}


def test_logger_name_is_unchanged():
    from src.autopilot_core import mutation_guards as mg

    assert mg.log.name == "autopilot.prompt_forge"
