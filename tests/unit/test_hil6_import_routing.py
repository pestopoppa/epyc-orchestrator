"""HIL-6 seam S5: AutoPilot import sites of the extracted guards, routed and audited.

The S1/S2 guards live in ``src.autopilot_core.mutation_guards``; ``prompt_forge`` re-exports the
same objects, so no import context can see a second copy (no dual-module hazard). S5 routes the
two AutoPilot modules that import ONLY extracted names straight to the guard module (they no
longer import the PromptForge proposer) and pins the audit of every remaining site.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _run(code: str, cwd: Path = ROOT) -> str:
    out = subprocess.run([sys.executable, "-c", code], cwd=cwd, capture_output=True, text=True,
                         timeout=120)
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


def _extracted_names() -> set[str]:
    from scripts.autopilot.species import prompt_forge as pf
    from src.autopilot_core import mutation_guards as mg

    return {n for n in vars(mg) if not n.startswith("__") and getattr(pf, n, None) is getattr(mg, n)
            and n not in {"annotations", "log", "ast", "hashlib", "json", "logging", "math", "os",
                          "re", "time", "dataclass", "Enum", "Path", "Any", "Callable"}}


def test_calibration_script_imports_the_guard_from_its_own_checkout(tmp_path):
    # Run from an unrelated cwd: the venv .pth points at the shared clone, so the script must put
    # its OWN checkout first, and must not import the PromptForge proposer.
    script = ROOT / "scripts" / "autopilot" / "prompt_interference_calibration.py"
    out = _run(
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('cal', {str(script)!r})\n"
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
        "import src.autopilot_core.mutation_guards as mg\n"
        "print(m.prompt_interference_lint is mg.prompt_interference_lint,"
        " any('prompt_forge' in k for k in sys.modules), mg.__file__)",
        cwd=tmp_path,
    )
    same, imported_pf, mg_file = out.split(" ", 2)
    assert (same, imported_pf) == ("True", "False")
    assert Path(mg_file).resolve() == (ROOT / "src/autopilot_core/mutation_guards.py").resolve()


def test_heldout_corpus_vocabulary_is_the_guard_module_in_both_contexts():
    out = _run(
        "import sys; sys.path.insert(0, 'scripts/autopilot'); sys.path.insert(0, '.')\n"
        "import species.heldout_effect_corpus as bare\n"
        "import scripts.autopilot.species.heldout_effect_corpus as pkg\n"
        "import src.autopilot_core.mutation_guards as mg\n"
        "import species.prompt_forge as pf\n"
        "print(bare is not pkg, bare.MutationEffect is pkg.MutationEffect is mg.MutationEffect"
        " is pf.MutationEffect, bare.MUTATION_EFFECT_RISK is mg.MUTATION_EFFECT_RISK)"
    )
    assert out == "True True True"


def test_remaining_autopilot_imports_of_extracted_names_are_audited():
    extracted = _extracted_names()
    assert {"MutationEffect", "eval_leakage_reason", "CODE_MUTATION_ALLOWLIST"} <= extracted
    sites = []
    for path in sorted((ROOT / "scripts" / "autopilot").rglob("*.py")):
        if path.name == "prompt_forge.py":
            continue
        for node in ast.walk(ast.parse(path.read_text(errors="replace"))):
            if isinstance(node, ast.ImportFrom) and (node.module or "").endswith("prompt_forge"):
                for alias in node.names:
                    if alias.name in extracted:
                        sites.append((str(path.relative_to(ROOT)), alias.name))
    # autopilot.py needs prompt_forge for new_file_mutation_root_labels and the module object
    # anyway; the list it imports is the guard module's own object, so routing it would remove
    # no hazard. Any NEW site of an extracted name should import from mutation_guards.
    assert sites == [("scripts/autopilot/autopilot.py", "CODE_MUTATION_ALLOWLIST")]
