"""seeding_types must import when generated stack priors are absent.

DEFAULT_ROLES used to be assigned before discover_active_roles() was defined, so the
fallback path raised NameError at import whenever stack_priors.yaml could not be read
(fresh trees, CI, a clone without generated files). Production never saw it because the
priors read short-circuits the `or`.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

MODULE = Path(__file__).resolve().parents[2] / "scripts" / "benchmark" / "seeding_types.py"


def test_imports_without_stack_priors(tmp_path: Path) -> None:
    # PROJECT_ROOT is derived from the file location; a bare copy has no orchestration/ tree,
    # so both the stack-prior read and the registry read come back empty.
    bench = tmp_path / "scripts" / "benchmark"
    bench.mkdir(parents=True)
    shutil.copy(MODULE, bench / "seeding_types.py")
    proc = subprocess.run(
        [sys.executable, "-c",
         "import seeding_types as s; print(s.DEFAULT_ROLES); print(sorted(s.ARCHITECT_ROLES))"],
        cwd=bench, capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert "NameError" not in proc.stderr
    assert proc.stdout.splitlines()[0] == "['frontdoor']"
    assert proc.stdout.splitlines()[1] == "['architect_general']"
