"""Exercise CI baseline and changed-file gate comparators on disposable inputs."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
CI_SCRIPTS = REPO_ROOT / "scripts" / "ci"


def _run_script(name: str, *args: str, cwd: Path = REPO_ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CI_SCRIPTS / name), *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def _write_pytest_case(path: Path, failures: list[tuple[str, str]]) -> None:
    cases = []
    for classname, name in failures:
        cases.append(
            f'<testcase classname="{classname}" name="{name}"><failure message="failed"/></testcase>'
        )
    passing = '<testcase classname="tests.unit.test_ok" name="test_ok" />'
    failure_count = len(failures)
    path.write_text(
        f'<testsuite tests="{failure_count + 1}" failures="{failure_count}" errors="0" skipped="0">'
        + passing
        + "".join(cases)
        + "</testsuite>",
        encoding="utf-8",
    )


def _pytest_baseline(path: Path, *, floor: int = 2) -> Path:
    baseline = path / "pytest-baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "source_head": "a" * 40,
                "minimum_collected": floor,
                "known_failure_ids": ["tests/unit/test_known.py::test_known"],
                "ci_exclusions": [],
            }
        ),
        encoding="utf-8",
    )
    return baseline


def test_pytest_baseline_accepts_known_ids_and_rejects_new_ids(tmp_path: Path) -> None:
    junit = tmp_path / "results.xml"
    baseline = _pytest_baseline(tmp_path)

    _write_pytest_case(junit, [("tests.unit.test_known", "test_known")])
    known = _run_script(
        "check_pytest_baseline.py",
        "--junit",
        str(junit),
        "--baseline",
        str(baseline),
        "--pytest-exit-code",
        "1",
    )
    assert known.returncode == 0
    assert "Known baseline failing/erroring IDs observed: 1 / 1" in known.stdout

    _write_pytest_case(junit, [("tests.unit.test_new", "test_new")])
    new = _run_script(
        "check_pytest_baseline.py",
        "--junit",
        str(junit),
        "--baseline",
        str(baseline),
        "--pytest-exit-code",
        "1",
    )
    assert new.returncode == 1
    assert "New failures:" in new.stdout


def test_pytest_baseline_enforces_collection_floor(tmp_path: Path) -> None:
    junit = tmp_path / "results.xml"
    baseline = _pytest_baseline(tmp_path, floor=3)
    _write_pytest_case(junit, [])

    result = _run_script(
        "check_pytest_baseline.py",
        "--junit",
        str(junit),
        "--baseline",
        str(baseline),
        "--pytest-exit-code",
        "0",
    )

    assert result.returncode == 1
    assert "Collection floor failed: 1 < 3." in result.stdout


def test_pytest_baseline_preserves_nested_class_and_parameter_identity(tmp_path: Path) -> None:
    junit = tmp_path / "results.xml"
    baseline = tmp_path / "pytest-baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "source_head": "c" * 40,
                "minimum_collected": 1,
                "known_failure_ids": [
                    "tests/unit/test_nested.py::TestOuter::TestInner::test_value[scope::value]"
                ],
                "ci_exclusions": [],
            }
        ),
        encoding="utf-8",
    )
    _write_pytest_case(
        junit,
        [("tests.unit.test_nested.TestOuter.TestInner", "test_value[scope::value]")],
    )

    result = _run_script(
        "check_pytest_baseline.py",
        "--junit",
        str(junit),
        "--baseline",
        str(baseline),
        "--pytest-exit-code",
        "1",
    )

    assert result.returncode == 0
    assert "Known baseline failing/erroring IDs observed: 1 / 1" in result.stdout


def test_ruff_baseline_accepts_known_diagnostics_and_rejects_new_ones(tmp_path: Path) -> None:
    diagnostic = {
        "filename": str(REPO_ROOT / "src" / "known.py"),
        "code": "F401",
        "message": "unused",
    }
    ruff_json = tmp_path / "ruff.json"
    baseline = tmp_path / "ruff-baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "source_head": "b" * 40,
                "known_diagnostics": [
                    {"path": "src/known.py", "code": "F401", "message": "unused"}
                ],
            }
        ),
        encoding="utf-8",
    )

    ruff_json.write_text(json.dumps([diagnostic]), encoding="utf-8")
    known = _run_script(
        "check_ruff_baseline.py",
        "--ruff-json",
        str(ruff_json),
        "--baseline",
        str(baseline),
        "--ruff-exit-code",
        "1",
    )
    assert known.returncode == 0
    assert "New diagnostics: 0" in known.stdout

    ruff_json.write_text(
        json.dumps(
            [
                diagnostic,
                {**diagnostic, "path": "src/new.py", "filename": str(REPO_ROOT / "src" / "new.py")},
            ]
        ),
        encoding="utf-8",
    )
    new = _run_script(
        "check_ruff_baseline.py",
        "--ruff-json",
        str(ruff_json),
        "--baseline",
        str(baseline),
        "--ruff-exit-code",
        "1",
    )
    assert new.returncode == 1
    assert "NEW x1: src/new.py: F401 unused" in new.stdout


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def _two_commit_repo(path: Path, before: str, after: str) -> str:
    path.mkdir()
    _git(path, "init", "-q")
    _git(path, "config", "user.email", "ci-test@example.invalid")
    _git(path, "config", "user.name", "CI test")
    source = path / "changed.py"
    source.write_text(before, encoding="utf-8")
    _git(path, "add", "changed.py")
    _git(path, "commit", "-qm", "base")
    base = _git(path, "rev-parse", "HEAD")
    source.write_text(after, encoding="utf-8")
    _git(path, "add", "changed.py")
    _git(path, "commit", "-qm", "candidate")
    return base


def test_changed_format_gate_reports_introduced_and_inherited_debt(tmp_path: Path) -> None:
    clean_base = _two_commit_repo(
        tmp_path / "introduced",
        "def add(a, b):\n    return a + b\n",
        "def add( a,b):return a+b\n",
    )
    introduced = _run_script(
        "check_changed_python_format.py",
        "--base",
        clean_base,
        cwd=tmp_path / "introduced",
    )
    assert introduced.returncode == 1
    assert "introduced or unclassified: changed.py" in introduced.stdout

    dirty_base = _two_commit_repo(
        tmp_path / "inherited",
        "def add( a,b):return a+b\n",
        "def add( a,b):return a+b\n# still inherited\n",
    )
    inherited = _run_script(
        "check_changed_python_format.py",
        "--base",
        dirty_base,
        cwd=tmp_path / "inherited",
    )
    assert inherited.returncode == 1
    assert "inherited baseline debt: changed.py" in inherited.stdout
