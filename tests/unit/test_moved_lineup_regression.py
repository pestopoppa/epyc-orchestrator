"""Run the compiler suite against both declared and relocated worker fleets."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

from tests.plugins.moved_lineup import PINNED_TEST

ROOT = Path(__file__).resolve().parents[2]


def test_compiler_suite_follows_moved_lineup_except_deliberate_pin(tmp_path, monkeypatch):
    # Ambient plugin/addopts injection must not change the compared test sets.
    monkeypatch.setenv("PYTEST_PLUGINS", "ssuf18_plugin_that_must_not_be_imported")
    monkeypatch.setenv("PYTEST_ADDOPTS", "--ssuf18-invalid-ambient-option")
    declarations = [ROOT / "orchestration" / name for name in (
        "model_registry.yaml", "launch_manifest.yaml", "stack_topology.yaml"
    )]
    before = {p: hashlib.sha256(p.read_bytes()).digest() for p in declarations}
    reports = {}
    for mode in ("baseline", "moved"):
        report = tmp_path / (mode + ".xml")
        command = [sys.executable, "-m", "pytest", "-q", "--runxfail",
                   "tests/unit/test_stack_priors_compiler.py", "--junitxml=" + str(report),
                   "-o", "cache_dir=" + str(tmp_path / mode / "pytest-cache")]
        if mode == "moved":
            command.extend(["-p", "tests.plugins.moved_lineup"])
        env = {key: value for key, value in os.environ.items()
               if key not in {"PYTEST_PLUGINS", "PYTHONPATH", "PYTHONPYCACHEPREFIX"}}
        env.update(PYTEST_ADDOPTS="", OMP_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
        completed = subprocess.run(command, cwd=ROOT, env=env, capture_output=True,
                                   text=True, timeout=60)
        assert completed.returncode == (0 if mode == "baseline" else 1), completed.stdout + completed.stderr
        document = ET.parse(report).getroot()
        suites = [document] if document.tag == "testsuite" else document.findall(".//testsuite")
        assert suites
        assert all(int(suite.attrib["errors"]) == 0 for suite in suites)
        for suite in suites:
            suite_cases = suite.findall("testcase")
            assert int(suite.attrib["tests"]) == len(suite_cases)
            for counter, child in (("failures", "failure"), ("errors", "error"), ("skipped", "skipped")):
                assert int(suite.attrib[counter]) == sum(case.find(child) is not None for case in suite_cases)
        cases = document.findall(".//testcase")
        assert cases
        assert not any(c.find("error") is not None or c.find("skipped") is not None for c in cases)
        reports[mode] = cases
    def identities(cases):
        return {(case.attrib.get("classname"), case.attrib["name"]) for case in cases}

    assert identities(reports["baseline"]) == identities(reports["moved"])
    assert len(identities(reports["baseline"])) == len(reports["baseline"])
    assert len(identities(reports["moved"])) == len(reports["moved"])
    assert not any(c.find("failure") is not None for c in reports["baseline"])
    failures = [c for c in reports["moved"] if c.find("failure") is not None]
    assert [c.attrib["name"] for c in failures] == [PINNED_TEST]
    assert 'assert regenerated.startswith("full:")' in (failures[0].find("failure").text or "")
    assert before == {p: hashlib.sha256(p.read_bytes()).digest() for p in declarations}
