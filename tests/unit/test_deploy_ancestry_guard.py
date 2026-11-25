"""HF-DEPLOY-ANCESTRY: reload ancestry guard + revert post-apply-hash guard (default off)."""

from __future__ import annotations

import argparse
import subprocess

import pytest

from scripts.autopilot.species import prompt_forge as pf
from scripts.server import stack_commands as sc
from src.features import reset_features

ENV = "ORCHESTRATOR_DEPLOY_ANCESTRY_GUARD"


@pytest.fixture(autouse=True)
def _flag(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    reset_features()
    yield
    reset_features()


def _on(monkeypatch):
    monkeypatch.setenv(ENV, "1")
    reset_features()


def _git(repo, *a):
    return subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def repo(tmp_path, monkeypatch):
    r = tmp_path / "repo"
    r.mkdir()
    _git(r, "init", "-q")
    _git(r, "config", "user.email", "t@t")
    _git(r, "config", "user.name", "t")
    shas = []
    for i in range(2):
        (r / "f").write_text(str(i))
        _git(r, "add", "f")
        _git(r, "commit", "-q", "-m", f"c{i}")
        shas.append(_git(r, "rev-parse", "HEAD"))
    monkeypatch.setitem(sc._PATHS, "project_root", r)
    return r, shas


def test_ancestry_refusal_cases(repo):
    r, (old, new) = repo
    assert sc._ancestry_refusal(new, old) is None  # descendant passes
    assert sc._ancestry_refusal(new, new) is None  # same ref passes
    assert "not a descendant" in sc._ancestry_refusal(old, new)  # older refused
    assert sc._ancestry_refusal(new, None) is None  # no history never refuses


def _run_reload(monkeypatch, repo, last, allow=False):
    r, _ = repo
    recorded = []
    monkeypatch.setattr(sc, "_last_deployed_ref", lambda: last)
    monkeypatch.setattr(sc, "_record_deployed_ref", recorded.append)
    monkeypatch.setattr(sc, "_capacity_launch_gate", lambda: True)

    class Stop(Exception):
        pass

    def boom(*a, **k):
        raise Stop

    monkeypatch.setattr(sc, "load_state", boom)
    args = argparse.Namespace(components=["x"], allow_rollback=allow)
    try:
        return sc.cmd_reload(args), "ran"
    except Stop:
        return None, "ran"


def test_reload_descendant_passes(monkeypatch, repo):
    _on(monkeypatch)
    assert _run_reload(monkeypatch, repo, repo[1][0]) == (None, "ran")


def test_reload_older_ref_refused(monkeypatch, repo, capsys):
    _on(monkeypatch)
    # HEAD is `new`; last deployed is a ref HEAD does not descend from
    r, (old, new) = repo
    _git(r, "checkout", "-q", old)
    assert _run_reload(monkeypatch, repo, new)[0] == 2
    assert "[REFUSED]" in capsys.readouterr().out


def test_reload_allow_rollback_passes(monkeypatch, repo):
    _on(monkeypatch)
    r, (old, new) = repo
    _git(r, "checkout", "-q", old)
    assert _run_reload(monkeypatch, repo, new, allow=True) == (None, "ran")


def test_reload_flag_off_unchanged(monkeypatch, repo):
    r, (old, new) = repo
    _git(r, "checkout", "-q", old)
    assert _run_reload(monkeypatch, repo, new) == (None, "ran")


def _mutation(tmp_path, monkeypatch, hash_):
    monkeypatch.setattr(pf, "PROJECT_ROOT", tmp_path)
    (tmp_path / "m.py").write_text("moved\n")
    return pf.CodeMutation(file="m.py", mutation_type="replace", description="d",
                           original_content="orig\n", mutated_content="new\n", post_apply_hash=hash_)


def _forge():
    return pf.PromptForge.__new__(pf.PromptForge)


def test_revert_refuses_moved_file(monkeypatch, tmp_path):
    _on(monkeypatch)
    m = _mutation(tmp_path, monkeypatch, pf._content_hash("new\n"))
    f = _forge()
    f.auto_commit = False
    with pytest.raises(RuntimeError, match="revert refused"):
        f.revert_code_mutation(m)
    assert (tmp_path / "m.py").read_text() == "moved\n"


def test_revert_matching_hash_reverts(monkeypatch, tmp_path):
    _on(monkeypatch)
    m = _mutation(tmp_path, monkeypatch, pf._content_hash("moved\n"))
    f = _forge()
    f.auto_commit = False
    f.revert_code_mutation(m)
    assert (tmp_path / "m.py").read_text() == "orig\n"


def test_revert_flag_off_unchanged(monkeypatch, tmp_path):
    m = _mutation(tmp_path, monkeypatch, pf._content_hash("new\n"))
    f = _forge()
    f.auto_commit = False
    f.revert_code_mutation(m)
    assert (tmp_path / "m.py").read_text() == "orig\n"


def test_apply_records_hash(monkeypatch, tmp_path):
    target = tmp_path / "t.py"
    target.write_text("orig\n")
    monkeypatch.setattr(pf, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(pf, "_resolve_code_mutation_target", lambda f: target)
    monkeypatch.setattr(pf, "CODE_MUTATION_ALLOWLIST", {"t.py"})
    monkeypatch.setattr(pf, "mutation_risk_gate_reason", lambda e: None)

    class Ctx:
        worktree_path = tmp_path

        def apply_file(self, rel, content):
            target.write_text(content)

    m = pf.CodeMutation(file="t.py", mutation_type="replace", description="d", original_content="orig\n",
                        mutated_content="new\n", syntax_valid=True)
    out = _forge().apply_code_mutation_in_context(Ctx(), m)
    assert out["status"] == "applied_isolated"
    assert m.post_apply_hash == pf._content_hash("new\n")
