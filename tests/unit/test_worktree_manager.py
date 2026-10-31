"""Tests for worktree_manager.py — git worktree isolation for PromptForge.

These tests use a temporary git repo to avoid touching the real project.
"""

import subprocess

import pytest

from scripts.autopilot.worktree_manager import WorktreeManager


@pytest.fixture
def temp_git_repo(tmp_path):
    """Create a temporary git repo with an initial commit."""
    repo = tmp_path / "repo"
    repo.mkdir()

    # Initialize repo
    subprocess.run(["git", "init"], cwd=repo, capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=repo, capture_output=True, check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=repo, capture_output=True, check=True,
    )

    # Create initial file and commit
    prompts_dir = repo / "orchestration" / "prompts"
    prompts_dir.mkdir(parents=True)
    (prompts_dir / "frontdoor.md").write_text("original frontdoor content")
    (repo / "src" / "escalation.py").parent.mkdir(parents=True)
    (repo / "src" / "escalation.py").write_text("# original escalation\ndef escalate(): pass\n")

    subprocess.run(["git", "add", "."], cwd=repo, capture_output=True, check=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=repo, capture_output=True, check=True,
    )

    # Create worktrees base dir
    (repo / "tmp" / "worktrees").mkdir(parents=True)

    return repo


def test_experiment_create_and_cleanup(temp_git_repo, monkeypatch):
    """Worktree is created and cleaned up."""
    import scripts.autopilot.worktree_manager as wm
    monkeypatch.setattr(wm, "WORKTREE_BASE", temp_git_repo / "tmp" / "worktrees")

    mgr = WorktreeManager(temp_git_repo)
    wt_path = temp_git_repo / "tmp" / "worktrees" / "test_trial"

    with mgr.experiment("test_trial") as ctx:
        assert ctx.worktree_path == wt_path
        assert wt_path.exists()
        assert (wt_path / "orchestration" / "prompts" / "frontdoor.md").exists()

    # Cleaned up after exit
    assert not wt_path.exists()


def test_apply_file_writes_to_both(temp_git_repo, monkeypatch):
    """apply_file writes to worktree and main repo."""
    import scripts.autopilot.worktree_manager as wm
    monkeypatch.setattr(wm, "WORKTREE_BASE", temp_git_repo / "tmp" / "worktrees")

    mgr = WorktreeManager(temp_git_repo)

    with mgr.experiment("test_apply") as ctx:
        ctx.apply_file(
            "orchestration/prompts/frontdoor.md",
            "mutated frontdoor content",
        )
        # Main repo has mutated content
        main_content = (temp_git_repo / "orchestration" / "prompts" / "frontdoor.md").read_text()
        assert main_content == "mutated frontdoor content"

        # Worktree also has mutated content
        wt_content = (ctx.worktree_path / "orchestration" / "prompts" / "frontdoor.md").read_text()
        assert wt_content == "mutated frontdoor content"

        # Reject
        ctx.reject()

    # After rejection, original is restored
    restored = (temp_git_repo / "orchestration" / "prompts" / "frontdoor.md").read_text()
    assert restored == "original frontdoor content"


def test_accept_commits_to_main(temp_git_repo, monkeypatch):
    """accept() commits the mutation in the main repo."""
    import scripts.autopilot.worktree_manager as wm
    monkeypatch.setattr(wm, "WORKTREE_BASE", temp_git_repo / "tmp" / "worktrees")

    mgr = WorktreeManager(temp_git_repo)

    with mgr.experiment("test_accept") as ctx:
        ctx.apply_file(
            "orchestration/prompts/frontdoor.md",
            "improved frontdoor",
        )
        ctx.accept("autopilot: improved frontdoor prompt")

    # Main repo has the mutation persisted
    content = (temp_git_repo / "orchestration" / "prompts" / "frontdoor.md").read_text()
    assert content == "improved frontdoor"

    # And it's committed
    result = subprocess.run(
        ["git", "log", "--oneline", "-1"],
        cwd=temp_git_repo, capture_output=True, text=True,
    )
    assert "improved frontdoor" in result.stdout


def test_accept_does_not_commit_or_unstage_unrelated_operator_work(temp_git_repo, monkeypatch):
    import scripts.autopilot.worktree_manager as wm
    monkeypatch.setattr(wm, "WORKTREE_BASE", temp_git_repo / "tmp" / "worktrees")

    unrelated = temp_git_repo / "src" / "escalation.py"
    unrelated.write_text("operator staged edit\n")
    subprocess.run(["git", "add", "--", "src/escalation.py"], cwd=temp_git_repo, check=True)

    with WorktreeManager(temp_git_repo).experiment("test_scoped_accept") as ctx:
        ctx.apply_file("orchestration/prompts/frontdoor.md", "accepted candidate")
        ctx.accept("autopilot: accept candidate")

    committed_unrelated = subprocess.run(
        ["git", "show", "HEAD:src/escalation.py"],
        cwd=temp_git_repo, capture_output=True, text=True, check=True,
    ).stdout
    staged_unrelated = subprocess.run(
        ["git", "show", ":src/escalation.py"],
        cwd=temp_git_repo, capture_output=True, text=True, check=True,
    ).stdout
    assert committed_unrelated == "# original escalation\ndef escalate(): pass\n"
    assert staged_unrelated == "operator staged edit\n"


def test_accept_commits_private_index_bytes_not_post_stage_worktree_edit(
    temp_git_repo, monkeypatch
):
    import scripts.autopilot.worktree_manager as wm

    monkeypatch.setattr(wm, "WORKTREE_BASE", temp_git_repo / "tmp" / "worktrees")
    unrelated = temp_git_repo / "src" / "escalation.py"
    unrelated.write_text("operator staged edit\n")
    subprocess.run(["git", "add", "--", "src/escalation.py"], cwd=temp_git_repo, check=True)
    staged_unrelated_before = subprocess.run(
        ["git", "ls-files", "--stage", "--", "src/escalation.py"],
        cwd=temp_git_repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout

    target = temp_git_repo / "orchestration" / "prompts" / "frontdoor.md"
    hooks = temp_git_repo / ".git" / "hooks"
    hooks.mkdir(exist_ok=True)
    pre_commit = hooks / "pre-commit"
    pre_commit.write_text(
        "#!/bin/sh\n"
        f"printf 'parallel post-stage edit\\n' > '{target}'\n"
    )
    pre_commit.chmod(0o755)

    with WorktreeManager(temp_git_repo).experiment("test_private_index_commit") as ctx:
        ctx.apply_file("orchestration/prompts/frontdoor.md", "evaluated candidate\n")
        ctx.accept("autopilot: accept evaluated candidate")

    committed_target = subprocess.run(
        ["git", "show", "HEAD:orchestration/prompts/frontdoor.md"],
        cwd=temp_git_repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert committed_target == "evaluated candidate\n"
    assert target.read_text() == "parallel post-stage edit\n"
    assert subprocess.run(
        ["git", "diff", "--cached", "--", "orchestration/prompts/frontdoor.md"],
        cwd=temp_git_repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout == ""
    assert subprocess.run(
        ["git", "ls-files", "--stage", "--", "src/escalation.py"],
        cwd=temp_git_repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout == staged_unrelated_before


def test_failed_commit_hook_restores_preimage_and_preserves_shared_index(temp_git_repo, monkeypatch):
    import scripts.autopilot.worktree_manager as wm
    monkeypatch.setattr(wm, "WORKTREE_BASE", temp_git_repo / "tmp" / "worktrees")
    unrelated = temp_git_repo / "src" / "escalation.py"
    unrelated.write_text("operator staged edit\n")
    subprocess.run(["git", "add", "--", "src/escalation.py"], cwd=temp_git_repo, check=True)
    staged_before = subprocess.run(
        ["git", "ls-files", "--stage", "--", "src/escalation.py"],
        cwd=temp_git_repo, capture_output=True, text=True, check=True,
    ).stdout

    hooks = temp_git_repo / ".git" / "hooks"
    hooks.mkdir(exist_ok=True)
    pre_commit = hooks / "pre-commit"
    pre_commit.write_text("#!/bin/sh\nexit 1\n")
    pre_commit.chmod(0o755)
    target = temp_git_repo / "orchestration" / "prompts" / "frontdoor.md"
    with pytest.raises(RuntimeError, match="experiment commit rejected"):
        with WorktreeManager(temp_git_repo).experiment("test_hook_reject") as ctx:
            ctx.apply_file("orchestration/prompts/frontdoor.md", "candidate")
            ctx.accept("autopilot: rejected by hook")

    assert target.read_text() == "original frontdoor content"
    assert subprocess.run(
        ["git", "ls-files", "--stage", "--", "src/escalation.py"],
        cwd=temp_git_repo, capture_output=True, text=True, check=True,
    ).stdout == staged_before
    assert subprocess.run(
        ["git", "show", "HEAD:orchestration/prompts/frontdoor.md"],
        cwd=temp_git_repo, capture_output=True, text=True, check=True,
    ).stdout == "original frontdoor content"


def test_auto_reject_on_exception_restores_candidate_and_new_file(temp_git_repo, monkeypatch):
    import scripts.autopilot.worktree_manager as wm
    monkeypatch.setattr(wm, "WORKTREE_BASE", temp_git_repo / "tmp" / "worktrees")

    target = temp_git_repo / "orchestration" / "prompts" / "frontdoor.md"
    new_file = temp_git_repo / "src" / "generated.py"
    with pytest.raises(RuntimeError, match="eval failed"):
        with WorktreeManager(temp_git_repo).experiment("test_exception_reject") as ctx:
            ctx.apply_file("orchestration/prompts/frontdoor.md", "candidate")
            ctx.apply_file("src/generated.py", "candidate module")
            raise RuntimeError("eval failed")

    assert target.read_text() == "original frontdoor content"
    assert not new_file.exists()


def test_code_mutation_context_restores_new_file_on_early_exit(temp_git_repo, monkeypatch):
    import scripts.autopilot.species.prompt_forge as prompt_forge_mod
    import scripts.autopilot.worktree_manager as wm
    from scripts.autopilot.species.prompt_forge import CodeMutation, MutationEffect, PromptForge

    monkeypatch.setattr(wm, "WORKTREE_BASE", temp_git_repo / "tmp" / "worktrees")
    monkeypatch.setattr(prompt_forge_mod, "PROJECT_ROOT", temp_git_repo)
    monkeypatch.setattr(prompt_forge_mod, "NEW_FILE_MUTATION_ROOT", temp_git_repo / "src")
    target = temp_git_repo / "src" / "generated.py"
    mutation = CodeMutation(
        file="src/generated.py",
        mutation_type="new_file",
        description="fixture candidate",
        original_content="",
        mutated_content="VALUE = 1\n",
    )
    mutation.syntax_valid = True
    mutation.effect = MutationEffect.CONSTRAIN
    forge = PromptForge(
        prompts_dir=temp_git_repo / "orchestration" / "prompts", auto_commit=False
    )
    with WorktreeManager(temp_git_repo).experiment("code_mutation_auto_reject") as ctx:
        result = forge.apply_code_mutation_in_context(ctx, mutation)
        assert result["status"] == "applied_isolated"
        assert target.read_text() == "VALUE = 1\n"
        # An undecided action exit models gate rejection and other early returns.

    assert not target.exists()


def test_auto_reject_on_no_decision(temp_git_repo, monkeypatch):
    """If neither accept nor reject is called, auto-reject."""
    import scripts.autopilot.worktree_manager as wm
    monkeypatch.setattr(wm, "WORKTREE_BASE", temp_git_repo / "tmp" / "worktrees")

    mgr = WorktreeManager(temp_git_repo)

    with mgr.experiment("test_auto_reject") as ctx:
        ctx.apply_file(
            "orchestration/prompts/frontdoor.md",
            "possibly bad mutation",
        )
        # Deliberately don't call accept() or reject()

    # Original restored
    content = (temp_git_repo / "orchestration" / "prompts" / "frontdoor.md").read_text()
    assert content == "original frontdoor content"


def test_multiple_files(temp_git_repo, monkeypatch):
    """Can apply mutations to multiple files in one experiment."""
    import scripts.autopilot.worktree_manager as wm
    monkeypatch.setattr(wm, "WORKTREE_BASE", temp_git_repo / "tmp" / "worktrees")

    mgr = WorktreeManager(temp_git_repo)

    with mgr.experiment("test_multi") as ctx:
        ctx.apply_file("orchestration/prompts/frontdoor.md", "new frontdoor")
        ctx.apply_file("src/escalation.py", "# new escalation\ndef escalate(): return True\n")
        ctx.reject()

    assert (temp_git_repo / "orchestration" / "prompts" / "frontdoor.md").read_text() == "original frontdoor content"
    assert "original escalation" in (temp_git_repo / "src" / "escalation.py").read_text()


def test_reject_restores_exact_uncommitted_main_worktree_preimage(temp_git_repo, monkeypatch):
    """Rollback preserves the bytes present at apply time, not the HEAD version."""
    import scripts.autopilot.worktree_manager as wm
    monkeypatch.setattr(wm, "WORKTREE_BASE", temp_git_repo / "tmp" / "worktrees")

    target = temp_git_repo / "orchestration" / "prompts" / "frontdoor.md"
    dirty_preimage = b"operator edit not committed\nwith a second line\n"
    target.write_bytes(dirty_preimage)

    with WorktreeManager(temp_git_repo).experiment("test_dirty_preimage") as ctx:
        ctx.apply_file("orchestration/prompts/frontdoor.md", "candidate mutation")
        ctx.reject()

    assert target.read_bytes() == dirty_preimage
    assert "operator edit" in subprocess.run(
        ["git", "diff", "--", "orchestration/prompts/frontdoor.md"],
        cwd=temp_git_repo, capture_output=True, text=True, check=True,
    ).stdout


def test_reject_distinguishes_empty_file_from_absent_file(temp_git_repo, monkeypatch):
    import scripts.autopilot.worktree_manager as wm
    monkeypatch.setattr(wm, "WORKTREE_BASE", temp_git_repo / "tmp" / "worktrees")

    empty = temp_git_repo / "empty.txt"
    empty.write_bytes(b"")
    absent = temp_git_repo / "new.txt"

    with WorktreeManager(temp_git_repo).experiment("test_empty_absent") as ctx:
        ctx.apply_file("empty.txt", "candidate")
        ctx.apply_file("new.txt", "candidate")
        ctx.reject()

    assert empty.exists() and empty.read_bytes() == b""
    assert not absent.exists()
