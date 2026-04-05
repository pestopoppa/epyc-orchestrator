"""Git worktree manager for isolated PromptForge experiments.

Creates temporary git worktrees for mutation experiments. The mutated file
is applied in the worktree (committed there for version history), then
copied into the main repo for eval (since the live server reads from main).
On rejection, the original is restored from the worktree's clean copy.

This prevents the corruption incidents seen in AR-3 where a bad mutation
left the main branch in a broken state — the worktree always holds a
clean pre-mutation snapshot for guaranteed recovery.

Usage:
    wt = WorktreeManager(project_root)
    with wt.experiment("trial_42") as ctx:
        # ctx.worktree_path is the worktree directory
        # ctx.branch_name is the temp branch

        # Apply mutation in worktree + copy to main for eval
        ctx.apply_file("orchestration/prompts/frontdoor.md", mutated_content)

        # Run eval... (reads from main repo)
        result = tower.hybrid_eval()

        # Accept or reject
        if result.quality > baseline:
            ctx.accept("autopilot: improved frontdoor prompt")
        else:
            ctx.reject()  # restores original in main
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator

log = logging.getLogger("autopilot.worktree")

DEFAULT_PROJECT_ROOT = Path("/mnt/raid0/llm/epyc-orchestrator")
WORKTREE_BASE = DEFAULT_PROJECT_ROOT / "tmp" / "worktrees"


@dataclass
class ExperimentContext:
    """Context for a worktree-isolated experiment."""

    worktree_path: Path
    branch_name: str
    project_root: Path
    _applied_files: dict[str, str] = field(default_factory=dict)  # rel_path -> original content
    _accepted: bool = False
    _decided: bool = False

    def apply_file(self, rel_path: str, content: str) -> None:
        """Apply mutated content: save original, write to worktree + main.

        Args:
            rel_path: Relative path from project root (e.g. "orchestration/prompts/frontdoor.md")
            content: Mutated file content
        """
        main_path = self.project_root / rel_path
        wt_path = self.worktree_path / rel_path

        # Save original content for rollback
        if rel_path not in self._applied_files:
            if main_path.exists():
                self._applied_files[rel_path] = main_path.read_text()
            else:
                self._applied_files[rel_path] = ""

        # Write to worktree (for version history)
        wt_path.parent.mkdir(parents=True, exist_ok=True)
        wt_path.write_text(content)

        # Commit in worktree
        _git(["add", rel_path], cwd=self.worktree_path)
        _git(
            ["commit", "-m", f"experiment: mutate {rel_path}"],
            cwd=self.worktree_path,
            check=False,  # may fail if no changes
        )

        # Copy to main repo for live eval
        main_path.parent.mkdir(parents=True, exist_ok=True)
        main_path.write_text(content)

    def accept(self, commit_message: str = "") -> None:
        """Accept the mutation — commit changes in main repo."""
        if self._decided:
            return
        self._decided = True
        self._accepted = True

        msg = commit_message or "autopilot: accept experiment"
        for rel_path in self._applied_files:
            _git(["add", rel_path], cwd=self.project_root)
        _git(["commit", "-m", msg], cwd=self.project_root, check=False)
        log.info("Experiment accepted: %s", msg)

    def reject(self) -> None:
        """Reject the mutation — restore originals in main repo."""
        if self._decided:
            return
        self._decided = True
        self._accepted = False

        for rel_path, original in self._applied_files.items():
            main_path = self.project_root / rel_path
            if original:
                main_path.write_text(original)
            elif main_path.exists():
                main_path.unlink()
            _git(["checkout", "--", rel_path], cwd=self.project_root, check=False)
        log.info("Experiment rejected, files restored")

    @property
    def accepted(self) -> bool:
        return self._accepted


class WorktreeManager:
    """Manages temporary git worktrees for isolated experiments."""

    def __init__(self, project_root: Path = DEFAULT_PROJECT_ROOT):
        self.project_root = project_root
        WORKTREE_BASE.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def experiment(self, name: str) -> Generator[ExperimentContext, None, None]:
        """Create a temporary worktree for an experiment.

        The worktree is automatically cleaned up on exit. If the experiment
        was not explicitly accepted or rejected, it is auto-rejected (safe default).

        Args:
            name: Experiment name (used for branch and directory naming)
        """
        branch = f"experiment/{name}"
        wt_path = WORKTREE_BASE / name

        # Clean up stale worktree if exists
        if wt_path.exists():
            self._remove_worktree(wt_path)

        # Create worktree from current HEAD
        try:
            _git(
                ["worktree", "add", "-b", branch, str(wt_path), "HEAD"],
                cwd=self.project_root,
            )
        except subprocess.CalledProcessError:
            # Branch may already exist — try without -b
            _git(["branch", "-D", branch], cwd=self.project_root, check=False)
            _git(
                ["worktree", "add", "-b", branch, str(wt_path), "HEAD"],
                cwd=self.project_root,
            )

        ctx = ExperimentContext(
            worktree_path=wt_path,
            branch_name=branch,
            project_root=self.project_root,
        )

        try:
            yield ctx
        finally:
            # Auto-reject if not explicitly decided
            if not ctx._decided:
                ctx.reject()

            # Always clean up worktree
            self._remove_worktree(wt_path)

            # Delete the temporary branch
            _git(["branch", "-D", branch], cwd=self.project_root, check=False)

    def _remove_worktree(self, wt_path: Path) -> None:
        """Remove a worktree, handling edge cases."""
        try:
            _git(
                ["worktree", "remove", "--force", str(wt_path)],
                cwd=self.project_root,
                check=False,
            )
        except Exception:
            pass
        # Force-remove if git worktree remove failed
        if wt_path.exists():
            shutil.rmtree(wt_path, ignore_errors=True)
        # Prune stale worktree refs
        _git(["worktree", "prune"], cwd=self.project_root, check=False)

    def list_worktrees(self) -> list[str]:
        """List active worktrees."""
        result = _git(
            ["worktree", "list", "--porcelain"],
            cwd=self.project_root,
            check=False,
        )
        return [
            line.split(" ", 1)[1]
            for line in result.stdout.splitlines()
            if line.startswith("worktree ")
        ]


def _git(
    args: list[str],
    cwd: Path | str = DEFAULT_PROJECT_ROOT,
    check: bool = True,
    timeout: int = 30,
) -> subprocess.CompletedProcess:
    """Run a git command."""
    cmd = ["git"] + args
    try:
        return subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=check,
        )
    except subprocess.CalledProcessError as e:
        log.warning("git %s failed: %s", " ".join(args), e.stderr.strip())
        raise
    except subprocess.TimeoutExpired:
        log.warning("git %s timed out after %ds", " ".join(args), timeout)
        raise
