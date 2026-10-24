"""Materialize the research MASTER registry from ``origin/main`` (never the shared clone).

The shared research clone (``/mnt/raid0/llm/epyc-inference-research``) is deliberately
left behind origin/main and must not be synced, so compiling from its working copy
silently drops landed declarations (2026-10-24: ten ``ctx_checkpoints: 32`` lines).
The compile pipeline therefore reads ``origin/main:orchestration/model_registry.yaml``
-- the ONLY research file the compile reads -- out of git into a private temp dir.

Provenance: the temp path is meaningless, so the materialized file is registered here
with a stable label (``origin/main:<relpath>``) and the origin/main commit; the
descriptor ``source_registries.research`` pin and the lean-registry banner consult
:func:`provenance_for` instead of recording a temp path. The label carries no SHA on
purpose -- the SHA lives in ``repo_commit`` (which the generated-artifact comparison
already ignores) and the content identity in ``sha256`` -- so an unrelated research
commit never churns the recorded path.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

RESEARCH_REPO = Path("/mnt/raid0/llm/epyc-inference-research")
REGISTRY_RELPATH = "orchestration/model_registry.yaml"
FETCH_TIMEOUT_S = 20
LABEL_PREFIX = "origin/main:"
# Explicit override for every default reader of the research MASTER registry (stack
# start, the compiler/descriptor/shared_with/reviewer-pool CLIs). A path given here or
# by a CLI flag wins over origin/main; nothing ever falls back to the shared clone.
RESEARCH_REGISTRY_ENV = "EPYC_RESEARCH_MASTER_REGISTRY"

# resolved materialized path -> (stable label, short commit)
_PROVENANCE: dict[Path, tuple[str, str]] = {}


def provenance_for(path: Path) -> tuple[str, str] | None:
    """(label, short origin/main commit) if ``path`` was materialized from origin/main."""
    try:
        return _PROVENANCE.get(Path(path).resolve())
    except OSError:
        return None


@dataclass
class MaterializedRegistry:
    path: Path
    sha: str
    label: str
    warnings: list[str] = field(default_factory=list)
    _tmp: tempfile.TemporaryDirectory | None = None

    def cleanup(self) -> None:
        _PROVENANCE.pop(self.path.resolve(), None)
        if self._tmp is not None:
            self._tmp.cleanup()
            self._tmp = None


class ResearchOriginError(RuntimeError):
    pass


def _git(repo: Path, *args: str, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True, text=True, timeout=timeout, check=False,
    )


def materialize_research_registry(
    repo: Path = RESEARCH_REPO, relpath: str = REGISTRY_RELPATH
) -> MaterializedRegistry:
    """``git fetch origin main`` (ref-only), then ``git show`` the registry into a temp dir."""
    warnings: list[str] = []
    try:
        fetch = _git(repo, "fetch", "--quiet", "origin", "main", timeout=FETCH_TIMEOUT_S)
        fetched = fetch.returncode == 0
        why = fetch.stderr.strip()[:200]
    except subprocess.TimeoutExpired:
        fetched, why = False, f"timed out after {FETCH_TIMEOUT_S}s"
    except OSError as exc:
        # No git binary (or it cannot be executed): there is no origin/main to read, and
        # the working copy is exactly what this module exists to avoid. Fail closed.
        raise ResearchOriginError(f"git is unavailable ({exc}); cannot read origin/main in {repo}") from exc
    if not fetched:
        warnings.append(
            f"git fetch origin main failed in {repo} ({why}); using the EXISTING "
            "origin/main ref, which may itself be stale"
        )
    sha = _git(repo, "rev-parse", "--verify", "origin/main^{commit}")
    if sha.returncode != 0:
        raise ResearchOriginError(f"origin/main does not resolve in {repo}: {sha.stderr.strip()}")
    full = sha.stdout.strip()
    blob = subprocess.run(
        ["git", "-C", str(repo), "show", f"{full}:{relpath}"],
        capture_output=True, check=False,
    )
    if blob.returncode != 0:
        raise ResearchOriginError(
            f"{full}:{relpath} not readable in {repo}: {blob.stderr.decode(errors='replace').strip()}"
        )
    tmp = tempfile.TemporaryDirectory(prefix="research-origin-main-")
    dest = Path(tmp.name) / relpath
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(blob.stdout)
    short = _git(repo, "rev-parse", "--short", full).stdout.strip() or full[:9]
    label = f"{LABEL_PREFIX}{relpath}"
    _PROVENANCE[dest.resolve()] = (label, short)
    return MaterializedRegistry(path=dest, sha=full, label=label, warnings=warnings, _tmp=tmp)


@dataclass
class ResolvedResearchRegistry:
    """The research MASTER registry a default reader should use, and where it came from."""

    path: Path
    source: str  # "explicit" | "env" | "origin/main"
    materialized: MaterializedRegistry | None = None

    @property
    def description(self) -> str:
        if self.materialized is not None:
            return f"{self.materialized.label} @ {self.materialized.sha}"
        return f"{self.path} ({self.source} override)"

    def cleanup(self) -> None:
        if self.materialized is not None:
            self.materialized.cleanup()


def _warn_stderr(message: str) -> None:
    print(f"WARNING: {message}", file=sys.stderr)


def resolve_research_registry(
    explicit: Path | str | None = None,
    *,
    flag: str = "--research-registry",
    env: Mapping[str, str] | None = None,
    repo: Path | None = None,
    warn: Callable[[str], None] = _warn_stderr,
) -> ResolvedResearchRegistry:
    """Resolve the research master for a DEFAULT reader: explicit > env > origin/main.

    * ``explicit`` (a CLI flag value) wins, then ``$EPYC_RESEARCH_MASTER_REGISTRY``.
    * Otherwise the registry is materialized from ``origin/main`` (ref-only fetch; a
      failed fetch falls back to the EXISTING origin/main ref, loudly via ``warn``).
    * If origin/main cannot be read at all (git missing, ref missing, blob missing) this
      raises :class:`ResearchOriginError` naming ``flag`` and the env var. It never falls
      back to the shared clone's working copy, which is deliberately left stale.

    The caller owns ``cleanup()`` of the result (removes the temp dir and provenance).
    """
    if explicit is not None and str(explicit) != "":
        return ResolvedResearchRegistry(path=Path(explicit), source="explicit")
    environ = os.environ if env is None else env
    from_env = (environ.get(RESEARCH_REGISTRY_ENV) or "").strip()
    if from_env:
        return ResolvedResearchRegistry(path=Path(from_env), source="env")
    repo = RESEARCH_REPO if repo is None else Path(repo)
    try:
        materialized = materialize_research_registry(repo=repo)
    except ResearchOriginError as exc:
        raise ResearchOriginError(
            f"cannot materialize the research master registry from origin/main: {exc}. "
            f"Refusing to fall back to the shared clone working copy {repo / REGISTRY_RELPATH} "
            f"(it is deliberately behind origin/main). Pass {flag} <path> or set "
            f"{RESEARCH_REGISTRY_ENV}=<path> to name a registry explicitly."
        ) from exc
    for message in materialized.warnings:
        warn(message)
    return ResolvedResearchRegistry(
        path=materialized.path, source="origin/main", materialized=materialized
    )
