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

import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

RESEARCH_REPO = Path("/mnt/raid0/llm/epyc-inference-research")
REGISTRY_RELPATH = "orchestration/model_registry.yaml"
FETCH_TIMEOUT_S = 20
LABEL_PREFIX = "origin/main:"

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
