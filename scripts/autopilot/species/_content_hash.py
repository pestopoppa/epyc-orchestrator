"""Content-hash helpers for NIB2-41 staleness detection.

sha256 over canonical file bytes. Keeps the same hash for identical content
regardless of mtime or inode — staleness is purely content-driven.

See `research/deep-dives/token-savior-extractable-patterns.md` §2 for the
origin of this primitive (Token Savior uses it for code-symbol staleness).
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def hash_file(path: Path) -> str | None:
    """Return the sha256 hex digest of file content, or None if absent."""
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_glob(root: Path, patterns: list[str]) -> dict[str, str]:
    """Return {relative_path: sha256_hex} for every file matching any pattern.

    Paths are reported relative to ``root`` so hashes survive cwd moves.
    Missing files are omitted (use ``hash_file`` directly for presence checks).
    """
    result: dict[str, str] = {}
    for pattern in patterns:
        for match in root.glob(pattern):
            if not match.is_file():
                continue
            digest = hash_file(match)
            if digest is None:
                continue
            rel = match.relative_to(root)
            result[str(rel)] = digest
    return result
