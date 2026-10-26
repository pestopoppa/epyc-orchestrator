"""Keep the pip compatibility extra aligned with uv's canonical dev group."""

from __future__ import annotations

import tomllib
from pathlib import Path


def test_dev_extra_matches_uv_dependency_group() -> None:
    project = Path(__file__).resolve().parents[2] / "pyproject.toml"
    metadata = tomllib.loads(project.read_text(encoding="utf-8"))

    dev_extra = metadata["project"]["optional-dependencies"]["dev"]
    dev_group = metadata["dependency-groups"]["dev"]

    assert dev_extra == dev_group
    assert [requirement for requirement in dev_group if requirement.startswith("ruff")] == [
        "ruff>=0.15.10"
    ]
