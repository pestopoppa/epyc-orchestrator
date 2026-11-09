"""RTG-02 zero-importer lister: synthetic-tree semantics plus one real-tree acceptance check."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis import zero_importer_modules as zim  # noqa: E402


def _w(root: Path, rel: str, text: str = "x = 1\n") -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def _tree(root: Path) -> None:
    _w(root, "pyproject.toml", '[project]\nname="t"\n[project.scripts]\norch = "src.cli_orch:main"\n')
    _w(root, "src/__init__.py", "")
    _w(root, "src/cli_orch.py", "def main(): pass\n")  # declared console script
    _w(root, "src/mcp_server.py", "if __name__ == '__main__': pass\n")  # extra entry point
    _w(root, "src/used_abs.py")
    _w(root, "src/used_rel.py")
    _w(root, "src/used_from.py")
    _w(root, "src/pkg/__init__.py", "")
    _w(root, "src/pkg/inner.py")
    _w(root, "src/pkg/sibling.py", "from . import inner\n")  # relative import
    _w(root, "src/dyn.py")
    _w(root, "src/orphan.py")  # nothing imports it
    _w(root, "src/tested_only.py")
    _w(root, "src/stemonly.py")
    _w(root, "src/consumer.py",
       "import src.used_abs\nfrom src import used_from\nfrom .used_rel import X\n"
       "import importlib\nimportlib.import_module('src.dyn')\n")
    _w(root, "scripts/tool.py", "from src.pkg.sibling import y\nimport stemonly\n")  # bare-name import
    _w(root, "tests/test_a.py", "import src.tested_only\n")


def test_synthetic_tree_candidates_and_entry_point_exclusion(tmp_path):
    _tree(tmp_path)
    rep = zim.build_report(tmp_path)
    names = {r["module"]: r for r in rep["candidates"]}
    assert "src.orphan" in names and names["src.orphan"]["method"] == "both"
    assert "src.tested_only" in names  # only tests import it
    assert names["src.tested_only"]["test_importers_dotted"] == 1
    # entry points are excluded from candidates and reported separately
    assert "src.cli_orch" not in names and "src.mcp_server" not in names
    excluded = {r["module"]: r["entry_point_reason"] for r in rep["excluded_entry_points"]}
    assert "console script `orch`" in excluded["src.cli_orch"]
    assert excluded["src.mcp_server"] == "explicit entry point"
    # used by absolute, from-import, relative, dynamic-string and sibling-relative imports
    for used in ("src.used_abs", "src.used_from", "src.used_rel", "src.dyn", "src.pkg.inner", "src.pkg.sibling"):
        assert used not in names, used
    # stem scan finds the bare `import stemonly`; the dotted scan does not
    assert "src.stemonly" in names and names["src.stemonly"]["method"] == "dotted-only"
    assert rep["counts"]["candidates_union"] == len(rep["candidates"])
    assert rep["counts"]["excluded_entry_points"] == 2
    assert all("wire_or_retire" in r for r in rep["candidates"])


def test_self_import_does_not_count_and_output_is_deterministic(tmp_path):
    _tree(tmp_path)
    _w(tmp_path, "src/orphan.py", "import src.orphan\n")
    a = zim.build_report(tmp_path)
    b = zim.build_report(tmp_path)
    assert a == b
    assert "src.orphan" in {r["module"] for r in a["candidates"]}


def test_markdown_and_cli(tmp_path, capsys):
    _tree(tmp_path)
    assert zim.main(["--root", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "`src.orphan`" in out and "Excluded entry points" in out
    assert zim.main(["--root", str(tmp_path), "--format", "json", "--out", str(tmp_path / "o.json")]) == 0
    assert '"candidates_union"' in (tmp_path / "o.json").read_text()


def test_real_tree_lists_at_least_twenty_and_excludes_entry_points():
    """Box acceptance: >=20 modules; pyproject console scripts and mcp_server.py excluded."""
    rep = zim.build_report(REPO_ROOT)
    names = {r["module"] for r in rep["candidates"]}
    assert len(names) >= 20, len(names)
    for entry in ("src.cli_orch", "src.cli_sessions", "src.registry_loader", "src.mcp_server"):
        assert entry not in names
    assert "src.mutation_ledger" in names  # the named instance (no production importer)


def test_non_python_launcher_reference_is_reported(tmp_path):
    _tree(tmp_path)
    _w(tmp_path, "scripts/run.sh", "python -m src.orphan --serve\n")
    rep = zim.build_report(tmp_path)
    row = next(r for r in rep["candidates"] if r["module"] == "src.orphan")
    assert row["non_python_launch_refs"] == ["scripts/run.sh"]
    assert "non-Python launcher" in row["wire_or_retire"]
    other = next(r for r in rep["candidates"] if r["module"] == "src.tested_only")
    assert other["non_python_launch_ref_count"] == 0
