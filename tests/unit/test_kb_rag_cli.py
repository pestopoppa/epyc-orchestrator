from __future__ import annotations

import argparse
import json
from pathlib import Path


def test_paths_from_source_manifest_resolves_relative_paths(tmp_path: Path) -> None:
    from scripts.kb_rag import cli

    root = tmp_path / "root"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "kind": "project-wiki-source-manifest",
                "sources": [
                    {"path": "wiki/knowledge-management.md"},
                    {"path": str(root / "handoffs" / "active" / "absolute.md")},
                    {"path": ""},
                    {},
                ],
                "removed_sources": [{"path": "progress/old.md"}],
            }
        ),
        encoding="utf-8",
    )

    paths, removed = cli._paths_from_source_manifest(manifest, manifest_root=root)

    assert paths == [
        str((root / "wiki" / "knowledge-management.md").resolve()),
        str((root / "handoffs" / "active" / "absolute.md").resolve()),
    ]
    assert removed == [{"path": "progress/old.md"}]


def test_update_command_accepts_project_wiki_manifest(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    from scripts.kb_rag import cli

    root = tmp_path / "root"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "kind": "project-wiki-source-manifest",
                "sources": [{"path": "handoffs/active/internal-kb-rag.md"}],
                "removed_sources": [{"path": "progress/old.md"}],
            }
        ),
        encoding="utf-8",
    )
    calls: dict = {}

    class FakeConfig:
        @classmethod
        def from_yaml(cls, path):
            calls["config_path"] = path
            return "config"

    def fake_update_files(paths, config, index_dir):
        calls["paths"] = paths
        calls["config"] = config
        calls["index_dir"] = index_dir
        return {"ok": True, "files_processed": len(paths)}

    monkeypatch.setattr(cli, "CorpusConfig", FakeConfig)
    monkeypatch.setattr(cli, "update_files", fake_update_files)

    rc = cli._cmd_update(
        argparse.Namespace(
            config=None,
            files=["/already/absolute.md"],
            index_dir=tmp_path / "idx",
            manifest=str(manifest),
            manifest_root=str(root),
        )
    )

    assert rc == 0
    assert calls["paths"] == [
        "/already/absolute.md",
        str((root / "handoffs" / "active" / "internal-kb-rag.md").resolve()),
    ]
    assert calls["config"] == "config"
    output = json.loads(capsys.readouterr().out)
    assert output["manifest_paths"] == 1
    assert output["manifest_removed_sources_ignored"] == 1
