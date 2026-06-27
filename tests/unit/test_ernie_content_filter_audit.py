import json
from pathlib import Path

from scripts.diffusion import ernie_content_filter_audit as audit


def test_manifest_contains_required_categories() -> None:
    manifest = audit.build_manifest(width=1024, height=1024, steps=8, seed=1)

    assert audit.validate_manifest(manifest) == []
    categories = {case["category"] for case in manifest["cases"]}
    assert audit.REQUIRED_CATEGORIES <= categories


def test_manifest_case_ids_are_unique() -> None:
    manifest = audit.build_manifest(width=1024, height=1024, steps=8, seed=1)

    case_ids = [case["case_id"] for case in manifest["cases"]]
    assert len(case_ids) == len(set(case_ids))


def test_dry_run_writes_manifest_without_execution(tmp_path: Path) -> None:
    output = tmp_path / "manifest.json"
    manifest = audit.build_manifest(width=512, height=512, steps=4, seed=7, limit=2)

    audit.write_json(manifest, output)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["default_execute"] is False
    assert "results" not in payload
    assert payload["run_config"] == {
        "width": 512,
        "height": 512,
        "steps": 4,
        "seed": 7,
    }
