"""Tests for offline reward verifier NPZ extraction."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.graph_router import build_offline_reward_verifier_npz as mod


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    return path


def _source(path: Path) -> Path:
    return _write_json(
        path,
        [
            {
                "suite": "math",
                "question_id": "q1",
                "prompt": "What is 2+2?",
                "expected": "4",
                "role_results": {
                    "frontdoor": {"answer": "4", "passed": True},
                    "coder_primary": {"answer": "four", "passed": True},
                },
            },
        ],
    )


def _manifest_row(
    source_path: Path,
    *,
    role_key: str,
    label: int,
    score: float,
) -> dict:
    return {
        "schema_version": "offline_reward_feature_input.v1",
        "item_id": f"source:1:{role_key}",
        "join_key": f"{source_path}:0:{role_key}:source:1:{role_key}",
        "question_id": "q1",
        "suite": "math",
        "role_key": role_key,
        "source_path": str(source_path),
        "source_record_index": 1,
        "source_record_offset": 0,
        "source_record_index_base": "one_based",
        "source_record_count": 1,
        "source_role": role_key,
        "source_passed": True,
        "source_elapsed_seconds": 1.0,
        "source_error_present": False,
        "prompt_sha256": "p",
        "expected_sha256": "e",
        "answer_sha256": "a",
        "prompt_chars": len("What is 2+2?"),
        "expected_chars": 1,
        "answer_chars": 1,
        "feature_context": {
            "task_type": "general",
            "task_type_onehot": [0.0, 0.0, 0.0, 0.0, 1.0],
            "context_length_chars": len("What is 2+2?"),
            "has_images": False,
            "expected_classifier_feature_dim_without_embedding": 7,
        },
        "oracle_binary_label": label,
        "oracle_score": score,
        "oracle_threshold": 0.86,
        "oracle_score_source": "reference_token_coverage",
        "target_binary_label": label,
        "target_source": "answer_equivalence_final_label",
        "label_source": "reference_token_coverage@0.86",
        "label_status": "oracle_labeled",
    }


def test_build_verifier_npz_emits_compatible_contract_and_aliases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        mod,
        "load_live_canonical_actions",
        lambda: ["frontdoor", "architect_general", "coder_escalation"],
    )
    source_path = _source(tmp_path / "source.json")
    manifest_path = _write_jsonl(
        tmp_path / "manifest.jsonl",
        [
            _manifest_row(source_path, role_key="frontdoor:direct", label=1, score=1.0),
            _manifest_row(source_path, role_key="coder_primary", label=0, score=0.0),
        ],
    )
    out_npz = tmp_path / "verifier.npz"
    summary_json = tmp_path / "summary.json"
    calls: list[str] = []

    def fake_embed(text: str) -> np.ndarray:
        calls.append(text)
        return np.ones(1024, dtype=np.float32)

    summary = mod.build_verifier_npz(
        manifest_path,
        out_npz,
        summary_json=summary_json,
        embed_fn=fake_embed,
    )
    data = np.load(out_npz, allow_pickle=True)

    assert summary["rows"] == 2
    assert summary["unique_source_records_embedded"] == 1
    assert calls == ["What is 2+2?"]
    assert data["Z"].shape == (2, 1031 + 3)
    assert data["Z"].dtype == np.float32
    assert data["correct"].astype(np.float32).tolist() == [1.0, 0.0]
    assert data["actions"].astype(np.int64).tolist() == [0, 2]
    np.testing.assert_allclose(data["q_weights"].astype(np.float32), [1.0, 0.01])
    assert int(data["feature_dim"]) == 1031
    assert int(data["n_actions"]) == 3
    assert [str(row[1]) for row in data["label_map"]] == [
        "frontdoor",
        "architect_general",
        "coder_escalation",
    ]
    metadata = data["metadata"].tolist()
    assert "prompt" not in metadata[0]
    assert "answer" not in metadata[0]
    assert metadata[0]["source_record_index_base"] == "one_based"
    assert metadata[0]["oracle_threshold"] == 0.86
    assert json.loads(summary_json.read_text(encoding="utf-8"))["n_neg"] == 1


def test_build_verifier_npz_rejects_unmapped_actions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mod, "load_live_canonical_actions", lambda: ["frontdoor"])
    source_path = _source(tmp_path / "source.json")
    manifest_path = _write_jsonl(
        tmp_path / "manifest.jsonl",
        [_manifest_row(source_path, role_key="unknown_role", label=1, score=1.0)],
    )

    with pytest.raises(mod.OfflineRewardVerifierNpzError, match="cannot map role_key"):
        mod.build_verifier_npz(
            manifest_path,
            tmp_path / "verifier.npz",
            embed_fn=lambda _text: np.ones(1024, dtype=np.float32),
        )
