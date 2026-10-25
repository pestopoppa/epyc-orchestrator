from __future__ import annotations

import json

from scripts.graph_router import reembed_episodic_store as reembed
from scripts.graph_router.reembed_episodic_store import retry_port_order
from scripts.maintenance import repair_episodic_embeddings as repair
from scripts.server.stack_manifest import EMBEDDER_PORTS


def test_retry_port_order_rotates_from_primary() -> None:
    assert retry_port_order(8102, [8100, 8101, 8102, 8103]) == [
        8102,
        8103,
        8100,
        8101,
    ]


def test_retry_port_order_handles_unknown_primary() -> None:
    assert retry_port_order(9000, [8100, 8101]) == [9000, 8100, 8101]


def test_repair_defaults_follow_stack_embedder_ports() -> None:
    assert repair.DEFAULT_EMBEDDER_SERVERS == len(EMBEDDER_PORTS)
    assert repair.DEFAULT_EMBEDDER_BASE_PORT == min(EMBEDDER_PORTS)


def test_memory_embedding_text_prefers_objective() -> None:
    text = reembed.memory_embedding_text(
        action="worker_general",
        action_type="routing",
        context={
            "objective": "Use this objective.",
            "task_description": "Not this legacy description.",
        },
        outcome="success",
    )

    assert text == "Use this objective."


def test_memory_embedding_text_uses_legacy_task_description() -> None:
    text = reembed.memory_embedding_text(
        action="frontdoor",
        action_type="routing",
        context={
            "task_type": "coder",
            "task_description": "Write a function that counts repeating subsequences.",
        },
        outcome="success",
    )

    assert text == "Write a function that counts repeating subsequences."


def test_rows_for_embedding_keeps_unknown_actions_for_faiss_repair() -> None:
    rows = [
        (
            "m1",
            "plan_review:add",
            "routing",
            json.dumps({"task_description": "Review this plan."}),
            "success",
            0.7,
        ),
        (
            "m2",
            "worker_general",
            "routing",
            json.dumps({"objective": "Solve this task."}),
            "success",
            0.9,
        ),
    ]

    valid_rows, skipped = reembed.rows_for_embedding(rows)

    assert skipped == {"bad_json": 0, "empty_text": 0}
    assert [row[0] for row in valid_rows] == ["m1", "m2"]
    assert valid_rows[0][1] == "plan_review:add"
    assert valid_rows[0][3] == "Review this plan."
    assert valid_rows[1][1] == "worker_general"


def test_rows_for_embedding_falls_back_to_metadata_when_text_fields_missing() -> None:
    rows = [
        (
            "m1",
            "worker_general",
            "routing",
            json.dumps({"task_type": "math", "question_id": "q-1"}),
            "success",
            0.5,
        )
    ]

    valid_rows, skipped = reembed.rows_for_embedding(rows)

    assert skipped == {"bad_json": 0, "empty_text": 0}
    assert valid_rows[0][0] == "m1"
    assert "action=worker_general" in valid_rows[0][3]
    assert "task_type=math" in valid_rows[0][3]
    assert "question_id=q-1" in valid_rows[0][3]


def test_load_only_ids_reads_newline_allowlist(tmp_path) -> None:
    ids_path = tmp_path / "ids.txt"
    ids_path.write_text("m1\n\nm2\n")

    assert reembed.load_only_ids(ids_path) == {"m1", "m2"}


def test_start_embedding_server_launches_kernel_store_binary_with_a_log(
    tmp_path, monkeypatch
) -> None:
    """A5-06: not the v9 in-tree build/bin binary, and not stdout/stderr to DEVNULL."""
    import subprocess
    from pathlib import Path
    from types import SimpleNamespace

    from src.registry import kernel_paths

    captured = {}

    def fake_popen(cmd, **kwargs):
        captured.update(kwargs, cmd=cmd)
        return SimpleNamespace(poll=lambda: None, kill=lambda: None)

    monkeypatch.setattr(reembed.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(reembed.requests, "get", lambda *a, **k: SimpleNamespace(status_code=200))
    monkeypatch.setattr(kernel_paths, "backend_ld_library_path", lambda backend: [f"/store/{backend}"])
    monkeypatch.setenv("ORCHESTRATOR_PATHS_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("LD_LIBRARY_PATH", "/ambient/lib")

    reembed.start_embedding_server(Path("/m.gguf"), 8199, 2)

    assert captured["cmd"][0] == str(kernel_paths.server_binary("cpu"))
    assert captured["env"]["LD_LIBRARY_PATH"] == "/store/cpu:/ambient/lib"
    assert Path(captured["stdout"].name) == tmp_path / "reembed_8199.log"
    assert captured["stderr"] is subprocess.STDOUT
