"""HAR-COMPACT-LEDGER-1: failed-action ledger + pinned-pointer restore across compaction.

Replay over five retained-trajectory shapes: every failed action line and every pinned pointer
(path, ID, number) of the pre-compaction context must be present byte-exact afterwards, across
two consecutive compactions, and the flag must default off.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.graph.compaction import (
    LEDGER_HEADER,
    _maybe_compact_context,
    apply_integrity,
    extract_failed_actions,
    extract_pinned_tokens,
    survivors_missing,
)

# (name, text, failed-action lines, pinned tokens) - each trajectory is long enough that the
# 20% keep-recent tail excludes the early failures.
FILLER = "\n".join(f"turn note {i}: reviewed output, nothing notable" for i in range(400))

TRAJECTORIES = {
    "shell_fail": (
        "$ pytest /mnt/raid0/llm/epyc-orchestrator/tests/unit/test_a.py\n"
        "exit code 2\n"
        "FileNotFoundError: /srv/data/run-7731/config.yaml\n"
        "commit 3f9a1c2b7d04 introduced it, box HAR-COMPACT-LEDGER-1, 1048576 bytes\n"
    ),
    "python_traceback": (
        "Traceback (most recent call last):\n"
        '  File "/workspace/src/x.py", line 12, in <module>\n'
        "ValueError: bad shape 4096 vs 2048\n"
    ),
    "tool_error": (
        "[tool read_file] error: permission denied /etc/shadow.bak\n"
        "retry with id 123e4567-e89b-12d3-a456-426614174000 gave rc=1\n"
    ),
    "numbers": (
        "server answered 503 on try 3; elapsed 12.75 s, port 8083\n"
        "Error: upstream timeout after 30000 ms\n"
    ),
    "clean": "all good, wrote /tmp/out/result_9921.json\n",
}


def _traj(name: str) -> str:
    return TRAJECTORIES[name] + FILLER + "\n" + "tail line without anything\n" * 200


@pytest.mark.parametrize("name", sorted(TRAJECTORIES))
def test_replay_survives_byte_exact(name):
    old = _traj(name)
    # a lossy compaction: only the recent tail plus a generic index survives
    compacted = "[Context Index]\n- Externalized.\n\n[Recent Context]\n" + old[-3000:]
    out = apply_integrity(old, compacted)
    for entry in extract_failed_actions(old):
        assert entry in out
    for tok in extract_pinned_tokens(old):
        assert tok in out
    assert survivors_missing(old, out) == ([], [])


def test_expected_extractions():
    old = _traj("shell_fail")
    fails = extract_failed_actions(old)
    assert "exit code 2" in fails
    assert "FileNotFoundError: /srv/data/run-7731/config.yaml" in fails
    toks = extract_pinned_tokens(old)
    for expected in (
        "/mnt/raid0/llm/epyc-orchestrator/tests/unit/test_a.py",
        "/srv/data/run-7731/config.yaml",
        "3f9a1c2b7d04",
        "HAR-COMPACT-LEDGER-1",
        "1048576",
    ):
        assert expected in toks
    assert "123e4567-e89b-12d3-a456-426614174000" in extract_pinned_tokens(_traj("tool_error"))
    assert "12.75" in extract_pinned_tokens(_traj("numbers"))
    assert extract_failed_actions(_traj("clean")) == []


def test_idempotent_when_nothing_missing():
    old = _traj("shell_fail")
    assert apply_integrity(old, old) == old
    once = apply_integrity(old, "short")
    assert apply_integrity(old, once) == once


def test_ledger_carries_across_two_compactions():
    old1 = _traj("shell_fail")
    c1 = apply_integrity(old1, "[Recent Context]\n" + old1[-500:])
    assert LEDGER_HEADER in c1
    old2 = c1 + "\n" + _traj("python_traceback")
    c2 = apply_integrity(old2, "[Recent Context]\n" + old2[-500:])
    for src in (old1, _traj("python_traceback")):
        for entry in extract_failed_actions(src):
            assert entry in c2


def test_blocks_are_bounded():
    old = "\n".join(f"error: failure number {i}" for i in range(500))
    assert len(extract_failed_actions(old)) == 50


# --- extractor gaps found by the retained-trajectory replay (scripts/har_compact_ledger_replay.py) ---
# Snippets are verbatim from data/bep_sandbox/*/traces/*.jsonl (TOC-SP-5 corpus).

REAL_LOOP_HALT = (
    "** LOOP HALTED ** You just emitted the SAME action with no progress. STOP repeating it. "
    "Do NOT peek/read again - you already have what you need."
)


def test_real_loop_halted_is_a_failed_action():
    assert extract_failed_actions("noise\n" + REAL_LOOP_HALT + "\nnoise") == [REAL_LOOP_HALT]


@pytest.mark.parametrize(
    "snippet, expected",
    [
        ("peek(99999, file_path='/shapes.py')", "/shapes.py"),  # root-level absolute path
        ('peek(1000, file_path="tax.py")', "tax.py"),  # bare file name
        ('{"path": "pkg/new.py", "operation": "create"}', "pkg/new.py"),  # relative path
        ('"postconditions": ["pytest tests/unit/test_mod.py passes"]', "tests/unit/test_mod.py"),
        ('peek(99999, file_path="/mnt/raid0/llm/tmp/report.py")', "/mnt/raid0/llm/tmp/report.py"),
    ],
)
def test_real_path_shapes_are_pinned(snippet, expected):
    assert expected in extract_pinned_tokens(snippet)


def test_url_tail_is_not_a_path():
    toks = extract_pinned_tokens("GET http://localhost:8102/v1/models.json ok")
    assert "models.json" not in toks and "/v1/models.json" not in toks


def test_real_path_and_halt_survive_compaction():
    old = "peek(99999, file_path='/shapes.py')\n" + REAL_LOOP_HALT + "\n" + FILLER + "\n" + "tail line\n" * 400
    out = apply_integrity(old, "[Recent Context]\n" + old[-500:])
    assert REAL_LOOP_HALT in out and "/shapes.py" in out


def test_flag_defaults_off():
    from src.features import Features

    assert Features().compaction_failed_action_ledger is False


def _ctx(context: str):
    state = SimpleNamespace(
        task_id="t1", context=context, turns=10, compaction_count=0, last_compaction_turn=0,
        context_file_paths=[], compaction_tokens_saved=0, aggregate_tokens=0, current_role="worker",
    )
    primitives = SimpleNamespace(llm_call=lambda *a, **k: "idx", registry=None, server_urls={})
    return SimpleNamespace(state=state, deps=SimpleNamespace(primitives=primitives, repl=None))


def _run(flag: bool, tmp_path, monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_PATHS_TMP_DIR", str(tmp_path))
    old = _traj("shell_fail")
    ctx = _ctx(old)
    feats = SimpleNamespace(
        session_compaction=True, session_compaction_llm_index=False, context_arm_ab=False,
        session_token_budget=False, compaction_failed_action_ledger=flag,
    )
    with patch("src.features.features", return_value=feats):
        asyncio.run(_maybe_compact_context(ctx, force=True))
    return old, ctx.state.context


def test_integration_flag_on_preserves(tmp_path, monkeypatch):
    old, new = _run(True, tmp_path, monkeypatch)
    assert new != old
    assert survivors_missing(old, new) == ([], [])


def test_integration_flag_off_is_unchanged_behaviour(tmp_path, monkeypatch):
    old, new = _run(False, tmp_path, monkeypatch)
    assert LEDGER_HEADER not in new
    assert "[Pinned Pointers]" not in new
