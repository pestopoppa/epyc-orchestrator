"""EV-5's opt-in T2 path owns exactly the generator and verifier it launched."""

import json
import subprocess
import sys

import pytest

from scripts.server import eval_process_verifier as native
from scripts.server.eval_process_verifier import OwnedT2Verifier, ProcessVerifierConfig


def _private_child():
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])


def test_owned_generation_drains_before_verifier_load_and_writes_receipt(tmp_path, monkeypatch):
    child = _private_child()
    order = []

    class FakeVerifier:
        def __init__(self, config):
            assert config.generation_drained
            self.lifecycle = {"drained": False}

        def __enter__(self):
            assert child.poll() is not None
            order.append("verifier_load")
            return self

        def __exit__(self, *_):
            self.lifecycle["drained"] = True
            order.append("verifier_drain")

        def verify(self, *_):
            return {"steps": []}

    monkeypatch.setattr(native, "NativeProcessVerifier", FakeVerifier)
    config = ProcessVerifierConfig(
        tmp_path / "model",
        tmp_path / "server",
        tmp_path,
        tmp_path / "verifier.log",
        generation_drained=True,
    )
    receipt_path = tmp_path / "generation-drained.json"
    try:
        session = OwnedT2Verifier(child, config=config, receipt_path=receipt_path)
        with session:
            assert session.verify("problem", "solution") == {"steps": []}
        session.finalize()  # Idempotent T2 end boundary.
        assert order == ["verifier_load", "verifier_drain"]
        assert session.lifecycle["generator_drained"] is True
        assert session.lifecycle["verifier_drained"] is True
        receipt = json.loads(receipt_path.read_text())
        assert receipt["generator_pid"] == child.pid
        assert receipt["verified_dead"] is True
        assert native._proc_identity(child.pid) != session.identity
    finally:
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=5)


def test_foreign_handle_refused_before_any_signal(tmp_path, monkeypatch):
    child = _private_child()
    original = native._proc_identity

    def foreign_identity(pid):
        parent_pid, ticks = original(pid)
        return parent_pid + 1, ticks

    monkeypatch.setattr(native, "_proc_identity", foreign_identity)
    config = ProcessVerifierConfig(
        tmp_path / "model",
        tmp_path / "server",
        tmp_path,
        tmp_path / "verifier.log",
        generation_drained=True,
    )
    try:
        with pytest.raises(ValueError, match="live direct child"):
            OwnedT2Verifier(child, config=config, receipt_path=tmp_path / "receipt")
        assert child.poll() is None
    finally:
        child.terminate()
        child.wait(timeout=5)


def test_real_t2_hook_drains_owned_child_on_cross_family_rejection(tmp_path, monkeypatch):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/autopilot"))
    from eval_tower import EvalTower, QuestionResult

    child = _private_child()
    config = ProcessVerifierConfig(
        tmp_path / "model",
        tmp_path / "server",
        tmp_path,
        tmp_path / "verifier.log",
        generation_drained=True,
    )
    try:
        session = OwnedT2Verifier(child, config=config, receipt_path=tmp_path / "receipt")
        tower = EvalTower()
        tower.set_question_artifact_dir(tmp_path)
        question = {"id": "a", "suite": "math", "prompt": "2+2?", "expected": "4"}
        monkeypatch.setattr(tower, "_load_pool", lambda: {"math": [question]})
        monkeypatch.setattr(tower, "_t1_core_exclusion_qids", lambda *a, **k: (set(), {}))
        monkeypatch.setattr(tower, "_load_tool_sentinels", lambda: [])
        monkeypatch.setattr(
            tower,
            "_eval_batch",
            lambda *_a, **_k: [
                QuestionResult(
                    "a",
                    "math",
                    "2+2?",
                    "4",
                    qid="a",
                    answer="4",
                    route_used="Qwen3-35B",
                    confidence=0.5,
                    disposition="scored",
                )
            ],
        )
        with pytest.raises(ValueError, match="same_family"):
            tower.eval_t2(n=1, process_verifier=session, verification_n=1)
        assert child.poll() is not None
        assert session.lifecycle["generator_drained"] is True
        assert session.lifecycle["verifier_drained"] is True
        assert (tmp_path / "receipt").is_file()
    finally:
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=5)


def test_real_t2_hook_drains_owned_child_on_batch_failure(tmp_path, monkeypatch):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/autopilot"))
    from eval_tower import EvalTower

    child = _private_child()
    config = ProcessVerifierConfig(
        tmp_path / "model",
        tmp_path / "server",
        tmp_path,
        tmp_path / "verifier.log",
        generation_drained=True,
    )
    try:
        session = OwnedT2Verifier(child, config=config, receipt_path=tmp_path / "receipt")
        tower = EvalTower()
        question = {"id": "a", "suite": "math", "prompt": "2+2?", "expected": "4"}
        monkeypatch.setattr(tower, "_load_pool", lambda: {"math": [question]})
        monkeypatch.setattr(tower, "_t1_core_exclusion_qids", lambda *a, **k: (set(), {}))
        monkeypatch.setattr(tower, "_load_tool_sentinels", lambda: [])

        def fail_batch(*_a, **_k):
            raise RuntimeError("private generation failed")

        monkeypatch.setattr(tower, "_eval_batch", fail_batch)
        with pytest.raises(RuntimeError, match="private generation failed"):
            tower.eval_t2(n=1, process_verifier=session)
        assert child.poll() is not None
        assert session.lifecycle["generator_drained"] is True
        assert (tmp_path / "receipt").is_file()
    finally:
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=5)


@pytest.mark.parametrize("failure", ["empty_pool", "artifact_root"])
def test_whole_t2_boundary_drains_on_early_and_artifact_failure(tmp_path, monkeypatch, failure):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/autopilot"))
    import eval_tower

    child = _private_child()
    config = ProcessVerifierConfig(
        tmp_path / "model",
        tmp_path / "server",
        tmp_path,
        tmp_path / "verifier.log",
        generation_drained=True,
    )
    try:
        session = OwnedT2Verifier(child, config=config, receipt_path=tmp_path / "receipt")
        tower = eval_tower.EvalTower()
        if failure == "empty_pool":
            monkeypatch.setattr(tower, "_load_pool", lambda: {})
            result = tower.eval_t2(process_verifier=session)
            assert result.quality_measured is False
        else:
            question = {"id": "a", "suite": "math", "prompt": "2+2?", "expected": "4"}
            monkeypatch.setattr(tower, "_load_pool", lambda: {"math": [question]})
            monkeypatch.setattr(tower, "_t1_core_exclusion_qids", lambda *a, **k: (set(), {}))
            monkeypatch.setattr(tower, "_load_tool_sentinels", lambda: [])
            monkeypatch.setattr(tower, "_eval_batch", lambda *_a, **_k: [])

            def fail_artifact_root(*_a, **_k):
                raise RuntimeError("artifact root unavailable")

            monkeypatch.setattr(eval_tower, "_eval_artifact_root", fail_artifact_root)
            with pytest.raises(RuntimeError, match="artifact root unavailable"):
                tower.eval_t2(n=1, process_verifier=session)
        assert child.poll() is not None
        assert session.lifecycle["generator_drained"] is True
        assert (tmp_path / "receipt").is_file()
    finally:
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=5)
