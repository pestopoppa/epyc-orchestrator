"""Helper-focused tests for scripts/benchmark/seed_specialist_routing.py."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import pytest


_ROOT = Path(__file__).resolve().parents[2]
_BENCH = _ROOT / "scripts" / "benchmark"
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_BENCH))


def _load_module(name: str):
    spec = importlib.util.spec_from_file_location(name, _BENCH / "seed_specialist_routing.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_signal_handlers_update_shutdown_and_close_paths():
    mod = _load_module("seed_specialist_routing_helpers_sig")
    mod.state.shutdown = False
    close_mock = Mock()
    mod.state.close_poll_client = close_mock

    mod._handle_sigint(None, None)
    assert mod.state.shutdown is True

    with patch.object(sys, "exit", side_effect=SystemExit(1)):
        with pytest.raises(SystemExit):
            mod._handle_sigint(None, None)
    close_mock.assert_called_once()

    mod.state.shutdown = False
    mod._handle_sigterm(None, None)
    assert mod.state.shutdown is True
    assert close_mock.call_count == 2


def test_load_from_dataset_adapter_import_and_success_paths(monkeypatch):
    mod = _load_module("seed_specialist_routing_helpers_adapter")

    import builtins

    orig_import = builtins.__import__

    def _fail_import(name, *args, **kwargs):
        if name == "dataset_adapters":
            raise ImportError("missing")
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fail_import)
    assert mod._load_from_dataset_adapter("suite", 1, 1) == []

    class _Adapter:
        total_available = 10

        @staticmethod
        def sample(n, seed):  # noqa: ANN001
            return [{"id": "q1", "suite": "s", "prompt": "p"}][:n]

    fake = ModuleType("dataset_adapters")
    fake.ADAPTER_SUITES = {"suite", "none"}
    fake.get_adapter = lambda name: _Adapter() if name == "suite" else None
    monkeypatch.setitem(sys.modules, "dataset_adapters", fake)
    monkeypatch.setattr(builtins, "__import__", orig_import)

    out = mod._load_from_dataset_adapter("suite", 1, 1)
    assert out == [{"id": "q1", "suite": "s", "prompt": "p"}]
    assert mod._load_from_dataset_adapter("unknown", 1, 1) == []
    assert mod._load_from_dataset_adapter("none", 1, 1) == []


def test_load_from_yaml_paths(tmp_path, monkeypatch):
    mod = _load_module("seed_specialist_routing_helpers_yaml")
    monkeypatch.setattr(mod, "DEBUG_PROMPTS_DIR", tmp_path)

    import builtins

    orig_import = builtins.__import__

    def _fail_yaml(name, *args, **kwargs):
        if name == "yaml":
            raise ImportError("missing yaml")
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fail_yaml)
    assert mod._load_from_yaml("math", 1, 1) == []
    monkeypatch.setattr(builtins, "__import__", orig_import)

    # Missing file
    assert mod._load_from_yaml("math", 1, 1) == []

    # Empty questions
    (tmp_path / "empty.yaml").write_text("questions: []\n")
    assert mod._load_from_yaml("empty", 1, 1) == []

    # Existing YAML path
    (tmp_path / "math.yaml").write_text(
        "questions:\n"
        "  - id: q1\n"
        "    prompt: 'What is 2+2?'\n"
        "    expected: '4'\n"
        "    image_path: ''\n"
    )
    out = mod._load_from_yaml("math", 2, 1)
    assert len(out) == 1
    assert out[0]["id"] == "q1"
    assert out[0]["dataset_source"] == "yaml"


def test_sample_unseen_questions_pool_fastpath(tmp_path, monkeypatch):
    mod = _load_module("seed_specialist_routing_helpers_pool_fast")
    pool_mod = ModuleType("question_pool")
    pool_mod.POOL_FILE = tmp_path / "pool.json"
    pool_mod.POOL_FILE.write_text("{}")
    pool_mod.build_pool = Mock()
    pool_mod.load_pool = lambda: {"pool": 1}
    pool_mod.sample_from_pool = lambda *a, **k: [{"id": "q1", "suite": "s"}]
    monkeypatch.setitem(sys.modules, "question_pool", pool_mod)

    out = mod.sample_unseen_questions(["s"], 1, set(), seed=1, use_pool=True)
    assert out == [{"id": "q1", "suite": "s"}]


def test_sample_unseen_questions_fallback_interleave_and_seen_filter(monkeypatch):
    mod = _load_module("seed_specialist_routing_helpers_pool_fallback")

    pool_mod = ModuleType("question_pool")
    pool_mod.POOL_FILE = Path("/tmp/nonexistent_pool.json")
    pool_mod.build_pool = Mock(side_effect=RuntimeError("no pool"))
    pool_mod.load_pool = Mock(side_effect=RuntimeError("no pool"))
    pool_mod.sample_from_pool = Mock(return_value=[])
    monkeypatch.setitem(sys.modules, "question_pool", pool_mod)

    def _from_adapter(suite, *_args, **_kwargs):
        if suite == "a":
            return [{"id": "a1", "suite": "a"}, {"id": "a_seen", "suite": "a"}]
        return [{"id": "b1", "suite": "b"}]

    monkeypatch.setattr(mod, "_load_from_dataset_adapter", _from_adapter)
    monkeypatch.setattr(mod, "_load_from_yaml", lambda *_a, **_k: [])

    out = mod.sample_unseen_questions(["a", "b"], 2, {"a_seen"}, seed=1, use_pool=True)
    assert [q["id"] for q in out] == ["a1", "b1"]


def test_sample_unseen_questions_pool_empty_then_yaml_fallback(tmp_path, monkeypatch):
    mod = _load_module("seed_specialist_routing_helpers_pool_empty")
    pool_mod = ModuleType("question_pool")
    pool_mod.POOL_FILE = tmp_path / "pool.json"
    pool_mod.POOL_FILE.write_text("{}")
    pool_mod.build_pool = Mock()
    pool_mod.load_pool = lambda: {"pool": 1}
    pool_mod.sample_from_pool = lambda *a, **k: []
    monkeypatch.setitem(sys.modules, "question_pool", pool_mod)

    monkeypatch.setattr(mod, "_load_from_dataset_adapter", lambda *_a, **_k: [])
    monkeypatch.setattr(
        mod,
        "_load_from_yaml",
        lambda suite, *_a, **_k: [{"id": f"{suite}_y1", "suite": suite}],
    )
    out = mod.sample_unseen_questions(["a"], 1, set(), seed=1, use_pool=True)
    assert out == [{"id": "a_y1", "suite": "a"}]


def test_run_batch_3way_health_and_empty_paths():
    mod = _load_module("seed_specialist_routing_helpers_batch")
    with patch.object(mod, "_check_server_health", return_value=False):
        with pytest.raises(mod.HealthCheckError):
            mod.run_batch_3way(
                suites=["s"],
                sample_per_suite=1,
                seed=1,
                url="http://localhost:8000",
                timeout=30,
                session_id="sess",
            )

    with (
        patch.object(mod, "_check_server_health", return_value=True),
        patch.object(mod, "load_seen_questions", return_value=set()),
        patch.object(mod, "sample_unseen_questions", return_value=[]),
    ):
        out = mod.run_batch_3way(
            suites=["s"],
            sample_per_suite=1,
            seed=1,
            url="http://localhost:8000",
            timeout=30,
            session_id="sess",
        )
    assert out == []


def test_print_3way_summary_no_results_and_with_data(capsys):
    mod = _load_module("seed_specialist_routing_helpers_summary")
    mod.print_3way_summary([])
    assert "No results to summarize." in capsys.readouterr().out

    rr_pass = SimpleNamespace(passed=True)
    rr_fail = SimpleNamespace(passed=False)
    result = mod.ThreeWayResult(
        suite="s",
        question_id="q1",
        prompt="p",
        expected="e",
        role_results={"frontdoor:direct": rr_pass, "frontdoor:repl": rr_fail},
        rewards={
            mod.ACTION_SELF_DIRECT: 1.0,
            mod.ACTION_SELF_REPL: 0.0,
            mod.ACTION_ARCHITECT: 1.0,
            mod.ACTION_WORKER: 1.0,
        },
        metadata={"tools_helped": True},
        rewards_injected=3,
    )
    mod.print_3way_summary([result])
    out = capsys.readouterr().out
    assert "3-WAY ROUTING EVALUATION SUMMARY" in out
    assert "Tools helped: 1" in out
    assert "Rewards injected: 3" in out


def test_retrieval_config_and_profile_application(monkeypatch):
    mod = _load_module("seed_specialist_routing_helpers_profile")
    args = SimpleNamespace(
        profile="infra-stable",
        cooldown=None,
        timeout=None,
        cost_lambda=0.3,
        confidence_threshold=None,
        confidence_estimator=None,
        confidence_trim_ratio=None,
        confidence_min_neighbors=None,
        warm_probability_hit=None,
        warm_probability_miss=None,
        warm_cost_fallback_s=None,
        cold_cost_fallback_s=None,
        calibrated_confidence_threshold=None,
        conformal_margin=None,
        risk_control_enabled=None,
        risk_budget_id=None,
        risk_gate_min_samples=None,
        risk_abstain_target_role=None,
        risk_gate_rollout_ratio=None,
        risk_gate_kill_switch=None,
        risk_budget_guardrail_min_events=None,
        risk_budget_guardrail_max_abstain_rate=None,
        prior_strength=None,
    )

    cfg = mod._build_retrieval_config_from_args(args)
    assert getattr(cfg, "cost_lambda") == 0.3

    monkeypatch.delenv("ORCHESTRATOR_DEFERRED_TOOL_RESULTS", raising=False)
    mod._apply_profile(args)
    assert args.cooldown == 2.0
    assert isinstance(args.timeout, int)
    assert os.environ.get("ORCHESTRATOR_DEFERRED_TOOL_RESULTS") == "1"
