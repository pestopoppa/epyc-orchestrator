"""Consumer-level tests for reward delivery propagation."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch


_ROOT = Path(__file__).resolve().parents[2]
_BENCH = _ROOT / "scripts" / "benchmark"
_AUTO = _ROOT / "scripts" / "autopilot" / "species"
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_BENCH))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_run_batch_3way_propagates_reward_delivery_summary():
    mod = _load_module("seed_specialist_routing_test", _BENCH / "seed_specialist_routing.py")
    prompt = {"id": "q1", "suite": "general", "prompt": "2+2?", "expected": "4"}
    role_results = {"SELF:direct": SimpleNamespace(passed=True)}
    delivery = {"submitted": 2, "acknowledged": 1, "failed": 1, "failure_reasons": {"ARCHITECT": "http_503"}}

    with (
        patch.object(mod, "_check_server_health", return_value=True),
        patch.object(mod, "evaluate_question_3way", return_value=(role_results, {"SELF:direct": 1.0}, {})),
        patch.object(mod, "_inject_3way_rewards_http", return_value=delivery),
        patch.object(mod, "checkpoint_result") as checkpoint_result,
        patch.object(mod, "record_seen") as record_seen,
        patch("httpx.Client", return_value=MagicMock()),
    ):
        mod.state.shutdown = False
        results = mod.run_batch_3way(
            suites=["general"],
            sample_per_suite=1,
            seed=1,
            url="http://localhost:8000",
            timeout=30,
            session_id="sess",
            questions_override=[prompt],
        )

    assert len(results) == 1
    assert results[0].rewards_delivery == delivery
    assert results[0].rewards_injected == 1
    checkpoint_result.assert_called_once()
    record_seen.assert_called_once_with("q1", "general", "sess")


def test_seeder_run_batch_accumulates_acknowledged_rewards():
    mod = _load_module("species_seeder_test", _AUTO / "seeder.py")
    question = {"id": "q1", "suite": "general", "prompt": "2+2?", "expected": "4"}
    role_results = {"SELF:direct": SimpleNamespace(passed=True)}
    delivery = {"submitted": 3, "acknowledged": 2, "failed": 1, "failure_reasons": {"ARCHITECT": "http_503"}}
    client_cm = MagicMock()
    client = Mock()
    client_cm.__enter__.return_value = client
    client_cm.__exit__.return_value = False

    with (
        patch.object(mod, "sample_unseen_questions", return_value=[question]),
        patch.object(mod, "evaluate_question_3way", return_value=(role_results, {"SELF:direct": 1.0}, {"avg_td_error": 0.2})),
        patch.object(mod, "_inject_3way_rewards_http", return_value=delivery),
        patch.object(mod.Seeder, "_get_memory_count", return_value=0),
        patch("httpx.Client", return_value=client_cm),
    ):
        seeder = mod.Seeder(url="http://localhost:8000", timeout=30, batch_size=1, dry_run=False)
        result = seeder.run_batch(n_questions=1, suites=["general"], seed=1)

    assert result.rewards_injected == 2
    assert result.rewards_delivery == [delivery]
    assert result.results[0].rewards_delivery == delivery
    assert result.results[0].rewards_injected == 2
