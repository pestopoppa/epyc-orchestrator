"""Unit tests for inference mixin in LLMPrimitives.

Tests InferenceMixin methods through LLMPrimitives integration:
- _real_call() with CachingBackend and ModelServer paths
- _call_caching_backend() with circuit breaker
- _real_batch() with ThreadPoolExecutor
- Error handling (connection errors, timeouts)
"""

from unittest.mock import Mock, patch

import pytest

from src.llm_primitives import LLMPrimitives
from src.llm_primitives.inference import _extract_port, _primary_url, _sampling_cache_key
from src.config import reset_config
from src.model_server import InferenceRequest, InferenceResult
from src.registry_loader import (
    AccelerationConfig,
    GenerationDefaults,
    MemoryConfig,
    ModelConfig,
    PerformanceMetrics,
    RoleConfig,
)


@pytest.fixture
def mock_backend():
    """Create a mock backend that implements the caching interface (no streaming)."""
    backend = Mock(spec=[])  # empty spec prevents auto-creating infer_stream_text
    backend.infer = Mock()
    return backend


@pytest.fixture
def mock_model_server():
    """Create a mock ModelServer (batch-only, no streaming)."""
    server = Mock(spec=[])  # empty spec prevents auto-creating attributes
    server.infer = Mock()
    return server


@pytest.fixture
def mock_health_tracker():
    """Create a mock BackendHealthTracker."""
    tracker = Mock()
    tracker.is_available = Mock(return_value=True)
    tracker.record_success = Mock()
    tracker.record_failure = Mock()
    return tracker


@pytest.fixture(autouse=True)
def writable_tmp_dir(monkeypatch, tmp_path):
    """Ensure inference lock uses a writable tmp dir in test sandbox."""
    monkeypatch.setenv("ORCHESTRATOR_PATHS_TMP_DIR", str(tmp_path))
    monkeypatch.delenv("INFERENCE_TAP_FILE", raising=False)
    monkeypatch.setattr("src.inference_tap._read_sentinel", lambda: "")
    monkeypatch.setattr("src.inference_tap._sentinel_cache", ("", 0.0))
    reset_config()
    yield
    reset_config()


class TestInferenceMixinRealCall:
    """Tests for _real_call() method."""

    def test_real_call_with_caching_backend(self, mock_backend, mock_health_tracker):
        """Test _real_call uses CachingBackend when available."""
        # Create LLMPrimitives with backend
        prims = LLMPrimitives(
            mock_mode=False,
            server_urls={"test_role": "http://localhost:8080"},
            health_tracker=mock_health_tracker,
        )
        prims._backends["test_role"] = mock_backend

        # Mock successful inference
        mock_backend.infer.return_value = InferenceResult(
            role="test_role",
            output="Backend response",
            tokens_generated=10,
            generation_speed=25.0,
            elapsed_time=0.4,
            success=True,
            prompt_eval_ms=100.0,
            generation_ms=300.0,
            predicted_per_second=25.0,
            http_overhead_ms=10.0,
        )

        result = prims._real_call("Test prompt", "test_role", n_tokens=128)

        assert result == "Backend response"
        mock_backend.infer.assert_called_once()
        assert prims.total_tokens_generated == 10
        assert prims.total_prompt_eval_ms == 100.0
        assert prims.total_generation_ms == 300.0
        assert prims.total_http_overhead_ms == 10.0

    def test_real_call_fallback_to_model_server(self, mock_model_server):
        """Test _real_call falls back to ModelServer when no backend."""
        prims = LLMPrimitives(
            mock_mode=False,
            model_server=mock_model_server,
        )

        # Mock successful inference
        mock_model_server.infer.return_value = InferenceResult(
            role="test_role",
            output="Server response",
            tokens_generated=15,
            generation_speed=30.0,
            elapsed_time=0.5,
            success=True,
        )

        result = prims._real_call(
            "Test prompt",
            "test_role",
            n_tokens=256,
            temperature=0.2,
            seed=1234,
            top_p=0.8,
            top_k=64,
        )

        assert result == "Server response"
        mock_model_server.infer.assert_called_once()
        # Check request structure
        call_args = mock_model_server.infer.call_args
        assert call_args[0][0] == "test_role"  # First positional arg is role
        request = call_args[0][1]
        assert isinstance(request, InferenceRequest)
        assert request.n_tokens == 256
        assert request.temperature == 0.2
        assert request.seed == 1234
        assert request.top_p == 0.8
        assert request.top_k == 64

    def test_real_call_records_inference_meta_for_non_frontdoor_model_server(self, mock_model_server):
        """Specialist roles should also publish timing metadata."""
        prims = LLMPrimitives(
            mock_mode=False,
            model_server=mock_model_server,
        )

        mock_model_server.infer.return_value = InferenceResult(
            role="coder_escalation",
            output="ok",
            tokens_generated=7,
            generation_speed=20.0,
            elapsed_time=0.2,
            success=True,
            prompt_eval_ms=11.0,
            generation_ms=22.0,
            http_overhead_ms=3.0,
        )

        prims._real_call("Test prompt", "coder_escalation", n_tokens=64)

        meta = getattr(prims, "_last_inference_meta", {})
        assert meta["role"] == "coder_escalation"
        assert meta["transport"] == "model_server"
        assert meta["prompt_ms"] == 11.0
        assert meta["gen_ms"] == 22.0
        assert meta["completion_reason"] == "unknown"

    def test_real_call_no_backend_raises_error(self):
        """Test _real_call raises error when no backend configured."""
        prims = LLMPrimitives(mock_mode=False)

        with pytest.raises(RuntimeError, match="No backend configured"):
            prims._real_call("Test prompt", "unknown_role")


class TestCallCachingBackend:
    """Tests for _call_caching_backend() method."""

    def test_primary_url_strips_full_marker(self):
        """The full-speed marker is config metadata, not part of the backend URL."""
        assert (
            _primary_url("full:http://localhost:8072,http://localhost:8082")
            == "http://localhost:8072"
        )
        assert _extract_port("full:http://localhost:8072,http://localhost:8082") == 8072

    def test_call_caching_backend_success(self, mock_backend, mock_health_tracker):
        """Test successful call to caching backend."""
        prims = LLMPrimitives(
            mock_mode=False,
            server_urls={"coder": "http://localhost:8081"},
            health_tracker=mock_health_tracker,
        )

        mock_backend.infer.return_value = InferenceResult(
            role="coder",
            output="def hello(): pass",
            tokens_generated=20,
            generation_speed=40.0,
            elapsed_time=0.5,
            success=True,
            predicted_per_second=40.0,
        )

        result = prims._call_caching_backend(
            mock_backend, "Write hello function", "coder", n_tokens=128
        )

        assert result == "def hello(): pass"
        assert prims.total_tokens_generated == 20
        mock_health_tracker.record_success.assert_called_once_with("http://localhost:8081")

    def test_call_caching_backend_uses_registry_role_config(self, mock_backend):
        """CachingBackend must not replace registry generation defaults with greedy."""
        role_config = RoleConfig(
            name="coder",
            tier="C",
            description="Coder",
            model=ModelConfig(
                name="coder-model",
                path="",
                quant="Q4_K_M",
                size_gb=0.0,
            ),
            acceleration=AccelerationConfig(type="baseline", temperature=None),
            performance=PerformanceMetrics(),
            memory=MemoryConfig(residency="warm"),
            generation_defaults=GenerationDefaults(temperature=0.3),
        )
        registry = Mock()
        registry.get_role.return_value = role_config
        prims = LLMPrimitives(mock_mode=False, registry=registry)

        def infer_with_config(config, request):
            assert config is role_config
            assert request.temperature is None
            return InferenceResult(
                role="coder",
                output="ok",
                tokens_generated=1,
                generation_speed=1.0,
                elapsed_time=0.1,
                success=True,
            )

        mock_backend.infer.side_effect = infer_with_config

        assert prims._call_caching_backend(mock_backend, "Prompt", "coder") == "ok"

    def test_call_caching_backend_forwards_sampling_params(self, mock_backend):
        prims = LLMPrimitives(mock_mode=False)
        mock_backend.infer.return_value = InferenceResult(
            role="coder",
            output="ok",
            tokens_generated=1,
            generation_speed=1.0,
            elapsed_time=0.1,
            success=True,
        )

        prims._call_caching_backend(
            mock_backend,
            "Prompt",
            "coder",
            temperature=0.2,
            seed=1234,
            top_p=0.8,
            top_k=64,
        )

        request = mock_backend.infer.call_args[0][1]
        assert request.temperature == 0.2
        assert request.seed == 1234
        assert request.top_p == 0.8
        assert request.top_k == 64

    def test_sampling_cache_key_only_includes_explicit_params(self):
        assert _sampling_cache_key() == ""
        assert _sampling_cache_key(temperature=0.0, seed=7, top_p=0.8, top_k=64) == (
            '{"seed":7,"temperature":0.0,"top_k":64,"top_p":0.8}'
        )

    def test_call_caching_backend_uses_concrete_url_for_full_speed_role(
        self, mock_backend, mock_health_tracker
    ):
        """Circuit tracking uses the live endpoint, not the full-speed config marker."""
        prims = LLMPrimitives(
            mock_mode=False,
            server_urls={"worker_general": "full:http://localhost:8072,http://localhost:8082"},
            health_tracker=mock_health_tracker,
        )

        mock_backend.infer.return_value = InferenceResult(
            role="worker_general",
            output="ok",
            tokens_generated=1,
            generation_speed=1.0,
            elapsed_time=0.1,
            success=True,
        )

        result = prims._call_caching_backend(
            mock_backend, "Test prompt", "worker_general", n_tokens=16
        )

        assert result == "ok"
        mock_health_tracker.is_available.assert_called_once_with("http://localhost:8072")
        mock_health_tracker.record_success.assert_called_once_with("http://localhost:8072")

    def test_call_caching_backend_records_inference_meta_for_non_frontdoor(
        self, mock_backend, mock_health_tracker
    ):
        """Caching backend path should publish timing metadata for specialist roles."""
        prims = LLMPrimitives(
            mock_mode=False,
            server_urls={"worker_fast": "http://localhost:8082"},
            health_tracker=mock_health_tracker,
        )

        mock_backend.infer.return_value = InferenceResult(
            role="worker_fast",
            output="done",
            tokens_generated=12,
            generation_speed=30.0,
            elapsed_time=0.4,
            success=True,
            prompt_eval_ms=9.0,
            generation_ms=31.0,
            http_overhead_ms=2.0,
            completion_reason="stop",
        )

        prims._call_caching_backend(
            mock_backend, "Do work", "worker_fast", n_tokens=96
        )

        meta = getattr(prims, "_last_inference_meta", {})
        assert meta["role"] == "worker_fast"
        assert meta["transport"] == "batch"
        assert meta["prompt_ms"] == 9.0
        assert meta["gen_ms"] == 31.0
        assert meta["completion_reason"] == "stop"

    def test_call_caching_backend_circuit_breaker_open(self, mock_backend, mock_health_tracker):
        """Test circuit breaker prevents call to unhealthy backend."""
        prims = LLMPrimitives(
            mock_mode=False,
            server_urls={"coder": "http://localhost:8081"},
            health_tracker=mock_health_tracker,
        )

        # Simulate circuit open
        mock_health_tracker.is_available.return_value = False

        with pytest.raises(RuntimeError, match="circuit open"):
            prims._call_caching_backend(mock_backend, "Test prompt", "coder", n_tokens=128)

        # Should not call backend
        mock_backend.infer.assert_not_called()

    def test_call_caching_backend_inference_failure(self, mock_backend, mock_health_tracker):
        """Test backend records failure when inference fails."""
        prims = LLMPrimitives(
            mock_mode=False,
            server_urls={"coder": "http://localhost:8081"},
            health_tracker=mock_health_tracker,
        )

        mock_backend.infer.return_value = InferenceResult(
            role="coder",
            output="",
            tokens_generated=0,
            generation_speed=0.0,
            elapsed_time=1.0,
            success=False,
            error_message="Model crashed",
        )

        with pytest.raises(RuntimeError, match="Inference failed"):
            prims._call_caching_backend(mock_backend, "Test prompt", "coder", n_tokens=128)

        mock_health_tracker.record_failure.assert_called_once_with("http://localhost:8081")

    def test_call_caching_backend_with_stop_sequences(self, mock_backend):
        """Test backend call with stop sequences."""
        prims = LLMPrimitives(mock_mode=False)

        mock_backend.infer.return_value = InferenceResult(
            role="worker",
            output="Output",
            tokens_generated=5,
            generation_speed=20.0,
            elapsed_time=0.25,
            success=True,
        )

        result = prims._call_caching_backend(
            mock_backend,
            "Test",
            "worker",
            n_tokens=64,
            stop_sequences=["END", "STOP"],
        )

        assert result == "Output"
        # Check that stop_sequences were passed
        call_args = mock_backend.infer.call_args
        request = call_args[0][1]
        assert request.stop_sequences == ["END", "STOP"]

    def test_call_caching_backend_locks_direct_single_instance_under_per_region(
        self, mock_backend, monkeypatch, tmp_path
    ):
        """Direct single-instance roles must still hold cpu_region_lock at idx=0."""
        monkeypatch.setenv("ORCHESTRATOR_PER_REGION_LOCKS", "1")
        monkeypatch.setenv("INFERENCE_TAP_FILE", str(tmp_path / "inference_tap.log"))
        monkeypatch.setattr(
            "src.runtime.instance_topology.get_instance_regions",
            lambda: {
                ("architect_general", 0): frozenset({"q0", "q1", "q2", "q3"}),
            },
        )

        lock_calls = []
        lock_active = []

        class FakeRegionLock:
            def __enter__(self):
                lock_active.append(True)
                return {}

            def __exit__(self, exc_type, exc, tb):
                lock_active.pop()
                return False

        def fake_region_lock(role, instance_idx, **kwargs):
            lock_calls.append((role, instance_idx, kwargs))
            return FakeRegionLock()

        monkeypatch.setattr(
            "src.runtime.cpu_region_lock.cpu_region_lock_for_instance",
            fake_region_lock,
        )

        def infer_while_locked(_role_config, _request):
            assert lock_active == [True]
            return InferenceResult(
                role="architect_general",
                output="done",
                tokens_generated=3,
                generation_speed=10.0,
                elapsed_time=0.3,
                success=True,
                prompt_eval_ms=4.0,
                generation_ms=300.0,
            )

        mock_backend.infer.side_effect = infer_while_locked
        prims = LLMPrimitives(
            mock_mode=False,
            server_urls={"architect_general": "http://localhost:8083"},
        )

        result = prims._call_caching_backend(
            mock_backend,
            "question",
            "architect_general",
            n_tokens=32,
        )

        assert result == "done"
        assert [(role, idx) for role, idx, _kwargs in lock_calls] == [
            ("architect_general", 0)
        ]
        assert lock_calls[0][2]["request_tag"] is None

        events_path = tmp_path / "inference_tap_events.jsonl"
        events = [line for line in events_path.read_text().splitlines() if line.strip()]
        assert events, "tap events should include direct lock metadata"
        first = __import__("json").loads(events[0])
        assert first["event"] == "start"
        assert first["lock_role"] == "architect_general"
        assert first["instance_idx"] == 0
        assert first["instance_shape"] == "full"
        assert first["instance_regions"] == ["q0", "q1", "q2", "q3"]

    def test_call_caching_backend_does_not_double_lock_concurrency_aware_backend(
        self, monkeypatch
    ):
        """CAB owns per-instance locking internally; the mixin must not wrap it."""
        monkeypatch.setenv("ORCHESTRATOR_PER_REGION_LOCKS", "1")

        def fail_region_lock(*_args, **_kwargs):
            raise AssertionError("ConcurrencyAwareBackend should manage its own lock")

        monkeypatch.setattr(
            "src.runtime.cpu_region_lock.cpu_region_lock_for_instance",
            fail_region_lock,
        )

        class FakeConcurrencyAwareBackend:
            _dispatch = object()
            _tap_dispatch_metadata = object()

            def infer(self, _role_config, _request):
                return InferenceResult(
                    role="frontdoor",
                    output="ok",
                    tokens_generated=1,
                    generation_speed=10.0,
                    elapsed_time=0.1,
                    success=True,
                )

        backend = FakeConcurrencyAwareBackend()
        prims = LLMPrimitives(
            mock_mode=False,
            server_urls={"frontdoor": "full:http://localhost:8070,http://localhost:8080"},
        )

        assert prims._call_caching_backend(backend, "hi", "frontdoor") == "ok"

    def test_shape_aware_concurrency_backend_defers_contention_gate_to_dispatch(
        self, monkeypatch
    ):
        """When B is armed, the pre-dispatch role-keyed gate must not mask the
        candidate-aware gate inside ConcurrencyAwareBackend._dispatch."""
        monkeypatch.setenv("ORCHESTRATOR_PER_REGION_LOCKS", "1")
        monkeypatch.setenv("ORCHESTRATOR_CROSS_ROLE_DISJOINT_PLACEMENT", "1")
        monkeypatch.setenv("ORCHESTRATOR_SHAPE_AWARE_CONTENTION", "1")

        class FailGate:
            def admit(self, *_args, **_kwargs):
                raise AssertionError("coarse pre-dispatch gate should be deferred")

        monkeypatch.setattr("src.scheduling.contention_gate.get_gate", lambda: FailGate())

        class FakeConcurrencyAwareBackend:
            _dispatch = object()
            _tap_dispatch_metadata = object()

            def infer(self, _role_config, request):
                assert getattr(request, "request_priority") == "background"
                assert getattr(request, "workload_class") == "campaign"
                assert getattr(request, "max_queue_wait_ms") == 123
                return InferenceResult(
                    role="frontdoor",
                    output="ok",
                    tokens_generated=1,
                    generation_speed=10.0,
                    elapsed_time=0.1,
                    success=True,
                )

        prims = LLMPrimitives(mock_mode=False)
        prims._backends["frontdoor"] = FakeConcurrencyAwareBackend()
        with prims.request_context(
            priority="background",
            workload_class="campaign",
            max_queue_wait_ms=123,
        ):
            assert prims._real_call("hi", "frontdoor") == "ok"

    def test_real_call_with_n_probs_uses_batch_path_and_preserves_rows(self):
        rows = [{"content": "o", "probs": [{"tok_str": "o", "prob": 0.8}]}]

        class FakeConcurrencyAwareBackend:
            _dispatch = object()
            _tap_dispatch_metadata = object()

            def infer_stream_text(self, _role_config, _request, on_chunk):  # noqa: ANN001
                raise AssertionError("n_probs requests must not use streaming")

            def infer(self, _role_config, request):
                assert request.n_probs == 7
                return InferenceResult(
                    role="frontdoor",
                    output="ok",
                    tokens_generated=1,
                    generation_speed=10.0,
                    elapsed_time=0.1,
                    success=True,
                    completion_probabilities=rows,
                )

        prims = LLMPrimitives(mock_mode=False)
        prims._backends["frontdoor"] = FakeConcurrencyAwareBackend()

        assert prims._real_call("hi", "frontdoor", n_probs=7) == "ok"
        assert prims._last_inference_meta["completion_probabilities"] == rows


class TestRealBatch:
    """Tests for _real_batch() method."""

    def test_real_batch_parallel_execution(self, mock_backend):
        """Test _real_batch executes calls in parallel."""
        prims = LLMPrimitives(mock_mode=False)
        prims._backends["worker"] = mock_backend

        # Mock responses
        def mock_infer(role_config, request):
            # Return different responses based on prompt
            prompt = request.prompt
            return InferenceResult(
                role="worker",
                output=f"Response to: {prompt[:20]}",
                tokens_generated=10,
                generation_speed=25.0,
                elapsed_time=0.1,
                success=True,
            )

        mock_backend.infer.side_effect = mock_infer

        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
        results = prims._real_batch(prompts, "worker")

        assert len(results) == 3
        assert "Prompt 1" in results[0]
        assert "Prompt 2" in results[1]
        assert "Prompt 3" in results[2]
        assert mock_backend.infer.call_count == 3

    def test_real_batch_handles_errors(self, mock_backend):
        """Test _real_batch handles individual call errors."""
        prims = LLMPrimitives(mock_mode=False)
        prims._backends["worker"] = mock_backend

        # Second call raises error
        call_count = [0]

        def mock_infer(role_config, request):
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("Simulated failure")
            return InferenceResult(
                role="worker",
                output="Success",
                tokens_generated=5,
                generation_speed=20.0,
                elapsed_time=0.1,
                success=True,
            )

        mock_backend.infer.side_effect = mock_infer

        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
        results = prims._real_batch(prompts, "worker")

        assert len(results) == 3
        assert results[0] == "Success"
        assert "[ERROR:" in results[1]  # Error formatted
        assert results[2] == "Success"

    def test_real_batch_no_backend_raises_error(self):
        """Test _real_batch raises error when no backend available."""
        prims = LLMPrimitives(mock_mode=False)

        with pytest.raises(RuntimeError, match="No backend configured"):
            prims._real_batch(["Prompt 1", "Prompt 2"], "unknown_role")

    def test_real_batch_with_model_server(self, mock_model_server):
        """Test _real_batch uses ModelServer when no backends."""
        prims = LLMPrimitives(mock_mode=False, model_server=mock_model_server)

        # Mock _real_call to verify it's used
        with patch.object(prims, "_real_call", return_value="Mocked response") as mock_call:
            prompts = ["P1", "P2"]
            results = prims._real_batch(prompts, "worker")

        assert len(results) == 2
        assert all(r == "Mocked response" for r in results)
        assert mock_call.call_count == 2


class TestWorkerPoolBatch:
    """Tests for _worker_pool_batch() method."""

    def test_worker_pool_batch_routes_correctly(self):
        """Test worker pool batch routes to correct task type."""
        mock_pool = Mock()
        mock_batch = object()
        mock_pool.batch = Mock(return_value=mock_batch)

        prims = LLMPrimitives(
            mock_mode=False,
            worker_pool=mock_pool,
            use_worker_pool=True,
        )

        prompts = ["Explore this", "Analyze that"]

        # Patch get_event_loop to return a non-running loop so we always
        # hit the asyncio.run() branch (event loop state varies in full suite).
        mock_loop = Mock()
        mock_loop.is_running.return_value = False

        with (
            patch("asyncio.get_event_loop", return_value=mock_loop),
            patch("asyncio.run") as mock_run,
        ):
            mock_run.return_value = ["Result 1", "Result 2"]
            results = prims._worker_pool_batch(prompts, "worker_explore")

        assert len(results) == 2
        mock_pool.batch.assert_called_once_with(prompts, task_type="worker_general")
        mock_run.assert_called_once_with(mock_batch)

    def test_worker_pool_batch_fallback_on_error(self, mock_model_server):
        """Test worker pool batch falls back on error."""
        mock_pool = Mock()
        mock_pool.batch.side_effect = RuntimeError("Pool unavailable")

        prims = LLMPrimitives(
            mock_mode=False,
            worker_pool=mock_pool,
            use_worker_pool=True,
            model_server=mock_model_server,
        )

        # Mock fallback
        with patch.object(
            prims, "_fallback_batch", return_value=["Fallback 1", "Fallback 2"]
        ) as mock_fallback:
            results = prims._worker_pool_batch(["P1", "P2"], "worker_code")

        assert results == ["Fallback 1", "Fallback 2"]
        mock_fallback.assert_called_once_with(["P1", "P2"], "worker_code")
