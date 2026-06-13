"""Tests for AdmissionController per-backend concurrency limiter."""

from pathlib import Path
import threading

import yaml

from src.api.admission import AdmissionController, _limits_from_stack_priors


class TestAdmissionController:
    """Tests for thread-safe admission control."""

    def test_unknown_backend_always_admitted(self):
        ctrl = AdmissionController({"http://localhost:8080": 1})
        assert ctrl.try_acquire("http://unknown:9999") is True

    def test_acquire_within_limit(self):
        ctrl = AdmissionController({"http://localhost:8080": 2})
        assert ctrl.try_acquire("http://localhost:8080") is True
        assert ctrl.try_acquire("http://localhost:8080") is True

    def test_acquire_beyond_limit_rejected(self):
        ctrl = AdmissionController({"http://localhost:8083": 1})
        assert ctrl.try_acquire("http://localhost:8083") is True
        # Second acquire should be rejected (limit=1)
        assert ctrl.try_acquire("http://localhost:8083") is False

    def test_release_allows_new_acquire(self):
        ctrl = AdmissionController({"http://localhost:8083": 1})
        assert ctrl.try_acquire("http://localhost:8083") is True
        assert ctrl.try_acquire("http://localhost:8083") is False
        ctrl.release("http://localhost:8083")
        assert ctrl.try_acquire("http://localhost:8083") is True

    def test_status_shows_in_flight(self):
        ctrl = AdmissionController({"http://localhost:8084": 2})
        ctrl.try_acquire("http://localhost:8084")
        status = ctrl.get_status()
        assert status["http://localhost:8084"]["limit"] == 2
        assert status["http://localhost:8084"]["in_flight"] == 1
        assert status["http://localhost:8084"]["available"] == 1

    def test_from_defaults_creates_controller(self):
        ctrl = AdmissionController.from_defaults()
        status = ctrl.get_status()
        assert "http://localhost:8084" not in status
        assert status["http://localhost:8070"]["limit"] == 2
        assert status["http://localhost:8080"]["limit"] == 2
        assert status["http://localhost:8083"]["limit"] == 2
        assert status["http://localhost:8082"]["limit"] == 1
        assert status["http://localhost:8086"]["limit"] == 2
        assert status["http://localhost:8087"]["limit"] == 1

    def test_limits_from_stack_priors_includes_shared_replica_ports(self, tmp_path: Path):
        priors = tmp_path / "stack_priors.yaml"
        priors.write_text(
            yaml.safe_dump(
                {
                    "roles": {
                        "frontdoor": {
                            "deployment_status": "live_stack",
                            "serving": {
                                "endpoint": "http://localhost:8070",
                                "ports": [8070, 8080, 8180],
                                "slots": 2,
                            },
                        },
                        "coder_escalation": {
                            "deployment_status": "live_stack",
                            "serving": {
                                "endpoint": "http://localhost:8070",
                                "ports": [8070, 8080, 8180],
                                "slots": 1,
                            },
                        },
                        "retired": {
                            "deployment_status": "benchmark_or_candidate",
                            "serving": {
                                "endpoint": "http://localhost:8084",
                                "ports": [8084],
                                "slots": 1,
                            },
                        },
                    }
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )

        limits = _limits_from_stack_priors(priors)

        assert limits["http://localhost:8070"] == 2
        assert limits["http://localhost:8080"] == 2
        assert limits["http://localhost:8180"] == 2
        assert "http://localhost:8084" not in limits

    def test_thread_safety(self):
        """Concurrent acquire/release should not corrupt state."""
        ctrl = AdmissionController({"http://localhost:8080": 2})
        errors = []

        def worker():
            try:
                for _ in range(100):
                    if ctrl.try_acquire("http://localhost:8080"):
                        ctrl.release("http://localhost:8080")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        # All slots should be available after all threads complete
        status = ctrl.get_status()
        assert status["http://localhost:8080"]["in_flight"] == 0
