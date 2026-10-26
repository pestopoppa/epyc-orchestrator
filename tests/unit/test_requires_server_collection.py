"""Keep live-server opt-in handling in the central collection hook."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


class _CollectionProbe:
    def __init__(self, collection_hook: Any) -> None:
        self.collection_hook = collection_hook
        self.collected = 0
        self.reports: list[tuple[str, str]] = []

    @staticmethod
    def pytest_addoption(parser: Any) -> None:
        parser.addoption("--run-server", action="store_true", default=False)
        parser.addoption("--run-live-models", action="store_true", default=False)
        parser.addoption("--run-materialized-trace", action="store_true", default=False)

    def pytest_collection_modifyitems(self, session: Any, config: Any, items: list[Any]) -> None:
        self.collection_hook(config, items)

    def pytest_collection_finish(self, session: Any) -> None:
        self.collected = len(session.items)

    def pytest_runtest_logreport(self, report: Any) -> None:
        self.reports.append((report.when, report.outcome))


def test_requires_server_is_skipped_by_default_and_opted_in_without_real_network(
    request: pytest.FixtureRequest, tmp_path: Path
) -> None:
    conftest_path = Path(__file__).resolve().parents[1] / "conftest.py"
    root_conftest = next(
        (
            plugin
            for plugin in request.config.pluginmanager.get_plugins()
            if getattr(plugin, "__file__", None)
            and Path(plugin.__file__).resolve() == conftest_path
        ),
        None,
    )
    assert root_conftest is not None
    collection_hook = getattr(root_conftest, "pytest_collection_modifyitems", None)
    assert callable(collection_hook)

    for run_server in (False, True):
        run_dir = tmp_path / ("opt-in" if run_server else "default")
        run_dir.mkdir()
        probe_file = run_dir / (
            "test_server_probe_opt_in.py" if run_server else "test_server_probe_default.py"
        )
        probe_file.write_text(
            """
import socket
from pathlib import Path
import pytest

@pytest.mark.requires_server
def test_mocked_server_probe(monkeypatch):
    def fake_connect(address, timeout=None):
        return ("mock-connection", address, timeout)
    monkeypatch.setattr(socket, "create_connection", fake_connect)
    connection = socket.create_connection(("127.0.0.1", 9001), timeout=0.1)
    assert connection == ("mock-connection", ("127.0.0.1", 9001), 0.1)
    Path(__file__).with_name("server-probe-ran").write_text("mock only")
""",
            encoding="utf-8",
        )
        probe = _CollectionProbe(collection_hook)
        args = [
            str(probe_file),
            "-q",
            "-p",
            "no:cacheprovider",
            "--basetemp",
            str(run_dir / "basetemp"),
        ]
        if run_server:
            args.append("--run-server")

        result = pytest.main(args, plugins=[probe])

        assert result == pytest.ExitCode.OK
        assert probe.collected == 1
        marker_file = run_dir / "server-probe-ran"
        if run_server:
            assert marker_file.read_text(encoding="utf-8") == "mock only"
            assert ("call", "passed") in probe.reports
        else:
            assert not marker_file.exists()
            assert any(outcome == "skipped" for _, outcome in probe.reports)
            assert not any(when == "call" for when, _ in probe.reports)
