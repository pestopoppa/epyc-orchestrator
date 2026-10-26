#!/usr/bin/env python3
"""Pytest configuration and fixtures for the orchestrator test suite.

Memory Safety:
    This file includes a memory guard that warns/exits if available RAM is below
    a safe threshold. This prevents crashes from running tests with pytest-xdist
    parallel workers that each load models.

    The 192-thread EPYC system with 1.13TB RAM can still crash if pytest spawns
    too many workers that each initialize the API (which loads the TaskEmbedder
    model). The lazy loading in src/api.py prevents this for mock mode tests,
    but this guard provides an additional safety net.

    The memory check is SKIPPED in CI environments (detected via CI or
    ORCHESTRATOR_MOCK_MODE env vars) since CI runners have limited RAM but
    only run mock-mode tests that don't load models.

Usage:
    pytest tests/              # Normal test run with memory check
    pytest tests/ -n 4         # Safe parallel execution (max 4 workers)
    pytest tests/ -n auto      # DANGEROUS on this machine - avoid

See also:
    - CLAUDE.md for memory constraints
    - research/ESCALATION_FLOW.md for memory pool architecture
"""

import atexit
import fnmatch
import importlib
import inspect
import json
import os
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# This conftest is loaded before test modules. Set the construction-time logger
# override before application imports so test-time default loggers stay isolated.
# Keep this as a literal to avoid importing application modules during pytest setup.
_PROGRESS_LOG_DIR_ENV = "ORCHESTRATOR_PROGRESS_LOG_DIR"
_PREVIOUS_PROGRESS_LOG_DIR = os.environ.get(_PROGRESS_LOG_DIR_ENV)
_TEST_PROGRESS_LOG_TEMP = tempfile.TemporaryDirectory(prefix="orchestrator-pytest-progress-")
os.environ[_PROGRESS_LOG_DIR_ENV] = _TEST_PROGRESS_LOG_TEMP.name

# Runtime stores that default into the checkout, pinned to a temp dir for the whole run
# (also before application imports: some bind their path at import).
_TEST_RUNTIME_STORES = tempfile.TemporaryDirectory(prefix="orchestrator-pytest-stores-")
_TEST_TRACE_DB_PATHS: set[Path] = set()
_RUNTIME_STORE_ENVS = {
    "ORCHESTRATOR_PATHS_LOG_DIR": "logs",
    "ORCHESTRATOR_REPLAY_CAS_DIR": "replay_cas",
    "ORCHESTRATOR_PATHS_SESSIONS_DIR": "sessions",
    "ORCHESTRATOR_DELEGATION_REPORT_DIR": "delegation_reports",
    "AUTOPILOT_EVAL_INSTRUMENT_LEDGER": "eval_instrument_ledger.json",
    # Successful tool audits may persist first-seen hosts beside this log.
    # Keep both writes under the per-process pytest temp root, never the checkout.
    "ORCHESTRATOR_TOOL_AUDIT_LOG": "tool_call_audit.jsonl",
    # Ruff is invoked by some subprocess tests; keep its cache out of the checkout too.
    "RUFF_CACHE_DIR": "ruff-cache",
}
_PREVIOUS_RUNTIME_STORE_ENVS = {key: os.environ.get(key) for key in _RUNTIME_STORE_ENVS}
for _env, _name in _RUNTIME_STORE_ENVS.items():
    os.environ[_env] = os.path.join(_TEST_RUNTIME_STORES.name, _name)
Path(os.environ["ORCHESTRATOR_PATHS_LOG_DIR"]).mkdir(parents=True, exist_ok=True)
_PRODUCTION_KERNEL_STORE_CPU = Path("/mnt/raid0/llm/kernels/production/cpu")


def _install_hermetic_kernel_store() -> None:
    """Give the suite a stub kernel store when the production one is absent.

    `src.config` resolves `src.registry.kernel_paths.backend_dir()` in dataclass
    default factories, and `src.api` builds a config at import — so without the
    store at `/mnt/raid0/llm/kernels/production/<backend>` this conftest could not
    even be imported, and no test ran anywhere but the production host.

    Only acts when the production store is missing and no store was named
    explicitly, so a run on the production host is unchanged. The stubs are inert
    executables: nothing in mock mode launches them.
    """
    if os.environ.get("ORCHESTRATOR_PATHS_KERNEL_ROOT") or _PRODUCTION_KERNEL_STORE_CPU.is_dir():
        return
    root = Path(tempfile.mkdtemp(prefix="orchestrator-test-kernels-"))
    atexit.register(shutil.rmtree, root, ignore_errors=True)
    os.environ["ORCHESTRATOR_PATHS_KERNEL_ROOT"] = str(root)

    from src.registry import kernel_paths

    for backend, binary in kernel_paths.BACKEND_BINARIES.items():
        backend_dir = kernel_paths.PRODUCTION_ROOT / backend
        backend_dir.mkdir(parents=True)
        stub = backend_dir / binary
        stub.write_text("#!/bin/sh\nexit 0\n")
        stub.chmod(0o755)


def _import_stack_manifest_off_serving_host() -> None:
    """Import `scripts.server.stack_manifest` on a machine smaller than the serving host.

    The module validates the declared lineup against THIS machine's MemTotal at
    import (a deliberate fail-fast for `orchestrator_stack.py`). On a test box that
    leg can only fail: it measures the wrong machine. If the host-RAM leg is the
    ONLY failure, re-import with /proc/meminfo unreadable, which the module already
    reports as host leg UNGATED ("capacity unknown", not "capacity fine"). Every
    other failure — the GPU leg, role classification, declaration parity — is left
    to surface in the tests that import the module, exactly as before.
    """
    try:
        import scripts.server.stack_manifest  # noqa: F401

        return
    except ValueError as exc:
        message = str(exc)
        if "device host (CPU RAM) OVERSUBSCRIBED" not in message or "device ROCm0" in message:
            return
    except Exception:
        return

    real_read_text = Path.read_text

    def read_text_without_meminfo(self, *args, **kwargs):
        if str(self) == "/proc/meminfo":
            raise OSError("hidden by tests/conftest.py: this box is not the serving host")
        return real_read_text(self, *args, **kwargs)

    Path.read_text = read_text_without_meminfo
    try:
        import scripts.server.stack_manifest  # noqa: F401
    except Exception:
        pass
    finally:
        Path.read_text = real_read_text


_install_hermetic_kernel_store()
_import_stack_manifest_off_serving_host()

from src.config import reset_config  # noqa: E402
from src.api.state import AppState  # noqa: E402


# Memory threshold in GB (fail if less available)
MEMORY_THRESHOLD_GB = 100

# Check if running in CI environment (GitHub Actions, etc.)
IS_CI = os.environ.get("CI") == "true" or os.environ.get("ORCHESTRATOR_MOCK_MODE") == "true"

# Maximum safe parallel workers for this machine
# -n 8 gives 4x speedup (67s → 17s), -n 4 gives 3x (67s → 23s)
MAX_SAFE_WORKERS = 8


def pytest_addoption(parser):
    """Add custom command-line options."""
    parser.addoption(
        "--run-server",
        action="store_true",
        default=False,
        help="Run tests that require a live llama-server instance",
    )
    parser.addoption(
        "--run-ocr-server",
        action="store_true",
        default=False,
        help="Run tests that require a live OCR server",
    )
    parser.addoption(
        "--run-materialized-trace",
        action="store_true",
        default=False,
        help="Read the host's materialized trace database in the smoke test",
    )
    parser.addoption(
        "--run-live-models",
        action="store_true",
        default=False,
        help="Run tests against live models (requires orchestrator)",
    )


def pytest_configure(config):
    """Check memory and register custom markers before running tests."""
    # Register custom markers
    config.addinivalue_line("markers", "heavy: marks tests that load models (may need more memory)")
    config.addinivalue_line("markers", "real_mode: marks tests that require real inference servers")
    config.addinivalue_line(
        "markers", "requires_server: marks tests that require a live llama-server"
    )
    config.addinivalue_line(
        "markers", "requires_live_models: marks tests that require live orchestrator models"
    )
    config.addinivalue_line("markers", "integration: marks integration tests")
    config.addinivalue_line("markers", "requires_materialized_trace: reads host trace data")

    # Check available memory (skip in CI - mock mode tests don't load models)
    if IS_CI:
        pass  # Skip memory check in CI environments
    else:
        try:
            import psutil

            free_gb = psutil.virtual_memory().available / (1024**3)

            if free_gb < MEMORY_THRESHOLD_GB:
                # Hard fail if memory is critically low
                pytest.exit(
                    f"DANGER: Only {free_gb:.0f}GB free RAM. "
                    f"Need {MEMORY_THRESHOLD_GB}GB+ for safe testing.\n"
                    "This machine can crash if tests load too many models.\n"
                    "Free up memory or wait for other processes to complete.",
                    returncode=1,
                )
            elif free_gb < MEMORY_THRESHOLD_GB * 2:
                # Warn if memory is getting low
                warnings.warn(
                    f"Low memory: {free_gb:.0f}GB free. Tests may be slow. Consider freeing memory.",
                    UserWarning,
                )
        except ImportError:
            warnings.warn(
                "psutil not installed - cannot check memory. Install with: pip install psutil",
                UserWarning,
            )


_REPO_ROOT = Path(__file__).resolve().parents[1]
_CHECKOUT_LEAK_EXEMPT = (
    "*__pycache__/*",
    ".pytest_cache/*",
    ".hypothesis/*",
    ".coverage*",  # tool caches
    # Leaks seen in a full run with no fix yet; shrink this list, never grow it.
    "logs/.bg_cleanup.lock",
    "logs/dispatch/*",
    "logs/progress/*",
    "logs/serving_calls/*",
    "logs/tool_call_audit.jsonl",
    "orchestration/procedures/state/test_*.json",
    "orchestration/repl_memory/sessions/*",
)


def _checkout_untracked() -> set[str] | None:
    """Untracked and ignored paths in this checkout; None when git cannot say."""
    try:
        out = subprocess.run(
            [
                "git",
                "-C",
                str(_REPO_ROOT),
                "status",
                "--porcelain",
                "--ignored",
                "--untracked-files=all",
                "-z",
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    paths = {entry[3:] for entry in out.split("\0") if entry[:2] in ("??", "!!")}
    return {p for p in paths if not any(fnmatch.fnmatch(p, pat) for pat in _CHECKOUT_LEAK_EXEMPT)}


def pytest_sessionstart(session):
    # Leak guard (A6-02): a test that writes into the checkout fails the run. --ignored
    # matters: every leak seen so far (MagicMock/, data/trace, logs/) is gitignored.
    if not hasattr(session.config, "workerinput"):
        session.config._checkout_before = _checkout_untracked()


def pytest_sessionfinish(session, exitstatus):
    before = getattr(session.config, "_checkout_before", None)
    after = _checkout_untracked() if before is not None else None
    leaked = sorted(after - before) if after is not None else []
    if leaked:
        listing = "\n  ".join(leaked[:50])
        session.config.get_terminal_writer().line(
            f"\nTests wrote {len(leaked)} new path(s) into the checkout {_REPO_ROOT}; "
            f"write under tmp_path instead:\n  {listing}",
            red=True,
        )
        if session.exitstatus == pytest.ExitCode.OK:
            session.exitstatus = pytest.ExitCode.TESTS_FAILED


@pytest.fixture(scope="session", autouse=True)
def _restore_progress_log_override_after_suite():
    """Keep every default ProgressLogger inside this pytest run's temp directory."""
    yield _TEST_PROGRESS_LOG_TEMP.name
    if _PREVIOUS_PROGRESS_LOG_DIR is None:
        os.environ.pop(_PROGRESS_LOG_DIR_ENV, None)
    else:
        os.environ[_PROGRESS_LOG_DIR_ENV] = _PREVIOUS_PROGRESS_LOG_DIR


@pytest.fixture(scope="session", autouse=True)
def _restore_runtime_store_overrides_after_suite():
    yield
    for key, previous in _PREVIOUS_RUNTIME_STORE_ENVS.items():
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


@pytest.fixture(autouse=True)
def _isolate_trace_store(tmp_path_factory):
    """Redirect default trace APIs, including aliases bound during collection.

    The path is captured in Python argument defaults, so patching the module
    constant alone cannot isolate emit() or ReviewTracingProcessor. Patch the
    callable objects themselves; explicit db_path arguments still work normally.
    A private MonkeyPatch keeps teardown independent of tests' config patches.
    """
    from src.trace import store
    from src.trace.emit import ReviewTracingProcessor, emit

    # Load the navigation surface before rebinding keyword-only defaults too.
    importlib.import_module("src.trace.navigation")
    production_db = store.DEFAULT_DB_PATH
    private_db = tmp_path_factory.mktemp("trace_store") / "events.sqlite"
    _TEST_TRACE_DB_PATHS.add(private_db)
    modules = [
        module
        for name, module in tuple(sys.modules.items())
        if name == "src.trace" or name.startswith("src.trace.")
    ]
    functions = {ReviewTracingProcessor.__init__, emit}
    for module in modules:
        functions.update(
            value
            for value in vars(module).values()
            if inspect.isfunction(value) and value.__module__.startswith("src.trace")
        )

    def redirect(value):
        # A module imported mid-test may retain that test's private default after
        # teardown. Recognize only our own prior paths when the next test begins.
        if isinstance(value, Path) and (value == production_db or value in _TEST_TRACE_DB_PATHS):
            return private_db
        return value

    with pytest.MonkeyPatch.context() as mp:
        # Children must pass this path explicitly via db_path= or CLI --db;
        # Python argument rebinding is local to this pytest worker process.
        mp.setenv("ORCHESTRATOR_TEST_TRACE_DB", str(private_db))
        # Includes CLI/package aliases loaded at collection. Later imports bind
        # the already-private store constant, including the trace sink factories.
        for module in modules:
            current = getattr(module, "DEFAULT_DB_PATH", None)
            if redirect(current) is not current:
                mp.setattr(module, "DEFAULT_DB_PATH", private_db)
        for function in functions:
            defaults = function.__defaults__ or ()
            if any(redirect(value) is not value for value in defaults):
                mp.setattr(function, "__defaults__", tuple(map(redirect, defaults)))
            if function.__kwdefaults__ and any(
                redirect(value) is not value for value in function.__kwdefaults__.values()
            ):
                mp.setattr(
                    function,
                    "__kwdefaults__",
                    {key: redirect(value) for key, value in function.__kwdefaults__.items()},
                )
        yield private_db


@pytest.fixture(scope="session", autouse=True)
def _pin_runtime_feature_flags(tmp_path_factory):
    """Pin the runtime feature-flag file so the suite is REPRODUCIBLE.

    `src.features.runtime_flags_path()` reads
    `orchestration/runtime_flags.json` — a gitignored file the RUNNING API
    rewrites at will (`set_by: api:127.0.0.1`). So unit results depended on
    whatever the live orchestrator last toggled, and could change *mid-run*:
    during a 2026-08-02 attribution pass the `memrl` flag flipped while the
    suite was executing.

    The cost of that was not hypothetical. Comparing a clean-worktree baseline
    against the working tree produced 18 apparent regressions; 17 of them were
    this file, because the worktree never had the live flags. `structured_tool_
    output: true` alone rewrites tool returns into a `ToolOutput` wrapper and
    fails four `test_tool_registry` cases that pass with flags neutral. Any
    attribution done without this pin is measuring operator toggles, not code.

    Pinned to an EMPTY flag set, so every test sees each feature at its declared
    default rather than at whatever production happens to be running. A test
    that specifically exercises flag loading should monkeypatch
    `ORCHESTRATOR_RUNTIME_FLAGS_PATH` to its own fixture; that still works,
    because this only sets the ambient default.
    """
    from src.features import RUNTIME_FLAGS_ENV

    previous = os.environ.get(RUNTIME_FLAGS_ENV)
    neutral = tmp_path_factory.mktemp("runtime_flags") / "runtime_flags.json"
    neutral.write_text(json.dumps({"flags": {}, "version": 1}), encoding="utf-8")
    os.environ[RUNTIME_FLAGS_ENV] = str(neutral)
    try:
        yield neutral
    finally:
        if previous is None:
            os.environ.pop(RUNTIME_FLAGS_ENV, None)
        else:
            os.environ[RUNTIME_FLAGS_ENV] = previous


@pytest.fixture(scope="session", autouse=True)
def _disable_kb_rag_query_length_log():
    """Keep the suite out of the live KB-RAG query-length telemetry (H2).

    `kb_rag.query()` appends to `data/kb_rag/telemetry/query_lengths.jsonl` by
    default; a test run from the shared clone would otherwise mix synthetic
    queries into the production observation. Tests that exercise the log
    monkeypatch the env var to their own tmp path.
    """
    from src.retrieval.kb_rag_query_telemetry import LOG_ENV

    previous = os.environ.get(LOG_ENV)
    os.environ[LOG_ENV] = "off"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(LOG_ENV, None)
        else:
            os.environ[LOG_ENV] = previous


@pytest.fixture(scope="session", autouse=True)
def _disable_serving_calls_log():
    """Keep the suite out of the live per-call serving log (``logs/serving_calls``).

    Every ``LlamaServerBackend`` call writes one record; a test run from the shared
    clone would otherwise mix mocked calls into the production observation. Tests
    that exercise the log monkeypatch the env var to their own tmp path.
    """
    from src.backends.serving_calls import LOG_ENV

    previous = os.environ.get(LOG_ENV)
    os.environ[LOG_ENV] = "off"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(LOG_ENV, None)
        else:
            os.environ[LOG_ENV] = previous


@pytest.fixture(scope="session", autouse=True)
def _disable_live_gpu_window():
    """Keep the suite off the live MI210 window file (parked roles).

    A real ``holder=autokernel`` window on this host would otherwise refuse every
    test call to a parked role. Tests that exercise parking point the env var at
    their own tmp file.
    """
    from src.runtime.gpu_window import PATH_ENV

    previous = os.environ.get(PATH_ENV)
    os.environ[PATH_ENV] = "off"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(PATH_ENV, None)
        else:
            os.environ[PATH_ENV] = previous


@pytest.fixture(scope="session", autouse=True)
def _disable_live_context_limit_reads():
    """Keep the suite off the live stack's ``GET /props`` (context limits).

    ``ContextLimitResolver`` reads each role's per-request n_ctx from the live
    llama-server; unit tests must resolve from the registry (or from an injected
    resolver) instead of whatever happens to be serving on this host.
    """
    from src.backends.context_limits import LIVE_ENV

    previous = os.environ.get(LIVE_ENV)
    os.environ[LIVE_ENV] = "off"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(LIVE_ENV, None)
        else:
            os.environ[LIVE_ENV] = previous


@pytest.fixture(autouse=True)
def _reset_context_limit_resolver():
    """Drop the process-wide context-limit cache between tests."""
    from src.backends.context_limits import set_context_limit_resolver

    set_context_limit_resolver(None)
    yield
    set_context_limit_resolver(None)


@pytest.fixture(autouse=True)
def _hermetic_long_prefill_lease(tmp_path_factory):
    """Keep tests off the HOST-WIDE long-prefill lease (KVU-15a).

    The lease is an flock on ``{tmp_dir}/kv_prefill_lease.{host}_{port}.lock`` —
    by default ``/mnt/raid0/llm/tmp``, where the LIVE API workers take it. A test
    that admits a long request on ``localhost:8083`` would otherwise hold the
    production lease (and stall real long prefills) for as long as the test
    holds it. Each test gets its own directory, so a lease a test never releases
    cannot leak into the next one either. Only this module's resolver is
    patched, never ``ORCHESTRATOR_TMP_DIR`` (see tests/unit/conftest.py for why
    a global tmp-dir override is the wrong seam).

    Uses a PRIVATE ``MonkeyPatch``, not the ``monkeypatch`` fixture: requesting
    that fixture from an autouse fixture declared before
    ``_reset_config_between_tests`` sets it up earlier and so tears it down
    LATER, i.e. a test's ``monkeypatch.setattr("src.config.get_config", ...)``
    would still be in place when ``reset_config()`` calls
    ``get_config.cache_clear()`` (AttributeError at teardown).
    """
    from src.runtime import long_prefill_lease

    lease_dir = tmp_path_factory.mktemp("hermetic_prefill_lease")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(long_prefill_lease, "lease_dir", lambda: lease_dir)
        yield lease_dir


@pytest.fixture(autouse=True)
def _reset_prefix_history():
    """Fresh KVU-15c prefix history per test: a prefix one test served must not
    credit another's request (its host-wide file already lives in the per-test
    lease dir above)."""
    from src.scheduling.prefix_history import set_prefix_history

    set_prefix_history(None)
    yield
    set_prefix_history(None)


@pytest.fixture(autouse=True)
def _isolate_omitted_episodic_store_path(tmp_path):
    """Keep default store construction private without changing explicit paths."""
    from orchestration.repl_memory.episodic_store import EpisodicStore
    from tests.fixtures.runtime_store_paths import private_default_store_init

    with pytest.MonkeyPatch.context() as isolation:
        isolation.setattr(EpisodicStore, "__init__", private_default_store_init(
            EpisodicStore.__init__, tmp_path / "default-episodic-store"))
        yield


@pytest.fixture(autouse=True)
def _reset_config_between_tests():
    """Ensure config cache is clean between tests.

    Prevents env var leaks from one test affecting another.
    Uses reset_config() which clears the lru_cache on get_config().
    """
    reset_config()
    yield
    reset_config()


def pytest_collection_modifyitems(config, items):
    """Add markers and warnings for parallel execution."""
    if not config.getoption("--run-server"):
        skip_server = pytest.mark.skip(reason="Need --run-server to run live server tests")
        for item in items:
            if "requires_server" in item.keywords:
                item.add_marker(skip_server)

    # Check if pytest-xdist is being used with too many workers
    try:
        num_workers = config.option.numprocesses
        if num_workers is not None:
            if num_workers == "auto":
                warnings.warn(
                    f"DANGER: pytest -n auto on 192-thread machine will spawn ~192 workers!\n"
                    f"Each worker may load models, causing memory exhaustion.\n"
                    f"Use: pytest -n {MAX_SAFE_WORKERS} instead.",
                    UserWarning,
                )
            elif isinstance(num_workers, int) and num_workers > MAX_SAFE_WORKERS:
                warnings.warn(
                    f"High parallelism: {num_workers} workers requested.\n"
                    f"Recommended max: {MAX_SAFE_WORKERS}. May cause memory issues.",
                    UserWarning,
                )
    except (AttributeError, TypeError):
        # pytest-xdist not installed or not using -n flag
        pass

    if not config.getoption("--run-materialized-trace"):
        skip_trace = pytest.mark.skip(
            reason="Need --run-materialized-trace to read host trace data"
        )
        for item in items:
            if "requires_materialized_trace" in item.keywords:
                item.add_marker(skip_trace)

    if not config.getoption("--run-live-models"):
        skip_live_models = pytest.mark.skip(reason="Need --run-live-models to run live model tests")
        for item in items:
            if "requires_live_models" in item.keywords:
                item.add_marker(skip_live_models)


@pytest.fixture
def mock_registry():
    """Shared registry mock fixture for chat pipeline tests."""
    registry = MagicMock()
    registry.routing_hints = {}
    return registry


@pytest.fixture
def mock_app_state(mock_registry):
    """Shared AppState mock fixture with common attributes used by route tests."""
    state = MagicMock(spec=AppState)
    state.progress_logger = MagicMock()
    state.hybrid_router = None
    state.tool_registry = MagicMock()
    state.script_registry = MagicMock()
    state.registry = mock_registry
    state.health_tracker = MagicMock()
    state.admission = MagicMock()
    state.increment_active = MagicMock()
    state.decrement_active = MagicMock()
    state.increment_request = MagicMock()
    return state


@pytest.fixture
def mock_llm_primitives():
    """Shared LLM primitives mock fixture with common telemetry fields."""
    primitives = MagicMock()
    primitives._backends = True
    primitives.total_tokens_generated = 100
    primitives.total_prompt_eval_ms = 50
    primitives.total_generation_ms = 200
    primitives._last_predicted_tps = 25.0
    primitives.total_http_overhead_ms = 10
    primitives.get_cache_stats.return_value = {"hits": 5, "misses": 2}
    primitives.llm_call.return_value = "Test response"
    return primitives
