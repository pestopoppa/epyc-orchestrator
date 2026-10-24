"""Contract tests for the shared bounded llama-mtmd CLI probe."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import os
from pathlib import Path
import signal
import shutil
import subprocess
import time

import pytest

from src.services.mtmd_probe import run_mtmd_probe
from src.services.lightonocr_llama_server import _probe_mtmd_cli
from src.vision.analyzers import vl_describe
from src.vision.analyzers.vl_describe import VLDescribeAnalyzer


ROOT = Path(__file__).resolve().parents[2]
_HELPER = ROOT / "scripts/lib/mtmd_probe.sh"


def _candidate(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/bash\n" + body, encoding="utf-8")
    path.chmod(0o755)
    return path


def test_shared_probe_merges_output_and_prefixes_candidate_library_dir(tmp_path, monkeypatch):
    cli = _candidate(tmp_path / "build" / "llama-mtmd-cli",
                     'printf "ld=%s\\n" "$LD_LIBRARY_PATH"\n'
                     'printf "version: 10303 (abc123)\\n" >&2\n')
    monkeypatch.setenv("LD_LIBRARY_PATH", "/old/lib:/older/lib")

    proc = run_mtmd_probe(cli)

    assert proc is not None and proc.returncode == 0
    assert proc.stdout.splitlines() == [
        f"ld={cli.parent.resolve()}:/old/lib:/older/lib",
        "version: 10303 (abc123)",
    ]


def test_shared_probe_moves_an_existing_candidate_dir_to_front_once(tmp_path, monkeypatch):
    cli = _candidate(tmp_path / "build" / "llama-mtmd-cli",
                     'printf "%s\\n" "$LD_LIBRARY_PATH"\n')
    resolved = str(cli.parent.resolve())
    monkeypatch.setenv("LD_LIBRARY_PATH", f"/old/lib:{resolved}:/older/lib:{resolved}")

    proc = run_mtmd_probe(cli)

    assert proc is not None and proc.returncode == 0
    assert proc.stdout.strip() == f"{resolved}:/old/lib:/older/lib"


def test_shared_probe_uses_symlink_target_directory_for_library_prefix(tmp_path, monkeypatch):
    target = _candidate(tmp_path / "real-build" / "llama-mtmd-cli",
                        'printf "ld=%s\\n" "$LD_LIBRARY_PATH"\n')
    alias_dir = tmp_path / "configured"
    alias_dir.mkdir()
    alias = alias_dir / "llama-mtmd-cli"
    alias.symlink_to(target)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/old/lib")

    proc = run_mtmd_probe(alias)

    assert proc is not None and proc.returncode == 0
    assert proc.stdout.strip() == f"ld={target.parent.resolve()}:/old/lib"


def test_shared_probe_preserves_nonzero_status_and_complete_merged_text(tmp_path):
    cli = _candidate(tmp_path / "bin" / "llama-mtmd-cli",
                     'printf "version: 10303 (abc123)\\n"\n'
                     'printf "probe failed\\n" >&2\nexit 7\n')

    proc = run_mtmd_probe(cli)

    assert proc is not None and proc.returncode == 7
    assert proc.stdout == "version: 10303 (abc123)\nprobe failed\n"
    assert _probe_mtmd_cli(cli) is None
    assert VLDescribeAnalyzer._mtmd_runs(cli) is None


def test_python_callers_keep_distinct_strict_and_tolerant_parsers(tmp_path):
    cli = _candidate(tmp_path / "bin" / "llama-mtmd-cli",
                     'printf "prefix version: build-10303\\n"\n')

    assert _probe_mtmd_cli(cli) is None
    assert VLDescribeAnalyzer._mtmd_runs(cli) == "prefix version: build-10303"


def test_shared_probe_reports_nonexecutable_candidate(tmp_path):
    proc = run_mtmd_probe(tmp_path / "missing" / "llama-mtmd-cli")

    assert proc is not None and proc.returncode == 126
    assert proc.stdout == ""


def test_vl_resolver_never_falls_back_to_a_build_tree(monkeypatch, tmp_path, caplog):
    """A5-M2: llama.cpp/build* trees hold retired kernels; a broken store binary is an error."""
    configured = tmp_path / "llama.cpp/configured/bin/llama-mtmd-cli"
    monkeypatch.setattr(vl_describe, "LLAMA_MTMD_CLI", configured)
    seen = []

    def probe(candidate):
        seen.append(candidate)
        return None if candidate == configured else "version: 10125 (0db32c06e)"

    monkeypatch.setattr(VLDescribeAnalyzer, "_mtmd_runs", staticmethod(probe))
    analyzer = object.__new__(VLDescribeAnalyzer)

    with caplog.at_level("ERROR"):
        assert analyzer._resolve_mtmd_cli() is None
    assert seen == [configured]
    assert "not runnable" in caplog.text

    monkeypatch.setattr(VLDescribeAnalyzer, "_mtmd_runs", staticmethod(lambda c: "version: 1 (a)"))
    assert analyzer._resolve_mtmd_cli() == configured


def test_vl_cli_run_leads_ld_library_path_with_the_binary_dir(monkeypatch, tmp_path):
    from types import SimpleNamespace

    cli = _candidate(tmp_path / "store/llama-mtmd-cli", "")
    analyzer = object.__new__(VLDescribeAnalyzer)
    analyzer.model_path, analyzer.mmproj_path = tmp_path / "m.gguf", tmp_path / "p.gguf"
    analyzer.prompt, analyzer.max_tokens, analyzer.threads = "describe", 8, 4
    monkeypatch.setattr(analyzer, "_resolve_mtmd_cli", lambda: cli, raising=False)
    captured = {}

    def fake_run(cmd, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(returncode=1, stderr="boom", stdout="")

    monkeypatch.setattr(vl_describe.subprocess, "run", fake_run)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/ambient/lib")
    analyzer._analyze_with_cli(tmp_path / "img.jpg", 0.0)

    assert captured["env"]["LD_LIBRARY_PATH"].split(":") == [str(cli.parent.resolve()), "/ambient/lib"]


def test_shared_probe_captures_timeout_status_and_output(tmp_path, monkeypatch):
    cli = _candidate(tmp_path / "bin" / "llama-mtmd-cli", 'exit 0\n')
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    timeout = _candidate(fake_bin / "timeout",
                         'printf "argv=%s\\n" "$*"\n'
                         'exit 124\n')
    monkeypatch.setenv("PATH", f"{fake_bin}:{os.environ.get('PATH', '')}")

    proc = run_mtmd_probe(cli)

    assert proc is not None and proc.returncode == 124
    assert proc.stdout.strip() == f"argv=--kill-after=2 20 {cli} --version"
    assert timeout.exists()


@pytest.mark.skipif(
    os.environ.get("EPYC_RUN_HOSTED_TIMEOUT_CONTROL") != "1",
    reason="real 20-second timeout and child-death control is hosted-only",
)
def test_timeout_kills_term_ignoring_child_and_leaves_no_process(tmp_path):
    pid_file = tmp_path / "child.pid"
    cli = tmp_path / "bin" / "llama-mtmd-cli"
    cli.parent.mkdir(parents=True)
    cli.write_text(
        "#!/usr/bin/env python3\n"
        "import os, signal, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        f"open({str(pid_file)!r}, 'w', encoding='utf-8').write(str(os.getpid()))\n"
        "while True: time.sleep(1)\n",
        encoding="utf-8",
    )
    cli.chmod(0o755)

    def snapshot(pid):
        """Return PID/start-time identity and state; zombies remain observable."""
        try:
            stat_text = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
        except FileNotFoundError:
            return None
        # comm is parenthesized and may itself contain spaces or ')'.
        fields_after_comm = stat_text[stat_text.rfind(")") + 2 :].split()
        identity = (pid, fields_after_comm[19])  # field 22: starttime
        return identity, fields_after_comm[0]  # field 3: process state

    def identity(pid):
        current = snapshot(pid)
        return current[0] if current is not None else None

    def wait_gone(expected, seconds=5):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if identity(expected[0]) != expected:
                return True
            time.sleep(0.05)
        return identity(expected[0]) != expected

    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mtmd-timeout-control")
    future = executor.submit(run_mtmd_probe, cli)
    child_identity = None
    try:
        # The probe must be live before we take the PID/start-time snapshot.
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                child_pid = int(pid_file.read_text(encoding="ascii").strip())
            except (FileNotFoundError, ValueError):
                time.sleep(0.05)
                continue
            current = snapshot(child_pid)
            if current is not None and current[1] not in {"Z", "X"}:
                child_identity = current[0]
                break
            child_identity = None
            time.sleep(0.05)
        assert child_identity is not None, "probe did not start a live captured child"

        proc = future.result(timeout=30)
        assert proc is not None and proc.returncode in (124, 137)
        assert wait_gone(child_identity), (
            f"timeout left captured TERM-ignoring child alive: {child_identity}"
        )
    finally:
        try:
            # Cleanup is limited to the exact captured process instance, never a reused PID.
            if child_identity is not None and identity(child_identity[0]) == child_identity:
                os.kill(child_identity[0], signal.SIGKILL)
                assert wait_gone(child_identity), (
                    f"captured child survived SIGKILL or was not reaped: {child_identity}"
                )
        finally:
            try:
                if not future.done():
                    try:
                        future.result(timeout=30)
                    except FutureTimeoutError as exc:
                        raise AssertionError("bounded MTMD probe future did not finish") from exc
            finally:
                executor.shutdown(wait=future.done(), cancel_futures=not future.done())


def _copy_env_library(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    lib = project / "scripts/lib"
    lib.mkdir(parents=True)
    shutil.copy2(ROOT / "scripts/lib/env.sh", lib / "env.sh")
    shutil.copy2(_HELPER, lib / "mtmd_probe.sh")
    return lib / "env.sh"


def _source_env(env_sh: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", 'set -euo pipefail; source "$1"; printf "%s\\n" "$ORCHESTRATOR_PATHS_LLAMA_MTMD"',
         "bash", str(env_sh)],
        env=env, text=True, capture_output=True, check=False,
    )


@pytest.mark.parametrize("store_body, warned", [
    ('printf "version: bad but nonzero\\n"\nexit 7\n', True),
    ('printf "version: 10303 (abc123)\\n"\n', False),
])
def test_env_sh_resolves_mtmd_only_from_the_store(tmp_path, store_body, warned):
    """A5-M2: never substitute a llama.cpp/build* binary; a broken store binary warns."""
    env_sh = _copy_env_library(tmp_path)
    store_cli = _candidate(tmp_path / "llm/kernels/production/cpu/llama-mtmd-cli", store_body)
    _candidate(tmp_path / "llm/llama.cpp/build/bin/llama-mtmd-cli",
               'printf "version: 10125 (0db32c06e)\\n"\n')
    env = dict(os.environ)
    env.update({
        "ORCHESTRATOR_PATHS_LLM_ROOT": str(tmp_path / "llm"),
        "ORCHESTRATOR_PATHS_PROJECT_ROOT": str(tmp_path / "project"),
        "ORCHESTRATOR_PATHS_LLAMA_CPP_BIN": str(store_cli.parent),
        "LD_LIBRARY_PATH": "/prior/lib",
    })
    env.pop("ORCHESTRATOR_PATHS_LLAMA_MTMD", None)

    result = _source_env(env_sh, env)

    assert result.returncode == 0
    assert result.stdout.strip() == str(store_cli)
    assert ("does not run" in result.stderr) is warned


def test_env_sh_preserves_explicit_mtmd_override_without_probing(tmp_path):
    env_sh = _copy_env_library(tmp_path)
    marker = tmp_path / "was-probed"
    _candidate(tmp_path / "project/llama.cpp/build/llama-mtmd-cli",
               f'touch "{marker}"\nprintf "version: 10303 (abc123)\\n"\n')
    override = str(tmp_path / "operator-selected/llama-mtmd-cli")
    env = dict(os.environ)
    env.update({
        "ORCHESTRATOR_PATHS_LLM_ROOT": str(tmp_path / "llm"),
        "ORCHESTRATOR_PATHS_PROJECT_ROOT": str(tmp_path / "project"),
        "ORCHESTRATOR_PATHS_LLAMA_CPP_BIN": str(tmp_path / "project/llama.cpp/build"),
        "ORCHESTRATOR_PATHS_LLAMA_MTMD": override,
    })

    result = _source_env(env_sh, env)

    assert result.returncode == 0
    assert result.stdout.strip() == override
    assert not marker.exists()
