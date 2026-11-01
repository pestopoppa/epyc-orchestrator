"""Registry recipe consumption is fail-closed and survives launch sanitization."""
import pytest

from scripts.server import stack_env
from scripts.server import orchestrator_stack as launcher


def test_recipe_env_survives_ambient_strip_and_override(monkeypatch):
    monkeypatch.setattr(stack_env, "recipe_for", lambda _: {"env": {"GGML_DECLARED": "1", "GGML_OVERRIDE": "recipe"}})
    monkeypatch.setattr(stack_env, "_role_env_overrides", lambda _: {"GGML_OVERRIDE": "role"})
    env = stack_env.build_launch_env("example", {"GGML_UNDECLARED": "ambient"})
    stack_env.strip_ambient_ggml(env, {**stack_env.recipe_env_for("example"), **stack_env._role_env_overrides("example")})
    assert env["GGML_DECLARED"] == "1"
    assert env["GGML_OVERRIDE"] == "role"
    assert "GGML_UNDECLARED" not in env


def test_recipe_shape_uses_machine_definition_and_rejects_drift(monkeypatch):
    role = "fixture"
    shape = next(name for name, (_, n) in launcher._CPU_SHAPES.items() if n == 96)
    cpus, threads = launcher._CPU_SHAPES[shape]
    monkeypatch.setitem(launcher.NUMA_CONFIG, role, {"instances": [(cpus, 1234, threads)]})
    monkeypatch.setitem(launcher.NUMA_INSTANCE_SHAPES, role, [shape])
    monkeypatch.setattr(launcher, "recipe_for", lambda _: {"cpu_shape": shape, "threads": threads})
    assert launcher._resolve_thread_count(role) == str(threads)
    monkeypatch.setattr(launcher, "recipe_for", lambda _: {"cpu_shape": "undefined", "threads": threads})
    with pytest.raises(ValueError, match="undefined recipe cpu_shape"):
        launcher._resolve_thread_count(role)
    monkeypatch.setattr(launcher, "recipe_for", lambda _: {"cpu_shape": shape, "threads": threads - 1})
    with pytest.raises(ValueError, match="disagree"):
        launcher._resolve_thread_count(role)


def test_process_attestation_reads_recipe_and_reports_drift(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from scripts.validate.check_shared_with_derivations import check_recipe_process

    role, pid = "fixture", 123
    recipe = {"threads": 96, "env": {"GGML_FA_SPLIT_KV": "0"}}
    sources = SimpleNamespace(server_mode={role: {"recipe": recipe, "port": 8074}},
                              master_lines=SimpleNamespace(line=lambda *args: 1),
                              master_path=tmp_path / "registry.yaml")
    proc = tmp_path / str(pid)
    proc.mkdir()
    (proc / "stat").write_text("123 (llama-server) S " + "0 " * 18 + "12345 0")
    # Build NUL-delimited records without ambiguous numeric escape sequences.
    (proc / "cmdline").write_bytes(b"\0".join([b"/bin/llama-server", b"--port", b"8074", b"-t", b"96", b""]))
    (proc / "environ").write_bytes(b"GGML_FA_SPLIT_KV=0\0")
    assert check_recipe_process(sources, role, pid, tmp_path) == []
    monkeypatch.setattr(stack_env, "_role_env_overrides", lambda _: {"GGML_FA_SPLIT_KV": "1"})
    (proc / "environ").write_bytes(b"GGML_FA_SPLIT_KV=1\0")
    assert check_recipe_process(sources, role, pid, tmp_path) == []
    monkeypatch.setattr(stack_env, "_role_env_overrides", lambda _: {})
    (proc / "cmdline").write_bytes(b"\0".join([b"/bin/llama-server", b"--port", b"8074", b"-t", b"48", b""]))
    (proc / "environ").write_bytes(b"GGML_FA_SPLIT_KV=1\0")
    findings = check_recipe_process(sources, role, pid, tmp_path)
    assert len(findings) == 2
    assert check_recipe_process(sources, role, pid + 1, tmp_path)
    (proc / "cmdline").write_bytes(b"\0".join([b"/bin/llama-server", b"--port", b"9999", b"-t", b"96", b""]))
    assert check_recipe_process(sources, role, pid, tmp_path)


def test_process_attestation_refuses_mixed_snapshots(tmp_path, monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace
    from scripts.validate.check_shared_with_derivations import check_recipe_process

    proc = tmp_path / "123"
    proc.mkdir()
    stat = proc / "stat"
    stat.write_text("123 (llama-server) S " + "0 " * 18 + "100 0")
    (proc / "environ").write_bytes(b"GGML_X=1\0")
    (proc / "cmdline").write_bytes(b"\0".join([b"llama-server", b"--port", b"1000", b"-t", b"96", b""]))
    sources = SimpleNamespace(server_mode={"fixture": {"port": 1000, "recipe": {"threads": 96, "env": {"GGML_X": "1"}}}},
                              master_lines=SimpleNamespace(line=lambda *args: 1), master_path=tmp_path / "registry")
    original_text = Path.read_text
    reads = 0
    def changed_start(path, *args, **kwargs):
        nonlocal reads
        text = original_text(path, *args, **kwargs)
        if path == stat:
            reads += 1
            if reads == 2:
                return text.replace("100 0", "101 0")
        return text
    monkeypatch.setattr(Path, "read_text", changed_start)
    assert "identity changed" in check_recipe_process(sources, "fixture", 123, tmp_path)[0].found
    monkeypatch.setattr(Path, "read_text", original_text)
    original_bytes = Path.read_bytes
    reads = 0
    def changed_env(path):
        nonlocal reads
        data = original_bytes(path)
        if path == proc / "environ":
            reads += 1
            if reads == 2:
                return b"GGML_X=2\0"
        return data
    monkeypatch.setattr(Path, "read_bytes", changed_env)
    assert "data changed" in check_recipe_process(sources, "fixture", 123, tmp_path)[0].found
