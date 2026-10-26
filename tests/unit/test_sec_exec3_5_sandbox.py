"""SEC-EXEC-3/5: kernel-enforced no-network and credential read-deny for model-exec children.

Kernel tests need Landlock + seccomp and SKIP (loudly, with the probe result) on a host that
cannot enforce; the fail-closed tests below them always run. Every kernel assertion has an
unsandboxed control run, so a pass cannot be vacuous.
"""

import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from src.features import reset_features
from src.repl_environment import REPLEnvironment, exec_sandbox, fence_kernel
from src.repl_environment.exec_containment import child_env
from src.repl_environment.exec_sandbox import ContainmentUnavailable, Policy


@pytest.fixture(scope="module")
def caps():
    exec_sandbox.reset_caches()
    c = exec_sandbox.capabilities()
    if not (c["landlock"] and c["netdeny"]):
        pytest.skip(f"host cannot enforce kernel containment: {c}")
    return c


@pytest.fixture
def audit(tmp_path, monkeypatch):
    path = tmp_path / "audit.jsonl"
    monkeypatch.setenv("ORCHESTRATOR_TOOL_AUDIT_LOG", str(path))
    return path


@pytest.fixture
def flags(monkeypatch):
    def _set(**kw):
        for name, on in kw.items():
            monkeypatch.setenv("ORCHESTRATOR_" + name.upper(), "1" if on else "0")
        reset_features()
        exec_sandbox.reset_caches()

    yield _set
    monkeypatch.undo()
    reset_features()
    exec_sandbox.reset_caches()


def _records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def _run(argv, *, policy=None, cwd, allow=(), deny=None, timeout=60):
    """Run argv, wrapped in the sandbox when ``policy`` is given. Returns CompletedProcess."""
    env = child_env(task_id="sandbox-test")
    if policy is not None:
        argv, extra = exec_sandbox.wrap(
            argv, policy=policy, extra_allow=[env["HOME"], cwd, *allow], deny_paths=deny if deny is not None else [],
        )
        env.update(extra)
    return subprocess.run(argv, env=env, cwd=str(cwd), capture_output=True, text=True, timeout=timeout)


# ── always-run: fail closed ─────────────────────────────────────────────────


@pytest.mark.parametrize("missing", ["landlock", "netdeny"])
def test_wrap_refuses_when_the_host_cannot_enforce(missing, monkeypatch):
    got = {"abi": 6, "landlock": True, "netdeny": True, "ts": 0.0}
    got[missing] = False
    monkeypatch.setattr(exec_sandbox, "capabilities", lambda: got)
    with pytest.raises(ContainmentUnavailable):
        exec_sandbox.wrap(["true"], policy=Policy(), deny_paths=[])


def test_shim_never_runs_the_command_when_it_cannot_restrict(tmp_path):
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"mode": "no-such-mode"}))
    proc = subprocess.run(
        [sys.executable, "-c", fence_kernel.shim_source(), str(spec), "--", "python3", "-c", "print('RAN')"],
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 126 and "RAN" not in proc.stdout
    assert "kernel enforcement failed" in proc.stderr


def test_shim_never_runs_the_command_on_a_corrupt_spec(tmp_path):
    spec = tmp_path / "spec.json"
    spec.write_text("{not json")
    proc = subprocess.run(
        [sys.executable, "-c", fence_kernel.shim_source(), str(spec), "--", "python3", "-c", "print('RAN')"],
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 126 and "RAN" not in proc.stdout


def test_credential_path_set_covers_the_handoff_list(tmp_path):
    home = str(tmp_path / "h")
    hf = str(tmp_path / "hf")
    paths = set(exec_sandbox.credential_paths(env={"HF_HOME": hf}, homes=[home]))
    for want in (
        f"{hf}/token", f"{hf}/stored_tokens",
        "/mnt/raid0/llm/cache/huggingface/token", "/mnt/raid0/llm/cache/huggingface/stored_tokens",
        f"{home}/.cache/huggingface/token", f"{home}/.git-credentials", f"{home}/.config/gh",
        f"{home}/.netrc", f"{home}/.ssh", f"{home}/.aws", f"{home}/.claude", f"{home}/.codex",
        f"{home}/.config/opencode",
    ):
        assert want in paths, want


def test_operator_extra_deny_paths_are_honoured(tmp_path):
    extra = str(tmp_path / "vault" / "secret")
    paths = exec_sandbox.credential_paths(env={exec_sandbox.DENY_EXTRA_ENV: extra}, homes=[str(tmp_path / "h")])
    assert extra in paths


def test_landlock_rules_deny_a_path_that_does_not_exist_yet(tmp_path):
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "keep.txt").write_text("x")
    token = str(tmp_path / "d" / "token")  # not created
    rules = fence_kernel.build_landlock_rules({token}, special={"/proc": fence_kernel.KIND_PROC})
    granted = {r[0] for r in rules}
    assert str(tmp_path / "d") not in granted, "the container directory was granted wholesale"
    assert str(tmp_path / "d" / "keep.txt") in granted
    assert token not in granted
    assert ["/proc", fence_kernel.KIND_PROC] in rules


def test_extra_grants_carve_fences_and_keep_a_private_run_directory(tmp_path):
    from src.repl_environment import _fence_kernel_shim as shim

    vault = tmp_path / "vault"
    (vault / "descendant").mkdir(parents=True)
    private = tmp_path / "private"
    private.mkdir()
    assert fence_kernel.build_landlock_rules({str(vault)}, root=str(vault)) == []
    assert fence_kernel.build_landlock_rules({str(vault)}, root=str(vault / "descendant")) == []
    spec = {"fenced": [str(vault)], "rules": []}
    extra = fence_kernel.safe_extra_landlock_rules([str(tmp_path), str(private)], spec)
    assert [str(private), fence_kernel.KIND_FULL] in extra
    assert [str(tmp_path), fence_kernel.KIND_FULL] not in extra
    assert [str(vault), fence_kernel.KIND_FULL] not in extra
    # Revalidation in the standalone shim also rejects legacy/broad and
    # symlink-repointed grants, independently of the parent's rule generation.
    alias = tmp_path / "alias"
    alias.symlink_to(vault, target_is_directory=True)
    safe = shim._k_safe_grants([str(tmp_path), str(alias), [str(private), shim._K_FULL]], [str(vault)])
    assert safe == [(str(private), shim._K_FULL)]


def test_extra_grants_preserve_denied_symlink_mount_and_proc_aliases(tmp_path, monkeypatch):
    from src.repl_environment import _fence_kernel_shim as shim

    physical = tmp_path / "physical"
    alias = tmp_path / "alias"
    (physical / "vault").mkdir(parents=True)
    (alias / "vault").mkdir(parents=True)
    mounts = [("fixture-dev", "/", str(physical)), ("fixture-dev", "/", str(alias))]
    monkeypatch.setattr(fence_kernel, "read_mounts", lambda: mounts)
    fenced = {str(physical / "vault")}
    expanded = fence_kernel.with_mount_aliases(fenced, mounts)
    assert str(alias / "vault") in expanded
    spec = {"fenced": list(fenced), "rules": [["/proc", fence_kernel.KIND_PROC]]}
    assert fence_kernel.safe_extra_landlock_rules([str(alias / "vault"), "/proc"], spec) == []
    assert shim._k_safe_grants([[str(alias / "vault"), shim._K_FULL]], expanded) == []

    symlink = tmp_path / "denied-link"
    symlink.symlink_to(physical / "vault", target_is_directory=True)
    assert fence_kernel.safe_extra_landlock_rules([str(physical / "vault")],
        {"fenced": [str(symlink)], "rules": []}) == []


@pytest.mark.parametrize("shape", ["ancestor", "exact", "descendant", "symlink"])
def test_sandbox_extras_never_reopen_a_denied_subtree(caps, tmp_path, shape):
    vault = tmp_path / "vault"
    nested = vault / "descendant"
    nested.mkdir(parents=True)
    secret = nested / "secret.txt"
    secret.write_text("FAKE-EXTRA-GRANT-SECRET")
    private = tmp_path / "private-run"
    private.mkdir()
    alias = tmp_path / "vault-alias"
    alias.symlink_to(vault, target_is_directory=True)
    extra = {"ancestor": tmp_path, "exact": vault, "descendant": nested, "symlink": alias}[shape]
    script = private / "probe.py"
    script.write_text(f"""from pathlib import Path
try:
    print(Path({str(secret)!r}).read_text())
except PermissionError:
    print('CRED-DENIED')
target = Path({str(private / 'owned.txt')!r})
target.write_text('owned')
print(target.read_text())
""")
    policy = Policy(no_network=False, hide_foreign_proc=False, scope=False)
    result = _run([sys.executable, str(script)], policy=policy, cwd=private,
                  allow=[str(extra)], deny=[str(vault)])
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.splitlines() == ["CRED-DENIED", "owned"]
    assert secret.read_text() == "FAKE-EXTRA-GRANT-SECRET"


def test_extra_ancestor_does_not_allow_creating_a_future_denied_file(caps, tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    future = vault / "future-token"
    private = tmp_path / "private-run"
    private.mkdir()
    script = private / "probe.py"
    script.write_text(f"""from pathlib import Path
try:
    Path({str(future)!r}).write_text('SHOULD-NOT-EXIST')
    print('CREATED')
except PermissionError:
    print('CREATE-DENIED')
Path({str(private / 'ok')!r}).write_text('owned')
print('OWNED-WRITE')
""")
    result = _run([sys.executable, str(script)],
                  policy=Policy(no_network=False, hide_foreign_proc=False, scope=False),
                  cwd=private, allow=[str(tmp_path)], deny=[str(future)])
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.splitlines() == ["CREATE-DENIED", "OWNED-WRITE"]
    assert not future.exists()


def test_large_carved_rules_roundtrip_without_an_oversized_environment(caps, tmp_path):
    shared = tmp_path / "shared"
    shared.mkdir()
    vault = shared / "vault"
    vault.mkdir()
    secret = vault / "token"
    secret.write_text("FAKE-LARGE-RULE-SECRET")
    private = shared / "private"
    private.mkdir()
    for index in range(1500):
        (shared / ("sibling-" + str(index) + "x" * 100)).write_text("fixture")
    script = private / "probe.py"
    script.write_text(f"""from pathlib import Path
try:
    print(Path({str(secret)!r}).read_text())
except PermissionError:
    print('CRED-DENIED')
Path('owned').write_text('owned')
print(Path('owned').read_text())
""")
    argv, extra = exec_sandbox.wrap([sys.executable, str(script)],
        policy=Policy(no_network=False, hide_foreign_proc=False, scope=False),
        extra_allow=[str(shared)], deny_paths=[str(vault)])
    payload = Path(extra["EPYC_FENCE_KERNEL_EXTRA_FILE"]).read_bytes()
    assert len(payload) > 131072  # larger than Linux's single env-string limit
    assert len(json.dumps(extra)) < 4096
    env = child_env(task_id="large-extra-control")
    env.update(extra)
    result = subprocess.run(argv, cwd=private, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.splitlines() == ["CRED-DENIED", "owned"]


def test_extra_grant_parent_refuses_tampered_cached_bytes(tmp_path, monkeypatch):
    path = tmp_path / "tampered.json"
    path.write_text('[["/unexpected", 1]]')
    monkeypatch.setattr(fence_kernel, "_write_cached", lambda *args: str(path))
    with pytest.raises(ValueError, match="digest mismatch"):
        fence_kernel.extra_grant_environment([])


@pytest.mark.parametrize("kind", ["symlink", "directory", "oversize", "invalid", "nonlist"])
def test_extra_file_transport_fails_closed(kind, tmp_path):
    from src.repl_environment import _fence_kernel_shim as shim

    path = tmp_path / "extra"
    digest = hashlib.sha256(b"[]").hexdigest()
    if kind == "symlink":
        target = tmp_path / "target"
        target.write_text("[]")
        path.symlink_to(target)
    elif kind == "directory":
        path.mkdir()
    elif kind == "oversize":
        with path.open("wb") as handle:
            handle.truncate(shim._K_MAX_EXTRA_BYTES + 1)
    else:
        data = b"{" if kind == "invalid" else b"{}"
        path.write_bytes(data)
        digest = hashlib.sha256(data).hexdigest()
    with pytest.raises((OSError, ValueError)):
        shim._k_read_extra_file(str(path), digest)


def test_extra_file_tamper_refuses_before_the_child_command(tmp_path):
    private = tmp_path / "private"
    private.mkdir()
    extra = fence_kernel.extra_grant_environment([[str(private), fence_kernel.KIND_FULL]])
    path = Path(extra["EPYC_FENCE_KERNEL_EXTRA_FILE"])
    original = path.read_bytes()
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"mode": "landlock", "rules": [], "fenced": []}))
    try:
        path.write_text("[]")
        env = child_env(task_id="tampered-extra-control")
        env.update(extra)
        result = subprocess.run([sys.executable, "-c", fence_kernel.shim_source(), str(spec),
                                 "--", sys.executable, "-c", "print('SHOULD-NOT-RUN')"],
                                env=env, cwd=private, capture_output=True, text=True, timeout=30)
        assert result.returncode == 126
        assert "digest mismatch" in result.stderr
        assert "SHOULD-NOT-RUN" not in result.stdout
    finally:
        path.write_bytes(original)


def test_child_env_has_a_token_free_hf_home_only_when_contained(monkeypatch, flags):
    monkeypatch.setenv("HF_HOME", "/mnt/raid0/llm/cache/huggingface")
    monkeypatch.setenv("HF_TOKEN", "hf_FAKEplantedtoken0123456789")
    flags(exec_containment=False)
    assert "HF_HOME" not in child_env(task_id="a")
    flags(exec_containment=True)
    env = child_env(task_id="a")
    assert env["HF_HOME"].startswith(env["HOME"]) and "raid0/llm/cache" not in env["HF_HOME"]
    assert env["HF_HUB_OFFLINE"] == "1" and "HF_TOKEN" not in env


# ── kernel: no network ──────────────────────────────────────────────────────

NET_PROBE = textwrap.dedent("""
    import errno, socket, sys
    port = int(sys.argv[1])
    out = []
    def attempt(name, fn):
        try:
            fn(); out.append(name + "=OK")
        except PermissionError:
            out.append(name + "=EACCES")
        except OSError as e:
            out.append(name + "=" + errno.errorcode.get(e.errno, str(e.errno)))
    def tcp():
        s = socket.create_connection(("127.0.0.1", port), timeout=3); s.close()
    attempt("tcp_connect", tcp)
    attempt("tcp_socket", lambda: socket.socket(socket.AF_INET, socket.SOCK_STREAM).close())
    attempt("udp_socket", lambda: socket.socket(socket.AF_INET, socket.SOCK_DGRAM).close())
    attempt("v6_socket", lambda: socket.socket(socket.AF_INET6, socket.SOCK_STREAM).close())
    attempt("raw_socket", lambda: socket.socket(socket.AF_INET, socket.SOCK_RAW, 1).close())
    attempt("unix_socket", lambda: socket.socket(socket.AF_UNIX, socket.SOCK_STREAM).close())
    def pair():
        a, b = socket.socketpair(); a.send(b"x"); assert b.recv(1) == b"x"
    attempt("socketpair", pair)
    print(" ".join(out))
""")


@pytest.fixture
def listener():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(8)
    yield srv.getsockname()[1]
    srv.close()


def _net(out):
    return dict(p.split("=", 1) for p in out.split())


def test_child_has_no_network_but_the_control_does(caps, listener, tmp_path):
    script = tmp_path / "net.py"
    script.write_text(NET_PROBE)
    control = _run(["python3", str(script), str(listener)], cwd=tmp_path)
    assert _net(control.stdout)["tcp_connect"] == "OK", control.stderr  # the probe itself works
    pol = Policy(deny_credentials=False, hide_foreign_proc=False)
    boxed = _run(["python3", str(script), str(listener)], policy=pol, cwd=tmp_path)
    assert boxed.returncode == 0, boxed.stderr
    got = _net(boxed.stdout)
    for name in ("tcp_connect", "tcp_socket", "udp_socket", "v6_socket", "raw_socket", "unix_socket"):
        assert got[name] == "EACCES", (name, got)
    assert got["socketpair"] == "OK", got  # anonymous AF_UNIX pairs (asyncio, multiprocessing) still work


def test_child_cannot_borrow_a_socket_or_peer_memory(caps, tmp_path):
    script = tmp_path / "sys.py"
    script.write_text(textwrap.dedent("""
        import ctypes, errno, os
        libc = ctypes.CDLL(None, use_errno=True)
        nr = {"x86_64": {"io_uring_setup": 425, "pidfd_getfd": 438, "ptrace": 101},
              "aarch64": {"io_uring_setup": 425, "pidfd_getfd": 438, "ptrace": 117}}[os.uname().machine]
        res = []
        for name, n in nr.items():
            rc = libc.syscall(n, 0, 0, 0, 0)
            res.append(f"{name}={'EACCES' if rc == -1 and ctypes.get_errno() == errno.EACCES else rc}")
        print(" ".join(res))
    """))
    boxed = _run(["python3", str(script)], policy=Policy(deny_credentials=False, hide_foreign_proc=False), cwd=tmp_path)
    assert boxed.returncode == 0, boxed.stderr
    assert all(v == "EACCES" for v in _net(boxed.stdout).values()), boxed.stdout


# ── kernel: credential files ────────────────────────────────────────────────


def _make_home(tmp_path):
    home = tmp_path / "home"
    hf = tmp_path / "hf"
    (home / ".config" / "gh").mkdir(parents=True)
    (home / ".claude").mkdir()
    (home / "notes").mkdir()
    hf.mkdir()
    (hf / "stored_tokens").mkdir()
    files = {
        "git": home / ".git-credentials", "gh": home / ".config" / "gh" / "hosts.yml",
        "claude": home / ".claude" / ".credentials.json", "hf_token": hf / "token",
        "hf_stored": hf / "stored_tokens" / "work", "ordinary": home / "notes" / "readme.txt",
    }
    for p in files.values():
        p.write_text("FAKE-SECRET-VALUE")
    return home, hf, files


def test_credential_files_are_unreadable_to_the_child_and_ordinary_files_are_not(caps, tmp_path):
    home, hf, files = _make_home(tmp_path)
    work = tmp_path / "work"
    work.mkdir()
    creds = exec_sandbox.credential_paths(env={"HF_HOME": str(hf)}, homes=[str(home)])
    script = work / "read.py"
    script.write_text(textwrap.dedent(f"""
        import json, os
        paths = {json.dumps({k: str(v) for k, v in files.items()})}
        res = {{}}
        for k, p in paths.items():
            try:
                open(p).read(); res[k] = "READ"
            except PermissionError:
                res[k] = "EACCES"
        for k, p in {{"git": paths["git"], "hf_token": paths["hf_token"]}}.items():
            try:
                open(p, "a").write("x"); res[k + "_write"] = "WROTE"
            except PermissionError:
                res[k + "_write"] = "EACCES"
        try:
            open(os.path.join({str(home / ".config" / "gh")!r}, "planted"), "w").write("x"); res["gh_create"] = "CREATED"
        except PermissionError:
            res["gh_create"] = "EACCES"
        late = {str(home / ".netrc")!r}  # created AFTER the spec was built
        try:
            open(late).read(); res["netrc_late"] = "READ"
        except PermissionError:
            res["netrc_late"] = "EACCES"
        except FileNotFoundError:
            res["netrc_late"] = "MISSING"
        print(json.dumps(res))
    """))
    pol = Policy(no_network=False, hide_foreign_proc=False)
    control = _run(["python3", str(script)], cwd=work)
    assert "READ" in control.stdout, control.stderr  # control: unsandboxed, the secrets ARE readable
    spec_probe = exec_sandbox.wrap(["true"], policy=pol, deny_paths=creds)  # builds + caches the spec
    assert spec_probe
    (home / ".netrc").write_text("FAKE-SECRET-VALUE")
    boxed = _run(["python3", str(script)], policy=pol, cwd=work, allow=[work], deny=creds)
    assert boxed.returncode == 0, boxed.stderr
    got = json.loads(boxed.stdout)
    for k in ("git", "gh", "claude", "hf_token", "hf_stored"):
        assert got[k] == "EACCES", (k, got)
    assert got["ordinary"] == "READ", got
    assert got["git_write"] == "EACCES" and got["hf_token_write"] == "EACCES" and got["gh_create"] == "EACCES", got
    assert got["netrc_late"] == "EACCES", got


def test_other_processes_proc_files_are_hidden_but_own_proc_works(caps, tmp_path):
    planted = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        env={"PATH": os.environ.get("PATH", ""), "PLANTED_SECRET": "sk-FAKE-planted-in-a-peer"},
    )
    try:
        script = tmp_path / "proc.py"
        script.write_text(textwrap.dedent(f"""
            import os
            res = []
            for name in ("environ", "cmdline", "mem"):
                try:
                    open("/proc/{planted.pid}/" + name, "rb").read(16); res.append(name + "=READ")
                except PermissionError:
                    res.append(name + "=EACCES")
                except OSError as e:
                    res.append(name + "=" + str(e.errno))
            res.append("self_status=" + ("READ" if "Name" in open("/proc/self/status").read() else "NO"))
            res.append("cpuinfo=" + ("READ" if open("/proc/cpuinfo").read(10) else "NO"))
            print(" ".join(res))
        """))
        control = _run(["python3", str(script)], cwd=tmp_path)
        assert "environ=READ" in control.stdout, control.stdout + control.stderr  # same-uid peer IS readable
        boxed = _run(["python3", str(script)], policy=Policy(no_network=False, deny_credentials=False), cwd=tmp_path)
        assert boxed.returncode == 0, boxed.stderr
        got = _net(boxed.stdout)
        assert got["environ"] == "EACCES" and got["cmdline"] == "EACCES", got
        assert got["mem"] in ("EACCES", "1"), got
        assert got["self_status"] == "READ" and got["cpuinfo"] == "READ", got
        # a descendant that only runs ordinary tools still works
        sub = _run(["python3", "-c", "import subprocess; print(subprocess.run(['echo','ok'],capture_output=True,text=True).stdout.strip())"],
                   policy=Policy(no_network=False, deny_credentials=False), cwd=tmp_path)
        assert sub.stdout.strip() == "ok", sub.stderr
    finally:
        planted.kill()
        planted.wait(timeout=10)


def test_child_cannot_signal_processes_outside_its_domain(caps, tmp_path):
    if caps["abi"] < 6:
        pytest.skip(f"Landlock signal scoping needs ABI 6, host has {caps['abi']}")
    victim = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        script = tmp_path / "kill.py"
        script.write_text(f"import os, signal\ntry:\n    os.kill({victim.pid}, signal.SIGUSR2); print('SENT')\nexcept PermissionError:\n    print('EPERM')\n")
        boxed = _run(["python3", str(script)], policy=Policy(no_network=False, deny_credentials=False, hide_foreign_proc=False), cwd=tmp_path)
        assert boxed.stdout.strip() == "EPERM", boxed.stdout + boxed.stderr
        time.sleep(0.2)
        assert victim.poll() is None, "the victim was signalled"
    finally:
        victim.send_signal(signal.SIGKILL)
        victim.wait(timeout=10)


# ── through the REPL tools, behind the flag ─────────────────────────────────


def test_repl_children_are_contained_when_the_flag_is_on(caps, audit, flags, tmp_path, listener, monkeypatch):
    secret = tmp_path / "vault" / "secret.txt"
    secret.parent.mkdir()
    secret.write_text("FAKE-SECRET-VALUE")
    monkeypatch.setenv(exec_sandbox.DENY_EXTRA_ENV, str(secret))
    flags(exec_containment=True)
    repl = REPLEnvironment(context="t")
    out = repl._run_python_code(
        "import socket\n"
        f"try:\n    socket.create_connection(('127.0.0.1', {listener}), timeout=3); print('CONNECTED')\n"
        "except PermissionError:\n    print('NET-DENIED')\n"
        f"try:\n    print(open({str(secret)!r}).read())\nexcept PermissionError:\n    print('CRED-DENIED')\n"
        "print('sum', 2 + 3)\n"
    )
    assert "NET-DENIED" in out and "CRED-DENIED" in out and "sum 5" in out, out
    assert "FAKE-SECRET-VALUE" not in out and "CONNECTED" not in out
    assert "hello" in repl._run_shell("echo hello")
    assert [r["outcome"] for r in _records(audit)] == ["ok", "ok"]


def test_repl_flag_off_children_are_not_sandboxed(audit, flags):
    flags(exec_containment=False)
    out = REPLEnvironment(context="t")._run_python_code(
        "import socket\nsocket.socket(socket.AF_INET, socket.SOCK_STREAM).close()\nprint('SOCKET-OK')\n"
    )
    assert "SOCKET-OK" in out


@pytest.mark.parametrize("tool,call", [
    ("run_python_code", lambda r: r._run_python_code("print('SHOULD-NOT-RUN')")),
    ("run_shell", lambda r: r._run_shell("echo SHOULD-NOT-RUN")),
])
def test_repl_refuses_visibly_when_containment_is_unavailable(tool, call, audit, flags, monkeypatch, caplog):
    flags(exec_containment=True)
    monkeypatch.setattr(
        exec_sandbox, "capabilities", lambda: {"abi": 0, "landlock": False, "netdeny": False, "ts": 0.0},
    )
    with caplog.at_level("WARNING"):
        out = call(REPLEnvironment(context="t"))
    assert out.startswith("[ERROR: EXEC CONTAINMENT") and "SHOULD-NOT-RUN" not in out
    assert any("REFUSED" in r.getMessage() for r in caplog.records)
    rec = _records(audit)[-1]
    assert rec["tool"] == tool and rec["outcome"] == "refused" and "Landlock" in rec["refusal_reason"]
