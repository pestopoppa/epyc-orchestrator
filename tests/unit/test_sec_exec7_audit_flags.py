"""SEC-EXEC-7: first_seen_host and write_outside_root flags in the tool-call audit log + daily summary."""

import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.features import reset_features
from src.repl_environment import REPLEnvironment, write_audit
from src.registry.tool_registry import ToolRegistry, load_from_yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "security"))
import tool_audit_summary as S  # noqa: E402


def _make_web_repl():
    registry = ToolRegistry()
    load_from_yaml(registry, ROOT / "orchestration" / "tool_registry.yaml")
    registry.load_permissions_from_registry(ROOT / "orchestration" / "model_registry.yaml")
    return REPLEnvironment(context="t", tool_registry=registry, role="frontdoor")


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

    yield _set
    monkeypatch.undo()
    reset_features()


@pytest.fixture
def narrow_tmp(tmp_path, monkeypatch):
    """gettempdir() -> a private dir, so '/var/tmp' and the repo count as OUTSIDE the roots."""
    d = tmp_path / "scratch"
    d.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(d))
    return d


def _records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


# ── run_shell: static check ─────────────────────────────────────────────────


@pytest.mark.parametrize("parts,expected", [
    (["ls", "-la"], None),
    (["cat", "a.txt"], None),
    (["sort", "-o", "out.txt", "in.txt"], ["out.txt"]),
    (["sort", "--output=out.txt", "in.txt"], ["out.txt"]),
    (["sort", "-oout.txt", "in.txt"], ["out.txt"]),
    (["sort", "-r", "in.txt"], None),
    (["uniq", "in.txt", "out.txt"], ["out.txt"]),
    (["uniq", "-f", "2", "in.txt"], None),
    (["uniq", "-f", "2", "in.txt", "out.txt"], ["out.txt"]),
    (["git", "diff", "--output=patch.diff"], ["patch.diff"]),
    (["git", "diff", "--output", "patch.diff"], ["patch.diff"]),
    (["git", "status"], None),
    (["sed", "-i", "s/a/b/", "f.txt"], ["f.txt"]),
    (["sed", "-i.bak", "-e", "s/a/b/", "f.txt"], ["f.txt"]),
    (["sed", "-n", "p", "f.txt"], None),
    (["sed", "--in-place", "s/a/b/"], []),
])
def test_shell_write_targets(parts, expected):
    assert write_audit.shell_write_targets(parts) == expected


def test_shell_write_flag_roots_and_unknown_targets(tmp_path):
    roots = [os.path.realpath(str(tmp_path))]
    inside = str(tmp_path / "out.txt")
    assert write_audit.shell_write_flag(["sort", "-o", inside, "x"], "/", roots) is False
    assert write_audit.shell_write_flag(["sort", "-o", "/var/tmp/out.txt", "x"], "/", roots) is True
    assert write_audit.shell_write_flag(["sort", "-o", "rel.txt", "x"], str(tmp_path), roots) is False
    assert write_audit.shell_write_flag(["sort", "-o", "rel.txt", "x"], "/srv", roots) is True
    assert write_audit.shell_write_flag(["sed", "--in-place", "s/a/b/"], str(tmp_path), roots) is True  # unknown target
    assert write_audit.shell_write_flag(["ls"], "/", roots) is False
    # a symlink out of the root is judged by where it points
    link = tmp_path / "link"
    link.symlink_to("/var/tmp")
    assert write_audit.shell_write_flag(["sort", "-o", str(link / "x"), "i"], "/", roots) is True


def test_repl_shell_records_write_flag_only_when_the_flag_is_on(audit, flags, narrow_tmp):
    done = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
    repl = REPLEnvironment(context="t")
    with patch("src.repl_environment.external_access.subprocess.run", return_value=done):
        flags(exec_write_flags=False)
        repl._run_shell("sort -o out.txt in.txt")
        flags(exec_write_flags=True)
        repl._run_shell("sort -o out.txt in.txt")  # cwd = project root: outside the roots
        repl._run_shell(f"sort -o {narrow_tmp}/out.txt in.txt")
        repl._run_shell("ls")
        repl._run_shell("python3 -c 'print(1)'")  # refused before launch: not analysed
    recs = _records(audit)
    assert [r["write_outside_root"] for r in recs] == [None, True, False, False, None]
    assert recs[-1]["outcome"] == "refused"


# ── run_python_code: audit hook ─────────────────────────────────────────────


def _var_tmp_target():
    p = f"/var/tmp/secexec7-{uuid.uuid4().hex}.txt"
    try:
        with open(p, "w"):
            pass
        os.unlink(p)
    except OSError:
        pytest.skip("/var/tmp not writable")
    return p


def test_python_hook_flags_out_of_root_writes_and_not_in_root_ones(audit, flags, narrow_tmp):
    flags(exec_write_flags=True)
    target = _var_tmp_target()
    repl = REPLEnvironment(context="t")
    try:
        inside = repl._run_python_code(
            "import os\nopen(os.path.join(os.environ['HOME'], 'ok.txt'), 'w').write('x')\n"
            "print(open('/etc/hostname').read() != '')\nprint('INSIDE-DONE')\n"
        )
        outside = repl._run_python_code(f"open({target!r}, 'w').write('x')\nprint('OUTSIDE-DONE')\n")
        removed = repl._run_python_code(f"import os\nos.remove({target!r})\nprint('REMOVED')\n")
        renamed = repl._run_python_code(
            "import os, tempfile\nd = tempfile.mkdtemp()\nopen(d + '/a', 'w').close()\n"
            f"os.rename(d + '/a', {target!r})\nprint('RENAMED')\n"
        )
    finally:
        if os.path.exists(target):
            os.unlink(target)
    assert "INSIDE-DONE" in inside and "OUTSIDE-DONE" in outside and "REMOVED" in removed and "RENAMED" in renamed
    recs = _records(audit)
    assert [r["write_outside_root"] for r in recs] == [False, True, True, True]
    assert all(r["tool"] == "run_python_code" and r["outcome"] == "ok" for r in recs)


def test_python_hook_records_subprocess_spawns(audit, flags, narrow_tmp):
    flags(exec_write_flags=True)
    REPLEnvironment(context="t")._run_python_code("import subprocess\nsubprocess.run(['true'])\n")
    REPLEnvironment(context="t")._run_python_code("print(1)\n")
    recs = _records(audit)
    assert [r["spawned_process"] for r in recs] == [True, False]


def test_python_hook_is_transparent_to_output_and_tracebacks(audit, flags, narrow_tmp):
    repl = REPLEnvironment(context="t")
    code = "import sys\nprint('out'); print('err', file=sys.stderr)\n\ndef f():\n    raise ValueError('boom')\nf()\n"
    flags(exec_write_flags=False)
    plain = repl._run_python_code(code)
    flags(exec_write_flags=True)
    hooked = repl._run_python_code(code)
    assert "ValueError: boom" in plain and "ValueError: boom" in hooked
    assert "_epyc_wa_main" not in hooked and "<string>" not in hooked
    assert hooked.count("File ") == plain.count("File ")  # same frames: the bootstrap frame is dropped
    syn = repl._run_python_code("def (:\n")
    assert "SyntaxError" in syn and "_epyc_wa_main" not in syn
    sx = repl._run_python_code("import sys\nsys.exit(3)\n")
    assert _records(audit)[-1]["exit"] == 3 and sx is not None


def test_python_flag_off_launches_exactly_as_before(audit, flags):
    flags(exec_write_flags=False)
    seen = {}
    real = subprocess.run

    def spy(argv, *a, **kw):
        seen["argv"] = argv
        return real(argv, *a, **kw)

    with patch("src.repl_environment.external_access.subprocess.run", spy):
        REPLEnvironment(context="t")._run_python_code("print('hi')")
    assert seen["argv"][0] == "python3" and seen["argv"][1].endswith(".py") and "-c" not in seen["argv"]
    assert _records(audit)[-1]["write_outside_root"] is None


def test_unstarted_hook_is_not_analysed_rather_than_clean(tmp_path):
    side = tmp_path / "side"
    side.write_text("")  # the bootstrap never wrote its start marker
    assert write_audit.read_python_flags(str(side), ["/x"]) == (None, None)
    assert write_audit.read_python_flags(str(tmp_path / "missing"), ["/x"]) == (None, None)
    side.write_text("S\t\nW\t/x/a\n")
    assert write_audit.read_python_flags(str(side), ["/x"]) == (False, False)
    side.write_text("S\t\nW\t/x/a\nW\t/y/b\nP\t\n")
    assert write_audit.read_python_flags(str(side), ["/x"]) == (True, True)
    side.write_text("S\t\nW\t?\n")  # a path the hook could not resolve is flagged, never waved through
    assert write_audit.read_python_flags(str(side), ["/x"]) == (True, False)


# ── first-seen destination host ─────────────────────────────────────────────


def test_first_seen_host_persists_only_successful_fetches(tmp_path):
    seen = str(tmp_path / "seen.txt")
    assert write_audit.first_seen_host("a.example.com", seen, persist=False) is True
    assert write_audit.first_seen_host("a.example.com", seen, persist=False) is True  # failed fetch: still new
    assert write_audit.first_seen_host("a.example.com", seen, persist=True) is True
    assert write_audit.first_seen_host("a.example.com", seen, persist=True) is False
    assert write_audit.first_seen_host(None, seen, persist=True) is None
    # survives a restart: a fresh in-memory cache reads the file
    write_audit._seen.clear()
    assert write_audit.first_seen_host("a.example.com", seen, persist=True) is False
    # another worker's addition is seen without a restart
    Path(seen).write_text(Path(seen).read_text() + "b.example.com\n")
    assert write_audit.first_seen_host("b.example.com", seen, persist=True) is False


def test_repl_web_fetch_marks_first_seen_then_not(audit):
    resp = MagicMock(text="x", headers={"Content-Type": "text/plain"})
    resp.raise_for_status = MagicMock()
    repl = _make_web_repl()
    with patch("src.repl_environment.web_guard.fetch", return_value=resp):
        repl._web_fetch("https://Docs.Example.com/a")
        repl._web_fetch("https://docs.example.com/b")
        repl._web_fetch("https://other.example.com/")
    recs = _records(audit)
    assert [(r["host"], r["first_seen_host"]) for r in recs] == [
        ("docs.example.com", True), ("docs.example.com", False), ("other.example.com", True),
    ]
    with patch("src.repl_environment.web_guard.fetch", side_effect=RuntimeError("down")):
        repl._web_fetch("https://flaky.example.com/")
        repl._web_fetch("https://flaky.example.com/")
    errs = _records(audit)[-2:]
    assert [r["first_seen_host"] for r in errs] == [True, True] and errs[0]["outcome"] == "error"


def test_non_web_tools_carry_no_host(audit):
    REPLEnvironment(context="t")._run_shell("echo hi")
    rec = _records(audit)[-1]
    assert rec["host"] is None and rec["first_seen_host"] is None and rec["write_outside_root"] is None


# ── zero-token daily summary ────────────────────────────────────────────────


def _rec(ts, tool="run_shell", outcome="ok", **kw):
    base = {
        "ts": ts, "tool": tool, "text": "t", "sha256": "ab" * 32, "outcome": outcome, "role": "frontdoor",
        "request_id": "r1", "refusal_reason": None, "host": None, "first_seen_host": None,
        "write_outside_root": None, "spawned_process": None,
    }
    base.update(kw)
    return base


def test_summary_lists_flagged_calls_across_rotated_files(tmp_path, capsys):
    log = tmp_path / "tool_call_audit.jsonl"
    now = time.time()
    old = [_rec(now - 10 * 86400, first_seen_host=True, tool="web_fetch", host="ancient.example.com")]
    rotated = [
        _rec(now - 3600, tool="web_fetch", host="new.example.com", first_seen_host=True),
        _rec(now - 3500, tool="web_fetch", host="seen.example.com", first_seen_host=False),
        _rec(now - 3000, tool="run_python_code", write_outside_root=True, spawned_process=True, text="open('/x','w')"),
    ]
    live = [
        _rec(now - 60, tool="run_python_code", write_outside_root=False),
        _rec(now - 50, outcome="refused", refusal_reason="EXEC CONTAINMENT: interpreter"),
        _rec(now - 40, tool="web_fetch", outcome="refused", host="169.254.169.254", first_seen_host=True,
             refusal_reason="IPv4 address in non-global range 169.254.0.0/16"),
    ]
    (tmp_path / "tool_call_audit.jsonl.2").write_text("\n".join(map(json.dumps, old)) + "\n")
    (tmp_path / "tool_call_audit.jsonl.1").write_text("\n".join(map(json.dumps, rotated)) + "\nnot json\n")
    log.write_text("\n".join(map(json.dumps, live)) + "\n")
    files = S.log_files(str(log))
    assert [f.name for f in files] == ["tool_call_audit.jsonl.2", "tool_call_audit.jsonl.1", "tool_call_audit.jsonl"]
    records, bad = S.read_records(files, now - 86400)
    summary = S.summarize(records, bad)
    assert summary["total"] == 6 and summary["unparseable_lines"] == 1
    assert {r["host"] for r in summary["first_seen_host"]} == {"new.example.com", "169.254.169.254"}
    assert len(summary["write_outside_root"]) == 1 and len(summary["spawned_process"]) == 1
    assert {(r["tool"], r["count"]) for r in summary["refusals"]} == {("run_shell", 1), ("web_fetch", 1)}
    out_dir = tmp_path / "daily"
    assert S.main(["--log", str(log), "--out-dir", str(out_dir), "--json"]) == 0
    md = next(out_dir.glob("*.md")).read_text()
    assert "new.example.com" in md and "ancient.example.com" not in md and "non-global range" in md
    assert next(out_dir.glob("*.json")).exists()


def test_summary_without_a_log_is_an_error_not_an_empty_report(tmp_path):
    assert S.main(["--log", str(tmp_path / "nope.jsonl")]) == 2
