"""REPL tap header carries optional role/request_id/task_id; the old format is unchanged when they are absent."""

from __future__ import annotations

import src.graph.repl_tap as tap


def _write(tmp_path, monkeypatch, **kw):
    p = tmp_path / "repl_tap.log"
    monkeypatch.setattr(tap, "_REPL_TAP_PATH", str(p))
    monkeypatch.setattr(tap, "_IN_PYTEST", False)
    tap.tap_write_repl_exec("print(1)", 3, **kw)
    return p.read_text()


def test_meta_on_header_line(tmp_path, monkeypatch):
    t = _write(tmp_path, monkeypatch, role="worker_general", request_id="r-1", task_id="t-9")
    head = t.splitlines()[1]
    assert head.endswith("REPL turn 3 | role=worker_general request_id=r-1 task_id=t-9")
    assert "[turn 3] $ python3 <<'CODE'\nprint(1)\nCODE\n" in t


def test_old_format_when_absent(tmp_path, monkeypatch):
    t = _write(tmp_path, monkeypatch)
    assert t.splitlines()[1].endswith("REPL turn 3")
    (tmp_path / "repl_tap.log").unlink()
    t2 = _write(tmp_path, monkeypatch, role="a b")
    assert t2.splitlines()[1].endswith("REPL turn 3 | role=a")
