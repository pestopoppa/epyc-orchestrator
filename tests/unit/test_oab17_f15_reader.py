"""OAB-17 F15: ScopedReader regular-file guard, per-file denial in definitions, per-thread alternatives."""
from __future__ import annotations

import os
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.api.routes.chat_pipeline import scout_stage as ss


def _scope(root, deny=()):
    return SimpleNamespace(root=str(root), read_denial=lambda p: "denied" if p in deny else None)


def test_fifo_with_source_suffix_does_not_block(tmp_path):
    fifo = tmp_path / "evil.c"
    os.mkfifo(fifo)
    reader = ss.ScopedReader(_scope(tmp_path))
    with pytest.raises(ValueError, match="not a regular file"):
        reader._text(fifo)                 # would block forever on open() before the fix


def test_definitions_skip_denied_files(tmp_path):
    ok, hidden = tmp_path / "a.c", tmp_path / "b.c"
    for f in (ok, hidden):
        f.write_text("int kernel_fn(int x) {\n return x;\n}\n")
    reader = ss.ScopedReader(_scope(tmp_path, deny={os.path.realpath(hidden)}))
    found = [p.name for p, _ in reader.definitions("kernel_fn")]
    assert found == ["a.c"]


def test_alternatives_are_per_thread(tmp_path):
    reader = ss.ScopedReader(_scope(tmp_path))
    reader._alt.value = ["main-thread"]
    seen = {}
    t = threading.Thread(target=lambda: seen.setdefault("v", getattr(reader._alt, "value", None)))
    t.start(); t.join()
    assert seen["v"] is None and reader._alt.value == ["main-thread"]
