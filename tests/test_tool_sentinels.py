"""Static (no-inference) guards for the get_eval_secret tool-use sentinels.

The threat the earlier design got wrong: secrets were hardcoded in source/YAML,
so a model could grep them instead of calling the tool. These tests enforce the
corrected design:
  * secret VALUES are minted at runtime — NOT present in eval_secret.py source,
    NOT present as `expected:` in tool_sentinels.yaml;
  * the runtime ground-truth file lives OUTSIDE read_file's allowed roots;
  * each question references a valid name and pins the REPL path + substring
    scoring.
No orchestrator / model / network required.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parents[1]
TOOL_SENTINELS = _REPO / "scripts" / "autopilot" / "tool_sentinels.yaml"
EVAL_SECRET_SRC = _REPO / "src" / "tools" / "eval_secret.py"
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.tools.eval_secret import (  # noqa: E402
    SECRET_NAMES,
    SECRETS_PATH,
    _SECRETS,
)

# read_file's allowed roots (src/tools/file/read.py). The runtime secret file
# must be OUTSIDE these so the model-under-test can't read it.
READ_FILE_ROOTS = ("/mnt/raid0/llm/", "/tmp/")
_NAME_RE = re.compile(r'name=\\?"([a-zA-Z0-9_]+)\\?"')


def _load() -> list[dict]:
    return yaml.safe_load(TOOL_SENTINELS.read_text()) or []


def test_nonempty_repl_substring():
    qs = _load()
    assert len(qs) >= 3, f"need >=3 tool_use questions, got {len(qs)}"
    for q in qs:
        assert q.get("suite") == "tool_use"
        assert q.get("force_mode") == "repl", f"{q['id']} not repl-pinned"
        assert q.get("scoring_method") == "substring", f"{q['id']} not substring-scored"
        assert "get_eval_secret" in q["prompt"], f"{q['id']} doesn't invoke get_eval_secret"
        names = _NAME_RE.findall(q["prompt"])
        assert names, f"{q['id']}: no name=\"...\" arg"
        for n in names:
            assert n.lower() in SECRET_NAMES, f"{q['id']}: name {n!r} not in SECRET_NAMES"


def test_repl_prompts_require_executable_tool_code():
    """REPL-pinned sentinels must ask for executable tool code, not prose.

    force_mode="repl" sends the answer through the Python REPL. Asking for
    "no code" or plain-text output traps models in comment/prose-only turns and
    hides the actual tool-use signal behind a nudge failure.
    """
    forbidden = (
        "do not include any code",
        "no code",
        "do not output code",
        "plain text",
        "<answer>",
    )
    for q in _load():
        prompt = q["prompt"]
        lower = prompt.lower()
        assert "executable python" in lower, f"{q['id']}: prompt must demand executable Python"
        assert 'TOOL("get_eval_secret"' in prompt or 'CALL("get_eval_secret"' in prompt, (
            f"{q['id']}: prompt must show an executable get_eval_secret call"
        )
        assert "FINAL(" in prompt, f"{q['id']}: prompt must show FINAL(secret)"
        assert "discarded and scored incorrect" in lower, (
            f"{q['id']}: prompt must state the consequence of violating the REPL contract"
        )
        for phrase in forbidden:
            assert phrase not in lower, f"{q['id']}: prompt still contains prose-only phrase {phrase!r}"


def test_no_secret_values_in_source():
    """Secrets must be runtime-only: empty at import, and the source must not
    contain a baked-in secret literal (the `EVS-<hex>` shape)."""
    assert _SECRETS == {}, "eval_secret._SECRETS must be empty at import (runtime-only)"
    src = EVAL_SECRET_SRC.read_text()
    assert not re.search(r"EVS-[0-9a-f]{16}", src), "a real secret literal is committed in source"


def test_no_secret_values_in_yaml():
    """tool_sentinels.yaml must NOT carry real expected values — only the
    non-matching runtime placeholder (so a repo grep yields nothing usable)."""
    for q in _load():
        exp = str(q.get("expected", ""))
        assert exp == "__resolved_at_runtime__", f"{q['id']}: expected leaks a value: {exp!r}"
        assert not re.search(r"EVS-[0-9a-f]{16}", exp)


def test_runtime_secret_file_is_unreadable_by_read_file():
    """The persisted ground-truth path must be outside read_file's allowed roots
    so the model-under-test cannot read it via the read_file tool."""
    p = str(SECRETS_PATH)
    assert not any(p.startswith(r) for r in READ_FILE_ROOTS), (
        f"runtime secrets path {p} is inside a read_file root — model could read it"
    )


def test_runtime_generation_roundtrip(tmp_path, monkeypatch):
    """Generate → persist → load yields high-entropy values for every name, and
    those values are NOT present in source or YAML."""
    import src.tools.eval_secret as es

    monkeypatch.setattr(es, "SECRETS_PATH", tmp_path / "secrets.json")
    minted = es.generate_and_persist_secrets()
    assert set(minted) == set(SECRET_NAMES)
    loaded = es.load_persisted_secrets()
    assert loaded == minted
    src = EVAL_SECRET_SRC.read_text()
    yml = TOOL_SENTINELS.read_text()
    for name in SECRET_NAMES:
        val = es.get_eval_secret(name)
        assert val == minted[name] and len(val) >= 12
        assert val not in src and val not in yml, f"{name} secret leaked to disk"
    assert es.get_eval_secret("nope").startswith("ERROR")


def test_load_or_create_is_idempotent_across_reloads(tmp_path, monkeypatch):
    """A reload (fresh in-memory, existing valid file) must LOAD the same secret,
    not re-mint — else the harness's `expected` drifts mid-trial and tool_use
    scores wrong even when the tool was called (the 2026-06-05 bug)."""
    import src.tools.eval_secret as es

    monkeypatch.setattr(es, "SECRETS_PATH", tmp_path / "s.json")
    es._SECRETS = {}
    first = es.generate_and_persist_secrets()
    # simulate an orchestrator reload: new process => empty memory, same file.
    es._SECRETS = {}
    second = es.generate_and_persist_secrets()
    assert second == first, "reload re-minted instead of loading the existing file"
    assert es.load_persisted_secrets() == first
    # repeated registration within one process is also stable.
    assert es.generate_and_persist_secrets() == first


def test_concurrent_first_start_converges_on_one_set(tmp_path):
    """uvicorn --workers first-start: N processes racing on an ABSENT file must
    all converge on ONE secret set (atomic load-or-create, no torn writes)."""
    import json as _json
    import os as _os
    import subprocess
    import sys as _sys

    secrets_path = tmp_path / "race_secrets.json"
    code = (
        "import json, sys\n"
        f"sys.path.insert(0, {str(_REPO)!r})\n"
        "import src.tools.eval_secret as es\n"
        "print(json.dumps(es.generate_and_persist_secrets()))\n"
    )
    env = dict(_os.environ, EVAL_SECRETS_PATH=str(secrets_path))
    procs = [
        subprocess.Popen(
            [_sys.executable, "-c", code], env=env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        for _ in range(8)
    ]
    outs = []
    for p in procs:
        out, err = p.communicate(timeout=60)
        assert p.returncode == 0, f"worker failed: {err}"
        outs.append(_json.loads(out.strip().splitlines()[-1]))
    first = outs[0]
    assert all(o == first for o in outs), f"workers diverged: {outs}"
    assert _json.loads(secrets_path.read_text()) == first  # file == winning set
