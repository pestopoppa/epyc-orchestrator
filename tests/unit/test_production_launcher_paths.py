"""Production shell entry path controls; Python children are intercepted, not run.

No kernel, model, HTTP health call, live process or runtime config is accessed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

SOURCE_ROOT = Path(os.environ.get("LAUNCHER_SOURCE_ROOT", Path(__file__).resolve().parents[2]))


def launch(tmp_path, *, overrides=None, dot_env="", args=("--status",)):
    root = tmp_path / "checkout"
    for relative in (
        "scripts/server/launch_production.sh",
        "scripts/lib/env.sh",
        "scripts/server/orchestrator_stack.py",
    ):
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SOURCE_ROOT / relative, destination)
    if dot_env:
        (root / ".env").write_text(dot_env)
    commands = tmp_path / "commands.jsonl"
    control_bin = tmp_path / "control-bin"
    control_bin.mkdir()
    interpreter = control_bin / "python3"
    interpreter.write_text(
        f"#!{sys.executable}\n"
        "import json,os,sys\n"
        'keys=("ORCHESTRATOR_PATHS_PROJECT_ROOT","ORCHESTRATOR_PATHS_LOG_DIR",'
        '"ORCHESTRATOR_PATHS_REGISTRY_PATH","ORCHESTRATOR_PATHS_SESSIONS_DIR")\n'
        'record={"argv":sys.argv[1:],"paths":{k:os.environ.get(k) for k in keys}}\n'
        'with open(os.environ["CONTROL_COMMANDS"],"a") as out:\n'
        ' out.write(json.dumps(record)+"\\n")\n'
    )
    interpreter.chmod(0o755)
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith("ORCHESTRATOR_PATHS_")
        and k not in ("PROJECT_ROOT", "LOG_DIR", "LLM_ROOT")
    }
    env.update(
        PATH=str(control_bin) + ":" + os.environ["PATH"],
        CONTROL_COMMANDS=str(commands),
        ORCHESTRATOR_PATHS_LLAMA_CPP_BIN=str(tmp_path / "unused-kernel"),
        ORCHESTRATOR_PATHS_LLAMA_MTMD=str(tmp_path / "unused-mtmd"),
    )
    env.update(overrides or {})
    result = subprocess.run(
        ["bash", str(root / "scripts/server/launch_production.sh"), *args],
        env=env,
        text=True,
        capture_output=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    return root, [json.loads(line) for line in commands.read_text().splitlines()]


def test_default_production_entry_uses_its_checkout(tmp_path):
    root, rows = launch(tmp_path)
    assert len(rows) == 1
    assert rows[0]["argv"] == [str(root / "scripts/server/orchestrator_stack.py"), "status"]
    paths = rows[0]["paths"]
    assert paths["ORCHESTRATOR_PATHS_PROJECT_ROOT"] == str(root)
    assert paths["ORCHESTRATOR_PATHS_LOG_DIR"] == str(root / "logs")
    assert paths["ORCHESTRATOR_PATHS_REGISTRY_PATH"] == str(
        root / "orchestration/model_registry.yaml"
    )
    assert paths["ORCHESTRATOR_PATHS_SESSIONS_DIR"] == str(
        root / "orchestration/repl_memory/sessions"
    )


def test_default_no_argument_entry_passes_paths_to_every_python_child(tmp_path):
    root, rows = launch(tmp_path, args=())
    assert [row["argv"][-1] for row in rows] == ["hot", "start", "status"]
    assert all(row["paths"]["ORCHESTRATOR_PATHS_LOG_DIR"] == str(root / "logs") for row in rows)


@pytest.mark.parametrize("source", ["environment", "dot-env"])
def test_explicit_project_root_preserved(tmp_path, source):
    custom = str(tmp_path / "custom-project")
    options = (
        {"overrides": {"ORCHESTRATOR_PATHS_PROJECT_ROOT": custom}}
        if source == "environment"
        else {"dot_env": f"ORCHESTRATOR_PATHS_PROJECT_ROOT={custom}\n"}
    )
    _, rows = launch(tmp_path, **options)
    assert rows[0]["paths"]["ORCHESTRATOR_PATHS_PROJECT_ROOT"] == custom
    assert rows[0]["paths"]["ORCHESTRATOR_PATHS_LOG_DIR"] == custom + "/logs"


@pytest.mark.parametrize("source", ["environment", "dot-env"])
def test_explicit_log_destination_preserved(tmp_path, source):
    custom = str(tmp_path / "retained-logs")
    options = (
        {"overrides": {"ORCHESTRATOR_PATHS_LOG_DIR": custom}}
        if source == "environment"
        else {"dot_env": f"ORCHESTRATOR_PATHS_LOG_DIR={custom}\n"}
    )
    _, rows = launch(tmp_path, **options)
    assert rows[0]["paths"]["ORCHESTRATOR_PATHS_LOG_DIR"] == custom


def test_conditional_dot_env_project_root_wins_over_wrapper_default(tmp_path):
    custom = str(tmp_path / "conditional-project")
    _, rows = launch(
        tmp_path,
        dot_env=(
            'ORCHESTRATOR_PATHS_PROJECT_ROOT="${ORCHESTRATOR_PATHS_PROJECT_ROOT:-' + custom + '}"\n'
        ),
    )
    assert rows[0]["paths"]["ORCHESTRATOR_PATHS_PROJECT_ROOT"] == custom
    assert rows[0]["paths"]["ORCHESTRATOR_PATHS_LOG_DIR"] == custom + "/logs"


@pytest.mark.parametrize("source", ["default", "environment", "conditional-dot-env"])
def test_unrelated_document_launcher_source_prefix_keeps_prior_paths(tmp_path, source):
    """Run the unrelated caller through its real source statement, then stop.

    The OCR exec is excluded; this checks shared-library compatibility only.
    """
    root = tmp_path / "other-checkout"
    relative = "scripts/document/start_lightonocr_llama.sh"
    for name in (relative, "scripts/lib/env.sh"):
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SOURCE_ROOT / name, target)
    custom = str(tmp_path / "custom-project")
    if source == "conditional-dot-env":
        (root / ".env").write_text(
            'ORCHESTRATOR_PATHS_PROJECT_ROOT="${ORCHESTRATOR_PATHS_PROJECT_ROOT:-' + custom + '}"\n'
        )
    lines = (root / relative).read_text().splitlines()
    end = lines.index('source "${SCRIPT_DIR}/../lib/env.sh"')
    prefix = root / "scripts/document/source-prefix-control.sh"
    prefix.write_text(
        "\n".join(lines[: end + 1]) + "\n"
        '"$CONTROL_PYTHON" -c \'import json,os; print(json.dumps({k:os.environ.get(k) '
        'for k in ["ORCHESTRATOR_PATHS_PROJECT_ROOT",'
        '"ORCHESTRATOR_PATHS_LOG_DIR"]}))\'\n'
    )
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith("ORCHESTRATOR_PATHS_")
        and k not in ("_EPYC_ENV_PROJECT_ROOT_DEFAULT", "PROJECT_ROOT", "LOG_DIR", "LLM_ROOT")
    }
    env.update(
        CONTROL_PYTHON=sys.executable,
        ORCHESTRATOR_PATHS_LLAMA_CPP_BIN=str(tmp_path / "unused-kernel"),
        ORCHESTRATOR_PATHS_LLAMA_MTMD=str(tmp_path / "unused-mtmd"),
    )
    if source == "environment":
        env["ORCHESTRATOR_PATHS_PROJECT_ROOT"] = custom
    result = subprocess.run(
        ["bash", str(prefix)], env=env, text=True, capture_output=True, timeout=15
    )
    assert result.returncode == 0, result.stderr
    paths = json.loads(result.stdout)
    expected = "/mnt/raid0/llm/claude" if source == "default" else custom
    assert paths == {
        "ORCHESTRATOR_PATHS_PROJECT_ROOT": expected,
        "ORCHESTRATOR_PATHS_LOG_DIR": expected + "/logs",
    }
