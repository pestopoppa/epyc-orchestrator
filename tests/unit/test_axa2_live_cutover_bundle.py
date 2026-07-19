from __future__ import annotations

import json
import subprocess
from pathlib import Path

from scripts.benchmark import axa2_live_cutover_bundle as bundle


class FakeRunner:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, argv, **kwargs):  # noqa: ANN001 - subprocess-compatible test double
        self.calls.append(list(argv))
        stdout = ""
        if argv[:2] == ["git", "rev-parse"]:
            stdout = "abc123\n"
        elif argv[:2] == ["git", "branch"]:
            stdout = "main\n"
        elif argv[:2] == ["git", "status"]:
            stdout = ""
        elif argv[:2] == ["ps", "-eo"]:
            stdout = "123 bash bash\n"
        elif argv[:1] == ["rocm-smi"]:
            stdout = "No KFD PIDs\n"
        return subprocess.CompletedProcess(argv, 0, stdout, "")


def test_dry_bundle_writes_no_inference_operator_script(tmp_path: Path, monkeypatch) -> None:
    fake = FakeRunner()
    monkeypatch.setattr(bundle.subprocess, "run", fake)
    out = tmp_path / "axa2"
    args = bundle.parse_args(
        [
            "--output",
            str(out),
            "--policy-enabled",
            "--cpu-quant",
            "q4_k_m",
            "--gpu-quant",
            "q4_k_m",
        ]
    )

    summary = bundle.write_dry_bundle(args)

    assert summary["status"] == "prepared_no_inference"
    assert summary["no_inference"] is True
    assert summary["production_v6_touch_authorized"] is False
    assert summary["decision"]["should_cutover"] is True
    assert (out / "prompt.txt").exists()
    assert (out / "policy_decision.json").exists()
    operator = (out / "operator_run.sh").read_text()
    assert "--execute" in operator
    assert "--policy-enabled" in operator
    assert 'CPU_URL="${CPU_URL:?set CPU_URL to the CPU llama-server base URL}"' in operator
    assert '--cpu-url "$CPU_URL"' in operator
    assert "--cpu-url '${CPU_URL" not in operator
    assert "--quant-change-role-allowlist --cpu-quant" not in operator
    forbidden = ("llama-server", "llama-bench", "orchestrator_stack.py", "rocprof", "perf record")
    assert not any(any(word in " ".join(call) for word in forbidden) for call in fake.calls)

    loaded = json.loads((out / "summary.json").read_text())
    assert loaded["schema"] == bundle.SCHEMA


def test_first_char_divergence_handles_equal_prefixes() -> None:
    assert bundle.first_char_divergence("abc", "abc") is None
    assert bundle.first_char_divergence("abc", "abd") == 2
    assert bundle.first_char_divergence("abc", "abcd") == 3


def test_execute_bundle_policy_declined_still_records_continuity(tmp_path: Path, monkeypatch) -> None:
    def fake_post(base_url, payload, *, timeout_s):  # noqa: ANN001
        return {
            "ok": True,
            "status": 200,
            "elapsed_s": 0.01,
            "json": {"content": f"{base_url}:{payload['n_predict']}"},
        }

    fake = FakeRunner()
    monkeypatch.setattr(bundle.subprocess, "run", fake)
    monkeypatch.setattr(bundle, "post_completion", fake_post)
    out = tmp_path / "run"
    args = bundle.parse_args(
        [
            "--execute",
            "--output",
            str(out),
            "--cpu-url",
            "http://127.0.0.1:19001",
            "--gpu-url",
            "http://127.0.0.1:19002",
        ]
    )

    summary = bundle.execute_bundle(args)

    assert summary["status"] == "policy_declined"
    assert summary["decision"]["reason"] == "disabled"
    assert summary["continuity"]["first_char_divergence"] is not None
    assert (out / "responses" / "continuity_cpu.response.json").exists()
    assert (out / "responses" / "continuity_gpu.response.json").exists()
    assert not (out / "responses" / "cpu_prefix.response.json").exists()
    events = [json.loads(line)["event"] for line in (out / "events.jsonl").read_text().splitlines()]
    assert events == ["teleport_candidate", "fallback"]
