"""td16_sidecar_job.sh: the Qwen3.x sidecar must start with --reasoning off, else the JSON-mode arm returns empty content
(td16-label-word-bias-20261109: J answered 0 of 56, 'Empty generation after ~40 s')."""
import subprocess
from pathlib import Path

SH = Path(__file__).resolve().parents[2] / "scripts" / "benchmark" / "td16_sidecar_job.sh"


def test_script_parses():
    assert subprocess.run(["bash", "-n", str(SH)]).returncode == 0


def test_reasoning_off_is_default_for_qwen3_and_gated_by_preflight():
    t = SH.read_text()
    assert 'REASONING_ARGS="--reasoning off"' in t and "*qwen3*" in t
    assert "$REASONING_ARGS" in t.split("setsid taskset")[1].split("&\n")[0]
    assert "first-reply preflight FAIL" in t


def test_reasoning_arg_selection_logic():
    snippet = SH.read_text().split("if [ -z \"${REASONING_ARGS+x}\" ]; then")[1].split("\nfi\n")[0]
    for model, want in (("Qwen3.6-35B-A3B-MTP-Q8_0.gguf", "--reasoning off"), ("gemma-4-26B.gguf", "")):
        out = subprocess.run(["bash", "-c", f'MODEL={model}\nif [ -z "${{REASONING_ARGS+x}}" ]; then{snippet}\nfi\necho -n "$REASONING_ARGS"'], capture_output=True, text=True)
        assert out.stdout == want, out
