"""Model table for the probe v2 direct-call picker validation (OFF vs hardened, per model) and the code-mode arms.

``run_probe.py --model-key 35b|27b|flash`` selects one row (its ``model_id`` becomes the unit-id prefix and the ``model`` field of every
record, so the three models never collide in one results file). The launcher (not this file) owns the server and its private port;
this table carries identity and serving profile only, taken from ``orchestration/launch_manifest.yaml`` / ``model_registry.yaml`` and
``SELECTOR_HARDENING.md`` section 3. Verify the weight path and kernel against the launcher's precond before a live run.
"""

from __future__ import annotations

MODELS: dict[str, dict[str, str]] = {
    "35b": {"model_id": "35b", "weights": "/mnt/raid0/llm/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf", "kernel": "cpu",
            "registry_role": "frontdoor (+ worker_general alias)", "profile": "MoE A3B Q8, CPU: prompt tokens are expensive, prefix-cache stability matters"},
    "27b": {"model_id": "27b", "weights": "/mnt/raid0/llm/models/Qwen3.8-27B-Q8_0.gguf", "kernel": "gpu",
            "registry_role": "architect_critic / coder_escalation", "profile": "dense Q8 on the MI210: fast prefill; REPL arms floored in v1, native A scored 0.30"},
    "flash": {"model_id": "flash", "weights": "/mnt/raid0/llm/models/unsloth/Qwen3.8-Flash-Next-GGUF/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf",
              "kernel": "cpu", "registry_role": "architect_general", "profile": "UD-IQ4_XS, the most heavily quantized model of the three: largest distractor penalty expected"},
}


def resolve(key: str) -> dict[str, str]:
    if key not in MODELS:
        raise KeyError(f"unknown model key {key!r}; known: {sorted(MODELS)}")
    return MODELS[key]
