# NeuralTxt Offline Reward-Oracle Smoke (2026-06-21)

Observation-only A9 scorer run for `paperbd/neuraltxt-reward-tiny` on existing local seeding rows from `benchmarks/results/orchestrator/seeding_20260305_192809.json`.

This report intentionally commits summaries only. Intermediate row JSONL and scored JSONL stayed in `/mnt/raid0/llm/tmp/` and are not committed.

The scorer ran in a disposable CPU-only Python 3.11 venv at `/mnt/raid0/llm/tmp/a9-neuraltxt-cpu-venv` with `torch-2.12.1+cpu`, `transformers`, `safetensors`, and `huggingface_hub`. No llama servers, AutoPilot state, or live routing paths were changed.

Status: observation, not decision. The sample is a narrow local smoke, not an acceptance gate for using NeuralTxt labels in NEXT-A2/A3.
