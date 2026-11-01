# DCP-8 offline paired instruction assembly

Source commit: `19655f323b23870c18e241067bbe04dbb2fc69c3`; base `3182846824ad50ec6201fde523f25aa1c5ecab51`. This instrument covers exactly the two first-party CJ-7c cases, clamp and unique. It assembles one native DCP bundle per case and two native root LM prompts using the sealed instruction pair; it invokes no model, produces no quality or timing verdict, and does not change the live/default DCP path. The seal over case identity, initial file and both instructions is SHA-256 `d70514c5492cf9c0b425ce18fd5d84d0c0e364c8e6760ffb6d56095dca73f8a2`. Raw CLI output: [pairs.json](pairs.json), SHA-256 `b49c1bdaee0bdae9fcacf81860f9d9f3316d731553d73f05eae1606927d54a32`.

Git blob identities at the source commit:

| Path | Git blob SHA-1 |
|---|---|
| `src/context_discovery.py` | `923e748929bb163f2c45446b75a24e2528b9c7af` |
| `src/context_assembly.py` | `9ba50c358e61766ef63340580a70016d4f3a5954` |
| `src/prompt_builders/builder.py` | `8047ae193820d82f8d3d5a8f207662a66be71a86` |
| `src/prompt_builders/review.py` | `5f4b7f5400a1f0694e5a7e81303d1273c91eed29` |
| `scripts/benchmark/paired_edit_instrument.py` | `2c46ff6df6a854bf4cacfa788eed307688381725` |
| `scripts/benchmark/dcp_paired_instruction.py` | `63ad6eefac225b2c7aeb65d2d888d37117ba4a12` |
| `tests/unit/test_dcp_paired_instruction.py` | `a385480a4659e7ebec8a643bb2f3a0ad6ea0b42c` |

[Focused controls](focused.txt): 49 passed including native DCP, root prompt and malformed pair/drift controls. [Broader controls](validation.txt): 202 passed, 1 failed; the existing architect investigate assertion expects text absent from its fallback. [Isolated baseline reproduction](baseline-failure.txt) fails the same test without importing this instrument. The architect implementation and test, DCP assembly, prompt builder and paired fixture were byte-identical to the base. The native DCP assembly callee was not edited; its available GitNexus graph was 219 commits stale and reported HIGH impact for five callers.

Runs used the R4 physical guard under ToolUID, lock and affinity CPUs 78–79, admission and supervisor, 900-second deadline and OMP/BLAS thread limits of one. The supervisor applies a sampled 7 GiB aggregate RSS stop against the standing 8 GiB cap; this is not a kernel-enforced memory peak. Focused run sampled peak 176,058,368 bytes, returned zero and drained its child tree. These are execution custody controls, not model-performance measurements.
