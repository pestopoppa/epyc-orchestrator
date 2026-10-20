| lane | threshold | non-finite | cos min (docs) | cos min (queries) | cos mean (docs) | norm range | verdict |
|---|---|---|---|---|---|---|---|
| gguf-bf16 | 0.99 | 0 | 0.999990 | 0.999994 | 0.999997 | 1.0000-1.0000 | PASS |
| gguf-q8_0 | 0.98 | 0 | 0.999063 | 0.999776 | 0.999838 | 1.0000-1.0000 | PASS |
| st-bf16 | 0.99 | 0 | 0.999919 | 0.999948 | 0.999967 | 0.9963-1.0039 | PASS |
| st-fp32-textonly | None | 0 | 1.000000 | 1.000000 | 1.000000 | 1.0000-1.0000 | None |

PARTIAL: 211 chunks (>= 200: True); query set = proxy-first-line (45 queries), NOT the 45 behavioural questions (intake-1357#03)
