# Jev OOD calibration input fixture

`val.jsonl` is copied byte-for-byte from [`scienthoon/jev-ood-calibration` at commit `914d87ab16517f14e833cff63d73b51235b14bd6`](https://github.com/scienthoon/jev-ood-calibration/tree/914d87ab16517f14e833cff63d73b51235b14bd6/data).

- Upstream path: `data/val.jsonl`
- SHA-256: `763949cdf07f422cbb3c4bd040392944034ae4caf23a07de08fcef04cfd1eb33`
- Upstream README declares its generated `data/` files CC0. The repository's MIT license applies to code, not the frozen data file.
- The fixture has 900 JSONL rows (300 ticket records × `choice`, `score`, `noul`). The generator independently corrupts each answer label with 5% probability. Adapter outputs retain those source observed labels unchanged and store clean oracle labels separately.

The adapter's `--fresh-seed` reimplements the pinned CLI recipe: generate 3,000 tickets with 5% label noise, then use the same seed to shuffle ticket groups and take the 10% validation split (300 tickets). It records the generation and split seeds, recipe commit and digest, serialized population digest, exact overlap with the frozen states, and makes no novelty or exposure claim. The generated `fresh_known` and `tier_ablated` rows share state identities; the ablation drops only `customer_tier` from the model input.

Usage (model-free):

```bash
python -m src.typed_decisions.jev_ood_calibration \
  --input tests/fixtures/typed_decisions/jev_ood_calibration/val.jsonl \
  --output-dir /tmp/td17-fixtures \
  --fresh-seed 20261101
```

The model-facing JSONL files contain no labels. Each has a separate `.sealed-truth.jsonl` companion. To feed a JSONL arm to typed-decision calls, use `load_fixture_inputs()` and pass each `FixtureInput.state` and `.questions` to `run_mode_benchmark`; the serialized questions match the existing bench loader schema. This adapter does not call a model or compute metrics.

## Upstream code attribution

The deterministic fresh-population recipe is reimplemented from `data/generate.py` at the pinned commit above; no downloaded upstream code is executed. Upstream code license (MIT):

Copyright (c) 2026 scienthoon

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
