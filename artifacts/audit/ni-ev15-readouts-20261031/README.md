# EV-15 distributional readout source proof — 2026-10-31

## Scope and provenance

This is a pure NumPy implementation of the eight distributional comparison
readouts in Table 3 of Wang, Zhang, and Choi, “Improving LLM-as-a-Judge
Inference with the Judgment Distribution,” arXiv:2503.03064v2. The paper
identifies its license as CC BY 4.0; this proof and implementation cite the
paper and are independently derived from its Table 3 and Appendix B.1:
[paper HTML](https://arxiv.org/html/2503.03064v2),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

The implementation exposes mode, mean, rounded mean, median, first
percentile, risk-averse mean, quantile integral, and probability of
superiority. It accepts NumPy float32 or float64 arrays with arbitrary leading
batch dimensions and a final score axis, validates both distributions, and
does not mutate inputs. It does not integrate with a live judge, alter a score
policy, or emit measurement/quality/calibration claims.

Where Table 3 does not settle mode ties or exact nearest-score ties for rounded
mean, the API raises `ValueError`. Quantiles use the left inverse. Appendix
B.1's zero-over-zero mean convention is implemented as zero. Risk-averse mean
recomputes the complete mean-comparator denominator after shifting each
support by its lower semideviation. Probability of superiority calculates
upper-tail mass from actual scores strictly above each support point, so a
within-tolerance simplex residual cannot become fictitious win/loss mass.

## New hand-calculated fixture evidence

The focused tests include newly derived small fixtures, explicitly not the
historical intake-1169 vectors:

- K=2 checks the readout outputs against hand calculations.
- K=5 uses `p1=(.5,0,0,0,.5)`, `p2=(0,1,0,0,0)`; mean is positive and shifted
  RAM is negative.
- K=9 uses `p1=(0,.3,0,0,0,0,0,0,.7)`,
  `p2=(.2,0,0,0,0,0,0,0,.8)`; quantile integral is `+0.1` while independent
  probability of superiority is `−0.04`.
- A tolerated float64 K=1 residual still gives PS zero because every pair
  ties. A second PS check confirms antisymmetry and unchanged inputs.
- Additional checks cover arbitrary leading batch dimensions, float32 and
  float64 preservation, caller immutability, tied/degenerate cases, invalid
  probabilities, NaN, shape/dtype mismatches, and bounded large-K sum
  tolerance.

## Historical intake golden coverage gap

The intake-1169 K=5 RAM-vs-mean and K=9 quantile-integral-vs-PS entries were
searched in the owning ROOT intake artifact and reachable Git history. The
current entry at ROOT `research/intake_index.yaml` lines 89581–89650 records
that those goldens were produced but does not include their input vectors or
expected outputs. The reachable publication commit
`654cf0fee52205646adbb1dc21b5a3124c9b3454` changes the intake index and
`.research-session.json`; that session file only summarizes the vectors and
does not preserve numeric values. No reference-implementation repository was
opened or copied. Consequently, these new fixtures do not establish
conformance to the historical vectors, and EV-15a remains open pending their
recovery.

## Verification

Focused CPU-only tests passed on the released APP lane at
`b4c52752d489a1049948bfc9e5a3d6ced351c80c` before the source additions:

- `pytest -q tests/unit/test_distributional_readout.py`: 10 passed.
- CPU affinity: 128–129; lock: `/mnt/raid0/llm/tmp/compute-flags/prep-locks/32-33.lock`.
- GPU visibility disabled; OMP, MKL, OpenBLAS, NumExpr threads set to 1;
  900-second timeout; sampled maximum RSS 7,356 KiB (8 GiB ceiling).
- `ruff check` passed; `ruff format --check` passed; no whitespace errors.
- Repository `make gates` passed (schema, shellcheck, format, lint).

Raw outputs: [`focused-tests.log`](focused-tests.log), [`ruff.log`](ruff.log),
[`make-gates.log`](make-gates.log), and [`diff-check.log`](diff-check.log).
File hashes for the source, tests, and proof files are in [`SHA256SUMS`](SHA256SUMS).

SHA-256:

```
ca60873c24b304e048eb91d69d9bcb7c969b55e5ee6e8cf9ffa30b9c1c58c70d  src/typed_decisions/distributional_readout.py
48df8410caa78010c34f9950381c0a1b79727ea86443ff555eba64ffbded1ef9  tests/unit/test_distributional_readout.py
```
