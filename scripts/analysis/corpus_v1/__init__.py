"""Near-miss decision corpus v1 builder (H4 RC-3).

Package of per-source miners + an assembler that builds the versioned
reviewer-calibration corpus `nearmiss-v1`. See README.md for the full design.

No module here performs any model/inference call. All defect synthesis is
rule-based. Journals and dataset files are read-only inputs.
"""
