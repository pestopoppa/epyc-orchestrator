"""Degenerate-generation detector for streamed chat completions (flag ``stream_degeneration_guard``).

Mirrors taskflow's DegenerationGuard (scripts/coordination/taskflow/harness/local_llm.py in
epyc-root, commit ff849bbd0). On Qwen3.6-35B-A3B about 1% of requests degenerate into repetitive
output and run to the output cap at near-zero MTP acceptance.

``feed(delta, timings)`` per streamed chunk returns ``(reason, detail)`` when the request should
be aborted, else ``None``. reason is ``"acceptance"`` or ``"repetition"``.
  acceptance: with per-chunk timings (request ``timings_per_token: true``), rolling draft acceptance
              over the last >= MTP_WINDOW predicted tokens is below MTP_MIN
  repetition: over the last REPEAT_WINDOW chars, the share of word NGRAM-grams that are repeats
              is >= REPEAT_MAX
"""

from __future__ import annotations

MTP_WINDOW, MTP_MIN = 512, 0.2
REPEAT_WINDOW, NGRAM, REPEAT_MAX, MIN_CHARS = 4000, 6, 0.6, 2000
MIN_GRAMS = 50

REASON_ACCEPTANCE = "acceptance"
REASON_REPETITION = "repetition"


class DegenerationGuard:
    def __init__(self) -> None:
        self.text = ""
        self.marks: list[tuple[int, int, int]] = []  # (predicted_n, draft_n, draft_accepted)
        self.predicted_n = 0

    def feed(self, delta: str = "", timings: dict | None = None) -> tuple[str, str] | None:
        if delta:
            self.text = (self.text + delta)[-REPEAT_WINDOW * 2:]
        if timings and timings.get("draft_n") is not None:
            self.marks.append((
                int(timings.get("predicted_n") or 0),
                int(timings.get("draft_n") or 0),
                int(timings.get("draft_n_accepted") or 0),
            ))
            p1, d1, a1 = self.marks[-1]
            self.predicted_n = max(self.predicted_n, p1)
            for p0, d0, a0 in reversed(self.marks[:-1]):
                if p1 - p0 >= MTP_WINDOW:
                    if d1 - d0 > 0 and (a1 - a0) / (d1 - d0) < MTP_MIN:
                        return REASON_ACCEPTANCE, f"mtp-acceptance {(a1 - a0) / (d1 - d0):.2f} over {p1 - p0} tokens"
                    break
        if len(self.text) >= MIN_CHARS:
            ratio = repeat_ratio(self.text[-REPEAT_WINDOW:])
            if ratio is not None and ratio >= REPEAT_MAX:
                return REASON_REPETITION, f"repeat-ratio {ratio:.2f}"
        return None


def repeat_ratio(text: str) -> float | None:
    """Share of word 6-grams that are repeats, or None when there are too few grams to judge."""
    w = text.split()
    grams = [tuple(w[i:i + NGRAM]) for i in range(max(0, len(w) - NGRAM + 1))]
    if len(grams) < MIN_GRAMS:
        return None
    return 1 - len(set(grams)) / len(grams)
