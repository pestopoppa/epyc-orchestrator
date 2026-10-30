"""EV-5 process feedback. Diagnostic attribution; never changes scorer verdicts.

The backend is an explicitly owned, sequential verifier session. The caller must
finish and drain generation before entering it. This module never stops serving
processes or loads a model until every selected generator passes EV-6.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import time
from typing import Any

from src.autopilot_core.verification_families import cross_family_status


def verification_prompt(problem: str, solution: str) -> str:
    """The pinned ThinkPRM model-card prompt (no gold answer exposed)."""
    return (
        "You are given a math problem and a proposed step-by-step solution:\n\n"
        f"[Math Problem]\n\n{problem}\n\n[Solution]\n\n{solution}\n\n"
        "Review and critique each step in the proposed solution to determine whether "
        "each step is correct. If the solution is incomplete, only verify the provided steps"
    )


def parse_step_verdicts(text: str) -> list[dict[str, Any]]:
    # Native generations also say "The step is boxed{correct}" inside a
    # numbered Step N block. Attribute only explicit labels within that block.
    headers = list(re.finditer(r"Step\s+(\d+)(?:\s*:|\s+is\b)", text, re.IGNORECASE))
    verdicts = []
    for index, header in enumerate(headers):
        end = headers[index + 1].start() if index + 1 < len(headers) else len(text)
        body = text[header.end() : end]
        label = re.search(r"\\boxed\s*\{(correct|incorrect)\}", body, re.IGNORECASE)
        if label:
            verdicts.append(
                {
                    "step": int(header[1]),
                    "verdict": label[1].lower(),
                    "span": [header.end() + label.start(), header.end() + label.end()],
                }
            )
    return verdicts


def binary_score(response: dict, label_ids: dict[str, int]) -> dict[str, Any]:
    """Normalize two OBSERVED log probabilities; missing support is never zero.

    Native /completion pre-sampling logprobs include no sampling/grammar bias.
    An equal positive logit bias nominates the two single-token labels into the
    returned top-2. Its common additive shift cancels in their conditional ratio.
    """
    rows = response.get("completion_probabilities") or response.get("probs") or []
    top = rows[0].get("top_logprobs", []) if rows else []
    found = {int(p["id"]): float(p["logprob"]) for p in top if "id" in p and "logprob" in p}
    raw = {label: found.get(tid) for label, tid in label_ids.items()}
    out = {
        "label_ids": label_ids,
        "label_logprobs": raw,
        "raw_response": response,
        "method": "equal_bias_two_label_conditional_v1",
        "common_logit_bias": 100.0,
    }
    if any(v is None or not math.isfinite(v) for v in raw.values()):
        return {**out, "score": None, "status": "both_label_logprobs_unavailable"}
    yes, no = raw["Yes"], raw["No"]
    shift = max(yes, no)
    a, b = math.exp(yes - shift), math.exp(no - shift)
    return {**out, "score": a / (a + b), "status": "measured"}


def selected_results(results: list, n: int) -> list:
    if not 0 <= n <= 100:
        raise ValueError("process verification_n must be between 0 and 100")
    eligible = [
        r
        for r in results
        if not r.error
        and r.disposition == "scored"
        and r.answer
        and r.eval_partition == "core"
        and math.isfinite(r.confidence)
        and 0 <= r.confidence <= 1
    ]
    return sorted(eligible, key=lambda r: (abs(r.confidence - 0.5), r.qid or r.question_id))[:n]


def verify_results(
    results: list,
    backend: Any,
    *,
    n: int,
    artifact_path: Path,
    deadline_monotonic: float | None = None,
) -> dict:
    selected = selected_results(results, n)
    # Entire preflight precedes __enter__: same-family models never load.
    for row in selected:
        independent, reason = cross_family_status(row.route_used, backend.model_identity)
        if not independent:
            raise ValueError(f"process verifier rejected {row.route_used!r}: {reason}")
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    completed = 0
    with artifact_path.open("x", encoding="utf-8") as fh:

        def append(record):
            fh.write(json.dumps(record, allow_nan=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

        try:
            if selected:
                with backend as session:
                    for row in selected:
                        if (
                            deadline_monotonic is not None
                            and time.monotonic() >= deadline_monotonic
                        ):
                            break
                        feedback = session.verify(row.prompt, row.answer)
                        feedback.update(
                            {
                                "generator": row.route_used,
                                "verifier": backend.model_identity,
                                "independence": "cross_family",
                                "confidence": row.confidence,
                                "confidence_source": row.confidence_source,
                                "uncertainty_distance": abs(row.confidence - 0.5),
                            }
                        )
                        row.details["process_verification"] = feedback
                        append({"qid": row.qid or row.question_id, "details": row.details})
                        completed += 1
            partial = completed < len(selected)
            append(
                {
                    "kind": "process_verification_partial"
                    if partial
                    else "process_verification_complete",
                    "n": completed,
                }
            )
        except BaseException as exc:
            append(
                {
                    "kind": "process_verification_interrupted",
                    "n": completed,
                    "error": type(exc).__name__,
                }
            )
            raise
    return {
        "enabled": True,
        "selected_n": len(selected),
        "completed_n": completed,
        "requested_n": n,
        "partial": partial,
        "artifact_path": str(artifact_path),
        "artifact_fsync_returned": True,
        "binary_proxy_ties_are_not_uncertainty": any(
            r.confidence_source == "binary_correctness_proxy" for r in selected
        ),
    }
