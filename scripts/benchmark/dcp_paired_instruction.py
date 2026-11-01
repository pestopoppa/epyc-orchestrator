"""Offline DCP prompt-construction instrument for the two first-party CJ-7c cases.

Run with ``python -m scripts.benchmark.dcp_paired_instruction``. No model calls,
execution of edits, workspace search, or quality/performance judgments are made.
The native DCP bundle is assembled once per case from its frozen fixture file;
the same rendered bundle is then supplied to both native prompt-builder calls.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from scripts.benchmark.paired_edit_instrument import FIXTURES
from src.context_discovery import DiscoveredHit, assemble_delegation_bundle, render_bundle
from src.prompt_builders import build_root_lm_prompt

CONDITIONS = ("descriptive", "minimal")
FIXTURE_SEAL_SHA256 = "d70514c5492cf9c0b425ce18fd5d84d0c0e364c8e6760ffb6d56095dca73f8a2"


@dataclass(frozen=True)
class SealedCase:
    problem_id: str
    initial: str
    instructions: tuple[tuple[str, str], ...]


# Pin source fixture values at import, including both conditions, before assembly.
SEALED_CASES = tuple(
    SealedCase(f.id, f.initial, (("descriptive", f.descriptive), ("minimal", f.minimal)))
    for f in FIXTURES
)


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def assemble_pairs(cases: tuple[SealedCase, ...] = SEALED_CASES) -> list[dict]:
    """Produce complete paired prompts, rejecting altered or incomplete fixture manifests."""
    if _digest([(c.problem_id, c.initial, c.instructions) for c in SEALED_CASES]) != FIXTURE_SEAL_SHA256:
        raise ValueError("CJ-7c fixture seal changed")
    if not cases or len(cases) != len(SEALED_CASES):
        raise ValueError("missing or extra sealed case")
    expected = {case.problem_id: case for case in SEALED_CASES}
    if len(expected) != len(SEALED_CASES) or len({c.problem_id for c in cases}) != len(cases):
        raise ValueError("duplicate problem identity")
    if {c.problem_id for c in cases} != set(expected):
        raise ValueError("unknown or missing problem identity")
    result = []
    for case in cases:
        names = [name for name, _ in case.instructions]
        if len(names) != len(CONDITIONS) or set(names) != set(CONDITIONS):
            raise ValueError("missing, duplicate or unknown condition")
        if case != expected[case.problem_id]:
            raise ValueError("sealed case changed")
        instructions = dict(case.instructions)
        if any(not text.strip() for text in instructions.values()) or len(set(instructions.values())) != 2:
            raise ValueError("invalid paired instructions")
        path = f"{case.problem_id}.py"
        files = {path: case.initial}
        bundle = assemble_delegation_bundle(
            f"Edit {path}", budget=2000,
            code_search_fn=lambda _query, limit: [DiscoveredHit(path, score=1.0)],
            file_reader_fn=files.get, bundle_id=f"cj7c-{case.problem_id}",
        )
        context = render_bundle(bundle, file_reader_fn=files.get)
        if not context or len(bundle.included()) != 1 or bundle.included()[0].path != path:
            raise ValueError("fixture bundle missing")
        manifest = bundle.manifest()
        identity = _digest({"problem_id": case.problem_id, "initial": case.initial,
                            "instructions": case.instructions, "manifest": manifest,
                            "context": context})
        prompts = {
            condition: build_root_lm_prompt(
                state="", original_prompt=f"{instructions[condition]}\n\nReturn the complete Python file.",
                corpus_context=context,
            ) for condition in CONDITIONS
        }
        # Refuse a builder configuration that drops, transforms, or duplicates the axis.
        left, right = (prompts[c] for c in CONDITIONS)
        descriptive, minimal = (instructions[c] for c in CONDITIONS)
        if (left.count(descriptive) != 1 or right.count(minimal) != 1
                or left.replace(descriptive, minimal, 1) != right):
            raise ValueError("prompt builder did not preserve the controlled axis")
        result.append({"problem_id": case.problem_id, "identity_sha256": identity,
                       "bundle_manifest": manifest, "bundle_context": context,
                       "conditions": [{"condition": c, "instruction": instructions[c],
                                       "prompt": prompts[c]} for c in CONDITIONS]})
    return result


def main() -> None:
    print(json.dumps({"scope": "two first-party CJ-7c fixtures; offline prompt assembly only",
                      "fixture_seal_sha256": FIXTURE_SEAL_SHA256,
                      "cases": assemble_pairs()}, indent=2))


if __name__ == "__main__":
    main()
