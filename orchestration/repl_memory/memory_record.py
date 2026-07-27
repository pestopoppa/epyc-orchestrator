"""memory_record.py — the single shape every episodic memory is written in.

WHY THIS EXISTS
---------------
A 2026-07-27 audit of the episodic store found that its six write sites had each
grown their own record shape and their own embedding text, with four measurable
consequences:

1. **Half the store was not memory.** 27,123 of 54,960 "routing" rows carried
   ``{question_id, elapsed_seconds, tokens_generated, predicted_tps, ...}`` and
   NO task text — performance telemetry embedded as text into a semantic index.
2. **Objectives were truncated to 200 chars at write time**, so the text was
   *gone*, not merely unindexed. 12.6% of rows sat exactly at the cap, and only
   2,639 distinct objectives existed across 54,960 rows (96.9% collision).
3. **At least four embedding conventions** coexisted, so the vector encoded
   *which writer produced the row* — a probe read the writer path off the
   embedding at ROW-AUC 0.906-0.940, and the two paths differed up to 8x in
   failure rate.
4. **Nothing stored the work.** No answer, no tool calls, no REPL steps, no
   reasoning. So retrieval could only ever return "here is a similar task stub,
   and it succeeded" — never "here is how it was solved". That ceiling is
   visible in SkillBank's output: all 57 distilled skills are thin routing
   heuristics, because a routing heuristic is the only thing extractable from
   ``(stub, role, success-bit)``.

Operator decision, 2026-07-27: store the works — answers, tool calls, REPL
steps, reasoning traces. Storage is not the constraint (answers measure ~1.1 KB
mean; p50 is 58 tokens), and distillation is measured at 10:1.

THE ONE INVARIANT
-----------------
**What gets EMBEDDED is not what gets STORED.**

    embedding_text()  -> the task only, one convention, bounded length.
                         This is what similarity search matches on, so it must
                         describe the QUESTION. Embedding the answer would make
                         retrieval match solutions against solutions.

    to_context()      -> the full record: untruncated objective plus the work.
                         Never fed to the embedder.

Telemetry rides in ``metrics`` and is excluded from the embedding text by
construction, which is what keeps number-blobs out of the semantic index.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

#: Cap on the text handed to the embedder. BGE-large truncates around 512
#: tokens anyway, so a generous character bound loses nothing while keeping the
#: convention stable. THIS DOES NOT TRUNCATE STORED TEXT — see to_context().
EMBED_TEXT_MAX_CHARS = 2000

#: Schema marker on every stored context, so a future reader can tell records
#: written under this contract from the pre-2026-07-27 free-form ones.
RECORD_VERSION = 1


@dataclass
class MemoryRecord:
    """One episodic memory, in the shape every write site must produce."""

    # ---- the task (drives the embedding) ----
    objective: str
    task_type: str | None = None
    priority: str | None = None

    # ---- the work (stored, never embedded) ----
    answer: str | None = None
    tool_calls: list[Any] = field(default_factory=list)
    repl_steps: list[Any] = field(default_factory=list)
    reasoning: str | None = None

    # ---- provenance and telemetry (stored, never embedded) ----
    source: str = "unknown"
    metrics: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def embedding_text(self) -> str:
        """The ONE embedding convention, for every write site.

        Matches the historical ``type: | objective: | priority:`` shape so
        vectors written before and after this change remain comparable, but is
        now produced in exactly one place instead of four.
        """
        parts: list[str] = []
        if self.task_type:
            parts.append(f"type:{self.task_type}")
        parts.append(f"objective:{(self.objective or '').strip()}")
        if self.priority:
            parts.append(f"priority:{self.priority}")
        text = " | ".join(parts)
        return text[:EMBED_TEXT_MAX_CHARS]

    def to_context(self) -> dict[str, Any]:
        """The stored payload. Full fidelity — nothing here is truncated.

        ``objective`` is the complete text: the old 200-char cap destroyed it at
        write time and is not reproduced.
        """
        ctx: dict[str, Any] = {
            "record_version": RECORD_VERSION,
            "task_type": self.task_type,
            "objective": self.objective,
            "priority": self.priority,
            "source": self.source,
        }
        work: dict[str, Any] = {}
        if self.answer is not None:
            work["answer"] = self.answer
        if self.tool_calls:
            work["tool_calls"] = self.tool_calls
        if self.repl_steps:
            work["repl_steps"] = self.repl_steps
        if self.reasoning is not None:
            work["reasoning"] = self.reasoning
        if work:
            ctx["work"] = work
        if self.metrics:
            ctx["metrics"] = self.metrics
        if self.extra:
            ctx["extra"] = self.extra
        return ctx

    def is_task_memory(self) -> bool:
        """True when this row belongs in the semantic index at all.

        A record with no objective text is telemetry. Embedding it produces a
        vector of a number-blob, which is how half the store came to be
        unsearchable noise.
        """
        return bool((self.objective or "").strip())


def build_memory_record(
    *,
    objective: str | None,
    task_type: str | None = None,
    priority: str | None = None,
    answer: str | None = None,
    tool_calls: list[Any] | None = None,
    repl_steps: list[Any] | None = None,
    reasoning: str | None = None,
    source: str = "unknown",
    metrics: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> MemoryRecord:
    """Construct a MemoryRecord. The only supported way to build one."""
    return MemoryRecord(
        objective=objective or "",
        task_type=task_type,
        priority=priority,
        answer=answer,
        tool_calls=list(tool_calls or []),
        repl_steps=list(repl_steps or []),
        reasoning=reasoning,
        source=source,
        metrics=dict(metrics or {}),
        extra=dict(extra or {}),
    )


def record_from_legacy_context(context: dict[str, Any]) -> MemoryRecord:
    """Adapt a pre-2026-07-27 free-form context dict to a MemoryRecord.

    Both historical writer shapes are recognised:

      path A (progress log): ``{task_type, objective, priority}``
      path B (external):     ``{task_description, source, question_id,
                                elapsed_seconds, tokens_generated, ...}``

    Path B's non-task keys are routed into ``metrics`` rather than being
    embedded, which is the whole point — it is what stops telemetry entering the
    semantic index.
    """
    ctx = dict(context or {})
    objective = ctx.pop("objective", None) or ctx.pop("task_description", None)
    task_type = ctx.pop("task_type", None)
    priority = ctx.pop("priority", None)
    source = ctx.pop("source", "legacy")
    work = ctx.pop("work", {}) or {}
    return MemoryRecord(
        objective=objective or "",
        task_type=task_type,
        priority=priority,
        answer=work.get("answer"),
        tool_calls=list(work.get("tool_calls") or []),
        repl_steps=list(work.get("repl_steps") or []),
        reasoning=work.get("reasoning"),
        source=source,
        # everything left over is telemetry/bookkeeping — stored, not embedded
        metrics=ctx,
    )


def context_size_bytes(context: dict[str, Any]) -> int:
    """Serialized size of a stored context, for budget accounting."""
    return len(json.dumps(context, default=str).encode("utf-8"))
