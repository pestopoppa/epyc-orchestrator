"""RD-2 — two-turn rubric reviewer engine (author once, grade cheap).

The reviewer is a *two-turn* system (intake-834 economics: authoring $0.245 vs
grading $0.003; grading-model capability barely matters). Turn 1: a heavyweight
model **authors** a per-domain rubric ONCE and it is cached. Turn 2: a cheap
model **grades** each candidate against that rubric, per item, aggregating a
weighted score S. On a CPU stack the heavyweight decode is the cost, so re-review
is a cheap grade, never a heavyweight pass.

Model-agnostic by construction
-------------------------------
Neither turn imports an LLM client. Both take a *completion callable*
``Callable[[str], str]`` (prompt -> raw model text). Tests drive the engine with
stub callables → **zero inference**. Actually wiring the callables to llama.cpp
(and GBNF-constraining their output via ``review_grammar``) is the caller's job.

Artifacts (H2)
--------------
* Authored rubrics validate against ``orchestration/review_rubric.schema.json``.
* Per-item grades + the full rubric are surfaced on the result object so the H4
  calibration ledger can persist them (a rubric-*generator* fine-tune beats a
  binary-classifier fine-tune on identical data — do NOT throw the structure
  away). **This module does not write the ledger** — it only returns the data.

Decision bands (observation-grade defaults)
-------------------------------------------
S≥0.85 approve / S≤0.5 reject / middle → request_changes|request_evidence. These
seed from the paper's ROC (GT-pass cluster 0.85-1.0, GT-fail disperse ~0.4-0.5)
and are **observation-grade priors only**: they become decision-gating solely
after re-measurement on corpus v1 under protocol P-REV-1 (MEASUREMENT.md). They
are configurable via ``DecisionBands`` so re-measurement can retune without a
code edit.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from src.proactive_delegation.review_grammar import _extract_json_object

# orchestration/ lives at the repo root: .../src/proactive_delegation/<this> -> parents[2]
_ORCH_DIR = Path(__file__).resolve().parents[2] / "orchestration"
_REVIEW_RUBRIC_SCHEMA = _ORCH_DIR / "review_rubric.schema.json"
_REPO_ROOT = Path(__file__).resolve().parents[2]

# Default rubric cache location. JSON (not sqlite): the cache is a small,
# human-inspectable map and the two-turn design values transparency (you should
# be able to read the cached rubric a grader is applying). Gitignored via the
# ``data/review/`` entry in .gitignore (rebuildable — re-authoring re-populates).
DEFAULT_CACHE_PATH = _REPO_ROOT / "data" / "review" / "rubric_cache.json"

RUBRIC_SCHEMA_VERSION = "1.0.0"

CompletionFn = Callable[[str], str]


# ── module-level configuration (RD-11 tuning surface; re-measure before gating) ──


@dataclass(frozen=True)
class DecisionBands:
    """Configurable S→decision bands + the near-edge majority-of-k window.

    Defaults are observation-grade priors (intake-834); re-measure under P-REV-1
    before any of these gate a real keep/revert/deploy decision.
    """

    approve_at: float = 0.85  # S >= approve_at            -> approve
    reject_at: float = 0.50  # S <= reject_at             -> reject
    # else (reject_at < S < approve_at) -> request_changes | request_evidence
    edge_margin: float = 0.05  # |S - edge| <= edge_margin  -> majority-of-k re-grade
    binarize_at: float = 0.50  # per-item raw score >= this -> binary pass (1)
    # In the middle band, a failing *critical* (weight-3) item routes to
    # request_evidence (we want an objective check on a critical miss before we
    # hand back subjective feedback); otherwise request_changes.
    critical_weight: int = 3


DEFAULT_BANDS = DecisionBands()


# ── result objects ────────────────────────────────────────────────────


@dataclass
class ItemGrade:
    """One rubric item's grade. ``raw_score`` is the grader's [0,1]; ``binary``
    is the {0,1} used in S = Σ(w·s)/Σw."""

    item: str
    axis: str
    weight: int
    raw_score: float
    binary: int
    graded: bool = True  # False = grader omitted this item (scored 0, conservative)
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "item": self.item,
            "axis": self.axis,
            "weight": self.weight,
            "raw_score": self.raw_score,
            "binary": self.binary,
            "graded": self.graded,
            "note": self.note,
        }


@dataclass
class GradingPass:
    """A single grading pass (majority-of-k audit trail)."""

    S: float
    decision: str
    per_item: list[ItemGrade]
    parse_ok: bool = True
    parse_detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "S": self.S,
            "decision": self.decision,
            "per_item": [g.to_dict() for g in self.per_item],
            "parse_ok": self.parse_ok,
            "parse_detail": self.parse_detail,
        }


@dataclass
class GradeResult:
    """Aggregate grading verdict + everything H4 needs to persist.

    Carries the full rubric and per-item grades (do not discard — H4 corpus rows
    need the transferable structure), plus every pass for the majority-of-k audit
    trail and a per-model flakiness estimate.
    """

    rubric_id: str
    rubric_version: str
    rubric_ref: str
    S: float
    decision: str
    confidence: float
    per_item: list[ItemGrade]
    passes: list[GradingPass]
    k_used: int
    near_edge: bool
    flakiness: float
    rubric: dict[str, Any]
    bands: DecisionBands

    def to_dict(self) -> dict[str, Any]:
        return {
            "rubric_id": self.rubric_id,
            "rubric_version": self.rubric_version,
            "rubric_ref": self.rubric_ref,
            "S": self.S,
            "decision": self.decision,
            "confidence": self.confidence,
            "per_item": [g.to_dict() for g in self.per_item],
            "passes": [p.to_dict() for p in self.passes],
            "k_used": self.k_used,
            "near_edge": self.near_edge,
            "flakiness": self.flakiness,
            "rubric": self.rubric,
            "bands": {
                "approve_at": self.bands.approve_at,
                "reject_at": self.bands.reject_at,
                "edge_margin": self.bands.edge_margin,
                "binarize_at": self.bands.binarize_at,
            },
        }


class RubricAuthoringError(ValueError):
    """The author model's emission could not be turned into a schema-valid rubric."""


class RubricGradingError(ValueError):
    """A grading pass could not be parsed (all k passes failed to parse)."""


# ── rubric cache (keyed (task_class, domain, version)) ─────────────────


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _cache_key(task_class: str, domain: str) -> str:
    return f"{task_class}\x1f{domain}"


def _bump_version(version: str) -> str:
    """Bump the minor component (template-drift refresh = new cacheable artifact)."""
    try:
        major, minor, patch = (int(p) for p in version.split("."))
    except (ValueError, AttributeError):
        return "1.0.0"
    return f"{major}.{minor + 1}.0"


class RubricCache:
    """JSON-file cache of authored rubrics keyed by (task_class, domain, version).

    Layout::

        {
          "<task_class>\\x1f<domain>": {
            "latest": "1.1.0",
            "versions": {"1.0.0": <rubric>, "1.1.0": <rubric>}
          }
        }

    ``get`` returns the latest version (or a specific one). A cache hit never
    calls the author model. Invalidation is explicit: ``author_rubric(refresh=
    True)`` bumps the version and re-authors (the template-drift hook).
    """

    def __init__(self, path: Path | str = DEFAULT_CACHE_PATH):
        self.path = Path(path)
        self._data: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    def _flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2, sort_keys=True), encoding="utf-8")

    def get(
        self, task_class: str, domain: str, version: str | None = None
    ) -> dict[str, Any] | None:
        entry = self._data.get(_cache_key(task_class, domain))
        if not entry:
            return None
        versions = entry.get("versions", {})
        if version is not None:
            return versions.get(version)
        latest = entry.get("latest")
        return versions.get(latest) if latest else None

    def latest_version(self, task_class: str, domain: str) -> str | None:
        entry = self._data.get(_cache_key(task_class, domain))
        return entry.get("latest") if entry else None

    def put(self, task_class: str, domain: str, rubric: dict[str, Any]) -> None:
        key = _cache_key(task_class, domain)
        entry = self._data.setdefault(key, {"latest": None, "versions": {}})
        entry["versions"][rubric["version"]] = rubric
        entry["latest"] = rubric["version"]
        self._flush()


# ── schema access ──────────────────────────────────────────────────────


def load_review_rubric_schema() -> dict[str, Any]:
    return json.loads(_REVIEW_RUBRIC_SCHEMA.read_text(encoding="utf-8"))


def _validate_rubric(rubric: dict[str, Any]) -> None:
    from jsonschema import Draft202012Validator

    validator = Draft202012Validator(load_review_rubric_schema())
    errors = sorted(validator.iter_errors(rubric), key=lambda e: list(e.absolute_path))
    if errors:
        msgs = [
            f"{'$' + ''.join(f'.{p}' for p in e.absolute_path)}: {e.message}" for e in errors[:10]
        ]
        raise RubricAuthoringError("rubric failed schema validation: " + "; ".join(msgs))


# ── prompt builders (content is not executed in tests — passed to the callable) ──


def _author_prompt(task_class: str, domain: str, context: str) -> str:
    return (
        "You are authoring a REUSABLE grading rubric that a cheaper model will "
        "apply to many candidates. Emit a JSON object with an `items` array; each "
        'item is {"text": <criterion phrased as a checkable question>, "axis": '
        '<grading axis>, "weight": 1|2|3}. Do not grade any candidate — write the '
        "rubric only.\n"
        f"task_class: {task_class}\n"
        f"domain: {domain}\n"
        f"context:\n{context}\n"
    )


def _grading_prompt(rubric: dict[str, Any], sanitized_candidate: str) -> str:
    lines = [
        "Grade the candidate against EACH rubric item independently. For each "
        'item emit {"item": <id>, "score": 0 or 1, "note": <optional>}; 1 = the '
        "criterion is met, 0 = not met. Then emit an overall `decision`. Judge "
        "pointwise — this single candidate only.",
        "RUBRIC:",
    ]
    for it in rubric.get("items", []):
        lines.append(f"  {it['id']} [{it['axis']} w{it['weight']}]: {it['text']}")
    lines.append("CANDIDATE:")
    lines.append(sanitized_candidate)
    return "\n".join(lines)


# ── turn 1: author_rubric ──────────────────────────────────────────────


def author_rubric(
    task_class: str,
    domain: str,
    context: str,
    author_complete: CompletionFn,
    *,
    refresh: bool = False,
    cache: RubricCache | None = None,
) -> dict[str, Any]:
    """Author (or fetch cached) a per-(task_class, domain) rubric.

    On a cache HIT with ``refresh=False`` the author model is NOT called — the
    cached rubric is returned verbatim. On a MISS, or when ``refresh=True`` (the
    template-drift invalidation hook), ``author_complete(prompt)`` is invoked, its
    text parsed into rubric items, wrapped with a stable ``rubric_id`` and a
    semver ``version`` (bumped on refresh), validated against
    review_rubric.schema.json, cached, and returned.
    """
    cache = cache if cache is not None else RubricCache()

    if not refresh:
        cached = cache.get(task_class, domain)
        if cached is not None:
            return cached

    raw = author_complete(_author_prompt(task_class, domain, context))
    obj_text = _extract_json_object(raw or "")
    if obj_text is None:
        raise RubricAuthoringError("author model produced no JSON object")
    try:
        parsed = json.loads(obj_text)
    except json.JSONDecodeError as exc:
        raise RubricAuthoringError(f"author JSON decode error: {exc}") from exc
    if not isinstance(parsed, dict) or "items" not in parsed:
        raise RubricAuthoringError("author output missing `items`")

    items = _normalize_items(parsed["items"])
    if not items:
        raise RubricAuthoringError("author produced zero valid items")

    prev = cache.latest_version(task_class, domain)
    version = _bump_version(prev) if (refresh and prev) else (prev or "1.0.0")
    if refresh and prev is None:
        version = "1.0.0"
    rubric_id = _slug(f"{domain}:{task_class}")

    provenance = parsed.get("provenance") if isinstance(parsed.get("provenance"), dict) else {}
    rubric: dict[str, Any] = {
        "schema_version": RUBRIC_SCHEMA_VERSION,
        "rubric_id": rubric_id,
        "version": version,
        "created_at": _now_utc(),
        "domain": domain,
        "title": str(parsed.get("title") or f"{task_class} / {domain} rubric"),
        "items": items,
        "provenance": {"role": "reviewer", **provenance, "task_class": task_class},
    }
    if isinstance(parsed.get("grading_scale"), dict):
        rubric["grading_scale"] = parsed["grading_scale"]

    _validate_rubric(rubric)
    cache.put(task_class, domain, rubric)
    return rubric


def _normalize_items(raw_items: Any) -> list[dict[str, Any]]:
    """Coerce model item output into schema-valid items (ids R1.., weight∈{1,2,3})."""
    out: list[dict[str, Any]] = []
    if not isinstance(raw_items, list):
        return out
    for idx, it in enumerate(raw_items, start=1):
        if not isinstance(it, dict):
            continue
        text = str(it.get("text", "")).strip()
        if not text:
            continue
        axis = str(it.get("axis", "general")).strip() or "general"
        out.append(
            {
                "id": f"R{idx}",  # renumber to guarantee ^R[0-9]+$ regardless of model
                "text": text,
                "axis": axis,
                "weight": _coerce_weight(it.get("weight", 2)),
            }
        )
    # ids must be unique + sequential after filtering-out invalid entries
    for new_idx, item in enumerate(out, start=1):
        item["id"] = f"R{new_idx}"
    return out


def _coerce_weight(weight: Any) -> int:
    try:
        w = int(round(float(weight)))
    except (ValueError, TypeError):
        return 2
    return min(3, max(1, w))


def _slug(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    return s or "rubric"


# ── turn 2: grade_candidate ────────────────────────────────────────────


def grade_candidate(
    rubric: dict[str, Any],
    sanitized_candidate: str,
    grader_complete: CompletionFn,
    k: int = 1,
    *,
    bands: DecisionBands | None = None,
) -> GradeResult:
    """Grade one candidate against ``rubric`` and map S → a decision band.

    Runs one grading pass. If ``k > 1`` **and** the resulting S lands within
    ``bands.edge_margin`` of a band edge (judge flakiness 2-9% is exactly where a
    single pass is unreliable), runs up to ``k`` total passes and decides by
    majority vote over per-pass decisions (ties broken by the median-S band).
    Cheap path stays one call unless we are genuinely on an edge.
    """
    bands = bands or DEFAULT_BANDS
    weights = {it["id"]: (it["axis"], it["weight"]) for it in rubric.get("items", [])}
    prompt = _grading_prompt(rubric, sanitized_candidate)

    passes: list[GradingPass] = []
    first = _one_pass(prompt, grader_complete, rubric, weights, bands)
    passes.append(first)

    near_edge = _near_edge(first.S, bands)
    if k > 1 and near_edge:
        for _ in range(k - 1):
            passes.append(_one_pass(prompt, grader_complete, rubric, weights, bands))

    ok_passes = [p for p in passes if p.parse_ok]
    if not ok_passes:
        raise RubricGradingError(
            f"all {len(passes)} grading pass(es) failed to parse: {passes[0].parse_detail}"
        )

    decision, chosen = _aggregate(ok_passes, bands)
    flakiness = _flakiness(ok_passes, decision)
    version = str(rubric.get("version", "0.0.0"))
    rid = str(rubric.get("rubric_id", "unknown"))
    # confidence: high when far from edges + passes agree; lower near edges / on disagreement
    confidence = round(max(0.0, min(1.0, (1.0 - flakiness) * _edge_distance(chosen.S, bands))), 4)

    return GradeResult(
        rubric_id=rid,
        rubric_version=version,
        rubric_ref=f"{rid}@{version}",
        S=chosen.S,
        decision=decision,
        confidence=confidence,
        per_item=chosen.per_item,
        passes=passes,
        k_used=len(passes),
        near_edge=near_edge,
        flakiness=flakiness,
        rubric=rubric,
        bands=bands,
    )


def _one_pass(
    prompt: str,
    grader_complete: CompletionFn,
    rubric: dict[str, Any],
    weights: dict[str, tuple[str, int]],
    bands: DecisionBands,
) -> GradingPass:
    raw = grader_complete(prompt)
    obj_text = _extract_json_object(raw or "")
    if obj_text is None:
        return GradingPass(0.0, "request_changes", [], parse_ok=False, parse_detail="no JSON object")
    try:
        parsed = json.loads(obj_text)
    except json.JSONDecodeError as exc:
        return GradingPass(0.0, "request_changes", [], parse_ok=False, parse_detail=str(exc))
    if not isinstance(parsed, dict) or not isinstance(parsed.get("grades"), list):
        return GradingPass(0.0, "request_changes", [], parse_ok=False, parse_detail="no grades[]")

    graded_scores: dict[str, tuple[float, str]] = {}
    for g in parsed["grades"]:
        if not isinstance(g, dict):
            continue
        item_id = str(g.get("item", ""))
        if item_id not in weights:
            continue  # ignore grades for items not in the rubric
        try:
            raw_score = float(g.get("score", 0.0))
        except (ValueError, TypeError):
            raw_score = 0.0
        graded_scores[item_id] = (max(0.0, min(1.0, raw_score)), str(g.get("note", "")))

    per_item: list[ItemGrade] = []
    for item_id, (axis, weight) in weights.items():
        if item_id in graded_scores:
            raw_score, note = graded_scores[item_id]
            per_item.append(
                ItemGrade(
                    item=item_id,
                    axis=axis,
                    weight=weight,
                    raw_score=raw_score,
                    binary=1 if raw_score >= bands.binarize_at else 0,
                    graded=True,
                    note=note,
                )
            )
        else:
            # missing grade = conservative fail (binary 0), recorded as ungraded
            per_item.append(
                ItemGrade(item=item_id, axis=axis, weight=weight, raw_score=0.0, binary=0, graded=False)
            )

    S = _weighted_score(per_item)
    decision = _band_decision(S, per_item, bands)
    return GradingPass(S=S, decision=decision, per_item=per_item)


def _weighted_score(per_item: list[ItemGrade]) -> float:
    total_w = sum(g.weight for g in per_item)
    if total_w == 0:
        return 0.0
    return round(sum(g.weight * g.binary for g in per_item) / total_w, 6)


def _band_decision(S: float, per_item: list[ItemGrade], bands: DecisionBands) -> str:
    if S >= bands.approve_at:
        return "approve"
    if S <= bands.reject_at:
        return "reject"
    # middle band: a failing CRITICAL item wants objective evidence before we
    # route subjective feedback; otherwise ask the author for changes.
    critical_fail = any(
        g.binary == 0 and g.weight >= bands.critical_weight for g in per_item
    )
    return "request_evidence" if critical_fail else "request_changes"


def _near_edge(S: float, bands: DecisionBands) -> bool:
    return (
        abs(S - bands.approve_at) <= bands.edge_margin
        or abs(S - bands.reject_at) <= bands.edge_margin
    )


def _edge_distance(S: float, bands: DecisionBands) -> float:
    """Distance from the nearer band edge, normalized to [0,1] (1 = far from edge)."""
    d = min(abs(S - bands.approve_at), abs(S - bands.reject_at))
    span = max(bands.approve_at - bands.reject_at, 1e-9)
    return max(0.0, min(1.0, d / span))


def _aggregate(passes: list[GradingPass], bands: DecisionBands) -> tuple[str, GradingPass]:
    """Majority decision across passes; ties broken by the median-S pass's band."""
    if len(passes) == 1:
        return passes[0].decision, passes[0]

    counts: dict[str, int] = {}
    for p in passes:
        counts[p.decision] = counts.get(p.decision, 0) + 1
    top = max(counts.values())
    winners = {d for d, c in counts.items() if c == top}

    ordered = sorted(passes, key=lambda p: p.S)
    median_pass = ordered[len(ordered) // 2]
    if len(winners) == 1:
        decision = next(iter(winners))
    else:
        decision = _band_decision(median_pass.S, median_pass.per_item, bands)
    # representative pass = one whose decision matches the aggregate (median-S among them)
    matching = [p for p in ordered if p.decision == decision]
    chosen = matching[len(matching) // 2] if matching else median_pass
    return decision, chosen


def _flakiness(passes: list[GradingPass], decision: str) -> float:
    """Fraction of passes whose decision disagrees with the aggregate (per-model
    judge-flakiness signal; the design says log this)."""
    if len(passes) <= 1:
        return 0.0
    disagree = sum(1 for p in passes if p.decision != decision)
    return round(disagree / len(passes), 4)
