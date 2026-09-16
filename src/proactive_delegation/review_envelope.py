"""RA-12 — immutable machine-review envelope: build, sign, validate, bind (pure).

A machine review is valid ONLY for the exact bytes and versions it was produced
against. This module binds a model-emitted review body to those inputs and
refuses every attempt to present it as current once they change
(``orchestration/machine_review_envelope.schema.json``; intake-1004).

Three mechanisms, each closing a distinct hole:

1. **Binding.** ``ReviewBinding`` records source + candidate content hash and
   version, reviewer model + quant, prompt bundle hash, pipeline version and
   review-schema version. ``check_binding`` compares it field-by-field against
   the CURRENT inputs; any difference makes the review STALE, with the changed
   fields named.
2. **Signed body.** The review body is wrapped as ``{binding_digest, review}``
   and the envelope stores the hash of that signed body. An old body re-wrapped
   in an envelope that claims current inputs is refused because the body's own
   ``binding_digest`` names the inputs it was really produced against. This is
   integrity, not authentication: there is no key, so it stops splicing, not a
   forger who recomputes every hash on purpose.
3. **Objection typing.** Machine objections start as ``unverified_lead``. Only
   a CURRENT primary artifact (its hash must equal the current source or
   candidate hash) or an objective verifier can resolve one, and no machine
   objection is ever independent corroboration.

Staleness is DERIVED at read time and never written into an envelope
(immutable structure). ``current_view`` returns at most one current envelope
and keeps every other one as stale history, so stale objections can never be
merged into a current page.

NO inference and no I/O beyond reading the schema file.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ENVELOPE_SCHEMA_VERSION = "1.0.0"
SCHEMA_PATH = (
    Path(__file__).resolve().parents[2] / "orchestration" / "machine_review_envelope.schema.json"
)

UNVERIFIED_LEAD = "unverified_lead"
CONFIRMED = "confirmed"
REFUTED = "refuted"
PRIMARY_ARTIFACT = "primary_artifact"
OBJECTIVE_VERIFIER = "objective_verifier"


class ReviewEnvelopeError(ValueError):
    """An envelope or signed body is malformed or internally inconsistent."""


class StaleReviewError(ReviewEnvelopeError):
    """A review was presented as current, but its binding does not match current inputs."""

    def __init__(self, envelope_id: str, reasons: Sequence[str]):
        self.envelope_id = envelope_id
        self.reasons = tuple(reasons)
        super().__init__(f"review {envelope_id} is not current: {'; '.join(self.reasons)}")


# --------------------------------------------------------------------------- #
# Hashing
# --------------------------------------------------------------------------- #
def canonical_hash(obj: Any) -> str:
    """``sha256:<hex>`` over sorted-key compact JSON (same form as review_ledger)."""
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()


def content_hash(content: str | bytes) -> str:
    """``sha256:<hex>`` over raw content (text is UTF-8 encoded)."""
    data = content.encode("utf-8") if isinstance(content, str) else content
    return "sha256:" + hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------- #
# Binding
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ReviewBinding:
    """Every input a machine review is valid for. Any change = stale review."""

    source_hash: str
    source_version: str
    candidate_hash: str
    candidate_version: str
    reviewer_model: str
    reviewer_quant: str
    prompt_bundle_hash: str
    pipeline_version: str
    review_schema_version: str
    source_ref: str | None = None
    candidate_ref: str | None = None
    reviewer_model_hash: str | None = None

    def as_schema(self) -> dict[str, Any]:
        """Render the schema's nested ``binding`` object (None fields omitted)."""
        source: dict[str, Any] = {"content_hash": self.source_hash, "version": self.source_version}
        if self.source_ref is not None:
            source["ref"] = self.source_ref
        candidate: dict[str, Any] = {
            "content_hash": self.candidate_hash,
            "version": self.candidate_version,
        }
        if self.candidate_ref is not None:
            candidate["ref"] = self.candidate_ref
        reviewer: dict[str, Any] = {"model": self.reviewer_model, "quant": self.reviewer_quant}
        if self.reviewer_model_hash is not None:
            reviewer["model_hash"] = self.reviewer_model_hash
        return {
            "source": source,
            "candidate": candidate,
            "reviewer": reviewer,
            "prompt_bundle_hash": self.prompt_bundle_hash,
            "pipeline_version": self.pipeline_version,
            "review_schema_version": self.review_schema_version,
        }

    @classmethod
    def from_schema(cls, binding: Mapping[str, Any]) -> ReviewBinding:
        try:
            source = binding["source"]
            candidate = binding["candidate"]
            reviewer = binding["reviewer"]
            return cls(
                source_hash=source["content_hash"],
                source_version=source["version"],
                candidate_hash=candidate["content_hash"],
                candidate_version=candidate["version"],
                reviewer_model=reviewer["model"],
                reviewer_quant=reviewer["quant"],
                prompt_bundle_hash=binding["prompt_bundle_hash"],
                pipeline_version=binding["pipeline_version"],
                review_schema_version=binding["review_schema_version"],
                source_ref=source.get("ref"),
                candidate_ref=candidate.get("ref"),
                reviewer_model_hash=reviewer.get("model_hash"),
            )
        except (KeyError, TypeError) as exc:
            raise ReviewEnvelopeError(f"malformed binding: {exc!r}") from exc

    def digest(self) -> str:
        return canonical_hash(self.as_schema())


#: Fields whose change invalidates a review. Locators (``*_ref``) are NOT material:
#: moving a file without changing its bytes does not make the review stale.
MATERIAL_BINDING_FIELDS: tuple[str, ...] = tuple(
    name for name in ReviewBinding.__dataclass_fields__ if not name.endswith("_ref")
)


def changed_binding_fields(recorded: ReviewBinding, current: ReviewBinding) -> list[str]:
    """Material fields whose recorded value differs from the current one."""
    rec, cur = asdict(recorded), asdict(current)
    return [name for name in MATERIAL_BINDING_FIELDS if rec[name] != cur[name]]


# --------------------------------------------------------------------------- #
# Emission
# --------------------------------------------------------------------------- #
def sign_review(binding: ReviewBinding, review: Mapping[str, Any]) -> dict[str, Any]:
    """Wrap a review body with the digest of the inputs it was produced against."""
    return {"binding_digest": binding.digest(), "review": dict(review)}


def load_schema() -> dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def schema_errors(envelope: Mapping[str, Any]) -> list[str]:
    """jsonschema errors for an envelope (empty list == schema-valid)."""
    from jsonschema import Draft202012Validator

    validator = Draft202012Validator(load_schema())
    return [
        f"{'/'.join(str(p) for p in err.absolute_path) or '$'}: {err.message}"
        for err in sorted(validator.iter_errors(envelope), key=lambda e: list(e.absolute_path))
    ]


def new_objection(objection_id: str, text: str) -> dict[str, Any]:
    """A machine objection as emitted: always an unverified lead, never corroboration."""
    return {
        "objection_id": objection_id,
        "text": text,
        "status": UNVERIFIED_LEAD,
        "independent_corroboration": False,
    }


def build_envelope(
    *,
    envelope_id: str,
    binding: ReviewBinding,
    signed_body: Mapping[str, Any],
    validation_status: str,
    objections: Iterable[Mapping[str, Any]] = (),
    supersedes: str | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build a schema-valid envelope. Raises ``ReviewEnvelopeError`` otherwise.

    The signed body must have been signed for THIS binding: an envelope cannot be
    emitted around a body produced against other inputs.
    """
    if signed_body.get("binding_digest") != binding.digest():
        raise ReviewEnvelopeError(
            "signed body was produced against a different binding; re-run the review"
        )
    envelope = {
        "envelope_schema_version": ENVELOPE_SCHEMA_VERSION,
        "envelope_id": envelope_id,
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
        "binding": binding.as_schema(),
        "binding_digest": binding.digest(),
        "output_hash": canonical_hash(dict(signed_body)),
        "validation_status": validation_status,
        "supersedes": supersedes,
        "objections": [dict(o) for o in objections],
    }
    errors = schema_errors(envelope)
    if errors:
        raise ReviewEnvelopeError("envelope violates schema: " + "; ".join(errors))
    return envelope


# --------------------------------------------------------------------------- #
# Validation: is THIS review current for THESE inputs?
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BindingCheck:
    envelope_id: str
    current: bool
    reasons: tuple[str, ...]
    changed_fields: tuple[str, ...]


def check_binding(
    envelope: Mapping[str, Any],
    signed_body: Mapping[str, Any],
    current: ReviewBinding,
) -> BindingCheck:
    """Decide whether ``envelope`` + ``signed_body`` is a current review for ``current``.

    Every failed check is reported (not just the first) so the audit trail says
    WHY a review was refused. A review is current only if all of these hold:
    the envelope is schema-valid; its ``binding_digest`` matches its binding;
    the body hashes to ``output_hash``; the body was signed for the envelope's
    binding; the body validated at emission; and no material binding field
    differs from the current inputs.
    """
    envelope_id = str(envelope.get("envelope_id") or "<unknown>")
    reasons: list[str] = []
    changed: list[str] = []

    errors = schema_errors(envelope)
    if errors:
        return BindingCheck(envelope_id, False, tuple(f"schema: {e}" for e in errors), ())

    recorded = ReviewBinding.from_schema(envelope["binding"])
    if envelope["binding_digest"] != recorded.digest():
        reasons.append("binding_digest does not match the recorded binding (binding edited after signing)")
    if canonical_hash(dict(signed_body)) != envelope["output_hash"]:
        reasons.append("review body does not hash to output_hash (body swapped or edited)")
    if signed_body.get("binding_digest") != recorded.digest():
        reasons.append("review body was signed for different inputs than the envelope claims")
    if envelope["validation_status"] != "valid":
        reasons.append(f"review body failed validation at emission ({envelope['validation_status']})")

    changed = changed_binding_fields(recorded, current)
    if changed:
        reasons.append("inputs changed since review: " + ", ".join(changed))
    # A binding forged to claim current inputs around an old body is caught by the
    # "signed for different inputs" check above: the body's own digest names the
    # inputs it was really produced against, and editing the binding cannot change it.

    return BindingCheck(envelope_id, not reasons, tuple(reasons), tuple(changed))


def require_current(
    envelope: Mapping[str, Any],
    signed_body: Mapping[str, Any],
    current: ReviewBinding,
) -> BindingCheck:
    """``check_binding`` that raises ``StaleReviewError`` instead of returning a refusal."""
    result = check_binding(envelope, signed_body, current)
    if not result.current:
        raise StaleReviewError(result.envelope_id, result.reasons)
    return result


# --------------------------------------------------------------------------- #
# Supersession + current view
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CurrentView:
    """At most one current review; everything else is stale history, never merged."""

    current: Mapping[str, Any] | None
    stale_history: tuple[Mapping[str, Any], ...]
    refusals: tuple[BindingCheck, ...]

    @property
    def objections(self) -> tuple[Mapping[str, Any], ...]:
        """Objections shown on a current page: the current envelope's ONLY."""
        return tuple(self.current["objections"]) if self.current is not None else ()


def check_supersession_chain(envelopes: Sequence[Mapping[str, Any]]) -> list[str]:
    """Structural problems with ``supersedes`` links (dangling, forked, cyclic, duplicate)."""
    problems: list[str] = []
    ids = [e.get("envelope_id") for e in envelopes]
    known = set(ids)
    if len(known) != len(ids):
        problems.append("duplicate envelope_id")
    successors: dict[str, str] = {}
    for env in envelopes:
        parent = env.get("supersedes")
        if parent is None:
            continue
        if parent not in known:
            problems.append(f"{env['envelope_id']} supersedes unknown envelope {parent}")
        elif parent in successors:
            problems.append(
                f"{parent} is superseded twice ({successors[parent]}, {env['envelope_id']})"
            )
        else:
            successors[parent] = env["envelope_id"]
    parent_of = {e["envelope_id"]: e.get("supersedes") for e in envelopes}
    for start in parent_of:
        seen: set[str] = set()
        node: str | None = start
        while node is not None and node in parent_of:
            if node in seen:
                problems.append(f"supersession cycle through {start}")
                break
            seen.add(node)
            node = parent_of[node]
    return problems


def current_view(
    envelopes: Sequence[Mapping[str, Any]],
    bodies: Mapping[str, Mapping[str, Any]],
    current: ReviewBinding,
) -> CurrentView:
    """Select the single current review for ``current`` inputs.

    ``bodies`` maps ``envelope_id`` -> signed body. An envelope is a candidate
    only if it passes ``check_binding`` and has not been superseded. More than
    one surviving candidate is refused (ambiguous), never merged.
    """
    problems = check_supersession_chain(envelopes)
    if problems:
        raise ReviewEnvelopeError("invalid supersession chain: " + "; ".join(problems))
    superseded = {e["supersedes"] for e in envelopes if e.get("supersedes")}
    live: list[Mapping[str, Any]] = []
    refusals: list[BindingCheck] = []
    for env in envelopes:
        body = bodies.get(env["envelope_id"])
        if body is None:
            refusals.append(BindingCheck(env["envelope_id"], False, ("review body missing",), ()))
            continue
        result = check_binding(env, body, current)
        if not result.current:
            refusals.append(result)
        elif env["envelope_id"] in superseded:
            refusals.append(BindingCheck(env["envelope_id"], False, ("superseded",), ()))
        else:
            live.append(env)
    if len(live) > 1:
        raise ReviewEnvelopeError(
            "more than one current review for the same inputs: "
            + ", ".join(e["envelope_id"] for e in live)
        )
    chosen = live[0] if live else None
    history = tuple(e for e in envelopes if e is not chosen)
    return CurrentView(chosen, history, tuple(refusals))


# --------------------------------------------------------------------------- #
# Objections
# --------------------------------------------------------------------------- #
def resolve_objection(
    objection: Mapping[str, Any],
    *,
    status: str,
    kind: str,
    ref: str,
    resolver_hash: str,
    current: ReviewBinding,
) -> dict[str, Any]:
    """Return a NEW objection resolved by a primary artifact or an objective verifier.

    A primary artifact resolves only if it is CURRENT: its hash must equal the
    current source or candidate hash. An objective verifier result is accepted
    as-is (its own version pinning belongs to the verification report). The
    result is still ``independent_corroboration: false``; record it by emitting
    a new envelope that supersedes the old one.
    """
    if status not in (CONFIRMED, REFUTED):
        raise ReviewEnvelopeError(f"resolution status must be confirmed|refuted, got {status!r}")
    if kind not in (PRIMARY_ARTIFACT, OBJECTIVE_VERIFIER):
        raise ReviewEnvelopeError(f"resolver kind must be primary_artifact|objective_verifier, got {kind!r}")
    if kind == PRIMARY_ARTIFACT and resolver_hash not in (current.source_hash, current.candidate_hash):
        raise ReviewEnvelopeError(
            "primary artifact is not current (hash matches neither current source nor candidate)"
        )
    resolved = dict(objection)
    resolved.update(
        {
            "status": status,
            "independent_corroboration": False,
            "resolved_by": {"kind": kind, "ref": ref, "content_hash": resolver_hash},
        }
    )
    return resolved


def independent_corroboration_count(objections: Iterable[Mapping[str, Any]]) -> int:
    """How many objections count as independent corroboration: always 0 for machine objections.

    Raises if any objection claims otherwise, since that claim is itself a defect.
    """
    for obj in objections:
        if obj.get("independent_corroboration") is not False:
            raise ReviewEnvelopeError(
                f"objection {obj.get('objection_id')} claims independent corroboration"
            )
    return 0


def confirmed_objections(objections: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Objections a current primary artifact or objective verifier has CONFIRMED."""
    return [o for o in objections if o.get("status") == CONFIRMED and o.get("resolved_by")]
