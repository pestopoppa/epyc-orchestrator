"""Assembly-time CandidatePackage sanitizer (control-plane spec §13.2 / §20.5).

PRIMARY sanitization for the reviewer plane. The reviewer-visible ``sanitized_view``
is BUILT here — this is the *assembly-time* half of the sanitization contract
declared by ``orchestration/candidate_package.schema.json``. ``review_service`` only
performs defense-in-depth at consumption time; it MUST NOT be the place a package is
first made safe (review_service.py:728: "sanitization is an assembly-time contract").

This module hardens the CP3 assembly-layer security gaps:

  * §13.2 control 6 — **package path allowlist + secret redaction.** Outputs whose
    ``ref`` is a file locator OUTSIDE the package allowlist are dropped from the
    reviewer view (a file outside the package root may hold unrelated secrets and is
    never projected); secret-like tokens embedded in retained inline content are
    masked. Every removal/mask is recorded in ``sanitization.redactions`` — redaction
    is FLAGGED, never silent.
  * §13.2 controls 5,7 — **no silent truncation.** Outputs are bounded by a retention
    cap that prioritizes material work-products and risk-flagged content over prose
    filler, so a buried critical output survives the downstream render cap instead of
    being silently dropped. Anything the cap drops is recorded in
    ``sanitization.truncation_manifest``.
  * §13.1 control/data separation (assembly half) — the sanitized_view carries an
    explicit ``untrusted_content_policy`` declaring candidate content is DATA. NOTE:
    the render-side quarantine (wrapping candidate text in an explicit prompt
    delimiter so embedded instructions are inert) is ``review_service``'s renderer's
    job and is tracked separately as a FROZEN render-layer item — this module cannot
    strip in-content injection without deleting the candidate the reviewer must read.

Pure-stdlib, no inference, no I/O: given a full package dict, returns a new dict with
a hardened ``sanitized_view``. The full package (author framing fields included) is
preserved verbatim for audit — only the reviewer projection is sanitized.
"""

from __future__ import annotations

import hashlib
import os
import re
from typing import Any

# Framing fields the schema forbids inside ``sanitized_view`` (intake-837/838):
# author self-assessment / confidence / quality labels shift verdicts ~18-29pp.
BANNED_FRAMING_FIELDS = (
    "author_self_assessment",
    "author_confidence_assertion",
    "quality_labels",
)

#: Repo-relative prefixes a candidate output ``ref`` file locator may point at.
#: Anything else (absolute paths, ``..`` escapes, home paths) is out-of-package.
PACKAGE_PATH_ALLOWLIST: tuple[str, ...] = ("src/", "tests/", "orchestration/")

#: Reviewer-visible output retention cap. Matches the downstream render cap so the
#: assembly builder — NOT the renderer — owns the truncation decision and records it.
MAX_REVIEWER_OUTPUTS = 8

#: Sanitization policy version stamped into the audit trail.
SANITIZER_POLICY_VERSION = "cp3-assembly-sanitizer"

# High-signal markers that flag an output as material for the reviewer (a defect /
# security concern that MUST NOT be buried past the retention cap). Case-insensitive.
_RISK_MARKERS = (
    "critical",
    "security",
    "regression",
    "vulnerab",
    "vuln",
    "secret",
    "fail",
    "error",
    "auth",
    "injection",
    "bypass",
    "traversal",
    "rce",
)

# Material output types rank above prose filler when the retention cap forces a choice.
_MATERIAL_TYPES = ("diff", "file", "artifact", "answer")

# Secret-like token patterns. Matches are masked in-place in retained inline content;
# only the redaction CATEGORY is recorded (never the secret value) per review policy.
_SECRET_PATTERNS: tuple[tuple[str, "re.Pattern[str]"], ...] = (
    ("private_key", re.compile(r"-----BEGIN[A-Z0-9 ]*PRIVATE KEY-----")),
    ("openai_key", re.compile(r"\bsk-[A-Za-z0-9_-]{8,}")),
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("bearer_token", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{16,}")),
    (
        "generic_secret_assignment",
        re.compile(
            r"(?i)\b(?:api[_-]?key|secret|token|password|passwd|access[_-]?key)\b"
            r"\s*[=:]\s*['\"]?[A-Za-z0-9/+_.\-]{12,}"
        ),
    ),
)

# A ref is treated as a file locator (path-allowlist gated) only if it is a single
# whitespace-free token that looks like a path — never multi-word inline content.
_PATH_SHAPED = re.compile(r"^[~./]*[\w][\w./+\-]*$")


def _looks_like_path(output_type: str, ref: str) -> bool:
    """True if this output ``ref`` is a file locator subject to the path allowlist.

    ``type == "file"`` is always a locator. Otherwise a ref is a locator only if it
    is a single path-shaped token containing a separator — inline code/answers (which
    carry spaces/newlines) are candidate DATA and are never path-gated.
    """
    s = ref.strip()
    if output_type == "file":
        return True
    if not s or "\n" in s or " " in s:
        return False
    if not _PATH_SHAPED.match(s):
        return False
    return "/" in s or s.startswith(("/", "~", "."))


def _ref_within_allowlist(ref: str, allowlist: tuple[str, ...]) -> bool:
    """True if a path-locator ref stays inside the repo-relative package allowlist.

    Canonicalizes with ``normpath`` first so ``src/../etc/passwd`` and symlink-style
    traversal cannot smuggle an out-of-package path past a naive prefix check.
    Absolute and home paths are out-of-package by construction.
    """
    s = ref.strip()
    if not s:
        return True  # empty locator is not a path to gate (schema minLength catches it)
    if s.startswith(("/", "~")):
        return False
    norm = os.path.normpath(s)
    if norm == ".." or norm.startswith(".." + os.sep) or norm.startswith("../") or os.path.isabs(norm):
        return False
    return any(norm == a.rstrip("/") or norm.startswith(a.rstrip("/") + "/") for a in allowlist)


def _redact_secrets(text: str) -> tuple[str, list[str]]:
    """Mask secret-like tokens in ``text``. Returns (masked_text, categories_hit)."""
    hits: list[str] = []
    masked = text
    for category, pattern in _SECRET_PATTERNS:
        if pattern.search(masked):
            if category not in hits:
                hits.append(category)
            masked = pattern.sub(f"[REDACTED:{category}]", masked)
    return masked, hits


def _digest(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:16]


def _output_priority(output: dict[str, Any]) -> tuple[int, int]:
    """Materiality rank for retention: (risk_flagged, material_type). Higher = keep."""
    blob = f"{output.get('ref', '')}\n{output.get('label', '')}".lower()
    risk = 1 if any(m in blob for m in _RISK_MARKERS) else 0
    material = 1 if str(output.get("type", "")) in _MATERIAL_TYPES else 0
    return (risk, material)


def _sanitize_outputs(
    outputs: list[dict[str, Any]],
    allowlist: tuple[str, ...],
    cap: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any] | None]:
    """Apply path-allowlist redaction, secret masking, then materiality-priority cap.

    Returns (sanitized_outputs, redactions, truncation_manifest_or_None).
    """
    redactions: list[dict[str, Any]] = []
    kept: list[dict[str, Any]] = []

    for idx, out in enumerate(outputs):
        if not isinstance(out, dict):
            continue
        otype = str(out.get("type", "artifact"))
        ref = str(out.get("ref", ""))

        # (1) Package path allowlist — drop out-of-package file locators entirely.
        if _looks_like_path(otype, ref) and not _ref_within_allowlist(ref, allowlist):
            redactions.append(
                {
                    "index": idx,
                    "category": "path_allowlist",
                    "reason": "output ref points outside the package path allowlist",
                    "original_type": otype,
                }
            )
            continue

        # (2) Secret masking in retained inline content (defense-in-depth).
        new_out = dict(out)
        masked_ref, ref_hits = _redact_secrets(ref)
        label_hits: list[str] = []
        if "label" in new_out:
            masked_label, label_hits = _redact_secrets(str(new_out["label"]))
            if label_hits:
                new_out["label"] = masked_label
        if ref_hits:
            new_out["ref"] = masked_ref
        hits = ref_hits + [h for h in label_hits if h not in ref_hits]
        if hits:
            redactions.append(
                {
                    "index": idx,
                    "category": "secret",
                    "reason": "secret-like token masked in output content",
                    "patterns": hits,
                }
            )
        kept.append({"_orig_index": idx, "_out": new_out})

    # (3) Retention cap — prioritize material/risk-flagged outputs, never silently drop.
    manifest: dict[str, Any] | None = None
    if len(kept) > cap:
        ordered = sorted(
            kept,
            key=lambda k: (_output_priority(k["_out"]), -k["_orig_index"]),
            reverse=True,
        )
        selected = ordered[:cap]
        dropped = ordered[cap:]
        selected_orig = {k["_orig_index"] for k in selected}
        # Emit the retained set back in original order for a natural reviewer reading.
        kept = [k for k in kept if k["_orig_index"] in selected_orig]
        manifest = {
            "output_cap": cap,
            "total_outputs": len(outputs),
            "retained": len(selected),
            "dropped": [
                {
                    "index": k["_orig_index"],
                    "type": str(k["_out"].get("type", "artifact")),
                    "reason": "exceeds_reviewer_output_cap",
                    "digest": _digest(str(k["_out"].get("ref", ""))),
                    "ref_preview": str(k["_out"].get("ref", ""))[:80],
                }
                for k in sorted(dropped, key=lambda k: k["_orig_index"])
            ],
        }

    sanitized_outputs = [k["_out"] for k in kept]
    return sanitized_outputs, redactions, manifest


def sanitize_candidate_package(
    full_pkg: dict[str, Any],
    *,
    allowlist: tuple[str, ...] = PACKAGE_PATH_ALLOWLIST,
    output_cap: int = MAX_REVIEWER_OUTPUTS,
) -> dict[str, Any]:
    """Build the reviewer-visible ``sanitized_view`` from a full CandidatePackage.

    Contract (assembly-time):
      * author framing FIELDS (self-assessment / confidence / quality labels) are
        stripped from the reviewer projection (recorded in ``removed_fields``);
      * output file locators outside ``allowlist`` are dropped, secret-like tokens in
        retained content are masked (recorded in ``redactions``);
      * outputs are bounded by ``output_cap`` with material/risk-flagged content
        prioritized so nothing critical is silently truncated (recorded in
        ``truncation_manifest``);
      * ``untrusted_content_policy`` declares candidate content is DATA (the assembly
        half of control/data separation; render-side quarantine is review_service's).

    The full package is preserved verbatim (author fields retained for audit); only
    the ``sanitized_view`` projection is sanitized. Pure function, no side effects.
    """
    removed = [f for f in BANNED_FRAMING_FIELDS if f in full_pkg]
    raw_outputs = list(full_pkg.get("outputs", []) or [])
    outputs, redactions, manifest = _sanitize_outputs(raw_outputs, allowlist, output_cap)

    sanitization: dict[str, Any] = {
        "applied": True,
        "removed_fields": removed,
        "policy_version": SANITIZER_POLICY_VERSION,
        "path_allowlist": list(allowlist),
    }
    if redactions:
        sanitization["redactions"] = redactions
    if manifest is not None:
        sanitization["truncation_manifest"] = manifest

    sanitized_view: dict[str, Any] = {
        "task_ref": full_pkg["task_ref"],
        "outputs": outputs,
        # Assembly-time control/data separation contract declaration. The render-side
        # enforcement (explicit prompt delimiter) is review_service's job (frozen).
        "untrusted_content_policy": {
            "candidate_text_is_data": True,
            "candidate_instructions_ignored": True,
            "authority_claims_require_ledger_proof": True,
        },
        "sanitization": sanitization,
    }
    if "objective" in full_pkg:
        # Verbatim: the objective is treated as DATA; in-content injection in it is a
        # render-layer quarantine concern, not stripped here (would delete signal).
        sanitized_view["objective"] = full_pkg["objective"]
    if "acceptance_checks" in full_pkg:
        sanitized_view["acceptance_checks"] = list(full_pkg["acceptance_checks"])

    out = {k: v for k, v in full_pkg.items() if k != "sanitized_view"}
    out["sanitized_view"] = sanitized_view
    return out


def sanitized_view_text(sanitized_view: dict[str, Any]) -> str:
    """Flatten the reviewer-visible content the way the reviewer prompt would see it."""
    parts: list[str] = [str(sanitized_view.get("objective", ""))]
    for o in sanitized_view.get("outputs", []) or []:
        parts.append(str(o.get("ref", "")))
        parts.append(str(o.get("label", "")))
    for c in sanitized_view.get("acceptance_checks", []) or []:
        parts.append(str(c.get("statement", "")))
    return "\n".join(parts)
