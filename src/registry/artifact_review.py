"""Uploader class + review record gate for tool-bearing roles (SEC-ART-3).

Owner handoff: handoffs/active/model-artifact-integrity.md (epyc-root).

Registry fields on ``roles.<role>.model`` (pass through ``registry_compiler.compile_lean`` unchanged):

  uploader_class: first_party | allowlisted_requantizer | community
  artifact_review:                       # required when a COMMUNITY artifact is tool-bearing
    record: <path to a YAML review record, relative to the orchestrator repo root or absolute>

The review record is a YAML file with:

  reviewed_by: <who>              reviewed_on: <YYYY-MM-DD>
  provenance: <free text; for a repo that presents itself as a copy of an upstream this holds the SEC-ART-9
              LFS-oid comparison and the list of files the upstream lacks>
  base_model_diff: <path to the diff/report>   OR   none_no_base_model   (only when no base exists)
  flag_check:                     # label-vs-function check of EVERY generation-changing launch flag
    <flag-or-knob>: {label: <what it claims>, function: <what it does>, verdict: consistent}

``declared_generation_flags`` is the set the record must cover: every knob the registry declares for that role
that changes generation (chat template file/kwargs, ...). The gate (``check_uploader_review``) runs inside
``registry_validator.validate_all`` (every ``orchestrator_stack.py start``) and the stack-change pipeline's
``artifact_integrity`` step. A community role without a complete record is an ERROR; a tool-bearing role with
no ``uploader_class`` at all is a WARNING until the strict switch (``REGISTRY_ARTIFACT_PINS_STRICT``) flips.

Which uploaders count as ``allowlisted_requantizer`` is an OPERATOR policy input; this module never infers it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

UPLOADER_CLASSES = ("first_party", "allowlisted_requantizer", "community")
NO_BASE = "none_no_base_model"
_VERDICT_OK = "consistent"
_REPO_ROOT = Path(__file__).resolve().parents[2]


def is_tool_bearing(role_cfg: dict[str, Any]) -> bool:
    """A role that can reach tools: any allowed category, or web access, in ``tool_permissions``."""
    perms = role_cfg.get("tool_permissions")
    if not isinstance(perms, dict):
        return False
    return bool(perms.get("web_access") or perms.get("allowed_categories"))


def declared_generation_flags(registry: dict[str, Any], role: str) -> list[str]:
    """Registry-declared knobs that change generation for ``role`` (what a review must label-check)."""
    flags: list[str] = []
    for section in ((registry.get("server_mode") or {}).get(role), (registry.get("roles") or {}).get(role)):
        if not isinstance(section, dict):
            continue
        if section.get("chat_template_file"):
            flags.append("--chat-template-file")
        if isinstance(section.get("chat_template_kwargs"), dict) and section["chat_template_kwargs"]:
            flags.append("chat_template_kwargs")
        for key in ("lora", "lora_path", "control_vector", "override_kv"):
            if section.get(key):
                flags.append(key)
    return sorted(set(flags))


def _resolve(path: str, repo_root: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else repo_root / p


def _record_errors(role: str, record_ref: Any, required_flags: list[str], repo_root: Path) -> list[str]:
    where = f"roles.{role}.model.artifact_review"
    if not isinstance(record_ref, dict) or not isinstance(record_ref.get("record"), str):
        return [f"{where}.record: a path to the review record is required for a community artifact"]
    path = _resolve(record_ref["record"], repo_root)
    try:
        rec = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        return [f"{where}.record {record_ref['record']}: unreadable ({exc})"]
    if not isinstance(rec, dict):
        return [f"{where}.record {record_ref['record']}: not a mapping"]
    errs: list[str] = []
    for key in ("reviewed_by", "reviewed_on", "provenance"):
        if not rec.get(key):
            errs.append(f"{where}.record: missing {key}")
    diff = rec.get("base_model_diff")
    if not diff:
        errs.append(f"{where}.record: missing base_model_diff (a path, or {NO_BASE})")
    elif diff != NO_BASE and not _resolve(str(diff), repo_root).exists():
        errs.append(f"{where}.record: base_model_diff {diff} does not exist")
    checked = rec.get("flag_check")
    if not isinstance(checked, dict):
        errs.append(f"{where}.record: missing flag_check mapping")
        checked = {}
    for flag in required_flags:
        entry = checked.get(flag)
        if not (isinstance(entry, dict) and entry.get("label") and entry.get("function")
                and entry.get("verdict") == _VERDICT_OK):
            errs.append(
                f"{where}.record: flag_check lacks a consistent label-vs-function entry for {flag}"
            )
    return errs


def check_uploader_review(
    registry: dict[str, Any], *, strict: bool = False, repo_root: Path = _REPO_ROOT
) -> tuple[list[str], list[str]]:
    """Return ``(errors, warnings)`` over every lineup role with a ``.gguf`` model."""
    errors: list[str] = []
    warnings: list[str] = []
    for role, cfg in sorted((registry.get("roles") or {}).items()):
        model = cfg.get("model") if isinstance(cfg, dict) else None
        if not isinstance(model, dict):
            continue
        path = model.get("path") or model.get("full_path")
        if not (isinstance(path, str) and path.endswith(".gguf")):
            continue
        klass = model.get("uploader_class")
        if klass is not None and klass not in UPLOADER_CLASSES:
            errors.append(f"roles.{role}.model.uploader_class={klass!r} not one of {UPLOADER_CLASSES}")
            continue
        if not is_tool_bearing(cfg):
            continue
        if klass is None:
            msg = f"roles.{role} is tool-bearing but roles.{role}.model.uploader_class is unset"
            (errors if strict else warnings).append(msg)
            continue
        if klass == "community":
            errors += _record_errors(
                role, model.get("artifact_review"), declared_generation_flags(registry, role), repo_root
            )
    return errors, warnings
