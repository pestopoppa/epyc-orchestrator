#!/usr/bin/env python3
"""Apply SEC-ART-1/2/4 backfill plans to a registry file WITHOUT disturbing its comments.

The master registry is a 13k-line hand-annotated YAML file; a yaml.dump round-trip would destroy the comments
that carry its history. This tool inserts the new keys line-by-line and then proves the edit by re-parsing:
the result must equal the original document plus EXACTLY the planned keys (no other change, no duplicate keys).

    apply_registry_integrity_pins.py --registry MASTER.yaml \\
        [--artifact-plan plan.json] [--template-plan plan.json] [--uploader-map map.json] [--dry-run]

  artifact plan  (backfill_registry_artifact_pins.py)  -> roles.<role>.model.{hf_repo,hf_revision,sha256,shard_sha256}
  template plan  (backfill_template_pins.py / template_provenance.py) -> server_mode.<role>.chat_template.
                 {served_template_sha256, upstream_template_sha256, template_diff, upstream_ref, upstream_revision}
  uploader map   {"<role>": "first_party|allowlisted_requantizer|community"} -> roles.<role>.model.uploader_class

Existing keys are never overwritten: a role that already carries a different value is reported and left alone.
Only `status: pin` plan rows are applied. The target is a MASTER registry (epyc-inference-research); never run this
against a generated lean view.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml

CLASSES = ("first_party", "allowlisted_requantizer", "community")


class ApplyError(RuntimeError):
    pass


def _section(lines: list[str], key: str) -> tuple[int, int]:
    """[start, end) of top-level `key:` (end = next unindented key line)."""
    start = next((i for i, ln in enumerate(lines) if ln == f"{key}:"), None)
    if start is None:
        raise ApplyError(f"top-level section {key!r} not found")
    end = len(lines)
    for i in range(start + 1, len(lines)):
        if lines[i] and not lines[i].startswith((" ", "#")):
            end = i
            break
    return start, end


def _child(lines: list[str], lo: int, hi: int, indent: int, key: str) -> tuple[int, int] | None:
    """[start, end) of the mapping child `key:` at `indent` spaces inside lines[lo:hi]; None if absent."""
    pad = " " * indent
    start = next((i for i in range(lo, hi) if lines[i].startswith(f"{pad}{key}:")
                  and (len(lines[i]) == len(pad) + len(key) + 1 or lines[i][len(pad) + len(key) + 1] in " #")), None)
    if start is None:
        return None
    end = hi
    for i in range(start + 1, hi):
        ln = lines[i]
        if ln.strip() and not ln.lstrip().startswith("#") and (len(ln) - len(ln.lstrip())) <= indent:
            end = i
            break
    return start, end


_HEXISH = re.compile(r"^[0-9a-f]{40}$|^[0-9a-f]{64}$")


def _scalar(v: Any) -> str:
    """Digests are double-quoted: an all-digit (or digits-e-digits) hex string would otherwise parse as a number."""
    v = str(v)
    return f'"{v}"' if _HEXISH.match(v) else v


def insert_model_pins(lines: list[str], role: str, pins: dict[str, Any]) -> list[str]:
    lo, hi = _section(lines, "roles")
    rr = _child(lines, lo + 1, hi, 2, role)
    if rr is None:
        raise ApplyError(f"roles.{role} not found")
    mr = _child(lines, rr[0] + 1, rr[1], 4, "model")
    if mr is None or lines[mr[0]].strip() != "model:":
        raise ApplyError(f"roles.{role}.model is not a block mapping")
    have = {m.group(1) for ln in lines[mr[0] + 1:mr[1]] if (m := re.match(r"^      ([A-Za-z0-9_]+):", ln))}
    anchor = next((i for i in range(mr[0] + 1, mr[1]) if lines[i].startswith("      path:")), None)
    if anchor is None:
        raise ApplyError(f"roles.{role}.model has no path line")
    new: list[str] = []
    for key in ("uploader_class", "hf_repo", "hf_revision", "sha256"):
        if key in pins and key not in have:
            new.append(f"      {key}: {_scalar(pins[key])}")
    if pins.get("shard_sha256") and "shard_sha256" not in have:
        new.append("      shard_sha256:")
        new += [f"        {name}: {_scalar(digest)}" for name, digest in sorted(pins["shard_sha256"].items())]
    return lines[:anchor + 1] + new + lines[anchor + 1:]


TEMPLATE_KEYS = (
    "served_template_sha256", "upstream_template_sha256", "template_diff",
    "upstream_ref", "upstream_revision", "template_diff_reviewed",
)


def insert_template_pin(lines: list[str], role: str, keys: dict[str, str]) -> list[str]:
    """Insert `chat_template.<key>` lines for every key in TEMPLATE_KEYS not already present."""
    lo, hi = _section(lines, "server_mode")
    rr = _child(lines, lo + 1, hi, 2, role)
    if rr is None:
        raise ApplyError(f"server_mode.{role} not found")
    cr = _child(lines, rr[0] + 1, rr[1], 4, "chat_template")
    ordered = [k for k in TEMPLATE_KEYS if k in keys]
    if cr is not None:
        if lines[cr[0]].strip() != "chat_template:":
            raise ApplyError(f"server_mode.{role}.chat_template is not a block mapping")
        have = {m.group(1) for ln in lines[cr[0] + 1:cr[1]] if (m := re.match(r"^      ([A-Za-z0-9_]+):", ln))}
        new = [f"      {k}: {_scalar(keys[k])}" for k in ordered if k not in have]
        if not new:
            return lines
        at = cr[0]
        for prefix in ("      served_template_sha256_12:", "      served_template_sha256:"):  # later one wins
            hit = next((i for i in range(cr[0] + 1, cr[1]) if lines[i].startswith(prefix)), None)
            if hit is not None:
                at = max(at, hit)
        return lines[:at + 1] + new + lines[at + 1:]
    # No block: add one after chat_template_file (the pilot roles), else model_role, else port (single-line anchors).
    at = next((i for i in range(rr[0] + 1, rr[1]) if lines[i].startswith("    chat_template_file:")), None)
    for anchor in ("    model_role:", "    port:"):
        if at is None:
            at = next((i for i in range(rr[0] + 1, rr[1]) if lines[i].startswith(anchor)), None)
    if at is None:
        raise ApplyError(f"server_mode.{role}: no anchor (chat_template_file/model_role/port) to attach a chat_template block")
    return lines[:at + 1] + ["    chat_template:"] + [f"      {k}: {_scalar(keys[k])}" for k in ordered] + lines[at + 1:]


def _expected(doc: dict[str, Any], model_pins: dict[str, dict], tmpl_pins: dict[str, dict[str, str]]) -> dict[str, Any]:
    out = copy.deepcopy(doc)
    for role, pins in model_pins.items():
        model = out["roles"][role]["model"]
        for k, v in pins.items():
            model.setdefault(k, v)
    for role, keys in tmpl_pins.items():
        block = out["server_mode"][role].setdefault("chat_template", {})
        for k, v in keys.items():
            block.setdefault(k, v)
    return out


def apply(text: str, artifact_rows: list[dict], template_rows: list[dict], uploader_map: dict[str, str],
          reviewed_note: str | None = None) -> tuple[str, list[str]]:
    notes: list[str] = []
    doc = yaml.safe_load(text)
    lines = text.split("\n")
    model_pins: dict[str, dict] = {}
    for r in artifact_rows:
        if r.get("status") != "pin":
            continue
        pins = {k: r[k] for k in ("hf_repo", "hf_revision", "sha256") if r.get(k)}
        if r.get("shard_sha256"):
            pins["shard_sha256"] = r["shard_sha256"]
        model_pins[r["role"]] = pins
    for role, klass in uploader_map.items():
        if klass not in CLASSES:
            raise ApplyError(f"uploader_class {klass!r} for {role} not in {CLASSES}")
        model_pins.setdefault(role, {})["uploader_class"] = klass
    tmpl_pins = {
        r["role"]: {**{k: str(r[k]) for k in TEMPLATE_KEYS if r.get(k)},
                    **({"template_diff_reviewed": reviewed_note} if reviewed_note and r.get("template_diff") else {})}
        for r in template_rows if r.get("status") == "pin"
    }
    for role, pins in sorted(model_pins.items()):
        existing = (doc.get("roles", {}).get(role, {}).get("model") or {})
        for k, v in pins.items():
            if k in existing and existing[k] != v:
                notes.append(f"roles.{role}.model.{k} already {existing[k]!r}; plan says {v!r} -- left alone")
        lines = insert_model_pins(lines, role, pins)
    for role, keys in sorted(tmpl_pins.items()):
        block = (doc.get("server_mode", {}).get(role, {}) or {}).get("chat_template") or {}
        for k, v in keys.items():
            if block.get(k) and block[k] != v:
                notes.append(f"server_mode.{role}.chat_template.{k} already {block[k]!r}; plan says {v!r} -- left alone")
        lines = insert_template_pin(lines, role, keys)
    new_text = "\n".join(lines)
    if (yaml.safe_load(new_text) != _expected(doc, model_pins, tmpl_pins)):
        raise ApplyError("post-edit document differs from original + planned keys; refusing to write")
    return new_text, notes


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--registry", required=True)
    ap.add_argument("--artifact-plan")
    ap.add_argument("--template-plan")
    ap.add_argument("--uploader-map")
    ap.add_argument("--reviewed-note", help="text recorded as chat_template.template_diff_reviewed for rows that carry a diff")
    ap.add_argument("--only-role", action="append", help="apply only these roles (repeatable); default all `pin` rows")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    path = Path(a.registry)
    arows = json.loads(Path(a.artifact_plan).read_text())["rows"] if a.artifact_plan else []
    trows = json.loads(Path(a.template_plan).read_text())["rows"] if a.template_plan else []
    umap = json.loads(Path(a.uploader_map).read_text()) if a.uploader_map else {}
    if a.only_role:
        keep = set(a.only_role)
        arows = [r for r in arows if r.get("role") in keep]
        trows = [r for r in trows if r.get("role") in keep]
        umap = {k: v for k, v in umap.items() if k in keep}
    try:
        new_text, notes = apply(path.read_text(), arows, trows, umap, a.reviewed_note)
    except ApplyError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1
    for n in notes:
        print(f"note: {n}")
    if a.dry_run:
        print("dry-run: edit verified, nothing written")
        return 0
    path.write_text(new_text)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
