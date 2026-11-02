"""Atomic source-bound CRM pair custody; not a measurement grading rule."""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "dtap.crm_pair_receipt.v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_FIELDS = ("source_tree_sha256", "model_sha256", "server_sha256", "template_sha256",
           "tool_schema_sha256", "case_sha256")


def read_journal(path):
    raw = Path(path).read_bytes()
    if not raw or not raw.endswith(b"\n"):
        raise ValueError("empty or truncated journal")
    previous = "0" * 64
    records = []
    for line in raw.splitlines(keepends=True):
        try:
            row = json.loads(line)
        except (ValueError, UnicodeDecodeError) as exc:
            raise ValueError("invalid journal JSON") from exc
        if not isinstance(row, dict) or row.get("previous_sha256") != previous:
            raise ValueError("journal hash chain mismatch")
        if not isinstance(row.get("kind"), str) or not isinstance(row.get("utc"), str):
            raise ValueError("journal event lacks kind/time")
        previous = hashlib.sha256(line).hexdigest()
        records.append(row)
    if records[0]["kind"] != "start":
        raise ValueError("journal lacks start")
    return raw, records


def validate_provenance(capture_kind, provenance):
    if capture_kind not in ("live", "fixture") or not isinstance(provenance, dict):
        raise ValueError("invalid capture kind or provenance")
    if not _COMMIT.fullmatch(str(provenance.get("source_commit", ""))):
        raise ValueError("source_commit must be a full commit")
    for field in _FIELDS:
        if not _SHA256.fullmatch(str(provenance.get(field, ""))):
            raise ValueError(f"missing or invalid {field}")
    if not isinstance(provenance.get("grant_id"), str) or not provenance["grant_id"]:
        raise ValueError("missing grant_id")
    for field in ("model_pid", "model_pid_starttime"):
        if type(provenance.get(field)) is not int or provenance[field] <= 0:
            raise ValueError(f"missing {field}")
    if provenance.get("backend") not in ("cpu", "gpu", "unknown"):
        raise ValueError("backend must be cpu, gpu, or unknown")
    model_file = provenance.get("model_file")
    if model_file is not None and (not isinstance(model_file, str) or not model_file
                                   or Path(model_file).name != model_file):
        raise ValueError("model_file must be a basename when provided")


def seal_pair(*, journal_path, output_path, capture_kind, provenance,
              grant_receipt_path, source_manifest_path, producer_path,
              claim_open_path=None, claim_close_path=None, claim_pins=None, posthash_path=None):
    """Seal producer facts; readers must independently verify original warrants."""
    validate_provenance(capture_kind, provenance)
    dependencies = {}
    for name, file in (("grant_receipt", grant_receipt_path), ("source_manifest", source_manifest_path),
                       ("producer", producer_path)):
        path = Path(file)
        data = path.read_bytes()
        if not data:
            raise ValueError(f"empty {name}")
        dependencies[name + "_path"] = str(path.resolve())
        dependencies[name + "_sha256"] = hashlib.sha256(data).hexdigest()
    raw, records = read_journal(journal_path)
    start = records[0].get("payload")
    if not isinstance(start, dict) or start.get("capture_kind") != capture_kind or start.get("provenance") != provenance:
        raise ValueError("capture provenance differs from native start")
    manifests = [r["payload"] for r in records if r["kind"] == "pair_manifest"]
    if len(manifests) != 1 or not isinstance(manifests[0], dict):
        raise ValueError("expected one native pair manifest")
    manifest = manifests[0].get("manifest", {})
    if (manifest.get("schema_sha256") != provenance["tool_schema_sha256"]
            or manifest.get("registry_sha256") != provenance["case_sha256"]):
        raise ValueError("case/schema differs from native manifest")
    terminals = [r for r in records if r["kind"] == "terminal"]
    if len(terminals) != 1 or not isinstance(terminals[0].get("payload"), dict):
        raise ValueError("expected one terminal event")
    if terminals[0] is not records[-1]:
        raise ValueError("terminal must be the final journal event")
    result = terminals[0]["payload"]
    arms = result.get("arms")
    if not isinstance(arms, dict) or set(arms) != {"control", "attack"}:
        raise ValueError("terminal must contain both arms")
    for arm, item in arms.items():
        if not isinstance(item, dict) or item.get("status") not in ("complete", "failed", "not_attempted"):
            raise ValueError(f"invalid {arm} status")
        if item.get("delivery") not in ("not_observed", "queued_not_confirmed", "delivered"):
            raise ValueError(f"invalid {arm} delivery")
        events = [r for r in records if isinstance(r.get("payload"), dict) and r["payload"].get("arm") == arm]
        expected = {"complete": "judged", "failed": "arm_failed", "not_attempted": "arm_skipped"}[item["status"]]
        if not any(r["kind"] == expected for r in events):
            raise ValueError(f"missing {arm} terminal evidence")
        if item["status"] == "complete" and not any(r["kind"] == "reset" for r in events):
            raise ValueError(f"missing {arm} reset")
        if item["status"] == "not_attempted" and (item["delivery"] != "not_observed"
                or any(r["kind"] in ("reset", "request", "model", "tool_attempt", "tool_result", "judged", "arm_failed")
                       for r in events)):
            raise ValueError(f"{arm} skipped but has attempted events")
        terminal_events = [r for r in events if r["kind"] == expected]
        if len(terminal_events) != 1 or terminal_events[0]["payload"].get("delivery") != item["delivery"]:
            raise ValueError(f"{arm} delivery differs from native terminal evidence")
    if arms["attack"]["status"] == "not_attempted" and arms["control"]["status"] != "failed":
        raise ValueError("attack may be skipped only after control failure")
    if arms["control"]["status"] == "not_attempted":
        raise ValueError("control cannot be skipped in a sealed pair")
    if result.get("cleanup") != "completed_by_backend" or not any(
            r["kind"] == "cleanup" and isinstance(r.get("payload"), dict)
            and r["payload"].get("status") == "completed_by_backend" for r in records):
        raise ValueError("successful backend cleanup not recorded")
    claim = {}
    if capture_kind == "live":
        if not claim_open_path or not claim_close_path or not isinstance(claim_pins, dict):
            raise ValueError("live seal requires provider open/close and independently selected pins")
        if not posthash_path:
            raise ValueError("live seal requires separately captured post-hash")
        from .crm_source_manifest import verify_manifest
        verify_manifest(source_manifest_path, provenance["source_commit"], provenance["source_tree_sha256"])
        from src.runtime.region_claim_receipt import verify as verify_claim
        open_path, close_path = Path(claim_open_path), Path(claim_close_path)
        opened, closed = verify_claim(open_path, close_path, **claim_pins)
        if opened["grant_id"] != provenance["grant_id"]:
            raise ValueError("provider grant ID differs from capture")
        if (Path(opened["grant_copy_path"]).resolve() != Path(grant_receipt_path).resolve()
                or opened["grant_sha256"] != dependencies["grant_receipt_sha256"]):
            raise ValueError("provider grant bytes differ from capture")
        def timestamp(value):
            result = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if result.tzinfo is None:
                raise ValueError("capture timestamp requires timezone")
            return result
        start, end = timestamp(records[0]["utc"]), timestamp(terminals[0]["utc"])
        if not (timestamp(opened["opened_utc"]) <= start <= end <= timestamp(closed["closed_utc"])
                and timestamp(opened["grant_start"]) <= start <= end <= timestamp(opened["grant_end"])):
            raise ValueError("capture interval is outside claim or grant")
        post_path = Path(posthash_path)
        post_data = post_path.read_bytes()
        post = json.loads(post_data)
        if (post.get("schema") != "dtap.crm_posthash.v1" or post.get("capture_kind") != "live"
                or post.get("fact", {}).get("sha256") != provenance["model_sha256"]
                or post.get("claim_close_sha256") != hashlib.sha256(close_path.read_bytes()).hexdigest()
                or post.get("pair_journal_sha256") != hashlib.sha256(raw).hexdigest()
                or timestamp(post["started_utc"]) < timestamp(closed["closed_utc"])):
            raise ValueError("post-hash does not bind capture/model/provider close")
        claim = {"claim_open_path": str(open_path.resolve()), "claim_open_sha256": hashlib.sha256(open_path.read_bytes()).hexdigest(),
                 "claim_close_path": str(close_path.resolve()), "claim_close_sha256": hashlib.sha256(close_path.read_bytes()).hexdigest(),
                 "claim_pins": dict(claim_pins), "posthash_path": str(post_path.resolve()),
                 "posthash_sha256": hashlib.sha256(post_data).hexdigest()}
    seal = {"schema": SCHEMA, "capture_kind": capture_kind, "journal_path": str(Path(journal_path).resolve()),
            "journal_sha256": hashlib.sha256(raw).hexdigest(), **dependencies, **claim, "provenance": dict(provenance),
            "arms": arms, "cleanup": result["cleanup"], "capture_start_utc": records[0]["utc"],
            "terminal_utc": terminals[0]["utc"], "sealed_utc": datetime.now(timezone.utc).isoformat()}
    data = (json.dumps(seal, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode()
    output = Path(output_path)
    fd, temporary = tempfile.mkstemp(prefix=".crm-seal-", dir=output.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output, follow_symlinks=False)
        directory_fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        os.unlink(temporary)
    return seal
