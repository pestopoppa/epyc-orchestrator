"""Opt-in physical region claim receipt boundary."""
import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.runtime import cpu_region_lock, region_lock_cli
from src.runtime import region_claim_receipt as receipt


def test_provider_receipts_bracket_real_flock_and_detect_tamper(tmp_path, monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_TMP_DIR", str(tmp_path / "locks"))
    monkeypatch.setenv("ORCHESTRATOR_CROSS_ROLE_DISJOINT_PLACEMENT", "1")
    monkeypatch.delenv("FILLER_DRYRUN", raising=False)
    monkeypatch.delenv("FILLER_SMOKE", raising=False)
    monkeypatch.setattr(region_lock_cli, "_preflight", lambda **_: None)
    now = datetime.now(timezone.utc)
    grant = tmp_path / "test.grant"
    grant.write_text(f"regions=24-47 start={(now-timedelta(minutes=1)).isoformat()} "
                     f"end={(now+timedelta(minutes=5)).isoformat()} holds=q1\n")
    prefix = tmp_path / "claim"
    open_file = Path(str(prefix) + ".open.json")
    close_file = Path(str(prefix) + ".close.json")
    code = ("from pathlib import Path; "
            f"assert Path({str(open_file)!r}).is_file(); "
            f"assert not Path({str(close_file)!r}).exists()")
    args = region_lock_cli.build_parser().parse_args([
        "run", "--cpu-list", "24-47", "--role", "build", "--timeout-s", "3",
        "--claim-receipt-prefix", str(prefix), "--claim-grant-file", str(grant),
        "--", sys.executable, "-c", code])
    args.command = args.command[1:]
    assert region_lock_cli._run(args) == 0
    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    pins = {"provider_sha256": sha(region_lock_cli.__file__),
            "lock_sha256": sha(cpu_region_lock.__file__), "receipt_sha256": sha(receipt.__file__),
            "expected_lock_root": tmp_path / "locks"}
    opened, closed = receipt.verify(open_file, close_file, **pins)
    assert opened["regions"] == ["q1"]
    assert closed["child_rc"] == 0
    assert grant.name == opened["grant_id"] + ".grant"
    grant.unlink()  # rowflow removes the live grant; immutable captured bytes survive.
    receipt.verify(open_file, close_file, **pins)
    with pytest.raises(ValueError, match="lock root"):
        receipt.verify(open_file, close_file, **{**pins, "expected_lock_root": tmp_path / "other"})
    with pytest.raises(ValueError, match="provider source pin"):
        receipt.verify(open_file, close_file, **{**pins, "provider_sha256": "0" * 64})
    grant_copy = Path(opened["grant_copy_path"])
    grant_bytes = grant_copy.read_bytes()
    grant_copy.write_bytes(grant_bytes + b" ")
    with pytest.raises(ValueError, match="grant identity changed"):
        receipt.verify(open_file, close_file, **pins)
    grant_copy.write_bytes(grant_bytes)
    obj = json.loads(close_file.read_text())
    obj["claim_id"] = "forged"
    close_file.write_text(json.dumps(obj))
    with pytest.raises(ValueError, match="claim_id mismatch"):
        receipt.verify(open_file, close_file, **pins)


def test_dryrun_and_unpaired_arguments_never_emit_provider_receipts(tmp_path, monkeypatch):
    monkeypatch.setenv("FILLER_DRYRUN", "1")
    prefix = tmp_path / "claim"
    args = region_lock_cli.build_parser().parse_args([
        "run", "--cpu-list", "24-47", "--claim-receipt-prefix", str(prefix),
        "--claim-grant-file", str(tmp_path / "x.grant"), "--", "true"])
    args.command = args.command[1:]
    with pytest.raises(SystemExit, match="real full-quarter"):
        region_lock_cli._run(args)
    assert not Path(str(prefix) + ".open.json").exists()


def test_claim_open_refuses_without_provider_owned_kernel_flock(tmp_path):
    now = datetime.now(timezone.utc)
    grant = tmp_path / "test.grant"
    grant.write_text(f"regions=24-47 start={(now-timedelta(minutes=1)).isoformat()} "
                     f"end={(now+timedelta(minutes=5)).isoformat()} holds=q1\n")
    (tmp_path / "cpu_region.GLOBAL.q1.lock").touch()
    with pytest.raises(RuntimeError, match="lacks one witnessed global flock"):
        receipt.open_receipt(tmp_path / "claim", grant, "24-47", frozenset({"q1"}),
                             Path(region_lock_cli.__file__), Path(cpu_region_lock.__file__), tmp_path)
    assert not (tmp_path / "claim.open.json").exists()
