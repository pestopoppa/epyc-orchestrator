"""Non-inference checks of the KPF-26 paired replay custody contract."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[2] / "scripts/trace/kpf26_paired_replay.py"
spec = importlib.util.spec_from_file_location("kpf26_paired_replay", PATH)
kpf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kpf)


def fixture(tmp_path):
    (tmp_path / "request.json").write_text('{"prompt":"test"}')
    (tmp_path / "expected.json").write_text('{"result":"yes"}')
    def h(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()
    m = {"schema": kpf.SCHEMA, "arm_order": ["off", "on"], "arm_timeout_s": 30, "max_occupancy_sample_gap_s": 2,
         "pins": {"model": "test", "server_binary_sha256": "a" * 64, "source_commit": "1234", "argv": "test", "executor_sha256": "b" * 64},
         "tasks": [{"id": "t1", "siblings": 1, "request": {"path": "request.json", "sha256": h(tmp_path / "request.json")},
                    "expected": {"path": "expected.json", "sha256": h(tmp_path / "expected.json")}}]}
    return m


def evidence(m, arm):
    task = m["tasks"][0]
    return {"arm": arm, "manifest_sha256": "c" * 64, "pins": m["pins"], "window_id": "win-" + arm,
            "flags": {"prefix_index": arm == "on", "fork": arm == "on", "lpm_max_skips": 1},
            "tasks": [{"id": "t1", "request_sha256": task["request"]["sha256"],
                       "expected_sha256": task["expected"]["sha256"], "start_s": 1, "end_s": 2,
                       "unique_cells_samples": [{"t": 1, "cells": 5}, {"t": 2, "cells": 7}],
                       "trunk": {"count": 1, "checkpoint_witness": "ckpt", "prefill_tokens": 3},
                       "children": [{"id": "c1", "status": "ok", "ttft_ms": 2, "admission_wait_ms": 0,
                                     "prefill_tokens": 3, "output_sha256": "d" * 64, "outcome": True,
                                     "n_fork_tokens": 2, "fork_task_slot_witness": "slot"}]}]}


def test_manifest_binding_and_duplicates(tmp_path):
    m = fixture(tmp_path)
    kpf.validate_manifest(m, tmp_path)
    m["tasks"].append(m["tasks"][0])
    with pytest.raises(ValueError, match="duplicate"):
        kpf.validate_manifest(m, tmp_path)
    m = fixture(tmp_path)
    (tmp_path / "request.json").write_text("changed")
    with pytest.raises(ValueError, match="pin mismatch"):
        kpf.validate_manifest(m, tmp_path)


@pytest.mark.parametrize("arm", ["off", "on"])
def test_complete_evidence_and_gaps(tmp_path, arm):
    m = fixture(tmp_path)
    e = evidence(m, arm)
    assert kpf.validate_evidence(e, m, "c" * 64, arm) == []
    e["tasks"][0]["unique_cells_samples"] = [{"t": 2, "cells": 7}]
    assert any("occupancy" in x for x in kpf.validate_evidence(e, m, "c" * 64, arm))
    e = evidence(m, arm)
    e["tasks"][0]["children"][0]["admission_wait_ms"] = None
    assert any("incomplete" in x for x in kpf.validate_evidence(e, m, "c" * 64, arm))


def test_fork_and_trunk_are_not_assumed(tmp_path):
    m = fixture(tmp_path)
    e = evidence(m, "on")
    e["tasks"][0]["trunk"]["checkpoint_witness"] = None
    e["tasks"][0]["children"][0]["fork_task_slot_witness"] = None
    errors = kpf.validate_evidence(e, m, "c" * 64, "on")
    assert any("trunk" in x for x in errors)
    assert any("fork" in x for x in errors)


def test_mock_executor_runs_both_arms_and_never_asserts_gate(tmp_path):
    m = fixture(tmp_path)
    exe = tmp_path / "executor"
    # Echo deterministic native-shaped mock evidence, never contacting a server.
    exe.write_text('#!/usr/bin/env python3\nimport hashlib,json,sys\n'
                   'a=sys.argv; raw=open(a[a.index("--manifest")+1],"rb").read(); m=json.loads(raw); '
                   'arm=a[a.index("--arm")+1]; t=m["tasks"][0]; '
                   'e={"arm":arm,"manifest_sha256":hashlib.sha256(raw).hexdigest(),"pins":m["pins"],'
                   '"window_id":arm,"flags":{"prefix_index":arm=="on","fork":arm=="on","lpm_max_skips":1},'
                   '"tasks":[{"id":t["id"],"request_sha256":t["request"]["sha256"],"expected_sha256":t["expected"]["sha256"],'
                   '"start_s":1,"end_s":2,"unique_cells_samples":[{"t":1,"cells":5},{"t":2,"cells":7}],'
                   '"trunk":{"count":1,"checkpoint_witness":"ckpt","prefill_tokens":3},'
                   '"children":[{"id":"c1","status":"ok","ttft_ms":2,"admission_wait_ms":0,"prefill_tokens":3,'
                   '"output_sha256":"d"*64,"outcome":True,"n_fork_tokens":2,"fork_task_slot_witness":"slot"}]}]}; '
                   'open(a[a.index("--evidence")+1],"w").write(json.dumps(e))\n')
    exe.chmod(0o755)
    m["pins"]["executor_sha256"] = hashlib.sha256(exe.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(m))
    assert kpf.run(manifest, exe, tmp_path / "out") == 0
    result = json.loads((tmp_path / "out/result.json").read_text())
    assert result["status"] == "OBSERVATION_COMPLETE"
    assert result["gate_pass"] is None
