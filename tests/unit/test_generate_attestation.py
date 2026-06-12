"""Tests for running-state attestation generation."""

from __future__ import annotations

from pathlib import Path

from scripts.attest import generate_attestation as attest


def test_parse_process_args_for_llama_server() -> None:
    cmdline = [
        "/opt/llama.cpp/build/bin/llama-server",
        "-m",
        "/models/model.gguf",
        "-md",
        "/models/draft.gguf",
        "--host",
        "127.0.0.1",
        "--port",
        "8070",
        "-np",
        "1",
        "-c",
        "32768",
        "-t",
        "96",
        "-ub",
        "8192",
        "-ctk",
        "q8_0",
        "-ctv",
        "q8_0",
        "--flash-attn",
        "on",
        "--mlock",
    ]

    assert attest.classify_process(cmdline) == "llama_server"
    parsed = attest.parse_process_args(cmdline)

    assert parsed["port"] == 8070
    assert parsed["model_path"] == "/models/model.gguf"
    assert parsed["draft_model_path"] == "/models/draft.gguf"
    assert parsed["parallel_slots"] == "1"
    assert parsed["context_length"] == "32768"
    assert parsed["threads"] == "96"
    assert parsed["ubatch_size"] == "8192"
    assert parsed["kv_cache_type_k"] == "q8_0"
    assert parsed["kv_cache_type_v"] == "q8_0"
    assert parsed["flash_attention"] == "on"
    assert parsed["mlock"] is True


def test_classify_process_does_not_match_supervisor_flag_text() -> None:
    cmdline = [
        "/usr/local/bin/earlyoom",
        "--ignore",
        "^(llama-server|sd-server)$",
    ]

    assert attest.classify_process(cmdline) is None
    assert attest.parse_process_args(cmdline) == {"port": None}


def test_registry_port_map_includes_numa_ports(tmp_path: Path) -> None:
    registry = tmp_path / "registry.yaml"
    registry.write_text(
        """
server_mode:
  frontdoor:
    port: 8070
    model_path: /models/frontdoor.gguf
    numa_ports: [8080, 8180]
  worker_vision:
    port: 8086
    model:
      name: Qwen2.5-VL
      path: /models/vl.gguf
""",
        encoding="utf-8",
    )

    ports = attest.load_registry_ports(registry)

    assert ports[8070][0]["registry_section"] == "server_mode.frontdoor"
    assert ports[8070][0]["port_kind"] == "primary"
    assert ports[8080][0]["port_kind"] == "numa_replica"
    assert ports[8180][0]["role"] == "frontdoor"
    assert ports[8086][0]["model_name"] == "Qwen2.5-VL"


def test_registry_port_map_preserves_shared_port_aliases(tmp_path: Path) -> None:
    registry = tmp_path / "registry.yaml"
    registry.write_text(
        """
server_mode:
  frontdoor:
    port: 8070
    model_path: /models/frontdoor.gguf
  coder_escalation:
    port: 8070
    model_path: /models/frontdoor.gguf
""",
        encoding="utf-8",
    )

    ports = attest.load_registry_ports(registry)

    assert [entry["registry_section"] for entry in ports[8070]] == [
        "server_mode.frontdoor",
        "server_mode.coder_escalation",
    ]


def test_llama_resolution_detects_cross_tree_mismatch() -> None:
    status = attest.llama_resolution_status(
        "/mnt/raid0/llm/ik_llama.cpp/build/bin/llama-server",
        {
            "libllama.so": "/mnt/raid0/llm/llama.cpp/build/bin/libllama.so",
            "libggml.so": "/mnt/raid0/llm/ik_llama.cpp/build/ggml/src/libggml.so",
        },
    )

    assert status["expected_tree"] == "/mnt/raid0/llm/ik_llama.cpp"
    assert status["issues"] == ["libllama.so_tree_mismatch:/mnt/raid0/llm/llama.cpp"]


def test_collect_feature_flags_detects_worker_drift(monkeypatch, tmp_path: Path) -> None:
    proc_root = tmp_path / "proc"
    for pid, value in ((101, "1"), (102, "0")):
        pid_dir = proc_root / str(pid)
        pid_dir.mkdir(parents=True)
        (pid_dir / "environ").write_bytes(f"ORCHESTRATOR_FEATURE_MODEL_FALLBACK={value}\0".encode())
    responses = [
        {
            "pid": 101,
            "flags": {"model_fallback": True},
            "sources": {"model_fallback": "ORCHESTRATOR_FEATURE_MODEL_FALLBACK"},
        },
        {
            "pid": 102,
            "flags": {"model_fallback": False},
            "sources": {"model_fallback": "ORCHESTRATOR_FEATURE_MODEL_FALLBACK"},
        },
    ]

    monkeypatch.setattr(attest, "_fetch_json", lambda _url: responses.pop(0))
    monkeypatch.setattr(
        attest,
        "load_declared_feature_env",
        lambda: {
            "status": "ok",
            "env": {"ORCHESTRATOR_FEATURE_MODEL_FALLBACK": "1"},
            "flag_env_names": {
                "model_fallback": "ORCHESTRATOR_FEATURE_MODEL_FALLBACK",
            },
            "flags": {"model_fallback": True},
        },
    )

    report = attest.collect_feature_flags(polls=2, delay_s=0, proc_root=proc_root)

    assert report["status"] == "warn"
    assert report["heterogeneous"] == {
        "model_fallback": {"101": True, "102": False},
    }
    assert report["intent_diffs"] == [
        {
            "pid": "102",
            "flag": "model_fallback",
            "expected": True,
            "actual": False,
            "source": "ORCHESTRATOR_FEATURE_MODEL_FALLBACK",
        }
    ]
    assert report["env_diffs"] == [
        {
            "pid": "102",
            "env": "ORCHESTRATOR_FEATURE_MODEL_FALLBACK",
            "expected": "1",
            "actual": "0",
        }
    ]


def test_build_serving_config_reports_numa_match() -> None:
    rows = attest.build_serving_config(
        [
            {
                "pid": 123,
                "kind": "llama_server",
                "port": 8070,
                "registry_matches": [{"registry_section": "server_mode.frontdoor"}],
                "args": {
                    "model_path": "/models/model.gguf",
                    "context_length": "32768",
                    "threads": "96",
                },
                "cpus_allowed_list": "0-3",
            }
        ],
        numa_ports={8070: {"role": "frontdoor", "cpu_list": "0-3", "threads": 96}},
    )

    assert rows[0]["numa_match"] is True
    assert rows[0]["numa_intent"]["role"] == "frontdoor"


def test_build_report_from_fake_proc(monkeypatch, tmp_path: Path) -> None:
    proc_root = tmp_path / "proc"
    pid_dir = proc_root / "123"
    pid_dir.mkdir(parents=True)
    (proc_root / "stat").write_text("btime 1700000000\n", encoding="utf-8")
    start_fields = ["0"] * 20
    start_fields[19] = "100"
    (pid_dir / "stat").write_text(
        "123 (llama-server) S " + " ".join(start_fields),
        encoding="utf-8",
    )
    (pid_dir / "status").write_text(
        "Name:\tllama-server\nState:\tS (sleeping)\nCpus_allowed_list:\t0-3\n",
        encoding="utf-8",
    )
    binary = tmp_path / "llama.cpp" / "build" / "bin" / "llama-server"
    binary.parent.mkdir(parents=True)
    binary.write_text("not an elf", encoding="utf-8")
    (pid_dir / "exe").symlink_to(binary)
    (pid_dir / "cmdline").write_bytes(
        b"/tmp/llama.cpp/build/bin/llama-server\0-m\0/models/model.gguf\0--port\08070\0"
    )
    registry = tmp_path / "registry.yaml"
    registry.write_text(
        "server_mode:\n  frontdoor:\n    port: 8070\n    model_path: /models/model.gguf\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        attest,
        "run_dynamic_checks",
        lambda _exe: {"status": "ok", "issues": [], "readelf": {}, "ldd": {}},
    )
    monkeypatch.setattr(
        attest,
        "load_numa_ports",
        lambda: {8070: {"role": "frontdoor", "cpu_list": "0-3", "threads": 4}},
    )

    report = attest.build_report(
        registry=registry,
        proc_root=proc_root,
        generated_at="2026-06-12T00:00:00Z",
    )

    assert report["summary"]["process_count"] == 1
    process = report["sections"]["processes"][0]
    assert process["pid"] == 123
    assert process["kind"] == "llama_server"
    assert process["port"] == 8070
    assert process["registry_matches"][0]["registry_section"] == "server_mode.frontdoor"
    assert process["cpus_allowed_list"] == "0-3"
    assert process["start_time"] == "2023-11-14T22:13:20Z"
    assert report["sections"]["feature_flags"]["status"] == "disabled"
    assert report["sections"]["serving_config"][0]["numa_match"] is True
    assert report["summary"]["issue_count"] == 0
