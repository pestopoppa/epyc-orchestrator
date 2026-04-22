"""Tests for NIB2-44 Agent-World env_synth scaffolding (AW-1..AW-5)."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, "/mnt/raid0/llm/epyc-orchestrator")

from scripts.autopilot.species.env_synth import (
    DifficultyBand,
    EnvSynth,
    EnvSynthAction,
    ETDAgent,
    MCPToolEntry,
    MCPToolRegistry,
    SolvabilityGate,
    SuiteStagnation,
    SynthesizedTask,
    T1TaskEntry,
    TaskSynthesizer,
    VerifierBuilder,
    VerifierSpec,
    VerifierType,
    arena_to_t1,
    diagnose_stagnation,
    flag_human_review,
    render_arena_rollup,
)
from scripts.autopilot.species.env_synth.task_synthesizer import make_fake_llm


# ── VerifierBuilder ────────────────────────────────────────────────

def test_regex_verifier_accepts_exact_and_rejects_others():
    scorer = VerifierBuilder.build(VerifierSpec(
        type=VerifierType.REGEX, pattern=r"^\d{3}$",
    ))
    assert scorer("123") == 1.0
    assert scorer(" 123 ") == 1.0   # whitespace stripped
    assert scorer("12") == 0.0
    assert scorer("abc") == 0.0


def test_regex_rejects_accept_all_pattern():
    with pytest.raises(ValueError):
        VerifierBuilder.build(VerifierSpec(type=VerifierType.REGEX, pattern=".*"))


def test_exact_match_case_insensitive_by_default():
    scorer = VerifierBuilder.build(VerifierSpec(
        type=VerifierType.EXACT_MATCH, reference="Canberra",
    ))
    assert scorer("canberra") == 1.0
    assert scorer(" CANBERRA  ") == 1.0
    assert scorer("Sydney") == 0.0


def test_f1_verifier_partial_overlap_scores_between_zero_and_one():
    scorer = VerifierBuilder.build(VerifierSpec(
        type=VerifierType.F1,
        reference="northern lights are caused by solar particles",
    ))
    s = scorer("solar particles create northern lights")
    assert 0.3 < s < 1.0
    assert scorer("pasta recipes") == 0.0


def test_f1_verifier_rejects_short_output():
    scorer = VerifierBuilder.build(VerifierSpec(
        type=VerifierType.F1, reference="answer", min_tokens=3,
    ))
    assert scorer("answer") == 0.0


def test_exact_match_rejects_empty_reference():
    with pytest.raises(ValueError):
        VerifierBuilder.build(VerifierSpec(type=VerifierType.EXACT_MATCH, reference="   "))


# ── MCPToolRegistry ────────────────────────────────────────────────

def _mk_tool(tool_id: str, env: str = "env_x") -> MCPToolEntry:
    return MCPToolEntry(
        tool_id=tool_id,
        name=f"tool {tool_id}",
        description="test tool",
        endpoint=f"http://example.com/{tool_id}/mcp",
        discovered_via="etd_web",
        environment_id=env,
    )


def test_registry_persists_and_reloads(tmp_path):
    path = tmp_path / "reg.jsonl"
    reg = MCPToolRegistry(path)
    reg.register(_mk_tool("t1"))
    reg.register(_mk_tool("t2", env="env_y"))

    reg2 = MCPToolRegistry(path)
    assert {e.tool_id for e in reg2.all()} == {"t1", "t2"}
    assert len(reg2.by_environment("env_y")) == 1


def test_registry_deactivate_excludes_from_active(tmp_path):
    reg = MCPToolRegistry(tmp_path / "reg.jsonl")
    reg.register(_mk_tool("t1"))
    reg.deactivate("t1")
    assert reg.active() == []
    assert reg.get("t1") is not None  # still present, just inactive


def test_registry_health_check_deactivates_after_failures(tmp_path):
    reg = MCPToolRegistry(tmp_path / "reg.jsonl")
    reg.register(_mk_tool("t_bad"))

    async def always_fail(entry):
        return False

    for _ in range(3):
        asyncio.run(reg.run_health_checks(always_fail, deactivate_after=3))

    assert reg.active() == []
    entry = reg.get("t_bad")
    assert entry is not None
    assert entry.consecutive_failures >= 3


# ── TaskSynthesizer ────────────────────────────────────────────────

def test_task_synthesizer_produces_valid_task_from_fake_llm():
    llm = make_fake_llm(
        verifier_type=VerifierType.EXACT_MATCH, reference="42",
    )
    synth = TaskSynthesizer(llm=llm)
    tools = [_mk_tool(f"t{i}") for i in range(3)]

    task = asyncio.run(
        synth.synthesize("env_x", tools, DifficultyBand.MEDIUM)
    )
    assert task is not None
    assert task.environment_id == "env_x"
    assert task.difficulty_band == DifficultyBand.MEDIUM
    assert task.verifier.type == VerifierType.EXACT_MATCH
    assert task.verifier.reference == "42"

    # Verifier should score correctly.
    scorer = VerifierBuilder.build(task.verifier)
    assert scorer("42") == 1.0
    assert scorer("wrong") == 0.0


def test_task_synthesizer_returns_none_on_bad_json():
    async def bad_llm(sys, usr):
        return "definitely not JSON"

    synth = TaskSynthesizer(llm=bad_llm, max_retries=1)
    tools = [_mk_tool("t0")]
    task = asyncio.run(
        synth.synthesize("env_x", tools, DifficultyBand.EASY)
    )
    assert task is None


# ── ETDAgent ────────────────────────────────────────────────────────

def test_etd_agent_enumerates_tools_and_persists(tmp_path):
    reg = MCPToolRegistry(tmp_path / "reg.jsonl")

    async def llm(sys, usr):
        return json.dumps([
            {"name": "MathTools", "description": "math MCP", "search_queries": ["math tools api"]},
            {"name": "WebStats", "description": "web stats MCP", "search_queries": ["web stats mcp"]},
        ])

    async def web_search(query, n):
        return [
            {"url": "https://math.example.com/mcp"},
            {"url": "https://math.example.com/blog"},  # not MCP endpoint
            {"url": "https://webstats.example.com/jsonrpc"},
        ][:n]

    async def fetch_url(url):
        return "<html/>"

    async def tool_enum(endpoint):
        return [
            MCPToolEntry(
                tool_id=f"tool_{hash(endpoint) % 10000}",
                name="discovered",
                description="via etd",
                endpoint=endpoint,
                discovered_via="etd_web",
            )
        ]

    agent = ETDAgent(
        llm=llm, web_search=web_search, fetch_url=fetch_url,
        tool_enum=tool_enum, registry=reg,
    )
    discoveries = asyncio.run(agent.discover("math tasks"))
    assert len(discoveries) == 2
    assert all(d.tools for d in discoveries)
    # Registry should now contain the tools with environment ids set.
    envs = {e.environment_id for e in reg.all()}
    assert all(env.startswith("env_") for env in envs)


# ── SolvabilityGate + EnvSynth end-to-end ──────────────────────────

def _minimal_stack(tmp_path, reference_ok: bool = True):
    reg = MCPToolRegistry(tmp_path / "reg.jsonl")

    async def llm(sys, usr):
        return json.dumps([
            {"name": "MathEnv", "description": "math", "search_queries": ["math mcp"]},
        ])

    async def web_search(query, n):
        return [{"url": "https://math.example.com/mcp"}]

    async def fetch_url(url):
        return ""

    async def tool_enum(endpoint):
        return [
            MCPToolEntry(
                tool_id="t_calc",
                name="calculator",
                description="does arithmetic",
                endpoint=endpoint,
                discovered_via="etd_web",
            )
        ]

    etd = ETDAgent(
        llm=llm, web_search=web_search, fetch_url=fetch_url,
        tool_enum=tool_enum, registry=reg,
    )

    synth_llm = make_fake_llm(
        verifier_type=VerifierType.EXACT_MATCH, reference="42",
    )
    synth = TaskSynthesizer(llm=synth_llm)

    async def reference_solver(prompt, tool_set):
        return reference_ok, 0.8 if reference_ok else 0.1, "mock"

    gate = SolvabilityGate(reference_solver=reference_solver)

    return EnvSynth(
        etd_agent=etd,
        task_synthesizer=synth,
        registry=reg,
        solvability_gate=gate,
        arena_path=tmp_path / "arena.jsonl",
        journal_path=tmp_path / "journal.jsonl",
    )


def test_env_synth_full_pipeline_persists_accepted_tasks(tmp_path):
    es = _minimal_stack(tmp_path, reference_ok=True)
    tasks = asyncio.run(es.discover_and_synthesize(
        "math", band=DifficultyBand.MEDIUM, tasks_per_env=2,
    ))
    assert tasks, "expected at least one accepted task"
    assert (tmp_path / "arena.jsonl").exists()
    assert (tmp_path / "journal.jsonl").exists()
    # Journal entry should carry the action type.
    journal_lines = (tmp_path / "journal.jsonl").read_text().splitlines()
    assert journal_lines
    record = json.loads(journal_lines[0])
    assert "environment_id" in record
    assert "synthesized_tasks" in record


def test_env_synth_rejects_when_reference_fails(tmp_path):
    es = _minimal_stack(tmp_path, reference_ok=False)
    tasks = asyncio.run(es.discover_and_synthesize(
        "math", band=DifficultyBand.MEDIUM, tasks_per_env=2,
    ))
    assert tasks == []


def test_env_synth_propose_actions_shape():
    es = _minimal_stack(Path("/tmp/envsynth_test_dir"))
    action = es.propose_actions("math", band=DifficultyBand.HARD, gap_descriptor="more math")
    assert action["type"] == "env_synth_cycle"
    assert action["difficulty_band"] == "hard"
    assert action["gap_descriptor"] == "more math"


# ── Gap diagnosis (AW-3) ───────────────────────────────────────────

def test_diagnose_stagnation_flags_flat_suites(tmp_path):
    journal = tmp_path / "j.jsonl"
    with journal.open("w") as f:
        for i in range(15):
            f.write(json.dumps({"suite": "math", "quality": 1.20, "trial_id": i}) + "\n")
        for i in range(15):
            # clearly improving
            f.write(json.dumps({"suite": "coder", "quality": 1.0 + i * 0.05, "trial_id": i}) + "\n")

    findings = diagnose_stagnation(journal, window=10, stagnation_threshold=0.01)
    suites = {f.suite for f in findings}
    assert "math" in suites
    assert "coder" not in suites


def test_render_arena_rollup_empty_and_populated():
    assert "No stagnation" in render_arena_rollup([])
    rollup = render_arena_rollup([SuiteStagnation(
        suite="math", latest_quality=1.2, window_slope=0.0, window_size=10,
        gap_descriptor="need more",
    )])
    assert "math" in rollup
    assert "need more" in rollup


# ── AW-5: EvalTower T1 integration ─────────────────────────────────

def test_arena_to_t1_projects_records_and_preserves_provenance(tmp_path):
    arena = tmp_path / "arena.jsonl"
    with arena.open("w") as f:
        f.write(json.dumps({
            "task_id": "envsynth_abc",
            "environment_id": "env_math",
            "tool_set": ["t_calc"],
            "prompt": "compute 6 × 7",
            "difficulty_band": "medium",
            "verifier": {"type": "exact_match", "reference": "42"},
            "ground_truth_hint": "42",
            "expected_tool_calls": [1, 2],
            "metadata": {},
            "persisted_at": "2026-04-22T00:00:00Z",
        }) + "\n")

    entries = arena_to_t1(arena, only_bands={"medium", "hard"})
    assert len(entries) == 1
    e = entries[0]
    assert e.task_id == "envsynth_abc"
    assert e.provenance["discovered_via"] == "env_synth"
    assert e.provenance["difficulty_band"] == "medium"
    assert e.provenance["verifier_type"] == "exact_match"
    assert e.scoring_config["verifier"]["reference"] == "42"


def test_flag_human_review_marks_at_threshold():
    entries = [
        T1TaskEntry(task_id="a"),
        T1TaskEntry(task_id="b"),
        T1TaskEntry(task_id="c"),
    ]
    flag_human_review(entries, failures_by_task={"a": 3, "b": 2, "c": 4}, fail_threshold=3)
    marked = {e.task_id for e in entries if e.flagged_for_review}
    assert marked == {"a", "c"}
