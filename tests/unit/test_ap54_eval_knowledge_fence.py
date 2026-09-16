"""AP-54 eval knowledge fence: per-request fence, touched-path recording, client flag.

Offline and inference-free. Covers:
  * a fenced request denies wiki / eval-gold reads through the registry, REPL,
    shell and python paths, as a tool error that does not raise;
  * an unfenced request is byte-identical to today (golden comparison);
  * the ordering hazard: an API without the field ignores it, and a response to
    a request without it carries no new key;
  * touched paths are recorded (bounded, paths only) and carried into question
    results, the trial summary and the journal measurement tuple;
  * the EvalTower client always sends the flag.

Run: .venv/bin/python -m pytest tests/unit/test_ap54_eval_knowledge_fence.py -q
"""
from __future__ import annotations

import asyncio
import contextvars
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT / "scripts" / "autopilot", REPO_ROOT / "scripts" / "benchmark", REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from src.builtin_tools import register_builtin_tools  # noqa: E402
from src.registry.tool_registry import ToolPermissions, ToolRegistry  # noqa: E402
from src.repl_environment import knowledge_fence as kf  # noqa: E402
from src.tool_registry import ToolCategory  # noqa: E402


@pytest.fixture(autouse=True)
def _no_leaked_carrier():
    kf.clear()
    yield
    kf.clear()


@pytest.fixture()
def tree(tmp_path: Path) -> Path:
    """A miniature llm root: one epyc-root checkout with a wiki, plus a plain file."""
    root = tmp_path / "llm"
    wiki = root / "epyc-root" / "wiki"
    wiki.mkdir(parents=True)
    (wiki / "INDEX.md").write_text("SECRET wiki content\n")
    gold = root / "epyc-inference-research" / "benchmarks" / "prompts"
    gold.mkdir(parents=True)
    (gold / "question_pool.jsonl").write_text('{"expected": "SECRET"}\n')
    plain = root / "work"
    plain.mkdir()
    (plain / "notes.txt").write_text("SECRET plain content\n")
    return root


def _registry() -> ToolRegistry:
    reg = ToolRegistry()
    register_builtin_tools(reg)
    reg.set_role_permissions(
        "frontdoor", ToolPermissions(allowed_categories=[ToolCategory.FILE])
    )
    return reg


# ── classification ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "path, reason",
    [
        ("/workspace/wiki/INDEX.md", "knowledge_root"),
        ("/mnt/raid0/llm/epyc-root/wiki/INDEX.md", "knowledge_root"),
        ("/mnt/raid0/llm/epyc-root.wt/handoffs/active/x.md", "knowledge_root"),
        ("/mnt/raid0/llm/root-archetype/wiki", "knowledge_root"),
        ("/workspace/docs/chapters/01.md", "knowledge_root"),
        ("/mnt/raid0/llm/epyc-inference-research/benchmarks/prompts/question_pool.jsonl", "eval_gold"),
        ("/mnt/raid0/llm/worktrees/x/benchmarks/results/eval/a.jsonl", "eval_gold"),
        ("/mnt/raid0/llm/epyc-orchestrator/scripts/autopilot/sentinel_questions.yaml", "eval_gold"),
        ("/mnt/raid0/llm/tmp/physreason/PhysReason_full/cal_problem_00292/problem.json", "eval_gold"),
        ("/mnt/raid0/llm/epyc-orchestrator/data/kb_rag/index-qd-v1/x", "knowledge_root"),
    ],
)
def test_deny_reason_fences_knowledge_and_gold(path: str, reason: str) -> None:
    assert kf.deny_reason(path) == reason


@pytest.mark.parametrize(
    "path",
    [
        "/mnt/raid0/llm/epyc-orchestrator/src/api/routes/chat.py",
        "/mnt/raid0/llm/epyc-orchestrator/benchmarks/images/vl/chartqa/chart_test_0758.png",
        "/mnt/raid0/llm/tmp/physreason/PhysReason_full/cal_problem_00292/images/a.jpg",
        "/tmp/scratch/wiki_notes.txt",
        "/mnt/raid0/llm/some-project/wiki/page.md",
    ],
)
def test_deny_reason_leaves_ordinary_paths_alone(path: str) -> None:
    assert kf.deny_reason(path) is None


def test_eval_secrets_file_is_fenced(monkeypatch, tmp_path: Path) -> None:
    secrets = tmp_path / "secrets.json"
    monkeypatch.setenv("EVAL_SECRETS_PATH", str(secrets))
    kf._explicit_roots_cached.cache_clear()
    try:
        assert kf.deny_reason(str(secrets)) == "knowledge_root"
    finally:
        kf._explicit_roots_cached.cache_clear()


# ── registry path: builtin file tools ────────────────────────────────────


def test_fenced_request_denies_wiki_read_without_raising(tree: Path) -> None:
    reg = _registry()
    wiki_file = str(tree / "epyc-root" / "wiki" / "INDEX.md")
    carrier = kf.begin(True)
    result = reg.invoke("read_file", "frontdoor", path=wiki_file)
    assert result["success"] is False
    assert result["error"].startswith(kf.DENY_PREFIX)
    assert "SECRET" not in json.dumps(result)
    snap = carrier.snapshot()
    assert snap["state"] == "active"
    assert snap["touched_paths"] == [wiki_file]
    assert snap["denied_paths"] == [wiki_file]
    assert snap["denied_count"] == 1
    # The denial is an ordinary failed invocation in the registry log.
    assert reg.get_invocation_log()[-1].success is False


def test_fenced_request_still_reads_ordinary_files(tree: Path) -> None:
    reg = _registry()
    plain = str(tree / "work" / "notes.txt")
    carrier = kf.begin(True)
    result = reg.invoke("read_file", "frontdoor", path=plain)
    assert result["success"] is True
    assert result["content"] == "SECRET plain content\n"
    assert carrier.snapshot()["touched_paths"] == [plain]
    assert carrier.snapshot()["denied_count"] == 0


def test_unfenced_request_is_byte_identical_to_legacy(tree: Path) -> None:
    """Golden: with no carrier, every builtin file tool returns exactly what the
    bare handler returns (the pre-fence behaviour), wiki included."""
    reg = _registry()
    wiki_file = str(tree / "epyc-root" / "wiki" / "INDEX.md")
    calls = [
        ("read_file", {"path": wiki_file}),
        ("list_files", {"path": str(tree / "epyc-root" / "wiki")}),
        ("list_files", {"path": str(tree), "pattern": "**/*"}),
        ("search_files", {"directory": str(tree), "content": "SECRET"}),
    ]
    kf.begin(None)
    assert kf.current() is None
    for name, kwargs in calls:
        legacy = reg._tools[name].handler(**kwargs)
        via_registry = reg.invoke(name, "frontdoor", **kwargs)
        assert json.dumps(via_registry, sort_keys=True) == json.dumps(legacy, sort_keys=True)
    # The wiki really is reachable when unfenced (that is the AP-54 finding).
    assert reg.invoke("read_file", "frontdoor", path=wiki_file)["content"].startswith("SECRET wiki")


def test_unarmed_eval_request_records_but_does_not_deny(tree: Path) -> None:
    reg = _registry()
    wiki_file = str(tree / "epyc-root" / "wiki" / "INDEX.md")
    carrier = kf.begin(False)
    result = reg.invoke("read_file", "frontdoor", path=wiki_file)
    assert result["success"] is True
    snap = carrier.snapshot()
    assert snap["state"] == "unarmed"
    assert snap["touched_paths"] == [wiki_file]
    assert snap["denied_count"] == 0


def test_fenced_walks_skip_fenced_files(tree: Path) -> None:
    reg = _registry()
    kf.begin(True)
    found = reg.invoke("search_files", "frontdoor", directory=str(tree), content="SECRET")
    paths = [m["path"] for m in found["matches"]]
    assert paths == [str(tree / "work" / "notes.txt")]
    listed = reg.invoke("list_files", "frontdoor", path=str(tree), pattern="**/*")
    assert not any("/wiki" in p or "benchmarks/prompts" in p for p in listed["files"])
    assert str(tree / "work" / "notes.txt") in listed["files"]
    assert kf.current().snapshot()["denied_count"] >= 2


def test_fenced_walk_rooted_above_a_concrete_fenced_dir_is_refused() -> None:
    kf.begin(True)
    msg = kf.check_tool_call("search_files", {"directory": "/", "content": "x"})
    assert msg and msg.startswith(kf.DENY_PREFIX) and "walk_contains_fenced_root" in msg


def test_doc_search_refused_only_when_armed() -> None:
    assert kf.check_tool_call("doc_search", {}) is None
    kf.begin(False)
    assert kf.check_tool_call("doc_search", {}) is None
    kf.begin(True)
    assert kf.check_tool_call("doc_search", {}).startswith(kf.DENY_PREFIX)


def test_touched_paths_are_bounded_and_paths_only() -> None:
    carrier = kf.begin(False)
    for i in range(kf.MAX_TOUCHED_PATHS + 10):
        kf.check_path(f"/tmp/x/{i}/" + ("a" * 400))
    snap = carrier.snapshot()
    assert len(snap["touched_paths"]) == kf.MAX_TOUCHED_PATHS
    assert snap["touched_paths_overflow"] == 10
    assert all(len(p) <= kf.MAX_PATH_CHARS for p in snap["touched_paths"])


# ── REPL path ────────────────────────────────────────────────────────────


def _fake_repl(prefix: str) -> SimpleNamespace:
    return SimpleNamespace(ALLOWED_FILE_PATHS=[prefix])


def test_repl_validate_file_path_denies_under_fence(tree: Path) -> None:
    from src.repl_environment.environment import REPLEnvironment

    fake = _fake_repl(str(tree) + "/")
    wiki_file = str(tree / "epyc-root" / "wiki" / "INDEX.md")
    plain = str(tree / "work" / "notes.txt")
    # Unfenced: unchanged.
    assert REPLEnvironment._validate_file_path(fake, wiki_file) == (True, None)
    kf.begin(True)
    ok, err = REPLEnvironment._validate_file_path(fake, wiki_file)
    assert ok is False and err.startswith(kf.DENY_PREFIX)
    assert REPLEnvironment._validate_file_path(fake, plain) == (True, None)
    # Paths outside the allow-list keep their original error, untouched by the fence.
    ok, err = REPLEnvironment._validate_file_path(fake, "/etc/passwd")
    assert ok is False and err.startswith("Path not in allowed locations")


def _fake_shell_env() -> SimpleNamespace:
    return SimpleNamespace(_exploration_calls=0, _exploration_log=MagicMock())


def test_run_shell_denies_fenced_paths_and_python(tree: Path) -> None:
    from src.repl_environment.external_access import _ExternalAccessMixin

    env = _fake_shell_env()
    wiki_file = str(tree / "epyc-root" / "wiki" / "INDEX.md")
    kf.begin(True)
    with patch("subprocess.run") as run:
        out = _ExternalAccessMixin._run_shell(env, f"cat {wiki_file}")
        assert out.startswith(f"[ERROR: {kf.DENY_PREFIX}")
        out = _ExternalAccessMixin._run_shell(env, "python3 -c 'print(1)'")
        assert out.startswith(f"[ERROR: {kf.DENY_PREFIX}")
        out = _ExternalAccessMixin._run_shell(env, "git show HEAD:README.md")
        assert out.startswith(f"[ERROR: {kf.DENY_PREFIX}")
        run.assert_not_called()


def test_run_shell_unfenced_runs_command(tree: Path) -> None:
    from src.repl_environment.external_access import _ExternalAccessMixin

    env = _fake_shell_env()
    wiki_file = str(tree / "epyc-root" / "wiki" / "INDEX.md")
    out = _ExternalAccessMixin._run_shell(env, f"cat {wiki_file}")
    assert out == "SECRET wiki content\n"


def test_check_shell_recursion_and_touched_paths(tree: Path) -> None:
    carrier = kf.begin(True)
    # grep -r with no path walks the orchestrator checkout, which holds benchmarks/prompts.
    assert kf.check_shell(["grep", "-r", "x"], str(REPO_ROOT)) is not None
    assert kf.check_shell(["find", "/"], "/tmp") is not None
    plain = str(tree / "work" / "notes.txt")
    assert kf.check_shell(["cat", plain], "/tmp") is None
    # awk/sed programs are not recorded as paths.
    assert kf.check_shell(["sed", "s/a/b/", plain], "/tmp") is None
    touched = carrier.snapshot()["touched_paths"]
    assert plain in touched
    assert "s/a/b/" not in touched


def test_run_python_code_refuses_literal_fenced_reference() -> None:
    kf.begin(True)
    assert kf.check_python_source("open('/mnt/raid0/llm/epyc-root/wiki/INDEX.md')")
    assert kf.check_python_source("url = 'https://en.wikipedia.org/wiki/Python'") is None
    kf.begin(None)
    assert kf.check_python_source("open('/mnt/raid0/llm/epyc-root/wiki/INDEX.md')") is None


# ── context propagation across thread hops ──────────────────────────────


def test_with_timeout_worker_thread_sees_the_carrier() -> None:
    import threading

    from src.tools.base import with_timeout

    @with_timeout(5)
    def probe() -> bool:
        return kf.armed()

    kf.begin(True)
    seen: list[bool] = []
    ctx = contextvars.copy_context()
    worker = threading.Thread(target=lambda: seen.append(ctx.run(probe)))
    worker.start()
    worker.join()
    assert seen == [True]


def test_parallel_dispatch_workers_see_the_carrier() -> None:
    from src.repl_environment.parallel_dispatch import _ParallelCall, execute_parallel_calls

    kf.begin(True)
    calls = [
        _ParallelCall(func_name="probe", args=(), kwargs={}, target_var=f"v{i}", index=i)
        for i in range(3)
    ]
    out = execute_parallel_calls(calls, {"probe": kf.armed}, state_lock=None)
    assert out == {"v0": True, "v1": True, "v2": True}


# ── API models and route: ordering hazard, byte-identical responses ─────


def test_chat_request_accepts_flag_and_tolerates_unknown_fields() -> None:
    from src.api.models import ChatRequest

    assert ChatRequest(prompt="q").eval_fence is None
    assert ChatRequest(prompt="q", eval_fence=True).eval_fence is True
    # An API build that lacks a field ignores it: extra is not 'forbid'. The
    # live client already relies on this for `scoring_method`.
    assert ChatRequest.model_config.get("extra") in (None, "ignore")
    req = ChatRequest.model_validate({"prompt": "q", "some_future_field": 1})
    assert not hasattr(req, "some_future_field")


def test_pre_fence_request_model_ignores_the_new_field() -> None:
    """The deployed API's ChatRequest has no `eval_fence`; simulate it exactly."""
    from src.api.models import ChatRequest

    fields = {k: (v.annotation, v) for k, v in ChatRequest.model_fields.items() if k != "eval_fence"}
    from pydantic import create_model

    OldChatRequest = create_model("OldChatRequest", **fields)
    old = OldChatRequest.model_validate({"prompt": "q", "eval_fence": True})
    assert "eval_fence" not in old.model_dump()


def test_chat_response_without_fence_is_byte_identical() -> None:
    from src.api.models import ChatResponse

    resp = ChatResponse(answer="ok", turns=1, elapsed_seconds=0.1, mock_mode=True)
    assert "eval_fence" not in resp.model_dump()
    assert "eval_fence" not in json.loads(resp.model_dump_json())
    fenced = resp.model_copy(update={"eval_fence": {"state": "active", "touched_paths": []}})
    assert fenced.model_dump()["eval_fence"] == {"state": "active", "touched_paths": []}


class _FakeHttpRequest:
    async def is_disconnected(self) -> bool:
        return False


@pytest.mark.asyncio
@pytest.mark.parametrize("flag, expected_state", [(True, "active"), (False, "unarmed"), (None, None)])
async def test_chat_route_arms_echoes_and_clears(tree: Path, flag, expected_state) -> None:
    from src.api.models import ChatRequest, ChatResponse
    from src.api.routes.chat import chat

    reg = _registry()
    wiki_file = str(tree / "epyc-root" / "wiki" / "INDEX.md")
    seen: dict = {}

    async def fake_handle_chat(*_args, **_kwargs):
        # Tools run in a worker thread in the real pipeline.
        seen["result"] = await asyncio.to_thread(
            reg.invoke, "read_file", "frontdoor", path=wiki_file
        )
        return ChatResponse(answer="ok", turns=1, elapsed_seconds=0.01, mock_mode=True)

    request = ChatRequest(prompt="t", eval_fence=flag)
    with patch("src.api.routes.chat._handle_chat", new=fake_handle_chat):
        response = await chat(request, _FakeHttpRequest(), MagicMock())

    assert kf.current() is None  # never outlives the request
    if expected_state is None:
        assert response.eval_fence is None
        assert "eval_fence" not in response.model_dump()
        assert seen["result"]["success"] is True
        return
    assert response.eval_fence["state"] == expected_state
    assert response.eval_fence["touched_paths"] == [wiki_file]
    assert seen["result"]["success"] is (flag is False)


# ── client: always sends the flag, records fence + touched paths ────────


class _Resp:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        return None


def test_call_orchestrator_forced_forwards_flag_only_when_given() -> None:
    import seeding_orchestrator

    client = Mock()
    client.post.return_value = _Resp(200, {"answer": "4"})
    seeding_orchestrator.call_orchestrator_forced(prompt="q", force_role="worker", client=client)
    assert "eval_fence" not in client.post.call_args.kwargs["json"]
    seeding_orchestrator.call_orchestrator_forced(
        prompt="q", force_role="worker", client=client, eval_fence=True
    )
    assert client.post.call_args.kwargs["json"]["eval_fence"] is True


def _question() -> dict:
    return {
        "id": "math-fence",
        "suite": "math",
        "prompt": "What is 2+2?",
        "expected": "4",
        "scoring_method": "exact_match",
    }


@pytest.mark.parametrize(
    "env, expected_flag", [(None, True), ("1", True), ("0", False), ("off", False)]
)
def test_eval_tower_always_sends_the_flag(monkeypatch, env, expected_flag) -> None:
    import eval_tower

    if env is None:
        monkeypatch.delenv("AUTOPILOT_EVAL_FENCE", raising=False)
    else:
        monkeypatch.setenv("AUTOPILOT_EVAL_FENCE", env)
    sent: list[dict] = []

    def _fake_call(**kwargs):
        sent.append(kwargs)
        return {"answer": "4", "tokens_generated": 3, "routed_to": "worker_math"}

    monkeypatch.setattr(eval_tower, "call_orchestrator_forced", _fake_call)
    with eval_tower.httpx.Client(timeout=1) as client:
        eval_tower.EvalTower()._eval_question(_question(), client)
    assert sent and all(call["eval_fence"] is expected_flag for call in sent)


def test_eval_tower_records_fence_and_touched_paths(monkeypatch) -> None:
    import eval_tower

    responses = iter(
        [
            {
                "answer": "4",
                "tokens_generated": 3,
                "routed_to": "worker_math",
                "eval_fence": {
                    "state": "active",
                    "touched_paths": ["/mnt/raid0/llm/epyc-root/wiki/INDEX.md"],
                    "denied_paths": ["/mnt/raid0/llm/epyc-root/wiki/INDEX.md"],
                    "denied_count": 1,
                },
            },
            # A pre-fence API: no echo at all.
            {"answer": "4", "tokens_generated": 3, "routed_to": "worker_math"},
        ]
    )
    monkeypatch.setattr(eval_tower, "call_orchestrator_forced", lambda **_k: next(responses))
    tower = eval_tower.EvalTower()
    with eval_tower.httpx.Client(timeout=1) as client:
        fenced = tower._eval_question(_question(), client)
        legacy = tower._eval_question(_question(), client)

    assert fenced.fence == "active"
    assert fenced.touched_paths == ["/mnt/raid0/llm/epyc-root/wiki/INDEX.md"]
    assert fenced.fence_denied_count == 1
    assert legacy.fence == "absent"
    assert legacy.touched_paths is None

    row = eval_tower._compact_question_result(fenced)
    assert row["fence"] == "active"
    assert row["touched_paths"] == ["/mnt/raid0/llm/epyc-root/wiki/INDEX.md"]
    assert row["fence_denied_count"] == 1
    legacy_row = eval_tower._compact_question_result(legacy)
    assert legacy_row["fence"] == "absent"
    assert "touched_paths" not in legacy_row

    summary = eval_tower._eval_fence_summary([fenced, legacy])
    assert summary["state"] == "mixed"
    assert summary["active"] == 1 and summary["absent"] == 1
    assert summary["rows_with_touched_paths"] == 1
    assert summary["denied_count"] == 1
    assert eval_tower._eval_fence_summary([fenced])["state"] == "active"

    aggregated = tower._aggregate([fenced, legacy], tier=1)
    assert aggregated.details["eval_fence"]["state"] == "mixed"


def test_compact_row_bounds_touched_paths() -> None:
    import eval_tower

    r = eval_tower.QuestionResult(
        question_id="q",
        suite="math",
        prompt="p",
        expected="e",
        fence="active",
        touched_paths=[f"/p/{i}" + "x" * 400 for i in range(100)],
    )
    row = eval_tower._compact_question_result(r)
    assert len(row["touched_paths"]) == eval_tower.MAX_RECORDED_TOUCHED_PATHS
    assert all(len(p) <= 256 for p in row["touched_paths"])


def test_compact_row_tolerates_duck_typed_legacy_rows() -> None:
    import eval_tower

    r = eval_tower.QuestionResult(question_id="q", suite="math", prompt="p", expected="e")
    legacy = SimpleNamespace(**{k: v for k, v in vars(r).items() if k not in {"fence", "touched_paths", "fence_denied_count"}})
    row = eval_tower._compact_question_result(legacy)
    assert row["fence"] == "absent"
    assert "touched_paths" not in row


def test_measurement_tuple_carries_fence_state() -> None:
    import experiment_journal as ej

    entry = ej.JournalEntry(
        trial_id=1,
        timestamp="2026-09-16T00:00:00+00:00",
        species="s",
        action_type="a",
        tier=1,
        quality=0.5,
        speed=1.0,
        cost=0.1,
        reliability=1.0,
        pareto_status="",
        eval_details={"details": {"n_scored": 3, "eval_fence": {"state": "active"}}},
    )
    assert ej.measurement_tuple(entry)["eval_fence"] == "active"
    entry.eval_details = {"details": {"n_scored": 3}}
    assert "eval_fence" not in ej.measurement_tuple(entry)


@pytest.mark.asyncio
async def test_unfenced_eval_request_is_logged_not_silent(caplog) -> None:
    from src.api.models import ChatRequest, ChatResponse
    from src.api.routes import chat as chat_mod

    async def fake_handle_chat(*_args, **_kwargs):
        return ChatResponse(answer="ok", turns=1, elapsed_seconds=0.01, mock_mode=True)

    chat_mod._UNFENCED_EVAL_REQUESTS = 0
    request = ChatRequest(prompt="t", workload_class="eval_batch")
    with patch("src.api.routes.chat._handle_chat", new=fake_handle_chat):
        with caplog.at_level("WARNING", logger="src.api.routes.chat"):
            response = await chat_mod.chat(request, _FakeHttpRequest(), MagicMock())
    assert response.eval_fence is None
    assert any("UNFENCED" in rec.getMessage() for rec in caplog.records)
