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
    kf.refresh_roots()
    try:
        assert kf.deny_reason(str(secrets)) == "knowledge_root"
    finally:
        kf.refresh_roots()


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



# ── review fixes (Fable review of a8bdb15f) ──────────────────────────────


@pytest.fixture()
def llm_root(tmp_path: Path, monkeypatch) -> Path:
    """A hermetic llm root with checkouts the name rule cannot see."""
    root = tmp_path / "llm"
    for rel in ("worktrees/mains/mainA", "tmp/deploy"):
        wiki = root / rel / "wiki"
        wiki.mkdir(parents=True)
        (wiki / "INDEX.md").write_text("SECRET wiki content\n")
    (root / "worktrees" / "mains" / "mainA" / "src").mkdir()
    (root / "worktrees" / "mains" / "mainA" / "src" / "ok.py").write_text("x = 1\n")
    (root / "work").mkdir()
    (root / "work" / "notes.txt").write_text("plain\n")
    monkeypatch.setattr(kf, "_llm_root", lambda: str(root))
    kf.refresh_roots()
    yield root
    kf.refresh_roots()


@pytest.mark.parametrize(
    "path",
    ["/mnt/raid0/llm/worktrees/mains/mainA/wiki/INDEX.md", "/mnt/raid0/llm/tmp/deploy/wiki/INDEX.md"],
)
def test_live_worktree_checkouts_are_fenced(path: str) -> None:
    if not Path(path).parent.is_dir():
        pytest.skip("checkout not present on this host")
    kf.refresh_roots()
    assert kf.deny_reason(path) == "knowledge_root"


def test_marker_discovered_checkouts_are_fenced(llm_root: Path) -> None:
    for rel in ("worktrees/mains/mainA", "tmp/deploy"):
        assert kf.deny_reason(str(llm_root / rel / "wiki" / "INDEX.md")) == "knowledge_root"
    assert kf.deny_reason(str(llm_root / "worktrees/mains/mainA/src/ok.py")) is None


def test_git_worktree_list_discovers_checkouts_outside_the_glob(tmp_path: Path, monkeypatch) -> None:
    import subprocess

    llm = tmp_path / "llm"
    repo = llm / "epyc-root"
    repo.mkdir(parents=True)
    git = ["git", "-c", "user.email=t@t", "-c", "user.name=t"]
    subprocess.run([*git, "init", "-q", str(repo)], check=True)
    (repo / "README").write_text("r\n")
    subprocess.run([*git, "-C", str(repo), "add", "README"], check=True)
    subprocess.run([*git, "-C", str(repo), "commit", "-q", "-m", "init"], check=True)
    far = tmp_path / "far" / "a" / "b" / "c" / "d" / "wt"
    subprocess.run([*git, "-C", str(repo), "worktree", "add", "-q", str(far)], check=True)
    (far / "handoffs").mkdir()
    assert str(far) in kf.discover_checkouts(str(llm))
    monkeypatch.setattr(kf, "_llm_root", lambda: str(llm))
    kf.refresh_roots()
    try:
        assert kf.deny_reason(str(far / "handoffs" / "x.md")) == "knowledge_root"
    finally:
        kf.refresh_roots()


def test_walks_rooted_at_a_worktree_are_refused(llm_root: Path) -> None:
    reg = _registry()
    main_a = str(llm_root / "worktrees" / "mains" / "mainA")
    kf.begin(True)
    for cmd in (["grep", "-r", "SECRET", main_a], ["find", main_a], ["ls", "-R", main_a]):
        msg = kf.check_shell(cmd, "/tmp")
        assert msg and "walk_contains_fenced_root" in msg, cmd
    assert kf.check_shell(["grep", "-r", "x"], main_a) is not None
    found = reg.invoke("search_files", "frontdoor", directory=main_a, content="SECRET")
    assert found["success"] is False and "walk_contains_fenced_root" in found["error"]
    # A single file in the same checkout outside the knowledge folders stays readable.
    assert kf.check_shell(["cat", f"{main_a}/src/ok.py"], "/tmp") is None


@pytest.mark.skipif(sys.version_info >= (3, 13), reason="realpath no longer raises here")
def test_unresolvable_path_is_denied_not_crashed() -> None:
    reg = _registry()
    path = "/proc/1/root/etc/passwd"
    try:
        __import__("os").path.realpath(path)
        pytest.skip("realpath resolves /proc/1/root on this host")
    except OSError:
        pass
    carrier = kf.begin(True)
    result = reg.invoke("read_file", "frontdoor", path=path)
    assert result["success"] is False
    assert "unresolvable_path" in result["error"]
    assert carrier.snapshot()["denied_paths"] == [path]
    kf.begin(False)
    assert reg.invoke("read_file", "frontdoor", path=path)["success"] is False  # OS refusal
    kf.begin(None)
    assert kf.check_tool_call("read_file", {"path": path}) is None


def test_check_tool_call_fails_closed_when_armed(monkeypatch) -> None:
    def boom(*_a, **_k):
        raise RuntimeError("classification exploded")

    monkeypatch.setattr(kf, "fence_roots", boom)
    kf.begin(True)
    assert kf.check_tool_call("read_file", {"path": "/tmp/x"}).startswith(kf.DENY_PREFIX)
    kf.begin(False)
    assert kf.check_tool_call("read_file", {"path": "/tmp/x"}) is None


# run_python_code runtime fence


def _run_python(code: str) -> str:
    from src.repl_environment.external_access import _ExternalAccessMixin

    return _ExternalAccessMixin._run_python_code(_fake_shell_env(), code)


_BYPASSES = {
    "concat": "print(open('{wiki_dir}'[:-5] + '/wiki/INDEX.md').read())",
    "pathlib": "from pathlib import Path\nprint((Path('{wiki_dir}').parent / 'wiki' / 'INDEX.md').read_text())",
    "os_walk": "import os\nfor r, d, f in os.walk('{root}'):\n    for n in f:\n        p = os.path.join(r, n)\n        print(p, open(p).read())",
    "subprocess_cat": "import subprocess\nprint(subprocess.run(['cat', '{wiki_dir}/INDEX.md'], capture_output=True, text=True).stdout)",
    "shell_true": "import subprocess\nprint(subprocess.run('cat {wiki_dir}/INDEX.md', shell=True, capture_output=True, text=True).stdout)",
    "os_system": "import os\nos.system('cat {wiki_dir}/INDEX.md')",
    "glob": "import glob\nprint(glob.glob('{root}/**/INDEX.md', recursive=True))",
    "nested_python": "import subprocess\nprint(subprocess.run(['python3', '-c', 'print(open(\"{wiki_dir}/INDEX.md\").read())'], capture_output=True, text=True).stdout)",
}


@pytest.mark.parametrize("name", sorted(_BYPASSES))
def test_run_python_code_bypasses_are_fenced(llm_root: Path, name: str) -> None:
    wiki_dir = str(llm_root / "worktrees" / "mains" / "mainA" / "wiki")
    code = _BYPASSES[name].format(wiki_dir=wiki_dir, root=str(llm_root))
    carrier = kf.begin(True)
    out = _run_python(code)
    assert "SECRET" not in out, out
    assert name == "os_walk" or name == "glob" or kf.DENY_PREFIX in out, out
    snap = carrier.snapshot()
    assert snap["denied_count"] >= 1, snap
    if name not in ("nested_python", "glob", "os_walk"):
        assert f"{wiki_dir}/INDEX.md" in snap["touched_paths"]
    # The plain file under the same root is still readable by the walk.
    if name == "os_walk":
        assert "plain" in out


def test_run_python_code_control_arm_records_without_denying(llm_root: Path) -> None:
    wiki_file = str(llm_root / "worktrees" / "mains" / "mainA" / "wiki" / "INDEX.md")
    carrier = kf.begin(False)
    out = _run_python(f"print(open('{wiki_file}').read())")
    assert "SECRET wiki content" in out
    snap = carrier.snapshot()
    assert wiki_file in snap["touched_paths"]
    assert snap["denied_count"] == 0


def test_run_python_code_unfenced_is_unchanged(llm_root: Path, monkeypatch) -> None:
    import subprocess

    wiki_file = str(llm_root / "worktrees" / "mains" / "mainA" / "wiki" / "INDEX.md")
    code = f"import sys\nprint(open('{wiki_file}').read())\nprint(sys.argv[0].endswith('.py'), __name__)\nraise ValueError('boom')"
    kf.begin(None)
    assert kf.python_fence_launch("/tmp/x.py", "/tmp") is None
    seen: list = []
    real_run = subprocess.run

    def spy(cmd, **kwargs):
        seen.append((cmd, kwargs))
        return real_run(cmd, **kwargs)

    monkeypatch.setattr(subprocess, "run", spy)
    out = _run_python(code)
    assert len(seen) == 1
    cmd, kwargs = seen[0]
    assert cmd[0] == "python3" and len(cmd) == 2 and "env" not in kwargs
    assert "SECRET wiki content" in out and "True __main__" in out
    assert 'raise ValueError' in out and "ValueError: boom" in out


def test_fenced_child_keeps_script_semantics(llm_root: Path) -> None:
    code = (
        "from __future__ import annotations\n"
        "import sys\n"
        "def f() -> undefined_name: return 1\n"
        "print(__name__, sys.argv[0].endswith('.py'), f())\n"
        "raise ValueError('boom')\n"
    )
    kf.begin(True)
    out = _run_python(code)
    assert "__main__ True 1" in out
    assert "ValueError: boom" in out
    assert 'line 5' in out
    assert "<string>" not in out


# control arm and shell evasions


def test_control_arm_is_distinguishable_from_an_old_api() -> None:
    import eval_tower

    control = {"eval_fence": {"state": "unarmed", "touched_paths": [], "denied_count": 0}}
    assert eval_tower._eval_fence_from_response(control) == ("control", [], 0)
    assert eval_tower._eval_fence_from_response({}) == ("absent", None, 0)
    r = eval_tower.QuestionResult(
        question_id="q1", qid="stable-q1", suite="math", prompt="p", expected="4",
        answer="4", correct=True, elapsed_s=1.0, fence="control", touched_paths=[],
    )
    assert eval_tower._compact_question_result(r) == {
        "qid": "stable-q1",
        "question_id": "q1",
        "suite": "math",
        "partition": "core",
        "correct": True,
        "latency_ms": 1000,
        "tokens_generated": 0,
        "tools_used": 0,
        "answer_hash": eval_tower.normalized_answer_hash("4"),
        "fence": "control",
        "touched_paths": [],
    }
    summary = eval_tower._eval_fence_summary([r])
    assert summary["state"] == "control"
    assert summary["control"] == 1 and summary["active"] == 0 and summary["absent"] == 0


@pytest.mark.parametrize(
    "cmd",
    [
        ["awk", 'BEGIN { while ((getline line < "/mnt/raid0/llm/epyc-root/wiki/INDEX.md") > 0) print line }'],
        ["awk", 'BEGIN { system("cat x") }'],
        ["awk", 'BEGIN { "cat x" | getline y }'],
        ["sed", "r /mnt/raid0/llm/epyc-root/wiki/INDEX.md", "/etc/hostname"],
        ["sed", "-e", "1r x", "/etc/hostname"],
        ["sed", "s/a/b/w out", "/etc/hostname"],
        ["sed", "-f", "prog.sed", "/etc/hostname"],
        ["git", "log", "--format=%B"],
        ["git", "log", "--pretty=full"],
    ],
)
def test_shell_io_evasions_are_refused_when_armed(cmd) -> None:
    kf.begin(True)
    assert kf.check_shell(cmd, "/tmp") is not None
    kf.begin(False)
    assert kf.check_shell(cmd, "/tmp") is None


@pytest.mark.parametrize(
    "cmd",
    [
        ["awk", "{print $1}", "/etc/hostname"],
        ["awk", "$1 > 3 || $2 == 1 {print}", "/etc/hostname"],
        ["sed", "s/a/b/g", "/etc/hostname"],
        ["sed", "-n", "/re/p", "/etc/hostname"],
        ["git", "log", "--oneline"],
        ["git", "status"],
    ],
)
def test_ordinary_shell_programs_still_run_when_armed(cmd) -> None:
    kf.begin(True)
    assert kf.check_shell(cmd, "/tmp") is None
