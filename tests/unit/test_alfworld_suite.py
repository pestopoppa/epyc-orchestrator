"""AP-ALFWORLD-SUITE: ALFWorld as a small rotating EvalTower slice (default off). Fakes only.

No real ALFWorld environment, no model, no server: the "env subprocess" is the REAL driver script run under the test
interpreter with a fake env adapter module planted in a fake AgentBench tree, and the "model" is a loopback HTTP fake.
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "autopilot"))

suite = importlib.import_module("alfworld_suite")
drv = importlib.import_module("alfworld_episode_driver")
eval_tower = importlib.import_module("eval_tower")
rigor = importlib.import_module("bsv2_accept_rigor")
actions = importlib.import_module("actions")

KINDS = ["pick_and_place", "pick_clean_then_place", "pick_heat_then_place", "pick_cool_then_place",
         "look_at_obj", "pick_two_obj"]
STALL_KIND = "pick_two_obj"  # the fake env never finishes these, so a model repeating one action trips the repeat guard
ENV_NAMES = ("AUTOPILOT_ALFWORLD_SUITE", "AUTOPILOT_ALFWORLD_N", "AUTOPILOT_ALFWORLD_MAX_STEPS",
             "AUTOPILOT_ALFWORLD_EPISODE_WALL_S", "AUTOPILOT_ALFWORLD_BATCH_WALL_S", "AUTOPILOT_ALFWORLD_LH_ROOT",
             "AUTOPILOT_ALFWORLD_PYTHON", "AUTOPILOT_ALFWORLD_DRIVER", "AUTOPILOT_ALFWORLD_ENV_ADAPTER",
             "AUTOPILOT_ALFWORLD_ENDPOINT", "AUTOPILOT_ALFWORLD_MODEL", "AUTOPILOT_ALFWORLD_ROLE",
             "AUTOPILOT_ALFWORLD_WORK_DIR", "AUTOPILOT_ALFWORLD_REQUEST_TIMEOUT_S", "AUTOPILOT_ALFWORLD_MAX_TOKENS",
             "AUTOPILOT_T1_CORE_ID", "AUTOPILOT_T1_CORE_PATH", "AUTOPILOT_W6_AUDIT_BLOCK")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


FAKE_ADAPTER = '''
class _Ep:
    def __init__(self, game):
        self.game, self.t = game, 0

    def reset(self):
        return "You are in a room. Your task is to: put a mug in the shelf.", ["go to shelf 1", "take mug 1", "look"]

    def step(self, action):
        self.t += 1
        won = action == "take mug 1" and "pick_two_obj" not in self.game
        return "ok " + action, won, won, ["go to shelf 1", "take mug 1", "look"]

    def close(self):
        pass


class Adapter:
    def __init__(self, root):
        self.root = root

    def task_instruction(self):
        return "FAKE INSTRUCTION"

    def process_action(self, action, choices):
        return action

    def open(self, game):
        if "BOOM" in game:
            raise RuntimeError("env boom")
        return _Ep(game)


def make(root):
    return Adapter(root)
'''


def make_root(tmp_path: Path, per_kind: int = 4) -> Path:
    """A fake Life-Harness checkout: AgentBench/data/alfworld split files + the fake env adapter module."""
    lh = tmp_path / "lh"
    ab = lh / "AgentBench"
    data = ab / "data" / "alfworld"
    data.mkdir(parents=True)
    train = {k: [f"json_2.1.1/train/{k}-{i}/trial_{i}/game.tw-pddl" for i in range(per_kind)] for k in KINDS}
    test = {k: [f"json_2.1.1/valid_unseen/{k}-{i}/trial_{i}/game.tw-pddl" for i in range(2)] for k in KINDS}
    (data / "train_valid.json").write_text(json.dumps(train))
    (data / "new_std.json").write_text(json.dumps(test))
    (ab / "fake_alf_env.py").write_text(FAKE_ADAPTER)
    return lh


class FakeModel:
    """Loopback OpenAI-compatible fake: records every request body, answers a fixed tool call."""

    def __init__(self, action: str = "take mug 1", status: int = 200, delay_s: float = 0.0):
        self.bodies: list[dict] = []
        self.action, self.status, self.delay_s = action, status, delay_s
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.bodies.append({"path": self.path, **body})
                time.sleep(outer.delay_s)
                payload = {"choices": [{"message": {"content": "", "tool_calls": [{"id": "c1", "type": "function",
                           "function": {"name": "take_action", "arguments": json.dumps({"action": outer.action})}}]}}],
                           "usage": {"prompt_tokens": 20, "completion_tokens": 7}}
                data = json.dumps(payload).encode()
                self.send_response(outer.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def env_for(monkeypatch, lh: Path, model: FakeModel | None = None, **extra) -> None:
    monkeypatch.setenv("AUTOPILOT_ALFWORLD_LH_ROOT", str(lh))
    monkeypatch.setenv("AUTOPILOT_ALFWORLD_PYTHON", sys.executable)
    monkeypatch.setenv("AUTOPILOT_ALFWORLD_ENV_ADAPTER", "fake_alf_env:make")
    monkeypatch.setenv("AUTOPILOT_ALFWORLD_WORK_DIR", str(lh.parent))
    if model is not None:
        monkeypatch.setenv("AUTOPILOT_ALFWORLD_ENDPOINT", model.base)
    for k, v in extra.items():
        monkeypatch.setenv(k, str(v))


# ---------------------------------------------------------------- draw / identity
def test_flag_default_off_and_registered() -> None:
    from src.features import _FEATURE_REGISTRY, Features

    spec = next(s for s in _FEATURE_REGISTRY if s.name == "autopilot_alfworld_suite")
    assert spec.default_test is False and spec.default_prod is False and spec.env_name == "AUTOPILOT_ALFWORLD_SUITE"
    assert Features().autopilot_alfworld_suite is False
    assert suite.flag_on() is False


def test_flag_on_reads_the_env(monkeypatch) -> None:
    monkeypatch.setenv("AUTOPILOT_ALFWORLD_SUITE", "1")
    assert suite.flag_on() is True


def test_draw_is_train_only_stratified_and_seeded(tmp_path) -> None:
    cfg = suite.SuiteConfig(lh_root=str(make_root(tmp_path)))
    games, sha = suite.load_split(cfg, suite.TRAIN_SPLIT)
    assert len(games) == 24 and len(sha) == 64
    a = suite.draw_slice(games, 5, seed=11)
    assert [g.qid for g in a] == [g.qid for g in suite.draw_slice(games, 5, seed=11)]
    assert len({g.kind for g in a}) == 5  # five episodes, five different task types
    assert all(g.split == "train_valid" and g.qid == f"alfworld:train_valid:{g.index}" for g in a)
    assert [g.qid for g in a] != [g.qid for g in suite.draw_slice(games, 5, seed=12)]


def test_draw_cycles_kinds_when_n_exceeds_kind_count(tmp_path) -> None:
    games, _ = suite.load_split(suite.SuiteConfig(lh_root=str(make_root(tmp_path))), suite.TRAIN_SPLIT)
    picked = suite.draw_slice(games, 8, seed=3)
    assert len(picked) == 8 and len({g.qid for g in picked}) == 8


def test_draw_skips_excluded_qids_and_can_run_dry(tmp_path) -> None:
    games, _ = suite.load_split(suite.SuiteConfig(lh_root=str(make_root(tmp_path))), suite.TRAIN_SPLIT)
    first = suite.draw_slice(games, 5, seed=11)
    second = suite.draw_slice(games, 5, seed=11, exclude_qids={g.qid for g in first})
    assert not {g.qid for g in first} & {g.qid for g in second}
    assert suite.draw_slice(games, 5, seed=11, exclude_qids={g.qid for g in games}) == []


def test_the_sealed_test_split_can_never_be_drawn(tmp_path) -> None:
    cfg = suite.SuiteConfig(lh_root=str(make_root(tmp_path)))
    test_games, _ = suite.load_split(cfg, suite.TEST_SPLIT)
    with pytest.raises(ValueError, match="R3"):
        suite.draw_slice(test_games, 5, seed=1)
    train, _ = suite.load_split(cfg, suite.TRAIN_SPLIT)
    with pytest.raises(ValueError, match="R3"):
        suite.draw_slice(train + test_games, 5, seed=1)
    sealed = suite.sealed_qids(cfg)
    assert len(sealed) == 12 and all(q.startswith("alfworld:new_std:") for q in sealed)
    assert suite.split_of(sealed[0]) == "new_std"


def test_rotation_changes_the_slice_and_suffix(tmp_path) -> None:
    cfg = suite.SuiteConfig(lh_root=str(make_root(tmp_path)))
    games, sha = suite.load_split(cfg, suite.TRAIN_SPLIT)
    a = suite.draw_slice(games, 5, suite.slice_seed(42, 0))
    b = suite.draw_slice(games, 5, suite.slice_seed(42, 1))
    assert [g.qid for g in a] != [g.qid for g in b]
    assert suite.core_id_suffix(a, sha) != suite.core_id_suffix(b, sha)
    assert suite.core_id_suffix(a, sha) == suite.core_id_suffix(list(a), sha)
    assert suite.core_id_suffix(a, sha) != suite.core_id_suffix(a, "other-data-sha")


def test_config_defaults_and_overrides(monkeypatch) -> None:
    cfg = suite.config_from_env()
    assert (cfg.n, cfg.max_steps, cfg.episode_wall_s) == (5, 30, 600.0)
    assert cfg.batch_budget_s == 5 * 600.0 + 120
    assert cfg.venv_python.endswith("AgentBench/.native/alfworld-venv/bin/python")
    monkeypatch.setenv("AUTOPILOT_ALFWORLD_N", "3")
    monkeypatch.setenv("AUTOPILOT_ALFWORLD_BATCH_WALL_S", "90")
    monkeypatch.setenv("AUTOPILOT_ALFWORLD_ROLE", "worker_general")
    cfg = suite.config_from_env()
    assert cfg.n == 3 and cfg.batch_budget_s == 90 and cfg.force_role == "worker_general"


# ---------------------------------------------------------------- driver episode loop (in process, fakes)
class FakeEpisode:
    def __init__(self, wins_on=None, done_on=None, boom_step=False):
        self.wins_on, self.done_on, self.boom_step, self.closed = wins_on, done_on, boom_step, False

    def reset(self):
        return "room. Your task is to: x", ["a", "b", "c"]

    def step(self, action):
        if self.boom_step:
            raise RuntimeError("tw crash")
        return "obs-" + action, action == self.wins_on, action == self.wins_on or action == self.done_on, ["a", "b", "c"]

    def close(self):
        self.closed = True


class FakeAdapter:
    def __init__(self, episode=None, open_error=None):
        self.episode, self.open_error = episode or FakeEpisode(wins_on="a"), open_error

    def task_instruction(self):
        return "INSTR"

    def process_action(self, action, choices):
        return action

    def open(self, game):
        if self.open_error:
            raise self.open_error
        return self.episode


def tool_resp(action, args=None):
    call = {"id": "c1", "type": "function",
            "function": {"name": "take_action", "arguments": args if args is not None else json.dumps({"action": action})}}
    return {"choices": [{"message": {"content": "", "tool_calls": [call]}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}}


def text_resp(text="hmm"):
    return {"choices": [{"message": {"content": text}}], "usage": {"prompt_tokens": 4, "completion_tokens": 2}}


def scripted(responses):
    queue = list(responses)
    seen = []

    def model(req, messages, timeout):
        seen.append((list(messages), timeout))
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    model.seen = seen
    return model


REQ = {"model": "m", "endpoint": "http://x/v1", "max_steps": 5, "episode_wall_s": 100.0, "request_timeout_s": 30.0}
EP = {"qid": "alfworld:train_valid:7", "index": 7, "game": "g"}


def test_episode_win_is_success_one_and_closes_env() -> None:
    adapter = FakeAdapter()
    row = drv.run_episode(adapter, REQ, EP, model=scripted([tool_resp("a")]))
    assert (row["success"], row["status"], row["detail"], row["steps"]) == (1, "completed", "won", 1)
    assert row["tokens_prompt"] == 10 and row["tokens_completion"] == 5 and adapter.episode.closed


def test_first_user_turn_and_tool_result_follow_agentbench_format() -> None:
    model = scripted([tool_resp("b"), tool_resp("a")])
    drv.run_episode(FakeAdapter(), REQ, EP, model=model)
    first = model.seen[0][0]
    assert first[0] == {"role": "system", "content": "INSTR"}
    assert first[1]["content"].startswith("Here is your task. ") and " AVAILABLE ACTIONS: a\nb\nc\n" in first[1]["content"]
    second = model.seen[1][0]
    assert second[-1]["role"] == "tool" and second[-1]["tool_call_id"] == "c1" and second[-1]["content"].startswith("obs-b")


def test_step_cap_is_solver_limit() -> None:
    cycle = [tool_resp(x) for x in ("b", "c", "b", "c", "b")]
    row = drv.run_episode(FakeAdapter(), REQ, EP, model=scripted(cycle))
    assert (row["success"], row["status"], row["detail"], row["steps"]) == (0, "solver_limit", "step_cap", 5)


def test_done_without_win_is_completed_failure() -> None:
    row = drv.run_episode(FakeAdapter(FakeEpisode(done_on="b")), REQ, EP, model=scripted([tool_resp("b")]))
    assert (row["success"], row["status"], row["detail"]) == (0, "completed", "done_without_win")


def test_repeating_one_action_three_times_is_candidate_error() -> None:
    row = drv.run_episode(FakeAdapter(), REQ, EP, model=scripted([tool_resp("b")] * 3))
    assert (row["success"], row["status"], row["detail"], row["steps"]) == (0, "candidate_error", "repeated_action", 3)


def test_no_tool_call_costs_a_step_and_gets_the_nudge() -> None:
    model = scripted([text_resp(), tool_resp("a")])
    row = drv.run_episode(FakeAdapter(), REQ, EP, model=model)
    assert row["success"] == 1 and row["steps"] == 2
    assert model.seen[1][0][-1] == {"role": "user", "content": drv.NO_TOOL_NUDGE}


def test_unusable_tool_arguments_are_answered_then_nudged() -> None:
    model = scripted([tool_resp("x", args="not json"), tool_resp("a")])
    row = drv.run_episode(FakeAdapter(), REQ, EP, model=model)
    assert row["success"] == 1
    tail = [m["role"] for m in model.seen[1][0][-3:]]
    assert tail == ["assistant", "tool", "user"]


def test_episode_wall_budget_is_solver_limit() -> None:
    ticks = iter([0.0, 1.0, 500.0, 500.0])  # start, step-1 check (ok), step-2 check (over 100 s), final elapsed
    row = drv.run_episode(FakeAdapter(), {**REQ, "episode_wall_s": 100.0}, EP,
                          model=scripted([tool_resp("b")]), clock=lambda: next(ticks))
    assert (row["status"], row["detail"], row["model_calls"]) == ("solver_limit", "episode_wall", 1)


def test_request_timeout_is_capped_by_the_remaining_wall() -> None:
    model = scripted([tool_resp("a")])
    drv.run_episode(FakeAdapter(), {**REQ, "episode_wall_s": 10.0, "request_timeout_s": 999.0}, EP, model=model)
    assert model.seen[0][1] <= 10.0


def test_endpoint_failure_is_infra_error_not_a_wrong_answer() -> None:
    row = drv.run_episode(FakeAdapter(), REQ, EP, model=scripted([drv.ModelError("infra", "http_503: down")]))
    assert (row["success"], row["status"]) == (0, "infra_error") and "model_error" in row["detail"]


def test_context_refusal_is_solver_limit() -> None:
    row = drv.run_episode(FakeAdapter(), REQ, EP, model=scripted([drv.ModelError("context", "http_400_context: too long")]))
    assert row["status"] == "solver_limit" and row["detail"].startswith("context_limit")


def test_env_failures_are_infra_error() -> None:
    row = drv.run_episode(FakeAdapter(open_error=RuntimeError("no game")), REQ, EP, model=scripted([]))
    assert row["status"] == "infra_error" and "env_error" in row["detail"]
    row = drv.run_episode(FakeAdapter(FakeEpisode(boom_step=True)), REQ, EP, model=scripted([tool_resp("a")]))
    assert row["status"] == "infra_error" and "tw crash" in row["detail"]


def test_malformed_model_response_is_infra_error() -> None:
    row = drv.run_episode(FakeAdapter(), REQ, EP, model=scripted([{"choices": []}]))
    assert (row["status"], row["detail"]) == ("infra_error", "malformed_model_response")


# ---------------------------------------------------------------- subprocess boundary (real driver, fake env + fake model)
def run_suite(monkeypatch, tmp_path, *, n=3, model=None, seed=5, rotation=0, **extra):
    lh = make_root(tmp_path)
    model = model or FakeModel()
    env_for(monkeypatch, lh, model, AUTOPILOT_ALFWORLD_N=n, AUTOPILOT_ALFWORLD_MAX_STEPS=6, **extra)
    run = suite.run_slice(tower_url="http://unused:1", base_seed=seed, rotation=rotation)
    return run, model


def test_slice_runs_through_the_subprocess_boundary(monkeypatch, tmp_path) -> None:
    run, model = run_suite(monkeypatch, tmp_path, n=6)
    try:
        assert isinstance(run, suite.SliceRun) and len(run.rows) == 6
        by_kind = {g.kind: row for g, row in zip(run.games, run.rows)}
        assert by_kind[STALL_KIND]["status"] == "candidate_error"  # same action 3x on a game that never finishes
        winners = [r for k, r in by_kind.items() if k != STALL_KIND]
        assert all(r["success"] == 1 and r["status"] == "completed" for r in winners)
        summary = run.descriptor()["summary"]
        assert summary["n"] == 6 and summary["successes"] == 5 and summary["status_split"]["candidate_error"] == 1
        assert run.driver["returncode"] == 0 and not run.driver["timed_out"]
        assert Path(run.artifact_dir, "request.json").exists() and Path(run.artifact_dir, "rows.jsonl").exists()
        body = model.bodies[0]
        assert body["path"] == "/v1/chat/completions" and body["model"] == "orchestrator"
        assert body["tools"][0]["function"]["name"] == "take_action" and "x_force_role" not in body
        assert body["messages"][0]["content"] == "FAKE INSTRUCTION"
    finally:
        model.close()


def test_routing_knobs_reach_the_model_request(monkeypatch, tmp_path) -> None:
    run, model = run_suite(monkeypatch, tmp_path, n=1, AUTOPILOT_ALFWORLD_ROLE="worker_general",
                           AUTOPILOT_ALFWORLD_MODEL="served-alias")
    try:
        assert run.rows[0]["status"] in {"completed", "candidate_error"}
        assert model.bodies[0]["x_force_role"] == "worker_general" and model.bodies[0]["model"] == "served-alias"
    finally:
        model.close()


def test_endpoint_defaults_to_the_tower_url(monkeypatch, tmp_path) -> None:
    lh = make_root(tmp_path)
    model = FakeModel()
    base = model.base.rsplit("/v1", 1)[0]
    env_for(monkeypatch, lh, None, AUTOPILOT_ALFWORLD_N=1)
    try:
        run = suite.run_slice(tower_url=base, base_seed=1, rotation=0)
        assert model.bodies and model.bodies[0]["path"] == "/v1/chat/completions"
        assert run.rows[0]["status"] != "infra_error"
    finally:
        model.close()


def test_dead_endpoint_rows_are_infra_error(monkeypatch, tmp_path) -> None:
    model = FakeModel(status=503)
    run, _ = run_suite(monkeypatch, tmp_path, n=2, model=model)
    try:
        assert [r["status"] for r in run.rows] == ["infra_error", "infra_error"]
        assert run.descriptor()["summary"]["scored_n"] == 0
    finally:
        model.close()


def test_missing_venv_python_yields_infra_rows_not_a_crash(monkeypatch, tmp_path) -> None:
    lh = make_root(tmp_path)
    env_for(monkeypatch, lh, None, AUTOPILOT_ALFWORLD_N=2, AUTOPILOT_ALFWORLD_PYTHON=str(tmp_path / "no-such-python"))
    run = suite.run_slice(tower_url="http://x", base_seed=1, rotation=0)
    assert [r["detail"] for r in run.rows] == ["driver_unavailable"] * 2 and run.driver["error"] == "driver_unavailable"


def test_driver_crash_leaves_infra_rows_with_the_exit_code(monkeypatch, tmp_path) -> None:
    crash = tmp_path / "crash_driver.py"
    crash.write_text("import sys\nsys.exit(3)\n")
    lh = make_root(tmp_path)
    env_for(monkeypatch, lh, None, AUTOPILOT_ALFWORLD_N=2, AUTOPILOT_ALFWORLD_DRIVER=str(crash))
    run = suite.run_slice(tower_url="http://x", base_seed=1, rotation=0)
    assert [r["detail"] for r in run.rows] == ["driver_exit_3_no_row"] * 2 and run.driver["returncode"] == 3


def test_batch_wall_kills_the_driver_and_keeps_finished_rows(monkeypatch, tmp_path) -> None:
    slow = tmp_path / "slow_driver.py"
    slow.write_text(
        "import json, sys, time\n"
        "req = json.load(open(sys.argv[sys.argv.index('--request') + 1]))\n"
        "out = sys.argv[sys.argv.index('--out') + 1]\n"
        "first = req['episodes'][0]\n"
        "open(out, 'a').write(json.dumps({'qid': first['qid'], 'index': first['index'], 'success': 1, 'status': 'completed',"
        " 'detail': 'won', 'steps': 1}) + '\\n')\n"
        "time.sleep(300)\n")
    lh = make_root(tmp_path)
    env_for(monkeypatch, lh, None, AUTOPILOT_ALFWORLD_N=3, AUTOPILOT_ALFWORLD_DRIVER=str(slow),
            AUTOPILOT_ALFWORLD_BATCH_WALL_S=2)
    spawned: list[subprocess.Popen] = []

    def recording_popen(*a, **kw):
        proc = subprocess.Popen(*a, **kw)
        spawned.append(proc)
        return proc

    run = suite.run_slice(tower_url="http://x", base_seed=1, rotation=0, popen=recording_popen)
    assert run.driver["timed_out"] is True
    assert [r["status"] for r in run.rows] == ["completed", "infra_error", "infra_error"]
    assert run.rows[1]["detail"] == "batch_wall_timeout"
    assert spawned[0].poll() is not None  # the driver this module started is dead (verified, not assumed)


def test_unavailable_slice_is_a_note_not_a_crash(monkeypatch, tmp_path) -> None:
    env_for(monkeypatch, tmp_path / "missing-lh", None)
    out = suite.run_slice(tower_url="http://x", base_seed=1, rotation=0)
    assert out["status"] == "unavailable" and "FileNotFoundError" in out["reason"]
    lh = make_root(tmp_path)
    env_for(monkeypatch, lh, None)
    games, _ = suite.load_split(suite.config_from_env(), suite.TRAIN_SPLIT)
    out = suite.run_slice(tower_url="http://x", base_seed=1, rotation=0, exclude_qids={g.qid for g in games})
    assert out == {"status": "unavailable", "reason": "empty_draw_after_exclusions"}


def test_standalone_cli_writes_slice_result_and_exit_codes(monkeypatch, tmp_path, capsys) -> None:
    lh = make_root(tmp_path)
    model = FakeModel()
    env_for(monkeypatch, lh, model, AUTOPILOT_ALFWORLD_N=3, AUTOPILOT_ALFWORLD_MAX_STEPS=6)
    try:
        assert suite.main(["--out", str(tmp_path / "out"), "--seed", "9", "--rotation", "2"]) == 0
        result = json.loads((tmp_path / "out" / "slice_result.json").read_text())
        assert result["summary"]["n"] == 3 and len(result["rows"]) == 3 and result["split"] == "train_valid"
        assert result["success_rate"] is not None and result["tokens_per_episode"] > 0
        assert all(q.startswith("alfworld:train_valid:") for q in result["qids"])
        assert json.loads(capsys.readouterr().out)["summary"]["n"] == 3
    finally:
        model.close()
    dead = FakeModel(status=503)
    env_for(monkeypatch, lh, dead)
    try:
        assert suite.main(["--out", str(tmp_path / "out2")]) == 6  # infra_error rows: no verdict, re-run
    finally:
        dead.close()
    env_for(monkeypatch, tmp_path / "missing", None)
    assert suite.main(["--out", str(tmp_path / "out3")]) == 2
    assert json.loads((tmp_path / "out3" / "slice_result.json").read_text())["status"] == "unavailable"


# ---------------------------------------------------------------- EvalTower wiring
def _result_for_question(q: dict, *, correct: bool = True):
    return eval_tower.QuestionResult(
        question_id=q["id"], suite=q["suite"], prompt=q["prompt"], expected=q["expected"], correct=correct,
        tokens_generated=100, elapsed_s=1.0, eval_concurrency=2, eval_wall_s=10.0,
        eval_partition=q.get("eval_partition", "core"))


def make_tower(monkeypatch, *, exclude=None):
    tower = eval_tower.EvalTower()
    tower._pool = {"math": [{"id": "core", "suite": "math", "tier": 1, "prompt": "2+2?", "expected": "4"}]}
    monkeypatch.setattr(eval_tower.EvalTower, "_load_tool_sentinels", lambda self: [])
    monkeypatch.setattr(eval_tower.EvalTower, "_eval_batch",
                        lambda self, questions, client, **kw: [_result_for_question(q) for q in questions])
    tower.bsv2_exclude_qids = exclude
    return tower


def test_flag_off_leaves_t1_untouched(monkeypatch, tmp_path) -> None:
    lh = make_root(tmp_path)
    env_for(monkeypatch, lh, None)  # configured but the flag is off
    off = make_tower(monkeypatch).eval_t1(n=1, seed=7, trial_id=3)
    assert "alfworld_slice" not in off.details and "_alfw" not in off.core_id
    assert all(r["suite"] != "alfworld" for r in off.question_results)
    assert "alfworld" not in off.details.get("decision_excluded_partitions", [])
    assert "alfworld_slice" not in off.details["test_profile"]
    assert not any(tmp_path.glob("alfworld_suite_*"))  # nothing was even drawn


def test_flag_on_adds_a_partitioned_slice_and_a_new_core_id(monkeypatch, tmp_path) -> None:
    lh = make_root(tmp_path)
    model = FakeModel()
    env_for(monkeypatch, lh, model, AUTOPILOT_ALFWORLD_N=3, AUTOPILOT_ALFWORLD_MAX_STEPS=6)
    try:
        off = make_tower(monkeypatch).eval_t1(n=1, seed=7, trial_id=3)
        monkeypatch.setenv("AUTOPILOT_ALFWORLD_SUITE", "1")
        on = make_tower(monkeypatch).eval_t1(n=1, seed=7, trial_id=3)
    finally:
        model.close()
    assert on.core_id.startswith(off.core_id + "_alfw3_") and on.core_id != off.core_id
    rows = [r for r in on.question_results if r["suite"] == "alfworld"]
    assert len(rows) == 3
    assert all(r["qid"].startswith("alfworld:train_valid:") and r["partition"] == "alfworld" for r in rows)
    assert all(r["status"] in rigor.STATUSES for r in rows)
    # decision aggregate is the core eval only: scalar quality and n unchanged by the slice
    assert on.n_questions == off.n_questions == 1 and on.quality == off.quality
    assert on.details["decision_excluded_partitions"] == ["alfworld"]
    note = on.details["alfworld_slice"]
    assert note["status"] == "ran" and note["summary"]["n"] == 3 and note["split"] == "train_valid"
    assert note["qids"] == [r["qid"] for r in rows]
    profile = on.details["test_profile"]["alfworld_slice"]
    assert profile["qids"] == note["qids"] and profile["split"] == "train_valid"
    assert on.details["dataset_content_sha256"] == off.details["dataset_content_sha256"]  # decision draw unchanged
    assert on.details["full_batch_n_questions"] == 4


def test_slice_scores_and_dispositions(monkeypatch, tmp_path) -> None:
    lh = make_root(tmp_path)
    model = FakeModel()
    env_for(monkeypatch, lh, model, AUTOPILOT_ALFWORLD_N=6, AUTOPILOT_ALFWORLD_MAX_STEPS=6)
    try:
        run = suite.run_slice(tower_url="http://x", base_seed=5, rotation=0)
    finally:
        model.close()
    results = eval_tower._alfworld_question_results(run)
    by_kind = {r.details["alfworld"]["kind"]: r for r in results}
    won = by_kind["pick_and_place"]
    assert won.correct and won.disposition == "scored" and won.error is None and won.scoring_method == "episode_success"
    stalled = by_kind[STALL_KIND]
    assert not stalled.correct and stalled.disposition == "task_failed" and stalled.error
    assert all(r.eval_partition == "alfworld" and r.suite == "alfworld" for r in results)
    infra = eval_tower._alfworld_question_results(suite.SliceRun(
        games=run.games, cfg=run.cfg, data_sha=run.data_sha,
        rows=[suite._infra_row(g, "model_error: x") for g in run.games]))
    assert all(r.disposition == "infra_failed" and r.infra_reason == "model_error" and not r.correct for r in infra)


def test_compact_row_carries_status_only_when_declared() -> None:
    plain = eval_tower._compact_question_result(_result_for_question({"id": "q", "suite": "math", "prompt": "p", "expected": "e"}))
    assert "status" not in plain
    declared = _result_for_question({"id": "q", "suite": "alfworld", "prompt": "p", "expected": "won"})
    declared.details = {"rigor_status": "solver_limit"}
    assert eval_tower._compact_question_result(declared)["status"] == "solver_limit"


def test_bsv2_partition_excludes_proposal_qids_from_the_draw(monkeypatch, tmp_path) -> None:
    lh = make_root(tmp_path)
    model = FakeModel()
    env_for(monkeypatch, lh, model, AUTOPILOT_ALFWORLD_N=3, AUTOPILOT_ALFWORLD_MAX_STEPS=6, AUTOPILOT_ALFWORLD_SUITE="1")
    try:
        first = make_tower(monkeypatch).eval_t1(n=1, seed=7, trial_id=3)
        proposal = {r["qid"] for r in first.question_results if r["suite"] == "alfworld"}
        accept = make_tower(monkeypatch, exclude=proposal).eval_t1(n=1, seed=7, trial_id=3)
    finally:
        model.close()
    accept_qids = {r["qid"] for r in accept.question_results if r["suite"] == "alfworld"}
    assert len(accept_qids) == 3 and not accept_qids & proposal  # R1: disjoint accept draw
    assert accept.core_id != first.core_id


def test_slice_failure_never_breaks_t1(monkeypatch, tmp_path) -> None:
    env_for(monkeypatch, tmp_path / "missing", None, AUTOPILOT_ALFWORLD_SUITE="1")
    result = make_tower(monkeypatch).eval_t1(n=1, seed=7, trial_id=3)
    assert result.details["alfworld_slice"]["status"] == "unavailable"
    assert "_alfw" not in result.core_id and all(r["suite"] != "alfworld" for r in result.question_results)
    assert result.n_questions == 1

    def boom(**kw):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(suite, "run_slice", boom)
    result = make_tower(monkeypatch).eval_t1(n=1, seed=7, trial_id=3)
    assert result.details["alfworld_slice"] == {"status": "error", "reason": "RuntimeError: kaboom"}
    assert "_alfw" not in result.core_id


def test_diagnostic_observation_never_runs_the_slice(monkeypatch, tmp_path) -> None:
    env_for(monkeypatch, make_root(tmp_path), None, AUTOPILOT_ALFWORLD_SUITE="1")
    assert make_tower(monkeypatch)._alfworld_slice_for_t1(seed=1, trial_id=1, diagnostic=True) == (None, {})


# ---------------------------------------------------------------- the accept gate sees split-qualified ids
def test_rigor_gate_sees_alfworld_qids_and_status_split(monkeypatch, tmp_path) -> None:
    lh = make_root(tmp_path)
    model = FakeModel()
    env_for(monkeypatch, lh, model, AUTOPILOT_ALFWORLD_N=3, AUTOPILOT_ALFWORLD_MAX_STEPS=6, AUTOPILOT_ALFWORLD_SUITE="1")
    try:
        base = make_tower(monkeypatch).eval_t1(n=1, seed=7, trial_id=3)
        cand = make_tower(monkeypatch).eval_t1(n=1, seed=7, trial_id=3)
    finally:
        model.close()
    state = rigor.new_state()
    state["proposal_qids"] = ["math:other"]
    state["budget"] = {"unit": "rollouts", "limit": 100, "spent": 0}
    state["sealed_qids"] = suite.sealed_qids(suite.config_from_env())
    record = rigor.evaluate_rigor(rigor_state=state, baseline_rows=base.question_results, candidate_rows=cand.question_results,
                                  baseline_fresh=True, paired_accept=True, accuracy_delta=0.0)
    assert record["rules"]["R3"]["ok"] and record["rules"]["R1"]["ok"]
    split = record["rules"]["R4"]["status_split"]["baseline"]
    assert sum(split.values()) == len(base.question_results) == 4  # 1 core row + 3 alfworld rows, all with a status
    assert split["completed"] >= 1
    # the proposal batch of the previous round (alfworld qids included) overlapping the accept eval is an R1 blocker
    prev = [r["qid"] for r in base.question_results if r["suite"] == "alfworld"]
    state["proposal_qids"] = prev
    overlap = rigor.evaluate_rigor(rigor_state=state, baseline_rows=base.question_results,
                                   candidate_rows=cand.question_results, baseline_fresh=True, paired_accept=True,
                                   accuracy_delta=0.0)
    assert not overlap["rules"]["R1"]["ok"] and set(overlap["rules"]["R1"]["overlap"]) >= set(prev[:3])
    # a sealed TEST qid in an accept eval is an R3 blocker
    leaked = [{"qid": state["sealed_qids"][0], "suite": "alfworld"}]
    bad = rigor.evaluate_rigor(rigor_state=state, baseline_rows=leaked, candidate_rows=leaked, baseline_fresh=True,
                               paired_accept=True, accuracy_delta=0.0)
    assert not bad["rules"]["R3"]["ok"] and not bad["accept"]


def test_actions_seal_the_test_split_only_when_the_flag_is_on(monkeypatch, tmp_path) -> None:
    from types import SimpleNamespace

    env_for(monkeypatch, make_root(tmp_path), None)
    ctx = SimpleNamespace(state={"bsv2_rigor": {"sealed_qids": ["pool:1"]}})
    actions._bsv2_seal_alfworld_test_split(ctx)
    assert ctx.state["bsv2_rigor"]["sealed_qids"] == ["pool:1"]  # flag off: untouched
    monkeypatch.setenv("AUTOPILOT_ALFWORLD_SUITE", "1")
    actions._bsv2_seal_alfworld_test_split(ctx)
    sealed = ctx.state["bsv2_rigor"]["sealed_qids"]
    assert "pool:1" in sealed and len([q for q in sealed if q.startswith("alfworld:new_std:")]) == 12
    actions._bsv2_seal_alfworld_test_split(ctx)  # idempotent
    assert ctx.state["bsv2_rigor"]["sealed_qids"] == sealed
    env_for(monkeypatch, tmp_path / "missing", None)
    actions._bsv2_seal_alfworld_test_split(ctx)  # unreadable split is logged, not fatal
    assert ctx.state["bsv2_rigor"]["sealed_qids"] == sealed


# ---------------------------------------------------------------- wiring checklist
def test_flag_is_fully_wired() -> None:
    import yaml

    config_applicator = importlib.import_module("config_applicator")
    assert "autopilot_alfworld_suite" in config_applicator.HOT_SWAP_FEATURES
    spec = yaml.safe_load((ROOT / "orchestration" / "runtime_flags.spec.yaml").read_text())
    assert spec["flags"]["autopilot_alfworld_suite"] == "baseline"
    assert "`autopilot_alfworld_suite`" in (ROOT / "scripts" / "autopilot" / "program.md").read_text()
    assert "`autopilot_alfworld_suite`" in (ROOT / "docs" / "chapters" / "01-runtime-environment.md").read_text()


def test_driver_is_python310_safe_and_standalone() -> None:
    import ast

    tree = ast.parse((ROOT / "scripts" / "autopilot" / "alfworld_episode_driver.py").read_text(), feature_version=(3, 10))
    top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]  # module level only
    imported = {a.name.split(".")[0] for n in top if isinstance(n, ast.Import) for a in n.names}
    imported |= {n.module.split(".")[0] for n in top if isinstance(n, ast.ImportFrom) and n.module}
    assert not imported & {"httpx", "requests", "src", "yaml", "eval_tower", "alfworld_suite"}
    assert imported <= set(sys.stdlib_module_names)  # the orchestrator never needs the env venv, nor vice versa


def test_subprocess_env_is_an_allowlist(monkeypatch) -> None:
    monkeypatch.setenv("SECRET_API_KEY", "x")
    monkeypatch.setenv("PATH", "/bin")
    env = suite._driver_env()
    assert "SECRET_API_KEY" not in env and env["PATH"] == "/bin" and env["OMP_NUM_THREADS"] == "1"
