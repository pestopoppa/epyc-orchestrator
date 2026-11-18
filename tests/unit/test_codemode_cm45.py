"""CM-4 (read-only typed classifier/q-scorer functions) and CM-5 (replay recording default-on with handles).

Every flag is default OFF; with the flags off nothing here may change a REPL global, a checkpoint key or a seed.
No model is called: the classifiers are the deterministic rule ones and the retriever is a fake."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.features import reset_features
from src.repl_environment import codemode_tools as cm
from src.repl_environment.environment import REPLEnvironment
from src.trace import store as trace_store

FLAGS = (cm.ENV_SEARCH, cm.ENV_CLASSIFY, cm.ENV_HANDLES, cm.ENV_CLASSIFY_BUDGET, cm.ENV_STUBS,
         "ORCHESTRATOR_REPLAY_RECORDING", "ORCHESTRATOR_FEATURE_REPLAY_RECORDING", "ORCHESTRATOR_STEP_EVENT_LOGGING",
         "ORCHESTRATOR_TOOL_SELECTOR")


@pytest.fixture(autouse=True)
def _clean(monkeypatch, tmp_path):
    for f in FLAGS:
        monkeypatch.delenv(f, raising=False)
    db = tmp_path / "events.sqlite"
    trace_store.ensure_schema(db).close()
    monkeypatch.setattr(trace_store, "DEFAULT_DB_PATH", db)
    monkeypatch.setenv("ORCHESTRATOR_REPLAY_CAS_DIR", str(tmp_path / "cas"))
    reset_features()
    yield
    reset_features()


NEW_NAMES = {"route_classify", "risk_bands", "q_scores"}


# ------------------------------------------------------------------------------------------------ CM-4
class _Mem:
    def __init__(self, action, role=None):
        self.action, self.context = action, ({"role": role} if role else {})


class _Hit:
    def __init__(self, action, q, role=None):
        self.memory, self.q_value = _Mem(action, role), q


class _Retriever:
    """Only ``retrieve_for_routing`` exists: any write or other call is an AttributeError (read-only proof)."""

    def __init__(self):
        self.calls = []

    def retrieve_for_routing(self, task_ir):
        self.calls.append(task_ir)
        return [_Hit("worker_general", 0.9), _Hit("worker_general", 0.5), _Hit("coder", 0.8, role="coder_escalation"),
                _Hit("architect", 0.1)]


def test_flag_off_installs_nothing_and_leaves_no_state():
    env = REPLEnvironment(context="c")
    assert not (NEW_NAMES & set(env._globals))
    assert getattr(env, "_classifier_budget", None) is None


def test_route_classify_typed_rows_and_trinity_role():
    rows = cm.route_classify(["Please review this answer I wrote: 2+2=5", "what is the capital of France"])
    assert len(rows) == 2
    for r in rows:
        assert set(r) == {"role", "strategy", "confidence", "matched_keywords", "trinity_role", "trinity_reason"}
        assert isinstance(r["role"], str) and isinstance(r["confidence"], float)
        assert isinstance(r["matched_keywords"], list)
    assert rows[0]["trinity_role"] == "verifier" and rows[1]["trinity_role"] == "worker"
    assert rows == cm.route_classify(["Please review this answer I wrote: 2+2=5", "what is the capital of France"])


def test_risk_bands_typed_rows():
    rows = cm.risk_bands(["When was the Treaty of Westphalia signed? Cite sources.", "hi"])
    assert [set(r) for r in rows] == [{"risk_band", "risk_score", "difficulty_band", "difficulty_score"}] * 2
    assert rows[0]["risk_score"] >= rows[1]["risk_score"]
    assert all(isinstance(r["risk_band"], str) and isinstance(r["difficulty_score"], float) for r in rows)


def test_q_scores_typed_ranked_read_only():
    r = _Retriever()
    out = cm.q_scores(r, ["route me"], k=2)
    assert out == [[{"role": "coder_escalation", "q_value": 0.8, "count": 1},
                    {"role": "worker_general", "q_value": 0.7, "count": 2}]]
    assert len(r.calls) == 1 and r.calls[0]["objective"] == "route me"
    with pytest.raises(RuntimeError):
        cm.q_scores(None, ["x"])


def test_cost_cap_per_call_total_budget_and_item_truncation():
    with pytest.raises(ValueError, match="capped at 200"):
        cm.route_classify(["x"] * 201)
    b = cm.ClassifierBudget(limit=5)
    cm.risk_bands(["a"] * 3, budget=b)
    with pytest.raises(ValueError, match="budget exhausted"):
        cm.route_classify(["a"] * 3, budget=b)
    assert b.used == 3  # a refused call spends nothing
    with pytest.raises(TypeError):
        cm.route_classify("one string, not a list")
    r = _Retriever()
    cm.q_scores(r, ["z" * 100_000])
    assert len(r.calls[0]["objective"]) <= cm.MAX_ITEM_CHARS
    assert cm.ClassifierBudget.from_env({cm.ENV_CLASSIFY_BUDGET: "7"}).limit == 7
    assert cm.ClassifierBudget.from_env({cm.ENV_CLASSIFY_BUDGET: "junk"}).limit == cm.DEFAULT_CLASSIFIER_BUDGET


def test_install_binds_functions_and_shares_one_budget(monkeypatch):
    monkeypatch.setenv(cm.ENV_CLASSIFY, "1")
    monkeypatch.setenv(cm.ENV_CLASSIFY_BUDGET, "3")
    g: dict = {}
    env = SimpleNamespace(llm_primitives=None, tool_registry=None, _retriever=_Retriever(), _hybrid_router=None)
    assert set(cm.install(env, g)) == NEW_NAMES
    assert len(g["route_classify"](["a", "b"])) == 2
    assert len(g["q_scores"](["c"])[0]) == 3
    with pytest.raises(ValueError, match="budget exhausted"):
        g["risk_bands"](["d"])
    g2: dict = {}
    env2 = SimpleNamespace(llm_primitives=None, tool_registry=None, _retriever=None, _hybrid_router=None)
    assert set(cm.install(env2, g2)) == {"route_classify", "risk_bands"}  # no retriever: no q_scores


def test_environment_exposes_functions_to_repl_code(monkeypatch):
    monkeypatch.setenv(cm.ENV_CLASSIFY, "1")
    env = REPLEnvironment(context="c")
    res = env.execute("rows = route_classify(['review this answer: 1+1=3', 'hello'])\nprint([r['trinity_role'] for r in rows])")
    assert not res.error and "verifier" in res.output and "worker" in res.output
    over = env.execute("risk_bands(['x'] * 500)")
    assert over.error and "capped" in over.error


# ------------------------------------------------------------------------------------------------ CM-5
def test_replay_default_on_matrix(monkeypatch):
    assert cm.replay_default_on({}) is False                                         # handles off
    assert cm.replay_default_on({cm.ENV_HANDLES: "1"}) is True
    assert cm.replay_default_on({cm.ENV_HANDLES: "1", "ORCHESTRATOR_REPLAY_RECORDING": "0"}) is False
    assert cm.replay_default_on({cm.ENV_HANDLES: "1", "ORCHESTRATOR_FEATURE_REPLAY_RECORDING": "0"}) is False
    monkeypatch.setenv(cm.ENV_HANDLES, "1")
    assert cm.replay_default_on() is True
    monkeypatch.setenv("ORCHESTRATOR_REPLAY_RECORDING", "0")
    assert cm.replay_default_on() is False


def _recorders(env):
    return env._replay_recorder, env._step_events


def test_environment_records_by_default_only_when_handles_on(monkeypatch):
    rec, steps = _recorders(REPLEnvironment(context="c"))
    assert rec is None and steps is None                                               # flags off: untouched
    monkeypatch.setenv(cm.ENV_HANDLES, "1")
    reset_features()
    rec, steps = _recorders(REPLEnvironment(context="c"))
    assert rec is not None and steps is not None and steps.cas is not None
    monkeypatch.setenv("ORCHESTRATOR_REPLAY_RECORDING", "0")                           # explicit off wins
    reset_features()
    rec, steps = _recorders(REPLEnvironment(context="c"))
    assert rec is None and steps is None


def test_handles_survive_checkpoint_restore_and_branch(monkeypatch):
    monkeypatch.setenv(cm.ENV_HANDLES, "1")
    reset_features()
    env = REPLEnvironment(context="c")
    assert not env.execute("store('big', list(range(1000)))\nstore('k', {'a': 1})").error
    cp = env.checkpoint()
    assert set(cp["handle_store"]) == {"big", "k"}
    branch = REPLEnvironment(context="c")
    branch.execute("store('stale', 1)")
    branch.restore(cp)
    assert branch._handle_store.handles().keys() == {"big", "k"}                       # replaced, not merged
    out = branch.execute("print(len(load('handle:big')), load('k'))")
    assert not out.error and out.output.strip() == "1000 {'a': 1}"
    # later writes on the branch do not leak back into the checkpoint's source
    branch.execute("store('k', 'changed')")
    assert env._handle_store.load("k") == {"a": 1}
    # a checkpoint taken WITHOUT handles clears a handles-on session on restore (a faithful branch)
    monkeypatch.delenv(cm.ENV_HANDLES)
    reset_features()
    plain = REPLEnvironment(context="c").checkpoint()
    assert "handle_store" not in plain
    monkeypatch.setenv(cm.ENV_HANDLES, "1")
    reset_features()
    branch.restore(plain)
    assert branch._handle_store.handles() == {}


def test_seed_carries_handle_store_only_when_present(tmp_path):
    from src.trace.replay.cas import CAS
    from src.trace.replay.seed import _handle_store_seed

    cas = CAS(tmp_path / "c")
    assert _handle_store_seed(SimpleNamespace(), cas) == {}                            # handles off: seed unchanged
    hs = cm.HandleStore()
    hs.begin_cell(); hs.store("a", [1]); hs.commit()
    s = _handle_store_seed(SimpleNamespace(_handle_store=hs), cas)
    assert cas.get_text(s["handle_store_cas"]) == '{"a": "[1]"}'
    assert _handle_store_seed(SimpleNamespace(_handle_store=cm.HandleStore()), cas) == {"handle_store_cas": None}
