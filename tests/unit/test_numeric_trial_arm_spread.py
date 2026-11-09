"""AP-68: planner-vs-sampler numeric trial spread/step, on synthetic journal + Optuna DB."""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.analysis import numeric_trial_arm_spread as mod  # noqa: E402

SURF = "chat_pipeline"
PARAM = "chat.try_cheap_first_q_threshold"  # bounds 0.3 .. 0.9


def _row(trial_id: int, params, surface: str = SURF, action_type: str = "numeric_trial") -> dict:
    return {"trial_id": trial_id, "action_type": action_type, "outcome_status": "ok",
            "reasoning": json.dumps({"type": "numeric_trial", "surface": surface, "params": params})}


def _journal(tmp_path: Path) -> Path:
    d = tmp_path / "journal"
    d.mkdir()
    (d / "autopilot_journal.jsonl").write_text("\n".join(json.dumps(r) for r in [
        _row(1, {PARAM: 0.3}),
        _row(2, {}),  # sampler-sourced: params stored empty
        _row(3, {PARAM: 0.9}, action_type="prompt_mutation"),  # not numeric
    ]) + "\n{not json\n")
    (d / "autopilot_journal_1.jsonl").write_text(json.dumps(_row(4, {PARAM: 0.6})) + "\n")
    return d


def _optuna(tmp_path: Path, studies: dict[str, list[float]]) -> Path:
    db = tmp_path / "optuna.db"
    con = sqlite3.connect(db)
    con.executescript("""
        create table studies(study_id integer primary key, study_name text);
        create table trials(trial_id integer primary key, number integer, study_id integer, state text);
        create table trial_params(param_id integer primary key, trial_id integer, param_name text,
                                  param_value float, distribution_json text);
    """)
    tid = 0
    for sid, (name, values) in enumerate(studies.items(), 1):
        con.execute("insert into studies values (?,?)", (sid, name))
        for number, v in enumerate(values):
            tid += 1
            con.execute("insert into trials values (?,?,?,?)", (tid, number, sid, "COMPLETE"))
            con.execute("insert into trial_params(trial_id,param_name,param_value) values (?,?,?)",
                        (tid, PARAM, v))
        con.execute("insert into trials values (?,?,?,?)", (tid + 1000, 99, sid, "RUNNING"))
    con.commit()
    con.close()
    return db


def test_planner_trials_come_from_explicit_params_across_shards(tmp_path):
    trials = mod.planner_trials(_journal(tmp_path))
    assert [(t["trial_id"], t["params"][PARAM]) for t in trials] == [(1, 0.3), (4, 0.6)]


def test_surface_of_study_names():
    assert mod._surface_of("autopilot_chat_pipeline") == SURF
    assert mod._surface_of("autopilot_think_harder_epoch7") == "think_harder"
    assert mod._surface_of("autopilot_memrl_retrieval_quarantined_20260613_786_797") == "memrl_retrieval"
    assert mod._surface_of("other_study") is None
    assert mod._surface_of("autopilot_unknown_surface") is None


def test_normalize_to_bounds():
    b = mod._bounds()
    assert mod.normalize(b, SURF, PARAM, 0.3) == pytest.approx(0.0)
    assert mod.normalize(b, SURF, PARAM, 0.9) == pytest.approx(1.0)
    assert mod.normalize(b, SURF, PARAM, 0.6) == pytest.approx(0.5)
    assert mod.normalize(b, SURF, "nope", 0.6) is None
    assert mod.normalize(b, SURF, PARAM, True) is None
    assert mod.normalize(b, SURF, PARAM, float("nan")) is None


def test_report_spread_step_and_running_trials_excluded(tmp_path):
    db = _optuna(tmp_path, {"autopilot_chat_pipeline": [0.3, 0.9, 0.3, 0.9]})
    rep = mod.build_report(_journal(tmp_path), db)
    cell = next(c for c in rep["cells"] if c["param"] == PARAM)
    assert cell["planner"]["n"] == 2 and cell["sampler"]["n"] == 4  # RUNNING trial not counted
    assert cell["planner"]["mean_step"] == pytest.approx(0.5)  # 0.0 -> 0.5
    assert cell["sampler"]["mean_step"] == pytest.approx(1.0)
    assert cell["sampler"]["spread"] == pytest.approx(0.5)
    assert rep["planner_trials"] == 2 and rep["sampler_trials"] == 4
    assert rep["summary"]["cells_with_both_arms_n_ge_3"] == 0  # planner n=2


def test_missing_db_yields_empty_sampler_arm_and_cli_writes_output(tmp_path, capsys):
    journal = _journal(tmp_path)
    rep = mod.build_report(journal, tmp_path / "absent.db")
    assert rep["sampler_trials"] == 0
    out = tmp_path / "o.md"
    assert mod.main(["--journal-dir", str(journal),
                     "--optuna-db", str(tmp_path / "absent.db"), "--out", str(out)]) == 0
    assert "Numeric trial arms" in out.read_text()
