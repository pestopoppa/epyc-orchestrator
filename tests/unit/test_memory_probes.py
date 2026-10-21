"""M-27 fixtures: each advisory class has a fixture test (no model, no embedder)."""

import json
import sqlite3

from orchestration.repl_memory.distillation.literal_lint import extract_negative_capability, unlabelled_success
from orchestration.repl_memory.distillation.memory_probes import (
    classify_source,
    lint_extensions_skills_db,
    pair_census,
    recall_composition,
    retrieval_floor_report,
    source_composition,
    source_role_per_skill,
)
from orchestration.repl_memory.distillation.pii_seed import extract_pii_seed


def test_negative_capability():
    assert extract_negative_capability("The tool does not work with streams; you cannot use pandas here")
    assert extract_negative_capability("Prefer simple strategies") == []


def test_unlabelled_success():
    assert unlabelled_success("failure") and unlabelled_success(None) and unlabelled_success("mixed")
    assert not unlabelled_success("success")


def test_pii_seed_classes_and_exceptions():
    t = ("see /Users/bob/data and /home/alice/x, host 8.8.8.8 (not 192.168.1.5), "
         "postgres://u:p@db.example.org/x, key " + "AK" + "IA" + "ABCDEFGHIJKLMNOP, mail jo@corp.io")
    got = extract_pii_seed(t)
    assert {"PII_PATH", "PII_IP", "PII_URL", "PII_ID"} <= set(got)
    assert "8.8.8.8" in got["PII_IP"] and "192.168.1.5" not in got["PII_IP"]
    assert extract_pii_seed("GET /users/:id") == {}
    assert extract_pii_seed("Copyright jo@corp.io") == {}


def _skills_db(path):
    con = sqlite3.connect(path)
    con.execute("""CREATE TABLE skills (id TEXT, title TEXT, skill_type TEXT, principle TEXT, when_to_apply TEXT,
        task_types TEXT, source_trajectory_ids TEXT, source_outcome TEXT, confidence REAL, retrieval_count INT,
        effectiveness_score REAL, embedding_idx INT, created_at TEXT, updated_at TEXT, revision INT, deprecated INT,
        parent_id TEXT, teacher_model TEXT)""")
    rows = [("a", "A", "routing", "It does not work on port 8083", "w", '["chat"]', '["m1"]', "success", .5, 7, .2, 0, "1", "1", 1, 0, None, "t"),
            ("b", "B", "failure_lesson", "Use /data/x.csv", "w", '["chat"]', '["m2"]', "failure", .5, 25, .9, 1, "2", "2", 1, 0, None, "t"),
            ("c", "C", "routing", "General advice", "w", '["code"]', '["zz"]', "success", .5, 0, .1, 2, "3", "3", 1, 0, None, "t")]
    con.executemany("INSERT INTO skills VALUES (%s)" % ",".join("?" * 18), rows)
    con.commit(); con.close()


def test_skills_db_probes(tmp_path):
    _skills_db(tmp_path / "s.db")
    ext = lint_extensions_skills_db(tmp_path / "s.db")
    assert ext["negative_capability_flagged"] == 1 and ext["unlabelled_success_flagged"] == 1
    fl = retrieval_floor_report(tmp_path / "s.db")
    assert fl["floors"]["5"]["at_or_above_floor"] == 2 and fl["floors"]["20"]["at_or_above_floor"] == 1
    assert fl["floors"]["5"]["below_effectiveness_0.3"] == 1 and fl["floors"]["20"]["below_effectiveness_0.3"] == 0
    e = sqlite3.connect(tmp_path / "e.db")
    e.execute("CREATE TABLE memories (id TEXT, action TEXT)")
    e.execute("INSERT INTO memories VALUES ('m1','frontdoor:direct')")
    e.commit(); e.close()
    sr = source_role_per_skill(tmp_path / "s.db", [tmp_path / "e.db"])
    assert sr["skills_by_top_role"] == {"frontdoor": 1, "unresolved": 2}


def test_pair_census_band_and_literal_difference():
    def it(i, tt, text, vec):
        return {"id": i, "task_types": set(tt), "text": text, "vec": vec}
    items = [it("a", ["chat"], "Use /data/a.csv", [1, 0]),
             it("b", ["chat"], "Use /data/b.csv", [0.75, 0.66]),   # cos ~0.75, literals differ
             it("c", ["chat"], "Use /data/a.csv", [0.75, 0.66]),   # same literals as a
             it("d", ["code"], "Use /data/z.csv", [0.75, 0.66])]   # other task type
    r = pair_census(items)
    assert r["pairs_same_task_type"] == 3 and r["pairs_in_band"] == 2
    assert r["pairs_in_band_literals_differ"] == 1 and len(r["examples"]) == 1


def test_source_classes_and_recall_composition():
    assert classify_source("progress_log") == "autopilot" and classify_source("zzz") == "unlabelled"
    assert source_composition(["seed", "external", None])["counts"] == {"seed": 1, "eval_or_external": 1, "unlabelled": 1}
    r = recall_composition(lambda q, k: ["seed", "progress_log"][:k], queries=["q1", "q2"], k=2)
    assert r["pooled"] == {"seed": 2, "autopilot": 2}


def test_distillation_report_call_totals():
    from orchestration.repl_memory.distillation.pipeline import DistillationReport
    d = DistillationReport(teacher_calls=3, prompt_chars=4000, response_chars=800).to_dict()
    assert d["teacher_calls"] == 3 and d["approx_prompt_tokens"] == 1000 and d["approx_response_tokens"] == 200
