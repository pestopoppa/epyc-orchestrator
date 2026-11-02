"""P17.BT-5a synthesizer and backtest on synthetic journals; no model or endpoint."""

import hashlib
import json
import random

from scripts.autopilot.experiment_journal import JournalEntry
from scripts.autopilot.feedback_synthesizer import (
    MIN_HISTORY,
    backtest,
    coverage_ceiling,
    extract_pairs,
    param_key,
    render_hints,
    run,
    spearman,
    synthesize_hints,
)


def entry(trial_id, *, quality=1.0, speed=10.0, params=None, surface="surf",
          action_type="numeric_trial", **changes):
    snapshot = {"type": action_type, "surface": surface}
    if params is not None:
        snapshot["params"] = params
    args = dict(trial_id=trial_id, timestamp=f"2026-07-01T00:{trial_id % 60:02d}:00Z",
                species="numeric_swarm", action_type=action_type, tier=1, quality=quality,
                speed=speed, cost=0.1, reliability=1.0, pareto_status="dominated",
                config_snapshot=snapshot, keep_revert_decision="revert")
    args.update(changes)
    return JournalEntry(**args)


def signal_corpus(n=90, slope=1.0, seed=7):
    rng = random.Random(seed)
    rows = [entry(i, quality=1.0, params=None, action_type="seed_batch") for i in range(4)]
    for i in range(4, 4 + n):
        x = rng.random()
        rows.append(entry(i, quality=1.0 + slope * x + rng.gauss(0, 0.05),
                          speed=10.0, params={"surf.x": x}))
    return rows


def test_param_key_and_spearman():
    assert param_key("memrl_retrieval", "semantic_k") == "memrl_retrieval.semantic_k"
    assert param_key(None, "repl.turn_token_cap") == "repl.turn_token_cap"
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0
    assert spearman([1, 2, 3], [5, 5, 5]) == 0.0
    assert abs(spearman([1, 1, 2, 3], [1, 2, 3, 4]) - 0.9486832980505138) < 1e-12


def test_extract_pairs_applies_trust_filter_and_skips_unlogged_params():
    rows = signal_corpus(n=12)
    rows += [
        entry(100, params={"surf.x": 0.5}, bug_corrupted_by="contention"),
        entry(101, params={"surf.x": 0.5}, keep_revert_decision="excluded"),
        entry(102, params={}),
        entry(103, params={"flag": True}),
        entry(104, params={"x": 0.5}, quality=0.0),
    ]
    pairs = extract_pairs(rows)
    ids = {p.trial_id for p in pairs}
    assert not ids & {100, 101, 102, 103, 104}
    assert all(set(p.params) == {"surf.x"} for p in pairs)


def test_directional_hint_sign_and_scalar_fallback():
    pairs = extract_pairs(signal_corpus(n=40))
    hints = synthesize_hints(pairs)
    assert hints["surf.x"].kind == "directional" and hints["surf.x"].sign == 1
    assert render_hints(hints).startswith("- increase surf.x")
    assert synthesize_hints(pairs[:MIN_HISTORY - 1]) == {}
    assert render_hints({}) == "(no feedback hint: scalar path)"
    negative = synthesize_hints(extract_pairs(signal_corpus(n=40, slope=-1.0)))
    assert negative["surf.x"].sign == -1


def test_walk_forward_never_sees_the_future():
    rows = signal_corpus(n=40)
    first = backtest(extract_pairs(rows))["per_trial"]
    # Invert every LATER trial's outcome; earlier hints must not change.
    cut = 25
    flipped = [r if r.trial_id < cut else entry(r.trial_id, quality=3.0 - r.quality,
                                                params=r.config_snapshot.get("params"),
                                                action_type=r.action_type)
               for r in rows]
    second = backtest(extract_pairs(flipped))["per_trial"]
    for a, b in zip(first, second):
        if a["trial_id"] < cut:
            assert a["hints_available"] == b["hints_available"] and a["units"] == b["units"]


def test_signal_corpus_passes_and_scrambled_control_does_not():
    result = backtest(extract_pairs(signal_corpus(n=90)), perm_reps=400)
    assert result["verdict"] == "SIGNAL"
    assert result["uplift_follow_vs_baseline"] > result["scrambled_control"]["null_uplift_mean"]
    assert result["scrambled_control"]["one_sided_p"] <= 0.05


def test_no_relationship_is_not_signal_and_small_corpus_is_inconclusive():
    noise = backtest(extract_pairs(signal_corpus(n=90, slope=0.0)), perm_reps=400)
    assert noise["verdict"] != "SIGNAL"
    small = backtest(extract_pairs(signal_corpus(n=10)), perm_reps=50)
    assert small["verdict"] == "INCONCLUSIVE_INSUFFICIENT_COVERAGE"
    assert coverage_ceiling(extract_pairs(signal_corpus(n=10)))["reachable"] is False


def test_run_reads_every_shard_and_leaves_them_unchanged(tmp_path):
    rows = signal_corpus(n=30)
    def dump(e):
        d = {k: getattr(e, k) for k in ("trial_id", "timestamp", "species", "action_type", "tier",
                                        "quality", "speed", "cost", "reliability",
                                        "pareto_status", "config_snapshot",
                                        "keep_revert_decision")}
        return json.dumps(d) + "\n"
    base = tmp_path / "autopilot_journal.jsonl"
    rotated = tmp_path / "autopilot_journal_1.jsonl"
    base.write_text("".join(dump(e) for e in rows[:15]))
    rotated.write_text("".join(dump(e) for e in rows[15:]))
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (base, rotated)}
    record = run(tmp_path)
    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (base, rotated)}
    assert before == after
    assert [s["name"] for s in record["inputs"]["shards"]] == [base.name, rotated.name]
    assert record["inputs"]["trial_entries"] == len(rows)
    assert record["corpus"]["trial_id_range"][1] == rows[-1].trial_id
