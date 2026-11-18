#!/usr/bin/env python3
"""Runner for the codemode A/B/C probe. Resumable: one ``episode`` record per unit in ``OUT/results.jsonl``.

    run_probe.py --out OUT --model-id ID --url http://127.0.0.1:PORT --embed-url http://127.0.0.1:PORT2 --embed-id ID
    run_probe.py --mock --out OUT [--limit N]          # no model, no server: scripted solver (smoke)
    run_probe.py --dry-run --out OUT                   # prints the plan only, writes nothing
    run_probe.py ... --arm-set v2                      # probe v2: A,B (reused controls) + A_pick,B1,B2,C3 (PREDECLARED_v2.md)

Stages (predeclared in PREDECLARED.md): 1 = singles + N in {5,20}; 2 = N=50; 3 = N in {100,200}.
After stage 1 and 2 the futility / descriptive-answer gates of ``score.evaluate`` are applied; a stop writes
``OUT/TERMINAL.json``. rc: 0 complete (or gate stop) | 75 stop-file/deadline (resumable) | 3 environment.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ORCH_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ORCH_ROOT))

from scripts.benchmark.codemode_ab import score, score_v2  # noqa: E402
from scripts.benchmark.codemode_ab.arms import ARMS, ARMS_V1, ARMS_V2, K_SELECTED, run_episode  # noqa: E402
from scripts.benchmark.codemode_ab.tasks import STAGES, build_tasks  # noqa: E402
from scripts.benchmark.codemode_ab.world import GROUP_DESC, World  # noqa: E402

SCRATCH = "/mnt/raid0/llm/tmp/"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class Out:
    def __init__(self, out: Path, mode: str):
        self.dir, self.mode = out, mode
        self.res = out / "results.jsonl"

    def append(self, rec: dict) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        with self.res.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"mode": self.mode, "at": now(), **rec}, default=str) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def transcript(self, uid: str, turns: list) -> str:
        """Full per-turn transcript of one episode -> ``transcripts/<uid>.jsonl`` (one JSON object per turn). Returns the relative path."""
        d = self.dir / "transcripts"
        d.mkdir(parents=True, exist_ok=True)
        name = uid.replace("|", "__").replace("/", "_") + ".jsonl"
        with (d / name).open("w", encoding="utf-8") as fh:
            for t in turns:
                fh.write(json.dumps(t, default=str) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        return f"transcripts/{name}"

    def done(self) -> set[str]:
        s: set[str] = set()
        if self.res.exists():
            for ln in self.res.read_text().splitlines():
                try:
                    r = json.loads(ln)
                except ValueError:
                    continue
                if r.get("mode") == self.mode and r.get("kind") == "episode":
                    s.add(r["unit_id"])
        return s


def plan(seeds: list[int], arms: tuple[str, ...], stages: list[int]) -> list[tuple]:
    tasks = [t for t in build_tasks(seeds) if t.stage in stages]
    # pairing-friendly order: stage, seed, task, arm (all arms of a task are adjacent)
    return sorted(((t.stage, t.seed, t.task_id, a, t) for t in tasks for a in arms), key=lambda x: x[:4])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-id", default="mock")
    ap.add_argument("--model-key", help="35b | 27b | flash: the models_v2.py row; sets --model-id (the three models are run one at a time)")
    ap.add_argument("--url")
    ap.add_argument("--embed-url")
    ap.add_argument("--embed-id", default="bge-large-en-v1.5")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--arms", default=",".join(ARMS_V1))
    ap.add_argument("--arm-set", choices=("v1", "v2"), help="v2 = A,B,A_pick,B1,B2,C3 (overrides --arms)")
    ap.add_argument("--stages", default="1,2,3")
    ap.add_argument("--limit", type=int, default=0, help="smoke: stop after N units")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="record mode=smoke (never counts toward the live row)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--work", default="/mnt/raid0/llm/tmp/codemode-ab-20261117/work")
    a = ap.parse_args(argv)
    if a.model_key:
        from scripts.benchmark.codemode_ab.models_v2 import resolve

        try:
            a.model_id = resolve(a.model_key)["model_id"]
        except KeyError as exc:
            print(exc, file=sys.stderr)
            return 3
    seeds = [int(x) for x in a.seeds.split(",")]
    arms = tuple(x for x in a.arms.split(",") if x)
    if a.arm_set == "v2":
        arms = ("A", "B") + ARMS_V2
    bad = [x for x in arms if x not in ARMS]
    if bad:
        print(f"unknown arm(s): {bad}; known: {ARMS}", file=sys.stderr)
        return 3
    stages = [int(x) for x in a.stages.split(",")]
    units = plan(seeds, arms, stages)
    if a.dry_run:
        by = {}
        for st, _s, _tid, arm, t in units:
            by.setdefault(f"stage{st}", {}).setdefault(t.family, 0)
            by[f"stage{st}"][t.family] += 1
        print(json.dumps({"dry_run": True, "units": len(units), "arms": arms, "seeds": seeds, "per_stage_family_units": by,
                          "k_selected": K_SELECTED}, indent=1))
        return 0
    out = Out(Path(a.out), "smoke" if (a.smoke or a.mock) else "full")
    if not a.mock and not (a.url and a.embed_url):
        print("need --url and --embed-url (or --mock)", file=sys.stderr)
        return 3
    work = Path(a.work)
    if not str(work.resolve()).startswith(SCRATCH):
        print(f"--work must be under {SCRATCH}", file=sys.stderr)
        return 3
    stopf = os.environ.get("FILLER_STOPFILE", "")
    deadline = int(os.environ.get("FILLER_DEADLINE", "0"))
    template = World(work / "template", ORCH_ROOT)
    os.environ.setdefault("ORCHESTRATOR_TOOL_SELECTOR_LOG", str(out.dir / "picker_events.jsonl"))
    selectors = build_selectors(a, template, out)
    register_native_picker(selectors, template)
    done = out.done()
    cat_uid = f"{a.model_id}|catalog"
    if not any(json.loads(ln).get("unit_id") == cat_uid for ln in (out.res.read_text().splitlines() if out.res.exists() else []) if ln.strip()):
        from scripts.benchmark.codemode_ab.arms import static_tool_names

        td = template.tool_dicts()
        out.append({"kind": "catalog", "unit_id": cat_uid, "model": a.model_id, "n_tools": len(td), "n_synthetic": len(template.synthetic),
                    "n_real": len(td) - len(template.synthetic) - 6, "static_tools": static_tool_names(template),
                    "schema_json_chars": len(json.dumps(td)), "n_groups": len(template.groups)})
    last_stage, n_run = None, 0
    for st, seed, tid, arm, task in units:
        if last_stage is not None and st != last_stage:
            if gate(out, last_stage, a.model_id):
                return 0
        last_stage = st
        uid = f"{a.model_id}|{arm}|{tid}"
        if uid in done:
            continue
        if (stopf and os.path.exists(stopf)) or (deadline and time.time() > deadline - 30):
            return 75
        world = World(work / "units" / f"{'m' if a.mock else 'r'}_{a.model_id}_{arm}_{tid}", ORCH_ROOT)
        if a.mock:
            from scripts.benchmark.codemode_ab.mock import MockLLM

            llm = MockLLM(task, world, seed)
        else:
            llm = selectors["llm"]
        try:
            rec = run_episode(task, arm, world, llm, seed * 1000, selectors=selectors)
        except Exception as exc:  # noqa: BLE001 - a harness crash is a record, never a silent skip
            rec = {"arm": arm, "task_id": tid, "family": task.family, "n": task.n, "seed": seed, "tier": task.tier, "stage": st,
                   "success": False, "in_tokens": 0, "out_tokens": 0, "end_reason": f"llm_error:harness:{type(exc).__name__}:{exc}"[:200],
                   "wall_s": 0.0, "turns": 0, "failed_snippet_rate": 0.0, "param_accuracy": None, "duplicate_writes": 0}
        tr = rec.pop("_transcript", None)
        if tr is not None:  # every episode gets a transcript file; a harness crash (no transcript) is caught by smoke_check_v2
            rec["transcript_path"] = out.transcript(uid, tr)
        out.append({"kind": "episode", "unit_id": uid, "model": a.model_id, **rec})
        n_run += 1
        if a.limit and n_run >= a.limit:
            return 0
    if last_stage is not None and gate(out, last_stage, a.model_id, final=True):
        return 0
    (out.dir / "TERMINAL.json").write_text(json.dumps({"state": "complete", "mode": out.mode, "at": now()}))
    out.append({"kind": "terminal", "unit_id": f"{a.model_id}|terminal", "state": "complete"})
    return 0


def register_native_picker(selectors: dict, template: World) -> None:
    """Arm A_pick runs the PRODUCTION native picker: register the C1 embedder as its provider and warm the memoised selector once
    (cold document embedding must not eat the 500 ms budget of the first episode)."""
    emb = selectors.get("embedder")
    if emb is None:
        return
    from src.prompt_builders import tool_selector as ts
    from scripts.benchmark.codemode_ab.arms import openai_tool

    ts.set_native_selector_provider(ts.make_native_provider("embed", embedder=emb))
    from src.prompt_builders.tool_picker import memoised_embedding_selector

    dicts = [d for d in (ts.openai_tool_to_dict(openai_tool(t)) for t in template.tool_dicts()) if d]
    memoised_embedding_selector(emb, dicts)


def build_selectors(a: argparse.Namespace, template: World, out: Out) -> dict:
    from scripts.benchmark.codemode_ab.llm import HttpEmbedder, HttpLLM
    from src.prompt_builders.tool_selector import EmbeddingToolSelector, OneShotToolSelector

    tools = template.tool_dicts()
    if a.mock:
        from scripts.benchmark.codemode_ab.mock import MockEmbedder, MockPrimitives

        kw = {"files": ["director", "file", "json", "csv", "read"], "data_formats": ["json", "csv"], "catalog": ["price", "sku"],
              "ledger": ["ledger", "post"], "math_stats": ["statistics", "mean"]}
        emb = MockEmbedder()
        return {"C1": EmbeddingToolSelector(emb, tools), "embedder": emb,
                "C2": OneShotToolSelector(MockPrimitives(kw), tools, template.groups, GROUP_DESC), "llm": None}
    llm = HttpLLM(a.url, a.model_id)
    emb = HttpEmbedder(a.embed_url, a.embed_id)
    return {"C1": EmbeddingToolSelector(emb, tools), "embedder": emb,
            "C2": OneShotToolSelector(llm, tools, template.groups, GROUP_DESC), "llm": llm}


def gate(out: Out, stage: int, model: str, final: bool = False) -> bool:
    """Interim look after ``stage``; True = a predeclared stop fired (TERMINAL written)."""
    recs = score.load(out.res) if out.mode == "full" else [r for r in _smoke_records(out)]
    scorer = score_v2 if any(r.get("arm") in ARMS_V2 for r in recs) else score
    res = scorer.evaluate(recs, stage)
    gate_rec = {"kind": "gate", "unit_id": f"{model}|gate|stage{stage}", "stage": stage, "interim": res["interim"], "final": final}
    out.append(gate_rec)
    if final or stage >= 3:
        return False
    look = res["interim"]["n20"] if stage == 1 else res["interim"]["n50"]
    reason = None
    if look["stop_on_futility"]:
        reason = f"stop_on_futility_stage{stage}"
    elif stage == 2 and look.get("stop_on_answer_descriptive"):
        reason = "stop_on_answer_n50_descriptive_interim"
    if reason:
        (out.dir / "TERMINAL.json").write_text(json.dumps({"state": reason, "mode": out.mode, "interim": look, "at": now()}))
        out.append({"kind": "terminal", "unit_id": f"{model}|terminal", "state": reason})
        return True
    return False


def _smoke_records(out: Out) -> list[dict]:
    rs = []
    for ln in out.res.read_text().splitlines():
        try:
            r = json.loads(ln)
        except ValueError:
            continue
        if r.get("kind") == "episode" and r.get("mode") == out.mode:
            rs.append(r)
    return rs


if __name__ == "__main__":
    sys.exit(main())
