"""HIL-6j: the entrypoint for the HIL-6i first-feature pilot, and for nothing else.

**Scope.** On 2026-10-22 the operator lifted the 2026-09-16 "no loop run" deferment *for the HIL-6
pilot only* (decision tool, coordinator session 3c4ca599, "Lift for this pilot only"). This module
is that lift's code. It runs ONE design, ``pilots.first_feature`` (HS-4 §3.5, the architect
delegation prompt); every other design id is refused. It has no scheduler, cron, AutoPilot or
generic hook: a calendar-granted filler job invokes it, one window at a time::

    python -m src.harness_loop.pilot_run --design hil6i-first-feature \\
        --operator-lift 2026-10-22-pilot-only --run-dir DIR --phase preflight|freeze|run|status ...

**Refusals** (checked before anything else, every phase that can reach a model):

* ``WINDOW_CONFIRMED=1`` is not set (only a calendar grant may set it, compute-fillers.md);
* the ``--operator-lift`` token is not exactly :data:`LIFT_TOKEN`;
* ``--design`` is not :data:`PILOT_DESIGN_ID` (or the feature id it names);
* the pilot validator fails: ``pilot.unmet_requirements`` is non-empty for either cycle's design,
  or ``pilot.CHECKLIST`` does not cover HIL-2 items 1-22.

**Resumable per cycle and per arm.** Every ``/chat`` response is cached under the run directory,
keyed by the arm's content sha256, the item and the run index; a proposal, a self-judged choice
and each judge chunk are cached too. Re-running ``--phase run`` after a stop replays cached work
for free and continues at the first missing item. Steps that completed are recorded in
``state.json`` and in the LOOP journal (``hil6j_step_done``); the attempt ledger resumes an
interrupted attempt instead of reserving its alpha twice (``ResumingLedger``). The STOP file and
the deadline are checked before every uncached request; a stop raises :class:`PilotStopped`.

**Isolation.** Every ``/chat`` request goes through the driver's evaluator, which arms the AP-54
fence (``substrate.fenced_eval_request``) and refuses any echo that is not ``active``. Arms are
byte-pinned prompt trees under the private API's prompt-root base; the live prompt file is never
written. While ``--phase run`` is active, AutoPilot's PromptForge is told to leave the target
alone (``src.autopilot_core.mutation_exclusions``); the exclusion is set at the start of the phase
and cleared in ``finally`` (the filler job also clears it in its EXIT trap).

Every number this produces is an **observation**: ``protocol_id`` stays ``None`` (MEASUREMENT.md).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
import time
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from src.harness_loop import substrate
from src.harness_loop.attempt_ledger import CLOSE as LEDGER_CLOSE, OPEN as LEDGER_OPEN, AttemptLedger
from src.harness_loop.chat_evaluator import ChatArmEvaluator, ItemScore, pass_share
from src.harness_loop.credibility_judge import (
    CROSS_FAMILY_JUDGE,
    PINNED_JUDGE,
    CredibilityMatrixJudge,
    cloud_cli_transport,
)
from src.harness_loop.driver import CycleOutcome, CycleSpec, SiblingLoopDriver
from src.harness_loop.editor import LocalEditorProposer, ReferenceEditorProposer
from src.harness_loop.interfaces import (
    ArmResult,
    ArmSpec,
    Candidate,
    Decision,
    ProposalResult,
    ProposerView,
    SplitRef,
    TokenCount,
    TraceRef,
    VerdictMatrix,
    sha256_text,
    sign_test_p,
)
from src.harness_loop.pilot import CHECKLIST, unmet_requirements
from src.harness_loop.pilots import first_feature as ff
from src.harness_loop.report import adoption_readout, power_readout
from src.harness_loop.transport import (
    CHAT_PATH,
    V1_CHAT_PATH,
    EgressClearance,
    JsonTransport,
    TransportError,
    http_json_transport,
    openai_message_text,
)

# ── scope ─────────────────────────────────────────────────────────────────────

PILOT_DESIGN_ID = "hil6i-first-feature"
ACCEPTED_DESIGN_IDS = frozenset({PILOT_DESIGN_ID, ff.FEATURE_ID})
LIFT_TOKEN = "2026-10-22-pilot-only"
LIFT_RECORD = (
    "Operator decision 2026-10-22 (decision tool, coordinator 3c4ca599): lift the 2026-09-16 "
    "'no loop run' deferment for the HIL-6 pilot only ('Lift for this pilot only')."
)
EXCLUSION_OWNER = "hil6j-pilot"
JOURNAL_STEP_EVENT = "hil6j_step_done"

EXIT_OK = 0
EXIT_REFUSED = 2
EXIT_REFREEZE = 3
EXIT_BLOCKED = 5
EXIT_STOPPED = 75

LINEAGES = ff.LINEAGES  # ("local_editor", "reference_editor")
REFERENCE_EDITOR_JUDGE_ENTRY = CROSS_FAMILY_JUDGE  # cloud_judges.yaml entry whose model is gpt-5.6-luna
JUDGE_ENTRY = PINNED_JUDGE  # sonnet-5-5-low
JUDGE_CHUNK = 21  # validation instances per judge call (420 / 21 = 20 calls per matrix)
TRACE_CAP_PER_STRATUM = 8  # proposer index: at most 40 trajectories
EXCERPT_CHARS = 600
MATCHED_ROLLOUT_K = 8  # item 1: 2 cycles x 2 candidates x 2 runs
LOOP_BUDGET = 3  # architect max_loops (final-result selection rule, item 14)
EGRESS_SUITE_ID = "hil6i-five-strata"
FREEZE_SEEN_WINDOW_DAYS = 60

#: Mean seconds per architect-delegated /chat request by scoring method (the strata are named by
#: their scoring method here). Sources: March-April 2026 delegated /chat medians (f1 18.8 s,
#: exact-match 18.2 s, code execution 224 s; research benchmarks/results/eval/3way_2026*.jsonl,
#: an older architect model) and the 2026-09-23 Flash-Next direct CPU bench (multiple choice
#: 8.0 s/item; artifacts/architect-bench-cpu-20260923). Means are set above those medians because
#: p90s are 2-5x the median. An ESTIMATE: no delegated /chat latency exists for Flash-Next.
SECONDS_PER_REQUEST = {
    "multiple_choice": 15.0,
    "f1": 30.0,
    "exact_match": 28.0,
    "substring": 28.0,
    "code_execution": 180.0,
}
#: The stratum order of ff.STRATA, as scoring methods, for the estimate (math mixes exact/substring).
STRATUM_METHODS = ("multiple_choice", "f1", "exact_match", "exact_match", "code_execution")
STACK_BRINGUP_S = 15 * 60  # load the three private servers + API, own-PID proofs, preflight


class PilotRefused(RuntimeError):
    """A precondition of the operator's lift is not met; nothing was started."""


class PilotBlocked(RuntimeError):
    """A named input is missing (e.g. the operator's egress clearance); nothing was evaluated."""


class PilotStopped(Exception):
    """The STOP file appeared or the deadline passed; the run resumes from its cache."""


class RefreezeRequired(RuntimeError):
    """The live target changed after freeze: candidates would be stale (design §2)."""


# ── refusals ──────────────────────────────────────────────────────────────────


def refusal_reasons(*, design: str, lift: str | None, env: Mapping[str, str]) -> list[str]:
    out: list[str] = []
    if env.get("WINDOW_CONFIRMED") != "1":
        out.append("WINDOW_CONFIRMED=1 is not set (only a calendar-granted window may set it)")
    if lift != LIFT_TOKEN:
        out.append(f"--operator-lift must be {LIFT_TOKEN!r} (the 2026-10-22 pilot-only lift)")
    if design not in ACCEPTED_DESIGN_IDS:
        out.append(f"design {design!r} is not the HIL-6i pilot; the lift covers {PILOT_DESIGN_ID!r} only")
    return out


def validator_reasons(splits: ff.FrozenSplits,
                      unmet_fn: Callable[[Any], list[tuple[str, str]]] | None = None) -> list[str]:
    """The HIL-2 items 1-22 CHECKLIST plus the HIL-6 pilot requirements, for both cycle designs."""
    unmet_fn = unmet_fn or unmet_requirements
    out: list[str] = []
    missing = sorted(set(range(1, 23)) - set(CHECKLIST))
    if missing:
        out.append(f"pilot.CHECKLIST does not cover HIL-2 items {missing}")
    for cycle in (1, 2):
        for key, why in unmet_fn(ff.first_feature_design(splits, cycle=cycle)):
            out.append(f"cycle {cycle} design unmet {key}: {why}")
    return out


# ── paths and small persistence helpers ───────────────────────────────────────


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, default=_jsonable) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    return str(obj)


def _load(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default


@dataclass(frozen=True)
class RunPaths:
    root: Path

    @property
    def freeze(self) -> Path:
        return self.root / "freeze.json"

    @property
    def items(self) -> Path:
        return self.root / "items.jsonl"

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    @property
    def state(self) -> Path:
        return self.root / "state.json"

    @property
    def cache(self) -> Path:
        return self.root / "cache" / "chat.jsonl"

    @property
    def texts(self) -> Path:
        return self.root / "texts"

    @property
    def proposals(self) -> Path:
        return self.root / "proposals"

    @property
    def judge(self) -> Path:
        return self.root / "judge"

    @property
    def journal(self) -> Path:
        return self.root / "loop-journal"

    @property
    def results(self) -> Path:
        return self.root / "results"

    @property
    def report(self) -> Path:
        return self.root / "report.json"


class TextStore:
    """Arm bytes by sha256 (``texts/<sha>.md``): incumbent, control and every candidate."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, text: str) -> str:
        sha = sha256_text(text)
        p = self.root / f"{sha}.md"
        if not p.exists():
            p.write_text(text, encoding="utf-8")
        return sha

    def get(self, sha: str) -> str:
        text = (self.root / f"{sha}.md").read_text(encoding="utf-8")
        if sha256_text(text) != sha:
            raise ValueError(f"text store entry {sha} is corrupt")
        return text


class ChatCache:
    """Append-only, fsynced JSONL of every /chat outcome, keyed by ``request_id``."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._rows: dict[str, dict[str, Any]] = {}
        if path.exists():
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        continue  # torn tail of an interrupted append; the item is re-requested
                    self._rows[rec["request_id"]] = rec

    def get(self, rid: str) -> dict[str, Any] | None:
        return self._rows.get(rid)

    def put(self, rid: str, rec: Mapping[str, Any]) -> None:
        row = {"request_id": rid, **rec}
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        self._rows[rid] = row

    def __len__(self) -> int:
        return len(self._rows)


_KEEP = ("answer", "eval_fence", "tokens_generated", "tokens_used", "delegation_diagnostics",
         "routed_to", "mode", "error", "error_code")


def slim_response(resp: Mapping[str, Any]) -> dict[str, Any]:
    out = {k: resp.get(k) for k in _KEEP if k in resp}
    events = resp.get("delegation_events")
    if isinstance(events, list):
        out["delegation_events_n"] = len(events)
        out["delegation_roles"] = sorted({str(e.get("to_role") or e.get("role") or "")
                                          for e in events if isinstance(e, Mapping)} - {""})
    return out


def classify_infra(exc: BaseException) -> str:
    msg = str(exc).lower()
    if "timed out" in msg or "timeout" in msg:
        return "timeout"
    return "backend_outage"


def cached_transport(inner: JsonTransport, cache: ChatCache, should_stop: Callable[[], bool],
                     clock: Callable[[], float] = time.monotonic) -> JsonTransport:
    """Replays cached outcomes; checks STOP/deadline before every uncached request."""

    def call(body: Mapping[str, Any]) -> Mapping[str, Any]:
        rid = str(body["request_id"])
        hit = cache.get(rid)
        if hit is None:
            if should_stop():
                raise PilotStopped(f"stop requested before {rid}")
            t0 = clock()
            try:
                resp = inner(body)
            except TransportError as exc:
                cache.put(rid, {"error": str(exc)[:500], "infra": classify_infra(exc),
                                "elapsed_s": round(clock() - t0, 3), "at": _now()})
                raise
            cache.put(rid, {"response": slim_response(resp), "elapsed_s": round(clock() - t0, 3),
                            "at": _now()})
            hit = cache.get(rid)
        if "error" in hit:
            raise TransportError(hit["error"])
        return hit["response"]

    return call


# ── freeze ────────────────────────────────────────────────────────────────────


def iter_pool(path: Path) -> Iterable[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("__pool_metadata__"):
                continue
            yield row


def pool_metadata(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        first = json.loads(fh.readline())
    return {k: first.get(k) for k in ("generated_at", "generator", "total_questions", "corpus_version")
            if k in first}


def gold_self_check_exclusions(rows: Iterable[Mapping[str, Any]], orch_root: Path) -> dict[str, str]:
    """EV-19 outcomes ``wrong`` / ``scorer_error`` per qid, computed in-process (zero inference)."""
    bench = str(orch_root / "scripts" / "benchmark")
    if bench not in sys.path:
        sys.path.insert(0, bench)
    from seeding_scoring import score_answer_or_error  # type: ignore

    from src.autopilot_core.gold_self_check import classify

    out: dict[str, str] = {}
    for row in rows:
        outcome, detail = classify(row, score_answer_or_error)
        if outcome in ("wrong", "scorer_error"):
            out[str(row["id"])] = f"{outcome}:{detail}"
    return out


def core_v2_ids(path: Path) -> set[str]:
    out: set[str] = set()
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("__core_metadata__"):
            continue
        qid = row.get("id") or row.get("question_id")
        if qid:
            out.add(str(qid))
    return out


_ID_RE = re.compile(r'"(?:id|question_id|prompt_id)"\s*:\s*"([^"]{1,200})"')
SEEN_SCAN_SUFFIXES = (".jsonl", ".json")
SEEN_SCAN_MAX_BYTES = 512 << 20


def seen_since(roots: Sequence[Path], pool_ids: set[str], cutoff_epoch: float) -> tuple[set[str], list[str]]:
    """Pool ids named in any eval artifact modified at or after ``cutoff_epoch``.

    The declared freeze rule for "qids seen in any evaluation within 60 days" (no host ledger
    defines "seen"; P-QUAL-PROMO names none). Over-inclusive on purpose: a file's mtime stands in
    for the date of every id it names. Returns the ids and the files that contributed.
    """
    seen: set[str] = set()
    files: list[str] = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.suffix not in SEEN_SCAN_SUFFIXES or not path.is_file():
                continue
            try:
                st = path.stat()
            except OSError:
                continue
            if st.st_mtime < cutoff_epoch or st.st_size > SEEN_SCAN_MAX_BYTES:
                continue
            hit = False
            try:
                with open(path, encoding="utf-8", errors="ignore") as fh:
                    for line in fh:
                        for qid in _ID_RE.findall(line):
                            if qid in pool_ids:
                                seen.add(qid)
                                hit = True
            except OSError:
                continue
            if hit:
                files.append(str(path))
    return seen, files


def freeze(paths: RunPaths, *, rows: Sequence[Mapping[str, Any]], exclusions: Mapping[str, str],
           control_text: str, live_target_sha: str, pool_meta: Mapping[str, Any],
           extra: Mapping[str, Any] | None = None, frozen_at: str | None = None) -> dict[str, Any]:
    """Draw the splits once (pure ``build_splits``), pin the control bytes, write ``freeze.json``."""
    if paths.freeze.exists():
        raise RefreezeRequired(f"{paths.freeze} exists; a re-freeze is a new run directory")
    frozen_at = frozen_at or _now()
    splits = ff.build_splits(rows, exclude_qids=exclusions.keys(), frozen_at=frozen_at)
    wanted = splits.train.qids | splits.validation.qids | splits.arrival.qids | splits.arrival_2.qids | splits.test.qids
    texts = TextStore(paths.texts)
    control_sha = texts.put(control_text)
    paths.root.mkdir(parents=True, exist_ok=True)
    with open(paths.items, "w", encoding="utf-8") as fh:
        for row in rows:
            if row.get("id") in wanted:
                fh.write(json.dumps(row, sort_keys=True) + "\n")
    doc = {
        "schema": "epyc.orchestrator.hil6j_pilot_freeze.v1",
        "design": PILOT_DESIGN_ID,
        "feature_id": ff.FEATURE_ID,
        "frozen_at": frozen_at,
        "seed": splits.seed,
        "manifest_sha256": splits.manifest_sha256,
        "splits": {name: {"split_id": s.split_id, "role": s.role, "qids": sorted(s.qids)}
                   for name, s in (("train", splits.train), ("validation", splits.validation),
                                   ("arrival", splits.arrival), ("arrival_2", splits.arrival_2),
                                   ("test", splits.test))},
        "control_sha256": control_sha,
        "control_sha256_at_design": ff.TARGET_SHA256_AT_DESIGN,
        "live_target_sha256": live_target_sha,
        "exclusions": {"n": len(exclusions), "by_reason": dict(Counter(v.split(":")[0] for v in exclusions.values()))},
        "pool": dict(pool_meta),
        **(dict(extra) if extra else {}),
    }
    _dump(paths.freeze, doc)
    return doc


def load_frozen(paths: RunPaths, state: Mapping[str, Any]) -> ff.FrozenSplits:
    doc = _load(paths.freeze)
    if doc is None:
        raise PilotBlocked("no freeze.json: run --phase freeze first")
    released = state.get("released", {})

    def ref(name: str) -> SplitRef:
        s = doc["splits"][name]
        arrived = doc["frozen_at"] if name in ("train", "validation", "test") else released.get(name, "")
        return SplitRef(s["split_id"], s["role"], frozenset(s["qids"]), arrived_at=arrived)

    return ff.FrozenSplits(train=ref("train"), validation=ref("validation"), arrival=ref("arrival"),
                           arrival_2=ref("arrival_2"), test=ref("test"), seed=doc["seed"],
                           manifest_sha256=doc["manifest_sha256"])


def load_items(paths: RunPaths) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with open(paths.items, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            out[str(row["id"])] = row
    return out


# ── arms: prompt trees, requests, scoring ─────────────────────────────────────


def materialize_arm(base: Path, template_dir: Path, target_rel: str, text: str) -> Path:
    """A full copy of the prompt tree with the target replaced, at ``<base>/hil6j-<sha12>``.

    The resolver falls back to family/fallback constants (not the live tree) for files a root
    lacks, so an arm root must carry every prompt, not only the target. Idempotent; verified.
    """
    sha = sha256_text(text)
    root = base / f"hil6j-{sha[:12]}"
    target = root / Path(target_rel).name
    if target.exists() and sha256_text(target.read_text(encoding="utf-8")) == sha:
        return root
    tmp = base / f".hil6j-{sha[:12]}.{os.getpid()}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    shutil.copytree(template_dir, tmp, symlinks=False)
    (tmp / Path(target_rel).name).write_text(text, encoding="utf-8")
    if root.exists():
        shutil.rmtree(root)
    os.replace(tmp, root)
    return root


def item_key(qid: str, run: int) -> str:
    return f"{qid}@r{run}"


def split_key(key: str) -> tuple[str, int]:
    qid, _, r = key.rpartition("@r")
    return qid, int(r)


def request_id(prefix: str, sha: str, key: str) -> str:
    rid = f"hil6j-{prefix}-{sha[:12]}-{key}"
    if len(rid) > 128:
        rid = f"hil6j-{prefix}-{sha[:12]}-{hashlib.sha256(key.encode()).hexdigest()[:24]}"
    return rid


class _RunItems(Mapping[str, Mapping[str, Any]]):
    def __init__(self, rows: Mapping[str, Mapping[str, Any]]) -> None:
        self._rows = rows

    def __getitem__(self, key: str) -> Mapping[str, Any]:
        qid, run = split_key(key)
        return {**self._rows[qid], "_key": key, "_run": run}

    def __iter__(self):
        return iter(())

    def __len__(self) -> int:
        return len(self._rows)

    def get(self, key, default=None):
        try:
            return self[key]
        except (KeyError, ValueError):
            return default


def score_item(item: Mapping[str, Any], response: Mapping[str, Any],
               score_fn: Callable[[str, str, dict], bool]) -> ItemScore:
    """Pass/fail by the vendored pool scorer; treatment fired iff phase A of the delegated
    architect path ran (``delegation_diagnostics.phases_count >= 1``) under the arm's root."""
    answer = str(response.get("answer") or "")
    passed = bool(score_fn(answer, str(item.get("expected") or ""), dict(item))) if answer.strip() else False
    diag = response.get("delegation_diagnostics") or {}
    fired = isinstance(diag, Mapping) and int(diag.get("phases_count") or 0) >= 1
    return ItemScore(passed=passed, fired=fired)


def add_code_exec_path(research_root: Path) -> None:
    """The vendored scorer imports ``code_exec_scorer`` lazily; it lives in the research repo's
    ``scripts/benchmark`` (stdlib only). Appended, so it can never shadow an orchestrator module."""
    bench = str(Path(research_root) / "scripts" / "benchmark")
    if bench not in sys.path:
        sys.path.append(bench)


def default_score_fn(orch_root: Path, research_root: Path | None = None) -> Callable[[str, str, dict], bool]:
    bench = str(orch_root / "scripts" / "benchmark")
    if bench not in sys.path:
        sys.path.insert(0, bench)
    if research_root is not None:
        add_code_exec_path(research_root)
    from scripts.benchmark.answer_scoring_vendored import score_response

    return score_response


@dataclass(frozen=True)
class RunMeanSignTest:
    """Design §6 + the pinned ``ff.DECISION_RULE``: each item's score is the mean over its runs
    (0, 1/2 or 1); a win is an item whose candidate mean exceeds the incumbent's; the exact
    one-sided sign test at the attempt's alpha with the declared win-share floor decides.
    Items with no scored run on either arm are left out. Ties go to the incumbent (item 17a)."""

    alpha: float = ff.DECISION_RULE.alpha
    min_win_share: float = ff.DECISION_RULE.min_win_share

    @staticmethod
    def item_means(result: ArmResult) -> dict[str, float]:
        acc: dict[str, list[float]] = {}
        for key, ok in result.item_outcomes.items():
            if ok is None:
                continue
            qid, _ = split_key(key) if "@r" in key else (key, 1)
            acc.setdefault(qid, []).append(1.0 if ok else 0.0)
        return {q: sum(v) / len(v) for q, v in acc.items()}

    def decide(self, incumbent: ArmResult, candidate: ArmResult) -> Decision:
        if incumbent.metric != candidate.metric or incumbent.direction != candidate.direction:
            return Decision(False, "metric_mismatch")
        a, b = self.item_means(incumbent), self.item_means(candidate)
        common = sorted(set(a) & set(b))
        if not common:
            return Decision(False, "no_paired_items")
        wins = sum(1 for q in common if b[q] > a[q])
        losses = sum(1 for q in common if a[q] > b[q])
        d = wins + losses
        p = sign_test_p(wins, d)
        share = wins / d if d else 0.0
        detail = f"paired_wins:{wins}/{d} n:{len(common)} p:{p:.3g} alpha:{self.alpha:g}"
        if d and p <= self.alpha and share >= self.min_win_share:
            return Decision(True, f"candidate_better:{detail}")
        return Decision(False, f"incumbent_holds:{detail}")


class PilotEvaluator:
    """Implements ``Evaluator``. Runs each arm ``runs`` times per item through the fenced
    ``ChatArmEvaluator`` (item keys ``<qid>@r<n>``) and applies the design's primary slice: an
    infra failure counts as a failure, classified by reason in :attr:`infra`."""

    def __init__(self, *, transport: JsonTransport, cache: ChatCache, items: Mapping[str, Mapping[str, Any]],
                 texts: TextStore, root_for: Callable[[str], Path], server_urls: Mapping[str, str],
                 score_fn: Callable[[str, str, dict], bool], runs: int = ff.RUNS,
                 timeout_s: int = 900, prefix: str = "arm", allow_delegation: bool = True) -> None:
        self.cache = cache
        self.items = items
        self.texts = texts
        self.root_for = root_for
        self.server_urls = dict(server_urls)
        self.runs = runs
        self.timeout_s = int(timeout_s)
        self.prefix = prefix
        self.allow_delegation = allow_delegation
        self.infra: dict[tuple[str, str], dict[str, str]] = {}
        self._inner = ChatArmEvaluator(
            transport=transport, items=_RunItems(items), request_builder=self.build,
            scorer=lambda item, resp: score_item(item, resp, score_fn),
            content_for=lambda arm: texts.get(arm.content_sha256),
            metric=ff.PRIMARY_METRIC, unit=ff.PRIMARY_UNIT, direction=ff.PRIMARY_DIRECTION)

    def build(self, arm: ArmSpec, arm_text: str, item: Mapping[str, Any]) -> dict[str, Any]:
        body = {
            "prompt": item["prompt"],
            "context": item.get("context") or "",
            "real_mode": True,
            "mock_mode": False,
            "force_role": ff.CONSUMER_ROLE,
            "allow_delegation": self.allow_delegation,
            "x_orchestrator_prompt_root": str(self.root_for(arm.content_sha256)),
            "workload_class": "eval_batch",
            "request_id": request_id(self.prefix, arm.content_sha256, item["_key"]),
            "server_urls": dict(self.server_urls),
            "timeout_s": self.timeout_s,
        }
        if self.allow_delegation:
            body["force_mode"] = "delegated"
        return body

    def evaluate(self, arm: ArmSpec, split: SplitRef, runs: int | None = None) -> ArmResult:
        n = runs or self.runs
        expanded = SplitRef(split.split_id, split.role,
                            frozenset(item_key(q, r) for q in split.qids for r in range(1, n + 1)),
                            arrived_at=split.arrived_at, labels_resolved_at=split.labels_resolved_at)
        res = self._inner.evaluate(arm, expanded)
        diag = self._inner.last_diagnostics
        errors = dict(diag.transport_errors) if diag else {}
        self.infra[(arm.arm_id, split.split_id)] = {k: classify_infra(TransportError(v)) for k, v in errors.items()}
        if errors:
            outcomes = dict(res.item_outcomes)
            for k in errors:
                outcomes[k] = False  # primary slice: infra failures count as failures (P-AB-1)
            res = replace(res, item_outcomes=outcomes, value=pass_share(outcomes),
                          scored=sum(1 for v in outcomes.values() if v is not None))
        return res

    def answer(self, sha: str, qid: str, run: int = 1) -> str:
        hit = self.cache.get(request_id(self.prefix, sha, item_key(qid, run))) or {}
        return str((hit.get("response") or {}).get("answer") or "")


# ── proposers, judge and the editor transports ────────────────────────────────


def qwen_editor_transport(inner: JsonTransport) -> JsonTransport:
    """compute-fillers rule 15: every chat request to a Qwen3.6+ editor disables thinking unless
    pre-registered, and an empty reply fails loudly instead of parsing as no candidates."""

    def call(body: Mapping[str, Any]) -> Mapping[str, Any]:
        payload = dict(body)
        payload.setdefault("chat_template_kwargs", {"enable_thinking": False})
        resp = inner(payload)
        text = openai_message_text(resp) if isinstance(resp, Mapping) else None
        if not (text or "").strip():
            raise TransportError("local editor returned empty content (thinking on, or no reply)")
        return resp

    return call


EDITOR_REPLY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"text": {"type": "string"}, "rationale": {"type": "string"}},
                "required": ["text", "rationale"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["candidates"],
    "additionalProperties": False,
}


def codex_editor_transport(spec: Any, model: str, run_fn: Any = None) -> JsonTransport:
    """The reference editor over the codex CLI (cloud_judges ``codex_exec``): an OpenAI-shaped
    request in, an OpenAI-shaped response out, schema-bound. Usage is not exposed (``None``)."""

    def call(body: Mapping[str, Any]) -> Mapping[str, Any]:
        from src.typed_decisions import cloud_judges

        prompt = "\n\n".join(str(m.get("content") or "") for m in body.get("messages") or ())
        try:
            res = cloud_judges.run_cloud_judge(spec, prompt, EDITOR_REPLY_SCHEMA, model=model, run_fn=run_fn)
        except cloud_judges.CloudJudgeError as exc:
            raise TransportError(str(exc)) from exc
        return {"choices": [{"message": {"content": json.dumps(res.payload)}}], "usage": None,
                "resolved_model": res.resolved_model}

    return call


class CachedProposer:
    """A proposer whose proposal for one (lineage, cycle, view) is computed once and replayed."""

    def __init__(self, inner: Any, path: Path, texts: TextStore) -> None:
        self.inner = inner
        self.model_id = inner.model_id
        self.path = path
        self.texts = texts

    def propose(self, view: ProposerView, *, max_candidates: int) -> ProposalResult:
        key = sha256_text(json.dumps([view.incumbent_text, view.selection_rule,
                                      [dataclasses.asdict(t) for t in view.traces],
                                      list(view.rejected_history), max_candidates]))
        doc = _load(self.path)
        if doc is None or doc.get("view_sha256") != key:
            res = self.inner.propose(view, max_candidates=max_candidates)
            doc = {"view_sha256": key, "model_id": self.model_id, "at": _now(),
                   "candidates": [dataclasses.asdict(c) for c in res.candidates],
                   "tokens": [dataclasses.asdict(t) for t in res.tokens],
                   "parse": dataclasses.asdict(getattr(self.inner, "last_parse", None))
                   if getattr(self.inner, "last_parse", None) is not None else None,
                   "identity": self.inner.identity() if hasattr(self.inner, "identity") else {}}
            _dump(self.path, doc)
        cands = tuple(Candidate(**c) for c in doc["candidates"])
        for c in cands:
            self.texts.put(c.mutated_text)
        return ProposalResult(cands, tuple(TokenCount(**t) for t in doc["tokens"]))


class ChunkedJudge:
    """Implements ``Judge`` over :data:`JUDGE_CHUNK`-instance calls of the typed credibility judge
    (a 420-instance matrix does not fit one call). Every cell is the judge's own; the merged
    ``chosen`` is the majority of the chunk choices (ties: none), then the zero-LLM argmax check
    runs on the merged counts. Chunks are cached, so a resumed cycle re-asks nothing."""

    def __init__(self, inner: CredibilityMatrixJudge, cache_dir: Path, *, chunk: int = JUDGE_CHUNK) -> None:
        self.inner = inner
        self.model_id = inner.model_id
        self.cache_dir = cache_dir
        self.chunk = chunk

    def judge(self, candidates: Sequence[Candidate], instances: Sequence[str]) -> VerdictMatrix:
        cids = [c.candidate_id for c in candidates]
        cells: dict[str, dict[str, str]] = {c: {} for c in cids}
        choices: list[str | None] = []
        tokens: list[TokenCount] = []
        for i in range(0, len(instances), self.chunk):
            part = list(instances[i:i + self.chunk])
            key = sha256_text(json.dumps([cids, part]))[:20]
            path = self.cache_dir / f"chunk-{key}.json"
            doc = _load(path)
            if doc is None:
                m = self.inner.judge(candidates, part)
                doc = {"cells": {c: dict(r) for c, r in m.cells.items()}, "chosen": m.chosen,
                       "tokens": [dataclasses.asdict(t) for t in m.tokens], "at": _now()}
                _dump(path, doc)
            for c, row in doc["cells"].items():
                cells.setdefault(c, {}).update(row)
            choices.append(doc["chosen"])
            tokens.extend(TokenCount(**t) for t in doc["tokens"])
        counts = Counter(c for c in choices if c is not None)
        chosen = None
        if counts:
            top = counts.most_common()
            if len(top) == 1 or top[0][1] > top[1][1]:
                chosen = top[0][0]
        return VerdictMatrix(judge_model=self.model_id, cells=cells, chosen=chosen, tokens=tuple(tokens))


class ResumingLedger(AttemptLedger):
    """Item 21 on a resumed run: an attempt opened for the same candidate, cycle and split and
    never closed is RESUMED (its alpha was reserved once and stays reserved), not re-opened."""

    def open(self, candidate_id: str, alpha: float, **context: Any) -> int:
        closed = {int(e["attempt_index"]) for e in self._journal.ledger_events(LEDGER_CLOSE)}
        for e in self._journal.ledger_events(LEDGER_OPEN):
            if (e.get("candidate_id") == candidate_id and int(e["attempt_index"]) not in closed
                    and e.get("cycle") == context.get("cycle") and e.get("split") == context.get("split")):
                return int(e["attempt_index"])
        return super().open(candidate_id, alpha, **context)


def load_clearance(path: Path | None) -> dict[str, EgressClearance]:
    """The operator's egress clearance file: one entry per egress, covering all five strata.

    ``{"cleared_by": "...", "cleared_at": "...", "clearances": [{"egress": "<cloud_judges egress>",
    "suites": [...]}]}``. Absent file -> ``{}`` (the cloud parts refuse; content stays on host).
    """
    if path is None or not path.exists():
        return {}
    doc = json.loads(path.read_text(encoding="utf-8"))
    who = str(doc.get("cleared_by") or "")
    out: dict[str, EgressClearance] = {}
    for c in doc.get("clearances") or ():
        suites = set(c.get("suites") or ())
        if not set(ff.STRATA) <= suites:
            continue
        out[str(c["egress"])] = EgressClearance(egress=str(c["egress"]), suite_id=EGRESS_SUITE_ID, cleared_by=who)
    return out


# ── cloud live verification and the estimates ─────────────────────────────────

LIVE_PROBE_SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}},
                     "required": ["ok"], "additionalProperties": False}
LIVE_PROBE_PROMPT = 'Reply with exactly the JSON object {"ok": true}. Nothing else.'


def cloud_live_verification(registry: Mapping[str, Any], run_fn: Any = None,
                            clock: Callable[[], float] = time.time) -> list[dict[str, Any]]:
    """One tiny schema-bound call per cloud transport the pilot uses. No suite content is sent.
    The result goes into the run manifest; ``verified_live`` in the registry is never edited."""
    from src.typed_decisions import cloud_judges

    out = []
    for entry, model in ((REFERENCE_EDITOR_JUDGE_ENTRY, ff.REFERENCE_EDITOR_MODEL),
                         (JUDGE_ENTRY, ff.JUDGE_PINNED_MODEL)):
        spec = registry.get(entry)
        rec: dict[str, Any] = {"entry": entry, "model": model, "at": _now(), "ok": False}
        if spec is None:
            rec["error"] = "entry missing from cloud_judges registry"
            out.append(rec)
            continue
        rec.update(transport=spec.transport, egress=spec.egress, registry_verified_live=spec.verified_live)
        t0 = clock()
        try:
            res = cloud_judges.run_cloud_judge(spec, LIVE_PROBE_PROMPT, LIVE_PROBE_SCHEMA, model=model, run_fn=run_fn)
            rec.update(ok=res.payload.get("ok") is True, payload=res.payload, resolved_model=res.resolved_model)
        except Exception as exc:  # recorded, never raised: the manifest carries the failure
            rec["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        rec["elapsed_s"] = round(clock() - t0, 2)
        out.append(rec)
    return out


def estimate_cloud_calls() -> dict[str, int]:
    """Upper bound on cloud CLI calls over the whole pilot (printed at preflight)."""
    v = dict(ff.PER_STRATUM)["validation"] * len(ff.STRATA)
    chunks = math.ceil(v / JUDGE_CHUNK)
    out = {
        "preflight_live_verification": 2,
        "reference_editor_proposals (codex, 1 per cycle)": ff.CYCLES,
        f"judge_matrix_chunks (claude, {chunks} per lineage-cycle with survivors)": chunks * len(LINEAGES) * ff.CYCLES,
    }
    out["total_upper_bound"] = sum(out.values())
    return out


def estimate_requests() -> dict[str, Any]:
    """/chat request counts per step (upper bound: every cycle commits and every verification
    mode picks a different arm) and the wall-time ESTIMATE from :data:`SECONDS_PER_REQUEST`."""
    per = dict(ff.PER_STRATUM)
    n_s = len(ff.STRATA)
    mean_s = sum(SECONDS_PER_REQUEST[m] for m in STRATUM_METHODS) / len(STRATUM_METHODS)
    v, t1, t2, t3, test = (per["validation"] * n_s, per["train"] * n_s, per["arrival"] * n_s,
                           per["arrival_2"] * n_s, per["test"] * n_s)
    r = ff.RUNS
    lin = len(LINEAGES)
    cand = ff.MAX_CANDIDATES_PER_CYCLE
    steps = {
        "c1.traces (incumbent on T1, 1 run)": t1,
        "c1.aa (incumbent on V, A/A)": v * r,
        "c1.candidates (2 lineages x 2 x V x 2 runs)": lin * cand * v * r,
        "c1.arrival (committed on T2)": lin * t2 * r,
        "c1.diag.verification_modes (none/self_judged on T2, if different)": lin * 2 * t2 * r,
        "c1.diag.all_off (allow_delegation=False on V)": v * r,
        "c1.diag.reference_trajectory_arm (V)": v * r,
        "c1.diag.matched_rollout (V, runs 3..8)": v * (MATCHED_ROLLOUT_K - r),
        "c2.traces (lineage incumbent on T1+T2, 1 run)": lin * (t1 + t2),
        "c2.incumbents (V, A/A, if cycle 1 committed)": lin * v * r,
        "c2.candidates": lin * cand * v * r,
        "c2.arrival (committed on T3)": lin * t3 * r,
        "c2.diag.verification_modes (T3, if different)": lin * 2 * t3 * r,
        "test (2 committed + control, 2 runs)": 3 * test * r,
    }
    total = sum(steps.values())
    per_cycle = {
        "cycle1": sum(n for k, n in steps.items() if k.startswith("c1.")),
        "cycle2": sum(n for k, n in steps.items() if k.startswith("c2.")),
        "test": steps["test (2 committed + control, 2 runs)"],
    }
    lower = total - (steps["c1.diag.verification_modes (none/self_judged on T2, if different)"]
                     + steps["c2.diag.verification_modes (T3, if different)"]
                     + steps["c2.incumbents (V, A/A, if cycle 1 committed)"]
                     + lin * t1)  # cycle-2 T1 traces replay when the incumbent did not change
    return {
        "requests_by_step": steps,
        "requests_total_upper": total,
        "requests_total_lower": lower,
        "mean_s_per_request_estimate": round(mean_s, 1),
        "hours_by_cycle_upper": {k: round(n * mean_s / 3600, 1) for k, n in per_cycle.items()},
        "hours_total_upper": round(total * mean_s / 3600, 1),
        "hours_total_lower": round(lower * mean_s / 3600, 1),
        "stack_bringup_s_per_window": STACK_BRINGUP_S,
        "concurrency": "1 (architect_general serves one slot, as in production)",
    }


# ── the pilot run ─────────────────────────────────────────────────────────────


@dataclass
class Context:
    args: argparse.Namespace
    paths: RunPaths
    state: dict[str, Any]
    splits: ff.FrozenSplits
    items: dict[str, dict[str, Any]]
    texts: TextStore
    cache: ChatCache
    journal: Any
    evaluator: PilotEvaluator
    control_text: str
    pins: dict[str, Any]
    markers: tuple[str, ...]
    make_local: Callable[[], Any]
    make_reference: Callable[[], Any]
    make_judge: Callable[[], Any]
    off_evaluator: PilotEvaluator
    log: Callable[[str], None] = print

    def save(self) -> None:
        _dump(self.paths.state, self.state)

    def done(self, step: str) -> bool:
        return step in self.state.setdefault("steps", {})

    def mark(self, step: str, **info: Any) -> None:
        at = _now()
        self.state.setdefault("steps", {})[step] = {"at": at, **info}
        self.save()  # full record in state.json; the journal carries the step boundary only
        self.journal.append_ledger_event({"type": JOURNAL_STEP_EVENT, "step": step, "at": at})
        self.log(f"step done: {step}")


def arm_spec(arm_id: str, role: str, sha: str, pins: Mapping[str, Any]) -> ArmSpec:
    return ArmSpec(arm_id, role, ff.FEATURE_ID, ff.TARGET_PATH, sha, dict(pins))


def build_view(ctx: Context, incumbent_text: str, incumbent_sha: str, tune_qids: Iterable[str],
               *, extra_trace: TraceRef | None = None) -> ProposerView:
    """The item-7 index-style view: per stratum, the first :data:`TRACE_CAP_PER_STRATUM` tuning
    items in seeded order, their question excerpt and the mechanical events of the incumbent's
    first run (decision path, delegated roles, break reason). No outcome, no answer, no label."""
    by_stratum: dict[str, list[str]] = {}
    for q in sorted(tune_qids, key=lambda q: ff._order_key(ff.SPLIT_SEED, q)):
        s = ctx.items[q]["suite"]
        if len(by_stratum.setdefault(s, [])) < TRACE_CAP_PER_STRATUM:
            by_stratum[s].append(q)
    traces: list[TraceRef] = []
    for s in ff.STRATA:
        for q in by_stratum.get(s, ()):
            hit = ctx.cache.get(request_id("arm", incumbent_sha, item_key(q, 1))) or {}
            resp = hit.get("response") or {}
            diag = resp.get("delegation_diagnostics") or {}
            events = ["error" if "error" in hit else "responded"]
            if isinstance(diag, Mapping):
                events.append(f"loops:{int(diag.get('loops') or 0)}")
                if diag.get("break_reason"):
                    events.append(f"break:{diag['break_reason']}")
                if diag.get("cap_reached"):
                    events.append("loop_cap_reached")
            events += [f"delegated_to:{r}" for r in resp.get("delegation_roles") or ()]
            traces.append(TraceRef(trace_id=request_id("arm", incumbent_sha, item_key(q, 1)),
                                   role=ff.CONSUMER_ROLE, events=tuple(events),
                                   excerpt=str(ctx.items[q]["prompt"])[:EXCERPT_CHARS]))
    if extra_trace is not None:
        traces.append(extra_trace)
    rejected = substrate.render_rejections_for_proposer(ctx.paths.journal, ff.TARGET_PATH)
    return ProposerView(
        feature_id=ff.FEATURE_ID, home=ff.HOME, target_path=ff.TARGET_PATH, target_kind=ff.TARGET_KIND,
        incumbent_text=incumbent_text, traces=tuple(traces),
        selection_rule=(f"tuning-split items ordered by sha256({ff.SPLIT_SEED}, qid); the first "
                        f"{TRACE_CAP_PER_STRATUM} per stratum; events from the incumbent's first run"),
        rejected_history=(rejected,) if rejected.strip() else ())


def ensure_traces(ctx: Context, sha: str, qids: Iterable[str], split_id: str) -> None:
    """Evaluate the incumbent once (run 1) on the tuning items, so the view has trajectories."""
    tune = SplitRef(split_id, "train", frozenset(qids))
    ctx.evaluator.evaluate(arm_spec(f"trace-{sha[:12]}", "incumbent", sha, ctx.pins), tune, runs=1)


def check_live_target(ctx: Context) -> None:
    live = Path(ctx.args.live_target)
    sha = hashlib.sha256(live.read_bytes()).hexdigest() if live.exists() else ""
    frozen = _load(ctx.paths.freeze)["live_target_sha256"]
    if sha != frozen:
        raise RefreezeRequired(f"live target {live} sha {sha[:12]} != frozen {frozen[:12]}")


def power_gate(ctx: Context, aa: ArmResult) -> dict[str, Any]:
    """Design §7 step 1: measured A/A discordance replaces the assumed one before any candidate."""
    means = RunMeanSignTest.item_means(aa)
    keyed = {}
    for key, ok in aa.item_outcomes.items():
        qid, run = split_key(key)
        keyed.setdefault(qid, {})[run] = ok
    pairs = [(r.get(1), r.get(2)) for r in keyed.values() if r.get(1) is not None and r.get(2) is not None]
    psi = (sum(1 for a, b in pairs if a != b) / len(pairs)) if pairs else 0.0
    n = len(means)
    share = (sum(means.values()) / n) if n else None
    notes = []
    psi_used = psi
    if psi == 0.0:
        # Undefined replacement: the two runs coincide (deterministic sampler), so A/A noise is 0
        # and says nothing about candidate discordance. The declared assumption stands.
        psi_used = ff.ASSUMED_DISCORDANCE
        notes.append("A/A discordance is 0 (the two incumbent runs coincide); the declared "
                     f"assumed discordance {ff.ASSUMED_DISCORDANCE:g} stands, reported, not hidden")
    d = max(1, round(n * psi_used)) if n else 0
    k = ff.sign_test_critical_wins(d, ff.ALPHA_PER_ATTEMPT) if d else 0
    win_share_at_mde = min(1.0, 0.5 + ff.MDE / (2 * psi_used))
    power = ff.sign_test_power(d, k, win_share_at_mde) if d and k else 0.0
    reasons = []
    if power < 0.8:
        reasons.append(f"power at MDE {ff.MDE:g} is {power:.3f} < 0.8 (measured discordance {psi:.3f})")
    if share is not None and share > ff.CEILING_PASS_SHARE:
        reasons.append(f"incumbent pass share {share:.3f} > {ff.CEILING_PASS_SHARE:.3f} (ceiling)")
    readout = power_readout(aa, paired_n=n, expected_discordant=d, min_win_share=(k / d if d else None))
    return {"measured_discordance": psi, "discordance_used": psi_used, "notes": notes, "paired_n": n, "expected_discordant": d, "critical_wins": k,
            "power_at_mde": power, "incumbent_pass_share": share, "underpowered": bool(reasons),
            "reasons": reasons, "readout": readout.as_record()}


def run_lineage_cycle(ctx: Context, lineage: str, cycle: int, incumbent_text: str) -> dict[str, Any]:
    step = f"c{cycle}.{lineage}"
    if ctx.done(step):
        return ctx.state["steps"][step]
    check_live_target(ctx)
    inc_sha = ctx.texts.put(incumbent_text)
    design = ff.first_feature_design(ctx.splits, cycle=cycle)
    tune = design.train
    view = build_view(ctx, incumbent_text, inc_sha, tune.qids)
    proposer = CachedProposer(ctx.make_local() if lineage == "local_editor" else ctx.make_reference(),
                              ctx.paths.proposals / f"{lineage}-c{cycle}.json", ctx.texts)
    judge = ctx.make_judge()
    if judge is not None:
        judge = ChunkedJudge(judge, ctx.paths.judge / f"{lineage}-c{cycle}")
    driver = SiblingLoopDriver(
        design=design, journal=ctx.journal, journal_dir=ctx.paths.journal, proposer=proposer,
        evaluator=ctx.evaluator, decision_rule=RunMeanSignTest(), judge=judge,
        leakage=ff.make_candidate_screen(ctx.control_text), answer_markers=ctx.markers)
    driver.ledger = ResumingLedger(ctx.journal, alpha_budget=design.alpha_budget)
    role = "incumbent"
    spec = CycleSpec(cycle_index=cycle, tune_split=tune,
                     incumbent=arm_spec(f"{lineage}-c{cycle}-incumbent-{inc_sha[:12]}", role, inc_sha, ctx.pins),
                     incumbent_text=incumbent_text, pins=ctx.pins,
                     max_candidates=ff.MAX_CANDIDATES_PER_CYCLE, alpha_per_attempt=ff.ALPHA_PER_ATTEMPT)
    out: CycleOutcome = driver.run_cycle(view, spec)
    proposal = _load(proposer.path) or {}
    order = [c["candidate_id"] for c in proposal.get("candidates", [])]
    screened = [c for c in order if c not in out.rejected]
    info = {
        "lineage": lineage, "cycle": cycle, "outcome": out.outcome,
        "incumbent_sha256": inc_sha,
        "committed_arm_id": out.committed_arm_id,
        "committed_sha256": ctx.texts.put(out.committed_text),
        "candidates": {c["candidate_id"]: sha256_text(c["mutated_text"]) for c in proposal.get("candidates", [])},
        "first_screened": screened[0] if screened else None,
        "rejected": dict(out.rejected),
        "decisions": {k: {"keep": d.keep, "reason": d.reason, "claim_grade": d.claim_grade}
                      for k, d in out.decisions.items()},
        "lint": dict(out.lint),
        "judge_argmax_ok": out.judge_argmax_ok,
        "judge_chosen": None,
        "tokens_by_seat": dict(out.tokens_by_seat),
        "validation": {k: _result_record(r) for k, r in out.validation.items()},
        "arrival": _result_record(out.arrival) if out.arrival else None,
    }
    ctx.mark(step, **info)
    return ctx.state["steps"][step]


def _result_record(r: ArmResult) -> dict[str, Any]:
    return {"arm_id": r.arm_id, "split_id": r.split_id, "value": r.value, "scored": r.scored,
            "attempted": r.attempted, "treatment_fired": r.treatment_fired, "fence_state": r.fence_state,
            "tokens": [dataclasses.asdict(t) for t in r.tokens],
            "adoption": dataclasses.asdict(adoption_readout(r)),
            "item_outcomes": dict(r.item_outcomes)}


SELF_JUDGE_SCHEMA_TEXT = ('Reply with exactly one JSON object: {"choice": "<candidate_id>"} to keep one '
                          'candidate, or {"choice": null} to keep the current instructions.')


def self_judged_choice(ctx: Context, lineage: str, cycle: int, incumbent_text: str,
                       candidates: Mapping[str, str]) -> str | None:
    """Item 4 ``self_judged``: the LOCAL editor picks among one lineage's candidates, no labels."""
    path = ctx.paths.proposals / f"{lineage}-c{cycle}-self-judged.json"
    doc = _load(path)
    if doc is None:
        if not candidates:
            doc = {"choice": None, "reason": "no candidates"}
        else:
            parts = [f"CURRENT INSTRUCTIONS:\n```\n{incumbent_text}\n```"]
            for cid, sha in candidates.items():
                parts.append(f"CANDIDATE {cid}:\n```\n{ctx.texts.get(sha)}\n```")
            prompt = ("You maintain the instructions of an assistant. Below are its current instructions "
                      "and proposed replacements. Without any test results, decide which version you "
                      "would deploy.\n\n" + "\n\n".join(parts) + "\n\n" + SELF_JUDGE_SCHEMA_TEXT)
            editor = ctx.make_local()
            resp = editor._transport({"model": editor.model_id, "messages": [{"role": "user", "content": prompt}],
                                      "temperature": 0.0, "max_tokens": 256, "stream": False})
            from src.harness_loop.transport import extract_json_object

            obj = extract_json_object(openai_message_text(resp) or "") or {}
            choice = obj.get("choice")
            doc = {"choice": choice if choice in candidates else None, "raw": obj}
        doc["at"] = _now()
        _dump(path, doc)
    return doc["choice"]


def verification_modes(ctx: Context, cycle: int, info: Mapping[str, Any], arrival: SplitRef) -> dict[str, Any]:
    """Item 4: each selection mode's choice, evaluated on the cycle's arrival split. Only ``with``
    commits; ``none`` and ``self_judged`` are reported, evaluated only when they pick another arm."""
    lineage = info["lineage"]
    inc_text = ctx.texts.get(info["incumbent_sha256"])
    modes = {"with": info["committed_sha256"],
             "none": info["candidates"].get(info["first_screened"]) if info["first_screened"] else info["incumbent_sha256"]}
    screened = {c: s for c, s in info["candidates"].items() if c not in info["rejected"]}
    choice = self_judged_choice(ctx, lineage, cycle, inc_text, screened)
    modes["self_judged"] = screened[choice] if choice else info["incumbent_sha256"]
    out: dict[str, Any] = {"choices": modes, "arrival": {}}
    for mode, sha in modes.items():
        if mode == "with":
            out["arrival"][mode] = info["arrival"]
            continue
        r = ctx.evaluator.evaluate(arm_spec(f"{lineage}-c{cycle}-{mode}-{sha[:12]}", "candidate", sha, ctx.pins), arrival)
        out["arrival"][mode] = _result_record(r)
    return out


def run_pilot(ctx: Context) -> int:
    """All steps in order; each is skipped when ``state.json`` says it is done."""
    s, ev = ctx.splits, ctx.evaluator
    control_sha = ctx.texts.put(ctx.control_text)
    control = arm_spec(f"control-{control_sha[:12]}", "unmodified_control", control_sha, ctx.pins)

    if not ctx.done("c1.traces"):
        ensure_traces(ctx, control_sha, s.train.qids, s.train.split_id)
        ctx.mark("c1.traces", n=len(s.train.qids))
    if not ctx.done("c1.aa"):
        aa = ev.evaluate(arm_spec(f"incumbent-c1-{control_sha[:12]}", "incumbent", control_sha, ctx.pins), s.validation)
        gate = power_gate(ctx, aa)
        ctx.mark("c1.aa", power=gate, result=_result_record(aa))
    gate = ctx.state["steps"]["c1.aa"]["power"]
    if gate["underpowered"] and not ctx.args.ignore_power_gate:
        ctx.log(f"underpowered before any candidate: {gate['reasons']}")
        if not ctx.done("underpowered"):
            ctx.mark("underpowered", reasons=gate["reasons"])
        return write_report(ctx)

    committed: dict[str, str] = {}
    for lineage in LINEAGES:
        info = run_lineage_cycle(ctx, lineage, 1, ctx.control_text)
        committed[lineage] = info["committed_sha256"]
    ctx.state.setdefault("released", {}).setdefault("arrival", _now())
    ctx.save()
    s = ctx.splits = load_frozen(ctx.paths, ctx.state)

    if not ctx.done("c1.diag"):
        diag: dict[str, Any] = {"verification_modes": {}}
        for lineage in LINEAGES:
            diag["verification_modes"][lineage] = verification_modes(ctx, 1, ctx.state["steps"][f"c1.{lineage}"], s.arrival)
        off = ctx.off_evaluator.evaluate(arm_spec(f"all-off-{control_sha[:12]}", "unmodified_control", control_sha, ctx.pins), s.validation)
        diag["all_off"] = _result_record(off)
        diag["reference_trajectory_arm"] = reference_trajectory_arm(ctx, control_sha)
        diag["matched_rollout"] = matched_rollout(ctx, control_sha)
        ctx.mark("c1.diag", **diag)

    for lineage in LINEAGES:
        inc_text = ctx.texts.get(committed[lineage])
        inc_sha = committed[lineage]
        step = f"c2.traces.{lineage}"
        if not ctx.done(step):
            ensure_traces(ctx, inc_sha, s.train.qids | s.arrival.qids, f"{s.train.split_id}+{s.arrival.split_id}")
            ctx.mark(step)
        info = run_lineage_cycle(ctx, lineage, 2, inc_text)
        committed[lineage] = info["committed_sha256"]
    ctx.state["released"].setdefault("arrival_2", _now())
    ctx.save()
    s = ctx.splits = load_frozen(ctx.paths, ctx.state)

    if not ctx.done("c2.diag"):
        ctx.mark("c2.diag", verification_modes={
            lin: verification_modes(ctx, 2, ctx.state["steps"][f"c2.{lin}"], s.arrival_2) for lin in LINEAGES})

    if not ctx.done("test"):
        arms: list[ArmSpec] = []
        for lineage in LINEAGES:
            sha = committed[lineage]
            if sha != control_sha and all(a.content_sha256 != sha for a in arms):
                arms.append(arm_spec(f"{lineage}-committed-{sha[:12]}", "candidate", sha, ctx.pins))
        arms.append(control)
        results = final_test(ctx, arms)
        ctx.mark("test", arms=[a.arm_id for a in arms], results={r.arm_id: _result_record(r) for r in results})
    return write_report(ctx)


def final_test(ctx: Context, arms: Sequence[ArmSpec]) -> tuple[ArmResult, ...]:
    """Item 3, paired: one touch of TEST for every final arm. A resumed touch (the journal holds
    the touch event and state says unfinished) re-evaluates the same arms from the cache."""
    from src.harness_loop.driver import TEST_EVENT

    design = ff.first_feature_design(ctx.splits, cycle=2)
    driver = SiblingLoopDriver(design=design, journal=ctx.journal, journal_dir=ctx.paths.journal,
                               proposer=ctx.make_local(), evaluator=ctx.evaluator,
                               decision_rule=RunMeanSignTest(), leakage=lambda *a, **k: None)
    touched = ctx.journal.ledger_events(TEST_EVENT)
    if not touched:
        return driver.final_test_paired(arms[0], *arms[1:])
    if touched[-1].get("arm_ids") != [a.arm_id for a in arms]:
        raise RuntimeError("item 3: the test split was touched for a different arm set")
    return tuple(driver._evaluate(a, design.test) for a in arms)


def reference_trajectory_arm(ctx: Context, control_sha: str) -> dict[str, Any]:
    """M9 diagnostic (design §4): a separate local-editor proposal whose dossier also carries one
    T1 trajectory that reached a scorer-passing answer. Never committed, never in the ledger."""
    trace = None
    for q in sorted(ctx.splits.train.qids, key=lambda q: ff._order_key(ff.SPLIT_SEED, q)):
        hit = ctx.cache.get(request_id("arm", control_sha, item_key(q, 1))) or {}
        resp = hit.get("response") or {}
        ans = str(resp.get("answer") or "")
        row = ctx.items[q]
        if ans.strip() and ctx.args._score_fn(ans, str(row.get("expected") or ""), dict(row)):
            trace = TraceRef(trace_id=f"reference-trajectory-{q}", role=ff.CONSUMER_ROLE,
                             events=("reference_trajectory",),
                             excerpt=(str(row["prompt"])[:EXCERPT_CHARS] + "\nFINAL ANSWER: " + ans[:400]))
            break
    if trace is None:
        return {"status": "no scorer-passing T1 trajectory"}
    view = build_view(ctx, ctx.control_text, control_sha, ctx.splits.train.qids, extra_trace=trace)
    prop = CachedProposer(ctx.make_local(), ctx.paths.proposals / "reference-trajectory-c1.json", ctx.texts)
    res = prop.propose(view, max_candidates=1)
    if not res.candidates:
        return {"status": "no candidate", "trace": trace.trace_id}
    cand = res.candidates[0]
    reason = ff.make_candidate_screen(ctx.control_text)(cand.mutated_text, original=cand.original_text,
                                                         trace_context=substrate.render_trace_context(view.traces))
    if reason:
        return {"status": "screened_out", "reason": reason, "trace": trace.trace_id}
    sha = ctx.texts.put(cand.mutated_text)
    r = ctx.evaluator.evaluate(arm_spec(f"reftraj-{sha[:12]}", "candidate", sha, ctx.pins), ctx.splits.validation)
    return {"status": "evaluated", "trace": trace.trace_id, "result": _result_record(r),
            "note": "diagnostic only: outside selection, never committed, not in the attempt ledger"}


def _norm_answer(a: str) -> str:
    return " ".join(a.strip().lower().split())


def matched_rollout(ctx: Context, control_sha: str) -> dict[str, Any]:
    """Item 1: the incumbent sampled k = 8 times per V item; label-free majority over answers."""
    ev = ctx.evaluator
    arm = arm_spec(f"matched-rollout-{control_sha[:12]}", "matched_rollout_baseline", control_sha, ctx.pins)
    ev.evaluate(arm, ctx.splits.validation, runs=MATCHED_ROLLOUT_K)
    outcomes: dict[str, bool | None] = {}
    coincide = 0
    for q in sorted(ctx.splits.validation.qids):
        answers = [ev.answer(control_sha, q, r) for r in range(1, MATCHED_ROLLOUT_K + 1)]
        normed = [_norm_answer(a) for a in answers if a.strip()]
        if not normed:
            outcomes[q] = False
            continue
        top = Counter(normed).most_common(1)[0][0]
        if len(set(normed)) == 1:
            coincide += 1
        chosen = next(a for a in answers if _norm_answer(a) == top)
        row = ctx.items[q]
        outcomes[q] = bool(ctx.args._score_fn(chosen, str(row.get("expected") or ""), dict(row)))
    return {"k": MATCHED_ROLLOUT_K, "value": pass_share(outcomes), "n": len(outcomes),
            "items_all_samples_coincide": coincide, "item_outcomes": outcomes,
            "note": "ChatRequest has no temperature: coinciding samples are reported, not hidden"}


# ── report and VB-HIL-WRITE finalize ──────────────────────────────────────────


def bootstrap_paired_ci(a: Mapping[str, float], b: Mapping[str, float], *, n_boot: int = 10_000,
                        seed: int = 0) -> tuple[float, float, float] | None:
    common = sorted(set(a) & set(b))
    if not common:
        return None
    diffs = [b[q] - a[q] for q in common]
    rng = random.Random(seed)
    n = len(diffs)
    stats = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return (sum(diffs) / n, stats[int(0.025 * n_boot)], stats[int(0.975 * n_boot) - 1])


def wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def write_report(ctx: Context) -> int:
    steps = ctx.state.get("steps", {})
    rep: dict[str, Any] = {
        "schema": "epyc.orchestrator.hil6j_pilot_report.v1",
        "design": PILOT_DESIGN_ID, "feature_id": ff.FEATURE_ID, "lift": LIFT_RECORD,
        "decision_grade": "observation", "protocol_id": None, "at": _now(),
        "fake_backends": bool(ctx.args.fake_backends),
        "power": steps.get("c1.aa", {}).get("power"),
        "underpowered": "underpowered" in steps,
        "cycles": {k: {kk: v for kk, v in steps[k].items() if kk not in ("validation",)}
                   for k in steps if re.fullmatch(r"c[12]\.(local_editor|reference_editor)", k)},
        "infra_failures": {f"{a}|{sp}": Counter(v.values()) for (a, sp), v in ctx.evaluator.infra.items() if v},
    }
    test = steps.get("test")
    if test:
        res = test["results"]
        control_id = test["arms"][-1]
        cmeans = _means(res[control_id]["item_outcomes"])
        rep["test"] = {}
        for arm_id, r in res.items():
            m = _means(r["item_outcomes"])
            share = (sum(m.values()) / len(m)) if m else None
            entry = {"pass_share": share, "wilson95": wilson(share or 0.0, len(m)), "n": len(m),
                     "treatment_fired": r["treatment_fired"]}
            if arm_id != control_id:
                entry["paired_vs_control"] = bootstrap_paired_ci(cmeans, m)
                entry["decision"] = dataclasses.asdict(RunMeanSignTest().decide(
                    _as_result(res[control_id]), _as_result(r)))
            rep["test"][arm_id] = entry
        rep["regression_readout"] = regression_readout(steps)
        if not ctx.args.fake_backends:  # epyc_root None: hil_ab_sidecar resolves $EPYC_ROOT / the root-main view
            rep["finalize"] = finalize_beliefs(ctx, test)
    _dump(ctx.paths.report, rep)
    complete = bool(test)
    ctx.log(f"report written: {ctx.paths.report} complete={complete}")
    if complete or rep["underpowered"]:
        (ctx.paths.root / "PILOT_DONE").write_text(_now() + "\n")
    return EXIT_OK


def _means(outcomes: Mapping[str, Any]) -> dict[str, float]:
    return RunMeanSignTest.item_means(_as_result({"item_outcomes": outcomes}))


def _as_result(rec: Mapping[str, Any]) -> ArmResult:
    return ArmResult(arm_id=str(rec.get("arm_id", "")), split_id=str(rec.get("split_id", "")),
                     metric=ff.PRIMARY_METRIC, value=rec.get("value"), unit=ff.PRIMARY_UNIT,
                     direction=ff.PRIMARY_DIRECTION, scored=int(rec.get("scored") or 0),
                     attempted=int(rec.get("attempted") or 0), item_outcomes=dict(rec["item_outcomes"]))


def regression_readout(steps: Mapping[str, Any]) -> dict[str, Any]:
    """Item 18 (``hil6i-arch-deleg-regress-v1``): V items the cycle-1 incumbent passed in both
    runs, and the right-to-wrong count per evaluated validation arm."""
    aa = steps.get("c1.aa", {}).get("result")
    if not aa:
        return {}
    keep = {q for q, v in _means(aa["item_outcomes"]).items() if v == 1.0}
    out: dict[str, Any] = {"set_id": ff.REGRESSION_SET_ID, "n": len(keep), "right_to_wrong": {}}
    for k, st in steps.items():
        for arm_id, r in (st.get("validation") or {}).items():
            m = _means(r["item_outcomes"])
            out["right_to_wrong"][arm_id] = sum(1 for q in keep if m.get(q, 1.0) < 1.0)
    return out


def finalize_beliefs(ctx: Context, test: Mapping[str, Any]) -> dict[str, Any]:
    """VB-HIL-WRITE: one finalize call per paired comparison on TEST (control vs each committed
    arm), each control under its own run_id; ``protocol_id`` stays None."""
    from src.harness_loop.finalize import finalize_run, observation

    out: dict[str, Any] = {}
    res = test["results"]
    control_id = test["arms"][-1]
    ctx.paths.results.mkdir(parents=True, exist_ok=True)
    results_path = ctx.paths.results / "test_results.json"
    _dump(results_path, res)
    head = ctx.args.orch_sha or ""
    for arm_id in test["arms"][:-1]:
        run_id = f"hil6j-{ctx.splits.manifest_sha256[:12]}-{arm_id}"
        try:
            obs = []
            for aid, role in ((control_id, "unmodified_control"), (arm_id, "candidate")):
                r = _as_result(res[aid])
                sha = aid.rsplit("-", 1)[-1]
                full = next(p.stem for p in ctx.paths.texts.glob(f"{sha}*.md"))
                arm = arm_spec(aid, role, full, ctx.pins)
                obs.append(observation(r, arm, objective="task_pass_share", suite_id=EGRESS_SUITE_ID,
                                       suite_version=ctx.splits.manifest_sha256, split="test",
                                       native_verdict="observation", prompt_param_path=str(ctx.paths.texts / f"{full}.md"),
                                       harness_pin=head, config_sha256=ctx.splits.manifest_sha256))
            path = finalize_run(ctx.paths.results / run_id, run_id=run_id, results_path=results_path,
                                observations=obs, pins=ctx.pins, protocol_id=None, epyc_root=ctx.args.epyc_root)
            out[arm_id] = {"ok": True, "path": str(path)}
        except Exception as exc:  # recorded in the report; the raw results stay on disk
            out[arm_id] = {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
    return out


# ── fakes (dry run only) ──────────────────────────────────────────────────────


class FakeBackends:
    """Deterministic in-process stand-ins for the private API, the local editor and both cloud
    CLIs. Used ONLY by ``--fake-backends`` (the job's dry run); every artifact says fake."""

    def __init__(self, items_path: Path | None = None) -> None:
        self.calls = Counter()
        self._items: dict[str, Mapping[str, Any]] = {}
        if items_path is not None and items_path.exists():
            for line in items_path.read_text().splitlines():
                row = json.loads(line)
                self._items[row["prompt"][:200]] = row

    def chat(self, body: Mapping[str, Any]) -> Mapping[str, Any]:
        self.calls["chat"] += 1
        if body.get("eval_fence") is not True:
            raise AssertionError("fence not armed")
        row = self._items.get(str(body.get("prompt"))[:200], {})
        h = int(hashlib.sha256(str(body["request_id"]).encode()).hexdigest()[:8], 16)
        good = h % 3 != 0
        answer = str(row.get("expected") or "x") if good else "wrong"
        return {"answer": answer, "eval_fence": {"state": "active", "denied_count": 0},
                "tokens_generated": 10 + h % 50,
                "delegation_diagnostics": {"phases_count": 1 if body.get("allow_delegation") else 0,
                                           "loops": 1, "break_reason": ""},
                "delegation_events": []}

    def editor(self, body: Mapping[str, Any]) -> Mapping[str, Any]:
        self.calls["editor"] += 1
        text = str(body["messages"][-1]["content"])
        if "CURRENT INSTRUCTIONS" in text:
            return {"choices": [{"message": {"content": '{"choice": null}'}}]}
        m = re.search(r"```\n(.*?)\n```", text, re.S)
        cur = m.group(1) if m else ""
        variants = [cur.replace("Decision:", f"Think step by step ({i}).\nDecision:", 1) for i in (1, 2)]
        return {"choices": [{"message": {"content": json.dumps(
            {"candidates": [{"text": v, "rationale": "fake"} for v in variants]})}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "prompt_tokens_details": {"cached_tokens": 0}}}

    def cloud_run(self, argv: list[str], stdin_text: str, timeout_s: float, cwd: str) -> tuple[int, str, str]:
        self.calls["cloud"] += 1
        if "ok" in stdin_text and "true" in stdin_text and "candidates" not in stdin_text and "rows" not in stdin_text:
            payload: dict[str, Any] = {"ok": True}
        elif "hil_credibility_judge_request" in stdin_text:
            req = json.loads(stdin_text[stdin_text.index("{"):stdin_text.rindex("}") + 1])
            payload = {"rows": [{"candidate_id": c["candidate_id"],
                                 "cells": [{"instance_id": i["instance_id"], "verdict": "unresolved", "reason": "fake"}
                                           for i in req["instances"]]} for c in req["candidates"]],
                       "chosen": None}
        else:
            m = re.search(r"```\n(.*?)\n```", stdin_text, re.S)
            cur = m.group(1) if m else ""
            payload = {"candidates": [{"text": cur.replace("Decision:", "Be brief.\nDecision:", 1), "rationale": "fake"}]}
        if "-o" in argv:
            Path(argv[argv.index("-o") + 1]).write_text(json.dumps(payload))
            return 0, "", ""
        return 0, json.dumps({"structured_output": payload, "modelUsage": {"fake": {}}}), ""


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="python -m src.harness_loop.pilot_run", description=__doc__.split("\n")[0])
    ap.add_argument("--design", required=True)
    ap.add_argument("--operator-lift")
    ap.add_argument("--phase", required=True,
                    choices=["preflight", "freeze", "run", "status", "estimate", "exclude-set", "exclude-clear"])
    ap.add_argument("--run-dir", type=Path)
    ap.add_argument("--pool", type=Path, help="research benchmarks/prompts/question_pool.jsonl")
    ap.add_argument("--research-root", type=Path, default=Path("/workspace/repos/epyc-inference-research"),
                    help="research checkout (code_exec_scorer for the code-execution stratum)")
    ap.add_argument("--core-v2", type=Path, help="orchestrator benchmarks/prompts/core_v2.jsonl")
    ap.add_argument("--seen-root", type=Path, action="append", default=[],
                    help="eval-artifact roots scanned for qids seen within 60 days (repeatable)")
    ap.add_argument("--live-target", default="/mnt/raid0/llm/epyc-orchestrator/" + ff.TARGET_PATH)
    ap.add_argument("--prompt-template-dir", type=Path, help="the pinned worktree's orchestration/prompts")
    ap.add_argument("--prompt-root-base", type=Path, help="the private API's ORCHESTRATOR_PROMPT_ROOT_OVERRIDE_BASE")
    ap.add_argument("--api-url", help="private orchestrator API base URL")
    ap.add_argument("--editor-url", help="private local-editor llama-server base URL")
    ap.add_argument("--server-urls", type=Path, help="JSON {role: url} of the PRIVATE servers, sent per request")
    ap.add_argument("--served-ctx", type=int, default=65536)
    ap.add_argument("--request-timeout-s", type=int, default=900)
    ap.add_argument("--egress-clearance", type=Path, help="operator-issued egress clearance JSON")
    ap.add_argument("--epyc-root", help="root checkout for the VB-HIL-WRITE carrier (default: resolved)")
    ap.add_argument("--orch-sha", default="", help="pinned orchestrator commit (recorded as harness_pin)")
    ap.add_argument("--stopfile", type=Path)
    ap.add_argument("--deadline", type=float, default=0.0, help="epoch seconds; no new request after it")
    ap.add_argument("--skip-cloud-verify", action="store_true", help="preflight without the two live calls")
    ap.add_argument("--fake-backends", action="store_true", help="dry run: in-process fakes only")
    ap.add_argument("--ignore-power-gate", action="store_true", help=argparse.SUPPRESS)
    return ap.parse_args(argv)


def _should_stop(args: argparse.Namespace) -> Callable[[], bool]:
    def stop() -> bool:
        if args.stopfile is not None and args.stopfile.exists():
            return True
        return bool(args.deadline) and time.time() >= args.deadline

    return stop


def _orch_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_context(args: argparse.Namespace, fakes: FakeBackends | None) -> Context:
    paths = RunPaths(args.run_dir)
    state = _load(paths.state, {}) or {}
    splits = load_frozen(paths, state)
    items = load_items(paths)
    texts = TextStore(paths.texts)
    cache = ChatCache(paths.cache)
    journal = substrate.open_loop_journal(paths.journal)
    doc = _load(paths.freeze)
    control_text = texts.get(doc["control_sha256"])
    score_fn = default_score_fn(_orch_root(), args.research_root)
    args._score_fn = score_fn
    if fakes is not None:
        chat_t, editor_t, run_fn = fakes.chat, fakes.editor, fakes.cloud_run
    else:
        if not (args.api_url and args.editor_url and args.server_urls and args.prompt_root_base
                and args.prompt_template_dir):
            raise PilotBlocked("--api-url, --editor-url, --server-urls, --prompt-root-base and "
                               "--prompt-template-dir are required for a live run")
        chat_t = http_json_transport(CHAT_PATH, base_url=args.api_url, timeout_s=args.request_timeout_s + 60)
        editor_t = http_json_transport(V1_CHAT_PATH, base_url=args.editor_url, timeout_s=1800)
        run_fn = None
    urls = json.loads(args.server_urls.read_text()) if args.server_urls else {}
    base = args.prompt_root_base or (paths.root / "prompt_roots")
    template = args.prompt_template_dir or (_orch_root() / "orchestration" / "prompts")
    base.mkdir(parents=True, exist_ok=True)

    def root_for(sha: str) -> Path:
        return materialize_arm(base, template, ff.TARGET_PATH, texts.get(sha))

    transport = cached_transport(chat_t, cache, _should_stop(args))
    common = dict(transport=transport, cache=cache, items=items, texts=texts, root_for=root_for,
                  server_urls=urls, score_fn=score_fn, timeout_s=args.request_timeout_s)
    evaluator = PilotEvaluator(**common)
    off = PilotEvaluator(**common, prefix="off", allow_delegation=False)
    from src.harness_loop import finalize as _fin  # noqa: F401  (import-time check of the hook module)
    from scripts.benchmark.hil_ab_sidecar import arm_pins, canonical_sha256, compaction_policy_digest

    pins = arm_pins(tool_surface_sha256=canonical_sha256({"orchestrator_commit": args.orch_sha or "unpinned",
                                                          "basis": "commit-pinned tool surface"}),
                    served_context_window=args.served_ctx, wall_time_budget_s=float(args.request_timeout_s),
                    turn_budget=LOOP_BUDGET,
                    final_result_selection_rule="architect D| answer, else the synthesized answer after <= 3 loops",
                    compaction_policy_sha256=compaction_policy_digest())
    markers = tuple(sorted({str(items[q].get("expected") or "") for q in splits.validation.qids | splits.test.qids
                            if len(str(items[q].get("expected") or "")) >= 16}))
    from src.typed_decisions.cloud_judges import load_registry

    registry = load_registry()
    clearances = load_clearance(args.egress_clearance)
    if fakes is not None:
        clearances = {spec.egress: EgressClearance(spec.egress, EGRESS_SUITE_ID, "dry-run fake (no egress)")
                      for spec in registry.values()}

    def make_local() -> LocalEditorProposer:
        return LocalEditorProposer(transport=qwen_editor_transport(editor_t), model_id=ff.LOCAL_EDITOR_MODEL,
                                   seed=0, answer_markers=markers)

    def make_reference() -> ReferenceEditorProposer:
        spec = registry[REFERENCE_EDITOR_JUDGE_ENTRY]
        if spec.egress not in clearances:
            raise PilotBlocked(f"no operator egress clearance for {spec.egress!r}: the reference editor "
                               f"would send T1 excerpts off-host")
        return ReferenceEditorProposer(model_id=ff.REFERENCE_EDITOR_MODEL,
                                       transport=codex_editor_transport(spec, ff.REFERENCE_EDITOR_MODEL, run_fn),
                                       judge_model=ff.JUDGE_MODEL, local_editor_model=ff.LOCAL_EDITOR_MODEL,
                                       egress=spec.egress, egress_clearance=clearances[spec.egress],
                                       answer_markers=markers)

    def make_judge() -> CredibilityMatrixJudge:
        spec = registry[JUDGE_ENTRY]
        if spec.egress not in clearances:
            raise PilotBlocked(f"no operator egress clearance for {spec.egress!r}: the judge would send V items off-host")

        def evidence(c: Candidate, iid: str) -> str:
            return f"candidate answer: {evaluator.answer(sha256_text(c.mutated_text), iid)[:1500]}\n" \
                   f"original-instructions answer: {evaluator.answer(sha256_text(c.original_text), iid)[:1500]}"

        return CredibilityMatrixJudge(transport=cloud_cli_transport(run_fn),
                                      proposer_models=[ff.LOCAL_EDITOR_MODEL, ff.REFERENCE_EDITOR_MODEL],
                                      local_editor_model=ff.LOCAL_EDITOR_MODEL, reference_editor_model=ff.REFERENCE_EDITOR_MODEL,
                                      egress_clearance=clearances[spec.egress], judge_name=JUDGE_ENTRY,
                                      registry=registry, task_fn=lambda iid: str(items[iid]["prompt"]),
                                      evidence_fn=evidence,
                                      call_log_sink=lambda rec: _append_jsonl(paths.judge / "calls.jsonl", rec))

    return Context(args=args, paths=paths, state=state, splits=splits, items=items, texts=texts, cache=cache,
                   journal=journal, evaluator=evaluator, control_text=control_text, pins=pins, markers=markers,
                   make_local=make_local, make_reference=make_reference, make_judge=make_judge, off_evaluator=off)


def _append_jsonl(path: Path, rec: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, sort_keys=True, default=str) + "\n")


def do_freeze(args: argparse.Namespace) -> int:
    paths = RunPaths(args.run_dir)
    if not args.pool:
        raise PilotBlocked("--pool is required for freeze")
    orch = _orch_root()
    strata = set(ff.STRATA)
    rows = [r for r in iter_pool(args.pool) if r.get("suite") in strata]
    pool_ids = {str(r["id"]) for r in rows}
    exclusions: dict[str, str] = {}
    gold = gold_self_check_exclusions(rows, orch)
    exclusions.update(gold)
    for q in core_v2_ids(args.core_v2 or orch / "benchmarks" / "prompts" / "core_v2.jsonl") & pool_ids:
        exclusions.setdefault(q, "core_v2:member")
    cutoff = time.time() - FREEZE_SEEN_WINDOW_DAYS * 86400
    seen, files = seen_since(args.seen_root, pool_ids, cutoff)
    for q in seen:
        exclusions.setdefault(q, "seen_60d:artifact")
    live = Path(args.live_target)
    control_text = live.read_text(encoding="utf-8")
    try:
        add_code_exec_path(args.research_root)
        import code_exec_scorer  # type: ignore  # noqa: F401

        sandbox = "code_exec_scorer importable (local isolated subprocess)"
    except Exception as exc:
        raise PilotBlocked(f"code-execution scorer unavailable at freeze ({exc}); dropping that stratum "
                           f"is a re-design, not a silent shrink") from exc
    doc = freeze(paths, rows=rows, exclusions=exclusions, control_text=control_text,
                 live_target_sha=sha256_text(control_text), pool_meta=pool_metadata(args.pool),
                 extra={"seen_rule": {"window_days": FREEZE_SEEN_WINDOW_DAYS, "cutoff_epoch": int(cutoff),
                                      "roots": [str(r) for r in args.seen_root], "files": files,
                                      "rule": "pool ids named in any .json/.jsonl under the roots with mtime >= cutoff"},
                        "sandbox": sandbox, "lift": LIFT_RECORD})
    print(json.dumps({"frozen": doc["manifest_sha256"], "excluded": doc["exclusions"]}, indent=1))
    return EXIT_OK


def do_preflight(args: argparse.Namespace, fakes: FakeBackends | None) -> int:
    from src.typed_decisions.cloud_judges import load_registry

    paths = RunPaths(args.run_dir)
    manifest = _load(paths.manifest, {}) or {}
    manifest.update(schema="epyc.orchestrator.hil6j_pilot_manifest.v1", design=PILOT_DESIGN_ID,
                    lift=LIFT_RECORD, lift_token=LIFT_TOKEN, orch_sha=args.orch_sha,
                    cloud_call_estimate=estimate_cloud_calls(), request_estimate=estimate_requests(),
                    fake_backends=bool(fakes))
    print("cloud call estimate:", json.dumps(manifest["cloud_call_estimate"], indent=1))
    print("request/wall estimate:", json.dumps(manifest["request_estimate"], indent=1))
    if not args.skip_cloud_verify:
        ver = cloud_live_verification(load_registry(), run_fn=fakes.cloud_run if fakes else None)
        manifest.setdefault("cloud_live_verification", []).append({"at": _now(), "results": ver})
        print("cloud live verification:", json.dumps(ver, indent=1, default=str))
    clear = load_clearance(args.egress_clearance)
    manifest["egress_clearance"] = {e: c.as_record() for e, c in clear.items()}
    _dump(paths.manifest, manifest)
    if fakes is None:
        registry = load_registry()
        missing = [registry[e].egress for e in (REFERENCE_EDITOR_JUDGE_ENTRY, JUDGE_ENTRY)
                   if e in registry and registry[e].egress not in clear]
        if missing:
            print(f"BLOCKED: no operator egress clearance (--egress-clearance) for {missing}; the "
                  "reference editor and the judge would send suite content off-host")
            return EXIT_BLOCKED
    if not args.skip_cloud_verify and not all(r["ok"] for r in manifest["cloud_live_verification"][-1]["results"]):
        print("BLOCKED: a cloud transport failed its live verification (see manifest)")
        return EXIT_BLOCKED
    return EXIT_OK


def main(argv: Sequence[str] | None = None, *, env: Mapping[str, str] | None = None,
         fakes: FakeBackends | None = None) -> int:
    args = parse_args(argv)
    env = os.environ if env is None else env
    if args.phase == "estimate":
        print(json.dumps({"cloud_calls": estimate_cloud_calls(), "requests": estimate_requests()}, indent=1))
        return EXIT_OK
    if args.phase == "exclude-clear":  # always allowed: the job's EXIT trap must be able to clear
        from src.autopilot_core import mutation_exclusions as mx

        n = mx.clear_exclusion(owner=EXCLUSION_OWNER, target=ff.TARGET_PATH)
        print(f"exclusion cleared ({n} entr{'y' if n == 1 else 'ies'}) in {mx.exclusions_file()}")
        return EXIT_OK
    reasons = refusal_reasons(design=args.design, lift=args.operator_lift, env=env)
    if reasons:
        for r in reasons:
            print(f"REFUSED: {r}", file=sys.stderr)
        return EXIT_REFUSED
    if args.run_dir is None:
        print("REFUSED: --run-dir is required", file=sys.stderr)
        return EXIT_REFUSED
    if args.fake_backends and fakes is None:
        fakes = FakeBackends(RunPaths(args.run_dir).items)
    try:
        if args.phase == "exclude-set":
            from src.autopilot_core import mutation_exclusions as mx

            p = mx.set_exclusion(ff.TARGET_PATH, owner=EXCLUSION_OWNER, reason=f"HIL-6j pilot run {args.run_dir}")
            print(f"exclusion set: {ff.TARGET_PATH} in {p}")
            return EXIT_OK
        if args.phase == "freeze":
            return do_freeze(args)
        # every phase below needs the frozen splits and the validator
        state = _load(RunPaths(args.run_dir).state, {}) or {}
        splits = load_frozen(RunPaths(args.run_dir), state)
        vr = validator_reasons(splits)
        if vr:
            for r in vr:
                print(f"REFUSED (validator): {r}", file=sys.stderr)
            return EXIT_REFUSED
        if args.phase == "preflight":
            return do_preflight(args, fakes)
        if args.phase == "status":
            print(json.dumps({"steps": sorted(state.get("steps", {})), "released": state.get("released", {}),
                              "done": (RunPaths(args.run_dir).root / "PILOT_DONE").exists()}, indent=1))
            return EXIT_OK
        # phase == "run"
        from src.autopilot_core import mutation_exclusions as mx

        ctx = build_context(args, fakes)
        mx.set_exclusion(ff.TARGET_PATH, owner=EXCLUSION_OWNER, reason=f"HIL-6j pilot run {args.run_dir}")
        try:
            return run_pilot(ctx)
        finally:
            mx.clear_exclusion(owner=EXCLUSION_OWNER, target=ff.TARGET_PATH)
    except PilotStopped as exc:
        print(f"STOPPED (resumable): {exc}")
        return EXIT_STOPPED
    except RefreezeRequired as exc:
        print(f"REFREEZE REQUIRED: {exc}", file=sys.stderr)
        return EXIT_REFREEZE
    except PilotBlocked as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return EXIT_BLOCKED


if __name__ == "__main__":
    raise SystemExit(main())
