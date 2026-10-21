"""AutoPilot's consumer half of the shared belief reader (ROOT `scripts/vidya/loop_beliefs.py`).

MANDATORY, NOT ADVISORY (operator rules 2026-10-21). Before every planner turn the loop calls
`prepare()`, which runs the ROOT reader as a CHILD PROCESS (never imported: a slow ledger is killed
at our deadline). Only a validated `ready` answer can be turned into a `ReadyBeliefs`, and
`planner_coordinator.plan_with_providers` refuses to run without one -- a plan drafted without a
verified, current ledger state is impossible by construction.

When the answer is refused (or the child times out / crashes / prints garbage) the loop generates
NO new hypothesis. Degraded mode keeps compute busy only with PRESEEDED, not-yet-tested items --
open operator hypotheses carrying a `proposed_action` -- filtered through the reader's
`check-seeds` against the LAST KNOWN-GOOD snapshot. An admissible seed is not dispatched
unreviewed: `mint_seed_review()` (the only constructor of `SeedReviewToken`) lets the binding
critic review it via `planner_coordinator.review_preseeded_seed`, which never drafts. When the
ledger recovers, degraded-mode results are re-checked against the fresh snapshot and any whose
premise has since been flagged gets a review flag (never an automatic reversal).

Configuration is argv on `autopilot.py start` (see `BeliefSettings`); there is no off switch.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

log = logging.getLogger("autopilot")

SCRIPT_DIR = Path(__file__).resolve().parent
ORCH_ROOT = SCRIPT_DIR.parents[1]

CONSUMER = "autopilot.planner"
REPLY_FIELD = "vidya_claim_ids"
ANSWER_SCHEMA = "epyc.vidya.loop_beliefs.v1"
LEDGER_STATE_SCHEMA = "epyc.vidya.ledger_state.v1"
SEED_CHECK_SCHEMA = "epyc.vidya.seed_check.v1"
KNOWN_GOOD_SCHEMA = "epyc.vidya.loop_known_good.v1"
RECEIPT_SCHEMA = "epyc.vidya.loop_beliefs_planner_receipt.v1"
DEGRADED_SCHEMA = "epyc.autopilot.belief_degraded.v1"
REVIEW_FLAG_SCHEMA = "epyc.autopilot.belief_review_flag.v1"

DEFAULT_ROOT_REPO = Path("/workspace")
#: Used only when the default checkout has no reader (a lagging /workspace on 2026-10-21 was
#: 466 commits behind origin/main and lacked scripts/vidya/loop_beliefs.py). The reader's alarm
#: state is the canonical bus (session_bus.get_bus_root, not __file__-relative), so running it
#: from another checkout of the same ROOT repo does not fork alarm state.
ROOT_REPO_FALLBACKS = (Path("/mnt/raid0/llm/views/epyc-root-main"),)
DEFAULT_ESCALATE_AFTER_S = 3600.0
#: Measured 2026-10-21: one `prepare` over the 30.8k-frame canonical ledger takes 20-25 s on a
#: 4-core niced slice (verify ~8 s, read ~3 s, fold ~5 s, index sha ~2 s) and grows with the
#: ledger; a 30 s deadline turned a healthy reader into `reader_timeout` + degraded mode.
DEFAULT_READER_TIMEOUT_S = 120.0
ALARM_TIMEOUT_S = 10.0
#: Bounded sleep when degraded mode has nothing admissible (no planner call, no trial consumed).
DEGRADED_BACKOFF_S = 60.0
DEGRADED_LOG_PATH = ORCH_ROOT / "logs" / "loop_beliefs_degraded.jsonl"
REVIEW_FLAGS_PATH = ORCH_ROOT / "logs" / "loop_beliefs_review_flags.jsonl"

EXIT_READY, EXIT_REFUSED = 0, 3
#: DESIGN §7.1 evidence tiers. Only `target` and `sibling` ids may be cited as premises; `host`
#: (device/backend-only, e.g. INF-70 bandwidth ceilings) is background and is refused.
TIERS = ("target", "sibling", "host")

Runner = Callable[..., Any]


class BeliefContextRefused(RuntimeError):
    """Planning was attempted without a validated `ready` belief answer."""


@dataclass(frozen=True)
class BeliefSettings:
    """Belief-reader wiring for one AutoPilot process (argv on `start`; no off switch).

    `alarm_state`, `alarm_config` and `index` exist so tests can keep the child away from real
    alarms and the canonical index; production leaves them None (reader defaults)."""

    root_repo: Path = DEFAULT_ROOT_REPO
    ledger: Path | None = None
    known_good: Path | None = None
    escalate_after_s: float = DEFAULT_ESCALATE_AFTER_S
    reader_timeout_s: float = DEFAULT_READER_TIMEOUT_S
    alarm_state: Path | None = None
    alarm_config: Path | None = None
    index: Path | None = None
    consumer: str = CONSUMER
    python: str = field(default_factory=lambda: sys.executable)
    degraded_log: Path = DEGRADED_LOG_PATH
    review_flags: Path = REVIEW_FLAGS_PATH

    @property
    def reader(self) -> Path:
        return Path(self.root_repo) / "scripts" / "vidya" / "loop_beliefs.py"

    def _alarm_args(self) -> list[str]:
        args: list[str] = []
        if self.alarm_state:
            args += ["--alarm-state", str(self.alarm_state)]
        if self.alarm_config:
            args += ["--alarm-config", str(self.alarm_config)]
        return args

    def describe(self) -> dict[str, Any]:
        return {"root_repo": str(self.root_repo), "reader": str(self.reader),
                "ledger": str(self.ledger) if self.ledger else None,
                "known_good": str(self.known_good) if self.known_good else None,
                "escalate_after_s": self.escalate_after_s,
                "reader_timeout_s": self.reader_timeout_s, "consumer": self.consumer}


def resolve_root_repo(explicit: str | Path | None = None) -> Path:
    """`explicit` (argv) wins verbatim. Otherwise the first of DEFAULT_ROOT_REPO and
    ROOT_REPO_FALLBACKS that actually holds the reader; none -> DEFAULT_ROOT_REPO, so a missing
    reader still raises the critical `reader_missing` alarm in `prepare`."""
    if explicit:
        return Path(explicit)
    for candidate in (DEFAULT_ROOT_REPO, *ROOT_REPO_FALLBACKS):
        if (candidate / "scripts" / "vidya" / "loop_beliefs.py").is_file():
            if candidate != DEFAULT_ROOT_REPO:
                log.warning("belief reader not found under %s; using %s", DEFAULT_ROOT_REPO,
                            candidate)
            return candidate
    return DEFAULT_ROOT_REPO


# ------------------------------------------------------------------------------ targets


def _role_backend(launch_runtime: Mapping[str, Any]) -> str:
    """gpu when the role's launch flags place it on a ROCm/HIP/CUDA device (or offload layers),
    else cpu -- the same two backends `loop_beliefs.Target` accepts."""
    flags = launch_runtime.get("flags") if isinstance(launch_runtime, Mapping) else None
    flags = flags if isinstance(flags, Mapping) else {}
    device = str(flags.get("device") or "").strip().lower()
    if device.startswith(("rocm", "hip", "cuda")):
        return "gpu"
    layers = flags.get("n_gpu_layers")
    if isinstance(layers, str) and layers.strip().lower() == "all":
        return "gpu"
    if isinstance(layers, int) and not isinstance(layers, bool) and layers > 0:
        return "gpu"
    return "cpu"


def stack_targets(records: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """Belief targets for the LIVE stack: one per distinct (model file, quant, backend), read from
    the compiled stack priors AutoPilot already assembles its stack from
    (`src.registry.stack_priors.live_stack_role_records`). `label` lists the roles sharing it.
    A role without a model path is skipped (an incomplete target is an error, not a wildcard)."""
    if records is None:
        from src.registry.stack_priors import live_stack_role_records

        records = live_stack_role_records()
    merged: dict[tuple[str, str, str], dict[str, Any]] = {}
    for role, record in (records or {}).items():
        if not isinstance(record, Mapping):
            continue
        launch = ((record.get("serving") or {}).get("launch") or {})
        model_path = str((launch.get("requirements") or {}).get("model_path") or "").strip()
        if not model_path:
            continue
        quant = str((record.get("model") or {}).get("quant") or "").strip()
        backend = _role_backend(launch.get("runtime") or {})
        key = (Path(model_path).name, quant.upper(), backend)
        entry = merged.setdefault(key, {"model_path": model_path, "quant": quant,
                                        "backend": backend, "_roles": []})
        entry["_roles"].append(str(record.get("role") or role))
    out = []
    for entry in merged.values():
        roles = sorted(set(entry.pop("_roles")))
        out.append({**entry, "label": "stack:" + ",".join(roles)})
    return sorted(out, key=lambda t: (t["backend"], t["model_path"], t["quant"]))


def hypothesis_targets(hypotheses: Iterable[Any]) -> list[dict[str, Any]]:
    """Targets an OPEN operator hypothesis names explicitly in its `proposed_action`: a
    `model_path`/`model_file` ending in .gguf plus an explicit `backend` (cpu|gpu). Anything less
    is skipped, never guessed."""
    out = []
    for item in hypotheses:
        action = getattr(item, "proposed_action", None)
        if not getattr(item, "is_open", False) or not isinstance(action, Mapping):
            continue
        model = str(action.get("model_path") or action.get("model_file") or "").strip()
        backend = str(action.get("backend") or "").strip().lower()
        if not model.lower().endswith(".gguf") or backend not in ("cpu", "gpu"):
            continue
        target: dict[str, Any] = {"model_path": model, "backend": backend,
                                  "quant": str(action.get("quant") or "").strip(),
                                  "label": f"hypothesis:{getattr(item, 'id', '?')}"}
        if isinstance(action.get("context_tokens"), int) and not isinstance(
                action.get("context_tokens"), bool):
            target["context_tokens"] = action["context_tokens"]
        out.append(target)
    return out


def derive_targets(*, records: Mapping[str, Any] | None = None,
                   hypotheses: Iterable[Any] | None = None,
                   hypotheses_loader: Callable[[], Iterable[Any]] | None = None
                   ) -> list[dict[str, Any]]:
    """Targets for this turn's `prepare`: the live stack plus models open hypotheses name.
    Deduplicated on (model file, quant, backend). Never raises: an empty or unreadable stack
    logs a warning and yields [] (the planner then gets ledger-wide flags only)."""
    targets: list[dict[str, Any]] = []
    try:
        targets = stack_targets(records)
    except Exception as exc:  # noqa: BLE001 -- belief context must not crash the loop
        log.warning("belief targets: live stack unreadable (%s: %s)", type(exc).__name__, exc)
    if not targets:
        log.warning("belief targets: no live stack roles resolved; asking for ledger-wide "
                    "flags only (no per-target evidence this turn)")
    try:
        hyps = list(hypotheses) if hypotheses is not None else (
            list(hypotheses_loader()) if hypotheses_loader is not None else [])
        extra = hypothesis_targets(hyps)
    except Exception as exc:  # noqa: BLE001
        log.warning("belief targets: operator hypotheses unreadable (%s: %s)",
                    type(exc).__name__, exc)
        extra = []
    seen = {(Path(t["model_path"]).name, str(t["quant"]).upper(), t["backend"]) for t in targets}
    for target in extra:
        key = (Path(target["model_path"]).name, str(target["quant"]).upper(), target["backend"])
        if key not in seen:
            seen.add(key)
            targets.append(target)
    return targets


# ------------------------------------------------------------------------------ sealed answers

_SEAL = object()


@dataclass(frozen=True)
class ReadyBeliefs:
    """A validated `ready` answer. Only `validate_ready` can build one (sealed constructor)."""

    answer: Mapping[str, Any]
    _seal: object = field(repr=False, compare=False, default=None)

    def __post_init__(self) -> None:
        if self._seal is not _SEAL:
            raise BeliefContextRefused("ReadyBeliefs can only be built by validate_ready()")

    @property
    def ledger_state(self) -> dict[str, Any]:
        return dict(self.answer["ledger_state"])

    @property
    def ledger_state_digest(self) -> str:
        return str(self.answer.get("ledger_state_digest") or "")

    @property
    def evidence(self) -> dict[str, Any]:
        return dict(self.answer["evidence"])

    @property
    def known_good_path(self) -> str:
        return str(self.answer.get("known_good_path") or "")


def _check_answer(answer: Any) -> None:
    if not isinstance(answer, Mapping) or answer.get("schema") != ANSWER_SCHEMA:
        raise ValueError("not a loop_beliefs answer")
    if answer.get("status") != "ready":
        raise ValueError(f"belief context refused: {answer.get('reason')}")
    state = answer.get("ledger_state")
    if (not isinstance(state, Mapping) or state.get("schema") != LEDGER_STATE_SCHEMA
            or state.get("fresh") is not True
            or not isinstance(state.get("frame_count"), int)
            or isinstance(state.get("frame_count"), bool)
            or not state.get("head_hash")
            or not isinstance(answer.get("evidence"), Mapping)):
        raise ValueError("belief context lacks a verified ledger state")
    if not answer.get("ledger_state_digest"):
        raise ValueError("belief context lacks a ledger_state_digest")
    _check_v2_keys(answer["evidence"])


def _num(value: Any) -> float | None:
    """The one tolerant numeric parse: the reader may send a number or a decimal string."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _check_v2_keys(evidence: Mapping[str, Any]) -> None:
    """DESIGN §7.4 answer keys (`tiers`, `tried_index`, `nonempty_target`, `banner`, per-claim
    `tier`). ALL OPTIONAL: an older answer carries none and passes. A key that is present must
    have its documented shape, so a malformed one is refused instead of rendered."""
    for key in ("tiers", "tiers_shown"):  # both live under `evidence`, never top level
        tiers = evidence.get(key)
        if tiers is not None and (not isinstance(tiers, Mapping) or any(
                k not in TIERS or _num(v) is None for k, v in tiers.items())):
            raise ValueError(f"belief evidence.{key} is malformed")
    banner = evidence.get("banner")
    if banner is not None and not isinstance(banner, str):
        raise ValueError("belief evidence.banner is malformed")
    for key in ("host_claim_ids", "not_evidence_claim_ids", "refuted_claim_ids"):
        ids = evidence.get(key)
        if ids is not None and (not isinstance(ids, (list, tuple))
                                or any(not isinstance(i, str) for i in ids)):
            raise ValueError(f"belief evidence.{key} is malformed")
    index = evidence.get("tried_index")
    if index is not None and not isinstance(index, (str, list, tuple, Mapping)):
        raise ValueError("belief evidence.tried_index is malformed")
    nonempty = evidence.get("nonempty_target")
    if nonempty is not None and not isinstance(nonempty, bool):
        raise ValueError("belief evidence.nonempty_target is malformed")
    claims = evidence.get("claims")
    if claims is not None:
        if not isinstance(claims, (list, tuple)):
            raise ValueError("belief evidence.claims is malformed")
        for row in claims:
            if isinstance(row, Mapping) and row.get("tier") is not None \
                    and row.get("tier") not in TIERS:
                raise ValueError(f"belief evidence claim tier {row.get('tier')!r} is unknown")


def validate_ready(answer: Any) -> ReadyBeliefs:
    """Mirror of `loop_beliefs.validate_ready`; raises ValueError on anything but a verified,
    current `ready` answer."""
    _check_answer(answer)
    return ReadyBeliefs(answer=json.loads(json.dumps(answer, default=str)), _seal=_SEAL)


def require_ready(obj: Any) -> ReadyBeliefs:
    """The planner-drafting guard. Anything but a sealed, re-validating ReadyBeliefs raises --
    including a `SeedReviewToken` (degraded-mode review never licenses drafting)."""
    if not isinstance(obj, ReadyBeliefs) or obj._seal is not _SEAL:
        raise BeliefContextRefused(
            "planner drafting requires a validated `ready` belief answer "
            f"(got {type(obj).__name__})")
    try:
        _check_answer(obj.answer)
    except ValueError as exc:
        raise BeliefContextRefused(str(exc)) from exc
    return obj


@dataclass(frozen=True)
class SeedReviewToken:
    """License to send ONE preseeded item to the critic in degraded mode. Minted only by
    `mint_seed_review` from an admissible `check-seeds` row checked against a known-good state."""

    item_id: str
    action: Mapping[str, Any]
    rationale: Mapping[str, Any]
    premises: tuple[str, ...]
    seeded_against: Any
    checked_against_digest: str
    checked_against: Mapping[str, Any] | None
    seed_check_row: Mapping[str, Any]
    description: str = ""
    _seal: object = field(repr=False, compare=False, default=None)

    def __post_init__(self) -> None:
        if self._seal is not _SEAL:
            raise BeliefContextRefused("SeedReviewToken can only be minted by mint_seed_review()")


def mint_seed_review(item: Mapping[str, Any], seed_check: Mapping[str, Any]) -> SeedReviewToken:
    """Raises BeliefContextRefused unless `seed_check` is a seed_check.v1 result, checked against
    a known-good state, whose row for `item` is admissible with the same premises/seeded_against."""
    if not isinstance(seed_check, Mapping) or seed_check.get("schema") != SEED_CHECK_SCHEMA:
        raise BeliefContextRefused("seed review needs a check-seeds result")
    digest = seed_check.get("checked_against_digest")
    if not digest:
        raise BeliefContextRefused("seed review needs a known-good ledger state")
    item_id = str(item.get("id", ""))
    row = next((r for r in seed_check.get("items") or [] if r.get("id") == item_id), None)
    if row is None or row.get("admissible") is not True or row.get("reasons"):
        raise BeliefContextRefused(f"seed {item_id!r} is not admissible: "
                                   f"{(row or {}).get('reasons')}")
    premises = [str(p) for p in item.get("premises") or []]
    if [str(p) for p in row.get("premises") or []] != premises:
        raise BeliefContextRefused(f"seed {item_id!r}: checked premises differ from the item")
    if row.get("seeded_against") != item.get("seeded_against") or not item.get("seeded_against"):
        raise BeliefContextRefused(f"seed {item_id!r}: seeded_against differs from the check")
    action = item.get("action")
    if not isinstance(action, Mapping) or not action.get("type"):
        raise BeliefContextRefused(f"seed {item_id!r} carries no action")
    return SeedReviewToken(
        item_id=item_id, action=json.loads(json.dumps(dict(action), default=str)),
        rationale=json.loads(json.dumps(dict(item.get("rationale") or {}), default=str)),
        premises=tuple(premises), seeded_against=item.get("seeded_against"),
        checked_against_digest=str(digest), checked_against=seed_check.get("checked_against"),
        seed_check_row=dict(row), description=str(item.get("description") or ""), _seal=_SEAL)


def require_seed_review(obj: Any) -> SeedReviewToken:
    if not isinstance(obj, SeedReviewToken) or obj._seal is not _SEAL:
        raise BeliefContextRefused(
            f"degraded-mode review requires a minted SeedReviewToken (got {type(obj).__name__})")
    return obj


# ------------------------------------------------------------------------------ child process


@dataclass
class PrepareOutcome:
    status: str  # "ready" | "refused"
    ready: ReadyBeliefs | None
    reason: str = ""
    detail: str = ""
    answer: dict[str, Any] | None = None
    consumer_alarm: dict[str, Any] | None = None
    last_known_good: dict[str, Any] | None = None
    seconds: float = 0.0

    def refusal(self) -> dict[str, Any]:
        return {"reason": self.reason, "detail": self.detail[:300],
                "consumer_alarm": self.consumer_alarm,
                "last_known_good_digest": (self.last_known_good or {}).get("ledger_state_digest")}


def _fallback_alarm(settings: BeliefSettings, reason: str, detail: str, run: Runner
                    ) -> dict[str, Any]:
    """The reader itself is missing, so its `alarm` subcommand is too: raise a CRITICAL alarm
    straight through ROOT's alarm channel (a missing reader is a deployment defect, not a
    transient). Best effort; never raises."""
    channel = settings.reader.parents[2] / "scripts" / "coordination" / "alarm_channel.py"
    out: dict[str, Any] = {"invoked": False, "channel": str(channel)}
    if not channel.is_file():
        out["error"] = "alarm channel missing"
        return out
    argv = [settings.python, str(channel)]
    if settings.alarm_state is not None:
        argv += ["--state", str(settings.alarm_state)]
    if settings.alarm_config is not None:
        argv += ["--config", str(settings.alarm_config)]
    argv += ["raise", "--severity", "critical", "--key",
             f"belief-ledger-not-current-escalated.{settings.consumer}", "--message",
             f"{settings.consumer}: belief reader missing ({reason}); no new strategies are "
             "being generated", "--evidence", json.dumps({"reason": reason,
                                                         "detail": detail[:300]})]
    try:
        proc = run(argv, capture_output=True, text=True, check=False, timeout=ALARM_TIMEOUT_S)
        out.update(invoked=True, returncode=getattr(proc, "returncode", None))
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"[:300]
    return out


def raise_alarm(settings: BeliefSettings, reason: str, detail: str, *,
                runner: Runner | None = None) -> dict[str, Any]:
    """Best-effort `alarm` subcommand for a consumer-side failure. Never raises."""
    run = runner or subprocess.run
    out: dict[str, Any] = {"reason": reason, "invoked": False}
    if not settings.reader.is_file():
        out["error"] = f"reader missing at {settings.reader}; alarm subcommand unavailable"
        out["fallback"] = _fallback_alarm(settings, reason, detail, run)
        if not out["fallback"].get("invoked"):
            log.critical("BELIEF LEDGER ALARM UNDELIVERABLE (%s): %s", reason, out["error"])
        return out
    argv = [settings.python, str(settings.reader), "alarm", "--consumer", settings.consumer,
            "--reason", reason, "--detail", detail[:500],
            "--escalate-after-s", str(settings.escalate_after_s), *settings._alarm_args()]
    try:
        proc = run(argv, capture_output=True, text=True, check=False,
                   timeout=max(0.1, min(ALARM_TIMEOUT_S, settings.reader_timeout_s)))
        out["invoked"] = True
        out["returncode"] = getattr(proc, "returncode", None)
        try:
            out["result"] = json.loads(getattr(proc, "stdout", "") or "null")
        except ValueError:
            out["result"] = None
    except Exception as exc:  # noqa: BLE001 -- the refusal stands even if the alarm fails
        out["error"] = f"{type(exc).__name__}: {exc}"[:300]
    return out


def prepare(settings: BeliefSettings, targets: Iterable[Mapping[str, Any]] | None = None, *,
            runner: Runner | None = None) -> PrepareOutcome:
    """Run `loop_beliefs.py prepare` for this planner turn. Never raises."""
    run = runner or subprocess.run
    started = time.monotonic()

    def refused(reason: str, detail: str, *, alarm: bool, answer: dict | None = None
                ) -> PrepareOutcome:
        consumer_alarm = raise_alarm(settings, reason, detail, runner=runner) if alarm else None
        return PrepareOutcome("refused", None, reason, detail, answer, consumer_alarm,
                              (answer or {}).get("last_known_good"),
                              round(time.monotonic() - started, 3))

    if not settings.reader.is_file():
        return refused("reader_missing", str(settings.reader), alarm=True)
    argv = [settings.python, str(settings.reader), "prepare", "--consumer", settings.consumer,
            "--target-json", json.dumps(list(targets or [])), "--reply-field", REPLY_FIELD,
            "--escalate-after-s", str(settings.escalate_after_s), *settings._alarm_args()]
    if settings.ledger:
        argv += ["--ledger", str(settings.ledger)]
    if settings.known_good:
        argv += ["--known-good", str(settings.known_good)]
    if settings.index:
        argv += ["--index", str(settings.index)]
    try:
        proc = run(argv, capture_output=True, text=True, check=False,
                   timeout=max(0.1, settings.reader_timeout_s))
    except subprocess.TimeoutExpired:
        return refused("reader_timeout", f"no answer within {settings.reader_timeout_s}s",
                       alarm=True)
    except Exception as exc:  # noqa: BLE001
        return refused("reader_crash", f"{type(exc).__name__}: {exc}", alarm=True)
    stdout = str(getattr(proc, "stdout", "") or "")
    returncode = getattr(proc, "returncode", None)
    try:
        answer = json.loads(stdout)
        if not isinstance(answer, dict) or answer.get("schema") != ANSWER_SCHEMA:
            raise ValueError("unexpected schema")
    except (ValueError, TypeError) as exc:
        stderr = str(getattr(proc, "stderr", "") or "")[-300:]
        return refused("reader_garbage", f"exit={returncode}; {exc}; {stderr}", alarm=True)
    if answer.get("status") == "refused" and returncode == EXIT_REFUSED:
        # The reader already raised (and escalates) the operator alarm itself.
        return refused(str(answer.get("reason") or "refused"), str(answer.get("detail") or ""),
                       alarm=False, answer=answer)
    if returncode != EXIT_READY:
        return refused("reader_invalid_answer",
                       f"exit={returncode} status={answer.get('status')}", alarm=True,
                       answer=answer)
    try:
        ready = validate_ready(answer)
    except ValueError as exc:
        return refused("reader_invalid_answer", str(exc), alarm=True, answer=answer)
    return PrepareOutcome("ready", ready, answer=answer,
                          seconds=round(time.monotonic() - started, 3))


def check_seeds(settings: BeliefSettings, items: list[Mapping[str, Any]], *,
                runner: Runner | None = None) -> dict[str, Any] | None:
    """`check-seeds` against the known-good snapshot; None when the child fails (then nothing is
    admissible -- a seed that cannot be checked is not run)."""
    run = runner or subprocess.run
    if not settings.reader.is_file():
        return None
    payload = [{"id": str(i.get("id", "")), "premises": list(i.get("premises") or []),
                "seeded_against": i.get("seeded_against")} for i in items]
    argv = [settings.python, str(settings.reader), "check-seeds", "--consumer", settings.consumer,
            "--items-json", json.dumps(payload, default=str)]
    if settings.known_good:
        argv += ["--known-good", str(settings.known_good)]
    try:
        proc = run(argv, capture_output=True, text=True, check=False,
                   timeout=max(0.1, settings.reader_timeout_s))
        result = json.loads(str(getattr(proc, "stdout", "") or ""))
    except Exception as exc:  # noqa: BLE001
        log.warning("belief check-seeds failed: %s: %s", type(exc).__name__, exc)
        return None
    if (getattr(proc, "returncode", None) != 0 or not isinstance(result, dict)
            or result.get("schema") != SEED_CHECK_SCHEMA
            or not isinstance(result.get("items"), list)):
        log.warning("belief check-seeds returned an invalid answer")
        return None
    return result


# ------------------------------------------------------------------------------ planner side


def known_good_flags(ready: ReadyBeliefs) -> dict[str, Any]:
    """Ledger-wide contested/withdrawn ids from the snapshot `prepare` just wrote (only when its
    digest equals this answer's), unioned with the target evidence's flagged ids."""
    ev = ready.evidence
    contested = set(ev.get("contested_claim_ids") or [])
    withdrawn = set(ev.get("withdrawn_claim_ids") or [])
    aliases: dict[str, str] = {}
    source = "evidence_only"
    path = ready.known_good_path
    if path:
        try:
            body = json.loads(Path(path).read_text())
        except (OSError, ValueError):
            body = None
        if (isinstance(body, dict) and body.get("schema") == KNOWN_GOOD_SCHEMA
                and body.get("ledger_state_digest") == ready.ledger_state_digest):
            contested |= set(body.get("contested") or [])
            withdrawn |= set(body.get("withdrawn") or [])
            aliases = dict(body.get("aliases") or {})
            source = "known_good_snapshot"
    return {"contested": contested, "withdrawn": withdrawn, "aliases": aliases,
            "host": host_claim_ids(ev),
            "refuted": {str(i) for i in ev.get("refuted_claim_ids") or ()
                        if isinstance(i, str)},
            "source": source,
            "ledger_state_digest": ready.ledger_state_digest}


def _claim_rows(ev: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [r for r in ev.get("claims") or () if isinstance(r, Mapping) and r.get("claim_id")]


def host_claim_ids(ev: Mapping[str, Any]) -> set[str]:
    """Ids this answer labelled `tier: host`. An older answer labels none, so none is refused."""
    ids = {str(r["claim_id"]) for r in _claim_rows(ev) if r.get("tier") == "host"}
    for key in ("host_claim_ids", "not_evidence_claim_ids"):
        ids |= {str(i) for i in ev.get(key) or () if isinstance(i, str)}
    return ids


def declared_premises(rationale: Any) -> list[str]:
    raw = rationale.get(REPLY_FIELD) if isinstance(rationale, Mapping) else None
    if not isinstance(raw, list):
        return []
    return [str(cid) for cid in raw if isinstance(cid, str) and cid]


def premise_flag_reason(premises: Iterable[str], flags: Mapping[str, Any] | None) -> str | None:
    """Pure. The skip reason when a declared premise is flagged withdrawn/contested, else None."""
    if not flags:
        return None
    aliases = flags.get("aliases") or {}
    withdrawn = set(flags.get("withdrawn") or ())
    contested = set(flags.get("contested") or ())
    host = set(flags.get("host") or ())
    refuted = set(flags.get("refuted") or ())
    # `host` is last: a withdrawn/contested host claim still reports the stronger reason.
    for status, bucket in (("withdrawn", withdrawn), ("contested", contested),
                           ("refuted", refuted),
                           ("host-tier (background only, not a premise)", host)):
        hits = [p for p in premises if aliases.get(p, p) in bucket]
        if hits:
            return f"vidya premise {status}: claim_id={','.join(sorted(set(hits)))}"
    return None


def _neutralise(text: str) -> str:
    """CT-12.3a: claim text is derived from ingested (partly external) sources, so break chat
    control-token spellings before it reaches a planner prompt. Idempotent with the API-ingress
    neutraliser every HTTP planner provider already crosses; honours its kill switch."""
    try:
        from src.security.control_tokens import neutralise_control_tokens
    except ImportError:  # standalone import without the orchestrator tree: ingress still covers it
        return text
    return neutralise_control_tokens(text)


#: Bounds for the lines this bridge adds around the reader's own section (which has its own cap).
MAX_CITABLE_IDS = 30
MAX_HOST_IDS = 6
MAX_INDEX_CHARS = 3000


def _tried_index_text(index: Any) -> str:
    """The tried-mechanisms index as one bounded line. The reader sends a string or a list of
    ids / `{mechanism_id, outcome}` rows; anything else renders nothing."""
    if isinstance(index, str):
        text = index.strip()
    elif isinstance(index, (list, tuple)):
        parts = []
        for item in index:
            if isinstance(item, Mapping):
                mech = str(item.get("mechanism_id") or item.get("id") or "").strip()
                outcome = str(item.get("outcome") or "").strip()
                if mech:
                    parts.append(f"{mech} {outcome}".strip())
            elif isinstance(item, str) and item.strip():
                parts.append(item.strip())
        text = " · ".join(parts)
    else:
        return ""
    return text[:MAX_INDEX_CHARS]


def _value_lines(ev: Mapping[str, Any], section: str) -> list[str]:
    """The DESIGN §7.4 additions, rendered under the ledger header. Empty for an older answer."""
    lines: list[str] = []
    tiers = ev.get("tiers")
    if isinstance(tiers, Mapping):
        lines.append("  Evidence tiers: " + ", ".join(
            f"{name} {int(_num(tiers.get(name)) or 0)}" for name in TIERS) + ".")
        if ev.get("nonempty_target") is False:
            lines.append("  No belief is scoped to this turn's target; sibling/host lines are "
                         "context, not evidence about it.")
    rows = _claim_rows(ev)
    if any(r.get("tier") for r in rows):
        citable = [str(r["claim_id"]) for r in rows if r.get("tier") in ("target", "sibling")]
        host = [str(r["claim_id"]) for r in rows if r.get("tier") == "host"]
        if citable:
            lines.append(f"  Citable in {REPLY_FIELD} (target/sibling tier): "
                         + ", ".join(citable[:MAX_CITABLE_IDS])
                         + (f" (+{len(citable) - MAX_CITABLE_IDS} more)"
                            if len(citable) > MAX_CITABLE_IDS else ""))
        if host:
            lines.append("  Background only, NOT citable as a premise (host tier): "
                         + ", ".join(host[:MAX_HOST_IDS])
                         + (f" (+{len(host) - MAX_HOST_IDS} more)"
                            if len(host) > MAX_HOST_IDS else ""))
    tried = _tried_index_text(ev.get("tried_index"))
    if tried and "Tried on" not in section:
        lines.append(f"  Tried mechanisms (id outcome): {tried}")
    return lines


def render_prompt_block(ready: ReadyBeliefs) -> str:
    state = ready.ledger_state
    fresh = state.get("freshness") or {}
    header = (f"  Ledger verified current for this turn: {state.get('frame_count')} frames, head "
              f"seq {state.get('head_seq')} ({str(state.get('head_hash'))[:23]}), state "
              f"{ready.ledger_state_digest[:23]}; sync_action={fresh.get('sync_action')}.")
    ev = ready.evidence
    section = str(ev.get("section") or "")
    extra = _value_lines(ev, section)
    if extra:
        header = header + "\n" + "\n".join(extra)
    # Every belief-derived string (section, ids, index text) funnels through ONE neutralise call.
    if section.strip():
        return _neutralise(f"{header}\n{section}")
    return _neutralise(
        f"{header}\n  (no ingested claim applies to this turn's targets: "
        f"{', '.join(ev.get('reasons') or []) or 'none'}; ledger-wide contested/withdrawn "
        "claims are still enforced against declared premises before dispatch)")


def build_planner_receipt(ready: ReadyBeliefs | None, declared: Iterable[str],
                          *, seed_review: SeedReviewToken | None = None) -> dict[str, Any]:
    declared = [str(c) for c in declared]
    if seed_review is not None:
        return {"schema": RECEIPT_SCHEMA, "status": "degraded_seed_review",
                "seed_id": seed_review.item_id, "premises": list(seed_review.premises),
                "seeded_against": seed_review.seeded_against,
                "checked_against_digest": seed_review.checked_against_digest,
                "checked_against": seed_review.checked_against,
                "ledger_state": None, "ledger_state_digest": None}
    if ready is None:
        return {"schema": RECEIPT_SCHEMA, "status": "absent", "ledger_state": None,
                "ledger_state_digest": None, "declared_reliance_claim_ids": declared}
    ev = ready.evidence
    host = host_claim_ids(ev)
    # Host-tier ids were shown (as background) but are never usable premises.
    usable = [str(c) for c in ev.get("claim_ids") or [] if str(c) not in host]
    contested = [str(c) for c in ev.get("contested_claim_ids") or []]
    withdrawn = [str(c) for c in ev.get("withdrawn_claim_ids") or []]
    accepted = sorted(set(declared) & set(usable))
    rejected = []
    for cid in sorted(set(declared) - set(usable)):
        reason = ("withdrawn" if cid in withdrawn else "contested" if cid in contested
                  else "host_tier" if cid in host else "not_presented")
        rejected.append({"claim_id": cid, "reason": reason})
    return {
        "schema": RECEIPT_SCHEMA, "status": "ready", "consumer": ready.answer.get("consumer"),
        "as_of": ready.answer.get("as_of"), "ledger_state": ready.ledger_state,
        "ledger_state_digest": ready.ledger_state_digest,
        "evidence_status": ev.get("status"), "section_sha256": ev.get("section_sha256") or "",
        "exposed_usable_claim_ids": usable, "exposed_contested_claim_ids": contested,
        "exposed_withdrawn_claim_ids": withdrawn,
        "exposed_host_claim_ids": sorted(host),
        "evidence_tiers": dict(ev["tiers"]) if isinstance(ev.get("tiers"), Mapping) else None,
        "nonempty_target": ev.get("nonempty_target"),
        "section_bytes": ev.get("section_bytes"),
        "declared_reliance_claim_ids": sorted(set(declared)),
        "accepted_reliance_claim_ids": accepted, "rejected_reliance": rejected,
        "reliance_status": "planner_declared" if accepted else "not_declared",
    }


# ------------------------------------------------------------------------------ degraded mode


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as fh:
            fh.write(json.dumps(dict(row), default=str, separators=(",", ":")) + "\n")
    except OSError as exc:
        log.warning("belief degraded log append failed (%s): %s", path, exc)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = Path(path).read_text().splitlines()
    except OSError:
        return rows
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def degraded_excluded_ids(path: Path) -> set[str]:
    """Seeds already dispatched, refused by the critic / a deterministic guard, or skipped at
    dispatch, in degraded mode -- never re-reviewed every turn."""
    out: set[str] = set()
    for row in read_jsonl(path):
        if row.get("kind") == "degraded_dispatch":
            out.add(str(row.get("item_id")))
        elif row.get("kind") == "degraded_review" and row.get("verdict") in {
                "reject", "revise", "deterministic_block"}:
            out.add(str(row.get("item_id")))
        elif row.get("kind") == "degraded_superseded" and row.get("skip_reason"):
            out.add(str(row.get("item_id")))
    return out


def operator_seed_items(hypotheses: Iterable[Any], *, excluded: set[str],
                        blacklist_conflicts: Mapping[str, str] | None = None
                        ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Open operator hypotheses with a `proposed_action`, as check-seeds work items."""
    items: list[dict[str, Any]] = []
    ineligible: list[dict[str, Any]] = []
    conflicts = blacklist_conflicts or {}
    for hyp in hypotheses:
        item_id = f"operator_hypothesis:{hyp.id}"
        if not getattr(hyp, "is_open", False):
            continue
        if not hyp.proposed_action:
            ineligible.append({"id": item_id, "reasons": ["no_proposed_action"]})
            continue
        if item_id in excluded:
            ineligible.append({"id": item_id, "reasons": ["already_handled_in_degraded_mode"]})
            continue
        if hyp.id in conflicts:
            ineligible.append({"id": item_id, "reasons": ["blacklisted"],
                               "detail": str(conflicts[hyp.id])[:200]})
            continue
        seeded_against = getattr(hyp, "seeded_against", None) or {
            "kind": "operator", "author": hyp.stated_by, "at": hyp.stated}
        premises = [str(c) for c in getattr(hyp, "vidya_claim_ids", ()) or ()]
        items.append({
            "id": item_id, "source": "operator_hypothesis", "premises": premises,
            "seeded_against": seeded_against, "action": dict(hyp.proposed_action),
            "rationale": {"falsifier": hyp.falsifier, "rubric_scores": {},
                          REPLY_FIELD: premises},
            "description": (f"Operator hypothesis {hyp.id} (stated by {hyp.stated_by} on "
                            f"{hyp.stated}, confidence {hyp.confidence}): {hyp.hypothesis}"),
        })
    return items, ineligible


@dataclass
class DegradedSelection:
    item: dict[str, Any] | None
    seed_check: dict[str, Any] | None
    skipped: list[dict[str, Any]]
    reason: str = ""


def select_degraded_item(settings: BeliefSettings, items: list[dict[str, Any]], *,
                         runner: Runner | None = None) -> DegradedSelection:
    """First admissible seed (operator order) per `check-seeds` against the last known-good."""
    if not items:
        return DegradedSelection(None, None, [], "no_preseeded_items")
    result = check_seeds(settings, items, runner=runner)
    if result is None:
        return DegradedSelection(None, None, [], "seed_check_unavailable")
    rows = {str(r.get("id")): r for r in result.get("items") or []}
    skipped, chosen = [], None
    for item in items:
        row = rows.get(item["id"])
        if row is None:
            skipped.append({"id": item["id"], "reasons": ["missing_from_seed_check"]})
        elif row.get("admissible") is True and not row.get("reasons"):
            chosen = chosen or item
        else:
            skipped.append({"id": item["id"], "reasons": list(row.get("reasons") or []),
                            "flagged_premises": list(row.get("flagged_premises") or []),
                            "seeded_against": row.get("seeded_against")})
    return DegradedSelection(chosen, result, skipped,
                             "" if chosen else "no_admissible_seed")


def degraded_row(kind: str, settings: BeliefSettings, **fields: Any) -> dict[str, Any]:
    return {"schema": DEGRADED_SCHEMA, "kind": kind, "ts": time.time(),
            "consumer": settings.consumer, **fields}


def recovery_recheck(settings: BeliefSettings, ready: ReadyBeliefs, *,
                     runner: Runner | None = None) -> dict[str, Any]:
    """On `ready`: re-check every degraded-mode dispatch not yet re-checked against the fresh
    snapshot; append a review flag for each whose premise is now flagged. Never reverses."""
    rows = read_jsonl(settings.degraded_log)
    done = {str(rid) for row in rows if row.get("kind") == "recovery_recheck"
            for rid in row.get("receipt_ids") or []}
    pending = [row for row in rows if row.get("kind") == "degraded_dispatch"
               and str(row.get("receipt_id")) not in done]
    if not pending:
        return {"pending": 0, "flagged": []}
    result = check_seeds(settings, [{"id": row["receipt_id"], "premises": row.get("premises"),
                                     "seeded_against": row.get("seeded_against")}
                                    for row in pending], runner=runner)
    if result is None:
        return {"pending": len(pending), "flagged": [], "error": "seed_check_unavailable"}
    if result.get("checked_against_digest") != ready.ledger_state_digest:
        # Not the snapshot this ready answer wrote (known_good_error); retry next ready.
        return {"pending": len(pending), "flagged": [], "error": "snapshot_digest_mismatch"}
    by_id = {str(r.get("id")): r for r in result.get("items") or []}
    flagged = []
    for row in pending:
        check = by_id.get(str(row["receipt_id"])) or {}
        if check.get("flagged_premises"):
            flag = {"schema": REVIEW_FLAG_SCHEMA, "ts": time.time(), "consumer": settings.consumer,
                    "receipt_id": row["receipt_id"], "trial_id": row.get("trial_id"),
                    "item_id": row.get("item_id"), "action": row.get("action"),
                    "flagged_premises": check.get("flagged_premises"),
                    "seeded_against": row.get("seeded_against"),
                    "dispatched_against_digest": row.get("checked_against_digest"),
                    "checked_against_digest": result.get("checked_against_digest"),
                    "review": "premise refuted after a degraded-mode run; operator review",
                    "auto_reversal": False}
            append_jsonl(settings.review_flags, flag)
            flagged.append(flag)
    append_jsonl(settings.degraded_log, degraded_row(
        "recovery_recheck", settings, receipt_ids=[row["receipt_id"] for row in pending],
        checked_against_digest=result.get("checked_against_digest"),
        flagged_receipt_ids=[f["receipt_id"] for f in flagged]))
    return {"pending": len(pending), "flagged": flagged}


def receipt_id(trial_id: Any, item_id: str) -> str:
    return f"{trial_id}:{hashlib.sha256(item_id.encode()).hexdigest()[:12]}:{item_id}"


__all__ = [
    "BeliefContextRefused", "BeliefSettings", "DegradedSelection", "PrepareOutcome",
    "ReadyBeliefs", "SeedReviewToken", "build_planner_receipt", "check_seeds",
    "declared_premises", "degraded_excluded_ids", "degraded_row", "host_claim_ids",
    "known_good_flags",
    "mint_seed_review", "operator_seed_items", "premise_flag_reason", "prepare", "raise_alarm",
    "recovery_recheck", "render_prompt_block", "require_ready", "require_seed_review",
    "select_degraded_item", "validate_ready",
]
