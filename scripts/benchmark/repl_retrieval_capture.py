#!/usr/bin/env python3
"""Belief-kernel write side for the UFH-12 retrieval eval and the embedder speed block.

Tasks VB-UFH12-RETR (REPL-EMB-2.2/2.4) and VB-UFH12-EMBSPEED (REPL-EMB-2.7 speed block), filed in
``handoffs/active/vidya-belief-substrate-program.md``. The eval harness imports this module and
calls :class:`RetrievalRun` / :class:`SpeedRun`; ``finalize()`` writes one immutable record
(``<run_id>.record.json``, or ``<run_id>.speed.record.json``) and, beside it, the belief sidecar
(``<run_id>.belief_measurements.jsonl`` / ``<run_id>.speed.belief_measurements.jsonl``) with one
self-hashed row per claim. The epyc-root adapter ``scripts/vidya/adapters/repl_retrieval_eval.py``
loads THIS file (pinned by sha256) and calls :func:`validate_row` / :func:`validate_rows`, so the
writer and the reader share one definition of a well-formed row.

TWO INSTRUMENTS, NEVER POOLED. ``offline_recall`` is the offline retrieval eval (recall@k, paired
differences, per-arm cost); ``offline_speed`` is the speed block (throughput, latency, RSS). The
online ``spill_pointer_follows`` shadow (REPL-EMB-2.3) is a third instrument and has no writer
here: a row whose instrument is not one of those two is refused, a sidecar holds exactly one
instrument, and each source kind accepts exactly one.

THE PRE-REGISTERED RULE RIDES IN EVERY RECALL ROW (:data:`REGISTERED_RULE`, REPL-EMB-2.1: primary
k=5, margin M=0.10, equivalence X=0.05, n>=120, paired-bootstrap 95% lower bound > 0). A record that
carries different values is refused unless it names an ``amendment`` (dated approval + reference);
an amended rule is flagged in the claim text and in ``extra.provenance.rule``. The writer never
computes the kill/selection verdict; the producer's verdict is not projected.

ARM IDENTITY (:data:`ARM_FIELDS`): arm_id, model, revision, dim, dtype_or_quant, runtime,
prompt_hash, index_identity_digest, and ``role`` (BASELINE | CANDIDATE) -- ``role`` is the
ClaimTuple ``category``, which cannot be derived from the other fields. A lexical-mode arm carries
the dense-only fields as explicit nulls (an embedding model never produced a lexical number); a
dense or hybrid arm carries them all. One ``arm_id`` keeps one identity for the whole run.

QUERY CATEGORY (the ``category`` argument of ``add_recall`` / ``add_paired_diff``) is the query
stratum, a key of ``query_digests`` (for example ``probe45``, ``spill_follow``, ``context_bundle``).
The reserved stratum ``all`` is the pooled decision set; a recall or paired-diff row on it with
n below the rule's n_min is refused. Within one stratum every recall row must share one n (same
queries for every arm: a failure is a miss, never a drop). Stratum and mode are measurand axes, so
they are part of the locator.

THE SPEED BLOCK is only ever claimed with its context and its noise floor. ``SpeedRun`` requires
the full timing context (cpu-list, observed affinity subset, region-lock claim tag, loadavg at both
ends, frequency governor / range / cap flag) at construction, and ``finalize()`` raises if any
speed row has no A/A noise floor for its (arm, metric), carried in the metric's own unit and with
its replication unit (arm | session | process) and n.

NO GRADE IS WRITTEN. ``protocol_id`` is empty on every row -- no protocol for retrieval recall or
embedder speed is codified under ``measurement/protocols/`` -- so ``claim_tuple.grade()`` caps every
row at an observation. Never invent a field: a missing or malformed required element raises
:class:`CaptureError` (writer) or is listed by :func:`validate_row` (reader); nothing is defaulted.

USAGE (the harness owns the timing; this module only records)::

    import repl_retrieval_capture as cap

    window = cap.HostWindow("24-71", "<held region-lock claim tag>")
    window.begin()          # before the first timed leg
    ...                     # run the legs
    window.finish()         # after the last one
    run = cap.RetrievalRun(
        run_dir, run_id, cap.registered_rule(), corpus_digests, query_digests, label_digests,
        arm_manifest_sha256, window.context(),
        frozen_at_launch_path=run_dir / "FROZEN-AT-LAUNCH.sha256", harness_path=__file__)
    run.add_recall(arm, "probe45", "hybrid", 5, "recall", hits / n, n, hits, [lo, hi], config)
    run.add_paired_diff(arm, lexical, "all", 5, diff, [lo, hi], n, 10000, seed,
                        mode_a="hybrid", mode_b="lexical")
    run.add_cost(arm, cpu_s, p50_ms, p95_ms, build_s, n_queries, mode="hybrid")
    sidecar = run.finalize()
    speed = cap.SpeedRun(run_dir, run_id, corpus_digests, query_digests, arm_manifest_sha256,
                         window.context())
    speed.add_speed(arm, "chunks_per_s", v, "chunks/s", n, spread, "chunks/s", config)
    speed.add_noise_floor(arm, "chunks_per_s", aa, "chunks/s", replication_unit="arm", n=3)
    speed.finalize()

BEHAVIOUR-NEUTRAL. Nothing here times, spawns or signals anything. :class:`HostWindow` reads
``/proc`` and ``/sys`` strictly before the first and after the last timed leg.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import socket
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

CAPTURE_SCHEMA = "epyc.vidya.repl_retrieval_capture.v1"
RETR_RECORD_SCHEMA = "epyc.repl_retrieval_eval.record.v1"
SPEED_RECORD_SCHEMA = "epyc.repl_embedder_speed.record.v1"
RETR_SOURCE_KIND = "repl-retrieval-eval"
SPEED_SOURCE_KIND = "repl-embedder-speed"
INSTRUMENT_RECALL = "offline_recall"
INSTRUMENT_SPEED = "offline_speed"
INSTRUMENTS = (INSTRUMENT_RECALL, INSTRUMENT_SPEED)
SOURCE_INSTRUMENT = {RETR_SOURCE_KIND: INSTRUMENT_RECALL, SPEED_SOURCE_KIND: INSTRUMENT_SPEED}
SIDECAR_SUFFIX = ".belief_measurements.jsonl"
RECORD_SUFFIX = ".record.json"
#: The hook landed 2026-10-20. The pilot ``artifacts/research/emb-speed-20261020/results.jsonl``
#: is retrospective and contended; it is never replayed through this writer.
HOOK_SINCE = "2026-10-20T00:00:00+00:00"

#: REPL-EMB-2.1, registered 2026-09-27 (operator-approved). Changing any value needs a dated,
#: operator-approved amendment, carried as ``rule["amendment"]``.
REGISTERED_RULE: dict[str, Any] = {
    "registration": "REPL-EMB-2.1",
    "registered_on": "2026-09-27",
    "k": 5,
    "M": 0.10,
    "X": 0.05,
    "n_min": 120,
    "ci_level": 0.95,
    "ci_condition": "paired-bootstrap lower bound > 0",
}
RULE_KEYS = tuple(REGISTERED_RULE)

RECALL_KS = (1, 3, 5, 10)
MODES = ("lexical", "dense", "hybrid")
ROLES = ("BASELINE", "CANDIDATE")
POOLED_CATEGORY = "all"
KINDS_RETR = ("recall", "paired_diff", "cost")
KIND_SPEED = "speed"
REPLICATION_UNITS = ("arm", "session", "process")

ARM_FIELDS = (
    "arm_id",
    "model",
    "revision",
    "dim",
    "dtype_or_quant",
    "runtime",
    "prompt_hash",
    "index_identity_digest",
    "role",
)
DENSE_ONLY_FIELDS = ("model", "revision", "dim", "dtype_or_quant", "prompt_hash")

HOST_BASE_KEYS = ("hostname", "cpu_list")
HOST_TIMING_KEYS = ("region_lock_claim", "cpus_allowed_observed", "loadavg", "frequency")

#: cost metric -> (unit, direction)
COST_METRICS: dict[str, tuple[str, str]] = {
    "cpu_s_per_query": ("cpu-s/query", "lower_better"),
    "query_p50_ms": ("ms", "lower_better"),
    "query_p95_ms": ("ms", "lower_better"),
    "index_build_s": ("s", "lower_better"),
}
#: speed metric -> (unit, direction). The unit strings are exact; a different spelling is refused.
SPEED_METRICS: dict[str, tuple[str, str]] = {
    "chunks_per_s": ("chunks/s", "higher_better"),
    "tokens_per_s": ("tokens/s", "higher_better"),
    "query_p50_ms": ("ms", "lower_better"),
    "query_p95_ms": ("ms", "lower_better"),
    "query_max_ms": ("ms", "lower_better"),
    "cpu_s_per_query": ("cpu-s/query", "lower_better"),
    "index_build_s_4096": ("s", "lower_better"),
    "rss_mb": ("MB", "lower_better"),
    "disk_mb": ("MB", "lower_better"),
}
RECALL_UNIT = "fraction of queries with >=1 returned pointer overlapping a gold span"
DIFF_UNIT = "recall difference (A minus B, paired by query)"

ROW_FIELDS = (
    "schema",
    "source_kind",
    "instrument",
    "run_id",
    "producer",
    "emitted_at",
    "date",
    "measurement_id",
    "kind",
    "metric",
    "arm_id",
    "mode",
    "query_category",
    "k",
    "value",
    "unit",
    "metric_direction",
    "category",
    "claim",
    "protocol_id",
    "reps",
    "reps_basis",
    "samples",
    "record_file",
    "record_sha256",
    "extra",
    "row_sha256",
)
EXTRA_FIELDS = ("locator", "arm", "against", "config_echo", "provenance", "provenance_sha256")
PROVENANCE_FIELDS = ("instrument", "run", "rule", "digests", "host_context")
#: the elements MEASUREMENT.md:13 and ClaimTuple require of every claim
CLAIM_TUPLE_FIELDS = (
    "metric",
    "protocol_id",
    "reps",
    "reps_basis",
    "date",
    "category",
    "metric_direction",
    "unit",
    "record_sha256",
)
NOISE_FIELDS = ("aa_spread", "unit", "replication_unit", "n", "scope")


class CaptureError(RuntimeError):
    """The capture refuses; nothing is written."""


# ------------------------------------------------------------------ hashing


def canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def content_hash(obj: Any) -> str:
    return hashlib.sha256(canonical(obj)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row_digest(row: Mapping[str, Any]) -> str:
    return content_hash({k: v for k, v in row.items() if k != "row_sha256"})


def measurement_identity(*, run_id: str, kind: str, record_sha256: str, key: Sequence[Any]) -> str:
    prefix = "replspd-" if kind == KIND_SPEED else "replret-"
    return prefix + content_hash([run_id, kind, list(key), record_sha256])[:32]


def registered_rule() -> dict[str, Any]:
    """A fresh copy of the REPL-EMB-2.1 rule, for the harness to pass as ``rule``."""
    return dict(REGISTERED_RULE)


def record_path_for(run_dir: str | Path, run_id: str, *, speed: bool = False) -> Path:
    return Path(run_dir) / (
        f"{run_id}.speed{RECORD_SUFFIX}" if speed else f"{run_id}{RECORD_SUFFIX}"
    )


def sidecar_path(record_path: str | Path) -> Path:
    """``<base>.record.json`` -> ``<base>.belief_measurements.jsonl`` (any other name: by stem)."""
    path = Path(record_path)
    if path.name.endswith(RECORD_SUFFIX):
        return path.with_name(path.name[: -len(RECORD_SUFFIX)] + SIDECAR_SUFFIX)
    return path.with_name(path.stem + SIDECAR_SUFFIX)


# ------------------------------------------------------------------ primitives


def _is_num(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _hex64(value: Any) -> bool:
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def _valid_k(value: Any) -> bool:
    return _is_int(value) and value in RECALL_KS


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _utc(text: Any) -> datetime | None:
    if not isinstance(text, str) or not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def _day(text: Any) -> date | None:
    if not isinstance(text, str):
        return None
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def _same(have: Any, want: Any) -> bool:
    if _is_num(want) and not isinstance(want, bool):
        return _is_num(have) and math.isclose(have, want, rel_tol=0.0, abs_tol=1e-12)
    return have == want and type(have) is type(want)


def _jsonable(obj: Any, what: str) -> Any:
    try:
        return json.loads(json.dumps(obj, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise CaptureError(f"{what} is not finite JSON: {exc}") from exc


def _parse_cpus(text: str) -> set[int]:
    out: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            if int(lo) > int(hi):
                raise ValueError(f"descending cpu range {part!r}")
            out.update(range(int(lo), int(hi) + 1))
        elif part:
            out.add(int(part))
    if not out:
        raise ValueError("empty cpu list")
    return out


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


# ------------------------------------------------------------------ rule


def check_rule(rule: Any) -> list[str]:
    """Every reason ``rule`` is not the REPL-EMB-2.1 registration (or a dated amendment of it)."""
    if not isinstance(rule, Mapping):
        return ["rule must be an object"]
    problems: list[str] = []
    unknown = sorted(set(rule) - set(RULE_KEYS) - {"amendment"})
    missing = [key for key in RULE_KEYS if key not in rule]
    if unknown:
        problems.append(f"rule carries unknown keys {unknown}")
    if missing:
        return problems + [f"rule lacks {missing}"]
    differs = [key for key in RULE_KEYS if not _same(rule[key], REGISTERED_RULE[key])]
    amendment = rule.get("amendment")
    if amendment is None:
        if "amendment" in rule:
            problems.append("amendment is null: name a dated approval or omit the key")
        if differs:
            problems.append(
                f"rule differs from the REPL-EMB-2.1 registration on {differs} and names no dated "
                "amendment"
            )
        return problems
    if not (
        isinstance(amendment, Mapping)
        and set(amendment) == {"approved_on", "approval_ref"}
        and _day(amendment.get("approved_on")) is not None
        and _text(amendment.get("approval_ref"))
    ):
        problems.append("amendment must be {approved_on: YYYY-MM-DD, approval_ref: text}")
    if not differs:
        problems.append("amendment present but every value is the registered one")
    return problems


def rule_is_amended(rule: Mapping[str, Any]) -> bool:
    return rule.get("amendment") is not None


# ------------------------------------------------------------------ arm identity


def check_arm_identity(identity: Any, *, mode: str | None) -> list[str]:
    """Problems with an arm identity. ``mode`` lexical -> dense-only fields must be null."""
    if not isinstance(identity, Mapping):
        return ["arm identity must be an object"]
    if set(identity) != set(ARM_FIELDS):
        return [
            f"arm identity fields differ from {ARM_FIELDS}: {sorted(set(identity) ^ set(ARM_FIELDS))}"
        ]
    problems: list[str] = []
    for name in ("arm_id", "runtime"):
        if not _text(identity[name]):
            problems.append(f"{name} must be non-empty text")
    if not _hex64(identity["index_identity_digest"]):
        problems.append("index_identity_digest must be a sha256")
    if identity["role"] not in ROLES:
        problems.append(f"role must be one of {ROLES} (it is the ClaimTuple category)")
    if mode not in MODES:
        return problems + [f"mode must be one of {MODES}"]
    if mode == "lexical":
        carried = [f for f in DENSE_ONLY_FIELDS if identity[f] is not None]
        if carried:
            problems.append(
                f"a lexical arm must carry {carried} as null: no embedding model made it"
            )
        return problems
    for name in ("model", "revision", "dtype_or_quant"):
        if not _text(identity[name]):
            problems.append(f"{name} must be non-empty text for a {mode} arm")
    if not (_is_int(identity["dim"]) and identity["dim"] > 0):
        problems.append("dim must be a positive int for a dense or hybrid arm")
    if not _hex64(identity["prompt_hash"]):
        problems.append("prompt_hash must be a sha256 for a dense or hybrid arm")
    return problems


def arm_label(identity: Mapping[str, Any]) -> str:
    if identity.get("model") is None:
        return f"{identity['arm_id']} [lexical]"
    return (
        f"{identity['arm_id']} [{identity['model']}@{str(identity['revision'])[:12]} "
        f"{identity['dtype_or_quant']} {identity['dim']}d {identity['runtime'].split('/')[0]}]"
    )


# ------------------------------------------------------------------ host context


def check_host_context(ctx: Any, *, timing: bool) -> list[str]:
    """Problems with a host context. ``timing`` demands the full timing-leg context."""
    if not isinstance(ctx, Mapping):
        return ["host_context must be an object"]
    problems: list[str] = []
    if not _text(ctx.get("hostname")):
        problems.append("host_context.hostname must be non-empty text")
    declared: set[int] = set()
    try:
        declared = _parse_cpus(ctx["cpu_list"]) if isinstance(ctx.get("cpu_list"), str) else set()
        if not declared:
            problems.append("host_context.cpu_list must be a cpu list such as 24-71")
    except ValueError as exc:
        problems.append(f"host_context.cpu_list malformed: {exc}")
    present = [key for key in HOST_TIMING_KEYS if key in ctx]
    if not (timing or present):
        return problems
    for key in HOST_TIMING_KEYS:
        if key not in ctx:
            problems.append(f"timing context lacks {key}")
    if "region_lock_claim" in ctx and not _text(ctx["region_lock_claim"]):
        problems.append("region_lock_claim must be the held claim's tag")
    if "cpus_allowed_observed" in ctx:
        try:
            observed = _parse_cpus(ctx["cpus_allowed_observed"])
            if declared and not observed <= declared:
                problems.append("cpus_allowed_observed is not inside the declared cpu_list")
        except (ValueError, AttributeError):
            problems.append("cpus_allowed_observed malformed")
    if "loadavg" in ctx:
        load = ctx["loadavg"]
        ok = (
            isinstance(load, Mapping)
            and set(load) == {"start", "end"}
            and all(
                isinstance(load[s], list)
                and len(load[s]) == 3
                and all(_is_num(x) and x >= 0 for x in load[s])
                for s in ("start", "end")
            )
        )
        if not ok:
            problems.append("loadavg must be {start: [1m,5m,15m], end: [1m,5m,15m]}")
    if "frequency" in ctx:
        freq = ctx["frequency"]
        ok = (
            isinstance(freq, Mapping)
            and _text(freq.get("governor"))
            and _is_num(freq.get("mhz_min"))
            and _is_num(freq.get("mhz_max"))
            and 0 < freq["mhz_min"] <= freq["mhz_max"]
            and isinstance(freq.get("throttled"), bool)
        )
        if not ok:
            problems.append(
                "frequency must carry governor, 0 < mhz_min <= mhz_max and throttled (bool)"
            )
    return problems


def _read(path: str | Path) -> str:
    return Path(path).read_text()


def sample_host(cpu_list: str) -> dict[str, Any]:
    """One read-only sample of loadavg, observed affinity and the frequency state of ``cpu_list``.

    ``throttled`` is True when a frequency cap below the hardware maximum is in force on a claimed
    core; transient thermal events are not visible here, which is why mhz_min/mhz_max are recorded.
    Raises CaptureError rather than guess when the cpufreq tree is unavailable.
    """
    cpus = sorted(_parse_cpus(cpu_list))
    try:
        load = [float(x) for x in _read("/proc/loadavg").split()[:3]]
        allowed = next(
            line.split(":", 1)[1].strip()
            for line in _read("/proc/self/status").splitlines()
            if line.startswith("Cpus_allowed_list:")
        )
    except (OSError, StopIteration, ValueError) as exc:
        raise CaptureError(f"host sample unavailable: {exc}") from exc
    mhz: list[float] = []
    governors: set[str] = set()
    capped = False
    for cpu in cpus:
        base = Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq")
        try:
            mhz.append(int(_read(base / "scaling_cur_freq")) / 1000.0)
            governors.add(_read(base / "scaling_governor").strip())
            capped = capped or int(_read(base / "scaling_max_freq")) < int(
                _read(base / "cpuinfo_max_freq")
            )
        except (OSError, ValueError) as exc:
            raise CaptureError(f"cpufreq state of cpu{cpu} unavailable: {exc}") from exc
    return {
        "loadavg": load,
        "cpus_allowed_observed": allowed,
        "mhz": mhz,
        "governor": "+".join(sorted(governors)),
        "capped": capped,
    }


class HostWindow:
    """Sample the host before the first and after the last timed leg; emit a ``host_context``.

    ``region_lock_claim`` is the tag of the claim the caller holds (it is recorded, not verified:
    nothing here shells out). ``sample_fn`` is injectable for tests only.
    """

    def __init__(
        self,
        cpu_list: str,
        region_lock_claim: str,
        *,
        sample_fn: Callable[[str], dict] | None = None,
    ) -> None:
        self.cpu_list = cpu_list
        self.region_lock_claim = region_lock_claim
        self._sample = sample_fn or sample_host
        self.start: dict | None = None
        self.end: dict | None = None

    def begin(self) -> None:
        self.start = self._sample(self.cpu_list)

    def finish(self) -> None:
        self.end = self._sample(self.cpu_list)

    def context(self) -> dict[str, Any]:
        if self.start is None or self.end is None:
            raise CaptureError("host window was not sampled at both ends")
        mhz = list(self.start["mhz"]) + list(self.end["mhz"])
        ctx = {
            "hostname": socket.gethostname(),
            "cpu_list": self.cpu_list,
            "cpus_allowed_observed": self.start["cpus_allowed_observed"],
            "region_lock_claim": self.region_lock_claim,
            "loadavg": {"start": self.start["loadavg"], "end": self.end["loadavg"]},
            "frequency": {
                "governor": self.start["governor"],
                "mhz_min": min(mhz),
                "mhz_max": max(mhz),
                "throttled": bool(self.start["capped"] or self.end["capped"]),
            },
        }
        problems = check_host_context(ctx, timing=True)
        if problems:
            raise CaptureError("host context incomplete: " + "; ".join(problems))
        return ctx


# ------------------------------------------------------------------ metric tables


def recall_metric(k: int) -> str:
    return f"recall_at_{k}"


def diff_metric(k: int) -> str:
    return f"recall_at_{k}_paired_diff"


def _metric_table(kind: str, metric: str) -> tuple[str, str] | None:
    """(unit, direction) the metric must carry, or None when the metric is unknown for the kind."""
    if kind == "recall" and re.fullmatch(r"recall_at_(1|3|5|10)", metric):
        return RECALL_UNIT, "higher_better"
    if kind == "paired_diff" and re.fullmatch(r"recall_at_(1|3|5|10)_paired_diff", metric):
        return DIFF_UNIT, "higher_better"
    if kind == "cost":
        return COST_METRICS.get(metric)
    if kind == KIND_SPEED:
        return SPEED_METRICS.get(metric)
    return None


def _kind_source(kind: str) -> str:
    return SPEED_SOURCE_KIND if kind == KIND_SPEED else RETR_SOURCE_KIND


def config_leg(config_echo: Mapping[str, Any]) -> str:
    """A short digest of the config a speed or recall number was measured under (a locator part)."""
    return content_hash(config_echo)[:12]


def locator(row_key: Sequence[Any], run_id: str, kind: str) -> str:
    head = "replspd" if kind == KIND_SPEED else "replret"
    return ":".join([head, run_id, *[str(x) for x in row_key]])


def _row_key(row: Mapping[str, Any]) -> list[Any]:
    """The measurand axes of a row: what makes it a different claim from its neighbours."""
    kind, extra, metric = row["kind"], row["extra"], row["metric"]
    if kind == "recall":
        return [row["arm_id"], row["mode"], row["query_category"], f"k{row['k']}", metric]
    if kind == "paired_diff":
        against = row["samples"].get("against_arm_id")
        return [
            row["arm_id"],
            row["mode"],
            "vs",
            against,
            row["samples"].get("against_mode"),
            row["query_category"],
            f"k{row['k']}",
            metric,
        ]
    if kind == "cost":
        return [row["arm_id"], row["mode"], "cost", metric]
    return [row["arm_id"], metric, config_leg(extra["config_echo"])]


def _derive(row: Mapping[str, Any]) -> float:
    samples = row["samples"]
    if row["kind"] == "recall":
        return samples["hits"] / samples["n"]
    return samples["diff"] if row["kind"] == "paired_diff" else samples["value"]


# ------------------------------------------------------------------ validation (shared with the reader)


def claim_tuple_problems(row: Mapping[str, Any]) -> list[str]:
    """MEASUREMENT.md:13 -- a claim is (metric, protocol, n, date, attestation) plus its labels."""
    missing = [f for f in CLAIM_TUPLE_FIELDS if f not in row]
    if missing:
        return [f"claim tuple incomplete: {missing}"]
    problems = []
    if not _text(row["metric"]):
        problems.append("claim tuple: metric empty")
    if not (_is_int(row["reps"]) and row["reps"] >= 1):
        problems.append("claim tuple: reps must be a positive int")
    if not _text(row["reps_basis"]):
        problems.append("claim tuple: reps_basis empty")
    if row["protocol_id"] != "":
        problems.append("no protocol is codified for this source; protocol_id must be empty")
    if not _hex64(row["record_sha256"]):
        problems.append("claim tuple: record_sha256 (the attestation) must be a sha256")
    if _day(row["date"]) is None:
        problems.append("claim tuple: date malformed")
    return problems


def _validate_noise(floor: Any, row: Mapping[str, Any]) -> list[str]:
    if not isinstance(floor, Mapping) or set(floor) != set(NOISE_FIELDS):
        return ["speed row lacks its A/A noise floor " + "/".join(NOISE_FIELDS)]
    problems = []
    if not (_is_num(floor["aa_spread"]) and floor["aa_spread"] >= 0):
        problems.append("noise floor aa_spread must be a finite number >= 0")
    if floor["unit"] != row["unit"]:
        problems.append("noise floor must be carried in the metric's own unit")
    if floor["replication_unit"] not in REPLICATION_UNITS:
        problems.append(f"noise floor replication_unit must be one of {REPLICATION_UNITS}")
    if not (_is_int(floor["n"]) and floor["n"] >= 2):
        problems.append("noise floor needs n >= 2 A/A repeats")
    if floor["scope"] not in ("leg", "arm_metric"):
        problems.append("noise floor scope must be leg or arm_metric")
    return problems


def _validate_digest_map(name: str, value: Any, *, required: bool) -> list[str]:
    if value is None:
        return [] if not required else [f"digests.{name} missing"]
    if not (
        isinstance(value, Mapping)
        and value
        and all(_text(k) and _hex64(v) for k, v in value.items())
    ):
        return [f"digests.{name} must map names to sha256 digests"]
    return []


def _validate_provenance(prov: Any, row: Mapping[str, Any]) -> list[str]:
    if not isinstance(prov, Mapping) or set(prov) != set(PROVENANCE_FIELDS):
        return ["provenance must carry exactly " + ", ".join(PROVENANCE_FIELDS)]
    problems: list[str] = []
    speed = row["kind"] == KIND_SPEED
    inst = prov["instrument"]
    if not (
        isinstance(inst, Mapping)
        and _hex64(inst.get("capture_sha256"))
        and isinstance(inst.get("harness"), Mapping)
        and _text(inst["harness"].get("producer"))
        and (
            inst["harness"].get("harness_sha256") is None
            or _hex64(inst["harness"].get("harness_sha256"))
        )
    ):
        problems.append(
            "instrument must name the capture sha256 and the harness (producer, harness_sha256)"
        )
    run = prov["run"]
    start = _utc(run.get("started_at")) if isinstance(run, Mapping) else None
    end = _utc(run.get("finished_at")) if isinstance(run, Mapping) else None
    emitted = _utc(row["emitted_at"])
    if start is None or end is None or end < start:
        problems.append("run needs UTC started_at <= finished_at")
    else:
        if start < _utc(HOOK_SINCE):
            problems.append("run started before the hook existed (retrospective)")
        if emitted is None or emitted < end:
            problems.append("emitted_at precedes the run's finished_at")
    if speed:
        if prov["rule"] is not None:
            problems.append("a speed row carries no recall rule")
    else:
        problems += [f"rule: {p}" for p in check_rule(prov["rule"])]
    digests = prov["digests"]
    if not isinstance(digests, Mapping) or set(digests) != {
        "corpus",
        "query",
        "label",
        "arm_manifest_sha256",
        "frozen_at_launch_sha256",
    }:
        problems.append(
            "digests must carry corpus, query, label, arm_manifest_sha256, frozen_at_launch_sha256"
        )
    else:
        problems += _validate_digest_map("corpus", digests["corpus"], required=True)
        problems += _validate_digest_map("query", digests["query"], required=True)
        problems += _validate_digest_map("label", digests["label"], required=not speed)
        if speed and digests["label"] is not None:
            problems.append("a speed run has no gold labels; digests.label must be null")
        if not _hex64(digests["arm_manifest_sha256"]):
            problems.append("digests.arm_manifest_sha256 must be a sha256")
        if digests["frozen_at_launch_sha256"] is not None and not _hex64(
            digests["frozen_at_launch_sha256"]
        ):
            problems.append("digests.frozen_at_launch_sha256 must be a sha256 or null")
    timing = speed or row["kind"] == "cost"
    problems += [
        f"host_context: {p}" for p in check_host_context(prov["host_context"], timing=timing)
    ]
    return problems


def _validate_samples(row: Mapping[str, Any], prov: Mapping[str, Any] | None) -> list[str]:
    kind, samples = row["kind"], row["samples"]
    problems: list[str] = []
    if not isinstance(samples, Mapping):
        return ["samples must be an object"]

    def ci_ok(ci: Any, lo_bound: float, *, required: bool) -> bool:
        if ci is None:
            return not required
        return (
            isinstance(ci, list)
            and len(ci) == 2
            and all(_is_num(x) for x in ci)
            and lo_bound <= ci[0] <= ci[1] <= 1
        )

    if kind == "recall":
        if set(samples) != {"n", "hits", "ci95"}:
            return ["recall samples must be n, hits, ci95"]
        n, hits = samples["n"], samples["hits"]
        if not (_is_int(n) and n >= 1 and _is_int(hits) and 0 <= hits <= n):
            problems.append("recall needs int n >= 1 and 0 <= hits <= n")
        if not ci_ok(samples["ci95"], 0.0, required=False):
            problems.append("recall ci95 must be null or [lo, hi] with 0 <= lo <= hi <= 1")
        if not _valid_k(row["k"]):
            problems.append(f"k must be one of {RECALL_KS}")
        elif row["metric"] != recall_metric(row["k"]):
            problems.append("recall metric does not name its k")
        if row["mode"] not in MODES:
            problems.append(f"mode must be one of {MODES}")
        if (
            row["query_category"] == POOLED_CATEGORY
            and prov
            and isinstance(prov.get("rule"), Mapping)
        ):
            floor = prov["rule"].get("n_min")
            if _is_int(n) and _is_num(floor) and n < floor:
                problems.append(f"pooled category n={n} is below the rule's n_min={floor}")
        reps = n
    elif kind == "paired_diff":
        if set(samples) != {
            "against_arm_id",
            "against_mode",
            "diff",
            "ci95",
            "n",
            "resamples",
            "seed",
        }:
            return [
                "paired_diff samples must be against_arm_id, against_mode, diff, ci95, n, resamples, seed"
            ]
        n = samples["n"]
        if not (_is_int(n) and n >= 1):
            problems.append("paired diff needs int n >= 1")
        if not (_is_num(samples["diff"]) and -1 <= samples["diff"] <= 1):
            problems.append("paired diff must be finite in [-1, 1]")
        if not ci_ok(samples["ci95"], -1.0, required=True):
            problems.append("paired diff needs ci95 [lo, hi] with -1 <= lo <= hi <= 1")
        if not (
            _is_int(samples["resamples"]) and samples["resamples"] >= 1 and _is_int(samples["seed"])
        ):
            problems.append("paired diff needs int resamples >= 1 and int seed")
        if samples["against_mode"] not in MODES or row["mode"] not in MODES:
            problems.append(f"modes must be in {MODES}")
        if not _text(samples["against_arm_id"]):
            problems.append("against_arm_id empty")
        elif (samples["against_arm_id"], samples["against_mode"]) == (row["arm_id"], row["mode"]):
            problems.append("a paired diff of an arm against itself is vacuous")
        if not _valid_k(row["k"]) or row["metric"] != diff_metric(row["k"]):
            problems.append("paired diff metric does not name a valid k")
        if (
            row["query_category"] == POOLED_CATEGORY
            and prov
            and isinstance(prov.get("rule"), Mapping)
        ):
            floor = prov["rule"].get("n_min")
            if _is_int(n) and _is_num(floor) and n < floor:
                problems.append(f"pooled category n={n} is below the rule's n_min={floor}")
        reps = n
    elif kind == "cost":
        if set(samples) != {"value", "n"}:
            return ["cost samples must be value, n"]
        if not (_is_num(samples["value"]) and samples["value"] > 0):
            problems.append("cost value must be a finite number > 0")
        if not (_is_int(samples["n"]) and samples["n"] >= 1):
            problems.append("cost needs int n >= 1")
        if row["mode"] not in MODES:
            problems.append(f"mode must be one of {MODES}")
        reps = samples["n"]
    else:
        if set(samples) != {"value", "n", "spread", "spread_unit", "noise_floor"}:
            return ["speed samples must be value, n, spread, spread_unit, noise_floor"]
        if not (_is_num(samples["value"]) and samples["value"] > 0):
            problems.append("speed value must be a finite number > 0")
        if not (_is_int(samples["n"]) and samples["n"] >= 1):
            problems.append("speed needs int n >= 1")
        if not (
            _is_num(samples["spread"]) and samples["spread"] >= 0 and _text(samples["spread_unit"])
        ):
            problems.append("speed needs a finite spread >= 0 with its spread_unit")
        problems += _validate_noise(samples["noise_floor"], row)
        reps = samples["n"]
    if row["reps"] != reps:
        problems.append("reps does not equal the sample n")
    return problems


def validate_row(row: Any) -> list[str]:
    """Every reason ``row`` is not a well-formed capture row. Empty list = valid."""
    try:
        return _validate_row(row)
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        return [f"row is malformed beyond validation: {type(exc).__name__}: {exc}"]


def _validate_row(row: Any) -> list[str]:
    if not isinstance(row, Mapping):
        return ["row is not an object"]
    if "instrument" in row and row["instrument"] not in INSTRUMENTS:
        return [
            f"instrument {row['instrument']!r} is not one of {INSTRUMENTS}: online or foreign "
            "instruments never pool with the offline ones"
        ]
    if any(f not in row for f in CLAIM_TUPLE_FIELDS):
        return claim_tuple_problems(row)
    if not _is_num(row.get("value")):
        return ["value is not a finite number"]
    if set(row) != set(ROW_FIELDS):
        return [
            f"row fields differ from the capture contract: {sorted(set(row) ^ set(ROW_FIELDS))}"
        ]
    problems = claim_tuple_problems(row)
    if row["schema"] != CAPTURE_SCHEMA:
        problems.append("schema is foreign")
    if (
        row["source_kind"] not in SOURCE_INSTRUMENT
        or SOURCE_INSTRUMENT[row["source_kind"]] != row["instrument"]
    ):
        problems.append("source_kind and instrument do not pair")
    if row["row_sha256"] != row_digest(row):
        problems.append("row_sha256 does not re-derive")
    kind = row["kind"]
    if kind not in KINDS_RETR + (KIND_SPEED,) or _kind_source(kind) != row["source_kind"]:
        return problems + [f"kind {kind!r} does not belong to source {row['source_kind']!r}"]
    table = _metric_table(kind, row["metric"])
    if table is None:
        return problems + [f"unknown {kind} metric {row['metric']!r}"]
    if (row["unit"], row["metric_direction"]) != table:
        problems.append("unit/direction differ from the metric's declaration")
    extra = row["extra"]
    if not isinstance(extra, Mapping) or set(extra) != set(EXTRA_FIELDS):
        return problems + ["extra must carry exactly " + ", ".join(EXTRA_FIELDS)]
    prov = extra["provenance"]
    if extra.get("provenance_sha256") != content_hash(prov):
        problems.append("provenance_sha256 does not re-derive")
    problems += _validate_provenance(prov, row)
    if not isinstance(row["samples"], Mapping):
        return problems + ["samples must be an object"]
    sample_problems = _validate_samples(row, prov if isinstance(prov, Mapping) else None)
    if sample_problems:
        return problems + sample_problems
    try:
        value = _derive(row)
    except (KeyError, TypeError, ZeroDivisionError):
        return problems + ["value cannot be derived from samples"]
    if not _is_num(row["value"]) or not math.isclose(
        row["value"], value, rel_tol=1e-12, abs_tol=1e-15
    ):
        problems.append("value does not re-derive from samples")
    # axes by kind
    if kind in ("recall", "paired_diff"):
        if not _valid_k(row["k"]) or not _text(row["query_category"]):
            problems.append("recall rows need k and query_category")
    else:
        if row["k"] is not None or row["query_category"] is not None:
            problems.append("cost and speed rows carry no k or query_category")
    if kind == KIND_SPEED:
        if row["mode"] is not None:
            problems.append("a speed row has no retrieval mode")
        arm_mode = "dense"
    else:
        arm_mode = row["mode"]
    arm_problems = check_arm_identity(extra["arm"], mode=arm_mode)
    problems += [f"arm: {p}" for p in arm_problems]
    if not arm_problems:
        if extra["arm"]["arm_id"] != row["arm_id"]:
            problems.append("arm_id differs from extra.arm")
        if row["category"] != extra["arm"]["role"]:
            problems.append("category must equal the arm's role")
    if kind == "paired_diff":
        against_problems = check_arm_identity(
            extra["against"], mode=row["samples"].get("against_mode")
        )
        problems += [f"against: {p}" for p in against_problems]
        if not against_problems and extra["against"]["arm_id"] != row["samples"].get(
            "against_arm_id"
        ):
            problems.append("against_arm_id differs from extra.against")
    elif extra["against"] is not None:
        problems.append("only a paired diff names an arm it is against")
    echo = extra["config_echo"]
    if not isinstance(echo, Mapping) or (kind in ("recall", KIND_SPEED) and not echo):
        problems.append("config_echo must be an object (non-empty for recall and speed rows)")
    if extra["locator"] != locator(_row_key(row), row["run_id"], kind):
        problems.append("locator does not bind run, arm and the measurand axes")
    if row["measurement_id"] != measurement_identity(
        run_id=row["run_id"], kind=kind, record_sha256=row["record_sha256"], key=_row_key(row)
    ):
        problems.append("measurement_id does not re-derive")
    rf = row["record_file"]
    if not isinstance(rf, str) or not rf or "/" in rf or rf.startswith("."):
        problems.append("record_file must be a bare file name beside the sidecar")
    if not _text(row["claim"]):
        problems.append("claim text is empty")
    emitted = _utc(row["emitted_at"])
    if emitted is None or row["date"] != str(row["emitted_at"])[:10]:
        problems.append("emitted_at/date malformed")
    if not _text(row["producer"]) or not _text(row["run_id"]):
        problems.append("producer and run_id are required")
    return problems


def validate_rows(rows: Sequence[Any]) -> list[str]:
    """Per-row validity plus the sidecar-level contract. Empty list = a valid, single-run sidecar."""
    if not rows:
        return ["sidecar has no rows"]
    problems: list[str] = []
    for index, row in enumerate(rows):
        problems += [f"row {index}: {p}" for p in validate_row(row)]
    if problems:
        return problems
    if len({r["instrument"] for r in rows}) != 1 or len({r["source_kind"] for r in rows}) != 1:
        return ["sidecar mixes instruments or sources: offline_recall and offline_speed never pool"]
    keys = {
        (r["run_id"], r["record_file"], r["record_sha256"], r["extra"]["provenance_sha256"])
        for r in rows
    }
    if len(keys) != 1:
        problems.append("sidecar mixes runs, records or provenance")
    idents: set[str] = set()
    for row in rows:
        for ident in (row["measurement_id"], row["extra"]["locator"]):
            if ident in idents:
                problems.append(f"duplicate measurement {ident}")
            idents.add(ident)
    arms: dict[str, str] = {}
    for row in rows:
        for identity in (row["extra"]["arm"], row["extra"]["against"]):
            if identity is None:
                continue
            digest = content_hash(identity)
            if arms.setdefault(identity["arm_id"], digest) != digest:
                problems.append(f"arm_id {identity['arm_id']} carries two identities in one run")
    if rows[0]["source_kind"] == SPEED_SOURCE_KIND:
        return problems + _speed_set_problems(rows)
    return problems + _retrieval_set_problems(rows)


def _retrieval_set_problems(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    problems: list[str] = []
    recall = {
        (r["arm_id"], r["mode"], r["query_category"], r["k"]): r
        for r in rows
        if r["kind"] == "recall"
    }
    if not recall:
        problems.append("a retrieval sidecar needs at least one recall row")
    ns: dict[str, set[int]] = {}
    for row in recall.values():
        ns.setdefault(row["query_category"], set()).add(row["samples"]["n"])
    for category, counts in sorted(ns.items()):
        if len(counts) != 1:
            problems.append(
                f"query category {category!r} scored on different n across arms {sorted(counts)}: "
                "a failure is a miss, never a dropped query"
            )
    for row in rows:
        if row["kind"] != "paired_diff":
            continue
        s = row["samples"]
        a = recall.get((row["arm_id"], row["mode"], row["query_category"], row["k"]))
        b = recall.get((s["against_arm_id"], s["against_mode"], row["query_category"], row["k"]))
        if a is None or b is None:
            problems.append(
                f"paired diff {row['extra']['locator']} has no recall row for both sides"
            )
        elif not (a["samples"]["n"] == b["samples"]["n"] == s["n"]):
            problems.append(f"paired diff {row['extra']['locator']} n differs from its recall rows")
        elif not math.isclose(row["value"], a["value"] - b["value"], rel_tol=0.0, abs_tol=1e-9):
            problems.append(f"paired diff {row['extra']['locator']} is not recall(A) - recall(B)")
    return problems


def _speed_set_problems(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    problems: list[str] = []
    by_leg: dict[tuple[str, str], dict[str, float]] = {}
    for row in rows:
        leg = config_leg(row["extra"]["config_echo"])
        by_leg.setdefault((row["arm_id"], leg), {})[row["metric"]] = row["value"]
    for (arm, leg), got in by_leg.items():
        order = [got[m] for m in ("query_p50_ms", "query_p95_ms", "query_max_ms") if m in got]
        if order != sorted(order):
            problems.append(
                f"{arm} leg {leg}: latency percentiles are not ordered p50 <= p95 <= max"
            )
    return problems


# ------------------------------------------------------------------ claim text


def _rule_text(rule: Mapping[str, Any]) -> str:
    text = (
        f"{rule['registration']} (primary k={rule['k']}, M={rule['M']}, X={rule['X']}, "
        f"n>={rule['n_min']}, paired-bootstrap {rule['ci_level']:.0%} lower bound > 0)"
    )
    amendment = rule.get("amendment")
    if amendment:
        text += f" AMENDED, approved {amendment['approved_on']} ({amendment['approval_ref']})"
    return text


def _claim(kind: str, metric: str, row: Mapping[str, Any], prov: Mapping[str, Any]) -> str:
    arm = arm_label(row["_arm"])
    value, samples = row["_value"], row["_samples"]
    if kind == "recall":
        ci = samples["ci95"]
        ci_text = f", 95% CI [{ci[0]:.4f}, {ci[1]:.4f}]" if ci else ", no CI recorded"
        return (
            f"UFH-12 offline retrieval eval: {metric} {value:.4f} ({samples['hits']}/{samples['n']} "
            f"queries; every failure, invalid or timeout counted as a miss) for {arm}, "
            f"{row['_mode']} pipeline, query category '{row['_category']}'{ci_text}. Pre-registered "
            f"rule {_rule_text(prov['rule'])}. Instrument offline_recall; the online "
            "spill_pointer_follows shadow is a separate instrument and is never pooled."
        )
    if kind == "paired_diff":
        lo, hi = samples["ci95"]
        return (
            f"UFH-12 offline retrieval eval: paired (by query) {metric} {value:+.4f} for {arm} "
            f"[{row['_mode']}] minus {arm_label(row['_against'])} [{samples['against_mode']}] on "
            f"query category '{row['_category']}', n={samples['n']}, 95% paired-bootstrap CI "
            f"[{lo:+.4f}, {hi:+.4f}] ({samples['resamples']} resamples, seed {samples['seed']}). "
            f"Pre-registered rule {_rule_text(prov['rule'])}. Instrument offline_recall."
        )
    host = prov["host_context"]
    where = (
        f"cpu-list {host['cpu_list']} under region-lock claim {host['region_lock_claim']} "
        f"(loadavg {host['loadavg']['start'][0]:.1f} -> {host['loadavg']['end'][0]:.1f})"
    )
    if kind == "cost":
        return (
            f"UFH-12 offline retrieval eval cost: {metric} {value:g} {row['_unit']} over "
            f"{samples['n']} {'queries' if metric != 'index_build_s' else 'index build'} for {arm}, "
            f"{row['_mode']} pipeline, measured on {where}. Instrument offline_recall."
        )
    floor = samples["noise_floor"]
    return (
        f"UFH-12 embedder speed block: {metric} {value:g} {row['_unit']} (n={samples['n']}, spread "
        f"{samples['spread']:g} {samples['spread_unit']}) for {arm}; A/A noise floor "
        f"{floor['aa_spread']:g} {floor['unit']} ({floor['replication_unit']}-scoped, "
        f"{floor['scope']}, n={floor['n']}); {where}. Instrument offline_speed."
    )


# ------------------------------------------------------------------ the writer


def build_rows(
    record: Mapping[str, Any], *, record_file: str, record_sha256: str, emitted_at: str
) -> list[dict]:
    """Project one record into capture rows (deterministic given its inputs). Raises CaptureError."""
    try:
        return _build_rows(
            record, record_file=record_file, record_sha256=record_sha256, emitted_at=emitted_at
        )
    except (KeyError, TypeError, ValueError, ZeroDivisionError, AttributeError) as exc:
        raise CaptureError(f"record is malformed: {type(exc).__name__}: {exc}") from exc


def _build_rows(
    record: Mapping[str, Any], *, record_file: str, record_sha256: str, emitted_at: str
) -> list[dict]:
    speed = record.get("schema") == SPEED_RECORD_SCHEMA
    if record.get("schema") not in (RETR_RECORD_SCHEMA, SPEED_RECORD_SCHEMA):
        raise CaptureError(f"unknown record schema {record.get('schema')!r}")
    instrument = record.get("instrument")
    if instrument not in INSTRUMENTS or instrument != (
        INSTRUMENT_SPEED if speed else INSTRUMENT_RECALL
    ):
        raise CaptureError(
            f"instrument {instrument!r} does not match the record schema; only "
            f"{INSTRUMENTS} are written, never pooled"
        )
    source_kind = SPEED_SOURCE_KIND if speed else RETR_SOURCE_KIND
    when = emitted_at
    if _utc(when) is None:
        raise CaptureError("emitted_at must be a UTC timestamp")
    provenance = _jsonable(
        {
            "instrument": {
                "capture_sha256": record["capture_sha256"],
                "harness": {
                    "producer": record["producer"],
                    "harness_sha256": record["harness_sha256"],
                },
            },
            "run": {"started_at": record["started_at"], "finished_at": record["finished_at"]},
            "rule": None if speed else record["rule"],
            "digests": record["digests"],
            "host_context": record["host_context"],
        },
        "provenance",
    )
    prov_sha = content_hash(provenance)
    arms = record["arms"]
    rows: list[dict] = []
    for result in record["results"]:
        kind = result["kind"]
        arm = arms[result["arm_id"]]
        against = arms[result["against_arm_id"]] if kind == "paired_diff" else None
        mode = result.get("mode")
        category = result.get("query_category")
        k = result.get("k")
        if kind == "recall":
            metric = recall_metric(k)
            samples = {"n": result["n"], "hits": result["hits"], "ci95": result["ci95"]}
            value, reps, basis = result["hits"] / result["n"], result["n"], "queries scored"
        elif kind == "paired_diff":
            metric = diff_metric(k)
            samples = {
                "against_arm_id": result["against_arm_id"],
                "against_mode": result["against_mode"],
                "diff": result["diff"],
                "ci95": result["ci95"],
                "n": result["n"],
                "resamples": result["resamples"],
                "seed": result["seed"],
            }
            value, reps, basis = result["diff"], result["n"], "queries paired"
        elif kind == "cost":
            metric = result["metric"]
            samples = {"value": result["value"], "n": result["n"]}
            value, reps = result["value"], result["n"]
            basis = "index builds timed" if metric == "index_build_s" else "queries timed"
        else:
            metric = result["metric"]
            samples = {
                "value": result["value"],
                "n": result["n"],
                "spread": result["spread"],
                "spread_unit": result["spread_unit"],
                "noise_floor": result["noise_floor"],
            }
            value, reps, basis = result["value"], result["n"], "timed repeats"
        table = _metric_table(kind, metric)
        if table is None:
            raise CaptureError(f"unknown {kind} metric {metric!r}")
        unit, direction = table
        samples = _jsonable(samples, "samples")
        config_echo = _jsonable(result.get("config_echo", {}), "config_echo")
        row: dict[str, Any] = {
            "schema": CAPTURE_SCHEMA,
            "source_kind": source_kind,
            "instrument": instrument,
            "run_id": record["run_id"],
            "producer": record["producer"],
            "emitted_at": when,
            "date": when[:10],
            "measurement_id": "",
            "kind": kind,
            "metric": metric,
            "arm_id": result["arm_id"],
            "mode": mode,
            "query_category": category,
            "k": k,
            "value": value,
            "unit": unit,
            "metric_direction": direction,
            "category": arm["role"],
            "claim": "",
            "protocol_id": "",
            "reps": reps,
            "reps_basis": basis,
            "samples": samples,
            "record_file": record_file,
            "record_sha256": record_sha256,
            "extra": {
                "locator": "",
                "arm": arm,
                "against": against,
                "config_echo": config_echo,
                "provenance": provenance,
                "provenance_sha256": prov_sha,
            },
            "row_sha256": "",
        }
        key = _row_key(row)
        row["measurement_id"] = measurement_identity(
            run_id=record["run_id"], kind=kind, record_sha256=record_sha256, key=key
        )
        row["extra"]["locator"] = locator(key, record["run_id"], kind)
        row["claim"] = _claim(
            kind,
            metric,
            {
                "_arm": arm,
                "_against": against,
                "_value": value,
                "_samples": samples,
                "_mode": mode,
                "_category": category,
                "_unit": unit,
            },
            provenance,
        )
        row["row_sha256"] = row_digest(row)
        rows.append(row)
    problems = validate_rows(rows)
    if problems:
        raise CaptureError("refusing an invalid record: " + "; ".join(problems))
    return rows


class _Run:
    """What a retrieval run and a speed run share: digests, host context, arms, immutability."""

    SPEED = False

    def __init__(
        self,
        run_dir: str | Path,
        run_id: str,
        *,
        rule: Mapping[str, Any] | None,
        corpus_digests: Mapping[str, str],
        query_digests: Mapping[str, str],
        label_digests: Mapping[str, str] | None,
        arm_manifest_sha256: str,
        host_context: Mapping[str, Any],
        timing: bool,
        frozen_at_launch_path: str | Path | None,
        harness_path: str | Path | None,
        producer: str,
        clock: Callable[[], str] | None,
    ) -> None:
        if not (isinstance(run_id, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._@+-]*", run_id)):
            raise CaptureError(f"run_id {run_id!r} must be a bare file-name stem")
        self.run_dir = Path(run_dir)
        self.run_id = run_id
        self.producer = producer
        if not _text(producer):
            raise CaptureError("producer must be non-empty text")
        self._clock = clock or utc_now
        self.started_at = self._clock()
        started = _utc(self.started_at)
        if started is None or started < _utc(HOOK_SINCE):
            raise CaptureError(
                "a run must start at or after the hook date; retrospective data is not "
                "written through this API"
            )
        problems = check_host_context(host_context, timing=timing)
        if problems:
            raise CaptureError("host_context refused: " + "; ".join(problems))
        self.host_context = _jsonable(dict(host_context), "host_context")
        if not _hex64(arm_manifest_sha256):
            raise CaptureError("arm_manifest_sha256 must be a sha256")
        self.arm_manifest_sha256 = arm_manifest_sha256
        self.corpus_digests = self._digest_map("corpus_digests", corpus_digests)
        self.query_digests = self._digest_map("query_digests", query_digests)
        self.label_digests = (
            None if self.SPEED else self._digest_map("label_digests", label_digests)
        )
        self.rule = None if rule is None else self._check_rule(rule)
        self.frozen_at_launch_sha256 = self._frozen_at_launch(frozen_at_launch_path)
        self.harness_sha256 = file_sha256(harness_path) if harness_path is not None else None
        self._arms: dict[str, tuple[dict, str]] = {}
        self._results: list[dict] = []
        self._seen: set[tuple] = set()
        self._finalized = False
        self.record_path = record_path_for(self.run_dir, run_id, speed=self.SPEED)
        self.sidecar_path = sidecar_path(self.record_path)

    # -- construction helpers
    @staticmethod
    def _digest_map(name: str, value: Any) -> dict[str, str]:
        if not (
            isinstance(value, Mapping)
            and value
            and all(_text(k) and _hex64(v) for k, v in value.items())
        ):
            raise CaptureError(f"{name} must be a non-empty mapping of names to sha256 digests")
        return dict(value)

    @staticmethod
    def _check_rule(rule: Mapping[str, Any]) -> dict[str, Any]:
        problems = check_rule(rule)
        if problems:
            raise CaptureError("pre-registered rule refused: " + "; ".join(problems))
        return _jsonable(dict(rule), "rule")

    def _frozen_at_launch(self, path: str | Path | None) -> str | None:
        if path is None:
            return None
        target = Path(path)
        if not target.is_file():
            raise CaptureError(f"FROZEN-AT-LAUNCH file {target} does not exist")
        tokens = set(re.findall(r"\b[0-9a-f]{64}\b", target.read_text()))
        wanted = {"arm_manifest_sha256": self.arm_manifest_sha256}
        for group, digests in (
            ("corpus", self.corpus_digests),
            ("query", self.query_digests),
            ("label", self.label_digests or {}),
        ):
            wanted.update({f"{group}:{name}": digest for name, digest in digests.items()})
        absent = sorted(name for name, digest in wanted.items() if digest not in tokens)
        if absent:
            raise CaptureError(f"digests absent from {target.name}: {absent}")
        return file_sha256(target)

    def _register_arm(self, identity: Any, mode: str | None) -> dict:
        problems = check_arm_identity(identity, mode=mode)
        if problems:
            raise CaptureError("arm identity refused: " + "; ".join(problems))
        copy = _jsonable(dict(identity), "arm identity")
        digest = content_hash(copy)
        known = self._arms.get(copy["arm_id"])
        if known is not None and known[1] != digest:
            raise CaptureError(
                f"arm_id {copy['arm_id']} was already registered with a different identity"
            )
        self._arms[copy["arm_id"]] = (copy, digest)
        return copy

    def _once(self, key: tuple) -> None:
        if self._finalized:
            raise CaptureError("run already finalized")
        if key in self._seen:
            raise CaptureError(
                f"duplicate measurement {key}: aggregate repeats into n/spread, "
                "never record the same measurand twice"
            )
        self._seen.add(key)

    @staticmethod
    def _count(value: Any, name: str, *, floor: int = 1) -> int:
        if not (_is_int(value) and value >= floor):
            raise CaptureError(f"{name} must be an int >= {floor} (got {value!r})")
        return value

    @staticmethod
    def _positive(value: Any, name: str, *, allow_zero: bool = False) -> float:
        if not _is_num(value) or value < 0 or (value == 0 and not allow_zero):
            raise CaptureError(
                f"{name} must be a finite number {'>= 0' if allow_zero else '> 0'} (got {value!r})"
            )
        return float(value)

    @staticmethod
    def _config(config_echo: Any, name: str = "config_echo") -> dict:
        if not (isinstance(config_echo, Mapping) and config_echo):
            raise CaptureError(
                f"{name} must be a non-empty mapping (what config produced the number)"
            )
        return _jsonable(dict(config_echo), name)

    # -- finalize
    def _record(self, finished_at: str) -> dict:  # pragma: no cover - overridden
        raise NotImplementedError

    def _preflight(self) -> None:
        raise NotImplementedError

    def finalize(self) -> Path:
        """Write the record and its belief sidecar (all or nothing). Returns the sidecar path."""
        if self._finalized:
            raise CaptureError("run already finalized")
        self._preflight()
        finished_at = self._clock()
        record = self._record(finished_at)
        data = (json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
        record_sha = hashlib.sha256(data).hexdigest()
        rows = build_rows(
            record,
            record_file=self.record_path.name,
            record_sha256=record_sha,
            emitted_at=finished_at,
        )
        self.run_dir.mkdir(parents=True, exist_ok=True)
        for target in (self.record_path, self.sidecar_path):
            if target.exists():
                raise CaptureError(f"{target} already exists: records are immutable")
        tmp_record = self.record_path.with_name(self.record_path.name + ".tmp")
        tmp_sidecar = self.sidecar_path.with_name(self.sidecar_path.name + ".tmp")
        tmp_record.write_bytes(data)
        tmp_sidecar.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
        os.replace(tmp_record, self.record_path)
        os.replace(tmp_sidecar, self.sidecar_path)
        self._finalized = True
        return self.sidecar_path

    def _base_record(self, schema: str, instrument: str, finished_at: str) -> dict:
        return {
            "schema": schema,
            "instrument": instrument,
            "run_id": self.run_id,
            "producer": self.producer,
            "started_at": self.started_at,
            "finished_at": finished_at,
            "capture_sha256": file_sha256(Path(__file__).resolve()),
            "harness_sha256": self.harness_sha256,
            "rule": self.rule,
            "digests": {
                "corpus": self.corpus_digests,
                "query": self.query_digests,
                "label": self.label_digests,
                "arm_manifest_sha256": self.arm_manifest_sha256,
                "frozen_at_launch_sha256": self.frozen_at_launch_sha256,
            },
            "host_context": self.host_context,
            "arms": {arm_id: identity for arm_id, (identity, _) in sorted(self._arms.items())},
        }


class RetrievalRun(_Run):
    """The offline retrieval eval (REPL-EMB-2.2). Instrument ``offline_recall``.

    ``rule`` is :func:`registered_rule` (or the same plus a dated ``amendment``);
    ``corpus_digests`` / ``label_digests`` map names to sha256; ``query_digests`` maps each query
    category (stratum) to the sha256 of its frozen query file, ``all`` being the pooled decision
    set; ``host_context`` is :meth:`HostWindow.context` output (``hostname`` + ``cpu_list`` are
    enough for a run with no cost rows; ``add_cost`` needs the full timing context).
    """

    def __init__(
        self,
        run_dir: str | Path,
        run_id: str,
        rule: Mapping[str, Any],
        corpus_digests: Mapping[str, str],
        query_digests: Mapping[str, str],
        label_digests: Mapping[str, str],
        arm_manifest_sha256: str,
        host_context: Mapping[str, Any],
        *,
        frozen_at_launch_path: str | Path | None = None,
        harness_path: str | Path | None = None,
        producer: str = "repl_retrieval_eval",
        clock: Callable[[], str] | None = None,
    ) -> None:
        super().__init__(
            run_dir,
            run_id,
            rule=rule,
            corpus_digests=corpus_digests,
            query_digests=query_digests,
            label_digests=label_digests,
            arm_manifest_sha256=arm_manifest_sha256,
            host_context=host_context,
            timing=False,
            frozen_at_launch_path=frozen_at_launch_path,
            harness_path=harness_path,
            producer=producer,
            clock=clock,
        )

    def _category(self, category: Any) -> str:
        if category not in self.query_digests:
            raise CaptureError(
                f"query category {category!r} is not a key of query_digests "
                f"{sorted(self.query_digests)}"
            )
        return category

    def _pooled_n(self, category: str, n: int) -> None:
        if category == POOLED_CATEGORY and n < self.rule["n_min"]:
            raise CaptureError(
                f"pooled category '{POOLED_CATEGORY}' with n={n} is below the "
                f"pre-registered n_min={self.rule['n_min']}"
            )

    def add_recall(
        self,
        arm_identity: Mapping[str, Any],
        category: str,
        mode: str,
        k: int,
        metric: str,
        value: float,
        n: int,
        hits: int,
        ci95: Sequence[float] | None,
        config_echo: Mapping[str, Any],
    ) -> None:
        if metric != "recall":
            raise CaptureError(f"metric must be 'recall' (got {metric!r}); k is its own argument")
        if mode not in MODES:
            raise CaptureError(f"mode must be one of {MODES}")
        if k not in RECALL_KS or not _is_int(k):
            raise CaptureError(f"k must be one of {RECALL_KS}")
        category = self._category(category)
        n = self._count(n, "n")
        if not (_is_int(hits) and 0 <= hits <= n):
            raise CaptureError(f"hits must be an int in [0, n={n}] (got {hits!r})")
        if not _is_num(value) or not math.isclose(value, hits / n, rel_tol=1e-9, abs_tol=1e-12):
            raise CaptureError(f"value {value!r} does not re-derive from hits/n = {hits}/{n}")
        if ci95 is not None:
            if not (len(ci95) == 2 and all(_is_num(x) for x in ci95)):
                raise CaptureError("ci95 must be null or two finite numbers")
            if ci95[0] > ci95[1]:
                raise CaptureError(f"ci95 lower bound {ci95[0]} exceeds upper bound {ci95[1]}")
            if not 0 <= ci95[0] <= ci95[1] <= 1:
                raise CaptureError("a recall ci95 must lie inside [0, 1]")
            ci95 = [float(ci95[0]), float(ci95[1])]
        self._pooled_n(category, n)
        identity = self._register_arm(arm_identity, mode)
        self._once(("recall", identity["arm_id"], mode, category, k))
        self._results.append(
            {
                "kind": "recall",
                "arm_id": identity["arm_id"],
                "mode": mode,
                "query_category": category,
                "k": k,
                "n": n,
                "hits": hits,
                "ci95": ci95,
                "config_echo": self._config(config_echo),
            }
        )

    def add_paired_diff(
        self,
        a: Mapping[str, Any],
        b: Mapping[str, Any],
        category: str,
        k: int,
        diff: float,
        ci95: Sequence[float],
        n: int,
        resamples: int,
        seed: int,
        *,
        mode_a: str,
        mode_b: str,
    ) -> None:
        """``diff`` = recall(a, mode_a) - recall(b, mode_b), paired by query. ``mode_*`` are required
        keyword arguments because an arm's dense and hybrid pipelines are different claims."""
        if mode_a not in MODES or mode_b not in MODES:
            raise CaptureError(f"mode_a and mode_b must be in {MODES}")
        if k not in RECALL_KS or not _is_int(k):
            raise CaptureError(f"k must be one of {RECALL_KS}")
        category = self._category(category)
        n = self._count(n, "n")
        if not (_is_num(diff) and -1 <= diff <= 1):
            raise CaptureError(f"diff must be finite in [-1, 1] (got {diff!r})")
        if ci95 is None or not (len(ci95) == 2 and all(_is_num(x) for x in ci95)):
            raise CaptureError("a paired diff needs ci95 as two finite numbers")
        if ci95[0] > ci95[1]:
            raise CaptureError(f"ci95 lower bound {ci95[0]} exceeds upper bound {ci95[1]}")
        if not -1 <= ci95[0] <= ci95[1] <= 1:
            raise CaptureError("a paired-diff ci95 must lie inside [-1, 1]")
        resamples = self._count(resamples, "resamples")
        if not _is_int(seed):
            raise CaptureError("seed must be an int")
        self._pooled_n(category, n)
        ia, ib = self._register_arm(a, mode_a), self._register_arm(b, mode_b)
        if (ia["arm_id"], mode_a) == (ib["arm_id"], mode_b):
            raise CaptureError("a paired diff of an arm against itself is vacuous")
        self._once(("paired_diff", ia["arm_id"], mode_a, ib["arm_id"], mode_b, category, k))
        self._results.append(
            {
                "kind": "paired_diff",
                "arm_id": ia["arm_id"],
                "mode": mode_a,
                "against_arm_id": ib["arm_id"],
                "against_mode": mode_b,
                "query_category": category,
                "k": k,
                "diff": float(diff),
                "ci95": [float(ci95[0]), float(ci95[1])],
                "n": n,
                "resamples": resamples,
                "seed": seed,
                "config_echo": {},
            }
        )

    def add_cost(
        self,
        arm_identity: Mapping[str, Any],
        cpu_s_per_query: float,
        p50_ms: float,
        p95_ms: float,
        index_build_s: float,
        n_queries: int,
        *,
        mode: str,
    ) -> None:
        """Per-arm cost. ``mode`` is required keyword: a lexical, dense and hybrid pipeline of one
        arm have different costs. Needs the full timing host context (cpu-list, claim, load, freq)."""
        problems = check_host_context(self.host_context, timing=True)
        if problems:
            raise CaptureError(
                "cost rows are timing legs and need the full timing context: " + "; ".join(problems)
            )
        if mode not in MODES:
            raise CaptureError(f"mode must be one of {MODES}")
        n_queries = self._count(n_queries, "n_queries")
        values = {
            "cpu_s_per_query": self._positive(cpu_s_per_query, "cpu_s_per_query"),
            "query_p50_ms": self._positive(p50_ms, "p50_ms"),
            "query_p95_ms": self._positive(p95_ms, "p95_ms"),
            "index_build_s": self._positive(index_build_s, "index_build_s"),
        }
        if values["query_p95_ms"] < values["query_p50_ms"]:
            raise CaptureError("p95_ms is below p50_ms")
        identity = self._register_arm(arm_identity, mode)
        for metric, value in values.items():
            self._once(("cost", identity["arm_id"], mode, metric))
            self._results.append(
                {
                    "kind": "cost",
                    "arm_id": identity["arm_id"],
                    "mode": mode,
                    "metric": metric,
                    "value": value,
                    "n": 1 if metric == "index_build_s" else n_queries,
                    "config_echo": {},
                }
            )

    def _preflight(self) -> None:
        if not any(r["kind"] == "recall" for r in self._results):
            raise CaptureError("a retrieval record needs at least one recall row")

    def _record(self, finished_at: str) -> dict:
        record = self._base_record(RETR_RECORD_SCHEMA, INSTRUMENT_RECALL, finished_at)
        record["results"] = self._results
        return record


class SpeedRun(_Run):
    """The embedder speed block (REPL-EMB-2.7). Instrument ``offline_speed``.

    ``host_context`` must already be the full timing context (cpu-list, observed affinity, region-lock
    claim tag, loadavg start/end, frequency governor/range/throttled): a speed number without its
    window is refused at construction.
    """

    SPEED = True

    def __init__(
        self,
        run_dir: str | Path,
        run_id: str,
        corpus_digests: Mapping[str, str],
        query_digests: Mapping[str, str],
        arm_manifest_sha256: str,
        host_context: Mapping[str, Any],
        *,
        frozen_at_launch_path: str | Path | None = None,
        harness_path: str | Path | None = None,
        producer: str = "repl_embedder_speed",
        clock: Callable[[], str] | None = None,
    ) -> None:
        super().__init__(
            run_dir,
            run_id,
            rule=None,
            corpus_digests=corpus_digests,
            query_digests=query_digests,
            label_digests=None,
            arm_manifest_sha256=arm_manifest_sha256,
            host_context=host_context,
            timing=True,
            frozen_at_launch_path=frozen_at_launch_path,
            harness_path=harness_path,
            producer=producer,
            clock=clock,
        )
        self._floors: dict[tuple[str, str, str | None], dict] = {}

    def add_speed(
        self,
        arm_identity: Mapping[str, Any],
        metric: str,
        value: float,
        unit: str,
        n: int,
        spread: float,
        spread_unit: str,
        config_echo: Mapping[str, Any],
    ) -> None:
        if metric not in SPEED_METRICS:
            raise CaptureError(f"metric must be one of {sorted(SPEED_METRICS)}")
        if unit != SPEED_METRICS[metric][0]:
            raise CaptureError(
                f"{metric} is reported in {SPEED_METRICS[metric][0]!r}, not {unit!r}"
            )
        value = self._positive(value, "value")
        n = self._count(n, "n")
        spread = self._positive(spread, "spread", allow_zero=True)
        if not _text(spread_unit):
            raise CaptureError("spread_unit must be non-empty text")
        config = self._config(config_echo)
        identity = self._register_arm(arm_identity, "dense")
        leg = config_leg(config)
        self._once(("speed", identity["arm_id"], metric, leg))
        self._results.append(
            {
                "kind": "speed",
                "arm_id": identity["arm_id"],
                "metric": metric,
                "value": value,
                "n": n,
                "spread": spread,
                "spread_unit": spread_unit,
                "config_echo": config,
            }
        )

    def add_noise_floor(
        self,
        arm_identity: Mapping[str, Any],
        metric: str,
        aa_spread: float,
        unit: str,
        *,
        replication_unit: str,
        n: int,
        config_echo: Mapping[str, Any] | None = None,
    ) -> None:
        """The A/A spread of ``metric`` for this arm, in the metric's own unit.

        ``replication_unit`` (arm | session | process) and ``n`` (A/A repeats, >= 2) are required
        keyword arguments: a floor without them is not admissible as a gate input. With
        ``config_echo`` the floor binds to that one leg; without it, to every leg of (arm, metric).
        """
        if metric not in SPEED_METRICS:
            raise CaptureError(f"metric must be one of {sorted(SPEED_METRICS)}")
        if unit != SPEED_METRICS[metric][0]:
            raise CaptureError(
                f"a noise floor for {metric} is carried in {SPEED_METRICS[metric][0]!r}, "
                f"not {unit!r}"
            )
        aa_spread = self._positive(aa_spread, "aa_spread", allow_zero=True)
        if replication_unit not in REPLICATION_UNITS:
            raise CaptureError(f"replication_unit must be one of {REPLICATION_UNITS}")
        n = self._count(n, "noise-floor n", floor=2)
        identity = self._register_arm(arm_identity, "dense")
        leg = config_leg(self._config(config_echo)) if config_echo is not None else None
        key = (identity["arm_id"], metric, leg)
        if key in self._floors:
            raise CaptureError(f"duplicate noise floor for {key}")
        self._floors[key] = {
            "aa_spread": aa_spread,
            "unit": unit,
            "replication_unit": replication_unit,
            "n": n,
            "scope": "leg" if leg else "arm_metric",
        }

    def _preflight(self) -> None:
        if not self._results:
            raise CaptureError("a speed record needs at least one speed row")
        used: set[tuple] = set()
        for result in self._results:
            leg = config_leg(result["config_echo"])
            for key in (
                (result["arm_id"], result["metric"], leg),
                (result["arm_id"], result["metric"], None),
            ):
                if key in self._floors:
                    result["noise_floor"] = dict(self._floors[key])
                    used.add(key)
                    break
            else:
                raise CaptureError(
                    f"speed record lacks an A/A noise floor for arm {result['arm_id']} "
                    f"metric {result['metric']} (leg {leg}): a speed number is never "
                    "emitted without its floor"
                )
        orphans = sorted(set(self._floors) - used, key=str)
        if orphans:
            raise CaptureError(f"noise floors match no speed row: {orphans}")

    def _record(self, finished_at: str) -> dict:
        record = self._base_record(SPEED_RECORD_SCHEMA, INSTRUMENT_SPEED, finished_at)
        record["results"] = self._results
        return record


__all__ = [
    "ARM_FIELDS",
    "CAPTURE_SCHEMA",
    "COST_METRICS",
    "EXTRA_FIELDS",
    "HOOK_SINCE",
    "INSTRUMENTS",
    "INSTRUMENT_RECALL",
    "INSTRUMENT_SPEED",
    "MODES",
    "POOLED_CATEGORY",
    "PROVENANCE_FIELDS",
    "RECALL_KS",
    "REGISTERED_RULE",
    "REPLICATION_UNITS",
    "RETR_RECORD_SCHEMA",
    "RETR_SOURCE_KIND",
    "ROLES",
    "ROW_FIELDS",
    "SIDECAR_SUFFIX",
    "SOURCE_INSTRUMENT",
    "SPEED_METRICS",
    "SPEED_RECORD_SCHEMA",
    "SPEED_SOURCE_KIND",
    "CaptureError",
    "HostWindow",
    "RetrievalRun",
    "SpeedRun",
    "build_rows",
    "check_arm_identity",
    "check_host_context",
    "check_rule",
    "claim_tuple_problems",
    "config_leg",
    "content_hash",
    "file_sha256",
    "locator",
    "measurement_identity",
    "record_path_for",
    "registered_rule",
    "row_digest",
    "sample_host",
    "sidecar_path",
    "validate_row",
    "validate_rows",
]
