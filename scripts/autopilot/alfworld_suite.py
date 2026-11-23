"""ALFWorld as a small rotating EvalTower suite slice (AP-ALFWORLD-SUITE, default off).

Orchestrator-venv side. The native ALFWorld environment lives in the Life-Harness checkout's own Python 3.10 venv
(``<lh_root>/AgentBench/.native/alfworld-venv``), so nothing here imports it: an episode batch is run by
``scripts/autopilot/alfworld_episode_driver.py`` as a SUBPROCESS (request JSON in, one JSONL row per finished episode
out). Reused, not duplicated: that venv, the AgentBench data under ``<lh_root>/AgentBench/data/alfworld``, AgentBench's own
env wrapper / task text / action matcher (inside the driver), and the split-qualified id convention of
``scripts/harness/life_harness`` in epyc-root (``rigor.qualify``: ``<split>:<index>``, the index being the position in
the split's flattened game list, which is what the Life-Harness batches index too).

Design (each point is a decision, documented here and in ``scripts/autopilot/program.md``):

* **Flag.** ``autopilot_alfworld_suite`` (env ``AUTOPILOT_ALFWORLD_SUITE``), default off. Off = ``eval_t1`` runs none of this:
  no suite change, no new ``core_id``, no ALFWorld import, no file read.
* **Draw.** ``AUTOPILOT_ALFWORLD_N`` episodes (default 5) from the TRAIN split ``train_valid`` only. The sealed TEST
  split ``new_std`` is never read for a draw (``draw_slice`` raises on any qid outside the train split; the sealed qids
  are only ever returned by ``sealed_qids`` for the R3 check). Seeded, stratified round-robin over the six task types
  with a seeded type order, so five episodes cover five different task types. The seed is a hash of the eval's base seed
  and the core rotation index (``AUTOPILOT_EVAL_CORE_ROTATION_TRIALS`` trials per block), so the slice is stable inside a
  rotation block (paired arms see the same games) and rotates between blocks. Qids in ``exclude_qids`` (the BSV-2 rigor
  partition: proposal batch + sealed split) are skipped, so an accept draw is disjoint from the proposal draw (R1).
* **Identity.** qid ``alfworld:<split>:<index>`` (split-qualified, ``alfworld:train_valid:1873``). A slice also carries
  the sha256 of ``train_valid.json`` so a changed game list is a different instrument.
* **Scoring.** Episode success = 1 (the environment reported ``won``), else 0. Wrong-answer evidence: ``completed`` and
  ``solver_limit`` rows score 0 in the denominator; ``candidate_error`` (same action three times) is ``task_failed``
  (ETR-1: scores 0, stays in the denominator); ``infra_error`` (endpoint down / env raised) is ``infra_failed`` and
  carries no quality information. Row ``status`` is the R4 vocabulary of ``bsv2_accept_rigor``.
* **Budgets.** Per episode: ``AUTOPILOT_ALFWORLD_MAX_STEPS`` model turns (default 30; a turn with no tool call still
  costs one) and ``AUTOPILOT_ALFWORLD_EPISODE_WALL_S`` seconds (default 600). Whole batch:
  ``AUTOPILOT_ALFWORLD_BATCH_WALL_S`` (default ``n * episode_wall + 120``); on expiry the driver's own process group is
  killed and every episode with no row is ``infra_error: batch_wall_timeout``. Episodes run sequentially (the env wrapper
  serialises on one lock and the slice is small); R5 budget unit is rollouts = episodes.
* **Routing.** The served model under test is reached through the orchestrator API's OpenAI-compatible
  ``POST <tower url>/v1/chat/completions`` with the client tool ``take_action``; ``AUTOPILOT_ALFWORLD_ROLE`` (optional)
  sets ``x_force_role``, ``AUTOPILOT_ALFWORLD_MODEL`` the ``model`` field (default ``orchestrator``).
  ``AUTOPILOT_ALFWORLD_ENDPOINT`` overrides the base URL, e.g. the Life-Harness solver loopback proxy
  (``http://127.0.0.1:8400/v1``, ``scripts/harness/life_harness/proxy.py``) to reach a bare llama-server.
* **Decision weight.** The slice rows ride in partition ``alfworld`` and are EXCLUDED from the scalar quality / speed
  aggregate (five episodes cannot move a 0-3 mean without adding noise, and the batch wall clock would pollute the speed
  objective); they stay in ``question_results`` and ``details["alfworld_slice"]``, which is what the paired accept gate
  (``screen_paired_arms`` over qids, ``bsv2_accept_rigor`` R1-R5) reads.

Nothing here raises into the tower: an unusable setup yields rows flagged ``infra_error`` (or no slice at all and a
``status: unavailable`` note), never a crash and never a fabricated success.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

FLAG_ENV = "AUTOPILOT_ALFWORLD_SUITE"
SUITE = "alfworld"
PARTITION = "alfworld"
TRAIN_SPLIT = "train_valid"
TEST_SPLIT = "new_std"  # sealed: never drawn
DEFAULT_LH_ROOT = "/mnt/raid0/llm/worktrees/life-harness"
DRIVER_PATH = Path(__file__).resolve().with_name("alfworld_episode_driver.py")
STATUSES = ("completed", "solver_limit", "infra_error", "candidate_error")
_SAFE_ENV = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "USER")


def flag_on() -> bool:
    """The default-off suite flag, honouring a live ``POST /config`` override (same overlay as the LH flags)."""
    import harness_layers

    return harness_layers.flag_on(FLAG_ENV)


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    try:
        return max(minimum, int(os.environ.get(name, "").strip() or default))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default
    return value if value > 0 else default


@dataclass(frozen=True)
class SuiteConfig:
    n: int = 5
    max_steps: int = 30
    episode_wall_s: float = 600.0
    request_timeout_s: float = 180.0
    batch_wall_s: float = 0.0  # 0 = derived from n and episode_wall_s
    lh_root: str = DEFAULT_LH_ROOT
    python: str = ""  # "" = <lh_root>/AgentBench/.native/alfworld-venv/bin/python
    driver: str = str(DRIVER_PATH)
    env_adapter: str = "agentbench"
    endpoint: str = ""  # "" = <tower url>/v1
    model: str = "orchestrator"
    force_role: str = ""
    max_tokens: int = 1024
    work_dir: str = ""  # "" = system temp

    @property
    def agentbench_root(self) -> Path:
        return Path(self.lh_root) / "AgentBench"

    @property
    def venv_python(self) -> str:
        return self.python or str(self.agentbench_root / ".native" / "alfworld-venv" / "bin" / "python")

    @property
    def batch_budget_s(self) -> float:
        return self.batch_wall_s if self.batch_wall_s > 0 else self.n * self.episode_wall_s + 120.0

    def as_json(self) -> dict[str, Any]:
        return {"n": self.n, "max_steps": self.max_steps, "episode_wall_s": self.episode_wall_s,
                "batch_wall_s": self.batch_budget_s, "model": self.model, "force_role": self.force_role,
                "endpoint": self.endpoint or "<tower>/v1", "env_adapter": self.env_adapter}


def config_from_env() -> SuiteConfig:
    e = os.environ.get
    return SuiteConfig(
        n=_env_int("AUTOPILOT_ALFWORLD_N", 5),
        max_steps=_env_int("AUTOPILOT_ALFWORLD_MAX_STEPS", 30),
        episode_wall_s=_env_float("AUTOPILOT_ALFWORLD_EPISODE_WALL_S", 600.0),
        request_timeout_s=_env_float("AUTOPILOT_ALFWORLD_REQUEST_TIMEOUT_S", 180.0),
        batch_wall_s=_env_float("AUTOPILOT_ALFWORLD_BATCH_WALL_S", 0.0) if e("AUTOPILOT_ALFWORLD_BATCH_WALL_S") else 0.0,
        lh_root=(e("AUTOPILOT_ALFWORLD_LH_ROOT") or DEFAULT_LH_ROOT).strip(),
        python=(e("AUTOPILOT_ALFWORLD_PYTHON") or "").strip(),
        driver=(e("AUTOPILOT_ALFWORLD_DRIVER") or str(DRIVER_PATH)).strip(),
        env_adapter=(e("AUTOPILOT_ALFWORLD_ENV_ADAPTER") or "agentbench").strip(),
        endpoint=(e("AUTOPILOT_ALFWORLD_ENDPOINT") or "").strip(),
        model=(e("AUTOPILOT_ALFWORLD_MODEL") or "orchestrator").strip(),
        force_role=(e("AUTOPILOT_ALFWORLD_ROLE") or "").strip(),
        max_tokens=_env_int("AUTOPILOT_ALFWORLD_MAX_TOKENS", 1024),
        work_dir=(e("AUTOPILOT_ALFWORLD_WORK_DIR") or "").strip(),
    )


# ------------------------------------------------------------------ ids and games
def qualify(split: str, index: int | str) -> str:
    """``alfworld:<split>:<index>`` (the Life-Harness ``<split>:<index>`` convention, suite-prefixed)."""
    return f"{SUITE}:{split}:{index}"


def split_of(qid: str) -> str | None:
    parts = str(qid).split(":")
    return parts[1] if len(parts) == 3 and parts[0] == SUITE else None


@dataclass
class Game:
    qid: str
    split: str
    index: int
    kind: str  # task type, the key of the split file
    rel_path: str  # relative to AgentBench/data/alfworld
    abs_path: str


def load_split(cfg: SuiteConfig, split: str) -> tuple[list[Game], str]:
    """Flattened game list of one split file (file key order = AgentBench's own ordering) and the file's sha256."""
    path = cfg.agentbench_root / "data" / "alfworld" / f"{split}.json"
    raw = path.read_bytes()
    content = json.loads(raw)
    games: list[Game] = []
    for kind, files in content.items():
        for rel in files:
            idx = len(games)
            games.append(Game(qualify(split, idx), split, idx, str(kind), str(rel),
                              str(path.parent / rel)))
    return games, hashlib.sha256(raw).hexdigest()


def sealed_qids(cfg: SuiteConfig) -> list[str]:
    """R3: every qid of the sealed TEST split. Used only to seal them; never to draw."""
    games, _ = load_split(cfg, TEST_SPLIT)
    return [g.qid for g in games]


def slice_seed(base_seed: int, rotation: int) -> int:
    payload = f"alfworld-slice-v1\x00{int(base_seed)}\x00{int(rotation)}".encode()
    return int(hashlib.sha256(payload).hexdigest()[:16], 16)


def draw_slice(games: Sequence[Game], n: int, seed: int, *, exclude_qids: Iterable[str] | None = None) -> list[Game]:
    """Seeded stratified draw of up to ``n`` TRAIN games (round-robin over task types, seeded type order).

    Raises ``ValueError`` if any candidate is outside the train split: the sealed split can never be drawn.
    """
    excluded = {str(q) for q in (exclude_qids or ())}
    for g in games:
        if g.split != TRAIN_SPLIT:
            raise ValueError(f"R3: refusing to draw from split {g.split!r}; only {TRAIN_SPLIT!r} is drawable")
    rng = random.Random(seed)
    by_kind: dict[str, list[Game]] = {}
    for g in games:
        if g.qid not in excluded:
            by_kind.setdefault(g.kind, []).append(g)
    kinds = sorted(by_kind)
    rng.shuffle(kinds)
    for kind in kinds:
        rng.shuffle(by_kind[kind])
    picked: list[Game] = []
    while len(picked) < n and any(by_kind.values()):
        for kind in kinds:
            if by_kind[kind] and len(picked) < n:
                picked.append(by_kind[kind].pop())
    return picked


def slice_identity(picked: Sequence[Game], data_sha: str) -> str:
    h = hashlib.sha256(data_sha.encode())
    for g in picked:
        h.update(b"\x00" + g.qid.encode())
    return h.hexdigest()


def core_id_suffix(picked: Sequence[Game], data_sha: str) -> str:
    """Deterministic per-slice suffix: a different slice is a different instrument in the drift ledger."""
    return f"_alfw{len(picked)}_{slice_identity(picked, data_sha)[:8]}"


def slice_questions(picked: Sequence[Game], cfg: SuiteConfig, data_sha: str) -> list[dict[str, Any]]:
    """Question-shaped identity rows (dataset hash / test profile); the episode itself is run by the driver."""
    return [{"suite": SUITE, "id": g.qid, "qid": g.qid, "prompt": f"alfworld:{g.rel_path}", "expected": "won",
             "scoring_method": "episode_success",
             "scoring_config": {"split": g.split, "kind": g.kind, "max_steps": cfg.max_steps,
                                "episode_wall_s": cfg.episode_wall_s, "data_sha256": data_sha}}
            for g in picked]


# ------------------------------------------------------------------ running episodes
def _driver_env() -> dict[str, str]:
    env = {k: os.environ[k] for k in _SAFE_ENV if k in os.environ}
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    return env


def _infra_row(g: Game, detail: str) -> dict[str, Any]:
    return {"qid": g.qid, "index": g.index, "game": g.abs_path, "success": 0, "status": "infra_error",
            "detail": detail, "steps": 0, "model_calls": 0, "tokens_prompt": 0, "tokens_completion": 0,
            "elapsed_s": 0.0}


def _stop_own_group(proc: subprocess.Popen) -> None:
    """Stop the driver's OWN process group (a session this module created): SIGTERM, then SIGKILL, then verify."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        if proc.poll() is not None:
            break
        try:
            os.killpg(proc.pid, sig)
        except (ProcessLookupError, PermissionError):
            break
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            continue
    proc.poll()


def run_episodes(picked: Sequence[Game], cfg: SuiteConfig, *, tower_url: str,
                 popen: Callable[..., subprocess.Popen] = subprocess.Popen) -> dict[str, Any]:
    """Run the slice in the driver subprocess. Returns ``{"rows": [...], "artifact_dir", "driver": {...}}``."""
    if cfg.work_dir:
        Path(cfg.work_dir).mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="alfworld_suite_", dir=cfg.work_dir or None))
    request = {
        "agentbench_root": str(cfg.agentbench_root), "env_adapter": cfg.env_adapter,
        "endpoint": cfg.endpoint or tower_url.rstrip("/") + "/v1", "model": cfg.model, "force_role": cfg.force_role,
        "max_steps": cfg.max_steps, "episode_wall_s": cfg.episode_wall_s, "request_timeout_s": cfg.request_timeout_s,
        "max_tokens": cfg.max_tokens, "temperature": 0.0,
        "episodes": [{"qid": g.qid, "index": g.index, "game": g.abs_path} for g in picked],
    }
    req_path, out_path, log_path = work / "request.json", work / "rows.jsonl", work / "driver.log"
    req_path.write_text(json.dumps(request, indent=1, sort_keys=True))
    info: dict[str, Any] = {"python": cfg.venv_python, "driver": cfg.driver, "returncode": None, "timed_out": False}
    rows: dict[str, dict[str, Any]] = {}
    started = time.monotonic()
    if not Path(cfg.venv_python).exists() or not Path(cfg.driver).exists():
        info["error"] = "driver_unavailable"
    else:
        with log_path.open("w") as log:
            try:
                proc = popen([cfg.venv_python, "-u", cfg.driver, "--request", str(req_path), "--out", str(out_path)],
                             cwd=str(cfg.agentbench_root if cfg.agentbench_root.is_dir() else work), env=_driver_env(),
                             stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            except OSError as exc:
                info["error"] = f"driver_spawn_failed: {type(exc).__name__}"
                proc = None
            if proc is not None:
                try:
                    info["returncode"] = proc.wait(timeout=cfg.batch_budget_s)
                except subprocess.TimeoutExpired:
                    info["timed_out"] = True
                    _stop_own_group(proc)
                    info["returncode"] = proc.poll()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict) and row.get("qid"):
                rows[str(row["qid"])] = row
    reason = ("batch_wall_timeout" if info["timed_out"] else info.get("error")
              or f"driver_exit_{info['returncode']}_no_row")
    ordered = [rows.get(g.qid) or _infra_row(g, reason) for g in picked]
    info["elapsed_s"] = round(time.monotonic() - started, 3)
    return {"rows": ordered, "artifact_dir": str(work), "driver": info}


def status_split(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    split = {s: 0 for s in STATUSES}
    for r in rows:
        s = str(r.get("status") or "")
        split[s if s in split else "candidate_error"] += 1
    return split


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scored = [r for r in rows if r.get("status") != "infra_error"]
    return {"n": len(rows), "successes": sum(int(r.get("success") or 0) for r in rows),
            "scored_n": len(scored), "status_split": status_split(rows),
            "tokens": sum(int(r.get("tokens_prompt") or 0) + int(r.get("tokens_completion") or 0) for r in rows),
            "model_calls": sum(int(r.get("model_calls") or 0) for r in rows),
            "elapsed_s": round(sum(float(r.get("elapsed_s") or 0.0) for r in rows), 3)}


@dataclass
class SliceRun:
    """Everything one eval needs from the suite: the drawn games, their rows, and the identity pieces."""

    games: list[Game]
    rows: list[dict[str, Any]]
    data_sha: str
    cfg: SuiteConfig
    artifact_dir: str = ""
    driver: dict[str, Any] = field(default_factory=dict)

    @property
    def core_suffix(self) -> str:
        return core_id_suffix(self.games, self.data_sha)

    def questions(self) -> list[dict[str, Any]]:
        return slice_questions(self.games, self.cfg, self.data_sha)

    def descriptor(self) -> dict[str, Any]:
        return {"status": "ran", "suite": SUITE, "split": TRAIN_SPLIT, "sealed_split": TEST_SPLIT,
                "qids": [g.qid for g in self.games], "kinds": [g.kind for g in self.games],
                "data_sha256": self.data_sha, "config": self.cfg.as_json(), "artifact_dir": self.artifact_dir,
                "driver": self.driver, "summary": summarize(self.rows), "core_id_suffix": self.core_suffix}


def run_slice(*, tower_url: str, base_seed: int, rotation: int, exclude_qids: Iterable[str] | None = None,
              cfg: SuiteConfig | None = None,
              popen: Callable[..., subprocess.Popen] = subprocess.Popen) -> SliceRun | dict[str, Any]:
    """Draw and run one slice. Returns a ``SliceRun``, or ``{"status": "unavailable", "reason": ...}`` when no
    slice could be drawn (missing data, empty pool): the caller then leaves the eval exactly as the flag-off path."""
    cfg = cfg or config_from_env()
    try:
        games, data_sha = load_split(cfg, TRAIN_SPLIT)
        picked = draw_slice(games, cfg.n, slice_seed(base_seed, rotation), exclude_qids=exclude_qids)
    except (OSError, ValueError, KeyError) as exc:
        return {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"[:300]}
    if not picked:
        return {"status": "unavailable", "reason": "empty_draw_after_exclusions"}
    ran = run_episodes(picked, cfg, tower_url=tower_url, popen=popen)
    return SliceRun(games=picked, rows=ran["rows"], data_sha=data_sha, cfg=cfg,
                    artifact_dir=ran["artifact_dir"], driver=ran["driver"])


def main(argv: Sequence[str] | None = None) -> int:
    """Standalone slice run for the live inference row: no EvalTower, no Autopilot, the same draw and driver.

    ``python3 scripts/autopilot/alfworld_suite.py --out <dir> [--seed 42 --rotation 0 --tower-url http://localhost:8000]``
    writes ``<dir>/slice_result.json`` (descriptor, per-episode rows, successes, status split, episodes/hour,
    tokens/episode) and exits 0 when every episode produced a verdict, 6 when any row is ``infra_error`` (no verdict
    about the model: re-run, never score), 2 when no slice could be drawn. The model endpoint and budgets come from the
    ``AUTOPILOT_ALFWORLD_*`` env knobs, so the caller owns which served model is under test.
    """
    import argparse

    parser = argparse.ArgumentParser(description=main.__doc__.split("\n")[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rotation", type=int, default=0)
    parser.add_argument("--tower-url", default="http://localhost:8000")
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    run = run_slice(tower_url=args.tower_url, base_seed=args.seed, rotation=args.rotation)
    if isinstance(run, dict):
        (out / "slice_result.json").write_text(json.dumps(run, indent=1, sort_keys=True))
        return 2
    summary = summarize(run.rows)
    hours = summary["elapsed_s"] / 3600.0
    result = {**run.descriptor(), "rows": run.rows,
              "episodes_per_hour": round(len(run.rows) / hours, 2) if hours > 0 else None,
              "tokens_per_episode": round(summary["tokens"] / len(run.rows), 1) if run.rows else None,
              "success_rate": round(summary["successes"] / len(run.rows), 4) if run.rows else None}
    (out / "slice_result.json").write_text(json.dumps(result, indent=1, sort_keys=True, default=str))
    print(json.dumps({k: result[k] for k in ("summary", "episodes_per_hour", "tokens_per_episode", "success_rate")}, sort_keys=True))
    return 6 if summary["status_split"]["infra_error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
