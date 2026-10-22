#!/usr/bin/env python3
"""VB-HIL-WRITE producer call: emit ``harness_improvement.beliefs.jsonl`` at a HIL eval run's finalize.

The carrier and strict reader live in epyc-root
(``scripts/vidya/adapters/harness_improvement.py``, root ``7335da327``); this is the ORCHESTRATOR
side of the wire: it computes the HIL-2 item 14 arm pins from the objects the run actually used and
calls ``write_belief_measurements(...)``. Same opt-in location pattern as
``scripts/benchmark/toc_ab_sidecar.py``: the carrier is found under ``--epyc-root``.

No HIL feature-A/B eval runner exists yet (HIL-6, the loop architecture, is the operator's open
A/B/C choice), so :func:`finalize_hil_run` IS the finalize hook: whichever runner executes the first
HIL efficacy run calls it once, after the raw per-item result file is closed. It never backfills:
missing pins, a missing raw file, or an invalid row are refusals (``CaptureError`` from the
carrier, ``ValueError`` here), not guesses.

Item 14 pins (closed set, every one required), and how they are computed here:

* ``tool_surface_sha256``: sha256 over canonical JSON of the sorted ``[{name, parameters}]`` of the
  tools the arm exposed (:func:`tool_surface_digest`; accepts ``ToolRegistry.list_tools(role)``
  dicts or ``Tool`` objects; descriptions are excluded on purpose: they are prompt text and are
  bound by ``prompt_param_sha256``).
* ``compaction_policy_sha256``: sha256 over canonical JSON of the compaction policy in force
  (:func:`current_compaction_policy` reads the session-compaction config and flags).
* ``served_context_window``: the per-request window the arm was served (int, from the run).
* ``wall_time_budget_s`` / ``turn_budget``: the run's budgets.
* ``final_result_selection_rule``: the rule that picked the scored answer (e.g. ``"last FINAL()"``).

CLI (for a runner that is not Python)::

    hil_ab_sidecar.py --spec run.json --out-dir DIR [--epyc-root DIR]  (default: $EPYC_ROOT, else the epyc-root-main view, else /workspace)

where ``run.json`` = ``{"run_id", "results_path", "observations": [...], "arm_pins": {...},
"protocol_id"?}``; each observation is the carrier's field set without ``arm_pins`` (the run-level
pins are applied to every observation unless an observation carries its own).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PRODUCER = "epyc-orchestrator/scripts/benchmark/hil_ab_sidecar.py"
ARM_PIN_FIELDS = ("tool_surface_sha256", "served_context_window", "compaction_policy_sha256",
                  "wall_time_budget_s", "turn_budget", "final_result_selection_rule")
#: Observation fields the carrier reads (``write_belief_measurements`` docstring), minus arm_pins.
OBSERVATION_FIELDS = ("feature_id", "arm", "objective", "metric", "value", "unit", "metric_direction",
                      "native_verdict", "suite_id", "suite_version", "split", "scored", "attempted",
                      "prompt_param_path", "prompt_param_sha256", "harness_pin", "config_sha256",
                      "identities")


from scripts.lib.epyc_root import default_epyc_root  # noqa: E402


def canonical_sha256(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, default=str).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tool_surface_digest(tools: Iterable[Any]) -> str:
    """sha256 of the exposed tool surface: names and parameter schemas, sorted by name."""
    surface = []
    for t in tools:
        if isinstance(t, Mapping):
            name, params = t.get("name"), t.get("parameters", t.get("schema"))
        else:
            name, params = getattr(t, "name", None), getattr(t, "parameters", None)
        if not isinstance(name, str) or not name:
            raise ValueError(f"tool without a name in the tool surface: {t!r}")
        surface.append({"name": name, "parameters": params if params is not None else {}})
    names = [s["name"] for s in surface]
    if len(set(names)) != len(names):
        raise ValueError("duplicate tool names in the tool surface")
    return canonical_sha256(sorted(surface, key=lambda s: s["name"]))


def current_compaction_policy() -> dict[str, Any]:
    """The compaction policy in force: session-compaction flags plus the chat config knobs."""
    from src.features import features

    f = features()
    pol: dict[str, Any] = {k: bool(getattr(f, k, False)) for k in (
        "session_compaction", "session_compaction_llm_index", "two_level_condensation",
        "session_token_budget", "tool_output_compression", "escalation_compression")}
    try:
        from src.config import get_config

        chat = get_config().chat
        for k in ("session_compaction_keep_recent_ratio", "session_compaction_recompaction_interval",
                  "session_compaction_min_turns", "session_compaction_trigger_ratio"):
            pol[k] = getattr(chat, k, None)
    except Exception as exc:  # recorded, never guessed
        pol["config_unavailable"] = type(exc).__name__
    return pol


def compaction_policy_digest(policy: Mapping[str, Any] | None = None) -> str:
    return canonical_sha256(dict(policy) if policy is not None else current_compaction_policy())


def arm_pins(*, tools: Iterable[Any] | None = None, tool_surface_sha256: str | None = None,
             served_context_window: int, wall_time_budget_s: float, turn_budget: int,
             final_result_selection_rule: str, compaction_policy: Mapping[str, Any] | None = None,
             compaction_policy_sha256: str | None = None) -> dict[str, Any]:
    """Build the closed item-14 pin set. Give either ``tools`` or a precomputed digest."""
    if (tools is None) == (tool_surface_sha256 is None):
        raise ValueError("pass exactly one of tools / tool_surface_sha256")
    return {
        "tool_surface_sha256": tool_surface_sha256 or tool_surface_digest(tools or ()),
        "served_context_window": int(served_context_window),
        "compaction_policy_sha256": compaction_policy_sha256 or compaction_policy_digest(compaction_policy),
        "wall_time_budget_s": float(wall_time_budget_s),
        "turn_budget": int(turn_budget),
        "final_result_selection_rule": str(final_result_selection_rule),
    }


def _carrier(epyc_root: str | Path | None):
    from scripts.lib.epyc_root import resolve_epyc_root

    vidya = str(resolve_epyc_root(epyc_root) / "scripts" / "vidya")
    if not (Path(vidya) / "adapters" / "harness_improvement.py").is_file():
        raise FileNotFoundError(f"VB-HIL-WRITE carrier not found under {vidya}")
    if vidya not in sys.path:
        sys.path.insert(0, vidya)
    from adapters import harness_improvement  # type: ignore

    return harness_improvement


def finalize_hil_run(out_dir: str | Path, *, run_id: str, results_path: str | Path,
                     observations: Sequence[Mapping[str, Any]], pins: Mapping[str, Any],
                     protocol_id: str | None = None, epyc_root: str | Path | None = None,
                     emitted_at: str | None = None) -> Path:
    """THE finalize hook: one self-hashed row per run x arm x feature x objective metric.

    ``pins`` (from :func:`arm_pins`) apply to every observation that does not carry its own
    ``arm_pins`` (a candidate arm that changed the tool surface must carry its own).
    """
    obs = []
    for o in observations:
        unknown = sorted(set(o) - set(OBSERVATION_FIELDS) - {"arm_pins"})
        if unknown:
            raise ValueError(f"observation has fields the carrier does not read: {unknown}")
        row = {k: o[k] for k in OBSERVATION_FIELDS if k in o}
        row["arm_pins"] = dict(o.get("arm_pins") or pins)
        if sorted(row["arm_pins"]) != sorted(ARM_PIN_FIELDS):
            raise ValueError(f"arm_pins must be exactly {ARM_PIN_FIELDS}")
        pp = row.get("prompt_param_path")
        if pp and Path(pp).is_file() and "prompt_param_sha256" not in row:
            row["prompt_param_sha256"] = file_sha256(pp)
        obs.append(row)
    hi = _carrier(epyc_root)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    return hi.write_belief_measurements(out_dir, run_id=run_id, producer=PRODUCER,
                                        results_path=results_path, observations=obs,
                                        protocol_id=protocol_id, emitted_at=emitted_at)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epyc-root", default=default_epyc_root("scripts/vidya/adapters/harness_improvement.py"))
    a = ap.parse_args(argv)
    spec = json.loads(Path(a.spec).read_text(encoding="utf-8"))
    out = finalize_hil_run(a.out_dir, run_id=spec["run_id"], results_path=spec["results_path"],
                           observations=spec["observations"], pins=spec["arm_pins"],
                           protocol_id=spec.get("protocol_id"), epyc_root=a.epyc_root)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
