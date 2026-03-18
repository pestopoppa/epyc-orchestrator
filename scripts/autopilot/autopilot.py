#!/usr/bin/env python3
"""AutoPilot: Continuous recursive optimization for the EPYC orchestration stack.

Main controller loop: observe → reason → act → evaluate → record → meta-learn.

Usage:
    python autopilot.py start [--dry-run] [--max-trials N]
    python autopilot.py status
    python autopilot.py pause
    python autopilot.py resume
    python autopilot.py report
    python autopilot.py plot
    python autopilot.py checkpoint [--production-best]
    python autopilot.py restore [--checkpoint PATH]
"""

from __future__ import annotations

import argparse
import fcntl
import json
import logging
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
ORCH_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from experiment_journal import ExperimentJournal, JournalEntry
from pareto_archive import ParetoArchive, ParetoEntry
from safety_gate import EvalResult, SafetyGate
from eval_tower import EvalTower
from config_applicator import apply_params, health_check
from meta_optimizer import MetaOptimizer, SpeciesBudget
from progress_plots import generate_all_plots
from species import Seeder, NumericSwarm, PromptForge, StructuralLab

log = logging.getLogger("autopilot")

STATE_PATH = ORCH_ROOT / "orchestration" / "autopilot_state.json"
LOCK_PATH = ORCH_ROOT / "orchestration" / ".autopilot.lock"
ORCHESTRATOR_URL = "http://localhost:8000"
PLOT_INTERVAL = 10  # Generate plots every N trials

# ── Controller Prompt Template ───────────────────────────────────

CONTROLLER_PROMPT_TEMPLATE = """\
You are the AutoPilot meta-reasoning controller for an LLM orchestration stack.
Your job: analyze current system state and propose the SINGLE best next action.

## Current State

### Pareto Archive
{pareto_summary}

### Experiment Journal (last 20 entries)
{journal_summary}

### Seeder Status
{seeder_status}

### Species Effectiveness
{species_effectiveness}

### System Health
- Orchestrator: {health_status}
- Memory count: {memory_count}
- Q-value converged: {converged}

### Species Budget
{budget}

### Plot Paths (reference for trend analysis)
{plot_paths}

## Action Guidelines

1. If memories < 500: ALWAYS prioritize seeding (seed_batch)
2. If Q-values converged and models not trained: trigger train_routing_models
3. If models trained and not enabled: try structural_experiment with routing features
4. If stagnating (hv_slope < 0.001): try prompt_mutation or widen numeric search
5. If quality regression after changes: rollback to last good checkpoint
6. Consider the species budget allocation when choosing actions

## Available Actions

Respond with EXACTLY ONE action in a ```json:autopilot_actions block:

- Seed: {{"type": "seed_batch", "n_questions": 10-50, "suites": ["coder","math",...]}}
- Numeric: {{"type": "numeric_trial", "surface": "memrl_retrieval|think_harder|monitor|escalation", "params": {{}}}}
  (Leave params empty to let Optuna suggest; provide params to test specific values)
- Prompt: {{"type": "prompt_mutation", "file": "frontdoor.md", "mutation": "targeted_fix|compress|few_shot_evolution", "description": "..."}}
- Structural: {{"type": "structural_experiment", "flags": {{"feature_name": true/false}}}}
- Train: {{"type": "train_routing_models", "min_memories": 500}}
- Distill: {{"type": "distill_skillbank", "teacher": "claude", "categories": ["routing"]}}
- Reset: {{"type": "reset_memories", "keep_seen": true, "keep_skills": true}}
- Deep eval: {{"type": "deep_eval", "tier": 2}}
- Rollback: {{"type": "rollback", "to_checkpoint": "production_best"}}

Include brief reasoning before the action block.
"""


# ── State Management ─────────────────────────────────────────────


def load_state() -> dict[str, Any]:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {
        "trial_counter": 0,
        "session_id": None,
        "paused": False,
        "species_budget": SpeciesBudget().as_dict(),
        "td_errors": [],
    }


def save_state(state: dict[str, Any]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2, default=str))


# ── Claude CLI Controller ────────────────────────────────────────


def invoke_controller(
    prompt: str,
    session_id: str | None = None,
    timeout: int = 300,
) -> tuple[str, str | None]:
    """Invoke Claude CLI for meta-reasoning.

    Returns (response_text, session_id).
    """
    cmd = [
        "claude", "-p", prompt,
        "--output-format", "json",
        "--allowedTools", "Read,Grep,Glob",
    ]
    if session_id:
        cmd.extend(["--resume", session_id])

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(ORCH_ROOT),
        )
        stdout, stderr = proc.communicate(timeout=timeout)

        if proc.returncode != 0:
            log.error("Controller failed (rc=%d): %s", proc.returncode, stderr[:500])
            return "", session_id

        try:
            response = json.loads(stdout)
            new_session = response.get("session_id", session_id)
            return response.get("result", stdout), new_session
        except json.JSONDecodeError:
            return stdout, session_id

    except subprocess.TimeoutExpired:
        proc.kill()
        log.error("Controller timed out after %ds", timeout)
        return "", session_id
    except FileNotFoundError:
        log.error("Claude CLI not found")
        return "", session_id


def extract_action(text: str) -> dict[str, Any] | None:
    """Extract structured action from controller response."""
    marker = "```json:autopilot_actions"
    if marker in text:
        start = text.index(marker) + len(marker)
        end = text.index("```", start)
        try:
            return json.loads(text[start:end].strip())
        except json.JSONDecodeError as e:
            log.error("Failed to parse action JSON: %s", e)
            return None

    # Fallback: look for any JSON block
    if "```json" in text:
        start = text.index("```json") + len("```json")
        end = text.index("```", start)
        try:
            data = json.loads(text[start:end].strip())
            if "type" in data:
                return data
        except (json.JSONDecodeError, ValueError):
            pass

    return None


# ── Action Dispatch ──────────────────────────────────────────────


def dispatch_action(
    action: dict[str, Any],
    seeder: Seeder,
    swarm: NumericSwarm,
    forge: PromptForge,
    lab: StructuralLab,
    tower: EvalTower,
    gate: SafetyGate,
    archive: ParetoArchive,
    state: dict[str, Any],
) -> tuple[EvalResult | None, str]:
    """Execute an action and return (eval_result, species_name)."""
    action_type = action.get("type", "")
    log.info("Dispatching action: %s", action_type)

    if action_type == "seed_batch":
        n = action.get("n_questions", 10)
        suites = action.get("suites")
        result = seeder.run_batch(n_questions=n, suites=suites)
        # After seeding, run T0 eval
        eval_result = tower.eval_t0()
        return eval_result, "seeder"

    elif action_type == "numeric_trial":
        surface = action.get("surface", "memrl_retrieval")
        explicit_params = action.get("params", {})

        if explicit_params:
            # Apply explicit params
            apply_params(explicit_params)
        else:
            # Let Optuna suggest
            trial = swarm.suggest_trial(surface)
            apply_params(trial["params"])
            state["_current_optuna_trial"] = {
                "surface": surface,
                "trial_number": trial["trial_number"],
            }

        eval_result = tower.eval_t0()
        # Report to Optuna if we have a trial
        if "_current_optuna_trial" in state and eval_result:
            t = state.pop("_current_optuna_trial")
            swarm.report_result(
                t["surface"], t["trial_number"], eval_result.objectives
            )
        return eval_result, "numeric_swarm"

    elif action_type == "prompt_mutation":
        target = action.get("file", "frontdoor.md")
        mutation_type = action.get("mutation", "targeted_fix")
        description = action.get("description", "")

        mutation = forge.propose_mutation(
            target_file=target,
            mutation_type=mutation_type,
            description=description,
        )
        forge.apply_mutation(mutation)
        eval_result = tower.eval_t0()

        # Revert if quality drops
        verdict = gate.check(eval_result)
        if not verdict:
            log.warning("Prompt mutation failed safety gate, reverting")
            forge.revert_mutation(mutation)
            return eval_result, "prompt_forge"

        return eval_result, "prompt_forge"

    elif action_type == "structural_experiment":
        flags = action.get("flags", {})
        validation = lab.propose_flag_experiment(flags)
        if validation.get("status") != "valid":
            log.warning("Invalid flag experiment: %s", validation)
            return None, "structural_lab"

        lab.apply_flag_experiment(flags)
        eval_result = tower.eval_t0()

        # Revert if quality drops
        verdict = gate.check(eval_result)
        if not verdict:
            log.warning("Structural experiment failed safety gate, reverting")
            # Revert flags
            reverted = {k: not v for k, v in flags.items()}
            lab.apply_flag_experiment(reverted)

        return eval_result, "structural_lab"

    elif action_type == "train_routing_models":
        min_mem = action.get("min_memories", 500)
        lab.checkpoint_state(
            trial_id=state.get("trial_counter", 0),
            notes="Pre-training checkpoint",
        )
        result = lab.train_routing_models(min_memories=min_mem)
        log.info("Training result: %s", result)
        eval_result = tower.eval_t0()
        return eval_result, "structural_lab"

    elif action_type == "distill_skillbank":
        teacher = action.get("teacher", "claude")
        categories = action.get("categories", ["routing"])
        lab.checkpoint_state(
            trial_id=state.get("trial_counter", 0),
            notes="Pre-distillation checkpoint",
        )
        result = lab.distill_skillbank(teacher=teacher, categories=categories)
        log.info("Distillation result: %s", result)
        eval_result = tower.eval_t0()
        return eval_result, "structural_lab"

    elif action_type == "reset_memories":
        keep_seen = action.get("keep_seen", True)
        keep_skills = action.get("keep_skills", True)
        result = lab.reset_and_reseed(
            keep_seen=keep_seen,
            keep_skills=keep_skills,
            trial_id=state.get("trial_counter", 0),
        )
        log.info("Reset result: %s", result)
        return None, "structural_lab"

    elif action_type == "deep_eval":
        tier = action.get("tier", 2)
        eval_result = tower.evaluate(tier=tier)
        return eval_result, "seeder"

    elif action_type == "rollback":
        to_cp = action.get("to_checkpoint", "production_best")
        if to_cp == "production_best":
            lab.restore_checkpoint()
        else:
            lab.restore_checkpoint(Path(to_cp))
        gate.reset_failures()
        eval_result = tower.eval_t0()
        return eval_result, "structural_lab"

    else:
        log.warning("Unknown action type: %s", action_type)
        return None, "unknown"


# ── Main Loop ────────────────────────────────────────────────────


def run_loop(
    max_trials: int | None = None,
    dry_run: bool = False,
    use_controller: bool = True,
) -> None:
    """Main optimization loop."""
    state = load_state()
    journal = ExperimentJournal()
    archive = ParetoArchive()
    gate = SafetyGate()
    tower = EvalTower(url=ORCHESTRATOR_URL)
    meta = MetaOptimizer()

    seeder = Seeder(url=ORCHESTRATOR_URL, dry_run=dry_run)
    swarm = NumericSwarm()
    forge = PromptForge(auto_commit=not dry_run)
    lab = StructuralLab(orchestrator_url=ORCHESTRATOR_URL)

    # Load species budget from state
    if "species_budget" in state:
        b = state["species_budget"]
        meta.budget = SpeciesBudget(**b)

    # Load TD errors from state
    if "td_errors" in state:
        seeder._td_errors = [(i, e) for i, e in enumerate(state["td_errors"])]

    trial_counter = state.get("trial_counter", 0)
    plot_paths: list[str] = []

    # Graceful shutdown handler
    shutdown_requested = False

    def signal_handler(signum, frame):
        nonlocal shutdown_requested
        log.info("Shutdown requested (signal %d)", signum)
        shutdown_requested = True

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    log.info("AutoPilot starting (trial=%d, dry_run=%s)", trial_counter, dry_run)

    while not shutdown_requested:
        if max_trials and trial_counter >= max_trials:
            log.info("Max trials reached (%d)", max_trials)
            break

        if state.get("paused"):
            log.info("AutoPilot paused, waiting...")
            time.sleep(10)
            state = load_state()
            continue

        # Check orchestrator health
        if not dry_run and not health_check(ORCHESTRATOR_URL, retries=2):
            log.error("Orchestrator unhealthy, waiting 30s...")
            time.sleep(30)
            continue

        # ── 1. Observe ───────────────────────────────────────────
        memory_count = seeder.get_memory_count() if not dry_run else 0
        converged = seeder.is_converged
        hv_slope = archive.hypervolume_slope(50)

        # ── 2. Reason ────────────────────────────────────────────
        if use_controller:
            prompt = CONTROLLER_PROMPT_TEMPLATE.format(
                pareto_summary=archive.summary_text(),
                journal_summary=journal.summary_text(20),
                seeder_status=json.dumps(seeder.convergence_status(), indent=2),
                species_effectiveness=json.dumps(
                    journal.species_effectiveness(), indent=2
                ),
                health_status="OK" if not dry_run else "dry_run",
                memory_count=memory_count,
                converged=converged,
                budget=json.dumps(meta.budget.as_dict(), indent=2),
                plot_paths="\n".join(f"  - {p}" for p in plot_paths) or "  (none yet)",
            )

            response, session_id = invoke_controller(
                prompt, state.get("session_id")
            )
            state["session_id"] = session_id
            action = extract_action(response)
        else:
            # Autonomous mode: species selection by budget
            species = meta.select_species()
            action = _auto_action(species, memory_count, converged, seeder)

        if not action:
            log.warning("No action proposed, defaulting to seed_batch")
            action = {"type": "seed_batch", "n_questions": 10}

        # ── 3. Act ───────────────────────────────────────────────
        log.info("Trial %d: %s", trial_counter, json.dumps(action))

        if dry_run:
            eval_result = EvalResult(
                tier=0, quality=2.5, speed=15.0, cost=0.3, reliability=0.95
            )
            species_name = action.get("type", "unknown").split("_")[0]
        else:
            eval_result, species_name = dispatch_action(
                action, seeder, swarm, forge, lab, tower, gate, archive, state
            )

        # ── 4. Evaluate ─────────────────────────────────────────
        if eval_result is None:
            trial_counter += 1
            state["trial_counter"] = trial_counter
            save_state(state)
            continue

        # Safety gate
        verdict = gate.check(eval_result)
        failure_analysis = gate.analyze_failure(eval_result, verdict)
        if not verdict:
            log.warning(
                "Safety violations: %s", "; ".join(verdict.violations)
            )
            if gate.should_rollback():
                log.error("Consecutive failure limit reached, rolling back")
                lab.restore_checkpoint()
                gate.reset_failures()

        # ── 5. Record ────────────────────────────────────────────
        pareto_status = archive.update(
            ParetoEntry(
                trial_id=trial_counter,
                objectives=eval_result.objectives,
                config_snapshot=action,
                species=species_name,
                timestamp=datetime.now(timezone.utc).isoformat(),
                eval_tier=eval_result.tier,
                memory_count=memory_count,
                reasoning=json.dumps(action),
            )
        )

        # If new Pareto entry, promote to T1
        if pareto_status == "frontier" and eval_result.tier == 0 and not dry_run:
            log.info("Pareto candidate! Running T1 evaluation...")
            t1_result = tower.eval_t1()
            t1_verdict = gate.check(t1_result)
            if t1_verdict:
                eval_result = t1_result
                # Update Pareto entry with T1 results
                archive._frontier[-1].objectives = t1_result.objectives
                archive._frontier[-1].eval_tier = 1

        # Git tag
        git_tag = ""
        if not dry_run:
            git_tag = f"autopilot/trial-{trial_counter}"
            _git_tag(git_tag, f"Trial {trial_counter}: {species_name}/{action.get('type', '')}")

        journal.record(
            JournalEntry(
                trial_id=trial_counter,
                timestamp=datetime.now(timezone.utc).isoformat(),
                species=species_name,
                action_type=action.get("type", ""),
                tier=eval_result.tier,
                quality=eval_result.quality,
                speed=eval_result.speed,
                cost=eval_result.cost,
                reliability=eval_result.reliability,
                pareto_status=pareto_status,
                git_tag=git_tag,
                config_snapshot=action,
                memory_count=memory_count,
                active_flags=[],
                failure_analysis=failure_analysis,
                eval_details={
                    "per_suite_quality": eval_result.per_suite_quality,
                    "routing_distribution": eval_result.routing_distribution,
                    "details": eval_result.details,
                },
            )
        )

        # ── 6. Meta-learn ───────────────────────────────────────
        if meta.should_rebalance(trial_counter):
            meta.rebalance(
                species_effectiveness=journal.species_effectiveness(window=50),
                hv_slope=hv_slope,
                memory_count=memory_count,
                is_converged=converged,
            )
            state["species_budget"] = meta.budget.as_dict()

        # Generate plots periodically
        if trial_counter % PLOT_INTERVAL == 0:
            td_errors = seeder.td_errors
            state["td_errors"] = [e for _, e in td_errors]
            paths = generate_all_plots(archive, journal, td_errors)
            plot_paths = [str(p) for p in paths]

        # Save state
        trial_counter += 1
        state["trial_counter"] = trial_counter
        archive.save(state)
        save_state(state)

        log.info(
            "Trial %d complete: q=%.3f s=%.1f → %s (HV=%.4f)",
            trial_counter - 1,
            eval_result.quality,
            eval_result.speed,
            pareto_status,
            archive.hypervolume(),
        )

    # Shutdown: checkpoint + save
    log.info("AutoPilot shutting down (trial=%d)", trial_counter)
    archive.save(state)
    save_state(state)
    if not dry_run:
        lab.checkpoint_state(trial_id=trial_counter, notes="Shutdown checkpoint")


def _auto_action(
    species: str,
    memory_count: int,
    converged: bool,
    seeder: Seeder,
) -> dict[str, Any]:
    """Generate an action without LLM controller (autonomous fallback)."""
    if memory_count < 500 or species == "seeder":
        return {"type": "seed_batch", "n_questions": 10}
    elif species == "numeric_swarm":
        return {"type": "numeric_trial", "surface": "memrl_retrieval"}
    elif species == "prompt_forge":
        return {"type": "prompt_mutation", "file": "frontdoor.md", "mutation": "targeted_fix"}
    elif species == "structural_lab":
        if converged:
            return {"type": "train_routing_models", "min_memories": 500}
        return {"type": "structural_experiment", "flags": {"think_harder": True}}
    return {"type": "seed_batch", "n_questions": 10}


def _git_tag(tag: str, message: str) -> None:
    """Create a git tag."""
    try:
        subprocess.run(
            ["git", "tag", "-a", tag, "-m", message],
            capture_output=True, timeout=10,
            cwd=str(ORCH_ROOT),
        )
    except Exception:
        pass


# ── CLI Commands ─────────────────────────────────────────────────


def cmd_start(args: argparse.Namespace) -> None:
    """Start the optimization loop."""
    # Process lock
    lock_file = open(LOCK_PATH, "w")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("ERROR: Another AutoPilot instance is running")
        sys.exit(1)

    run_loop(
        max_trials=args.max_trials,
        dry_run=args.dry_run,
        use_controller=not args.no_controller,
    )


def cmd_status(args: argparse.Namespace) -> None:
    """Show current status."""
    state = load_state()
    archive = ParetoArchive()
    journal = ExperimentJournal()

    print("AutoPilot Status")
    print("=" * 50)
    print(f"Trial counter: {state.get('trial_counter', 0)}")
    print(f"Paused: {state.get('paused', False)}")
    print(f"Session ID: {state.get('session_id', 'none')}")
    print()
    print(archive.summary_text())
    print()
    print(journal.summary_text(10))


def cmd_pause(args: argparse.Namespace) -> None:
    state = load_state()
    state["paused"] = True
    save_state(state)
    print("AutoPilot paused")


def cmd_resume(args: argparse.Namespace) -> None:
    state = load_state()
    state["paused"] = False
    save_state(state)
    print("AutoPilot resumed")


def cmd_report(args: argparse.Namespace) -> None:
    """Generate markdown report."""
    journal = ExperimentJournal()
    archive = ParetoArchive()

    print("# AutoPilot Optimization Report")
    print()
    print(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    print()
    print("## Summary")
    print(journal.summary_text())
    print()
    print("## Pareto Frontier")
    print(archive.summary_text())
    print()
    print("## Species Effectiveness")
    eff = journal.species_effectiveness()
    for sp, stats in eff.items():
        print(f"  {sp}: {stats['pareto']:.0f}/{stats['total']:.0f} ({stats['rate']:.1%})")


def cmd_plot(args: argparse.Namespace) -> None:
    """Generate plots."""
    archive = ParetoArchive()
    journal = ExperimentJournal()
    state = load_state()
    td_errors = [(i, e) for i, e in enumerate(state.get("td_errors", []))]
    paths = generate_all_plots(archive, journal, td_errors)
    for p in paths:
        print(f"  {p}")


def cmd_checkpoint(args: argparse.Namespace) -> None:
    lab = StructuralLab()
    state = load_state()
    cp = lab.checkpoint_state(
        trial_id=state.get("trial_counter", 0),
        mark_production_best=args.production_best,
        notes="Manual checkpoint",
    )
    print(f"Checkpoint created: {cp}")


def cmd_restore(args: argparse.Namespace) -> None:
    lab = StructuralLab()
    path = Path(args.checkpoint) if args.checkpoint else None
    result = lab.restore_checkpoint(path)
    print(f"Restore result: {result}")


# ── Entry Point ──────────────────────────────────────────────────


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                ORCH_ROOT / "logs" / "autopilot.log", mode="a"
            ),
        ],
    )

    parser = argparse.ArgumentParser(
        description="AutoPilot: Continuous recursive optimization"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # start
    p_start = subparsers.add_parser("start")
    p_start.add_argument("--dry-run", action="store_true")
    p_start.add_argument("--max-trials", type=int, default=None)
    p_start.add_argument("--no-controller", action="store_true",
                         help="Use autonomous species selection instead of Claude CLI")
    p_start.set_defaults(func=cmd_start)

    # status
    p_status = subparsers.add_parser("status")
    p_status.set_defaults(func=cmd_status)

    # pause / resume
    p_pause = subparsers.add_parser("pause")
    p_pause.set_defaults(func=cmd_pause)
    p_resume = subparsers.add_parser("resume")
    p_resume.set_defaults(func=cmd_resume)

    # report
    p_report = subparsers.add_parser("report")
    p_report.set_defaults(func=cmd_report)

    # plot
    p_plot = subparsers.add_parser("plot")
    p_plot.set_defaults(func=cmd_plot)

    # checkpoint
    p_cp = subparsers.add_parser("checkpoint")
    p_cp.add_argument("--production-best", action="store_true")
    p_cp.set_defaults(func=cmd_checkpoint)

    # restore
    p_restore = subparsers.add_parser("restore")
    p_restore.add_argument("--checkpoint", type=str, default=None)
    p_restore.set_defaults(func=cmd_restore)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
