"""Auto-generate progress visualization plots for AutoPilot.

6 plots, overwritten in-place to orchestration/autopilot_plots/.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

log = logging.getLogger("autopilot.plots")

PLOTS_DIR = Path(__file__).resolve().parents[2] / "orchestration" / "autopilot_plots"


def ensure_matplotlib():
    """Import matplotlib with Agg backend."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_hypervolume_trend(
    history: list[tuple[int, float]], output_dir: Path | None = None
) -> Path:
    """Line chart: hypervolume over trial number."""
    plt = ensure_matplotlib()
    output_dir = output_dir or PLOTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    if history:
        trials, hvs = zip(*history)
        ax.plot(trials, hvs, "b-", linewidth=2)
        ax.fill_between(trials, hvs, alpha=0.1)
    ax.set_xlabel("Trial")
    ax.set_ylabel("Hypervolume")
    ax.set_title("AutoPilot: Hypervolume Trend")
    ax.grid(True, alpha=0.3)

    path = output_dir / "hypervolume_trend.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", path)
    return path


def plot_pareto_frontier_2d(
    frontier: list[dict], dominated: list[dict], output_dir: Path | None = None
) -> Path:
    """Scatter: Quality vs Speed, colored by species."""
    plt = ensure_matplotlib()
    output_dir = output_dir or PLOTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    species_colors = {
        "seeder": "#2196F3",
        "numeric_swarm": "#FF9800",
        "prompt_forge": "#4CAF50",
        "structural_lab": "#9C27B0",
    }

    fig, ax = plt.subplots(figsize=(10, 7))

    # Dominated points in grey
    if dominated:
        d_q = [p["objectives"][0] for p in dominated]
        d_s = [p["objectives"][1] for p in dominated]
        ax.scatter(d_s, d_q, c="lightgrey", s=20, alpha=0.4, label="Dominated")

    # Frontier points colored by species
    for species, color in species_colors.items():
        pts = [p for p in frontier if p.get("species") == species]
        if pts:
            q = [p["objectives"][0] for p in pts]
            s = [p["objectives"][1] for p in pts]
            ax.scatter(s, q, c=color, s=80, label=species, edgecolors="black", linewidth=0.5)

    ax.set_xlabel("Speed (tokens/s)")
    ax.set_ylabel("Quality (0-3)")
    ax.set_title("Pareto Frontier: Quality vs Speed")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    path = output_dir / "pareto_frontier_2d.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_species_effectiveness(
    effectiveness: dict[str, dict[str, float]], output_dir: Path | None = None
) -> Path:
    """Bar chart: Pareto improvement rate per species."""
    plt = ensure_matplotlib()
    output_dir = output_dir or PLOTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    species_colors = {
        "seeder": "#2196F3",
        "numeric_swarm": "#FF9800",
        "prompt_forge": "#4CAF50",
        "structural_lab": "#9C27B0",
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    species = list(effectiveness.keys())
    rates = [effectiveness[s]["rate"] for s in species]
    colors = [species_colors.get(s, "#999999") for s in species]

    bars = ax.bar(species, rates, color=colors, edgecolor="black", linewidth=0.5)
    for bar, sp in zip(bars, species):
        total = int(effectiveness[sp]["total"])
        pareto = int(effectiveness[sp]["pareto"])
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{pareto}/{total}",
            ha="center", va="bottom", fontsize=9,
        )

    ax.set_ylabel("Pareto Improvement Rate")
    ax.set_title("Species Effectiveness")
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis="y")

    path = output_dir / "species_effectiveness.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_per_suite_quality(
    trial_suite_quality: list[dict], output_dir: Path | None = None
) -> Path:
    """Heatmap: quality by suite x trial."""
    plt = ensure_matplotlib()
    import numpy as np
    output_dir = output_dir or PLOTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if not trial_suite_quality:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, "No data yet", ha="center", va="center", fontsize=14)
        path = output_dir / "per_suite_quality.png"
        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        return path

    # Collect all suites
    all_suites = sorted(
        set(s for entry in trial_suite_quality for s in entry.get("per_suite", {}))
    )
    trial_ids = [entry["trial_id"] for entry in trial_suite_quality]

    # Build matrix
    matrix = np.full((len(all_suites), len(trial_ids)), np.nan)
    for j, entry in enumerate(trial_suite_quality):
        for i, suite in enumerate(all_suites):
            if suite in entry.get("per_suite", {}):
                matrix[i, j] = entry["per_suite"][suite]

    fig, ax = plt.subplots(figsize=(max(10, len(trial_ids) * 0.3), max(5, len(all_suites) * 0.4)))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=3)
    ax.set_yticks(range(len(all_suites)))
    ax.set_yticklabels(all_suites, fontsize=8)
    if len(trial_ids) <= 50:
        ax.set_xticks(range(len(trial_ids)))
        ax.set_xticklabels(trial_ids, fontsize=7, rotation=45)
    ax.set_xlabel("Trial")
    ax.set_title("Per-Suite Quality Over Trials")
    fig.colorbar(im, ax=ax, label="Quality (0-3)")

    path = output_dir / "per_suite_quality.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_memory_convergence(
    td_errors: list[tuple[int, float]], output_dir: Path | None = None
) -> Path:
    """Line chart: Q-value TD error magnitude over seeding batches."""
    plt = ensure_matplotlib()
    output_dir = output_dir or PLOTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    if td_errors:
        batches, errors = zip(*td_errors)
        ax.plot(batches, errors, "r-", linewidth=1.5, alpha=0.7)
        # Running average
        window = min(10, len(errors))
        if window > 1:
            running_avg = []
            for i in range(len(errors)):
                start = max(0, i - window + 1)
                running_avg.append(sum(errors[start:i + 1]) / (i - start + 1))
            ax.plot(batches, running_avg, "r-", linewidth=2.5, label=f"MA({window})")
            ax.legend()

    ax.set_xlabel("Seeding Batch")
    ax.set_ylabel("|TD Error|")
    ax.set_title("Memory Convergence: Q-Value TD Error Magnitude")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.05, color="green", linestyle="--", alpha=0.5, label="Convergence threshold")
    ax.legend()

    path = output_dir / "memory_convergence.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_trial_timeline(
    entries: list[dict], output_dir: Path | None = None
) -> Path:
    """Gantt-like timeline: species colored bars, height = quality delta."""
    plt = ensure_matplotlib()
    output_dir = output_dir or PLOTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    species_colors = {
        "seeder": "#2196F3",
        "numeric_swarm": "#FF9800",
        "prompt_forge": "#4CAF50",
        "structural_lab": "#9C27B0",
    }

    fig, ax = plt.subplots(figsize=(12, 5))
    if entries:
        trial_ids = [e["trial_id"] for e in entries]
        deltas = [e.get("quality_delta", 0.0) for e in entries]
        colors = [species_colors.get(e.get("species", ""), "#999999") for e in entries]

        ax.bar(trial_ids, deltas, color=colors, edgecolor="black", linewidth=0.3)
        ax.axhline(y=0, color="black", linewidth=0.5)

        # Legend
        from matplotlib.patches import Patch
        legend_patches = [
            Patch(facecolor=c, label=s) for s, c in species_colors.items()
        ]
        ax.legend(handles=legend_patches, loc="upper right")

    ax.set_xlabel("Trial")
    ax.set_ylabel("Quality Delta vs Baseline")
    ax.set_title("Trial Timeline")
    ax.grid(True, alpha=0.3, axis="y")

    path = output_dir / "trial_timeline.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_all_plots(
    archive,  # ParetoArchive
    journal,  # ExperimentJournal
    td_errors: list[tuple[int, float]] | None = None,
    output_dir: Path | None = None,
) -> list[Path]:
    """Generate all 6 plots from current state."""
    output_dir = output_dir or PLOTS_DIR
    paths = []

    try:
        # 1. Hypervolume trend
        paths.append(plot_hypervolume_trend(archive.hypervolume_trend(), output_dir))

        # 2. Pareto frontier 2D
        frontier = [e.to_dict() for e in archive.frontier()]
        dominated = [
            e.to_dict()
            for e in archive._all_entries
            if e not in archive._frontier
        ]
        paths.append(plot_pareto_frontier_2d(frontier, dominated, output_dir))

        # 3. Species effectiveness
        eff = journal.species_effectiveness()
        paths.append(plot_species_effectiveness(eff, output_dir))

        # 4. Per-suite quality
        suite_data = []
        for e in journal.all_entries():
            if e.eval_details.get("per_suite_quality"):
                suite_data.append({
                    "trial_id": e.trial_id,
                    "per_suite": e.eval_details["per_suite_quality"],
                })
        paths.append(plot_per_suite_quality(suite_data, output_dir))

        # 5. Memory convergence
        paths.append(plot_memory_convergence(td_errors or [], output_dir))

        # 6. Trial timeline
        entries = journal.all_entries()
        baseline_q = entries[0].quality if entries else 0
        timeline_data = [
            {
                "trial_id": e.trial_id,
                "species": e.species,
                "quality_delta": e.quality - baseline_q,
            }
            for e in entries
        ]
        paths.append(plot_trial_timeline(timeline_data, output_dir))

    except Exception as e:
        log.error("Error generating plots: %s", e)

    return paths
