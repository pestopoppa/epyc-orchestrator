"""Species 3 — StructuralLab: Feature flags, routing model lifecycle, checkpointing.

Manages the routing intelligence lifecycle:
  checkpoint → train → A/B test → enable → monitor → reset → reseed
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger("autopilot.structural_lab")

ORCH_ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT_DIR = ORCH_ROOT / "orchestration" / "autopilot_checkpoints"
MEMORY_DIR = ORCH_ROOT / "orchestration" / "repl_memory" / "sessions"
SKILLS_DIR = ORCH_ROOT / "orchestration" / "repl_memory"
PROMPTS_DIR = ORCH_ROOT / "orchestration" / "prompts"
CLASSIFIER_CONFIG = ORCH_ROOT / "orchestration" / "classifier_config.yaml"

# Files to checkpoint
CHECKPOINT_FILES = {
    "episodic.db": MEMORY_DIR / "episodic.db",
    "embeddings.faiss": MEMORY_DIR / "embeddings.faiss",
    "id_map.npy": MEMORY_DIR / "id_map.npy",
    "skills.db": SKILLS_DIR / "skills.db",
    "skill_embeddings.faiss": SKILLS_DIR / "skill_embeddings.faiss",
    "routing_classifier_weights.npz": SKILLS_DIR / "routing_classifier_weights.npz",
    "graph_router_weights.npz": ORCH_ROOT / "scripts" / "graph_router" / "graph_router_weights.npz",
}


@dataclass
class CheckpointMeta:
    timestamp: str
    trial_id: int = -1
    hypervolume: float = 0.0
    feature_flags: dict[str, bool] = field(default_factory=dict)
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    memory_count: int = 0
    is_production_best: bool = False
    notes: str = ""


class StructuralLab:
    """Species 3: Feature flag experiments + routing intelligence lifecycle."""

    def __init__(self, orchestrator_url: str = "http://localhost:8000"):
        self.url = orchestrator_url

    # ── checkpointing ────────────────────────────────────────────

    def checkpoint_state(
        self,
        trial_id: int = -1,
        hypervolume: float = 0.0,
        feature_flags: dict[str, bool] | None = None,
        config_snapshot: dict[str, Any] | None = None,
        notes: str = "",
        mark_production_best: bool = False,
    ) -> Path:
        """Snapshot all routing intelligence files to timestamped directory."""
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        cp_dir = CHECKPOINT_DIR / ts
        cp_dir.mkdir(parents=True, exist_ok=True)

        # Copy files
        copied = []
        for name, src in CHECKPOINT_FILES.items():
            if src.exists():
                dst = cp_dir / name
                shutil.copy2(src, dst)
                copied.append(name)
                log.info("Checkpointed %s", name)

        # Copy prompts
        prompts_cp = cp_dir / "prompts"
        if PROMPTS_DIR.exists():
            shutil.copytree(PROMPTS_DIR, prompts_cp, dirs_exist_ok=True)
            copied.append("prompts/")

        # Copy classifier config
        if CLASSIFIER_CONFIG.exists():
            shutil.copy2(CLASSIFIER_CONFIG, cp_dir / "classifier_config.yaml")
            copied.append("classifier_config.yaml")

        # Memory count
        memory_count = self._get_memory_count()

        # Write metadata
        meta = CheckpointMeta(
            timestamp=ts,
            trial_id=trial_id,
            hypervolume=hypervolume,
            feature_flags=feature_flags or {},
            config_snapshot=config_snapshot or {},
            memory_count=memory_count,
            is_production_best=mark_production_best,
            notes=notes,
        )
        (cp_dir / "checkpoint_meta.json").write_text(
            json.dumps(meta.__dict__, indent=2, default=str)
        )

        # If production best, update symlink
        if mark_production_best:
            best_link = CHECKPOINT_DIR / "production_best"
            if best_link.is_symlink() or best_link.exists():
                best_link.unlink()
            best_link.symlink_to(cp_dir)
            log.info("Marked as production best: %s", ts)

        log.info("Checkpoint %s: %d files copied, %d memories", ts, len(copied), memory_count)
        return cp_dir

    def restore_checkpoint(self, checkpoint_path: Path | None = None) -> dict[str, Any]:
        """Restore routing intelligence from a checkpoint.

        If no path given, restores from production_best.
        """
        if checkpoint_path is None:
            checkpoint_path = CHECKPOINT_DIR / "production_best"
            if not checkpoint_path.exists():
                return {"status": "error", "error": "No production_best checkpoint"}

        if checkpoint_path.is_symlink():
            checkpoint_path = checkpoint_path.resolve()

        if not checkpoint_path.exists():
            return {"status": "error", "error": f"Checkpoint not found: {checkpoint_path}"}

        restored = []
        for name, dst in CHECKPOINT_FILES.items():
            src = checkpoint_path / name
            if src.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                restored.append(name)

        # Restore prompts
        prompts_cp = checkpoint_path / "prompts"
        if prompts_cp.exists():
            shutil.copytree(prompts_cp, PROMPTS_DIR, dirs_exist_ok=True)
            restored.append("prompts/")

        # Restore classifier config
        cc = checkpoint_path / "classifier_config.yaml"
        if cc.exists():
            shutil.copy2(cc, CLASSIFIER_CONFIG)
            restored.append("classifier_config.yaml")

        log.info("Restored %d files from %s", len(restored), checkpoint_path.name)
        return {"status": "ok", "restored": restored, "from": str(checkpoint_path)}

    def list_checkpoints(self) -> list[dict[str, Any]]:
        """List available checkpoints with metadata."""
        if not CHECKPOINT_DIR.exists():
            return []
        result = []
        for d in sorted(CHECKPOINT_DIR.iterdir()):
            if not d.is_dir() or d.is_symlink():
                continue
            meta_file = d / "checkpoint_meta.json"
            if meta_file.exists():
                meta = json.loads(meta_file.read_text())
            else:
                meta = {"timestamp": d.name}
            meta["path"] = str(d)
            result.append(meta)
        return result

    # ── routing model training ───────────────────────────────────

    def train_routing_models(self, min_memories: int = 500) -> dict[str, Any]:
        """Train MLP routing classifier + GAT GraphRouter from episodic memories."""
        memory_count = self._get_memory_count()
        if memory_count < min_memories:
            return {
                "status": "skipped",
                "reason": f"Insufficient memories: {memory_count} < {min_memories}",
            }

        results: dict[str, Any] = {"memory_count": memory_count}

        # Train MLP classifier
        try:
            classifier_script = ORCH_ROOT / "scripts" / "graph_router" / "train_routing_classifier.py"
            if classifier_script.exists():
                proc = subprocess.run(
                    ["python", str(classifier_script)],
                    capture_output=True, text=True, timeout=120,
                    cwd=str(ORCH_ROOT),
                )
                results["classifier"] = {
                    "status": "ok" if proc.returncode == 0 else "error",
                    "stdout": proc.stdout[-500:],
                    "stderr": proc.stderr[-500:],
                }
            else:
                results["classifier"] = {"status": "script_not_found"}
        except Exception as e:
            results["classifier"] = {"status": "error", "error": str(e)}

        # Train GAT GraphRouter
        try:
            gat_script = ORCH_ROOT / "scripts" / "graph_router" / "train_graph_router.py"
            if gat_script.exists():
                proc = subprocess.run(
                    ["python", str(gat_script)],
                    capture_output=True, text=True, timeout=300,
                    cwd=str(ORCH_ROOT),
                )
                results["graph_router"] = {
                    "status": "ok" if proc.returncode == 0 else "error",
                    "stdout": proc.stdout[-500:],
                    "stderr": proc.stderr[-500:],
                }
            else:
                results["graph_router"] = {"status": "script_not_found"}
        except Exception as e:
            results["graph_router"] = {"status": "error", "error": str(e)}

        return results

    # ── skillbank distillation ───────────────────────────────────

    def distill_skillbank(
        self,
        teacher: str = "claude",
        categories: list[str] | None = None,
    ) -> dict[str, Any]:
        """Run SkillBank distillation pipeline."""
        try:
            import sys
            sys.path.insert(0, str(ORCH_ROOT / "orchestration" / "repl_memory" / "distillation"))
            from pipeline import DistillationPipeline

            pipeline = DistillationPipeline(
                teacher_model=teacher,
                categories=categories or ["routing", "escalation", "tool_selection"],
            )
            result = pipeline.run()
            return {"status": "ok", "result": result}
        except ImportError:
            log.warning("DistillationPipeline not available")
            return {"status": "not_available"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    # ── memory reset ─────────────────────────────────────────────

    def reset_and_reseed(
        self,
        keep_seen: bool = True,
        keep_skills: bool = True,
        checkpoint_first: bool = True,
        trial_id: int = -1,
    ) -> dict[str, Any]:
        """Checkpoint → reset → ready for reseeding.

        NEVER resets without checkpointing first (unless checkpoint_first=False
        and caller has already checkpointed).
        """
        result: dict[str, Any] = {}

        if checkpoint_first:
            cp = self.checkpoint_state(
                trial_id=trial_id,
                notes="Pre-reset checkpoint",
            )
            result["checkpoint"] = str(cp)

        # Run reset script
        reset_script = ORCH_ROOT / "scripts" / "session" / "reset_episodic_memory.sh"
        if not reset_script.exists():
            return {"status": "error", "error": "Reset script not found"}

        cmd = ["bash", str(reset_script)]
        if keep_seen:
            cmd.append("--keep-seen")
        if keep_skills:
            cmd.append("--keep-skills")

        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=60,
                cwd=str(ORCH_ROOT),
            )
            result["reset"] = {
                "status": "ok" if proc.returncode == 0 else "error",
                "stdout": proc.stdout[-500:],
                "stderr": proc.stderr[-500:],
            }
        except Exception as e:
            result["reset"] = {"status": "error", "error": str(e)}

        return result

    # ── feature flag experiments ─────────────────────────────────

    def propose_flag_experiment(
        self,
        flags: dict[str, bool],
    ) -> dict[str, Any]:
        """Propose a feature flag experiment.

        Validates flag dependencies before applying.
        """
        import sys
        sys.path.insert(0, str(ORCH_ROOT / "src"))

        try:
            from features import Features
            test_features = Features(**flags)
            errors = test_features.validate()
            if errors:
                return {
                    "status": "invalid",
                    "errors": errors,
                    "proposed_flags": flags,
                }
        except Exception as e:
            return {"status": "error", "error": str(e)}

        return {"status": "valid", "flags": flags}

    def apply_flag_experiment(self, flags: dict[str, bool]) -> dict[str, Any]:
        """Apply feature flags via POST /config."""
        import httpx
        try:
            resp = httpx.post(f"{self.url}/config", json=flags, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"status": "error", "error": str(e)}

    # ── helpers ──────────────────────────────────────────────────

    def _get_memory_count(self) -> int:
        try:
            import sys
            sys.path.insert(0, str(ORCH_ROOT / "orchestration" / "repl_memory"))
            from episodic_store import EpisodicStore
            store = EpisodicStore()
            count = store.count("routing")
            store.close()
            return count
        except Exception:
            return 0

    def summary(self) -> dict[str, Any]:
        return {
            "memory_count": self._get_memory_count(),
            "checkpoints": len(self.list_checkpoints()),
            "has_production_best": (CHECKPOINT_DIR / "production_best").exists(),
        }
