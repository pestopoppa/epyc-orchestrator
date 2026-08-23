#!/usr/bin/env python3
"""Offline training pipeline for GraphRouter GAT weights.

Steps:
1. Load EpisodicStore snapshot
2. BipartiteRoutingGraph.sync_from_episodic_store() — cluster + edge compute
3. Export node features + adjacency + edge labels
4. Train GAT: edge masking (20% held out) -> predict -> BCE loss -> SGD
5. Save graph_router_weights.npz
6. Validate: held-out edge prediction accuracy

Usage:
    python3 scripts/graph_router/train_graph_router.py [--epochs 100] [--lr 0.001]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import yaml

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_graph_router")

DEFAULT_WEIGHTS_PATH = PROJECT_ROOT / "orchestration/repl_memory/graph_router_weights.npz"
DEFAULT_STACK_PRIORS_PATH = PROJECT_ROOT / "orchestration/derived/stack_priors.yaml"
DEFAULT_REGISTRY_PATH = PROJECT_ROOT / "orchestration/model_registry.yaml"

# NIB2-57a: a role with no measured t/s prior must not enter the router fleet
# as tps=0.0 silently — the GAT learner would read it as "slowest role", the
# same fabricated-implies-measured shape that let four unmeasured priors drive
# the router for weeks. Warn once per role; the fleet still degrades to 0.0
# (callers reach these paths only when generated priors are unavailable).
_warned_unmeasured_fleet_tps: set[str] = set()


def _warn_unmeasured_fleet_tps(role_id: str) -> None:
    """Warn (once per role) that the fleet tps for ``role_id`` is unmeasured."""
    if role_id in _warned_unmeasured_fleet_tps:
        return
    _warned_unmeasured_fleet_tps.add(role_id)
    logger.warning(
        "GraphRouter fleet role %r has NO measured t/s prior (stack priors and "
        "descriptors both lack one); fleet tps = 0.0, which the learner will "
        "read as the slowest role. Placement features for this role are "
        "UNMEASURED — measure it or accept the degraded feature.",
        role_id,
    )


def _coerce_float(value, default: float) -> float:
    if isinstance(value, (int, float)):
        parsed = float(value)
        return parsed if parsed > 0 else default
    return default


def _role_description(role: str, record: dict) -> str:
    model = record.get("model") if isinstance(record.get("model"), dict) else {}
    serving = record.get("serving") if isinstance(record.get("serving"), dict) else {}
    priors = record.get("priors") if isinstance(record.get("priors"), dict) else {}
    pieces = [
        str(record.get("display_name") or record.get("model_id") or role),
        f"role={role}",
    ]
    server_role = serving.get("server_role")
    if server_role:
        pieces.append(f"server={server_role}")
    modalities = model.get("modalities")
    if isinstance(modalities, list) and modalities:
        pieces.append("modalities=" + ",".join(str(item) for item in modalities))
    throughput = priors.get("throughput_tps")
    if isinstance(throughput, (int, float)):
        pieces.append(f"{float(throughput):g} t/s")
    return "; ".join(pieces)


def _descriptor_role_description(role: str, descriptor: dict) -> str:
    pieces = [
        str(descriptor.get("display_name") or descriptor.get("model_id") or role),
        f"role={role}",
    ]
    arch = descriptor.get("arch")
    if arch:
        pieces.append(f"arch={arch}")
    speed = descriptor.get("speed")
    if isinstance(speed, dict):
        tps = speed.get("solo_96t_tps") or speed.get("optimized_tps")
        if isinstance(tps, (int, float)):
            pieces.append(f"{float(tps):g} t/s")
    return "; ".join(pieces)


def _registry_role_ids(registry_path: Path) -> set[str]:
    try:
        loaded = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return set()
    roles = loaded.get("roles")
    if not isinstance(roles, dict):
        return set()
    return {role for role in roles if isinstance(role, str)}


def _compile_descriptors(registry_path: Path, active_roles: set[str] | None = None) -> dict:
    from src.registry.model_descriptors import compile_model_descriptors

    return compile_model_descriptors(
        lean_registry_path=registry_path,
        research_registry_path=None,
        active_roles=active_roles,
        allow_incomplete=True,
    )


def _descriptor_model_fleet(registry_path: Path = DEFAULT_REGISTRY_PATH) -> list[dict]:
    try:
        from src.roles import Role

        descriptors = _compile_descriptors(registry_path)
        models = descriptors.get("models")
        if models == []:
            descriptors = _compile_descriptors(
                registry_path,
                active_roles=_registry_role_ids(registry_path),
            )
    except Exception as exc:
        logger.warning("Could not derive degraded GraphRouter fleet from descriptors: %s", exc)
        return []

    models = descriptors.get("models")
    if not isinstance(models, list):
        return []

    fleet: list[dict] = []
    seen: set[str] = set()
    for descriptor in models:
        if not isinstance(descriptor, dict):
            continue
        bindings = descriptor.get("role_bindings")
        roles = bindings.get("roles") if isinstance(bindings, dict) else None
        if not isinstance(roles, list):
            continue
        serving = descriptor.get("serving")
        speed = descriptor.get("speed")
        ports = serving.get("ports") if isinstance(serving, dict) else None
        port = next((item for item in ports or () if isinstance(item, int)), 0)
        tps = 0.0
        if isinstance(speed, dict):
            tps = _coerce_float(speed.get("solo_96t_tps") or speed.get("optimized_tps"), 0.0)
        for role in roles:
            if not isinstance(role, str):
                continue
            canonical = Role.from_string(role)
            role_id = str(canonical) if canonical is not None else role
            if role_id in seen:
                continue
            seen.add(role_id)
            if tps <= 0.0:
                _warn_unmeasured_fleet_tps(role_id)
            fleet.append(
                {
                    "role_id": role_id,
                    "description": _descriptor_role_description(role_id, descriptor),
                    "port": port,
                    "tps": tps,
                    "tier": "UNKNOWN",
                    "gb": _coerce_float(descriptor.get("mem_gb"), 0.0),
                }
            )
    return fleet


def load_model_fleet(
    stack_priors_path: Path = DEFAULT_STACK_PRIORS_PATH,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
) -> list[dict]:
    """Load live GraphRouter model nodes from the generated stack-priors contract."""
    try:
        from src.registry.stack_priors import (
            load_stack_priors_artifact,
            live_stack_role_records,
            stack_prior_primary_port,
            stack_prior_serving,
        )

        artifact = load_stack_priors_artifact(stack_priors_path)
        live_roles = live_stack_role_records(stack_priors_path)
    except Exception as exc:
        logger.warning(
            "Using descriptor-derived GraphRouter model fleet; stack priors unavailable: %s", exc
        )
        return _descriptor_model_fleet(registry_path)

    if not isinstance(artifact, dict):
        logger.warning(
            "Using descriptor-derived GraphRouter model fleet; stack priors unavailable: cannot load contract"
        )
        return _descriptor_model_fleet(registry_path)

    roles = artifact.get("roles")
    if roles is not None and not isinstance(roles, dict):
        logger.warning(
            "Using descriptor-derived GraphRouter model fleet; stack priors roles field is invalid"
        )
        return _descriptor_model_fleet(registry_path)

    if not isinstance(roles, dict):
        logger.warning(
            "Using descriptor-derived GraphRouter model fleet; stack priors roles field is invalid"
        )
        return _descriptor_model_fleet(registry_path)

    if not isinstance(live_roles, dict):
        logger.warning(
            "Using descriptor-derived GraphRouter model fleet; stack priors roles field is invalid"
        )
        return _descriptor_model_fleet(registry_path)

    fleet: list[dict] = []
    for role, record in sorted(live_roles.items()):
        if not isinstance(role, str) or not isinstance(record, dict):
            continue

        serving = stack_prior_serving(record)
        priors = record.get("priors") if isinstance(record.get("priors"), dict) else {}
        model = record.get("model") if isinstance(record.get("model"), dict) else {}
        endpoint_port = stack_prior_primary_port(serving) or 0
        fleet_tps = _coerce_float(priors.get("throughput_tps"), 0.0)
        if fleet_tps <= 0.0:
            _warn_unmeasured_fleet_tps(role)
        fleet.append(
            {
                "role_id": role,
                "description": _role_description(role, record),
                "port": endpoint_port,
                "tps": fleet_tps,
                "tier": str(serving.get("tier") or "unknown").upper(),
                "gb": _coerce_float(model.get("mem_gb"), 0.0),
            }
        )

    if not fleet:
        logger.warning(
            "Using descriptor-derived GraphRouter model fleet; no live stack roles found"
        )
        return _descriptor_model_fleet(registry_path)
    return fleet


def populate_llm_roles(graph, embedder, model_fleet: list[dict] | None = None):
    """Populate LLMRole nodes from model fleet definition."""
    fleet = model_fleet if model_fleet is not None else load_model_fleet()
    for model in fleet:
        emb = embedder.embed_text(model["description"])
        graph.add_llm_role(
            role_id=model["role_id"],
            description=model["description"],
            embedding=emb,
            port=model["port"],
            tps=model["tps"],
            tier=model["tier"],
            gb=model["gb"],
        )
    logger.info("Populated %d LLM role nodes", len(fleet))


def build_training_data(graph):
    """Export graph structure for GAT training.

    Returns:
        node_features, edge_index, targets, qc_indices, llm_indices
    """
    node_feats = graph.get_node_features()
    edge_idx = graph.get_edge_index()
    perf_edges = graph.get_performance_edges()

    qc_ids = node_feats["query_cluster_ids"]
    llm_ids = node_feats["llm_role_ids"]

    if not qc_ids or not llm_ids:
        logger.error("Graph has no query clusters or LLM roles — cannot train")
        return None

    qc_idx_map = {id: i for i, id in enumerate(qc_ids)}
    llm_idx_map = {id: i for i, id in enumerate(llm_ids)}

    # Build target matrix (N_qc, N_llm) with success_rate as soft labels
    targets = np.zeros((len(qc_ids), len(llm_ids)), dtype=np.float32)
    for e in perf_edges:
        qi = qc_idx_map.get(e["to_cluster"])
        li = llm_idx_map.get(e["from_role"])
        if qi is not None and li is not None:
            targets[qi, li] = e["success_rate"]

    return node_feats, edge_idx, targets, np.arange(len(qc_ids)), np.arange(len(llm_ids))


def train(
    epochs: int = 100,
    lr: float = 0.001,
    val_split: float = 0.2,
    patience: int = 20,
    output_path: Path = DEFAULT_WEIGHTS_PATH,
    min_memories: int = 500,
):
    """Run offline GAT training pipeline."""
    from orchestration.repl_memory.embedder import TaskEmbedder
    from orchestration.repl_memory.episodic_store import EpisodicStore
    from orchestration.repl_memory.lightweight_gat import LightweightGAT
    from orchestration.repl_memory.routing_graph import BipartiteRoutingGraph

    logger.info("=== GraphRouter Training Pipeline ===")
    t0 = time.time()

    # 1. Load episodic store
    store = EpisodicStore()
    mem_count = store.count()
    logger.info("Episodic store: %d memories", mem_count)

    if mem_count < min_memories:
        logger.warning(
            "Insufficient memories (%d < %d). Skipping training.",
            mem_count,
            min_memories,
        )
        return False

    # 2. Initialize components
    embedder = TaskEmbedder()
    graph = BipartiteRoutingGraph()

    # 3. Populate LLM roles
    populate_llm_roles(graph, embedder)

    # 4. Sync graph from episodic store
    sync_stats = graph.sync_from_episodic_store(store, embedder)
    logger.info("Graph sync: %s", sync_stats)

    # 5. Build training data
    result = build_training_data(graph)
    if result is None:
        return False
    node_feats, edge_idx, targets, qc_indices, llm_indices = result
    logger.info(
        "Training data: %d query clusters, %d LLM roles, target shape %s",
        len(qc_indices),
        len(llm_indices),
        targets.shape,
    )

    # 6. Create validation mask (stratified by role)
    rng = np.random.default_rng(42)
    val_mask = np.zeros_like(targets, dtype=bool)
    for li in range(targets.shape[1]):
        nonzero = np.where(targets[:, li] > 0)[0]
        if len(nonzero) >= 2:
            n_val = max(1, int(len(nonzero) * val_split))
            val_idx = rng.choice(nonzero, n_val, replace=False)
            val_mask[val_idx, li] = True

    train_mask = ~val_mask
    logger.info(
        "Train edges: %d, Val edges: %d",
        int(train_mask.sum()),
        int(val_mask.sum()),
    )

    # 7. Train GAT
    gat = LightweightGAT()
    logger.info("GAT parameters: %d", gat.param_count)

    gat_edge_index = {
        "belongs_to": edge_idx["belongs_to"],
        "performance_on": edge_idx["performance_on"],
    }

    best_val_loss = float("inf")
    best_weights = None
    no_improve = 0

    for epoch in range(epochs):
        # Forward pass
        out = gat.forward(node_feats, gat_edge_index, training=True)
        preds = gat.predict_edges(out["query_cluster"], out["llm_role"])

        # Losses
        train_loss = gat.compute_loss(preds, targets, mask=train_mask.astype(np.float32))
        val_loss = gat.compute_loss(preds, targets, mask=val_mask.astype(np.float32))

        # Cosine LR decay
        current_lr = lr * 0.5 * (1 + np.cos(np.pi * epoch / epochs))

        # Compute gradients and update
        gradients = gat.get_gradients(
            node_feats,
            gat_edge_index,
            targets,
            qc_indices,
            llm_indices,
        )
        gat.update_weights(gradients, current_lr)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.copy() for k, v in gat._weights.items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 10 == 0 or epoch == epochs - 1:
            logger.info(
                "Epoch %3d/%d  train_loss=%.4f  val_loss=%.4f  lr=%.6f  patience=%d/%d",
                epoch,
                epochs,
                train_loss,
                val_loss,
                current_lr,
                no_improve,
                patience,
            )

        if no_improve >= patience:
            logger.info("Early stopping at epoch %d", epoch)
            break

    # 8. Restore best weights and save
    if best_weights:
        gat._weights = best_weights

    gat.save(output_path)

    # 9. Final validation
    out = gat.forward(node_feats, gat_edge_index)
    preds = gat.predict_edges(out["query_cluster"], out["llm_role"])
    final_loss = gat.compute_loss(preds, targets)

    # Edge prediction accuracy (threshold 0.5)
    binary_preds = (preds > 0.5).astype(np.float32)
    binary_targets = (targets > 0.5).astype(np.float32)
    accuracy = float((binary_preds == binary_targets).mean())

    elapsed = time.time() - t0
    logger.info(
        "=== Training complete in %.1fs ===\n"
        "  Final loss: %.4f\n"
        "  Edge accuracy: %.2f%%\n"
        "  Weights: %s\n"
        "  Graph: %s",
        elapsed,
        final_loss,
        accuracy * 100,
        output_path,
        graph.get_stats(),
    )
    return True


def main():
    parser = argparse.ArgumentParser(description="Train GraphRouter GAT weights")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--patience", type=int, default=20, help="Early stop patience")
    parser.add_argument(
        "--min-memories", type=int, default=500, help="Min episodic memories required"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_WEIGHTS_PATH),
        help="Output path for weights",
    )
    args = parser.parse_args()

    success = train(
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        output_path=Path(args.output),
        min_memories=args.min_memories,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
