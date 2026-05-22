"""Routing model bootstrap helpers for MemRL initialization."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class RoutingModelBundle:
    """Optional routing models loaded at API startup or MemRL lazy init."""

    graph_router: Any | None = None
    routing_graph: Any | None = None
    routing_classifier: Any | None = None
    frontdoor_verifier: Any | None = None

    def hybrid_router_kwargs(self) -> dict[str, Any]:
        """Return kwargs accepted by HybridRouter for optional models."""
        return {
            "graph_router": self.graph_router,
            "routing_classifier": self.routing_classifier,
            "frontdoor_verifier": self.frontdoor_verifier,
        }


def project_root() -> Path:
    """Return configured project root, falling back to the current directory."""
    try:
        from src.config import get_config

        return get_config().paths.project_root
    except Exception:
        return Path.cwd()


def build_routing_model_bundle(
    feature_flags: Any,
    embedder: Any,
    logger: logging.Logger,
    environ: Mapping[str, str] | None = None,
) -> RoutingModelBundle:
    """Load optional routing models and return them as one explicit bundle."""
    env = environ if environ is not None else os.environ
    graph_router, routing_graph = load_graph_router(feature_flags, embedder, logger)
    return RoutingModelBundle(
        graph_router=graph_router,
        routing_graph=routing_graph,
        routing_classifier=load_routing_classifier(feature_flags, logger, env),
        frontdoor_verifier=load_frontdoor_verifier(logger, env),
    )


def load_graph_router(
    feature_flags: Any,
    embedder: Any,
    logger: logging.Logger,
) -> tuple[Any | None, Any | None]:
    """Load the GNN cold-start routing signal when enabled."""
    if not getattr(feature_flags, "graph_router", False):
        return None, None

    try:
        from orchestration.repl_memory.graph_router_predictor import GraphRouterPredictor
        from orchestration.repl_memory.lightweight_gat import LightweightGAT
        from orchestration.repl_memory.routing_graph import BipartiteRoutingGraph

        routing_graph = BipartiteRoutingGraph()
        gat = LightweightGAT()
        weights_path = project_root() / "orchestration/repl_memory/graph_router_weights.npz"
        if weights_path.exists():
            gat.load(weights_path)
        graph_router = GraphRouterPredictor(routing_graph, gat, embedder)
        logger.info("GraphRouter initialized (GNN cold-start routing signal)")
        return graph_router, routing_graph
    except Exception as exc:
        logger.warning("GraphRouter init failed: %s", exc)
        return None, None


def load_routing_classifier(
    feature_flags: Any,
    logger: logging.Logger,
    environ: Mapping[str, str] | None = None,
) -> Any | None:
    """Load the MLP routing classifier fast-path when enabled."""
    if not getattr(feature_flags, "routing_classifier", False):
        return None

    env = environ if environ is not None else os.environ
    try:
        from orchestration.repl_memory.routing_classifier import (
            DEFAULT_WEIGHTS_PATH,
            RoutingClassifier,
        )

        weights_path = Path(
            env.get("ROUTING_CLASSIFIER_WEIGHTS", str(DEFAULT_WEIGHTS_PATH))
        )
        if not weights_path.exists():
            logger.warning(
                "routing_classifier flag is ON but weights not found at %s - fast-path disabled",
                weights_path,
            )
            return None

        routing_classifier = RoutingClassifier.load(weights_path)
        if routing_classifier is None:
            logger.warning(
                "RoutingClassifier.load returned None for %s - fast-path disabled",
                weights_path,
            )
            return None

        logger.info(
            "Routing classifier loaded: %d params, %d actions, weights=%s",
            routing_classifier.param_count,
            routing_classifier.n_actions,
            weights_path,
        )
        return routing_classifier
    except Exception as exc:
        logger.warning("Routing classifier init failed: %s", exc, exc_info=True)
        return None


def load_frontdoor_verifier(
    logger: logging.Logger,
    environ: Mapping[str, str] | None = None,
) -> Any | None:
    """Load the frontdoor-specialist verifier head when explicitly gated on."""
    env = environ if environ is not None else os.environ
    if env.get("ORCHESTRATOR_FRONTDOOR_VERIFIER_GATE", "0") != "1":
        return None

    try:
        from orchestration.repl_memory.verifier_head import (
            DEFAULT_WEIGHTS_PATH,
            VerifierHead,
        )

        weights_path = Path(env.get("FRONTDOOR_VERIFIER_WEIGHTS", str(DEFAULT_WEIGHTS_PATH)))
        if not weights_path.exists():
            logger.warning(
                "ORCHESTRATOR_FRONTDOOR_VERIFIER_GATE=1 but verifier weights not found at %s",
                weights_path,
            )
            return None

        frontdoor_verifier = VerifierHead.load(weights_path)
        if frontdoor_verifier is not None:
            logger.info(
                "Frontdoor verifier loaded: %d params, weights=%s",
                frontdoor_verifier.param_count,
                weights_path,
            )
        return frontdoor_verifier
    except Exception as exc:
        logger.warning("Frontdoor verifier init failed: %s", exc, exc_info=True)
        return None
