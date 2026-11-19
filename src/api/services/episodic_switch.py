"""M-12k memory-off switch: one predicate for every episodic LOOKUP on the chat routing path.

Lookups gated (each reads the HybridRouter / EpisodicStore):
  * initial-role routing      -- chat_pipeline.routing_decision.select_initial_route
  * execution-mode selection  -- chat_routing._select_mode (HybridRouter.route_with_mode)
  * try-cheap-first Q gate    -- chat.py phase B/C (retriever.retrieve_for_routing)
  * plan-review Q gate        -- chat_review.py phase B (retriever.retrieve_for_routing)

NOT gated: episodic WRITES (task-start logging, Q-scoring) and the model-initiated REPL ``route`` tool
(a tool call the model chooses to make, not a pipeline lookup). The switch has two layers, both read LIVE:

  * ``features().episodic_routing_lookups`` (default True; POST /config toggles it with no restart);
  * ``ChatRequest.episodic_lookups`` (None = follow the flag; False = off for this request only), so a
    memory-on and a memory-off arm can interleave on one server.

A request can only turn lookups OFF; ``True`` never overrides a flag that is off.
"""

from __future__ import annotations

from typing import Any

from src.features import features


def episodic_lookups_enabled(request: Any = None) -> bool:
    """True = episodic lookups run (today's behaviour). False = skip them."""
    if getattr(request, "episodic_lookups", None) is False:
        return False
    return bool(features().episodic_routing_lookups)
