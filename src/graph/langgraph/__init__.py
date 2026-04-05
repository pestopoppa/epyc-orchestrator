"""LangGraph migration module — Phase 1 hybrid bridge.

Provides a LangGraph StateGraph equivalent to the pydantic_graph orchestration
graph. During the hybrid phase, both backends coexist:

- ``pydantic_graph`` remains the production default
- ``langgraph`` backend is activated via ``langgraph_bridge`` feature flag

The migration preserves all node logic (``_execute_turn()`` and helpers are
framework-agnostic) — only the graph routing/transition layer changes.
"""
