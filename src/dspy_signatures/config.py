"""DSPy LM configuration for local llama-server endpoints (AP-18/AP-25).

Points DSPy at local OpenAI-compatible endpoints served by llama-server.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def configure_local_lm(
    base_url: str = "http://localhost:8082/v1",
    model: str = "local",
    max_tokens: int = 2048,
):
    """Configure DSPy to use local llama-server as the LM backend.

    Args:
        base_url: OpenAI-compatible /v1 endpoint URL.
        model: Model name (arbitrary for local; used in DSPy logging).
        max_tokens: Max generation tokens.

    Returns:
        The configured dspy.LM instance.
    """
    import dspy

    lm = dspy.LM(
        model=f"openai/{model}",
        api_base=base_url,
        api_key="not-needed",
        max_tokens=max_tokens,
    )
    dspy.configure(lm=lm)
    log.info("DSPy configured: %s → %s", model, base_url)
    return lm


def configure_rlm(
    main_lm_url: str = "http://localhost:8082/v1",
    sub_lm_url: str = "http://localhost:8080/v1",
    main_model: str = "coder",
    sub_model: str = "frontdoor",
    max_tokens: int = 4096,
):
    """Configure DSPy for RLM pattern with main LM + sub_lm (AP-25).

    The RLM pattern uses a capable main LM (coder) for code generation
    and a cheap sub_lm (frontdoor) for semantic queries and metadata
    exploration.

    Args:
        main_lm_url: Endpoint for the main (coder) model.
        sub_lm_url: Endpoint for the sub (frontdoor) model.
        main_model: Name for the main LM.
        sub_model: Name for the sub LM.
        max_tokens: Max generation tokens for main LM.

    Returns:
        Tuple of (main_lm, sub_lm) dspy.LM instances.
    """
    import dspy

    main_lm = dspy.LM(
        model=f"openai/{main_model}",
        api_base=main_lm_url,
        api_key="not-needed",
        max_tokens=max_tokens,
    )
    sub_lm = dspy.LM(
        model=f"openai/{sub_model}",
        api_base=sub_lm_url,
        api_key="not-needed",
        max_tokens=1024,
    )
    # Set main as default; sub_lm used explicitly via dspy.context(lm=sub_lm)
    dspy.configure(lm=main_lm)
    log.info("DSPy RLM configured: main=%s@%s, sub=%s@%s",
             main_model, main_lm_url, sub_model, sub_lm_url)
    return main_lm, sub_lm


def test_connection(base_url: str = "http://localhost:8082/v1") -> bool:
    """Health check: verify DSPy can reach the local LM endpoint.

    Returns True if the endpoint responds, False otherwise.
    """
    import httpx

    try:
        resp = httpx.get(f"{base_url}/models", timeout=5.0)
        return resp.status_code == 200
    except Exception as e:
        log.warning("LM endpoint unreachable at %s: %s", base_url, e)
        return False
