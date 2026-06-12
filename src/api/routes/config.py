"""Runtime configuration endpoint for hot-reloading feature flags."""

from __future__ import annotations

import logging
import os

from fastapi import APIRouter, HTTPException, Request, Depends

from src.features import (
    Features,
    feature_sources,
    set_features,
    write_runtime_flag_overrides,
)
from src.api.dependencies import dep_features

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/config")
async def update_config(
    request: Request,
    current: Features = Depends(dep_features),
):
    """Update feature flags at runtime without restarting the API.

    Restricted to localhost only to prevent unauthorized remote configuration
    changes. Accepts a JSON body with feature flag names as keys and booleans
    as values. Only known feature flags are applied; unknown keys are ignored.

    Returns:
        Updated feature flag summary.
    """
    client_ip = request.client.host if request.client else "unknown"
    if client_ip not in ("127.0.0.1", "::1", "localhost"):
        raise HTTPException(
            status_code=403,
            detail="Config changes only allowed from localhost",
        )
    body = await request.json()
    current_summary = current.summary()
    overrides = {k: bool(v) for k, v in body.items() if k in current_summary}
    if overrides:
        write_runtime_flag_overrides(overrides, set_by=f"api:{client_ip}")
    new = Features(**{**current_summary, **overrides})
    set_features(new)
    return {
        "status": "ok",
        "features": new.summary(),
        "sources": feature_sources(),
    }


@router.get("/config/attest")
async def attest_config(current: Features = Depends(dep_features)):
    """Return this worker's feature-flag state and value sources."""
    return {
        "pid": os.getpid(),
        "flags": current.summary(),
        "sources": feature_sources(),
    }
