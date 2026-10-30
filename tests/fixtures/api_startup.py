"""Explicit lifespan seam for route tests that install their own fake runtime state."""
from contextlib import asynccontextmanager


def reset_route_rate_limit(app):
    """Start a route client's real token buckets fresh without changing its policy."""
    from src.api.rate_limit import RateLimitMiddleware

    middleware = app.middleware_stack
    while middleware is not None:
        if isinstance(middleware, RateLimitMiddleware):
            middleware._buckets.clear()
        middleware = getattr(middleware, 'app', None)


@asynccontextmanager
async def route_only_lifespan(app):
    """Keep HTTP routing/REPL behavior without host startup or shutdown hooks."""
    reset_route_rate_limit(app)
    try:
        yield
    finally:
        reset_route_rate_limit(app)
