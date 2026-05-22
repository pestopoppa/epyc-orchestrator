"""HTTP health-probe helper for orchestrator-managed servers."""

from __future__ import annotations

import time


def wait_for_health(port: int, timeout: int = 120, path: str = "/health") -> bool:
    """Poll http://localhost:<port><path> until it returns 200 or `timeout` elapses.

    Used by every server-launch path (llama-server, orchestrator API, docker
    services, document_formalizer, sd, whisper) to gate "started" on actual
    readiness. Default of 120 s matches the registry fallback for
    health.server_startup; every production caller passes an explicit timeout.
    """
    import urllib.request
    import urllib.error

    url = f"http://localhost:{port}{path}"
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, TimeoutError, ConnectionResetError, OSError):
            pass
        time.sleep(2)
    return False
