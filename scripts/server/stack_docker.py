"""Docker container management for orchestrator-rostered services.

Currently the orchestrator stack rosters NextPLAID (multi-vector retrieval),
SearXNG (metasearch), and Crawl4AI (browser-backed extraction) as Docker
services. Container-management helpers are kept here so `orchestrator_stack.py`
doesn't have to know docker CLI semantics directly; it re-imports the four
public helpers.
"""

from __future__ import annotations

import subprocess
from datetime import datetime

from scripts.server.stack_health import wait_for_health
from scripts.server.stack_state import ProcessInfo


def _docker_available() -> bool:
    """Check if docker CLI is available."""
    try:
        result = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            capture_output=True, text=True, timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def docker_container_running(name: str) -> bool:
    """Check if a named Docker container is running."""
    try:
        result = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", name],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() == "true"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def start_docker_container(service: dict) -> ProcessInfo | None:
    """Start a Docker service. Removes any existing container with the same name first."""
    name = service["name"]
    port = service["port"]
    container_port = service.get("container_port", 8080)

    # Remove existing container (stopped or running) with same name
    subprocess.run(
        ["docker", "rm", "-f", name],
        capture_output=True, timeout=10,
    )

    cmd = ["docker", "run", "-d", "--name", name]
    if shm_size := service.get("shm_size"):
        cmd.extend(["--shm-size", str(shm_size)])
    cmd.extend(["-p", f"{port}:{container_port}"])
    for key, value in sorted(service.get("env", {}).items()):
        cmd.extend(["-e", f"{key}={value}"])
    for vol in service.get("volumes", []):
        cmd.extend(["-v", vol])
    cmd.append(service["image"])
    cmd.extend(service.get("args", []))

    print(f"  Starting {name} on port {port}...")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        print(f"    [FAIL] docker run failed: {result.stderr.strip()[:200]}")
        return None

    container_id = result.stdout.strip()[:12]
    print(f"    Container: {container_id}")

    # Wait for health
    health_path = service.get("health_path", "/health")
    print("    Waiting for health...")
    if wait_for_health(port, timeout=60, path=health_path):
        print(f"    [OK] {name} ready ({service['description']})")
        # Use container_id as PID placeholder (Docker manages the actual process)
        return ProcessInfo(
            role=name,
            pid=-1,  # Docker-managed, not a host PID
            port=port,
            started_at=datetime.now().isoformat(),
            model_path=service.get("model", service["image"]),
            log_file=f"docker logs {name}",
        )
    else:
        print(f"    [FAIL] {name} health check timed out")
        # Show last few log lines for debugging
        logs = subprocess.run(
            ["docker", "logs", "--tail", "10", name],
            capture_output=True, text=True, timeout=5,
        )
        if logs.stdout:
            print(f"    Last logs: {logs.stdout.strip()[:300]}")
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=5)
        return None


def stop_docker_container(name: str) -> bool:
    """Stop and remove a named Docker container."""
    result = subprocess.run(
        ["docker", "rm", "-f", name],
        capture_output=True, text=True, timeout=15,
    )
    return result.returncode == 0
