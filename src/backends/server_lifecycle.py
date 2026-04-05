"""Server lifecycle abstraction for inference backends (B6).

Abstracts the server spawn/manage/shutdown lifecycle into a backend-agnostic
interface, enabling support for vLLM, TGI, and other inference servers
alongside the current llama-server.

Cherry-picked from OpenGauss ``ManagedWorkflowSpec → ManagedContext →
LaunchPlan`` pipeline.

Design: does NOT modify ``orchestrator_stack.py``. Creates new abstractions
alongside it. The stack launcher can adopt these interfaces incrementally.
"""

from __future__ import annotations

import enum
import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Server capabilities and status
# ---------------------------------------------------------------------------


class ServerType(str, enum.Enum):
    """Supported inference server types."""

    LLAMA_SERVER = "llama-server"
    VLLM = "vllm"
    TGI = "tgi"


@dataclass(frozen=True)
class ServerCapabilities:
    """What an inference server supports."""

    streaming: bool = True
    prefix_caching: bool = False
    slot_management: bool = False  # KV slot save/restore
    grammar_constrained: bool = False
    json_schema: bool = False
    multimodal: bool = False
    max_batch_size: int = 1
    supported_quant_types: tuple[str, ...] = ()


@dataclass
class ServerStatus:
    """Current status of a running server instance."""

    healthy: bool = False
    pid: int | None = None
    port: int = 0
    url: str = ""
    model_loaded: str = ""
    memory_used_gb: float = 0.0
    slots_total: int = 0
    slots_active: int = 0
    uptime_seconds: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ServerConfig:
    """Configuration for launching a server instance."""

    server_type: ServerType
    model_path: str
    port: int
    host: str = "127.0.0.1"
    num_slots: int = 1
    context_length: int = 32768
    threads: int = 48
    gpu_layers: int = -1  # -1 = all
    extra_args: dict[str, Any] = field(default_factory=dict)

    # NUMA affinity (llama-server specific)
    numa_node: int | None = None
    cpu_cores: str = ""  # e.g. "0-47"

    # KV cache config
    kv_type_k: str = ""  # e.g. "q4_0"
    kv_type_v: str = ""  # e.g. "f16"
    kv_hadamard: bool = False


# ---------------------------------------------------------------------------
# ServerLifecycle Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class ServerLifecycle(Protocol):
    """Protocol for managing inference server lifecycle.

    Implementations handle server-specific launch, health checking,
    and shutdown logic.
    """

    @property
    def server_type(self) -> ServerType:
        """The type of server this lifecycle manages."""
        ...

    @property
    def capabilities(self) -> ServerCapabilities:
        """Static capabilities of this server type."""
        ...

    def build_launch_command(self, config: ServerConfig) -> list[str]:
        """Build the command line to launch the server.

        Args:
            config: Server configuration.

        Returns:
            List of command-line arguments.
        """
        ...

    def health_check(self, url: str) -> ServerStatus:
        """Check if a running server is healthy.

        Args:
            url: Base URL of the server (e.g., http://localhost:8080).

        Returns:
            ServerStatus with health information.
        """
        ...

    def get_status(self, url: str) -> ServerStatus:
        """Get detailed status of a running server.

        Args:
            url: Base URL of the server.

        Returns:
            ServerStatus with detailed metrics.
        """
        ...


# ---------------------------------------------------------------------------
# LlamaServerLifecycle — concrete implementation
# ---------------------------------------------------------------------------


class LlamaServerLifecycle:
    """Lifecycle manager for llama-server (llama.cpp).

    Builds launch commands matching the patterns in orchestrator_stack.py
    without modifying it.
    """

    @property
    def server_type(self) -> ServerType:
        return ServerType.LLAMA_SERVER

    @property
    def capabilities(self) -> ServerCapabilities:
        return ServerCapabilities(
            streaming=True,
            prefix_caching=True,
            slot_management=True,
            grammar_constrained=True,
            json_schema=True,
            multimodal=False,
            max_batch_size=4,
            supported_quant_types=(
                "q4_0", "q4_1", "q5_0", "q5_1", "q8_0",
                "q4_k_m", "q5_k_m", "q6_k", "iq4_xs",
            ),
        )

    def build_launch_command(self, config: ServerConfig) -> list[str]:
        """Build llama-server launch command."""
        cmd = ["llama-server"]
        cmd.extend(["-m", config.model_path])
        cmd.extend(["--port", str(config.port)])
        cmd.extend(["--host", config.host])
        cmd.extend(["-np", str(config.num_slots)])
        cmd.extend(["-c", str(config.context_length)])
        cmd.extend(["-t", str(config.threads)])

        if config.gpu_layers >= 0:
            cmd.extend(["-ngl", str(config.gpu_layers)])

        if config.kv_type_k:
            cmd.extend(["-ctk", config.kv_type_k])
        if config.kv_type_v:
            cmd.extend(["-ctv", config.kv_type_v])
        if config.kv_hadamard:
            cmd.append("--kv-hadamard")

        # Extra args (e.g., --slot-save-path, -ub, -ps)
        for key, val in config.extra_args.items():
            if val is True:
                cmd.append(f"--{key}")
            elif val is not False and val is not None:
                cmd.extend([f"--{key}", str(val)])

        return cmd

    def health_check(self, url: str) -> ServerStatus:
        """Check llama-server health via /health endpoint."""
        import httpx

        status = ServerStatus(url=url)
        try:
            resp = httpx.get(f"{url}/health", timeout=5.0)
            data = resp.json() if resp.status_code == 200 else {}
            status.healthy = resp.status_code == 200
            status.slots_total = data.get("slots_idle", 0) + data.get("slots_processing", 0)
            status.slots_active = data.get("slots_processing", 0)
        except Exception as exc:
            logger.debug("Health check failed for %s: %s", url, exc)
            status.healthy = False
        return status

    def get_status(self, url: str) -> ServerStatus:
        """Get detailed llama-server status via /slots endpoint."""
        import httpx

        status = self.health_check(url)
        if not status.healthy:
            return status

        try:
            resp = httpx.get(f"{url}/slots", timeout=5.0)
            if resp.status_code == 200:
                slots = resp.json()
                status.slots_total = len(slots)
                status.slots_active = sum(
                    1 for s in slots if s.get("state", 0) != 0
                )
                if slots:
                    status.model_loaded = slots[0].get("model", "")
                status.extra["slots"] = slots
        except Exception as exc:
            logger.debug("Status check failed for %s: %s", url, exc)

        return status


# ---------------------------------------------------------------------------
# VLLMLifecycle — stub implementation
# ---------------------------------------------------------------------------


class VLLMLifecycle:
    """Lifecycle manager for vLLM (stub).

    To be implemented when vLLM backend support is needed.
    """

    @property
    def server_type(self) -> ServerType:
        return ServerType.VLLM

    @property
    def capabilities(self) -> ServerCapabilities:
        return ServerCapabilities(
            streaming=True,
            prefix_caching=True,
            slot_management=False,
            grammar_constrained=False,
            json_schema=True,
            multimodal=True,
            max_batch_size=256,
            supported_quant_types=("awq", "gptq", "squeezellm"),
        )

    def build_launch_command(self, config: ServerConfig) -> list[str]:
        cmd = ["python3", "-m", "vllm.entrypoints.openai.api_server"]
        cmd.extend(["--model", config.model_path])
        cmd.extend(["--port", str(config.port)])
        cmd.extend(["--host", config.host])
        cmd.extend(["--max-model-len", str(config.context_length)])
        if config.gpu_layers >= 0:
            cmd.extend(["--tensor-parallel-size", "1"])
        return cmd

    def health_check(self, url: str) -> ServerStatus:
        import httpx

        status = ServerStatus(url=url)
        try:
            resp = httpx.get(f"{url}/health", timeout=5.0)
            status.healthy = resp.status_code == 200
        except Exception:
            status.healthy = False
        return status

    def get_status(self, url: str) -> ServerStatus:
        return self.health_check(url)


# ---------------------------------------------------------------------------
# TGILifecycle — stub implementation
# ---------------------------------------------------------------------------


class TGILifecycle:
    """Lifecycle manager for Text Generation Inference (stub).

    To be implemented when TGI backend support is needed.
    """

    @property
    def server_type(self) -> ServerType:
        return ServerType.TGI

    @property
    def capabilities(self) -> ServerCapabilities:
        return ServerCapabilities(
            streaming=True,
            prefix_caching=True,
            slot_management=False,
            grammar_constrained=True,
            json_schema=True,
            multimodal=False,
            max_batch_size=128,
            supported_quant_types=("awq", "gptq", "eetq", "bitsandbytes"),
        )

    def build_launch_command(self, config: ServerConfig) -> list[str]:
        cmd = ["text-generation-launcher"]
        cmd.extend(["--model-id", config.model_path])
        cmd.extend(["--port", str(config.port)])
        cmd.extend(["--hostname", config.host])
        cmd.extend(["--max-input-length", str(config.context_length)])
        return cmd

    def health_check(self, url: str) -> ServerStatus:
        import httpx

        status = ServerStatus(url=url)
        try:
            resp = httpx.get(f"{url}/health", timeout=5.0)
            status.healthy = resp.status_code == 200
        except Exception:
            status.healthy = False
        return status

    def get_status(self, url: str) -> ServerStatus:
        return self.health_check(url)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_lifecycle(server_type: ServerType | str) -> ServerLifecycle:
    """Get the lifecycle manager for a given server type.

    Args:
        server_type: Server type string or enum.

    Returns:
        Appropriate ServerLifecycle implementation.

    Raises:
        ValueError: If server type is not supported.
    """
    if isinstance(server_type, str):
        server_type = ServerType(server_type)

    _registry: dict[ServerType, type] = {
        ServerType.LLAMA_SERVER: LlamaServerLifecycle,
        ServerType.VLLM: VLLMLifecycle,
        ServerType.TGI: TGILifecycle,
    }

    cls = _registry.get(server_type)
    if cls is None:
        raise ValueError(f"Unsupported server type: {server_type}")
    return cls()
