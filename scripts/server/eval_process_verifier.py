"""Private CPU verifier session: explicit opt-in, one captured PID, no fleet edits."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import socket
import subprocess
import time

import httpx


@dataclass(frozen=True)
class ProcessVerifierConfig:
    model: Path
    server: Path
    library_dir: Path
    log_path: Path
    model_identity: str = "ThinkPRM-1.5B"
    max_tokens: int = 1024
    context_tokens: int = 4096
    timeout_s: float = 120.0
    seed: int = 42
    generation_drained: bool = False


class NativeProcessVerifier:
    def __init__(self, config: ProcessVerifierConfig):
        self.config = config
        self.model_identity = config.model_identity
        self.proc = None
        self.client = None
        self.log = None
        self.lifecycle = {"started": False, "drained": True}

    def __enter__(self):
        c = self.config
        if not c.generation_drained:
            raise ValueError("explicit generation_drained boundary required before verifier load")
        if not c.model.is_file() or not c.server.is_file():
            raise ValueError("verifier model and server must exist")
        if not 1 <= c.max_tokens <= 4096 or not 512 <= c.context_tokens <= 16384:
            raise ValueError("verifier token budget outside bounded limits")
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        self.url = f"http://127.0.0.1:{port}"
        c.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = c.log_path.open("a")
        env = dict(os.environ, LD_LIBRARY_PATH=str(c.library_dir), OMP_NUM_THREADS="1")
        # Clear inherited backend knobs before admitting the explicit CPU recipe.
        env = {k: v for k, v in env.items() if not k.startswith(("GGML_", "HIP_", "ROCR_"))}
        try:
            self.proc = subprocess.Popen(
                [
                    str(c.server),
                    "-m",
                    str(c.model),
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "-ngl",
                    "0",
                    "-t",
                    "1",
                    "-tb",
                    "1",
                    "-c",
                    str(c.context_tokens),
                    "-b",
                    "128",
                    "-ub",
                    "128",
                    "-np",
                    "1",
                    "--no-warmup",
                    "--jinja",
                ],
                env=env,
                stdout=self.log,
                stderr=subprocess.STDOUT,
            )
            self.lifecycle["drained"] = False
            self.client = httpx.Client(base_url=self.url, timeout=c.timeout_s, trust_env=False)
            deadline = time.monotonic() + min(c.timeout_s, 90)
            while time.monotonic() < deadline:
                if self.proc.poll() is not None:
                    raise RuntimeError("private verifier server exited during load")
                try:
                    if self.client.get("/health").status_code == 200:
                        self.lifecycle.update(started=True, pid=self.proc.pid, url=self.url)
                        return self
                except httpx.HTTPError:
                    pass
                time.sleep(0.1)
            raise TimeoutError("private verifier load timed out")
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *_):
        if self.client:
            self.client.close()
        if self.proc:
            if self.proc.poll() is None:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait(timeout=5)
            self.lifecycle.update(
                drained=self.proc.poll() is not None, returncode=self.proc.returncode
            )
        if self.log:
            self.log.close()

    def post(self, endpoint, payload):
        response = self.client.post(endpoint, json=payload)
        response.raise_for_status()
        return response.json()

    def verify(self, problem: str, solution: str) -> dict:
        from scripts.autopilot.process_verification import (
            binary_score,
            parse_step_verdicts,
            verification_prompt,
        )

        rendered = (
            self.post(
                "/apply-template",
                {
                    "messages": [
                        {"role": "user", "content": verification_prompt(problem, solution)}
                    ],
                    "add_generation_prompt": True,
                },
            )["prompt"]
            + "\nLet's verify step by step:"
        )
        cot = self.post(
            "/completion",
            {
                "prompt": rendered,
                "n_predict": self.config.max_tokens,
                "temperature": 0,
                "seed": self.config.seed,
                "cache_prompt": False,
            },
        )
        text = cot["content"]
        labels = {
            label: self.post("/tokenize", {"content": label, "add_special": False})["tokens"]
            for label in (" Yes", " No")
        }
        if any(len(ids) != 1 for ids in labels.values()) or labels[" Yes"] == labels[" No"]:
            probability = {
                "score": None,
                "status": "labels_not_distinct_single_tokens",
                "raw_label_tokenization": labels,
            }
        else:
            ids = {label.strip(): tokens[0] for label, tokens in labels.items()}
            # No grammar or truncation: equal bias changes neither label odds.
            # temperature=-1 requests pre-sampling softmax in the native server;
            # both observed logprobs are required, even when a label is unlikely.
            raw = self.post(
                "/completion",
                {
                    "prompt": rendered + text + "\nIs the solution correct?",
                    "n_predict": 1,
                    "temperature": -1,
                    "n_probs": 64,
                    "post_sampling_probs": False,
                    "return_tokens": True,
                    "logit_bias": [[tid, 100.0] for tid in ids.values()],
                    "seed": self.config.seed,
                    "cache_prompt": False,
                },
            )
            probability = binary_score(raw, ids)
            # Independently observe the original distribution as a native
            # control; an equal additive bias must preserve the two-label odds.
            unbiased_raw = self.post(
                "/completion",
                {
                    "prompt": rendered + text + "\nIs the solution correct?",
                    "n_predict": 1,
                    "temperature": -1,
                    "n_probs": 64,
                    "post_sampling_probs": False,
                    "return_tokens": True,
                    "seed": self.config.seed,
                    "cache_prompt": False,
                },
            )
            unbiased = binary_score(unbiased_raw, ids)
            unbiased["common_logit_bias"] = 0.0
            probability["unbiased_control"] = unbiased
            probability["equal_bias_ratio_validated"] = (
                probability["score"] is not None
                and unbiased["score"] is not None
                and abs(probability["score"] - unbiased["score"]) < 1e-5
            )
            if probability["score"] is not None and unbiased["score"] is not None:
                if not probability["equal_bias_ratio_validated"]:
                    raise ValueError("native equal-bias control changed conditional label odds")
            probability["raw_label_tokenization"] = labels
        return {
            "verification_cot": text,
            "steps": parse_step_verdicts(text),
            "probability": probability,
            "generation_settings": {
                "max_tokens": self.config.max_tokens,
                "seed": self.config.seed,
                "temperature": 0,
                "context_tokens": self.config.context_tokens,
            },
            "native_cot_response": cot,
        }
