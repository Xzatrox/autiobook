"""manage a local llama-server process for LLM inference."""

import shutil
import signal
import subprocess
import time
from dataclasses import dataclass, field
from typing import Optional

import requests


@dataclass
class LlamaServerConfig:
    """configuration for a local llama-server instance."""

    model: str  # path to main GGUF model
    draft_model: Optional[str] = None  # path to draft model for speculative decoding
    port: int = 8080
    ctx_size: int = 16384
    gpu_layers: int = 99
    draft_gpu_layers: int = 99
    draft_ctx_size: Optional[int] = None
    draft_max: int = 16
    draft_min: int = 1
    draft_p_min: float = 0.5
    extra_args: list[str] = field(default_factory=list)


class LlamaServer:
    """manages a llama-server subprocess lifecycle."""

    def __init__(self, config: LlamaServerConfig):
        self.config = config
        self._process: Optional[subprocess.Popen] = None

    @property
    def api_base(self) -> str:
        return f"http://localhost:{self.config.port}/v1"

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(self, timeout: float = 120.0) -> None:
        """start llama-server and wait until it's ready."""
        if self.running:
            return

        binary = shutil.which("llama-server")
        if not binary:
            raise FileNotFoundError(
                "llama-server not found. install with: brew install llama.cpp"
            )

        cmd = [
            binary,
            "-m", self.config.model,
            "--port", str(self.config.port),
            "-ngl", str(self.config.gpu_layers),
            "-c", str(self.config.ctx_size),
            "--reasoning-format", "none",
        ]

        if self.config.draft_model:
            cmd += [
                "-md", self.config.draft_model,
                "-ngld", str(self.config.draft_gpu_layers),
                "-cd", str(self.config.draft_ctx_size or self.config.ctx_size),
                "--draft-max", str(self.config.draft_max),
                "--draft-min", str(self.config.draft_min),
                "--draft-p-min", str(self.config.draft_p_min),
            ]

        cmd += self.config.extra_args

        print(f"llm-server: starting on port {self.config.port}...")
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # wait for health endpoint
        url = f"http://localhost:{self.config.port}/health"
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                raise RuntimeError(
                    f"llama-server exited with code {self._process.returncode}"
                )
            try:
                r = requests.get(url, timeout=2)
                if r.status_code == 200:
                    print("llm-server: ready")
                    return
            except (requests.ConnectionError, requests.ReadTimeout):
                pass
            time.sleep(1)

        self.stop()
        raise TimeoutError(f"llama-server did not become ready within {timeout}s")

    def stop(self) -> None:
        """stop the llama-server process."""
        if self._process is None:
            return

        if self._process.poll() is None:
            self._process.send_signal(signal.SIGTERM)
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()

        print("llm-server: stopped")
        self._process = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()
