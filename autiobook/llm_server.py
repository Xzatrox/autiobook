"""manage a local llama-server process for LLM inference."""

import os
import shutil
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import requests


@dataclass
class LlamaServerConfig:
    """configuration for a local llama-server instance."""

    model: str  # path to main GGUF model
    draft_model: Optional[str] = None  # path to draft model for speculative decoding
    port: int = 8080
    ctx_size: int = 8192
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
            # check common install locations
            candidates = [
                r"C:\Apps\llama-rocm-7.2.1\llama-b8407-windows-rocm-7.2.1-gfx110X-gfx115X-gfx120X-x64\llama-server.exe",
                r"C:\Apps\llama-bin-win-hip-radeon-x64\llama-server.exe",
                r"C:\Apps\llama-bin-win-vulkan-x64\llama-server.exe",
            ]
            for candidate in candidates:
                if Path(candidate).exists():
                    binary = candidate
                    break
        if not binary:
            raise FileNotFoundError(
                "llama-server not found. add its directory to PATH."
            )

        cmd = [
            binary,
            "-m", self.config.model,
            "--port", str(self.config.port),
            "-ngl", str(self.config.gpu_layers),
            "-c", str(self.config.ctx_size),
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

        env = dict(os.environ)
        env["HIP_VISIBLE_DEVICES"] = "0"
        env["ROCR_VISIBLE_DEVICES"] = "0"

        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
        )

        # print llama-server output in background
        def _log_output():
            for line in self._process.stdout:
                decoded = line.decode(errors="replace").rstrip()
                if any(k in decoded for k in ("GPU", "VRAM", "ROCm", "error", "warning", "loaded", "ready")):
                    print(f"  [llama] {decoded}")
        threading.Thread(target=_log_output, daemon=True).start()

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


@dataclass
class TransformersServerConfig:
    """configuration for a local transformers-based LLM server."""
    model: str  # HuggingFace model id or local path
    port: int = 8080
    max_new_tokens: int = 4096


class TransformersServer:
    """OpenAI-compatible LLM server using transformers + ROCm torch."""

    def __init__(self, config: TransformersServerConfig):
        self.config = config
        self._thread: Optional[threading.Thread] = None
        self._server = None
        self._ready = threading.Event()

    @property
    def api_base(self) -> str:
        return f"http://localhost:{self.config.port}/v1"

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print(f"llm-server: starting transformers server on port {self.config.port}...")
        if not self._ready.wait(timeout=600):
            raise TimeoutError("TransformersServer did not become ready within 600s")
        print("llm-server: ready")

    def _run(self) -> None:
        import torch
        import uvicorn
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"llm-server: loading {self.config.model}...")
        tokenizer = AutoTokenizer.from_pretrained(self.config.model)
        llm_device = "cuda:0"
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model,
            torch_dtype=torch.float16,
        )
        model = model.to(llm_device)
        model.eval()
        device_actual = next(model.parameters()).device
        print(f"llm-server: model loaded on {device_actual}")
        if "cpu" in str(device_actual):
            raise RuntimeError("model failed to load on GPU - OOM or unsupported device")

        app = FastAPI()
        max_new_tokens = self.config.max_new_tokens

        @app.get("/health")
        def health():
            return {"status": "ok"}

        @app.post("/v1/chat/completions")
        def chat_completions(body: dict):
            messages = body.get("messages", [])
            req_max = body.get("max_tokens", max_new_tokens)
            print(f"llm-server: generating (max_tokens={req_max}, msgs={len(messages)})...")

            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            print(f"llm-server: input tokens={inputs['input_ids'].shape[1]}, device={inputs['input_ids'].device}")

            with torch.inference_mode():
                import time as _time
                t0 = _time.monotonic()
                out = model.generate(
                    **inputs,
                    max_new_tokens=min(req_max, max_new_tokens),
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
                elapsed = _time.monotonic() - t0

            new_tokens = out[0][inputs["input_ids"].shape[1]:]
            content = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            print(f"llm-server: generated {len(new_tokens)} tokens in {elapsed:.1f}s ({len(new_tokens)/elapsed:.1f} tok/s)")

            return JSONResponse({
                "choices": [{"message": {"role": "assistant", "content": content}}],
                "model": self.config.model,
            })

        ready_event = self._ready

        @app.on_event("startup")
        async def on_startup():
            ready_event.set()

        config = uvicorn.Config(app, host="127.0.0.1", port=self.config.port, log_level="error")
        self._server = uvicorn.Server(config)
        self._server.run()

    def stop(self) -> None:
        if self._server:
            self._server.should_exit = True
        print("llm-server: stopped")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()
