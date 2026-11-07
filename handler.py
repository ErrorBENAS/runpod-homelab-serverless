"""RunPod Serverless worker: vLLM OpenAI server + thin async proxy handler.

How it works
------------
1. A prestart hook launches `vllm serve` on 127.0.0.1:8000 and waits for
   /health. RunPod holds queued jobs until the hook finishes, so the model is
   loaded once per worker, never per request.
2. `handler()` forwards each job to the local vLLM server and streams the
   result back. Because the backend speaks the OpenAI API, the endpoint works
   with RunPod's built-in OpenAI route:

       https://api.runpod.ai/v2/<ENDPOINT_ID>/openai/v1

   which is what Open WebUI (or the homelab proxy in homelab/runpod-proxy)
   points at.

Accepted job["input"] shapes
----------------------------
- RunPod OpenAI passthrough (sent by the platform on /openai/v1/*):
      {"openai_route": "/v1/chat/completions", "openai_input": {...}}
- Simple chat (for /run and /runsync callers):
      {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 256}
- Simple completion:
      {"prompt": "Once upon a time", "max_tokens": 64}
  Any other top-level keys (temperature, top_p, stop, ...) are passed through
  as OpenAI sampling parameters. "sampling_params": {...} is also accepted.

Local testing (no GPU needed)
-----------------------------
    WORKER_BACKEND=echo python handler.py --rp_serve_api --rp_api_concurrency 1 --rp_api_port 8080
    curl -s localhost:8080/runsync -H "Content-Type: application/json" -d @test_input.json

Use the local API rather than --test_input: in runpod 2.1 the --test_input
path does not iterate async-generator handlers (it prints the generator
object). Port 8080 because vLLM itself uses 8000. With a GPU and vLLM
installed, drop WORKER_BACKEND to run the real thing.
"""

from __future__ import annotations

import collections
import logging
import os
import shlex
import subprocess
import sys
import threading
import time
import urllib.request
from typing import Any, AsyncGenerator

import aiohttp
import runpod

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("worker")

# ---------------------------------------------------------------------------
# Configuration (all via endpoint environment variables)
# ---------------------------------------------------------------------------
# Default model: Qwen3 14B AWQ fits a 24 GB GPU with a 32k context. The homelab
# already runs the Qwen3 family in Ollama (qwen3:14b, qwen3:30b-a3b-instruct),
# so prompts and system messages carry over unchanged. For the bigger sibling
# on a 48 GB GPU use MODEL_NAME=Qwen/Qwen3-30B-A3B-Instruct-2507-FP8.
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3-14B-AWQ")
# The id Open WebUI shows in its model picker.
SERVED_MODEL_NAME = os.getenv("SERVED_MODEL_NAME", "runpod-qwen3-14b")
MAX_MODEL_LEN = os.getenv("MAX_MODEL_LEN", "32768")
GPU_MEMORY_UTILIZATION = os.getenv("GPU_MEMORY_UTILIZATION", "0.92")
# Anything else vLLM accepts, e.g. "--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser hermes"
VLLM_EXTRA_ARGS = os.getenv("VLLM_EXTRA_ARGS", "")

WORKER_BACKEND = os.getenv("WORKER_BACKEND", "vllm")  # "vllm" | "echo" (local plumbing tests)
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "16"))
STARTUP_TIMEOUT = int(os.getenv("VLLM_STARTUP_TIMEOUT", "1800"))
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "600"))

VLLM_HOST = "127.0.0.1"
VLLM_PORT = os.getenv("VLLM_PORT", "8000")
VLLM_URL = f"http://{VLLM_HOST}:{VLLM_PORT}"

# RunPod mounts network volumes and its model cache under /runpod-volume.
VOLUME_ROOT = "/runpod-volume"

vllm_process: subprocess.Popen | None = None
backend_ready = threading.Event()
startup_error: str | None = None
_tail: collections.deque[str] = collections.deque(maxlen=200)  # last vLLM log lines


# ---------------------------------------------------------------------------
# Backend startup (prestart hook)
# ---------------------------------------------------------------------------
def _configure_cache() -> None:
    """Keep Hugging Face downloads on the network volume / model cache if present.

    Cold starts then skip the multi-GB download. Falls back to the container's
    own disk when no volume is attached (or it is read-only and lacks the model).
    """
    hf_home = os.path.join(VOLUME_ROOT, "huggingface-cache")
    model_dir = os.path.join(hf_home, "hub", "models--" + MODEL_NAME.replace("/", "--"))
    if os.path.isdir(VOLUME_ROOT) and (os.access(VOLUME_ROOT, os.W_OK) or os.path.isdir(model_dir)):
        os.environ.setdefault("HF_HOME", hf_home)
        os.environ.setdefault("VLLM_CACHE_ROOT", os.path.join(VOLUME_ROOT, "vllm-cache"))
        if not os.access(VOLUME_ROOT, os.W_OK):
            os.environ.setdefault("HF_HUB_OFFLINE", "1")  # read-only model cache: don't try to write
    log.info("HF_HOME=%s", os.environ.get("HF_HOME", "(container default)"))


def _vllm_argv() -> list[str]:
    argv = [
        "vllm", "serve", MODEL_NAME,
        "--host", VLLM_HOST,
        "--port", VLLM_PORT,
        "--served-model-name", SERVED_MODEL_NAME,
        "--max-model-len", MAX_MODEL_LEN,
        "--gpu-memory-utilization", GPU_MEMORY_UTILIZATION,
        # (Request logging is off by default in current vLLM; the old
        # --disable-log-requests flag was removed and now aborts startup.)
    ]
    return argv + shlex.split(VLLM_EXTRA_ARGS)


def _pump(proc: subprocess.Popen) -> None:
    for line in proc.stdout:  # type: ignore[union-attr]
        sys.stdout.write(line)
        _tail.append(line)


def start_backend() -> None:
    """Launch vLLM and block until it answers /health (or fails)."""
    global vllm_process, startup_error
    if WORKER_BACKEND == "echo":
        log.info("WORKER_BACKEND=echo: skipping vLLM (local plumbing test)")
        backend_ready.set()
        return

    _configure_cache()
    argv = _vllm_argv()
    log.info("Starting: %s", " ".join(argv))
    vllm_process = subprocess.Popen(
        argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        errors="replace", env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    threading.Thread(target=_pump, args=(vllm_process,), daemon=True).start()

    deadline = time.monotonic() + STARTUP_TIMEOUT
    while time.monotonic() < deadline:
        if vllm_process.poll() is not None:
            # Keep the worker alive and report the cause on every job instead of
            # crash-looping (and burning GPU time) on e.g. an OOM or a bad flag.
            startup_error = (
                f"vLLM exited during startup (code {vllm_process.returncode}). "
                "Last log lines:\n" + "".join(list(_tail)[-25:])
            )
            log.error(startup_error)
            backend_ready.set()
            return
        try:
            with urllib.request.urlopen(f"{VLLM_URL}/health", timeout=5) as r:
                if r.status == 200:
                    log.info("vLLM ready: %s as '%s'", MODEL_NAME, SERVED_MODEL_NAME)
                    backend_ready.set()
                    return
        except Exception:
            pass
        time.sleep(2)

    startup_error = f"vLLM did not become healthy within {STARTUP_TIMEOUT}s"
    log.error(startup_error)
    backend_ready.set()


# ---------------------------------------------------------------------------
# Request handling
# ---------------------------------------------------------------------------
def _error(message: str, kind: str = "worker_error") -> dict:
    return {"error": {"message": message, "type": kind, "code": None}}


def normalize(job_input: dict) -> tuple[str, str, dict | None]:
    """Map any accepted input shape to (method, route, body)."""
    if job_input.get("openai_route"):
        body = job_input.get("openai_input")
        return ("POST" if body else "GET"), job_input["openai_route"], body

    params = {k: v for k, v in job_input.items() if k not in ("messages", "prompt", "sampling_params")}
    params.update(job_input.get("sampling_params") or {})
    params.setdefault("model", SERVED_MODEL_NAME)
    if "messages" in job_input:
        return "POST", "/v1/chat/completions", {**params, "messages": job_input["messages"]}
    if "prompt" in job_input:
        return "POST", "/v1/completions", {**params, "prompt": job_input["prompt"]}
    raise ValueError("input needs 'messages', 'prompt', or 'openai_route' (+ 'openai_input')")


def _echo(route: str, body: dict | None) -> dict:
    """Fake OpenAI response for WORKER_BACKEND=echo."""
    if route.endswith("/models"):
        return {"object": "list", "data": [{"id": SERVED_MODEL_NAME, "object": "model", "owned_by": "runpod"}]}
    body = body or {}
    last = (body.get("messages") or [{}])[-1].get("content") or body.get("prompt", "")
    return {
        "id": "echo-1", "object": "chat.completion", "created": int(time.time()),
        "model": body.get("model", SERVED_MODEL_NAME),
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": f"echo: {last}"}}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


async def handler(job: dict) -> AsyncGenerator[Any, None]:
    if startup_error:
        yield _error(startup_error, "startup_error")
        return
    try:
        method, route, body = normalize(job.get("input") or {})
    except ValueError as e:
        yield _error(str(e), "invalid_request_error")
        return

    if WORKER_BACKEND == "echo":
        yield _echo(route, body)
        return

    if vllm_process is not None and vllm_process.poll() is not None:
        yield _error("vLLM process is not running; worker is unhealthy")
        return

    stream = isinstance(body, dict) and body.get("stream") is True
    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.request(method, VLLM_URL + route, json=body) as resp:
                if resp.status >= 400:
                    yield _error(f"vLLM HTTP {resp.status}: {await resp.text()}", "backend_error")
                    return
                if stream:
                    # Raw SSE text; RunPod's /openai route re-emits it as a stream.
                    async for chunk in resp.content.iter_any():
                        yield chunk.decode("utf-8", errors="replace")
                else:
                    yield await resp.json(content_type=None)
    except aiohttp.ClientError as e:
        yield _error(f"request to vLLM failed: {e}", "backend_error")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    runpod.serverless.register_prestart_hook(start_backend)
    runpod.serverless.start({
        "handler": handler,
        # Accept no jobs (so their execution timeout doesn't tick) until vLLM is up.
        "concurrency_modifier": lambda _current: MAX_CONCURRENCY if backend_ready.is_set() else 0,
        # /run + /runsync callers get the full output, streamed or not.
        "return_aggregate_stream": True,
    })
