"""Tiny OpenAI-compatible proxy: homelab -> RunPod Serverless.

Runs on noblesseoblige next to Open WebUI. Open WebUI (and any bot on the
tailnet) talks plain OpenAI to http://runpod-proxy:8080/v1; this service adds
the RunPod key and forwards to https://api.runpod.ai/v2/<id>/openai/v1.

Why not point Open WebUI straight at RunPod?
- The RunPod API key lives in one root-only env file, not in every client.
- GET /v1/models is answered locally. Upstream, every model-list refresh is a
  queued job that can wake a GPU worker (and bill a cold start) just because
  someone opened the WebUI.
- Streaming, long cold-start timeouts and a /status view of RunPod workers.
"""

from __future__ import annotations

import os
import time

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

RUNPOD_API_KEY = os.environ["RUNPOD_API_KEY"]
RUNPOD_ENDPOINT_ID = os.environ["RUNPOD_ENDPOINT_ID"]
# Optional bearer that clients must send. Empty = no auth (only OK if bound to
# localhost / the docker network).
PROXY_API_KEY = os.getenv("PROXY_API_KEY", "")
# Model ids advertised to clients; must match the worker's SERVED_MODEL_NAME.
MODELS = [m.strip() for m in os.getenv("MODELS", "runpod-qwen3-14b").split(",") if m.strip()]
# Cold start (image pull + model load) can take minutes; don't give up early.
TIMEOUT = float(os.getenv("UPSTREAM_TIMEOUT", "600"))

BASE = f"{os.getenv('RUNPOD_API_BASE', 'https://api.runpod.ai/v2')}/{RUNPOD_ENDPOINT_ID}"
FORWARD = {"chat/completions", "completions", "embeddings"}

app = FastAPI(title="runpod-proxy", docs_url=None, redoc_url=None)
client = httpx.AsyncClient(
    headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"},
    timeout=httpx.Timeout(TIMEOUT, connect=15),
)


def _check_auth(request: Request) -> None:
    if PROXY_API_KEY and request.headers.get("authorization") != f"Bearer {PROXY_API_KEY}":
        raise HTTPException(401, "invalid proxy API key")


@app.get("/healthz")
async def healthz():
    return {"ok": True}


@app.get("/status")
async def status(request: Request):
    """RunPod worker/queue counts. Does not wake a worker."""
    _check_auth(request)
    r = await client.get(f"{BASE}/health")
    return JSONResponse(r.json(), status_code=r.status_code)


@app.get("/v1/models")
async def models(request: Request):
    _check_auth(request)
    now = int(time.time())
    return {"object": "list",
            "data": [{"id": m, "object": "model", "created": now, "owned_by": "runpod"} for m in MODELS]}


@app.post("/v1/{path:path}")
async def forward(path: str, request: Request):
    _check_auth(request)
    if path not in FORWARD:
        raise HTTPException(404, f"/v1/{path} is not forwarded")
    body = await request.body()
    upstream = client.build_request(
        "POST", f"{BASE}/openai/v1/{path}", content=body,
        headers={"Content-Type": "application/json"},
    )
    resp = await client.send(upstream, stream=True)
    media = resp.headers.get("content-type", "application/json")

    if "text/event-stream" in media:
        async def relay():
            try:
                async for chunk in resp.aiter_raw():
                    yield chunk
            finally:
                await resp.aclose()
        return StreamingResponse(relay(), status_code=resp.status_code, media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    content = await resp.aread()
    await resp.aclose()
    return Response(content, status_code=resp.status_code, media_type=media)
