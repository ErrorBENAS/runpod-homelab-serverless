# runpod-homelab-serverless

A RunPod Serverless LLM worker for the **noblesseoblige** homelab, built to deploy quickly. It scales to zero and speaks the OpenAI API, so Open WebUI can use it like any other model. It's for the big models the Radeon 780M iGPU can't run.

```
Open WebUI (:3000, noblesseoblige) ──► runpod-proxy (docker, ai_default net)
                                          │  adds RunPod key, answers /v1/models locally
                                          ▼
                     https://api.runpod.ai/v2/<ENDPOINT_ID>/openai/v1
                                          ▼
                RunPod worker: handler.py ──► vllm serve (Qwen3) on 127.0.0.1:8000
```

- **Worker**: the official `vllm/vllm-openai` image plus `handler.py`. A RunPod *prestart hook* starts `vllm serve` once per worker, and the handler forwards jobs to it, streaming included.
- **Default model**: `Qwen/Qwen3-14B-AWQ` on a 24 GB GPU. It's from the same family as the homelab's Ollama models (`qwen3:14b`, `qwen3:30b-a3b-instruct`), so prompts carry over. You can switch models with an environment variable.
- **Homelab side**: `homelab/runpod-proxy` is a small proxy (FastAPI + httpx) that runs next to Open WebUI. It's reachable only on localhost and the docker network unless you bind it to the Tailscale IP.

## Repository layout

| Path | What |
|---|---|
| `handler.py` | Serverless handler. It starts vLLM in a prestart hook and forwards jobs to it. |
| `Dockerfile` | `vllm/vllm-openai:v0.32.0` + RunPod SDK 2.1 |
| `requirements.txt` | What the worker adds on top of the base image |
| `test_input.json` | Sample job for local tests and the RunPod console |
| `client/runpod_client.py` | Client that needs only the standard library: `/runsync`, `/run` + poll, the OpenAI route, `/health` |
| `homelab/runpod-proxy/` | OpenAI-compatible proxy for noblesseoblige (docker compose) |
| `.env.example` | Client environment variables, plus a reference list of the worker's |

---

## Deploy (fastest path: RunPod GitHub integration)

1. **Connect GitHub**: RunPod console → **Settings → Connections → GitHub → Connect**. Give it access to this repo.
2. **Create the endpoint**: **Serverless → New Endpoint → Import Git Repository**. Pick `runpod-homelab-serverless`, branch `main`, Dockerfile path `Dockerfile` (repo root). RunPod builds the image, and the **Builds** tab shows progress. The first build takes a few minutes because it pulls the ~10 GB vLLM base image. Later builds reuse it. Pushing new commits starts a rebuild according to the endpoint's build settings.
3. **GPU and scaling**:

   | Model (`MODEL_NAME`) | GPU class | `MAX_MODEL_LEN` |
   |---|---|---|
   | `Qwen/Qwen3-14B-AWQ` (default) | **24 GB** (RTX 4090 / L4 / A5000) | 32768 |
   | `Qwen/Qwen3-30B-A3B-Instruct-2507-FP8` | **48 GB** (L40S best, native FP8; A6000 works) | 32768 |
   | `Qwen/Qwen3-30B-A3B-Instruct-2507` (BF16) | 80 GB (A100/H100) | 32768 |

   - **Active (min) workers: 0** means it scales to zero and you pay only while it runs.
   - **Max workers: 1–2** is plenty for one household.
   - **Idle timeout: 60–120 s** keeps the worker warm between chat turns. The 5 s default means a cold start on nearly every message.
   - **Execution timeout: 600 s**.
   - **FlashBoot: on**. It cuts repeat cold starts sharply.
4. **Environment variables** (Edit Endpoint → Environment Variables). Each one is optional and has a default:

   | Var | Default | Notes |
   |---|---|---|
   | `MODEL_NAME` | `Qwen/Qwen3-14B-AWQ` | Any HF repo vLLM can serve |
   | `SERVED_MODEL_NAME` | `runpod-qwen3-14b` | The name shown in Open WebUI. It must match the proxy's `MODELS` setting. |
   | `MAX_MODEL_LEN` | `32768` | Lower it if you hit out-of-memory errors at startup |
   | `GPU_MEMORY_UTILIZATION` | `0.92` | |
   | `VLLM_EXTRA_ARGS` | *(empty)* | Suggested: `--reasoning-parser qwen3` so Qwen3 "thinking" shows up as reasoning in Open WebUI. For tool calling, add `--enable-auto-tool-choice --tool-call-parser hermes`. |
   | `MAX_CONCURRENCY` | `16` | Concurrent requests per worker. vLLM batches them. |
   | `HF_TOKEN` | – | Only for gated models. Store it as a RunPod **Secret** and set the value to `{{ RUNPOD_SECRET_hf_token }}`. |
   | `REQUEST_TIMEOUT` / `VLLM_STARTUP_TIMEOUT` | `600` / `1800` | Seconds |

5. **Model caching** (optional, makes cold starts much faster): in the endpoint settings, set the **Model** field (cached models) to the same HF repo as `MODEL_NAME`, *or* attach a **network volume**. `handler.py` finds `/runpod-volume` and uses it automatically: `HF_HOME=/runpod-volume/huggingface-cache` for downloads and `/runpod-volume/vllm-cache` for vLLM's compile cache. Without either one, every new worker downloads the weights again (about 10 GB for the default model).

### Test it

```bash
export RUNPOD_API_KEY=...   RUNPOD_ENDPOINT_ID=...

# plain RunPod API (handler's simple chat input)
curl -s https://api.runpod.ai/v2/$RUNPOD_ENDPOINT_ID/runsync \
  -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
  -d @test_input.json

# OpenAI-compatible route (what Open WebUI uses)
curl -s https://api.runpod.ai/v2/$RUNPOD_ENDPOINT_ID/openai/v1/chat/completions \
  -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"runpod-qwen3-14b","messages":[{"role":"user","content":"hi"}],"max_tokens":64}'

# included client
python client/runpod_client.py "Explain Tailscale in one sentence"
python client/runpod_client.py --openai "Hello"
python client/runpod_client.py --health
```

The first request after the endpoint has been idle is a **cold start**: the image starts and the model loads, roughly 1–3 min with model caching and longer without. `/runsync` waits about 90 s. After that the client switches to polling `/status` by itself.

The plain API returns the result as `output: [ {...openai response...} ]`, a one-element list, because the handler streams. The client unwraps it.

---

## Homelab integration (noblesseoblige)

### Option A (recommended): local proxy next to Open WebUI

```bash
# from this repo on the PC
scp -r homelab/runpod-proxy homelab:/opt/apps/
ssh homelab
cd /opt/apps/runpod-proxy
cp .env.example runpod-proxy.env && chmod 600 runpod-proxy.env
nano runpod-proxy.env      # RUNPOD_API_KEY, RUNPOD_ENDPOINT_ID, PROXY_API_KEY (openssl rand -hex 24)
docker compose up -d --build
curl -s localhost:7598/healthz
```

Then in **Open WebUI → Admin Panel → Settings → Connections → OpenAI API → +**:

- URL: `http://runpod-proxy:8080/v1`. This goes over the shared `ai_default` docker network, with no host port and no tailnet hop.
- Key: the `PROXY_API_KEY` you generated.

`runpod-qwen3-14b` now appears in the model picker next to the Ollama and Hermes models. Hermes, the Grok bots, or n8n can use the same URL with the same key. To reach the proxy from other machines on the tailnet, set `BIND_ADDR=<server Tailscale IP>` in `/opt/apps/runpod-proxy/.env` and recreate the container. It's still never published on the LAN or through Funnel.

Why use the proxy: the RunPod key stays in one mode-600 file, and **`/v1/models` is answered locally**. Upstream, each model-list refresh is a queued job that can wake a GPU worker and bill you for a cold start just because someone opened the WebUI.

### Option B: point Open WebUI straight at RunPod

Add an OpenAI connection with:

- URL: `https://api.runpod.ai/v2/<ENDPOINT_ID>/openai/v1`
- Key: your RunPod API key. Prefer a **restricted key** scoped to this endpoint.
- **Model IDs**: add `runpod-qwen3-14b` explicitly. With an explicit list, Open WebUI doesn't need to call `/models` upstream, so opening the UI doesn't wake a worker.

### Option C: from your own code

```python
from openai import OpenAI
client = OpenAI(base_url="http://runpod-proxy:8080/v1", api_key=PROXY_API_KEY)   # or the RunPod URL + key
client.chat.completions.create(model="runpod-qwen3-14b", messages=[...], stream=True)
```

Or use `client/runpod_client.py`: it needs only the standard library and works with `RUNPOD_API_KEY` and `RUNPOD_ENDPOINT_ID`.

---

## Local development

The handler has an `echo` backend, so you can test the RunPod plumbing without a GPU:

```bash
pip install runpod~=2.1.0 aiohttp
WORKER_BACKEND=echo python handler.py --rp_serve_api --rp_api_concurrency 1 --rp_api_port 8080
curl -s localhost:8080/runsync -H "Content-Type: application/json" -d @test_input.json
```

Use the local API (`--rp_serve_api`), not `--test_input`. In runpod 2.1 the `--test_input` path doesn't iterate async-generator handlers. Use port 8080 because vLLM takes 8000 inside the worker.

On a machine with an NVIDIA GPU, run the real image:

```bash
docker build -t runpod-homelab-serverless .
docker run --gpus all --rm -p 8080:8080 -e MODEL_NAME=Qwen/Qwen3-0.6B -e MAX_MODEL_LEN=4096 \
  runpod-homelab-serverless --rp_serve_api --rp_api_concurrency 1 --rp_api_host 0.0.0.0 --rp_api_port 8080
curl -s localhost:8080/runsync -H "Content-Type: application/json" -d @test_input.json
```

## Fallback: manual Docker image

If you don't want the GitHub integration:

```bash
docker build --platform linux/amd64 -t <dockerhub-user>/runpod-homelab-serverless:0.1.0 .
docker push <dockerhub-user>/runpod-homelab-serverless:0.1.0
```

Then go to **Serverless → New Endpoint → Docker Image** and enter that tag. The GPU, scaling and environment settings are the same as above. Use a fixed version tag rather than `latest` so workers pull the build you expect.

## Swapping models / engines

- **Another model**: change `MODEL_NAME` and `SERVED_MODEL_NAME` on the endpoint, then update `MODELS` in `runpod-proxy.env`. No rebuild is needed.
- **Newer vLLM**: bump `ARG VLLM_VERSION` in the `Dockerfile` and push. The build fails if pip replaced the base image's vLLM.
- **Official worker**: `runpod-workers/worker-vllm` uses the same input contract, including `openai_route` and `openai_input`. Swapping to it doesn't change the proxy or Open WebUI.

## Security notes

- No secrets go in this repo. `.env` and `*.env` are gitignored. Keys live in RunPod Secrets, `/opt/apps/runpod-proxy/runpod-proxy.env` (mode 600), or Open WebUI's settings.
- The proxy binds to `127.0.0.1` by default and only joins the internal `ai_default` network. Expose it on the tailnet only if you need to.
- Use a RunPod API key restricted to this endpoint, with read/write on serverless only.
