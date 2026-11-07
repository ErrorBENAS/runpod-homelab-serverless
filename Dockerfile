# RunPod Serverless worker = official vLLM OpenAI image + a ~200-line handler.
#
# Why this base: vLLM already ships CUDA, PyTorch and a tuned OpenAI server, so
# the only layer we add is the RunPod SDK (seconds to build, tiny to pull on top
# of a cached base). Bump vLLM with one build arg:
#   docker build --build-arg VLLM_VERSION=v0.33.0 -t runpod-homelab-serverless .
ARG VLLM_VERSION=v0.32.0
FROM vllm/vllm-openai:${VLLM_VERSION}
ARG VLLM_VERSION

WORKDIR /app

COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt \
    # Guard rail: our installs must not swap out the base image's vLLM.
    && test "$(python3 -c 'import vllm; print(vllm.__version__)')" = "${VLLM_VERSION#v}"

ENV PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false

COPY handler.py .

# The base image's ENTRYPOINT is `vllm serve`; replace it so the RunPod worker
# owns the process and launches vLLM itself in its prestart hook.
ENTRYPOINT ["python3", "-u", "/app/handler.py"]
