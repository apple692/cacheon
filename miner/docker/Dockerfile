# Reference Cacheon miner image: vLLM OpenAI server, Qwen2.5-72B-Instruct from /models.
#
# Weights are never baked in — validators mount weights at /models (read-only).
# Dependencies come only from the base image; do not `RUN pip install` without pinning versions.
# Default base is :latest (upstream stack can change between builds). Pin a tag if you need
# reproducible perf, or rely on the manifest digest you commit on-chain.
#
# Build (from this directory):
#   docker build -t docker.io/YOURUSER/cacheon-miner:v1 .
# Optional: use a fixed vLLM tag instead of latest:
#   docker build --build-arg VLLM_BASE=vllm/vllm-openai:v0.8.4 -t ...

ARG VLLM_BASE=vllm/vllm-openai:latest
FROM ${VLLM_BASE}

# Same default as pre-FROM ARG so LABEL is set when no --build-arg is passed.
ARG VLLM_BASE=vllm/vllm-openai:latest
LABEL org.opencontainers.image.description="Cacheon reference miner (vLLM OpenAI server)"
LABEL ai.cacheon.vllm-base="${VLLM_BASE}"

# Defaults match Cacheon eval: weights mounted read-only at /models; API on 8000.
# Override at runtime: -e MODEL_PATH=... -e TP_SIZE=4 -e MAX_MODEL_LEN=65536 etc.
ENV MODEL_PATH=/models
ENV PORT=8000
ENV VLLM_DTYPE=bfloat16
ENV VLLM_GPU_MEMORY_UTILIZATION=0.90
# Batching / long-prefill (tune for your vLLM version + hardware; see README).
ENV VLLM_MAX_NUM_SEQS=32
ENV VLLM_MAX_NUM_BATCHED_TOKENS=8192
# Set to 0 on vLLM builds that do not support --enable-chunked-prefill.
ENV VLLM_ENABLE_CHUNKED_PREFILL=1

COPY entrypoint.sh /usr/local/bin/cacheon-miner-entrypoint.sh
RUN chmod +x /usr/local/bin/cacheon-miner-entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["/usr/local/bin/cacheon-miner-entrypoint.sh"]
