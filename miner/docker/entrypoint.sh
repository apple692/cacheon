#!/usr/bin/env bash
# Cacheon miner: OpenAI-compatible server on :8000, weights from MODEL_PATH (default /models).
# Mirrors validator baseline logic in validator/docker_eval.py for max_model_len and TP count.
#
# Scoring-oriented defaults (see miner/docker/README.md "Scoring-oriented tuning"):
#   - Long-context eval → optional chunked prefill (vLLM version dependent).
#   - Continuous batching caps → conservative defaults; eval is mostly sequential requests.
#   - No duplicate "warmup before /health": validator already discards the first 2 prompts.
#
# Always use a local model path (e.g. /models), never a Hugging Face hub id in production.
#
# Startup hygiene: do not `pip install` or download weights here — only exec the server.
# Readiness: rely on vLLM's /health (or your engine's equivalent) meaning "model can serve traffic".
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/models}"
PORT="${PORT:-8000}"
DTYPE="${VLLM_DTYPE:-bfloat16}"
GPU_MEM_UTIL="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-32}"
MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-8192}"
# 1|true|yes = pass --enable-chunked-prefill; 0|false|no = omit (use vLLM default). Disable if your vLLM build rejects the flag.
VLLM_ENABLE_CHUNKED_PREFILL="${VLLM_ENABLE_CHUNKED_PREFILL:-1}"
# Optional overrides; leave unset for validator-parity defaults.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
TP_SIZE="${TP_SIZE:-}"

gpu_count="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -c . || true)"
if [ "${gpu_count}" -lt 1 ]; then
  echo "error: no GPUs visible (nvidia-smi failed or zero GPUs)" >&2
  exit 1
fi

if [ -z "${MAX_MODEL_LEN}" ]; then
  max_model_len=32768
  if [ "${gpu_count}" -ge 8 ]; then
    max_model_len=131072
  elif [ "${gpu_count}" -ge 4 ]; then
    max_model_len=65536
  fi
else
  max_model_len="${MAX_MODEL_LEN}"
fi

if [ -n "${TP_SIZE}" ]; then
  tp="${TP_SIZE}"
else
  tp="${gpu_count}"
fi

args=(
  --model "${MODEL_PATH}"
  # Required: validator requests body uses "model": "Qwen2.5-72B-Instruct" (validator/docker_eval.py).
  --served-model-name Qwen2.5-72B-Instruct
  --generation-config vllm
  --max-model-len "${max_model_len}"
  --host 0.0.0.0
  --port "${PORT}"
  --dtype "${DTYPE}"
  --gpu-memory-utilization "${GPU_MEM_UTIL}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
  --trust-remote-code
)

case "$(printf '%s' "${VLLM_ENABLE_CHUNKED_PREFILL}" | tr '[:upper:]' '[:lower:]')" in
  1 | true | yes) args+=(--enable-chunked-prefill) ;;
  0 | false | no) ;;
  *)
    echo "error: VLLM_ENABLE_CHUNKED_PREFILL must be 1/true/yes or 0/false/no, got: ${VLLM_ENABLE_CHUNKED_PREFILL}" >&2
    exit 1
    ;;
esac

if [ "${tp}" -gt 1 ]; then
  args+=(--tensor-parallel-size "${tp}")
fi

# Extra vLLM CLI tokens (quote-sensitive — use simple space-separated flags).
extra_args=()
if [ -n "${VLLM_EXTRA_ARGS:-}" ]; then
  # shellcheck disable=SC2206
  extra_args=(${VLLM_EXTRA_ARGS})
fi

exec python3 -m vllm.entrypoints.openai.api_server "${args[@]}" "${extra_args[@]}" "$@"
