#!/usr/bin/env bash
# Test a Cacheon-style miner image: GET /health + POST /v1/chat/completions (stream + logprobs).
# Matches what the validator exercises (see validator/docker_eval.py).
#
# Requirements (--image mode): NVIDIA GPU(s), drivers, and NVIDIA Container Toolkit (--gpus all).
# Qwen2.5-72B needs multi-GPU in practice; CPU-only machines cannot run this stack meaningfully.
# Building the image (docker build) does not require a GPU.
#
# Usage — start container, run tests, remove container:
#   ./test_miner_image.sh --image docker.io/you/cacheon-miner:v1 --model-dir /path/to/Qwen2.5-72B-Instruct
#
# Usage — container already listening on localhost:
#   ./test_miner_image.sh --url http://127.0.0.1:8000
#
# Options:
#   --port N          host port when using --image (default 18000 to avoid clashes)
#   --wait SEC        max seconds to wait for /health (default 600)
#   --keep            do not remove container when using --image

set -euo pipefail

IMAGE=""
MODEL_DIR=""
BASE_URL=""
PORT="18000"
WAIT_SECS="600"
KEEP="0"
CID=""
TMP_OUT=""

usage() {
  sed -n '1,18p' "$0" | tail -n +2
  exit 1
}

cleanup() {
  rm -f "${TMP_OUT}"
  if [[ "$KEEP" == "1" ]]; then
    return 0
  fi
  if [[ -n "${CID}" ]]; then
    echo "Removing container ${CID}..."
    docker rm -f "${CID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image) IMAGE="${2:-}"; shift 2 ;;
    --model-dir) MODEL_DIR="${2:-}"; shift 2 ;;
    --url) BASE_URL="${2:-}"; shift 2 ;;
    --port) PORT="${2:-}"; shift 2 ;;
    --wait) WAIT_SECS="${2:-}"; shift 2 ;;
    --keep) KEEP="1"; shift ;;
    -h|--help) usage ;;
    *) echo "unknown arg: $1" >&2; usage ;;
  esac
done

if [[ -n "$BASE_URL" ]]; then
  :
elif [[ -n "$IMAGE" && -n "$MODEL_DIR" ]]; then
  if ! command -v docker >/dev/null 2>&1; then
    echo "error: docker not found" >&2
    exit 1
  fi
  if [[ ! -d "$MODEL_DIR" ]]; then
    echo "error: model dir not found: $MODEL_DIR" >&2
    exit 1
  fi
  echo "Starting container from ${IMAGE} (host port ${PORT} -> 8000)..."
  CID="$(
    docker run -d \
      --gpus all \
      --shm-size 16g \
      -v "${MODEL_DIR}:/models:ro" \
      -p "${PORT}:8000" \
      "${IMAGE}"
  )"
  BASE_URL="http://127.0.0.1:${PORT}"
  echo "Container id: ${CID}"
else
  echo "error: provide either (--image and --model-dir) or --url" >&2
  usage
fi

echo "Waiting for GET ${BASE_URL}/health (timeout ${WAIT_SECS}s)..."
start_ts=$(date +%s)
ok=""
while true; do
  if curl -sf --max-time 5 "${BASE_URL}/health" >/dev/null; then
    ok="1"
    break
  fi
  now=$(date +%s)
  if (( now - start_ts >= WAIT_SECS )); then
    break
  fi
  sleep 3
done

if [[ -z "$ok" ]]; then
  echo "FAIL: /health did not return HTTP 2xx within ${WAIT_SECS}s" >&2
  if [[ -n "${CID}" ]]; then
    echo "--- docker logs (tail) ---" >&2
    docker logs "${CID}" 2>&1 | tail -n 80 >&2 || true
  fi
  exit 1
fi
echo "OK: /health"

TMP_OUT="$(mktemp)"

echo "POST ${BASE_URL}/v1/chat/completions (stream=true, logprobs=true, temperature=0)..."
HTTP_CODE="$(
  curl -sS -N --max-time 120 \
    -o "${TMP_OUT}" \
    -w "%{http_code}" \
    -H "Content-Type: application/json" \
    -X POST "${BASE_URL}/v1/chat/completions" \
    -d @- <<'EOF'
{"model":"Qwen2.5-72B-Instruct","messages":[{"role":"user","content":"Reply with exactly one word: OK."}],"max_tokens":32,"temperature":0,"stream":true,"logprobs":true,"top_logprobs":5}
EOF
)"

if [[ "$HTTP_CODE" != "200" ]]; then
  echo "FAIL: chat/completions HTTP ${HTTP_CODE}" >&2
  head -c 2000 "${TMP_OUT}" >&2 || true
  exit 1
fi

if ! grep -qE '^data:' "${TMP_OUT}"; then
  echo "FAIL: expected SSE lines starting with 'data:'" >&2
  head -c 2000 "${TMP_OUT}" >&2 || true
  exit 1
fi

if ! grep -q 'chat.completion.chunk' "${TMP_OUT}"; then
  echo "FAIL: expected OpenAI stream chunks (chat.completion.chunk)" >&2
  head -c 2000 "${TMP_OUT}" >&2 || true
  exit 1
fi

if ! grep -q '"logprobs"' "${TMP_OUT}"; then
  echo "FAIL: expected logprobs in stream chunks (validator enables logprobs)" >&2
  head -c 2000 "${TMP_OUT}" >&2 || true
  exit 1
fi

if ! grep -q 'data: \[DONE\]' "${TMP_OUT}"; then
  echo "WARN: no 'data: [DONE]' seen (stream may still be valid); first 500 bytes:" >&2
  head -c 500 "${TMP_OUT}" >&2 || true
fi

echo "OK: streaming chat completions with logprobs"
echo "Sample stream head:"
head -n 8 "${TMP_OUT}" | sed 's/^/  /'

echo ""
echo "All checks passed."
