# Cacheon miner Docker image (reference)

This directory builds a **reference** miner image using **vLLM** with the same high-level settings the validator uses for its baseline (`/models`, `Qwen2.5-72B-Instruct`, tensor parallel aligned to visible GPUs, `generation-config vllm`). Replace this image with your own optimizations while keeping the **HTTP contract** (`GET /health`, `POST /v1/chat/completions` with streaming + logprobs).

### CUDA / PyTorch / vLLM versions

Inference **speed and CLI flags change between releases**.

This Dockerfile **defaults to `vllm/vllm-openai:latest`** so you always track the current upstream image (CUDA, PyTorch, vLLM, transformers, etc., as that tag points to today).

- **Tradeoff:** two `docker build` runs weeks apart can use **different** upstream stacks even if your Dockerfile did not change.
- **Reproducible builds:** pass a fixed tag, e.g.  
  `docker build --build-arg VLLM_BASE=vllm/vllm-openai:v0.8.4 -t ...`
- **Strongest pin on-chain:** after `docker push`, commit the **manifest digest** (`sha256:...`) with `miner/commit.py` — validators pull `image@digest`, so that locks the exact image bytes regardless of `:latest` drift until you push again and commit a new digest.
- **Advanced:** `FROM nvidia/cuda:12.x-devel-ubuntu22.04` plus `pip install torch==... vllm==...` with explicit pins; more work to keep wheels compatible than using **`vllm/vllm-openai:<tag>`**.

### Model path: use `/models`, not a Hub id

Use **`--model /models`** (or `MODEL_PATH=/models`) so the server reads the weights the validator bind-mounts. Do **not** use **`--model Qwen/Qwen2.5-72B-Instruct`** in the image you submit: that can trigger Hugging Face downloads, slower startup, and eval failures under the validator’s **internal-only** Docker network.

### Do not bake model weights into the image

Do **not** add lines like `RUN huggingface-cli download Qwen/Qwen2.5-72B-Instruct` (or multi‑GB `COPY` of checkpoints). That blows image size and pull time, can violate subnet image limits, and is redundant because validators already mount **`/models`**. The image should only **load** the mounted tree via **`--model /models`** (or `MODEL_PATH`).

### Lean container startup (§14)

Keep **install and compile in `docker build`**, not in `entrypoint.sh`:

- No **`pip install`** at container start.
- No **model download** at start (use **`/models`** only).
- The entrypoint should **only start the server** (this repo does that).

That reduces cold-start variance and avoids burning the validator’s startup window on work that belongs in the image layer cache.

### `/health` and readiness (§16)

The validator polls **`GET /health`** until **HTTP 200** (`validator/docker_eval.py`). If you return **200 before the model can serve** real `chat/completions` (streaming + logprobs, long context), the first requests can see **very bad TTFT** or errors.

With **vLLM’s** OpenAI API server, use its built-in **`/health`** semantics and avoid wrapping it with a proxy that reports ready too early. If you ship a **custom** server, gate **200** on: weights loaded, engine initialized, and the process is actually ready to handle eval-shaped traffic — not merely “process started”.

### Served name must match the harness

The validator always sends `"model": "Qwen2.5-72B-Instruct"` in `POST /v1/chat/completions`. Your server must accept that id (vLLM: **`--served-model-name Qwen2.5-72B-Instruct`**). A minimal Dockerfile that omits this often breaks eval even if `/models` is correct.

### Tensor parallel size

Do **not** hardcode `TP_SIZE=8` unless you only run on eight GPUs. The validator uses **`--gpus all`** on **4×H200-class** machines for V1; the container typically sees **four** devices, so tensor parallel should be **4** there. This entrypoint defaults **TP to the visible GPU count** (override with `-e TP_SIZE=...` only when you know the hardware).

### `vllm serve` vs `python -m vllm.entrypoints.openai.api_server`

Both can work on newer vLLM images; this repo uses the **`api_server`** module form for parity with the pinned baseline container args in code. If your base image only documents `vllm serve`, switch the `exec` line in `entrypoint.sh` to that CLI and keep the same flags.

### Scoring-oriented tuning (what the entrypoint applies)

The subnet score (when correct) blends **TTFT** and **decode throughput** about **50/50** on **median** per-prompt stats. The harness uses **long** PG19-style contexts, **`stream=true`**, **`logprobs=true`**, and sends requests **one at a time** (not a flood of parallel clients). It also **discards the first two prompts** as warmup for both baseline and challenger before aggregating scores.

This image applies:

| Default | Rationale |
|--------|-----------|
| `--max-num-seqs 32` and `--max-num-batched-tokens 8192` | Sensible continuous-batching anchors; tune with `VLLM_MAX_NUM_SEQS` / `VLLM_MAX_NUM_BATCHED_TOKENS`. Under sequential eval they mainly shape **scheduler / memory**, not “64-way client concurrency”. |
| `--enable-chunked-prefill` when `VLLM_ENABLE_CHUNKED_PREFILL=1` (default) | Long prefill is typical; chunked prefill often helps **throughput / scheduling** on large contexts. **A/B** against `VLLM_ENABLE_CHUNKED_PREFILL=0` on your vLLM build if TTFT regresses. If the flag is **unknown** to your vLLM version, set **`VLLM_ENABLE_CHUNKED_PREFILL=0`**. |
| No extra “warmup before `/health`” in the entrypoint | Readiness stays simple and fast; the validator’s **two** discarded prompts already pay most cold-start cost relevant to scoring. |
| Direct `python3 -m vllm.entrypoints.openai.api_server` | Avoids extra proxy latency on the streaming hot path. |

**Optional:** space-separated flags in `VLLM_EXTRA_ARGS` (e.g. experimental knobs). Prefer an **honest `/health`** over faking readiness; the harness already discards two warmup prompts after health passes.

## Step 1 — Model weights on disk (local test only)

Download or sync **`Qwen/Qwen2.5-72B-Instruct`** in Hugging Face layout to a directory on the host, for example:

`/data/models/Qwen2.5-72B-Instruct/`

That directory is what you mount as **`/models`** inside the container (same layout validators use under `CACHEON_MODEL_VOLUME`).

## Step 2 — Build the image

From **this** directory (`miner/docker/`). The Dockerfile **defaults to** `vllm/vllm-openai:latest`.

```bash
export IMAGE=docker.io/YOURUSER/cacheon-miner:v1

docker build -t "${IMAGE}" .
```

Use a **fixed tag** instead of `latest` when you want reproducible local builds:

```bash
docker build --build-arg VLLM_BASE=vllm/vllm-openai:v0.8.4 -t "${IMAGE}" .
```

## Step 3 — Run locally (sanity check)

You need NVIDIA drivers, the NVIDIA Container Toolkit, and enough GPU memory for 72B (typically **multiple GPUs** with tensor parallelism, matching how the validator runs with `--gpus all`).

```bash
export MODEL_DIR=/data/models/Qwen2.5-72B-Instruct

docker run --rm -it --gpus all \
  --shm-size 16g \
  -v "${MODEL_DIR}:/models:ro" \
  -p 8000:8000 \
  "${IMAGE}"
```

Wait until the server is up, then:

```bash
curl -sS "http://127.0.0.1:8000/health"
```

### Automated check (health + streaming logprobs)

**GPU:** The reference image runs vLLM on CUDA. The test script uses `docker run --gpus all`, so you need a **GPU machine** (typically **several GPUs** for 72B). **`docker build` does not need a GPU.** If the server is already running elsewhere, `./test_miner_image.sh --url http://...` only needs **network** access to that host from where you run the script (that host must still be GPU-backed for vLLM).

From `miner/docker/`, after the image is built (or with a container already on port 8000):

```bash
# Full flow: start container, wait for /health, POST chat/completions like the validator, then remove.
./test_miner_image.sh --image docker.io/YOURUSER/cacheon-miner:v1 --model-dir /data/models/Qwen2.5-72B-Instruct

# Only API checks (you started the container yourself):
./test_miner_image.sh --url http://127.0.0.1:8000
```

Options: `--port 18000` (host port when using `--image`), `--wait 600` (health timeout seconds), `--keep` (leave the container running after success).

## Step 4 — Log in to the registry and push

```bash
docker login
docker push "${IMAGE}"
```

## Step 5 — Copy the manifest digest

Validators pull **`image@sha256:...`**. After push, obtain the digest:

```bash
docker inspect --format='{{index .RepoDigests 0}}' "${IMAGE}"
```

Example output: `docker.io/youruser/cacheon-miner@sha256:abcdef...` — use the **`sha256:...`** part as `--digest` when committing.

## Step 6 — Commit on-chain

From the repo root:

```bash
python miner/commit.py \
  --image "${IMAGE}" \
  --digest 'sha256:YOUR_64_CHAR_HEX' \
  --wallet-name YOUR_WALLET \
  --wallet-hotkey YOUR_HOTKEY \
  --network finney \
  --netuid 14
```

(Testnet example: `--network test --netuid 470`.)

## Updating the image

1. Edit `Dockerfile` or `entrypoint.sh` (e.g. extra vLLM flags, different base tag).
2. Bump the image tag (e.g. `v2`), rebuild, push.
3. Resolve the **new** manifest digest and commit again — the old digest always points at the old image.

## Notes

- **Image size:** keep the layer under subnet rules (see project README; weights stay outside the image on `/models`).
- **vLLM API:** if a future `vllm/vllm-openai` image drops `python3 -m vllm.entrypoints.openai.api_server`, update `entrypoint.sh` to the supported launch command for that version.
- **Single-GPU local runs** of 72B may OOM; use the same class of multi-GPU hardware as production eval when possible.
