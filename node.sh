#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
source "$DIR/config.sh"
RANK="${1:?node rank required}"
mkdir -p "$HOME/.cache/"{huggingface,vllm,flashinfer,b12x} "$HOME/.triton"
IP="$HEAD_IP"
[[ "$RANK" == 1 ]] && IP="$WORKER_IP"
extra=()
[[ "$RANK" == 1 ]] && extra+=(--headless)
[[ -n "${SPEC_CONFIG:-}" ]] && extra+=(--speculative-config "$SPEC_CONFIG")
[[ -n "${EXTRA_ARGS:-}" ]] && extra+=($EXTRA_ARGS)
docker_extra=()
[[ -n "${DOCKER_ARGS:-}" ]] && docker_extra+=($DOCKER_ARGS)
exec docker run -d --name glm53-flash --init --restart no \
  --privileged --gpus all --network host --ipc host --shm-size 16g \
  --stop-timeout 60 --ulimit nofile=1048576:1048576 \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --env-file "$DIR/runtime.env" \
  -e VLLM_HOST_IP="$IP" -e HEAD_IP="$HEAD_IP" -e MASTER_PORT="$MASTER_PORT" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$HOME/.cache/vllm:/root/.cache/vllm" \
  -v "$HOME/.cache/flashinfer:/root/.cache/flashinfer" \
  -v "$HOME/.cache/b12x:/root/.cache/b12x" \
  -v "$HOME/.triton:/root/.triton" \
  "${docker_extra[@]}" \
  --entrypoint vllm "$IMAGE" serve "${MODEL:-nvidia/GLM-5.3-Flash-FP4-Local}" \
  --served-model-name nvidia/GLM-5.3-Flash-FP4-Local \
  --host 0.0.0.0 --port "$PORT" --tensor-parallel-size 2 \
  --mamba-cache-mode align --enable-prefix-caching --enable-chunked-prefill \
  --dtype bfloat16 --kv-cache-dtype fp8 --quantization modelopt_mixed \
  --attention-backend B12X --block-size 256 --moe-backend b12x \
  --linear-backend b12x --no-enable-flashinfer-autotune --load-format b12x \
  --max-model-len 131000 --max-num-seqs 4 --max-num-batched-tokens 4096 \
  --kv-cache-memory-bytes "$KV_CACHE_BYTES" \
  --reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice \
  --gpu-memory-utilization 0.73 --nnodes 2 --node-rank "$RANK" \
  --master-addr "$HEAD_IP" --master-port "$MASTER_PORT" "${extra[@]}"
