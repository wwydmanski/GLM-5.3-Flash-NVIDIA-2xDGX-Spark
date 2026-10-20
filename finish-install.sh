#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/glm53-flash"
echo "$(date -Is) Waiting for model and images"
while kill -0 283236 2>/dev/null; do
  sleep 60
done
# A complete snapshot is required before copying and offline inference.
SNAP="$HOME/.cache/huggingface/hub/models--nvidia--GLM-5.3-Flash-FP4-Local/snapshots/76348ddb2be2e35e39ecfc00e75dc07703f4f3d5"
for i in $(seq 1 40); do
  printf -v shard 'model-%05d-of-00040.safetensors' "$i"
  [[ -s "$SNAP/$shard" ]] || { echo "Missing shard: $shard"; exit 1; }
done
[[ -f "$SNAP/tokenizer.json" ]]
echo "$(date -Is) Copying model to worker"
rsync -a --info=progress2 "$HOME/.cache/huggingface/hub/models--nvidia--GLM-5.3-Flash-FP4-Local" nvidia@192.168.100.11:.cache/huggingface/hub/
while ! docker image inspect eugr/spark-vllm-b12x:latest >/dev/null 2>&1; do sleep 30; done
while ! ssh nvidia@192.168.100.11 'docker image inspect eugr/spark-vllm-b12x:latest >/dev/null 2>&1'; do sleep 30; done
echo "$(date -Is) Launching GLM"
./glm start
./glm wait 7200
echo "$(date -Is) API ready"
