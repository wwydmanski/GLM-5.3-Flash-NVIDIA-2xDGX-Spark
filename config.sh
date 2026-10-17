IMAGE=eugr/spark-vllm-b12x:latest
WORKER_HOST=nvidia@192.168.100.11
HEAD_IP=192.168.100.10
WORKER_IP=192.168.100.11
MASTER_PORT=29501
PORT=8000
# Explicit per-GPU KV budget instead of consuming all spare unified memory.
KV_CACHE_BYTES=4294967296
