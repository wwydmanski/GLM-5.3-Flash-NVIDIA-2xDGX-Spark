IMAGE=eugr/spark-vllm-b12x:latest
WORKER_HOST=nvidia@192.168.100.11
HEAD_IP=192.168.100.10
WORKER_IP=192.168.100.11
MASTER_PORT=29501
PORT=8000
# Explicit per-GPU KV budget instead of consuming all spare unified memory.
KV_CACHE_BYTES=4294967296
# MTP speculative decoding config (model-card default: 5 draft tokens).
SPEC_CONFIG='{"method":"mtp","num_speculative_tokens":3,"draft_sample_method":"probabilistic","rejection_sample_method":"standard","moe_backend":"humming","attention_backend":"B12X"}'
EXTRA_ARGS=''
DOCKER_ARGS='-e VLLM_MTP_NVFP4_LM_HEAD=0'
MODEL=
