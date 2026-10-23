#!/usr/bin/env bash
# vast.ai on-start: download GLM-5.3-Flash-FP4-Local and serve it on both RTX PRO 6000 (TP=2) with tuned MTP3.
# Logs: /workspace/glm/{download,vllm}.log. Kernel settings come from the template's Docker options (docker-options.txt).
mkdir -p /workspace/glm && cd /workspace/glm
export HF_HOME=/workspace/hf
(
  /usr/local/bin/hf download nvidia/GLM-5.3-Flash-FP4-Local > download.log 2>&1 || exit 1
  # The image's entrypoint sets the CUDA toolchain paths that b12x JIT needs (cuda.h, ptxas).
  exec /usr/local/bin/lil-entrypoint /opt/venv/bin/vllm serve nvidia/GLM-5.3-Flash-FP4-Local \
    --host 0.0.0.0 --port 8000 --served-model-name nvidia/GLM-5.3-Flash-FP4-Local \
    ${VLLM_API_KEY:+--api-key "$VLLM_API_KEY"} \
    --tensor-parallel-size 2 --quantization modelopt_mixed --load-format instanttensor --dtype bfloat16 \
    --attention-backend B12X --moe-backend b12x --linear-backend b12x --no-enable-flashinfer-autotune \
    --additional-config '{"glm53_kda_decode_backend":"auto","kda_prefill_backend":"b12x"}' \
    --block-size 256 --kv-cache-dtype fp8 --mamba-cache-mode align --enable-prefix-caching --enable-chunked-prefill \
    --max-model-len 131072 --max-num-seqs 4 --max-num-batched-tokens 4096 \
    --gpu-memory-utilization 0.97 --kv-cache-memory-bytes 7650410496 \
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' --cudagraph-capture-sizes 1 2 4 8 16 --max-cudagraph-capture-size 16 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3,"draft_sample_method":"probabilistic","rejection_sample_method":"standard","moe_backend":"b12x","attention_backend":"B12X"}' \
    --reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice > vllm.log 2>&1
) &
