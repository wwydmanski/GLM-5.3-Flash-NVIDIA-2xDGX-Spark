# GLM 5.3 Flash on the two GX10 nodes

Run on a DGX Spark (both nodes must be booted and reachable):

```sh
~/glm53-flash/safe-start.sh   # preferred start, see "Safe start" below
~/.local/bin/glm start
~/.local/bin/glm wait
~/.local/bin/glm stop
~/.local/bin/glm restart
~/.local/bin/glm status
~/.local/bin/glm logs
~/.local/bin/glm worker-logs
```

`glm` is also available by name in shells that include ~/.local/bin in PATH.
Containers run detached and survive SSH disconnection. After reboot use `glm
start` on the head. There is deliberately no independent Docker auto-restart,
which could start the head without the worker. Stopping affects only this model.

API: http://host:8000/v1
Model: nvidia/GLM-5.3-Flash-FP4-Local
Context: exactly 131000 tokens, FP8 KV, 4 concurrent sequences.
KV cache budget: 4 GiB per GPU (config.sh), not all free unified RAM.
The context is per request, the KV pool is shared among concurrent requests.

The NVIDIA model-card two-Spark B12X profile is used, with a tuned MTP setup:
3 draft tokens, probabilistic draft sampling and a BF16 draft lm_head
(`VLLM_MTP_NVFP4_LM_HEAD=0`). On replayed staging traffic this raised
position-1 acceptance from 0.72 to 0.86 and decode to 42.0 tok/s vs 37.0
without speculation (same sampling: MTP2 40.0, MTP4 41.8, MTP5 ~39.5). Change it via `SPEC_CONFIG` / `DOCKER_ARGS` in config.sh.
Files: node.sh (launch), runtime.env (kernel/network settings), config.sh
(image/network/KV budget/speculative config, optional `EXTRA_ARGS` and `MODEL`
override), glm (two-node lifecycle), safe-start.sh (guarded start).

## Safe start

Loading can exhaust the head's unified memory while b12x expands the CSF expert
scales; the node then freezes until a watchdog reboot, after which the GB10
clock may stay stuck at ~600 MHz until a physical power cycle.
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in runtime.env fixes the
peak. `safe-start.sh` additionally stops `infinity_embed` and drops the page
cache, runs `glm start`, kills GLM if MemAvailable falls under 2 GiB
(`MIN_AVAIL_KB`), logs to safe-start.log / safe-start.mem, and starts
`infinity_embed` again once the API is ready.

`start` stops the existing DeepSeek containers on both nodes, but preserves
those containers, their configuration, and all cached DeepSeek weights.
The unrelated embedding service and GitLab runner are not modified.
Do not run the DeepSeek launcher while GLM is running.

Official deployment documentation:
https://huggingface.co/nvidia/GLM-5.3-Flash-FP4-Local
