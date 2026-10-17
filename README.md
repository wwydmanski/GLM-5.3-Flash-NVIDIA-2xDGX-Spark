# GLM 5.3 Flash on the two GX10 nodes

Run on gx10-a55a (both nodes must be booted and reachable):

```sh
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

API: http://gx10-a55a:8000/v1
Model: nvidia/GLM-5.3-Flash-FP4-Local
Context: exactly 131000 tokens, FP8 KV, 4 concurrent sequences.
KV cache budget: 4 GiB per GPU (config.sh), not all free unified RAM.
The context is per request, the KV pool is shared among concurrent requests.

The NVIDIA model-card two-Spark B12X profile is used, including MTP5.
Files: node.sh (launch), runtime.env (kernel/network settings), config.sh
(image/network/KV budget), glm (two-node lifecycle).

`start` stops the existing DeepSeek containers on both nodes, but preserves
those containers, their configuration, and all cached DeepSeek weights.
The unrelated embedding service and GitLab runner are not modified.
Do not run the DeepSeek launcher while GLM is running.

Official deployment documentation:
https://huggingface.co/nvidia/GLM-5.3-Flash-FP4-Local
