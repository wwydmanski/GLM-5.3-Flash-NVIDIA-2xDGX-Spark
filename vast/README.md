# vast.ai template: GLM-5.3-Flash-FP4-Local on 2x RTX PRO 6000

One TP=2 vLLM replica on two RTX PRO 6000 Blackwell (96 GB each). It uses the NVIDIA
model-card kernel settings for this card, plus the MTP setup we validated on
real staging traffic.

| Field | Value |
|---|---|
| Image | `ghcr.io/local-inference-lab/vllm:release-10152026` (x86, sm_120a, public) |
| Launch mode | SSH (direct), so the on-start script runs |
| Docker options | contents of [`docker-options.txt`](docker-options.txt) (port 8000 + 60 `-e` kernel variables) |
| On-start script | contents of [`onstart.sh`](onstart.sh) |
| Disk | 300 GB (model 166 GB + JIT/compile caches) |
| GPUs | 2x RTX PRO 6000 on one host, CUDA driver >= 12.8 |
| Optional env | `VLLM_API_KEY` (strongly recommended: vast exposes port 8000 publicly), `HF_TOKEN` |

## Create it

In the UI: Templates → New, paste the fields above. Or with the CLI (`pip install vastai`,
`vastai set api-key ...`):

```sh
cd vast
vastai create template --name "GLM-5.3-Flash 2x RTX PRO 6000 (MTP3 prob)" \
  --image ghcr.io/local-inference-lab/vllm --image_tag release-10152026 \
  --env "$(cat docker-options.txt)" --onstart-cmd "$(cat onstart.sh)" \
  --disk_space 300 --ssh --direct \
  --search_params "num_gpus=2 gpu_name=RTX_PRO_6000_WS cuda_max_good>=12.8 disk_space>=300"
```

Check the exact `gpu_name` for the card first: `vastai search offers 'num_gpus=2' | grep -i "PRO 6000"`
(server and workstation editions are listed separately; both work).

To set the API key, add `-e VLLM_API_KEY=<secret>` to the instance's Docker options when renting
(do not bake it into a shared template).

## After start

The model download takes ~10-20 min, loading and kernel compilation another ~10 min.

```sh
tail -f /workspace/glm/download.log /workspace/glm/vllm.log
curl -s localhost:8000/v1/models
```

Externally: `http://<public ip>:<mapped port for 8000>/v1` (see the instance's IP & port info).
OpenAI-compatible; reasoning parser `glm45`, tool calling `glm47`.

## MTP settings and why

Measured by replaying real staging requests (Polish legal assistant traffic), first 100 requests:

| Setup | decode tok/s | accepted per position |
|---|---|---|
| MTP2, greedy draft, NVFP4 draft head (model card) | 37.0 | 0.76 / 0.56 |
| MTP2, probabilistic draft, BF16 draft head | 40.0 | 0.86 / 0.69 |
| **MTP3, probabilistic draft, BF16 draft head** | **42.0** | 0.86 / 0.69 / 0.54 |
| MTP4, same | 41.8 | 0.85 / 0.67 / 0.51 / 0.39 |
| MTP5, same | ~39.5-40.0 | 0.84 / 0.65 / 0.48 / 0.37 / ... |

(Numbers from the 2x DGX Spark deployment; the relative ordering is what carries over.)

- `draft_sample_method: probabilistic` + `rejection_sample_method: standard`: the draft samples from
  its distribution instead of taking argmax. That gives the biggest gain (+10 pt acceptance at position 1).
- BF16 draft head (`VLLM_MTP_NVFP4_LM_HEAD=0`, `VLLM_GLM53_MTP_DRAFT_HEAD=bf16`): the NVFP4-quantized
  draft lm_head costs acceptance.
- `num_speculative_tokens: 3`: past 3 drafts, the extra verify cost outweighs the extra accepted tokens.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`: avoids an out-of-memory peak while b12x
  prepares the expert scales after loading.
- Fine-tuning the MTP layer on our traffic is not worth it: best result +1.7% tokens/step
  (TV loss; soft-CE makes acceptance worse). Use the stock MTP weights.

`--max-num-seqs 4` with a 7.1 GiB/GPU KV budget at 131k context is the memory-safe setting we ran
on these cards; raise `--kv-cache-memory-bytes` / `--max-num-seqs` only after checking free VRAM.
