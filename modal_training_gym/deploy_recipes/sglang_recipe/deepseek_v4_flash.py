from dataclasses import dataclass

from modal_training_gym.deploy_recipes.sglang_recipe.recipe import SglangRecipe

# Flags mirror the official SGLang cookbook for B200 · DeepSeek-V4-Flash · FP4:
# https://lmsysorg.mintlify.app/cookbook/autoregressive/DeepSeek/DeepSeek-V4
#
# Key points for this model on a single B200 node:
#   - Flash is an FP4-MoE checkpoint → use --moe-runner-backend flashinfer_mxfp4
#     (a tensor-parallel MoE kernel). Do NOT use --moe-a2a-backend deepep, which
#     needs NVSHMEM/IBGDA RDMA networking and hangs at expert-location init on a
#     single-node deployment without that fabric.
#   - Pure TP=4, no DP attention (DP attention triggers a CUDA-graph "Hidden size
#     mismatch" / short-circuit-allreduce assertion on a single node here).
#   - --disable-flashinfer-autotune and --swa-full-tokens-ratio 0.1 are required
#     by the cookbook for the hybrid (CSA/HCA + sliding-window) attention.
_DEEPSEEK_V4_FLASH_DEFAULTS = {
    "gpu": "B200",
    "tp": 4,
    "context_length": 262144,
    "mem_fraction_static": 0.80,
    "chunked_prefill_size": 4096,
    "max_running_requests": 16,
    "extra_server_args": {
        "--trust-remote-code": "",
        "--moe-runner-backend": "flashinfer_mxfp4",
        "--disable-flashinfer-autotune": "",
        "--swa-full-tokens-ratio": "0.1",
    },
}


_SGLANG_DEFAULTS = SglangRecipe()


@dataclass
class DeepSeek_V4_Flash_SglangRecipe(SglangRecipe):
    """DeepSeek-V4-Flash (284B MoE, 13B active) on 4×B200 — sensible SGLang defaults."""

    def __post_init__(self) -> None:
        for key, val in _DEEPSEEK_V4_FLASH_DEFAULTS.items():
            if getattr(self, key) == getattr(_SGLANG_DEFAULTS, key):
                object.__setattr__(self, key, val)
