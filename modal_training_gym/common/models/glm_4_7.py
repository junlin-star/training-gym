"""GLM-4.7 (355B-A32B MoE) model spec as a concrete HFModelConfiguration subclass."""

import subprocess

from .base import HFModelConfiguration, ModelArchitecture, parse_glm_response

_TOOLS_PATH = "/opt/training-gym/tools"


def _disable_mtp_in_config(snapshot_dir: str) -> None:
    """Zero out ``num_nextn_predict_layers`` in config.json.

    The Slime model provider reads this field via the megatron-bridge
    ``AutoBridge.from_hf_pretrained`` path but does NOT override it from
    CLI args (only parallelism settings are overridden).  With PP > 1 the
    MTP embedding on the last pipeline stage collides with the main
    embedding on the first stage during ``broadcast_from_pp_rank``.
    Setting the field to 0 prevents the bridge from creating MTP layers.
    """
    import json
    import os

    cfg_path = os.path.join(snapshot_dir, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    if cfg.get("num_nextn_predict_layers", 0) == 0:
        return
    cfg["num_nextn_predict_layers"] = 0
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print("[glm_4_7] Patched config.json: num_nextn_predict_layers → 0")


class GLM_4_7(HFModelConfiguration):
    """GLM-4.7 (355B total, ~32B active) MoE model from Zhipu AI.

    Mixture-of-Experts with 160 routed experts + 1 shared expert,
    8 active per token. First 3 layers are dense.
    Pre-configured with base ``ModelArchitecture`` for Megatron-based
    frameworks (slime). Downloads from ``zai-org/GLM-4.7`` on HuggingFace.
    """

    response_parser = staticmethod(parse_glm_response)

    model_name = "zai-org/GLM-4.7"
    architecture = ModelArchitecture(
        num_layers=92,
        hidden_size=5120,
        ffn_hidden_size=12288,
        num_attention_heads=96,
        group_query_attention=True,
        num_query_groups=8,
        kv_channels=128,
        vocab_size=151552,
        normalization="RMSNorm",
        norm_epsilon=1e-5,
        swiglu=True,
        disable_bias_linear=False,
        qk_layernorm=True,
        untie_embeddings_and_output_weights=True,
        use_rotary_position_embeddings=True,
        rotary_base=1000000,
        num_experts=160,
        moe_ffn_hidden_size=1536,
        moe_shared_expert_intermediate_size=1536,
        moe_grouped_gemm=True,
        moe_shared_expert_gate=True,
        moe_router_topk=8,
        moe_router_score_function="softmax",
        moe_token_drop_policy="probs",
        moe_router_dtype="fp32",
        moe_permute_fusion=True,
        moe_aux_loss_coeff=0,
    )

    def download(self) -> None:
        from huggingface_hub import snapshot_download

        snapshot_dir = snapshot_download(repo_id=self.model_name)
        subprocess.check_call(
            [
                "python3",
                f"{_TOOLS_PATH}/ensure_glm_tokenizer.py",
                "--snapshot-dir",
                snapshot_dir,
            ],
        )
        _disable_mtp_in_config(snapshot_dir)
