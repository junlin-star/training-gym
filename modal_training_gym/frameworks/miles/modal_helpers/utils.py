import os
import shlex
from os import PathLike

_CONVERSION_EXTRA_ARGS = [
    ("decoder_first_pipeline_num_layers", "decoder-first-pipeline-num-layers"),
    ("decoder_last_pipeline_num_layers", "decoder-last-pipeline-num-layers"),
    ("mtp_num_layers", "mtp-num-layers"),
    ("make_vocab_size_divisible_by", "make-vocab-size-divisible-by"),
]


def is_local_checkpoint_ref(ref: str | PathLike) -> bool:
    return str(ref).startswith("/")


def resolve_checkpoint_ref(
    ref: str | PathLike,
    *,
    local_files_only: bool = True,
) -> str:
    ref_str = str(ref)
    if is_local_checkpoint_ref(ref_str):
        return ref_str

    from huggingface_hub import snapshot_download

    return snapshot_download(ref_str, local_files_only=local_files_only)


def get_checkpoint_conversion_policy(
    miles_cfg, model=None
) -> tuple[int, int, list[str]]:
    gpus_per_node = getattr(miles_cfg, "actor_num_gpus_per_node", 8)
    actor_nodes = getattr(miles_cfg, "actor_num_nodes", 1)
    tp = getattr(miles_cfg, "tensor_model_parallel_size", 1)
    pp = getattr(miles_cfg, "pipeline_model_parallel_size", 1)
    ep = getattr(miles_cfg, "expert_model_parallel_size", 1) or 1
    etp = getattr(miles_cfg, "expert_tensor_parallel_size", 1) or 1

    # When EP>1 we must convert with the full training parallelism so
    # Megatron doesn't attempt re-sharding at load time.
    if ep > 1:
        pp = 1
        world_size = actor_nodes * gpus_per_node
    else:
        world_size = tp * pp if (tp > 1 or pp > 1) else gpus_per_node
    max_world_size = actor_nodes * gpus_per_node
    if world_size > max_world_size:
        raise ValueError(
            f"checkpoint conversion world_size={world_size} exceeds actor cluster capacity "
            f"{actor_nodes}x{gpus_per_node}={max_world_size}"
        )

    for num_nodes in range(1, actor_nodes + 1):
        if world_size % num_nodes != 0:
            continue
        nproc_per_node = world_size // num_nodes
        if nproc_per_node > gpus_per_node:
            continue

        extra_args: list[str] = []
        # Always pass parallelism flags to override any values baked
        # into the model script's MODEL_ARGS.
        if tp > 1 or pp > 1 or ep > 1:
            extra_args += [
                f"--tensor-model-parallel-size {tp}",
                f"--pipeline-model-parallel-size {pp}",
            ]
        if ep > 1:
            extra_args += [
                f"--expert-model-parallel-size {ep}",
                f"--expert-tensor-parallel-size {etp}",
                # Prevent Megatron/spec from auto-inflating PP via MTP
                f"--transformer-pipeline-model-parallel-size {pp}",
            ]
        for attr, flag in _CONVERSION_EXTRA_ARGS:
            x = getattr(miles_cfg, attr, None)
            if x is not None:
                extra_args.append(f"--{flag} {x}")

        if model and getattr(model, "architecture", None):
            arch = model.architecture
            for attr, flag in [
                ("num_layers", "num-layers"),
                ("hidden_size", "hidden-size"),
                ("ffn_hidden_size", "ffn-hidden-size"),
                ("num_attention_heads", "num-attention-heads"),
                ("num_query_groups", "num-query-groups"),
                ("kv_channels", "kv-channels"),
                ("vocab_size", "vocab-size"),
                ("norm_epsilon", "norm-epsilon"),
                ("rotary_base", "rotary-base"),
                ("num_experts", "num-experts"),
                ("moe_ffn_hidden_size", "moe-ffn-hidden-size"),
                (
                    "moe_shared_expert_intermediate_size",
                    "moe-shared-expert-intermediate-size",
                ),
                ("moe_router_topk", "moe-router-topk"),
            ]:
                val = getattr(arch, attr, 0)
                if val:
                    extra_args.append(f"--{flag} {val}")
            if arch.group_query_attention:
                extra_args.append("--group-query-attention")
            if arch.swiglu:
                extra_args.append("--swiglu")
            if arch.disable_bias_linear:
                extra_args.append("--disable-bias-linear")
            if arch.qk_layernorm:
                extra_args.append("--qk-layernorm")
            if arch.untie_embeddings_and_output_weights:
                extra_args.append("--untie-embeddings-and-output-weights")
            if arch.normalization and arch.normalization != "LayerNorm":
                extra_args.append(f"--normalization {arch.normalization}")
            if arch.use_rotary_position_embeddings:
                extra_args.append("--use-rotary-position-embeddings")
                extra_args.append("--position-embedding-type rope")
            if getattr(arch, "moe_grouped_gemm", False):
                extra_args.append("--moe-grouped-gemm")
            if getattr(arch, "moe_shared_expert_gate", False):
                extra_args.append("--moe-shared-expert-gate")
            for str_attr, str_flag in [
                ("moe_router_score_function", "moe-router-score-function"),
                ("moe_token_drop_policy", "moe-token-drop-policy"),
                ("moe_router_dtype", "moe-router-dtype"),
            ]:
                val = getattr(arch, str_attr, "")
                if val:
                    extra_args.append(f"--{str_flag} {val}")
            if getattr(arch, "moe_permute_fusion", False):
                extra_args.append("--moe-permute-fusion")
            moe_aux = getattr(arch, "moe_aux_loss_coeff", None)
            if moe_aux is not None:
                extra_args.append(f"--moe-aux-loss-coeff {moe_aux}")
            if getattr(arch, "apply_layernorm_1p", False):
                extra_args.append("--apply-layernorm-1p")

        return num_nodes, nproc_per_node, extra_args

    raise ValueError(
        f"cannot find checkpoint conversion layout for world_size={world_size} "
        f"with actor_num_nodes={actor_nodes}, actor_num_gpus_per_node={gpus_per_node}"
    )


def prepare_miles_config(miles_cfg, model, tmpdir: str) -> None:
    import yaml

    from modal_training_gym.train_recipes.miles_recipe.recipe import YAML_CONFIG_FIELDS

    if (
        model
        and not model.model_path
        and model.model_name
        and not str(model.model_name).startswith("/")
    ):
        model.model_path = resolve_checkpoint_ref(model.model_name)

    for attr in ("hf_checkpoint", "load", "ref_load", "critic_load"):
        if val := getattr(miles_cfg, attr, None):
            object.__setattr__(miles_cfg, attr, resolve_checkpoint_ref(val))

    for field in YAML_CONFIG_FIELDS:
        if isinstance(val := getattr(miles_cfg, field, None), dict):
            path = os.path.join(tmpdir, f"{field}.yaml")
            with open(path, "w") as f:
                yaml.dump(val, f)
            print(f"Materialized {field} -> {path}")
            object.__setattr__(miles_cfg, field, path)


def build_train_cmd(miles_cfg, miles_root: str, model=None, dataset=None) -> str:
    train_script = (
        f"{miles_root}/{'train_async.py' if miles_cfg.async_mode else 'train.py'}"
    )
    cli_args = shlex.join(miles_cfg.cli_args(dataset=dataset, model=model))
    if miles_cfg.miles_model_script:
        inner = (
            f"source {miles_root}/{miles_cfg.miles_model_script} && "
            f"python3 {train_script} ${{MODEL_ARGS[@]}} {cli_args}"
        )
        return f"bash -c {shlex.quote(inner)}"
    return f"python3 {train_script} {cli_args}"
