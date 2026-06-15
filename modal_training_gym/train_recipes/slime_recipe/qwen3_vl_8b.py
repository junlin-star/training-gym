"""Qwen3-VL-8B recipe for vision-language GRPO on 1x8xH100."""

from dataclasses import field

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from modal_training_gym.train_recipes.slime_recipe.recipe import SlimeRecipe


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class Qwen3VL_Recipe(SlimeRecipe):
    """Qwen3-VL-8B vision-language GRPO on 1x8xH100, colocated.

    Carries the VL-specific defaults so a user only sets the reward:

      recipe = Qwen3VL_Recipe(custom_rm_function=my_reward)

    What's baked in and why:

    * ``use_dynamic_batch_size=False`` + ``qkv_format="bshd"`` + ``micro_batch_size=1``
      — VL models expand image patches into many tokens; padded (bshd) batches
      avoid the THD sequence packing path which may not be supported for VL.
    * ``sglang_mem_fraction_static=0.55`` — the frozen ViT encoder uses extra
      memory, so we give SGLang less static KV-cache allocation.
    * ``rollout_max_response_len=256`` — grounding predictions are short
      coordinate strings, not long chains of thought.
    * ``tensor_model_parallel_size=2`` — 8B model with vision encoder benefits
      from TP=2 on 8 GPUs.
    * ``megatron_to_hf_mode="bridge"`` — load HF -> Megatron through AutoBridge
      (which handles the VL checkpoint, incl. the ViT, with TP=2/PP=1). This skips
      slime's standalone ``convert_hf_to_torch_dist.py`` torch_dist pre-conversion,
      which assigns the VL model a pipeline stage (PP=2) the gym's conversion
      launcher doesn't expect (``world_size(2) % total_model_size(4) != 0``). Same
      reasoning as Qwen3_ASR_1_7b_Recipe.
    * ``freeze_params_name_list=["vision_model"]`` — freeze the ViT during RL. A
      short grounding run shouldn't perturb the pretrained visual features, and it
      cuts memory/compute. With the tower frozen its weights equal the base HF
      weights, so ``Qwen3VL_8B``'s export shim can identity-pass it on MB->HF
      export (see that model's ``compat_patches``).
    """

    gpu_type: str = "H100"
    colocate: bool = True
    tensor_model_parallel_size: int = 2
    sequence_parallel: bool = True
    rollout_num_gpus_per_engine: int = 1

    actor_num_nodes: int = 1
    actor_num_gpus_per_node: int = 8

    num_rollout: int = 15
    rollout_batch_size: int = 8
    n_samples_per_prompt: int = 4
    rollout_max_response_len: int = 256
    rollout_temperature: float = 1.0
    sglang_mem_fraction_static: float = 0.55

    global_batch_size: int = 16
    lr: float = 1e-6
    lr_decay_style: str = "constant"

    use_dynamic_batch_size: bool = False
    extra_config: dict | None = field(
        default_factory=lambda: {"qkv_format": "bshd", "micro_batch_size": 1}
    )

    save_interval: int = 10
    eval_interval: int | None = None

    # Load through megatron.bridge (AutoBridge) rather than slime's torch_dist
    # conversion tool — see class docstring.
    megatron_to_hf_mode: str = "bridge"

    # Freeze the vision tower; RL only updates the language backbone.
    freeze_params_name_list: list[str] | None = field(
        default_factory=lambda: ["vision_model"]
    )
