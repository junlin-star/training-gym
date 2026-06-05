from dataclasses import field
from typing import Callable

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from modal_training_gym.rollouts import transcription_rollout
from modal_training_gym.train_recipes.slime_recipe.recipe import SlimeRecipe


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class Qwen3ASR_Recipe(SlimeRecipe):
    """Qwen3-ASR-1.7B audio GRPO on 1×2×H100, colocated.

    Carries the ASR-specific defaults so a user only sets the reward (and W&B):

      recipe = Qwen3ASR_Recipe(custom_rm_function=wer_reward, wandb=...)

    What's baked in and why:

    * ``custom_generate_function=transcription_rollout`` — Qwen3-ASR is served by
      SGLang on ``/v1/audio/transcriptions`` (never chat completions), so it must
      be driven through the gym's audio-transcription rollout.
    * ``use_dynamic_batch_size=False`` + ``qkv_format="bshd"`` + ``micro_batch_size=1``
      — the native megatron-bridge Qwen3-ASR forward doesn't implement THD
      sequence packing ("packed_seq_params is not supported"). slime only builds
      packed_seq_params when qkv_format="thd" (its default), so padded "bshd"
      batches sidestep the unsupported path. bshd requires dynamic batching off +
      an explicit micro_batch_size. The launcher enforces this (model.requires_bshd).
    * ``sglang_mem_fraction_static=0.45`` — audio conditioning makes prompts much
      longer (expanded <audio_pad>) and adds the frozen audio tower, so free SGLang
      memory to give the colocated actor headroom (text-only runs use ~0.78).
    * Many samples/prompt + temperature 1.0 — Qwen3-ASR is near-deterministic on
      clean LibriSpeech, so this is how the GRPO group gets nonzero reward variance.

    Scale to a full 8×H100 node by passing ``actor_num_gpus_per_node=8`` (and a
    larger ``num_rollout``); everything else holds.
    """

    gpu_type: str = "H100"
    colocate: bool = True
    tensor_model_parallel_size: int = 1
    sequence_parallel: bool = False
    rollout_num_gpus_per_engine: int = 1

    actor_num_nodes: int = 1
    actor_num_gpus_per_node: int = 2

    # Qwen3-ASR is served on /v1/audio/transcriptions — always drive it through the
    # gym's transcription rollout rather than slime's default chat-completions one.
    custom_generate_function: Callable | None = transcription_rollout

    num_rollout: int = 8
    rollout_batch_size: int = 4
    n_samples_per_prompt: int = 8
    rollout_max_response_len: int = 128
    rollout_temperature: float = 1.0
    sglang_mem_fraction_static: float = 0.45

    global_batch_size: int = 8
    lr: float = 1e-6
    lr_decay_style: str = "cosine"

    # Padded (bshd) batches — required (see class docstring); enforced by the launcher.
    use_dynamic_batch_size: bool = False
    extra_config: dict | None = field(
        default_factory=lambda: {"qkv_format": "bshd", "micro_batch_size": 1}
    )

    # Save at the final rollout so the run produces a checkpoint to export to HF.
    save_interval: int = 8
