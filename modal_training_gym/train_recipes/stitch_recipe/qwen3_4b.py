from dataclasses import field

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from modal_training_gym.deploy_recipes.sglang_recipe.recipe import SglangRecipe
from modal_training_gym.train_recipes.stitch_recipe.recipe import StitchRecipe
from modal_training_gym.train_recipes.stitch_recipe.serve import StitchServeRecipe
from modal_training_gym.train_recipes.stitch_recipe.train import SlimeStitchTrainRecipe


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class Qwen3_4b_Stitch_Train(SlimeStitchTrainRecipe):
    """1×8×H200 slime actor cluster for Qwen3-4B, publishing sparse deltas."""

    gpu_type: str = "H200"
    actor_num_nodes: int = 1
    actor_num_gpus_per_node: int = 8

    num_rollout: int = 3
    rollout_batch_size: int = 64
    rollout_max_response_len: int = 4096
    rollout_temperature: float = 1.0
    n_samples_per_prompt: int = 8
    global_batch_size: int = 128

    tensor_model_parallel_size: int = 1
    sequence_parallel: bool = False
    max_tokens_per_gpu: int = 9216
    lr: float = 1e-6

    rm_type: str | None = "math"
    save_interval: int = 20


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class Qwen3_4b_Stitch_Serve(StitchServeRecipe):
    """Flash pool of single-H200 SGLang replicas serving Qwen3-4B rollouts."""

    sglang: SglangRecipe = field(
        default_factory=lambda: SglangRecipe(
            gpu="H200",
            tp=1,
            context_length=16384,
            # An in-place delta apply loads the new shard alongside the live
            # weights, so the static pool has to leave that much GPU headroom on
            # a replica that is already serving at full concurrency.
            mem_fraction_static=0.80,
            chunked_prefill_size=4096,
            extra_server_args={
                # The no-GDS fastsafetensors path — hosts have no nvidia-fs, and
                # it is also the load format the sidecar reuses for each delta
                # reload.
                "--load-format": "fastsafetensors",
                "--model-loader-extra-config": '{"enable_gds":false}',
                "--weight-loader-drop-cache-after-load": "",
                "--reasoning-parser": "qwen3",
                "--max-prefill-tokens": "4096",
            },
        )
    )
    concurrency: int = 64
    min_containers: int = 3


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class Qwen3_4b_Stitch_Recipe(StitchRecipe):
    """Qwen3-4B disaggregated GRPO: 1×8×H200 trainer + Modal Flash rollout pool.

    Mirrors the stitch ``qwen3_4b_delta_flash`` cookbook config: publish-only
    disk-delta weight sync over a Modal Volume bulletin board, rollouts served
    by a warm Flash pool of single-GPU SGLang servers.

    Override either half in place, e.g. for stitch's ``…_hillclimb`` budget::

        Qwen3_4b_Stitch_Recipe(
            train=Qwen3_4b_Stitch_Train(num_rollout=120, eval_interval=20),
        )
    """

    train: SlimeStitchTrainRecipe = field(default_factory=Qwen3_4b_Stitch_Train)  # type: ignore[assignment]
    serve: StitchServeRecipe = field(default_factory=Qwen3_4b_Stitch_Serve)
