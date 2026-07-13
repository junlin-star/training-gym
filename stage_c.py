"""Launch MegaGem Stage C on Training Gym.

Example production launch:

    MODAL_ENVIRONMENT=alex-dev .venv/bin/python stage_c.py --detach
"""

from __future__ import annotations

import argparse
import json

from modal_training_gym import (
    MegaGemStageCDataset,
    MegaGem_Qwen3_4B_SFT,
    MegaGem_Qwen3_4B_StageC_Recipe,
    TrainConfig,
)
from modal_training_gym.train_recipes.slime_recipe.megagem_qwen3_4b_stage_c import (
    megagem_stage_c_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MegaGem Stage C: 8 actor GPUs + 32 rollout GPUs on H200."
    )
    parser.add_argument("--num-rollout", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=10000)
    parser.add_argument("--num-prompts", type=int, default=128)
    parser.add_argument("--rollout-batch-size", type=int, default=128)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--rows-per-group", type=int, default=16)
    parser.add_argument(
        "--rotate-seats",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Round-robin the trainable seat across prompt groups.",
    )
    parser.add_argument("--save-interval", type=int, default=25)
    parser.add_argument("--lr", type=float, default=6e-6)
    parser.add_argument("--min-lr", type=float, default=6e-7)
    parser.add_argument("--kl-loss-coef", type=float, default=0.05)
    parser.add_argument("--rollout-max-response-len", type=int, default=1024)
    parser.add_argument("--rollout-temperature", type=float, default=0.85)
    parser.add_argument("--rollout-top-p", type=float, default=1.0)
    parser.add_argument("--global-batch-size", type=int, default=2048)
    parser.add_argument("--max-parallel-games", type=int, default=256)
    parser.add_argument("--extra-games-per-group", type=int, default=4)
    parser.add_argument("--min-success-games", type=int, default=12)
    parser.add_argument("--game-timeout-s", type=float, default=600.0)
    parser.add_argument("--length-penalty-target-tokens", type=int, default=512)
    parser.add_argument("--length-penalty-per-100-tokens", type=float, default=0.01)
    parser.add_argument("--length-penalty-cap", type=float, default=0.20)
    parser.add_argument(
        "--fail-open-groups",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use zero-advantage fallback rows when a prompt group fails. "
            "Disabled by default so rollout/import/API failures stop Stage C."
        ),
    )
    parser.add_argument("--wandb-project", default="megagem-stage-c")
    parser.add_argument("--wandb-group", default="stage-c-stability")
    parser.add_argument(
        "--wandb-secret",
        default="wandb-secret",
        help="Modal secret containing WANDB_API_KEY.",
    )
    parser.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable W&B logging for Slime/Megatron train metrics.",
    )
    parser.add_argument(
        "--detach",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Return after remote launch; the Modal training call keeps running.",
    )
    parser.add_argument(
        "--wait",
        dest="detach",
        action="store_false",
        help="Wait for the remote training call instead of returning after launch.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = MegaGem_Qwen3_4B_SFT()
    dataset = MegaGemStageCDataset(
        num_prompts=args.num_prompts,
        seed_start=args.seed_start,
        k=args.k,
        rows_per_group=args.rows_per_group,
        rotate_seats=args.rotate_seats,
    )
    recipe = MegaGem_Qwen3_4B_StageC_Recipe(
        num_rollout=args.num_rollout,
        save_interval=args.save_interval,
        lr=args.lr,
        min_lr=args.min_lr,
        kl_loss_coef=args.kl_loss_coef,
        rollout_max_response_len=args.rollout_max_response_len,
        eval_max_response_len=args.rollout_max_response_len,
        rollout_temperature=args.rollout_temperature,
        rollout_top_p=args.rollout_top_p,
        rollout_batch_size=args.rollout_batch_size,
        global_batch_size=args.global_batch_size,
    )
    recipe.extra_config = {
        **(recipe.extra_config or {}),
        "megagem_max_parallel_games": args.max_parallel_games,
        "megagem_extra_games_per_group": args.extra_games_per_group,
        "megagem_min_success_games": args.min_success_games,
        "megagem_game_timeout_s": args.game_timeout_s,
        "megagem_fail_open_groups": args.fail_open_groups,
        "megagem_length_penalty_target_tokens": args.length_penalty_target_tokens,
        "megagem_length_penalty_per_100_tokens": args.length_penalty_per_100_tokens,
        "megagem_length_penalty_cap": args.length_penalty_cap,
    }
    if args.wandb:
        from modal_training_gym import WandbConfig

        recipe.wandb = WandbConfig(
            project=args.wandb_project,
            group=args.wandb_group,
            modal_wandb_secret_name=args.wandb_secret,
        )

    print(json.dumps(megagem_stage_c_summary(recipe), indent=2))
    training_run = TrainConfig(
        model=model,
        dataset=dataset,
        recipe=recipe,
        detach=args.detach,
    )
    if args.detach:
        launched = training_run.launch(prepare_inputs=True)
        print(
            "Launched Stage C: "
            f"training_run_id={launched.training_run_id} "
            f"function_call={launched.function_call_id}"
        )
    else:
        result = training_run.train()
        print(f"Training finished: {result.training_run_id}")


if __name__ == "__main__":
    main()
