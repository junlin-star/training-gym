from modal_training_gym.train_recipes.slime_recipe.recipe import SlimeRecipe


_RECIPE_KW = {
    "gpu_type": "H100",
    "colocate": True,
    "tensor_model_parallel_size": 1,
    "sequence_parallel": False,
    "rollout_num_gpus_per_engine": 1,
    "num_rollout": 1,
    "rollout_batch_size": 1,
    "rollout_max_response_len": 32,
    "rollout_temperature": 1.0,
    "save_interval": 1,
}


def _flag_value(args: list[str], flag: str) -> str | None:
    try:
        return args[args.index(flag) + 1]
    except ValueError:
        return None


def test_colocated_recipe_disables_incompatible_prefill_graph() -> None:
    args = SlimeRecipe(**_RECIPE_KW).cli_args()

    assert _flag_value(args, "--sglang-cuda-graph-backend-prefill") == "disabled"


def test_non_colocated_recipe_keeps_sglang_default() -> None:
    args = SlimeRecipe(
        **{**_RECIPE_KW, "colocate": False, "rollout_num_gpus": 8}
    ).cli_args()

    assert "--sglang-cuda-graph-backend-prefill" not in args


def test_explicit_prefill_graph_backend_is_preserved() -> None:
    args = SlimeRecipe(
        **_RECIPE_KW,
        sglang_cuda_graph_backend_prefill="tc_piecewise",
    ).cli_args()

    assert _flag_value(args, "--sglang-cuda-graph-backend-prefill") == "tc_piecewise"


def test_extra_config_prefill_graph_backend_wins() -> None:
    args = SlimeRecipe(
        **_RECIPE_KW,
        extra_config={"sglang_cuda_graph_backend_prefill": "tc_piecewise"},
    ).cli_args()

    assert "--sglang-cuda-graph-backend-prefill" not in args
    assert "--custom-config-path" in args
