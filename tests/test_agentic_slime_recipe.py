from modal_training_gym import (
    Qwen3_6_27B,
    Qwen3_6_27b_Agentic_Recipe,
    Qwen3_6_27b_Recipe,
)


def test_agentic_recipe_inherits_qwen_preset_and_pins_fork() -> None:
    recipe = Qwen3_6_27b_Agentic_Recipe()

    assert isinstance(recipe, Qwen3_6_27b_Recipe)
    assert recipe.slime_git_repository == "https://github.com/modal-projects/slime.git"
    assert recipe.slime_git_revision == "ba324bebdd3a3cbfc1946b58404a012ad607f38b"
    assert recipe.data_volume_name == "slime-data"
    assert recipe.slime_model_script == "scripts/models/qwen3.5-27B.sh"
    assert recipe.sglang_speculative_algorithm == "EAGLE"


def test_agentic_recipe_emits_fork_rollout_configuration() -> None:
    recipe = Qwen3_6_27b_Agentic_Recipe()

    assert recipe.total_nodes == 6
    assert recipe.gpu_allocation.total_gpus == 48
    assert recipe.colocate is False
    assert recipe.pipeline_model_parallel_size == 1
    assert recipe.conversion_pipeline_model_parallel_size == 1
    assert recipe.context_parallel_size == 2
    assert recipe.extra_config["custom_generate_function_path"] == (
        "agentic_rl.generate.generate"
    )
    assert recipe.extra_config["training_gym_custom_rollout_log_function_path"] == (
        "agentic_rl.metrics.log_rollout_data"
    )
    fields = recipe._fields(model=Qwen3_6_27B())
    assert fields["save_debug_rollout_data"].endswith("/rollout_{rollout_id}.pt")
