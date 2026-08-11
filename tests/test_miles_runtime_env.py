"""Ray runtime env construction for miles jobs."""

from __future__ import annotations

from modal_training_gym.frameworks.miles import launcher
from modal_training_gym.frameworks.miles.launcher import build_ray_runtime_env
from modal_training_gym.train_recipes.miles_recipe.recipe import MilesRecipe


def test_ld_library_path_comes_from_the_container(monkeypatch):
    """Workers get the container's linker path, behind the system lib dir."""
    monkeypatch.setenv("LD_LIBRARY_PATH", "/usr/local/cuda/lib64:/wheel/nvidia/lib")

    env_vars = build_ray_runtime_env(
        head_addr="10.0.0.1", wandb_env={}, environment={}
    )["env_vars"]

    assert env_vars["LD_LIBRARY_PATH"] == (
        "/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:/wheel/nvidia/lib"
    )
    assert env_vars["MASTER_ADDR"] == "10.0.0.1"
    assert env_vars["no_proxy"] == "127.0.0.1,10.0.0.1"


def test_recipe_environment_still_wins(monkeypatch):
    monkeypatch.setenv("LD_LIBRARY_PATH", "/from/container")

    env_vars = build_ray_runtime_env(
        head_addr="10.0.0.1",
        wandb_env={},
        environment={
            "LD_LIBRARY_PATH": "/from/recipe",
            "PYTHONPATH": "/root/Megatron-LM/",
        },
    )["env_vars"]

    assert env_vars["LD_LIBRARY_PATH"] == "/from/recipe"
    assert env_vars["PYTHONPATH"] == "/root/Megatron-LM/"


def test_unset_container_path_yields_only_the_system_lib_dir(monkeypatch):
    """No empty entry, which the loader would read as the working directory."""
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)

    env_vars = build_ray_runtime_env(
        head_addr="10.0.0.1", wandb_env={}, environment={}
    )["env_vars"]

    assert env_vars["LD_LIBRARY_PATH"] == "/usr/lib/x86_64-linux-gnu"


def test_system_lib_dir_is_not_duplicated(monkeypatch):
    monkeypatch.setenv("LD_LIBRARY_PATH", "/usr/lib/x86_64-linux-gnu:/wheel/nvidia/lib")

    env_vars = build_ray_runtime_env(
        head_addr="10.0.0.1", wandb_env={}, environment={}
    )["env_vars"]

    assert env_vars["LD_LIBRARY_PATH"] == (
        "/usr/lib/x86_64-linux-gnu:/wheel/nvidia/lib"
    )


def test_wandb_env_is_preserved(monkeypatch):
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)

    env_vars = build_ray_runtime_env(
        head_addr="10.0.0.1",
        wandb_env={"WANDB_RUN_ID": "abc", "WANDB_RESUME": "allow"},
        environment={},
    )["env_vars"]

    assert env_vars["WANDB_RUN_ID"] == "abc"
    assert env_vars["WANDB_RESUME"] == "allow"


def test_substep_timing_is_forwarded_from_recipe(monkeypatch):
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    recipe = MilesRecipe(substep_timing="require")

    env_vars = build_ray_runtime_env(
        head_addr="10.0.0.1",
        wandb_env={},
        environment={},
        substep_timing=recipe.substep_timing,
    )["env_vars"]

    assert env_vars["TRAINING_GYM_SUBSTEP_TIMING"] == "require"


def test_pinned_image_patch_commands_keep_non_timing_patches_strict(monkeypatch):
    commands = []

    class FakeImage:
        @classmethod
        def from_registry(cls, image):
            return cls()

        def entrypoint(self, command):
            return self

        def run_commands(self, *values):
            commands.extend(values)
            return self

    monkeypatch.setattr(launcher, "Image", FakeImage)
    launcher._build_miles_base_image(MilesRecipe(substep_timing="require"))

    assert "TG_BEST_EFFORT_ENTRYPOINTS=1" not in commands[1]
    assert all(
        "TG_BEST_EFFORT_ENTRYPOINTS=1" not in command for command in commands[2:4]
    )
    assert "TRAINING_GYM_SUBSTEP_TIMING=require" in commands[4]
    assert "TG_BEST_EFFORT_ENTRYPOINTS=1" not in commands[4]


def test_pinned_image_timing_patch_is_best_effort_in_auto(monkeypatch):
    commands = []

    class FakeImage:
        @classmethod
        def from_registry(cls, image):
            return cls()

        def entrypoint(self, command):
            return self

        def run_commands(self, *values):
            commands.extend(values)
            return self

    monkeypatch.setattr(launcher, "Image", FakeImage)
    launcher._build_miles_base_image(MilesRecipe(substep_timing="auto"))

    assert "TG_BEST_EFFORT_ENTRYPOINTS=1" in commands[4]
    assert "TRAINING_GYM_SUBSTEP_TIMING=auto" in commands[4]
