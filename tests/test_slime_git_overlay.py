import pytest

from modal_training_gym.frameworks.slime.launcher import _slime_git_overlay_command
from modal_training_gym.train_recipes.slime_recipe import Qwen3_6_27b_Recipe


REPOSITORY = "https://github.com/modal-projects/slime.git"
REVISION = "ba324bebdd3a3cbfc1946b58404a012ad607f38b"


def test_slime_git_overlay_requires_repository_and_full_revision() -> None:
    with pytest.raises(ValueError, match="must be set together"):
        Qwen3_6_27b_Recipe(slime_git_repository=REPOSITORY)

    with pytest.raises(ValueError, match="full 40-character"):
        Qwen3_6_27b_Recipe(
            slime_git_repository=REPOSITORY,
            slime_git_revision="ba324beb",
        )


def test_slime_git_overlay_rejects_local_or_credentialed_sources() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        Qwen3_6_27b_Recipe(
            local_slime="/tmp/slime",
            slime_git_repository=REPOSITORY,
            slime_git_revision=REVISION,
        )

    with pytest.raises(ValueError, match="must not contain credentials"):
        Qwen3_6_27b_Recipe(
            slime_git_repository="https://token@example.com/slime.git",
            slime_git_revision=REVISION,
        )


def test_slime_git_overlay_command_is_revision_pinned() -> None:
    recipe = Qwen3_6_27b_Recipe(
        slime_git_repository=REPOSITORY,
        slime_git_revision=REVISION.upper(),
    )

    assert recipe.slime_git_revision == REVISION
    assert "fetch --depth=1 origin" in _slime_git_overlay_command(
        recipe.slime_git_repository or "", recipe.slime_git_revision or ""
    )
    assert REVISION in _slime_git_overlay_command(
        recipe.slime_git_repository or "", recipe.slime_git_revision or ""
    )
    assert "checkout --detach FETCH_HEAD" in _slime_git_overlay_command(
        recipe.slime_git_repository or "", recipe.slime_git_revision or ""
    )
