"""Build and publish the slime and Miles base images as Modal named Images.

Named Images let launchers reference a prebuilt image with
``Image.from_name(...)`` (via the ``named_image`` recipe field) instead of
pulling the registry image and applying the built-in patches on every app
build. Referencing a name never triggers a rebuild, so training launches keep
using the last successfully published image.

Usage:

    uv run scripts/publish_framework_images.py                    # both frameworks
    uv run scripts/publish_framework_images.py --framework slime
    uv run scripts/publish_framework_images.py --framework miles --tag v1

Each publish writes ``<name>:latest`` plus an optional extra ``--tag``. The
default names are ``slime-base`` and ``miles-base``; pass ``--name-prefix`` to
publish under an ``environment/`` prefix. Publishing into a *public*
environment (so other workspaces can consume the name) additionally requires
a Modal admin identity and a globally deployed underlying image — that path
is gated server-side and not exposed here.
"""

import argparse

import modal

from modal_training_gym.frameworks.miles.launcher import _build_miles_base_image
from modal_training_gym.frameworks.slime.launcher import _build_slime_base_image
from modal_training_gym.train_recipes.miles_recipe import MilesRecipe

DEFAULT_NAMES = {"slime": "slime-base", "miles": "miles-base"}


def _base_image(framework: str) -> modal.Image:
    if framework == "slime":
        return _build_slime_base_image()
    # Only the launcher-level fields with defaults matter for the base image;
    # construct without validation since MilesRecipe requires training fields.
    miles = MilesRecipe.__new__(MilesRecipe)
    object.__setattr__(miles, "named_image", None)
    object.__setattr__(miles, "docker_image", MilesRecipe.docker_image)
    object.__setattr__(miles, "image_env", {})
    object.__setattr__(miles, "image_run_commands", [])
    return _build_miles_base_image(miles)


def publish(framework: str, *, name_prefix: str, tag: str | None) -> None:
    image = _base_image(framework)
    name = DEFAULT_NAMES[framework]
    ref = f"{name_prefix}/{name}" if name_prefix else name

    app = modal.App.lookup("training-gym-image-builds", create_if_missing=True)
    with modal.enable_output():
        built = image.build(app)
    built.publish(f"{ref}:latest")
    print(f"published {ref}:latest")
    if tag:
        built.publish(f"{ref}:{tag}")
        print(f"published {ref}:{tag}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", choices=["slime", "miles", "all"], default="all")
    parser.add_argument(
        "--tag", default=None, help="Extra tag to publish in addition to :latest"
    )
    parser.add_argument(
        "--name-prefix",
        default="",
        help="Optional 'environment' prefix for the published name",
    )
    args = parser.parse_args()

    frameworks = ["slime", "miles"] if args.framework == "all" else [args.framework]
    for framework in frameworks:
        publish(framework, name_prefix=args.name_prefix, tag=args.tag)


if __name__ == "__main__":
    main()
