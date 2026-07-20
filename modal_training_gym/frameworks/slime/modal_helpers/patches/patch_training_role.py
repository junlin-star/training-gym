from pathlib import Path


def _patch_file(path: Path) -> None:
    if not path.exists():
        print(f"WARNING: {path} not found, skipping training-role patch")
        return

    source = path.read_text()
    assignment = "        self.args.training_gym_role = role\n"
    if assignment in source:
        return
    target = "        self.role = role\n"
    if target not in source:
        print(f"WARNING: Could not patch {path} with training role")
        return
    path.write_text(source.replace(target, target + assignment, 1))


if __name__ == "__main__":
    _patch_file(Path("/root/slime/slime/ray/train_actor.py"))
