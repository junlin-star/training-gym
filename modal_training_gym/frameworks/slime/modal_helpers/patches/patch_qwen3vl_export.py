"""Compat shim: make slime's MB->HF converter skip Qwen3-VL's vision tower.

slime's ``tools/convert_torch_dist_to_hf.py`` assumes a single decoder stack: for
any stacked ``.layers.`` param without an explicit index it asserts
``param.shape[0] == args.num_layers``. A VL checkpoint carries two stacks — the
language backbone (depth == num_layers) and the vision tower (a different depth) —
so the assertion blows up on the first ``vision_model.*`` param, before the
per-model ``convert_qwen3vl_to_hf`` even runs.

The vision tower is frozen during RL (see Qwen3VL_Recipe), so its trained weights
equal the base HF weights. We therefore skip ``vision_model.*`` in the converter
and let ``--add-missing-from-origin-hf`` fill the ViT + projector back from the
origin HF checkpoint. Only the trained language stack is converted from megatron.

Report upstream so this can be dropped. Idempotent. Run at image build:
    python patch_qwen3vl_export.py
"""

import pathlib

_MARKER = "PATCHED_VL_SKIP_VISION"
_CANDIDATES = [
    pathlib.Path("/root/slime/tools/convert_torch_dist_to_hf.py"),
]

_ANCHOR = '        name = f"module.module.{name}"\n'
_SKIP = (
    '        name = f"module.module.{name}"\n'
    "        if 'vision_model' in name:  # " + _MARKER + "\n"
    "            continue\n"
)


def main() -> None:
    target = next((p for p in _CANDIDATES if p.exists()), None)
    if target is None:
        print("compat: slime convert_torch_dist_to_hf.py not found; skipping")
        return
    src = target.read_text()
    if _MARKER in src:
        print("compat: qwen3_vl export vision-skip already applied")
        return
    if _ANCHOR not in src:
        print(
            "compat: WARNING - convert_torch_dist_to_hf get_named_params shape "
            "changed; skipping vision-skip patch"
        )
        return
    target.write_text(src.replace(_ANCHOR, _SKIP, 1))
    print("compat: patched slime MB->HF converter to skip Qwen3-VL vision tower")


if __name__ == "__main__":
    main()
