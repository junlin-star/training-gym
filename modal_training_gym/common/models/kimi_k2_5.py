from __future__ import annotations

import os
import subprocess
from pathlib import Path

from modal_training_gym.common.framework import TOOLS_REMOTE_PATH

from .base import HFModelConfiguration


class Kimi_K2_5(HFModelConfiguration):
    """Kimi-K2.5 model preset."""

    model_name = "moonshotai/Kimi-K2.5"
    model_path = "/checkpoints/Kimi-K2.5-bf16"
    int4_model_path = "/checkpoints/Kimi-K2.5-int4"

    @staticmethod
    def _patch_source_model(source_dir: Path) -> None:
        model_file = source_dir / "modeling_kimi_k25.py"
        if not model_file.is_file():
            raise FileNotFoundError(f"Expected Kimi source file at {model_file}")

        src = model_file.read_text()
        if "use_deterministic_attn: bool = False" in src:
            return

        ctor_old = (
            "    def __init__(\n"
            "        hidden_dim: int,\n"
            "        num_layers: int,\n"
            "        block_cfg: dict,\n"
            "        video_attn_type: str = 'spatial_temporal') -> None:\n"
        )
        ctor_new = (
            "    def __init__(\n"
            "        hidden_dim: int,\n"
            "        num_layers: int,\n"
            "        block_cfg: dict,\n"
            "        video_attn_type: str = 'spatial_temporal',\n"
            "        use_deterministic_attn: bool = False,\n"
            ") -> None:\n"
        )
        layer_old = (
            "            MoonViTEncoderLayer(\n"
            "                **block_cfg,\n"
            "                use_deterministic_attn=self.use_deterministic_attn)\n"
        )
        layer_new = (
            "            MoonViTEncoderLayer(\n"
            "                **block_cfg,\n"
            "                use_deterministic_attn=use_deterministic_attn)\n"
        )

        if ctor_old not in src or layer_old not in src:
            raise RuntimeError(
                "Unexpected Kimi source contents; the Kimi-K2.5 patch could not be applied cleanly."
            )

        src = src.replace(ctor_old, ctor_new, 1)
        src = src.replace(layer_old, layer_new, 1)
        model_file.write_text(src)

    def download(self) -> None:
        from huggingface_hub import snapshot_download

        source_dir = Path(snapshot_download(self.model_name, local_files_only=False))
        self._patch_source_model(source_dir)

        output_dir = Path(self.model_path or "/checkpoints/Kimi-K2.5-bf16")
        if output_dir.exists() and any(output_dir.iterdir()):
            object.__setattr__(self, "model_path", str(output_dir))
            return

        int4_dir = Path(self.int4_model_path)
        if not int4_dir.exists() or not any(int4_dir.iterdir()):
            int4_script = "/root/miles/tools/convert_hf_to_int4_direct.py"
            subprocess.run(
                [
                    "python",
                    int4_script,
                    "--model-dir",
                    str(source_dir),
                    "--save-dir",
                    str(int4_dir),
                    "--group-size",
                    os.environ.get("OPEN_TRAINING_INT4_GROUP_SIZE", "32"),
                ],
                check=True,
            )

        convert_script = f"{TOOLS_REMOTE_PATH}/convert_kimi_int4_to_bf16.py"
        os.makedirs(output_dir.parent, exist_ok=True)
        subprocess.run(
            [
                "python",
                convert_script,
                "--model-dir",
                str(int4_dir),
                "--output-dir",
                str(output_dir),
                "--overwrite",
            ],
            check=True,
        )
        object.__setattr__(self, "model_path", str(output_dir))
