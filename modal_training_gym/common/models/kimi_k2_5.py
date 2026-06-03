from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

from .base import HFModelConfiguration


class Kimi_K2_5(HFModelConfiguration):
    """Kimi-K2.5 model preset with Miles INT4/BF16 checkpoint preparation."""

    model_name = "moonshotai/Kimi-K2.5"
    model_path = "/checkpoints/Kimi-K2.5-bf16"
    int4_model_path = "/checkpoints/Kimi-K2.5-int4"

    @staticmethod
    def _patch_source_model(source_dir: str) -> None:
        """Patch Kimi source code for deterministic attention construction."""
        modeling_file = Path(source_dir) / "modeling_kimi_k25.py"
        if not modeling_file.exists():
            return

        source = modeling_file.read_text()
        patched = source.replace(
            "self.attn = KimiK2Attention(",
            "self.attn = KimiK2Attention(layer_idx=layer_idx, ",
        )
        patched = patched.replace(
            "self.layer_idx = layer_idx\n        self.q_proj",
            "self.layer_idx = layer_idx\n        self.layer_idx = layer_idx\n        self.q_proj",
        )
        if patched != source:
            modeling_file.write_text(patched)

    def download(self) -> str:
        source_dir = snapshot_download(self.model_name)
        self._patch_source_model(source_dir)

        if Path(self.model_path).exists():
            return self.model_path

        int4_path = Path(self.int4_model_path)
        if not int4_path.exists():
            env = os.environ.copy()
            env.setdefault("OPEN_TRAINING_INT4_GROUP_SIZE", "32")
            subprocess.run(
                [
                    sys.executable,
                    "/root/miles/tools/convert_hf_to_int4_direct.py",
                    "--model-dir",
                    source_dir,
                    "--save-dir",
                    self.int4_model_path,
                ],
                check=True,
                env=env,
            )

        subprocess.run(
            [
                sys.executable,
                "/opt/training-gym/tools/convert_kimi_int4_to_bf16.py",
                "--model-dir",
                self.int4_model_path,
                "--output-dir",
                self.model_path,
            ],
            check=True,
        )
        return self.model_path
