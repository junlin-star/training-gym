from __future__ import annotations

import shutil
from pathlib import Path

from .base import HFModelConfiguration, parse_kimi_k2_response


class Kimi_K2_5(HFModelConfiguration):
    response_parser = staticmethod(parse_kimi_k2_response)

    model_name = "moonshotai/Kimi-K2.5"

    @staticmethod
    def _transformers_module_name(checkpoint_path: str | Path) -> str:
        name = Path(checkpoint_path).name
        return name.replace("-", "_hyphen_").replace(".", "_dot_")

    @classmethod
    def _seed_dynamic_module_cache(cls, checkpoint_path: str | Path) -> None:
        checkpoint = Path(checkpoint_path)
        if not checkpoint.is_dir():
            return
        py_files = list(checkpoint.glob("*.py"))
        if not py_files:
            return
        module_dir = (
            Path.home()
            / ".cache"
            / "huggingface"
            / "modules"
            / "transformers_modules"
            / cls._transformers_module_name(checkpoint)
        )
        module_dir.mkdir(parents=True, exist_ok=True)
        for src in py_files:
            shutil.copy2(src, module_dir / src.name)
        init_file = module_dir / "__init__.py"
        if not init_file.exists():
            init_file.write_text("")
        print(f"Seeded Kimi dynamic module cache: {module_dir}")

    def prepare_runtime_cache(self) -> None:
        candidates = [
            self.model_path,
            "/checkpoints/Kimi-K2.5-int4",
            "/checkpoints/Kimi-K2.5-bf16",
        ]
        for candidate in candidates:
            if candidate:
                self._seed_dynamic_module_cache(candidate)

    def download(self) -> None:
        super().download()
        self.prepare_runtime_cache()
