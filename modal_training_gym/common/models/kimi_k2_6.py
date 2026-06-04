from __future__ import annotations

from .kimi_k2_5 import Kimi_K2_5


class Kimi_K2_6(Kimi_K2_5):
    model_name = "moonshotai/Kimi-K2.6"

    def prepare_runtime_cache(self) -> None:
        candidates = [
            self.model_path,
            "/checkpoints/Kimi-K2.6-int4",
            "/checkpoints/Kimi-K2.6-bf16",
        ]
        for candidate in candidates:
            if candidate:
                self._seed_dynamic_module_cache(candidate)
