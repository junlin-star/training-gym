from __future__ import annotations

from .base import HFModelConfiguration, parse_kimi_k2_response


class Kimi_K2_5(HFModelConfiguration):
    response_parser = staticmethod(parse_kimi_k2_response)

    model_name = "moonshotai/Kimi-K2.5"
    model_path = "/checkpoints/Kimi-K2.5-int4"

    def download(self) -> None:
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=self.model_name, local_dir=self.model_path)
