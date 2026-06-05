"""Compatibility shims for running Qwen3-ASR on upstream training-gym's NATIVE stack
(slime nightly: sglang 0.5.12 + megatron-bridge 0.5.0 + transformers 5.6).

Two small UPSTREAM gaps block Qwen3-ASR out of the box. Both are idempotent in-place
fixes applied once at image build. Each should be reported upstream so this entire
file can eventually be deleted (then the example is truly patch-free):

  1) megatron-bridge 0.5.0 `hf_qwen3_asr` config: `get_text_config()` reads
     `self.thinker_config`, but transformers 5.6's `validate_token_ids` calls it
     DURING `super().__init__()` — before `__init__` assigns `thinker_config` —
     so loading ANY Qwen3-ASR config raises AttributeError. (Same bug class sglang
     fixed in its config via PR #24187.) Fix: guard `get_text_config`.

  2) slime `processing_utils.load_processor`: for `qwen3_asr`, transformers'
     AutoProcessor returns a bare tokenizer, so slime falls back to its GLM-4V
     processor and crashes (`NoneType video_processor`). sglang already ships a
     `Qwen3ASRProcessor` — use it before the GLM-4V fallback.

  3) megatron-bridge `Qwen3ASRThinkerModel.__init__` hard-raises when
     `pg_collection is None`, but slime's megatron model_provider builds the model
     without one. The error message itself names the intended default
     (`ProcessGroupCollection.use_mpu_process_groups()`), which is available once
     slime has initialized model-parallel state — so default to it instead of
     raising. (Other megatron-bridge models accept a None pg_collection; Qwen3-ASR
     is the outlier.)

Run at image build:  python _native_qwen3asr_compat.py
"""
import pathlib


def _patch_bridge_config() -> None:
    for p in pathlib.Path("/usr/local/lib").glob(
        "python3.*/dist-packages/megatron/bridge/models/qwen3_asr/hf_qwen3_asr/"
        "configuration_qwen3_asr.py"
    ):
        s = p.read_text()
        if 'hasattr(self, "thinker_config")' in s:
            print("compat: bridge config already guarded:", p)
            continue
        needle = "return self.thinker_config.get_text_config()"
        if needle in s:
            s = s.replace(
                "        " + needle,
                '        if not hasattr(self, "thinker_config"):\n'
                "            return self\n"
                "        " + needle,
                1,
            )
            p.write_text(s)
            print("compat: patched bridge get_text_config guard:", p)


_QWEN3ASR_PROC_HELPER = '''

def _try_load_qwen3_asr_processor(name_or_path, **kwargs):
    """Qwen3-ASR has no transformers AutoProcessor entry; build sglang's composite
    WhisperFeatureExtractor + tokenizer processor (the same one the engine uses)."""
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(name_or_path, trust_remote_code=True)
    except Exception:
        return None
    if getattr(cfg, "model_type", "") != "qwen3_asr":
        return None
    try:
        from sglang.srt.configs.qwen3_asr import Qwen3ASRProcessor

        return Qwen3ASRProcessor.from_pretrained(name_or_path)
    except Exception:
        return None

'''


def _patch_slime_processor() -> None:
    p = pathlib.Path("/root/slime/slime/utils/processing_utils.py")
    if not p.exists():
        print("compat: slime processing_utils not found at", p)
        return
    s = p.read_text()
    if "_try_load_qwen3_asr_processor" in s:
        print("compat: slime processor already patched")
        return
    if "def _try_load_glm4v_processor(" not in s:
        print("compat: WARNING — slime load_processor shape changed; skipping")
        return
    s = s.replace(
        "def _try_load_glm4v_processor(",
        _QWEN3ASR_PROC_HELPER + "def _try_load_glm4v_processor(",
        1,
    )
    s = s.replace(
        "        proc = _try_load_glm4v_processor(name_or_path, **kwargs)",
        "        proc = _try_load_qwen3_asr_processor(name_or_path, **kwargs)"
        " or _try_load_glm4v_processor(name_or_path, **kwargs)",
        1,
    )
    p.write_text(s)
    print("compat: patched slime load_processor for qwen3_asr")


def _patch_bridge_pg_collection() -> None:
    for p in pathlib.Path("/usr/local/lib").glob(
        "python3.*/dist-packages/megatron/bridge/models/qwen3_asr/"
        "modeling_qwen3_asr/thinker_model.py"
    ):
        s = p.read_text()
        if "use_mpu_process_groups()" in s and "pg_collection = ProcessGroupCollection" in s:
            print("compat: bridge pg_collection already defaulted:", p)
            continue
        needle = (
            "        if pg_collection is None:\n"
            "            raise ValueError(\n"
            '                "pg_collection is required for Qwen3ASRThinkerModel. "\n'
            '                "Use ProcessGroupCollection.use_mpu_process_groups() to get the default collection."\n'
            "            )"
        )
        if needle in s:
            s = s.replace(
                needle,
                "        if pg_collection is None:\n"
                "            pg_collection = ProcessGroupCollection.use_mpu_process_groups()",
                1,
            )
            p.write_text(s)
            print("compat: defaulted bridge pg_collection:", p)
        else:
            print("compat: WARNING — pg_collection raise block not found; skipping")


if __name__ == "__main__":
    _patch_bridge_config()
    _patch_slime_processor()
    _patch_bridge_pg_collection()
