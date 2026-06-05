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


def _patch_export_converter() -> None:
    """Route Qwen3-ASR MB→HF export through megatron-bridge's own AutoBridge.

    slime's `tools/convert_torch_dist_to_hf.py` dispatches by model name to
    hand-written per-param converters; there is no qwen3_asr entry, so it falls to
    the qwen2 path and dies on the audio tower (`Unknown parameter name:
    ...thinker.audio_model...`). The native megatron-bridge already maps the whole
    model (LLM + QKV/MLP + `thinker.audio_model.** -> thinker.audio_tower.**`) and
    exposes `AutoBridge.export_ckpt`, so short-circuit qwen3_asr to it.
    """
    p = pathlib.Path("/root/slime/tools/convert_torch_dist_to_hf.py")
    if not p.exists():
        print("compat: slime convert tool not found at", p)
        return
    s = p.read_text()
    if "qwen3_asr export via megatron-bridge" in s:
        print("compat: export converter already patched")
        return
    anchor = '    print(f"loading model from {args.input_dir}")'
    if anchor not in s:
        print("compat: WARNING — convert tool shape changed; skipping export patch")
        return
    inject = (
        "    # qwen3_asr export via megatron-bridge (compat shim): slime has no\n"
        "    # qwen3_asr per-param converter; use the native bridge, which maps the\n"
        "    # full model incl. the audio tower.\n"
        "    if args.origin_hf_dir is not None:\n"
        "        from transformers import AutoConfig as _AC\n"
        "        if getattr(_AC.from_pretrained(args.origin_hf_dir, trust_remote_code=True), 'model_type', '') == 'qwen3_asr':\n"
        "            import sys as _sys\n"
        "            from megatron.bridge import AutoBridge as _AB\n"
        "            _AB.from_hf_pretrained(args.origin_hf_dir, trust_remote_code=True).export_ckpt(args.input_dir, args.output_dir)\n"
        "            print(f'[qwen3_asr] exported via megatron-bridge AutoBridge -> {args.output_dir}')\n"
        "            _sys.exit(0)\n"
    )
    s = s.replace(anchor, inject + anchor, 1)
    p.write_text(s)
    print("compat: patched export converter for qwen3_asr (megatron-bridge AutoBridge)")


if __name__ == "__main__":
    _patch_bridge_config()
    _patch_slime_processor()
    _patch_bridge_pg_collection()
    _patch_export_converter()
