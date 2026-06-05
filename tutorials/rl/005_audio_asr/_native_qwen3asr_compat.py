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


_QWEN3ASR_HF_CONVERTER = r'''"""Megatron -> HF converter for Qwen3-ASR (shipped into slime's megatron_to_hf).

slime dispatches "qwen3" to convert_qwen2_to_hf, which can't map Qwen3-ASR's audio
tower. This mirrors megatron-bridge's qwen3_asr mapping_registry: the LLM is the
standard Qwen3 decoder nested under thinker.language_model.* (same QKV / gated-MLP /
qk-norm splits as the qwen3_vl converter), and the frozen audio tower is an identity
passthrough (megatron thinker.audio_model.* -> HF thinker.audio_tower.*).
"""
import re

import torch


def convert_qwen3_asr_to_hf(args, name, param):
    # Frozen audio tower: identity passthrough.
    if name.startswith("module.module.thinker.audio_model."):
        hf = "thinker.audio_tower." + name[len("module.module.thinker.audio_model."):]
        return [(hf, param)]
    # Strip the thinker.language_model. nesting so the standard megatron decoder names
    # are exposed; emit HF names back into the ASR thinker. namespace.
    if name.startswith("module.module.thinker.language_model."):
        name = "module.module." + name[len("module.module.thinker.language_model."):]
    if name == "module.module.embedding.word_embeddings.weight":
        return [("thinker.model.embed_tokens.weight", param)]
    if name == "module.module.output_layer.weight":
        return [("thinker.lm_head.weight", param)]
    if name == "module.module.decoder.final_layernorm.weight":
        return [("thinker.model.norm.weight", param)]
    try:
        head_dim = args.kv_channels if args.kv_channels is not None else args.hidden_size // args.num_attention_heads
    except AttributeError:
        head_dim = args.hidden_size // args.num_attention_heads
    value_num_per_group = args.num_attention_heads // args.num_query_groups
    match = re.match(r"module\.module\.decoder\.layers\.(\d+)\.(.+)", name)
    if match:
        layer_idx, rest = match.groups()
        base = f"thinker.model.layers.{layer_idx}"
        if rest == "self_attention.linear_proj.weight":
            return [(f"{base}.self_attn.o_proj.weight", param)]
        if rest == "self_attention.linear_qkv.weight":
            param = param.view(args.num_query_groups, -1, head_dim, args.hidden_size)
            q, k, v = torch.split(param, [value_num_per_group, 1, 1], dim=1)
            return [
                (f"{base}.self_attn.q_proj.weight", q.reshape(-1, args.hidden_size)),
                (f"{base}.self_attn.k_proj.weight", k.reshape(-1, args.hidden_size)),
                (f"{base}.self_attn.v_proj.weight", v.reshape(-1, args.hidden_size)),
            ]
        if rest == "mlp.linear_fc1.weight":
            gate, up = param.chunk(2, dim=0)
            return [(f"{base}.mlp.gate_proj.weight", gate), (f"{base}.mlp.up_proj.weight", up)]
        if rest == "mlp.linear_fc2.weight":
            return [(f"{base}.mlp.down_proj.weight", param)]
        if rest == "self_attention.linear_qkv.layer_norm_weight":
            return [(f"{base}.input_layernorm.weight", param)]
        if rest == "mlp.linear_fc1.layer_norm_weight":
            return [(f"{base}.post_attention_layernorm.weight", param)]
        if rest == "self_attention.q_layernorm.weight":
            return [(f"{base}.self_attn.q_norm.weight", param)]
        if rest == "self_attention.k_layernorm.weight":
            return [(f"{base}.self_attn.k_norm.weight", param)]
    raise ValueError(f"Unknown parameter name: {name}")
'''


def _patch_export_converter() -> None:
    """Ship a qwen3_asr per-param converter into slime's megatron_to_hf + dispatch to it.

    slime's MB->HF tool routes "qwen3" to convert_qwen2_to_hf, which can't map the audio
    tower ("Unknown parameter name: ...thinker.audio_model..."). Add a qwen3_asr
    converter (mirroring megatron-bridge's mapping_registry) and route to it before the
    generic qwen3 branch. This uses slime's own dist-checkpoint loader, so it sidesteps
    the megatron.bridge.training <-> bundled-Megatron-LM version skew that breaks
    AutoBridge.export_ckpt in this image.
    """
    base = pathlib.Path("/root/slime/slime/backends/megatron_utils/megatron_to_hf")
    if not base.exists():
        print("compat: slime megatron_to_hf not found at", base)
        return
    (base / "qwen3_asr.py").write_text(_QWEN3ASR_HF_CONVERTER)
    init = base / "__init__.py"
    s = init.read_text()
    if "convert_qwen3_asr_to_hf" in s:
        print("compat: qwen3_asr HF converter already wired")
        return
    if "from .qwen2 import convert_qwen2_to_hf\n" not in s:
        print("compat: WARNING - megatron_to_hf __init__ shape changed; skipping export")
        return
    s = s.replace(
        "from .qwen2 import convert_qwen2_to_hf\n",
        "from .qwen2 import convert_qwen2_to_hf\nfrom .qwen3_asr import convert_qwen3_asr_to_hf\n",
        1,
    )
    dispatch_anchor = (
        '    elif "qwen2" in model_name or "qwen3" in model_name:\n'
        "        converted_named_tensors = convert_qwen2_to_hf(args, name, param)\n"
    )
    if dispatch_anchor not in s:
        print("compat: WARNING - megatron_to_hf dispatch shape changed; skipping export")
        return
    s = s.replace(
        dispatch_anchor,
        '    elif "qwen3asr" in model_name or "qwen3_asr" in model_name:\n'
        "        converted_named_tensors = convert_qwen3_asr_to_hf(args, name, param)\n"
        + dispatch_anchor,
        1,
    )
    init.write_text(s)
    print("compat: wired qwen3_asr HF converter into slime megatron_to_hf")


if __name__ == "__main__":
    _patch_bridge_config()
    _patch_slime_processor()
    _patch_bridge_pg_collection()
    _patch_export_converter()
