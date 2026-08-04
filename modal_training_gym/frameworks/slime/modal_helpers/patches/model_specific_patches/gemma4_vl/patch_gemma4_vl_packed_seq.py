"""Stop slime's THD packing from colliding with Gemma-4 VL's own attention mask.

``slime/backends/megatron_utils/data.py: get_batch`` concatenates a micro-batch into
``[1, T]`` and attaches ``PackedSeqParams(qkv_format="thd")`` regardless of
``args.qkv_format`` (the same upstream gap ``patch_qwen3_asr_packed_seq`` works
around). ``Gemma4VLModel.forward`` meanwhile builds its own dense ``[1, 1, T, T]``
mask — causal plus bidirectional attention within each run of image tokens — and
hands it to the language model together with those packed params, which TE cannot
honour at once.

Dropping ``packed_seq_params`` in the VL forward processes the stream as one contiguous
sequence and lets the dense mask govern attention. Keeping the dense mask preserves
bidirectional image attention, and with it parity against the SGLang rollouts.

Sound only while the ``[1, T]`` stream holds a single real sequence, which vision mode
arranges by pinning ``micro_batch_size=1`` and ``use_dynamic_batch_size=False`` (see
``Gemma4_26B_A4B_Recipe``). Outside that regime samples would be fused into one stream
and trained against a mask built for one — wrong gradients, no crash — so the injected
guard rejects those regimes, as slime's own ``slime_plugins/megatron_bridge/
qwen3_5_vl.py`` does for Qwen3.5-VL.

Report to slime, not megatron-bridge, and don't expect ``qkv_format`` to come back:
PR #2233 notes #2100 removed BSHD deliberately, so the accepted fix is to make a model
handle packed sequences.

Idempotent. Run at image build:  python patch_gemma4_vl_packed_seq.py
"""

import pathlib

MARKER = "PATCHED_GEMMA4_VL_PACKED_SEQ"

ANCHOR = '        """Forward pass combining HF vision encoder with Megatron language model."""'

_INJECT_TEMPLATE = """
        # @MARKER@: drop slime's thd packing so _compute_attention_mask governs
        # attention; only valid for a single-sequence stream, so reject the rest.
        if packed_seq_params is not None:
            from megatron.core import mpu as _gemma4_vl_mpu

            if _gemma4_vl_mpu.get_context_parallel_world_size() != 1:
                raise NotImplementedError(
                    "@MARKER@: Gemma-4 VL needs context_parallel_size=1 -- the dense "
                    "image mask spans the whole sequence, but slime hands each CP rank "
                    "a zigzag slice."
                )
            if input_ids is not None and input_ids.shape[0] != 1:
                raise NotImplementedError(
                    "@MARKER@: Gemma-4 VL expects a batch dimension of 1, got "
                    f"{input_ids.shape[0]}."
                )
            # slime appends padding as one extra segment, so one sequence means <= 2.
            _cu_seqlens = getattr(packed_seq_params, "cu_seqlens_q", None)
            if _cu_seqlens is not None and len(_cu_seqlens) - 1 > 2:
                raise NotImplementedError(
                    f"@MARKER@: this microbatch packs {len(_cu_seqlens) - 1} segments, "
                    "so it is not a single sequence. Gemma-4 VL needs "
                    "micro_batch_size=1 and use_dynamic_batch_size=False "
                    "(Gemma4_26B_A4B_Recipe sets both)."
                )
            try:
                from megatron.training import get_args as _gemma4_vl_get_args

                _gemma4_vl_args = _gemma4_vl_get_args()
            except Exception:
                _gemma4_vl_args = None
            if _gemma4_vl_args is not None and getattr(
                _gemma4_vl_args, "use_dynamic_batch_size", False
            ):
                raise NotImplementedError(
                    "@MARKER@: Gemma-4 VL needs use_dynamic_batch_size=False -- dynamic "
                    "batching packs several samples per microbatch, which this mask "
                    "cannot represent."
                )
            packed_seq_params = None"""

INJECT = _INJECT_TEMPLATE.replace("@MARKER@", MARKER)


def main() -> None:
    targets = list(
        pathlib.Path("/usr/local/lib").glob(
            "python3.*/dist-packages/megatron/bridge/models/gemma_vl/modeling_gemma4_vl.py"
        )
    )
    if not targets:
        print("WARNING: modeling_gemma4_vl.py not found; skipping VL packed-seq patch")
        return

    for path in targets:
        src = path.read_text()
        if MARKER in src:
            print("compat: modeling_gemma4_vl.py already patched:", path)
            continue
        if ANCHOR not in src:
            print(
                "WARNING: could not find Gemma4VLModel.forward docstring anchor in",
                path,
                "(upstream may have changed; VL packed-seq patch NOT applied)",
            )
            continue
        path.write_text(src.replace(ANCHOR, ANCHOR + INJECT, 1))
        print("compat: patched modeling_gemma4_vl.py packed_seq_params:", path)


if __name__ == "__main__":
    main()
