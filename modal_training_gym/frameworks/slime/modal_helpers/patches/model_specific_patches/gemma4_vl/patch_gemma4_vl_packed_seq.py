"""Stop slime's THD packing from colliding with Gemma-4 VL's own attention mask.

``slime/backends/megatron_utils/data.py: get_batch`` concatenates a micro-batch into
``[1, T]`` and attaches ``PackedSeqParams(qkv_format="thd")`` regardless of
``args.qkv_format`` (the same upstream gap ``patch_qwen3_asr_packed_seq`` works
around). ``Gemma4VLModel.forward`` meanwhile builds its own dense ``[1, 1, T, T]``
mask — causal plus bidirectional attention within each run of image tokens — and
hands it to the language model together with those packed params, which TE cannot
honour at once.

Dropping ``packed_seq_params`` in the VL forward processes the stream as one contiguous
sequence and lets the dense mask govern attention. Sound only because vision mode pins
``micro_batch_size=1`` (see ``Gemma4_26B_A4B_Recipe``), so the stream really does hold
one sample. Keeping the dense mask preserves bidirectional image attention, and with it
parity against the SGLang rollouts. Report upstream.

Idempotent. Run at image build:  python patch_gemma4_vl_packed_seq.py
"""

import pathlib

MARKER = "PATCHED_GEMMA4_VL_PACKED_SEQ"

ANCHOR = '        """Forward pass combining HF vision encoder with Megatron language model."""'

INJECT = f"""
        # {MARKER}: slime's get_batch always packs (thd) even under
        # qkv_format="bshd"; with micro_batch_size=1 the [1, T] stream is a single
        # sequence, so drop the packing and let _compute_attention_mask govern
        # attention (preserves bidirectional image attention / HF parity).
        packed_seq_params = None"""


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
