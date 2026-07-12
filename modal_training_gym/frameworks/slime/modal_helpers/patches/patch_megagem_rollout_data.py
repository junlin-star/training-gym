"""Thread MegaGem row metadata into slime rollout_data.

MegaGem Stage C emits one Slime sample per selected trainable turn row. The
sample carries row-level ``precomputed_reward`` and ``precomputed_advantage``
under ``sample.metadata["megagem"]``. Slime natively consumes ``sample.reward``,
but it does not carry arbitrary metadata into ``rollout_data`` where the native
``custom_advantage_function_path`` hook runs.

This patch mirrors the existing VPO metadata patch: copy the compact MegaGem
fields into train_data in ``_convert_samples_to_train_data`` and preserve them
through ``_split_train_data_by_dp``. If the pinned slime source drifts, fail the
image build rather than silently training with ordinary GRPO advantages.
"""

import pathlib


p = pathlib.Path("/root/slime/slime/ray/rollout.py")
src = p.read_text()

MARKER = "PATCHED_MEGAGEM_ROLLOUT_DATA"

anchor = """\
        # For rollout buffer
        if samples[0].metadata and "round_number" in samples[0].metadata:
            train_data["round_number"] = [sample.metadata["round_number"] for sample in samples]"""

new = f'''\
        # For rollout buffer
        if samples[0].metadata and "round_number" in samples[0].metadata:
            train_data["round_number"] = [sample.metadata["round_number"] for sample in samples]

        # MegaGem: row-level reward/advantage contract for Stage C.  {MARKER}
        if any(s.metadata and (s.metadata.get("megagem") or {{}}) for s in samples):
            _tg_megagem = [
                ((s.metadata or {{}}).get("megagem") or {{}})
                for s in samples
            ]
            train_data["megagem_precomputed_advantage"] = [
                float(m.get("precomputed_advantage", 0.0))
                for m in _tg_megagem
            ]
            train_data["megagem_precomputed_reward"] = [
                float(m.get("precomputed_reward", 0.0))
                for m in _tg_megagem
            ]
            train_data["megagem_group_key"] = [
                str(m.get("group_key", ""))
                for m in _tg_megagem
            ]'''

if MARKER in src:
    print("[patch_megagem_rollout_data] edit1 already applied")
elif anchor in src:
    src = src.replace(anchor, new, 1)
    print("[patch_megagem_rollout_data] edit1: threaded MegaGem metadata into train_data")
else:
    raise SystemExit(
        "[patch_megagem_rollout_data] FATAL: edit1 anchor (round_number passthrough in "
        "_convert_samples_to_train_data) not found — pinned slime source drifted."
    )

with_vpo_anchor = '                "round_number",\n                "test_case_passes",\n                "sample_indices",'
with_vpo_new = (
    '                "round_number",\n'
    '                "test_case_passes",\n'
    '                "megagem_precomputed_advantage",\n'
    '                "megagem_precomputed_reward",\n'
    '                "megagem_group_key",\n'
    '                "sample_indices",'
)
plain_anchor = '                "round_number",\n                "sample_indices",'
plain_new = (
    '                "round_number",\n'
    '                "megagem_precomputed_advantage",\n'
    '                "megagem_precomputed_reward",\n'
    '                "megagem_group_key",\n'
    '                "sample_indices",'
)

if '"megagem_precomputed_advantage",' in src:
    print("[patch_megagem_rollout_data] edit2 already applied")
elif with_vpo_anchor in src:
    src = src.replace(with_vpo_anchor, with_vpo_new, 1)
    print("[patch_megagem_rollout_data] edit2: preserved MegaGem keys through DP split")
elif plain_anchor in src:
    src = src.replace(plain_anchor, plain_new, 1)
    print("[patch_megagem_rollout_data] edit2: preserved MegaGem keys through DP split")
else:
    raise SystemExit(
        "[patch_megagem_rollout_data] FATAL: edit2 anchor (round_number/sample_indices "
        "whitelist in _split_train_data_by_dp) not found — pinned slime source drifted."
    )

p.write_text(src)
