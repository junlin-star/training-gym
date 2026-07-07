"""Thread the VPO per-test reward vector into slime's training data (for the VPO advantage fn).

VPO's coverage objective lives in OUR code as a slime `custom_advantage_function`
(train.vpo_advantage.vpo_advantages), invoked by compute_advantages_and_returns(args, rollout_data).
It needs each sample's per-test pass vector — which the rollout emits on
`sample.metadata["usaco"]["test_case_passes"]` (train/rollout.py, --vpo) — to be present in
`rollout_data`. This patch adds that one key in `_convert_samples_to_train_data`
(slime/ray/rollout.py), mirroring the existing `round_number` passthrough right above the anchor.

Anchor is verbatim from the pinned-era source (commit family of slimerl/slime@sha256:087a…,
nightly-dev-20260529a). This patch RAISES (fails loud) if the anchor is absent, so a source drift
aborts the image build rather than silently shipping VPO without its vector.

Registered in launcher._build_slime_base_image. Idempotent. Behaviour-neutral for non-VPO runs
(the key is only added when samples carry the usaco.test_case_passes metadata, i.e. under --vpo).

Executed at image-build time via ``python3 <this file>``.
"""

import pathlib

p = pathlib.Path("/root/slime/slime/ray/rollout.py")
src = p.read_text()

MARKER = "PATCHED_VPO_ROLLOUT_DATA"

anchor = """\
        # For rollout buffer
        if samples[0].metadata and "round_number" in samples[0].metadata:
            train_data["round_number"] = [sample.metadata["round_number"] for sample in samples]"""

new = f'''\
        # For rollout buffer
        if samples[0].metadata and "round_number" in samples[0].metadata:
            train_data["round_number"] = [sample.metadata["round_number"] for sample in samples]

        # VPO: per-test reward vector for the coverage advantage fn (train.vpo_advantage).  {MARKER}
        # gate on any sample
        if any(
            s.metadata and (s.metadata.get("usaco") or {{}}).get("test_case_passes") is not None
            for s in samples
        ):
            train_data["test_case_passes"] = [
                ((s.metadata or {{}}).get("usaco") or {{}}).get("test_case_passes") or []
                for s in samples
            ]'''

# emit test_case_passes into train_data (_convert_samples_to_train_data).
if MARKER in src:
    print("[patch_vpo_rollout_data] edit1 already applied")
elif anchor in src:
    src = src.replace(anchor, new, 1)
    print("[patch_vpo_rollout_data] edit1: threaded test_case_passes into _convert_samples_to_train_data")
else:
    raise SystemExit(
        "[patch_vpo_rollout_data] FATAL: edit1 anchor (round_number passthrough in "
        "_convert_samples_to_train_data) not found — pinned source drifted. Re-confirm before VPO."
    )

# Edit 2 — carry test_case_passes through the per-DP-rank split (_split_train_data_by_dp). Without
# this the key is in train_data but DROPPED when rollout_data is repackaged per DP rank, so it never
# reaches compute_advantages_and_returns → vpo_advantages raises
anchor2 = '                "round_number",\n                "sample_indices",'
new2 = '                "round_number",\n                "test_case_passes",\n                "sample_indices",'
if new2 in src:
    print("[patch_vpo_rollout_data] edit2 already applied")
elif anchor2 in src:
    src = src.replace(anchor2, new2, 1)
    print("[patch_vpo_rollout_data] edit2: added test_case_passes to _split_train_data_by_dp whitelist")
else:
    raise SystemExit(
        "[patch_vpo_rollout_data] FATAL: edit2 anchor (round_number/sample_indices in the "
        "_split_train_data_by_dp key list) not found — pinned source drifted. Re-confirm before VPO."
    )

p.write_text(src)
