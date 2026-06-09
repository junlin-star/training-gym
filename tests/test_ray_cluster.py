"""Cluster-scheduling helpers in common/ray_cluster: the RDMA-by-GPU-family rule
and the single-node identity path of clustered_if.
"""

import pytest

from modal_training_gym.common.ray_cluster import _supports_rdma, clustered_if


@pytest.mark.parametrize(
    "gpu", ["H100", "H200", "B200", "B300", "GB200", "H100:8", "h100"]
)
def test_supports_rdma_true(gpu):
    assert _supports_rdma(gpu)


@pytest.mark.parametrize("gpu", ["A100", "L40S", "A10G", "T4", "", "A100:8"])
def test_supports_rdma_false(gpu):
    assert not _supports_rdma(gpu)


def test_clustered_if_single_node_is_identity():
    def fn():
        return None

    # Single node: no @clustered, just a plain registration — fn returned unchanged.
    assert clustered_if(False, 1, gpu_type="H100")(fn) is fn
