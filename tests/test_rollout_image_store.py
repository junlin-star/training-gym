"""Input-image dedup across one rollout's samples.

The store caps *distinct* images. Whether the samples of a prompt group share one
decoded image object depends on how the framework materializes them, so identity
dedup has to stay an optimisation rather than the thing correctness rests on.

Encoding is stubbed: PIL only exists inside the training container, and stubbing
it also makes the encode count — the cost the cap exists to bound — assertable.
"""

import pytest

from modal_training_gym.common import sample_extraction
from modal_training_gym.common.sample_extraction import (
    RolloutImageStore,
    _raw_image_key,
)


class FakeImage:
    """Distinct object per instance, content decided by ``colour``."""

    def __init__(self, colour):
        self.colour = colour
        self.mode = "RGB"
        self.size = (8, 8)

    def tobytes(self):
        return self.colour.encode()


@pytest.fixture
def encodes(monkeypatch):
    calls = []

    def fake_encode(value):
        calls.append(value)
        return f"data:image/png;base64,{value.colour}"

    monkeypatch.setattr(sample_extraction, "_image_to_data_uri", fake_encode)
    return calls


def annotate_all(store, images):
    metas = []
    for image in images:
        metadata = {}
        store.annotate({"multimodal_inputs": {"images": [image]}}, metadata)
        metas.append(metadata)
    return metas


def test_shared_object_across_a_group_stores_bytes_once(encodes):
    shared = FakeImage("red")
    metas = annotate_all(RolloutImageStore(limit=4), [shared] * 4)

    assert len(encodes) == 1
    assert sum("image" in m for m in metas) == 1
    assert len({m["image_ref"] for m in metas}) == 1


def test_distinct_objects_with_equal_content_share_one_ref(encodes):
    """A framework that decodes per sample must not multiply stored images."""
    store = RolloutImageStore(limit=4)
    metas = annotate_all(store, [FakeImage("red") for _ in range(4)])

    assert store.count == 1
    assert sum("image" in m for m in metas) == 1
    assert len({m["image_ref"] for m in metas}) == 1


def test_per_sample_objects_do_not_multiply_encodes(encodes):
    """Cost scales with distinct images, not with rollout size."""
    store = RolloutImageStore(limit=16)
    groups = [FakeImage(c) for c in ("red", "blue") for _ in range(8)]

    annotate_all(store, groups)

    assert store.count == 2
    assert len(encodes) == 2


def test_group_straddling_the_cap_still_resolves_its_ref(encodes):
    """The cap bounds distinct images; it must not cut a prompt group in half."""
    store = RolloutImageStore(limit=1)
    kept = annotate_all(store, [FakeImage("red") for _ in range(3)])
    dropped = annotate_all(store, [FakeImage("blue") for _ in range(3)])

    assert store.count == 1
    assert all("image_ref" in m for m in kept)
    assert all(m == {} for m in dropped)


def test_cap_bounds_distinct_images(encodes):
    store = RolloutImageStore(limit=2)
    annotate_all(store, [FakeImage(c) for c in ("red", "blue", "green", "gold")])

    assert store.count == 2
    assert len(encodes) == 2


def test_zero_limit_disables_capture(encodes):
    metadata = {}
    store = RolloutImageStore(limit=0)

    assert (
        store.annotate({"multimodal_inputs": {"images": [FakeImage("red")]}}, metadata)
        is False
    )
    assert metadata == {}
    assert encodes == []


def test_key_is_content_derived_not_identity():
    """Deterministic guard on what stops a recycled id() serving wrong bytes."""
    assert _raw_image_key(FakeImage("red")) == _raw_image_key(FakeImage("red"))
    assert _raw_image_key(FakeImage("red")) != _raw_image_key(FakeImage("blue"))


def test_shared_object_is_content_keyed_once_per_object(encodes):
    """Keying is what every sample pays; a shared object must pay it once.

    The content key materialises a PIL image's raw pixels, so recomputing it per
    sample makes reporting scale with rollout size instead of distinct images.
    """
    shared = FakeImage("red")
    keyed = []
    shared.tobytes = lambda: (keyed.append(1), b"red")[1]

    annotate_all(RolloutImageStore(limit=4), [shared] * 8)

    assert len(keyed) == 1


def test_identity_keyed_candidates_are_pinned():
    """Without a content key, the store must own a reference to keep id() valid."""

    class Opaque:
        pass

    store = RolloutImageStore(limit=2)
    annotate_all(store, [Opaque()])

    assert len(store._pinned) == 1
