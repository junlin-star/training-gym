"""RolloutImageStore: dedup coverage and what content keying is allowed to cost.

Content keying a PIL image materialises its full raw pixel buffer and hashes it,
which the store pays *before* it knows whether the image can be captured. These
tests pin both halves of that trade: which samples come out annotated, and how
many times the expensive step runs.
"""

from typing import Any

import pytest

from modal_training_gym.common import sample_extraction as se


class FakeImage:
    """Stands in for a PIL image, counting the expensive part of keying.

    ``_raw_image_key`` content-keys anything exposing ``tobytes()``, so this is
    the same path a real image takes — and every ``tobytes()`` call is one full
    raw-pixel materialisation the store chose to pay for.
    """

    mode = "RGB"
    size = (8, 8)

    def __init__(self, content: bytes, counter: list[bytes]) -> None:
        self._content = content
        self._counter = counter

    def tobytes(self) -> bytes:
        self._counter.append(self._content)
        return self._content


@pytest.fixture
def encoder(monkeypatch):
    """Encode without PIL: a data-URI that varies only with image content."""
    monkeypatch.setattr(
        se,
        "_image_to_data_uri",
        lambda value: "data:image/png;base64," + value._content.hex(),
    )


def _run(groups: list[bytes], n_per: int, limit: int, *, share_object: bool = False):
    """Annotate ``n_per`` samples for each image in ``groups``, in group order.

    Unless ``share_object``, each member gets its own object holding equal
    content -- how miles builds a prompt group (deep copy per sample).
    """
    keyings: list[bytes] = []
    store = se.RolloutImageStore(limit)
    metas: list[dict[str, Any]] = []
    for content in groups:
        shared = FakeImage(content, keyings)
        for _ in range(n_per):
            image = shared if share_object else FakeImage(content, keyings)
            meta: dict[str, Any] = {}
            store.annotate({"multimodal_inputs": {"images": [image]}}, meta)
            metas.append(meta)
    return metas, keyings


def _coverage(metas: list[dict[str, Any]]) -> tuple[int, int, int]:
    """(samples annotated, samples carrying bytes, distinct refs)."""
    return (
        sum(1 for m in metas if m.get("image_ref")),
        sum(1 for m in metas if m.get("image")),
        len({m["image_ref"] for m in metas if m.get("image_ref")}),
    )


def test_group_shares_one_carrier(encoder):
    """Equal content on distinct objects still collapses to one copy of bytes."""
    metas, keyings = _run([b"a"], n_per=8, limit=16)
    assert _coverage(metas) == (8, 1, 1)
    # Distinct objects, so content keying is what proves them equal: once each.
    assert len(keyings) == 8


def test_shared_object_is_keyed_once(encoder):
    """A group handed one image object pays the raw-pixel pass a single time."""
    metas, keyings = _run([b"a"], n_per=8, limit=16, share_object=True)
    assert _coverage(metas) == (8, 1, 1)
    assert len(keyings) == 1


def test_keying_stops_once_the_limit_is_spent(encoder):
    """The cost tracks distinct images, not the sample count.

    Without the close, every one of the 320 samples would pay a full raw-pixel
    copy + hash to learn what the first miss past the limit already settled.
    """
    metas, keyings = _run([bytes([i]) for i in range(40)], n_per=8, limit=4)
    assert _coverage(metas) == (32, 4, 4)  # the 4 captured groups, in full
    assert len(keyings) == 4 * 8 + 1  # + the one miss that closes keying


def test_group_that_spends_the_limit_keeps_its_tail(encoder):
    """Closing on the first *miss* keeps the straddling group whole.

    Group 2's first sample spends the last of the budget; its remaining members
    still resolve against it, and only group 3 goes unannotated.
    """
    metas, keyings = _run([b"a", b"b", b"c"], n_per=8, limit=2)
    assert _coverage(metas) == (16, 2, 2)
    assert len(keyings) == 2 * 8 + 1
    assert all(not m for m in metas[16:])


def test_limit_of_zero_does_no_image_work(encoder):
    metas, keyings = _run([b"a", b"b"], n_per=4, limit=0)
    assert _coverage(metas) == (0, 0, 0)
    assert keyings == []


def test_unencodable_image_does_not_spend_the_budget(monkeypatch):
    """A candidate that fails to encode is remembered as unusable, not captured."""
    keyings: list[bytes] = []
    store = se.RolloutImageStore(1)
    monkeypatch.setattr(se, "_image_to_data_uri", lambda value: None)
    meta: dict[str, Any] = {}
    assert (
        store.annotate(
            {"multimodal_inputs": {"images": [FakeImage(b"a", keyings)]}}, meta
        )
        is False
    )
    assert store.count == 0 and meta == {}

    monkeypatch.setattr(
        se, "_image_to_data_uri", lambda value: "data:image/png;base64,b"
    )
    meta = {}
    assert (
        store.annotate(
            {"multimodal_inputs": {"images": [FakeImage(b"b", keyings)]}}, meta
        )
        is True
    )
    assert store.count == 1 and meta["image"]
