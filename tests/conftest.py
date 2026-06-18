"""Shared pytest fixtures and test doubles."""

from __future__ import annotations

import io

import pytest

from modal_training_gym.utils import metadata


class FakeVolume:
    """In-memory stand-in for a Modal Volume that is *not* attached.

    ``reload()`` raises like a real unattached/local volume; reads and writes
    operate on an in-memory dict. A correct metadata layer must still complete a
    ``save()`` against this — reload is only a freshness hint.
    """

    class _DirEntry:
        """Simple stand-in for Modal Volume directory entries."""

        def __init__(self, path: str):
            self.path = path

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}

    def reload(self) -> None:
        raise RuntimeError("reload() can only be called from within a running function")

    def read_file(self, path: str):
        if path not in self.files:
            raise FileNotFoundError(path)
        return [self.files[path]]

    def remove_file(self, path: str) -> None:
        if path not in self.files:
            raise FileNotFoundError(path)
        del self.files[path]

    def iterdir(self, path: str):
        """Return list of files in the given path (stub for Modal Volume.iterdir)."""
        prefix = path.rstrip("/") + "/"
        matching_files = [f for f in self.files.keys() if f.startswith(prefix)]
        return [self._DirEntry(f) for f in matching_files]

    def batch_upload(self, force: bool = False):
        files = self.files

        class _Batch:
            def __enter__(self):
                return self

            def __exit__(self, *_exc):
                return False

            def put_file(self, fileobj: io.BytesIO, path: str) -> None:
                files[path] = fileobj.read()

        return _Batch()


@pytest.fixture
def fake_volume(monkeypatch) -> FakeVolume:
    """Swap the metadata volume for an in-memory ``FakeVolume`` (no Modal, no GPU)."""
    vol = FakeVolume()
    monkeypatch.setattr(metadata, "_metadata_volume", lambda: vol)
    return vol
