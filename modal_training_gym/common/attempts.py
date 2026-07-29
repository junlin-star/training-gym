"""Immutable training-attempt namespaces and committed resume boundaries.

An attempt owns every file below ``<logical-run>/attempts/<attempt-id>``.  A
retry may read an earlier attempt, but it always writes to a fresh namespace.
Append-only, content-addressed publication receipts are authoritative. The
mutable ``latest_committed_boundary.json`` file is only a convenience cache
and may safely be stale after concurrent Volume commits.

This module deliberately does not decide *what* constitutes a complete
framework checkpoint.  Framework code must build a manifest whose inventory
contains every state component needed for an exact resume; this module verifies
that inventory before a launcher is allowed to consume it.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any

ATTEMPT_SCHEMA_VERSION = 1
BOUNDARY_SCHEMA_VERSION = 1
ATTEMPTS_DIRNAME = "attempts"
ATTEMPT_MANIFEST = "attempt.json"
BOUNDARIES_DIRNAME = "committed_boundaries"
BOUNDARY_PUBLICATIONS_DIRNAME = "committed_boundary_publications"
LATEST_BOUNDARY_POINTER = "latest_committed_boundary.json"
ACCEPTED_LINEAGE = "accepted_lineage.json"
RUN_CONTRACT_SCHEMA_VERSION = 1

_ATTEMPT_ID = re.compile(r"[0-9a-f]{32}")
_RUN_CONTRACT_SHA256 = re.compile(r"[0-9a-f]{64}")
_PUBLICATION_RECEIPT = re.compile(
    r"rollout_(?P<rollout>[0-9]{7,})_"
    r"(?P<attempt>[0-9a-f]{32})_(?P<digest>[0-9a-f]{64})\.json"
)
_LOCAL_PUBLICATION_LOCK = threading.RLock()


def new_attempt_id() -> str:
    """Return a random, path-safe execution-attempt identity."""
    return uuid.uuid4().hex


def validate_attempt_id(attempt_id: str) -> str:
    value = str(attempt_id)
    if not _ATTEMPT_ID.fullmatch(value):
        raise ValueError("attempt_id must be 32 lowercase hexadecimal characters")
    return value


def validate_run_contract_sha256(value: str) -> str:
    """Validate the canonical fingerprint for one scientific launch contract."""
    normalized = str(value)
    if not _RUN_CONTRACT_SHA256.fullmatch(normalized):
        raise ValueError(
            "run_contract_sha256 must be 64 lowercase hexadecimal characters"
        )
    return normalized


def run_contract_sha256(payload: dict[str, Any]) -> str:
    """Hash a JSON-native, canonical scientific launch contract.

    The caller is responsible for excluding credentials and attempt-local
    values. Rejecting non-finite floats and non-JSON objects keeps the same
    logical configuration byte-identical across processes and relaunches.
    """
    if not isinstance(payload, dict):
        raise TypeError("scientific run contract must be a mapping")
    canonical = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def _contained_path(root: Path, relative: Any, *, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{label} must be a nonempty relative path")
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"{label} must remain inside the logical run")
    resolved_root = root.resolve()
    resolved = (root / candidate).resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ValueError(f"{label} escapes the logical run")
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _require_real_directory(path: Path, *, label: str) -> None:
    """Reject symlinked namespace components before reading or writing them."""
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise ValueError(f"{label} is missing") from exc
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError(f"{label} must not be a symbolic link")
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"{label} must be a directory")


def _open_directory_at(parent_descriptor: int, name: str) -> int:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    return os.open(name, flags, dir_fd=parent_descriptor)


def _write_exclusive_at(
    directory_descriptor: int,
    name: str,
    payload: dict[str, Any],
) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(
        name,
        flags,
        0o644,
        dir_fd=directory_descriptor,
    )
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(_json_bytes(payload))
        stream.flush()
        os.fsync(stream.fileno())
    os.fsync(directory_descriptor)


def _require_attempt_namespace(root: Path, attempt_id: str) -> Path:
    """Return an attempt root only after verifying non-symlink directories."""
    _require_real_directory(root, label="logical run root")
    attempts = root / ATTEMPTS_DIRNAME
    _require_real_directory(attempts, label="attempts namespace")
    target = attempts / validate_attempt_id(attempt_id)
    _require_real_directory(target, label="attempt namespace")
    resolved_root = root.resolve()
    resolved_target = target.resolve()
    if resolved_target.parent != attempts.resolve():
        raise ValueError("attempt namespace escapes the logical run")
    if resolved_root not in resolved_target.parents:
        raise ValueError("attempt namespace escapes the logical run")
    return target


def _atomic_write(path: Path, payload: dict[str, Any], *, exclusive: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _json_bytes(payload)
    if exclusive:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            _fsync_directory(path.parent)
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        return

    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def attempt_root(logical_root: str | os.PathLike[str], attempt_id: str) -> Path:
    return Path(logical_root) / ATTEMPTS_DIRNAME / validate_attempt_id(attempt_id)


def create_attempt_namespace(
    logical_root: str | os.PathLike[str],
    *,
    run_id: str,
    attempt_id: str,
    attempt_count: int,
    initial_load: str = "",
    parent_boundary: dict[str, Any] | None = None,
    run_contract_sha256: str | None = None,
) -> Path:
    """Create a write-once attempt directory and its immutable owner manifest."""
    root = Path(logical_root)
    if root.name != run_id:
        raise ValueError(
            f"logical run root must end in run_id: root={root} run_id={run_id}"
        )
    if attempt_count < 1:
        raise ValueError("attempt_count must be positive")
    attempt_id = validate_attempt_id(attempt_id)
    contract_digest = (
        validate_run_contract_sha256(run_contract_sha256)
        if run_contract_sha256 is not None
        else ""
    )
    root.mkdir(parents=True, exist_ok=True)
    _require_real_directory(root, label="logical run root")
    target = attempt_root(root, attempt_id)

    parent: dict[str, Any] | None = None
    if parent_boundary is not None:
        parent_contract = str(parent_boundary.get("run_contract_sha256") or "")
        if contract_digest and parent_contract != contract_digest:
            raise ValueError(
                "parent boundary belongs to a different scientific run contract"
            )
        if parent_contract and not contract_digest:
            raise ValueError(
                "an unbound attempt cannot descend from a contract-bound boundary"
            )
        parent = {
            "attempt_id": validate_attempt_id(parent_boundary["attempt_id"]),
            "rollout_id": int(parent_boundary["rollout_id"]),
            "checkpoint_iteration": int(parent_boundary["checkpoint_iteration"]),
            "boundary_manifest": str(parent_boundary["boundary_manifest"]),
            "boundary_sha256": str(parent_boundary["boundary_sha256"]),
            "scientific_commit_id": str(
                parent_boundary.get("scientific_commit_id") or ""
            ),
            "run_contract_sha256": parent_contract,
            "pending_rollout_id": (
                int(parent_boundary["pending_rollout"]["rollout_id"])
                if isinstance(parent_boundary.get("pending_rollout"), dict)
                else None
            ),
        }
    payload = {
        "schema_version": ATTEMPT_SCHEMA_VERSION,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "attempt_count": int(attempt_count),
        "logical_root": str(root),
        "attempt_root": str(target),
        "initial_load": str(initial_load or ""),
        "run_contract_sha256": contract_digest,
        "parent_boundary": parent,
        "created_at_ns": time.time_ns(),
        "policy": "append_only_attempt_namespace",
    }
    root_descriptor = os.open(
        root,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    attempts_descriptor: int | None = None
    target_descriptor: int | None = None
    created_target = False
    try:
        try:
            os.mkdir(ATTEMPTS_DIRNAME, mode=0o755, dir_fd=root_descriptor)
            os.fsync(root_descriptor)
        except FileExistsError:
            pass
        try:
            attempts_descriptor = _open_directory_at(
                root_descriptor,
                ATTEMPTS_DIRNAME,
            )
        except OSError as exc:
            raise ValueError("attempts namespace must be a real directory") from exc
        os.mkdir(attempt_id, mode=0o755, dir_fd=attempts_descriptor)
        created_target = True
        os.fsync(attempts_descriptor)
        try:
            target_descriptor = _open_directory_at(
                attempts_descriptor,
                attempt_id,
            )
            _write_exclusive_at(target_descriptor, ATTEMPT_MANIFEST, payload)
        except BaseException:
            if target_descriptor is not None:
                try:
                    os.unlink(
                        ATTEMPT_MANIFEST,
                        dir_fd=target_descriptor,
                    )
                except OSError:
                    pass
            raise
    except BaseException:
        if attempts_descriptor is not None and created_target:
            try:
                os.rmdir(attempt_id, dir_fd=attempts_descriptor)
            except OSError:
                pass
        raise
    finally:
        if target_descriptor is not None:
            os.close(target_descriptor)
        if attempts_descriptor is not None:
            os.close(attempts_descriptor)
        os.close(root_descriptor)
    return target


def _validate_inventory(
    logical_root: Path,
    records: Any,
    *,
    verify_hashes: bool,
) -> list[dict[str, Any]]:
    if not isinstance(records, list) or not records:
        raise ValueError("committed boundary requires a nonempty file inventory")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in records:
        if not isinstance(raw, dict):
            raise ValueError("boundary inventory records must be mappings")
        relative = str(raw.get("path") or "")
        if relative in seen:
            raise ValueError(f"duplicate boundary inventory path: {relative}")
        seen.add(relative)
        path = _contained_path(logical_root, relative, label="inventory path")
        if not path.is_file():
            raise ValueError(f"boundary inventory file is missing: {relative}")
        size = int(raw.get("bytes", -1))
        if size <= 0 or path.stat().st_size != size:
            raise ValueError(f"boundary inventory size mismatch: {relative}")
        digest = str(raw.get("sha256") or "")
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"invalid boundary inventory digest: {relative}")
        if verify_hashes and _sha256(path) != digest:
            raise ValueError(f"boundary inventory digest mismatch: {relative}")
        normalized.append({"path": relative, "bytes": size, "sha256": digest})
    return normalized


def _validate_scientific_commit(
    logical_root: Path,
    payload: dict[str, Any],
    *,
    scientific_commit: Path,
) -> None:
    """Authenticate the framework commit instead of trusting wrapper assertions."""
    try:
        raw = json.loads(scientific_commit.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("scientific boundary commit is unreadable") from exc
    if not isinstance(raw, dict):
        raise ValueError("scientific boundary commit must be a JSON object")
    recorded_id = str(raw.pop("commit_id", "") or "")
    canonical = json.dumps(
        raw,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    computed_id = f"boundary-sha256:{hashlib.sha256(canonical).hexdigest()}"
    wrapper_id = str(payload.get("scientific_commit_id") or "")
    if recorded_id != computed_id or wrapper_id != computed_id:
        raise ValueError("scientific boundary content ID mismatch")

    expected = {
        "schema_version": BOUNDARY_SCHEMA_VERSION,
        "status": "committed",
        "run_id": logical_root.name,
        "attempt_id": payload["attempt_id"],
        "rollout_id": int(payload["rollout_id"]),
        "terminal": payload["terminal"],
    }
    run_contract = str(payload.get("run_contract_sha256") or "")
    if run_contract:
        expected["run_contract_sha256"] = validate_run_contract_sha256(run_contract)
    for key, value in expected.items():
        if raw.get(key) != value:
            raise ValueError(f"scientific boundary {key} mismatch")
    if str(raw.get("parent_commit_id") or "") != str(
        payload.get("parent_commit_id") or ""
    ):
        raise ValueError("scientific boundary parent commit mismatch")

    actor_checkpoint = raw.get("actor_checkpoint")
    publication = raw.get("publication")
    pipeline = raw.get("pipeline")
    if not isinstance(actor_checkpoint, dict) or int(
        actor_checkpoint.get("iteration", -1)
    ) != int(payload["rollout_id"]):
        raise ValueError("scientific boundary actor checkpoint mismatch")
    if not isinstance(publication, dict):
        raise ValueError("scientific boundary publication receipt is missing")
    if int(publication.get("published_weight_version", -1)) != int(
        payload.get("publication_version", -1)
    ):
        raise ValueError("scientific boundary publication version mismatch")
    if int(publication.get("completed_optimizer_step_index", -1)) != int(
        payload.get("completed_optimizer_step_index", -1)
    ):
        raise ValueError("scientific boundary optimizer watermark mismatch")
    if not isinstance(pipeline, dict):
        raise ValueError("scientific boundary pipeline receipt is missing")
    rich_pending = pipeline.get("pending_rollout")
    wrapper_pending = payload.get("pending_rollout")
    if (rich_pending is None) != (wrapper_pending is None):
        raise ValueError("scientific and generic pending-rollout receipts disagree")
    if isinstance(rich_pending, dict) and isinstance(wrapper_pending, dict):
        if int(rich_pending.get("rollout_id", -1)) != int(
            wrapper_pending.get("rollout_id", -1)
        ):
            raise ValueError("scientific pending-rollout ID mismatch")
        batch = rich_pending.get("batch")
        if not isinstance(batch, dict):
            raise ValueError("scientific pending-rollout batch receipt is missing")
        batch_path = Path(str(batch.get("path") or ""))
        wrapper_path = _contained_path(
            logical_root,
            wrapper_pending.get("path"),
            label="pending_rollout.path",
        )
        if not batch_path.is_absolute() or batch_path.resolve() != wrapper_path:
            raise ValueError("scientific pending-rollout path mismatch")


def validate_boundary_manifest(
    logical_root: str | os.PathLike[str],
    payload: dict[str, Any],
    *,
    verify_hashes: bool = True,
) -> dict[str, Any]:
    """Fail closed unless an immutable boundary and all inventoried files exist."""
    root = Path(logical_root)
    _require_real_directory(root, label="logical run root")
    if payload.get("schema_version") != BOUNDARY_SCHEMA_VERSION:
        raise ValueError("unsupported committed-boundary schema")
    if payload.get("status") != "committed":
        raise ValueError("resume boundary is not committed")
    if payload.get("run_id") != root.name:
        raise ValueError("committed-boundary run_id does not match logical root")
    payload_contract = str(payload.get("run_contract_sha256") or "")
    if payload_contract:
        validate_run_contract_sha256(payload_contract)

    attempt_id = validate_attempt_id(payload.get("attempt_id", ""))
    expected_attempt_root = _require_attempt_namespace(root, attempt_id).resolve()
    checkpoint = _contained_path(
        root, payload.get("checkpoint_path"), label="checkpoint_path"
    )
    if expected_attempt_root not in checkpoint.parents:
        raise ValueError("checkpoint_path is outside its attempt namespace")

    rollout_id = int(payload.get("rollout_id", -1))
    checkpoint_iteration = int(payload.get("checkpoint_iteration", -1))
    if rollout_id < 0 or checkpoint_iteration != rollout_id:
        raise ValueError("boundary rollout/checkpoint iteration mismatch")
    if not isinstance(payload.get("terminal"), bool):
        raise ValueError("boundary requires an explicit terminal flag")

    pending = payload.get("pending_rollout")
    if payload["terminal"] and pending is not None:
        raise ValueError("terminal boundary cannot contain a pending rollout")
    if pending is not None:
        if not isinstance(pending, dict):
            raise ValueError("pending_rollout must be a mapping or null")
        if int(pending.get("rollout_id", -1)) != rollout_id + 1:
            raise ValueError("pending rollout must immediately follow the boundary")
        pending_path = _contained_path(
            root, pending.get("path"), label="pending_rollout.path"
        )
        if expected_attempt_root not in pending_path.parents:
            raise ValueError("pending rollout is outside its attempt namespace")

    inventory = _validate_inventory(
        root,
        payload.get("inventory"),
        verify_hashes=verify_hashes,
    )
    inventory_paths = {record["path"] for record in inventory}
    checkpoint_relative = str(checkpoint.relative_to(root.resolve()))
    if not any(
        path == checkpoint_relative or path.startswith(f"{checkpoint_relative}/")
        for path in inventory_paths
    ):
        raise ValueError("boundary inventory does not contain checkpoint files")
    if pending is not None and str(pending.get("path")) not in inventory_paths:
        raise ValueError("boundary inventory does not contain the pending rollout")
    if payload.get("aux_state_complete") is not True:
        raise ValueError("boundary does not attest complete auxiliary async state")

    scientific_commit = _contained_path(
        root,
        payload.get("scientific_commit_path"),
        label="scientific_commit_path",
    )
    if expected_attempt_root not in scientific_commit.parents:
        raise ValueError("scientific commit is outside its attempt namespace")
    scientific_relative = str(scientific_commit.relative_to(root.resolve()))
    if scientific_relative not in inventory_paths:
        raise ValueError("boundary inventory does not contain the scientific commit")
    commit_id = str(payload.get("scientific_commit_id") or "")
    if not re.fullmatch(r"boundary-sha256:[0-9a-f]{64}", commit_id):
        raise ValueError("boundary requires a content-addressed scientific commit")
    if int(payload.get("publication_version", -1)) < 0:
        raise ValueError("boundary requires a nonnegative publication version")
    if int(payload.get("completed_optimizer_step_index", -1)) < 0:
        raise ValueError("boundary requires a nonnegative optimizer-step watermark")

    owner_path = expected_attempt_root / ATTEMPT_MANIFEST
    try:
        owner = json.loads(owner_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("attempt owner manifest is unreadable") from exc
    if (
        owner.get("schema_version") != ATTEMPT_SCHEMA_VERSION
        or owner.get("run_id") != root.name
        or owner.get("attempt_id") != attempt_id
        or Path(str(owner.get("attempt_root") or "")).resolve() != expected_attempt_root
    ):
        raise ValueError("attempt owner manifest does not match the boundary")
    owner_contract = str(owner.get("run_contract_sha256") or "")
    if owner_contract:
        validate_run_contract_sha256(owner_contract)
    if owner_contract != payload_contract:
        raise ValueError(
            "committed boundary scientific run contract does not match its attempt"
        )
    owner_parent = owner.get("parent_boundary")
    expected_parent_commit = (
        str(owner_parent.get("scientific_commit_id") or "")
        if isinstance(owner_parent, dict)
        else ""
    )
    if str(payload.get("parent_commit_id") or "") != expected_parent_commit:
        raise ValueError("scientific boundary parent commit does not match its attempt")
    _validate_scientific_commit(
        root,
        payload,
        scientific_commit=scientific_commit,
    )
    return dict(payload)


def _load_attempt_owner(root: Path, attempt_id: str) -> dict[str, Any]:
    attempt = _require_attempt_namespace(root, attempt_id)
    owner_path = attempt / ATTEMPT_MANIFEST
    try:
        metadata = owner_path.lstat()
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("attempt owner manifest must be a regular file")
        owner = json.loads(owner_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("attempt owner manifest is unreadable") from exc
    if (
        not isinstance(owner, dict)
        or owner.get("schema_version") != ATTEMPT_SCHEMA_VERSION
        or owner.get("run_id") != root.name
        or owner.get("attempt_id") != attempt_id
        or Path(str(owner.get("attempt_root") or "")).resolve() != attempt.resolve()
    ):
        raise ValueError("attempt owner manifest does not match its namespace")
    contract_digest = str(owner.get("run_contract_sha256") or "")
    if contract_digest:
        validate_run_contract_sha256(contract_digest)
    return owner


def _boundary_reference(boundary: dict[str, Any]) -> dict[str, Any]:
    return {
        "attempt_id": validate_attempt_id(boundary.get("attempt_id", "")),
        "rollout_id": int(boundary["rollout_id"]),
        "boundary_sha256": str(boundary["boundary_sha256"]),
        "scientific_commit_id": str(boundary["scientific_commit_id"]),
    }


def _normalize_predecessor(
    raw: Any,
    *,
    current_rollout_id: int,
) -> dict[str, Any] | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("boundary publication predecessor must be an object or null")
    predecessor = {
        "attempt_id": validate_attempt_id(raw.get("attempt_id", "")),
        "rollout_id": int(raw.get("rollout_id", -1)),
        "boundary_sha256": str(raw.get("boundary_sha256") or ""),
        "scientific_commit_id": str(raw.get("scientific_commit_id") or ""),
    }
    if predecessor["rollout_id"] < 0:
        raise ValueError("boundary publication predecessor rollout is invalid")
    if predecessor["rollout_id"] >= current_rollout_id:
        raise ValueError("boundary publication predecessor must be earlier")
    if not re.fullmatch(r"[0-9a-f]{64}", predecessor["boundary_sha256"]):
        raise ValueError("boundary publication predecessor digest is invalid")
    if not re.fullmatch(
        r"boundary-sha256:[0-9a-f]{64}",
        predecessor["scientific_commit_id"],
    ):
        raise ValueError("boundary publication predecessor commit ID is invalid")
    return predecessor


def _load_boundary_from_pointer(
    root: Path,
    pointer: dict[str, Any],
    *,
    verify_hashes: bool,
    label: str,
    require_predecessor: bool = False,
) -> dict[str, Any]:
    if not isinstance(pointer, dict):
        raise ValueError(f"{label} must be a JSON object")
    if pointer.get("schema_version") != BOUNDARY_SCHEMA_VERSION:
        raise ValueError(f"unsupported {label} schema")
    if pointer.get("run_id") != root.name:
        raise ValueError(f"{label} run_id mismatch")
    attempt_id = validate_attempt_id(pointer.get("attempt_id", ""))
    rollout_id = int(pointer.get("rollout_id", -1))
    if rollout_id < 0:
        raise ValueError(f"{label} rollout_id is invalid")
    attempt = _require_attempt_namespace(root, attempt_id)
    boundaries = attempt / BOUNDARIES_DIRNAME
    _require_real_directory(boundaries, label="committed-boundary namespace")
    manifest_path = _contained_path(
        root,
        pointer.get("boundary_manifest"),
        label="boundary_manifest",
    )
    expected_manifest = boundaries / f"rollout_{rollout_id:07d}.json"
    if manifest_path != expected_manifest.resolve():
        raise ValueError(f"{label} manifest path is not canonical")
    try:
        metadata = manifest_path.lstat()
    except FileNotFoundError as exc:
        raise ValueError("committed-boundary manifest is missing") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError("committed-boundary manifest must be a regular file")
    expected_digest = str(pointer.get("boundary_sha256") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_digest):
        raise ValueError(f"{label} manifest digest is invalid")
    if _sha256(manifest_path) != expected_digest:
        raise ValueError("committed-boundary manifest digest mismatch")
    try:
        payload = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("committed-boundary manifest is unreadable") from exc
    validated = validate_boundary_manifest(root, payload, verify_hashes=verify_hashes)
    for key in (
        "attempt_id",
        "rollout_id",
        "checkpoint_iteration",
        "scientific_commit_id",
        "run_contract_sha256",
    ):
        if (validated.get(key) or "") != (pointer.get(key) or ""):
            raise ValueError(f"{label} {key} mismatch")
    if require_predecessor and "predecessor" not in pointer:
        raise ValueError("boundary publication predecessor is missing")
    predecessor = _normalize_predecessor(
        pointer.get("predecessor"),
        current_rollout_id=rollout_id,
    )
    loaded = {
        **validated,
        "boundary_manifest": str(manifest_path.relative_to(root.resolve())),
        "boundary_sha256": expected_digest,
    }
    if require_predecessor:
        loaded["_publication_predecessor"] = predecessor
    return loaded


def _boundary_is_ancestor(
    root: Path,
    older: dict[str, Any],
    newer: dict[str, Any],
) -> bool:
    """Return whether ``older`` is on the immutable lineage ending at ``newer``."""
    older_attempt = str(older["attempt_id"])
    newer_attempt = str(newer["attempt_id"])
    older_rollout = int(older["rollout_id"])
    newer_rollout = int(newer["rollout_id"])
    if older_attempt == newer_attempt:
        if older_rollout < newer_rollout:
            return True
        return (
            older_rollout == newer_rollout
            and older["boundary_sha256"] == newer["boundary_sha256"]
            and older["scientific_commit_id"] == newer["scientific_commit_id"]
        )
    if older_rollout >= newer_rollout:
        return False
    # A publication edge that crosses attempts must name the child attempt's
    # exact authenticated parent boundary. Skipping across an older ancestor
    # would hide missing receipts and make a retried suffix ambiguous.
    owner = _load_attempt_owner(root, newer_attempt)
    parent = owner.get("parent_boundary")
    return (
        isinstance(parent, dict)
        and validate_attempt_id(parent.get("attempt_id", "")) == older_attempt
        and int(parent.get("rollout_id", -1)) == older_rollout
        and parent.get("boundary_sha256") == older["boundary_sha256"]
        and parent.get("scientific_commit_id") == older["scientific_commit_id"]
    )


def _select_publication_tip(
    root: Path,
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    """Select the unique tip of a serial, expected-predecessor history."""
    BoundaryIdentity = tuple[str, int, str, str]

    def identity(boundary: dict[str, Any]) -> BoundaryIdentity:
        return (
            str(boundary["attempt_id"]),
            int(boundary["rollout_id"]),
            str(boundary["boundary_sha256"]),
            str(boundary["scientific_commit_id"]),
        )

    def predecessor_identity(
        boundary: dict[str, Any],
    ) -> BoundaryIdentity | None:
        predecessor = boundary.get("_publication_predecessor")
        if predecessor is None:
            return None
        return (
            str(predecessor["attempt_id"]),
            int(predecessor["rollout_id"]),
            str(predecessor["boundary_sha256"]),
            str(predecessor["scientific_commit_id"]),
        )

    unique: dict[BoundaryIdentity, dict[str, Any]] = {}
    rollout_slots: dict[tuple[str, int], tuple[str, str]] = {}
    for candidate in candidates:
        attempt_id = str(candidate["attempt_id"])
        rollout_id = int(candidate["rollout_id"])
        digest = str(candidate["boundary_sha256"])
        scientific_commit_id = str(candidate["scientific_commit_id"])
        slot = (attempt_id, rollout_id)
        existing_content = rollout_slots.setdefault(
            slot,
            (digest, scientific_commit_id),
        )
        if existing_content != (digest, scientific_commit_id):
            raise ValueError(
                "conflicting committed boundaries occupy the same attempt rollout"
            )
        key = identity(candidate)
        existing = unique.setdefault(key, candidate)
        if existing.get("_publication_predecessor") != candidate.get(
            "_publication_predecessor"
        ):
            raise ValueError(
                "duplicate boundary publications disagree on their predecessor"
            )

    children: dict[BoundaryIdentity, list[BoundaryIdentity]] = {
        key: [] for key in unique
    }
    roots: list[BoundaryIdentity] = []
    for key, candidate in unique.items():
        predecessor_key = predecessor_identity(candidate)
        if predecessor_key is None:
            owner = _load_attempt_owner(root, str(candidate["attempt_id"]))
            if owner.get("parent_boundary") is not None:
                raise ValueError(
                    "committed-boundary publication root belongs to an attempt "
                    "that declares a parent"
                )
            roots.append(key)
            continue
        if predecessor_key not in unique:
            raise ValueError(
                "committed-boundary publication history is incomplete: "
                "a predecessor receipt is missing"
            )
        predecessor = unique[predecessor_key]
        if not _boundary_is_ancestor(root, predecessor, candidate):
            raise ValueError(
                "boundary publication does not descend from its predecessor"
            )
        children[predecessor_key].append(key)

    tips = [key for key, descendants in children.items() if not descendants]
    if (
        len(roots) != 1
        or len(tips) != 1
        or any(len(descendants) > 1 for descendants in children.values())
    ):
        raise ValueError(
            "committed-boundary publication history forked; concurrent attempts "
            "require operator resolution"
        )

    visited: set[BoundaryIdentity] = set()
    current = roots[0]
    while True:
        if current in visited:
            raise ValueError("committed-boundary publication history contains a cycle")
        visited.add(current)
        descendants = children[current]
        if not descendants:
            break
        current = descendants[0]
    if len(visited) != len(unique) or current != tips[0]:
        raise ValueError(
            "committed-boundary publication history forked; concurrent attempts "
            "require operator resolution"
        )
    return unique[tips[0]]


def _load_publication_receipts(
    root: Path,
    *,
    verify_hashes: bool,
) -> list[dict[str, Any]]:
    publications = root / BOUNDARY_PUBLICATIONS_DIRNAME
    try:
        publications.lstat()
    except FileNotFoundError:
        return []
    _require_real_directory(
        publications,
        label="committed-boundary publication namespace",
    )
    candidates: list[dict[str, Any]] = []
    for receipt_path in sorted(publications.iterdir()):
        if receipt_path.name.startswith("."):
            continue
        match = _PUBLICATION_RECEIPT.fullmatch(receipt_path.name)
        if match is None:
            raise ValueError(
                f"unexpected committed-boundary publication: {receipt_path.name}"
            )
        metadata = receipt_path.lstat()
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("boundary publication receipt must be a regular file")
        encoded = receipt_path.read_bytes()
        if hashlib.sha256(encoded).hexdigest() != match.group("digest"):
            raise ValueError("boundary publication receipt digest mismatch")
        try:
            pointer = json.loads(encoded)
        except json.JSONDecodeError as exc:
            raise ValueError("boundary publication receipt is unreadable") from exc
        if str(pointer.get("attempt_id") or "") != match.group("attempt") or int(
            pointer.get("rollout_id", -1)
        ) != int(match.group("rollout")):
            raise ValueError("boundary publication receipt filename mismatch")
        candidates.append(
            _load_boundary_from_pointer(
                root,
                pointer,
                verify_hashes=verify_hashes,
                label="committed-boundary publication",
                require_predecessor=True,
            )
        )
    return candidates


def _write_immutable_json(path: Path, payload: dict[str, Any]) -> None:
    encoded = _json_bytes(payload)
    try:
        _atomic_write(path, payload, exclusive=True)
    except FileExistsError:
        try:
            existing = path.read_bytes()
        except OSError as exc:
            raise ValueError(f"immutable file is unreadable: {path}") from exc
        if existing != encoded:
            raise ValueError(
                f"immutable file already exists with other content: {path}"
            )


def _write_publication_receipt(
    root: Path,
    pointer: dict[str, Any],
) -> Path:
    publications = root / BOUNDARY_PUBLICATIONS_DIRNAME
    publications.mkdir(exist_ok=True)
    _require_real_directory(
        publications,
        label="committed-boundary publication namespace",
    )
    digest = hashlib.sha256(_json_bytes(pointer)).hexdigest()
    receipt = publications / (
        f"rollout_{int(pointer['rollout_id']):07d}_"
        f"{pointer['attempt_id']}_{digest}.json"
    )
    _write_immutable_json(receipt, pointer)
    return receipt


def publish_committed_boundary(
    logical_root: str | os.PathLike[str],
    payload: dict[str, Any],
    *,
    verify_hashes: bool = True,
) -> dict[str, Any]:
    """Append a durable publication, then update a best-effort pointer cache.

    Modal Volumes do not provide distributed file locking and same-file writes
    are last-writer-wins. The immutable publication receipts are therefore the
    source of truth. The mutable pointer remains useful to people and legacy
    readers, but a stale concurrent write cannot regress a journal-aware reader.

    Framework integrations that just computed every inventory digest may pass
    ``verify_hashes=False`` to avoid rereading a very large checkpoint. Resume
    consumers always verify the persisted digests.
    """
    root = Path(logical_root)
    with _LOCAL_PUBLICATION_LOCK:
        validated = validate_boundary_manifest(
            root,
            payload,
            verify_hashes=verify_hashes,
        )
        attempt_id = validated["attempt_id"]
        rollout_id = int(validated["rollout_id"])
        # The prior boundary was authenticated when it became eligible and will
        # be fully revalidated by any resume consumer. Here only its immutable
        # manifest identity is needed for monotonic lineage checks.
        previous = load_latest_committed_boundary(root, verify_hashes=False)
        if previous is not None:
            previous_rollout = int(previous["rollout_id"])
            if rollout_id <= previous_rollout:
                raise ValueError(
                    "committed-boundary pointer may only advance to a later rollout"
                )
            if previous["attempt_id"] != attempt_id:
                owner = _load_attempt_owner(root, attempt_id)
                parent = owner.get("parent_boundary")
                if (
                    not isinstance(parent, dict)
                    or parent.get("attempt_id") != previous["attempt_id"]
                    or int(parent.get("rollout_id", -1)) != previous_rollout
                    or parent.get("boundary_sha256") != previous["boundary_sha256"]
                    or parent.get("scientific_commit_id")
                    != previous["scientific_commit_id"]
                ):
                    raise ValueError(
                        "new attempt does not descend from the current "
                        "committed boundary"
                    )
        attempt = _require_attempt_namespace(root, attempt_id)
        boundaries = attempt / BOUNDARIES_DIRNAME
        boundaries.mkdir(exist_ok=True)
        _require_real_directory(
            boundaries,
            label="committed-boundary namespace",
        )
        manifest = boundaries / f"rollout_{rollout_id:07d}.json"
        _write_immutable_json(manifest, validated)
        manifest_relative = str(manifest.resolve().relative_to(root.resolve()))
        manifest_digest = _sha256(manifest)
        pointer = {
            "schema_version": BOUNDARY_SCHEMA_VERSION,
            "run_id": root.name,
            "attempt_id": attempt_id,
            "rollout_id": rollout_id,
            "checkpoint_iteration": int(validated["checkpoint_iteration"]),
            "scientific_commit_id": str(validated["scientific_commit_id"]),
            "run_contract_sha256": str(validated.get("run_contract_sha256") or ""),
            "boundary_manifest": manifest_relative,
            "boundary_sha256": manifest_digest,
            "predecessor": (
                _boundary_reference(previous) if previous is not None else None
            ),
            "updated_at_ns": time.time_ns(),
        }
        # Receipt creation is the commit point. It precedes the mutable cache so
        # a crash cannot advertise a boundary that lacks a publication record.
        _write_publication_receipt(root, pointer)
        _atomic_write(root / LATEST_BOUNDARY_POINTER, pointer, exclusive=False)
        return pointer


def load_latest_committed_boundary(
    logical_root: str | os.PathLike[str],
    *,
    verify_hashes: bool = True,
    expected_run_contract_sha256: str | None = None,
    allow_legacy_pointer: bool = False,
) -> dict[str, Any] | None:
    """Resolve and verify the unique committed tip a retry may consume."""
    root = Path(logical_root)
    expected_contract = (
        validate_run_contract_sha256(expected_run_contract_sha256)
        if expected_run_contract_sha256 is not None
        else None
    )
    if not root.exists():
        return None
    _require_real_directory(root, label="logical run root")
    candidates = _load_publication_receipts(
        root,
        # Authenticating every historical checkpoint digest would make one
        # resume O(total history bytes). Receipt/manifest digests, paths, sizes,
        # owner manifests, and scientific commits are still checked for all
        # history; only the selected tip needs expensive inventory hashes.
        verify_hashes=False,
    )
    if candidates:
        # The pointer is deliberately not consulted here: it is a non-authority
        # cache and can be stale or corrupt after concurrent last-writer-wins
        # Volume commits. Every receipt is immutable and content-addressed.
        tip = _select_publication_tip(root, candidates)
        public_tip = {
            key: value
            for key, value in tip.items()
            if not key.startswith("_publication_")
        }
        if (
            expected_contract is not None
            and str(public_tip.get("run_contract_sha256") or "") != expected_contract
        ):
            raise ValueError(
                "latest committed boundary belongs to a different scientific "
                "run contract"
            )
        if not verify_hashes:
            return public_tip
        validated = validate_boundary_manifest(root, public_tip, verify_hashes=True)
        return {
            **validated,
            "boundary_manifest": public_tip["boundary_manifest"],
            "boundary_sha256": public_tip["boundary_sha256"],
        }

    pointer_path = root / LATEST_BOUNDARY_POINTER
    publication_path = root / BOUNDARY_PUBLICATIONS_DIRNAME
    if publication_path.exists():
        raise ValueError(
            "committed-boundary publication journal exists but has no "
            "authoritative receipts"
        )
    if not pointer_path.is_file():
        return None
    if not allow_legacy_pointer:
        raise ValueError(
            "resume requires an authoritative immutable boundary publication "
            "receipt; mutable legacy pointer fallback requires explicit opt-in"
        )
    try:
        pointer = json.loads(pointer_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("latest committed-boundary pointer is unreadable") from exc
    loaded = _load_boundary_from_pointer(
        root,
        pointer,
        verify_hashes=verify_hashes,
        label="committed-boundary pointer",
    )
    if (
        expected_contract is not None
        and str(loaded.get("run_contract_sha256") or "") != expected_contract
    ):
        raise ValueError(
            "latest committed boundary belongs to a different scientific run contract"
        )
    return loaded


def write_accepted_lineage(
    logical_root: str | os.PathLike[str],
    *,
    final_attempt_id: str,
    final_rollout_id: int,
) -> Path:
    """Freeze the successful chain and the accepted rollout range per attempt."""
    root = Path(logical_root)
    current = validate_attempt_id(final_attempt_id)
    end = int(final_rollout_id)
    if end < 0:
        raise ValueError("final_rollout_id must be nonnegative")
    latest = load_latest_committed_boundary(root, verify_hashes=True)
    if (
        latest is None
        or latest["attempt_id"] != current
        or int(latest["rollout_id"]) != end
    ):
        raise ValueError(
            "accepted lineage requires a final boundary committed by the "
            "successful attempt"
        )
    scientific = json.loads((root / str(latest["scientific_commit_path"])).read_text())
    if scientific.get("terminal") is not True:
        raise ValueError("accepted lineage requires a terminal scientific boundary")
    reverse_segments: list[dict[str, Any]] = []
    reverse_handoffs: list[dict[str, Any]] = []
    seen: set[str] = set()
    generation_end = end
    while True:
        if current in seen:
            raise ValueError("attempt lineage contains a cycle")
        seen.add(current)
        current_attempt_root = _require_attempt_namespace(root, current)
        attempt = _load_attempt_owner(root, current)
        parent = attempt.get("parent_boundary")
        start = int(parent["rollout_id"]) + 1 if isinstance(parent, dict) else 0
        if end < start:
            raise ValueError("accepted attempt segment has an empty rollout range")
        inherited_pending = (
            int(parent["pending_rollout_id"])
            if isinstance(parent, dict) and parent.get("pending_rollout_id") is not None
            else None
        )
        generation_start = (
            inherited_pending + 1 if inherited_pending is not None else start
        )
        segment = {
            "attempt_id": current,
            "attempt_root": str(
                current_attempt_root.resolve().relative_to(root.resolve())
            ),
            "first_train_rollout_id": start,
            "last_train_rollout_id": end,
            "first_generation_rollout_id": (
                generation_start if generation_start <= generation_end else None
            ),
            "last_generation_rollout_id": (
                generation_end if generation_start <= generation_end else None
            ),
        }
        reverse_segments.append(segment)
        if not isinstance(parent, dict):
            break
        if inherited_pending is not None:
            reverse_handoffs.append(
                {
                    "rollout_id": inherited_pending,
                    "generation_attempt_id": str(parent["attempt_id"]),
                    "training_attempt_id": current,
                }
            )
        end = int(parent["rollout_id"])
        generation_end = inherited_pending if inherited_pending is not None else end
        current = validate_attempt_id(parent["attempt_id"])

    payload = {
        "schema_version": 1,
        "run_id": root.name,
        "status": "accepted",
        "final_attempt_id": validate_attempt_id(final_attempt_id),
        "final_rollout_id": int(final_rollout_id),
        "run_contract_sha256": str(latest.get("run_contract_sha256") or ""),
        "segments": list(reversed(reverse_segments)),
        "pending_rollout_handoffs": list(reversed(reverse_handoffs)),
        "accepted_at_ns": time.time_ns(),
    }
    path = root / ACCEPTED_LINEAGE
    if path.is_file():
        try:
            existing = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("existing accepted lineage is unreadable") from exc
        immutable_fields = (
            "schema_version",
            "run_id",
            "status",
            "final_attempt_id",
            "final_rollout_id",
            "run_contract_sha256",
            "segments",
            "pending_rollout_handoffs",
        )
        if all(existing.get(key) == payload.get(key) for key in immutable_fields):
            return path
        raise ValueError("existing accepted lineage disagrees with successful chain")
    _atomic_write(path, payload, exclusive=True)
    return path
