from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from modal_training_gym.common.attempts import (
    ACCEPTED_LINEAGE,
    ATTEMPT_MANIFEST,
    BOUNDARY_PUBLICATIONS_DIRNAME,
    LATEST_BOUNDARY_POINTER,
    create_attempt_namespace,
    load_latest_committed_boundary,
    publish_committed_boundary,
    write_accepted_lineage,
)
from modal_training_gym.common.run import (
    mark_training_attempt_finished,
    mark_training_attempt_started,
    select_accepted_wandb_attempt,
)


def _record(root: Path, path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "path": str(path.relative_to(root)),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _boundary(
    logical_root: Path,
    *,
    attempt_id: str,
    rollout_id: int,
    pending: bool = True,
    parent_commit_id: str = "",
    terminal: bool = False,
    run_contract_sha256: str = "",
) -> dict:
    attempt = logical_root / "attempts" / attempt_id
    checkpoint = attempt / f"iter_{rollout_id:07d}"
    checkpoint.mkdir(parents=True)
    checkpoint_files = [
        checkpoint / "common.pt",
        checkpoint / ".metadata",
        checkpoint / "__0_0.distcp",
    ]
    for index, path in enumerate(checkpoint_files):
        path.write_bytes(f"checkpoint-{index}".encode())
    aux = attempt / "resume_aux" / f"rollout_{rollout_id:07d}.pt"
    aux.parent.mkdir(parents=True, exist_ok=True)
    aux.write_bytes(b"aux-state")
    inventory = [_record(logical_root, path) for path in checkpoint_files]
    inventory.append(_record(logical_root, aux))
    pending_record = None
    rich_pending = None
    if pending:
        pending_path = attempt / "pending" / f"rollout_{rollout_id + 1:07d}.pkl"
        pending_path.parent.mkdir(parents=True, exist_ok=True)
        pending_path.write_bytes(b"pending-batch")
        inventory.append(_record(logical_root, pending_path))
        pending_record = {
            "rollout_id": rollout_id + 1,
            "path": str(pending_path.relative_to(logical_root)),
        }
        rich_pending = {
            "rollout_id": rollout_id + 1,
            "batch": {"path": str(pending_path.resolve())},
        }
    scientific_payload = {
        "schema_version": 1,
        "status": "committed",
        "run_id": logical_root.name,
        "attempt_id": attempt_id,
        "parent_commit_id": parent_commit_id or None,
        "rollout_id": rollout_id,
        "terminal": terminal,
        "actor_checkpoint": {"iteration": rollout_id},
        "publication": {
            "published_weight_version": rollout_id + 1,
            "completed_optimizer_step_index": rollout_id,
        },
        "pipeline": {"pending_rollout": rich_pending},
        **({"run_contract_sha256": run_contract_sha256} if run_contract_sha256 else {}),
    }
    canonical = json.dumps(
        scientific_payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    scientific_id = f"boundary-sha256:{hashlib.sha256(canonical).hexdigest()}"
    scientific_commit = (
        attempt
        / "boundary_commits"
        / f"rollout_{rollout_id:07d}_{scientific_id[-16:]}.json"
    )
    scientific_commit.parent.mkdir(parents=True, exist_ok=True)
    scientific_commit.write_text(
        json.dumps(
            {"commit_id": scientific_id, **scientific_payload},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    inventory.append(_record(logical_root, scientific_commit))
    boundary = {
        "schema_version": 1,
        "status": "committed",
        "run_id": logical_root.name,
        "attempt_id": attempt_id,
        "rollout_id": rollout_id,
        "terminal": terminal,
        "checkpoint_iteration": rollout_id,
        "checkpoint_path": str(checkpoint.relative_to(logical_root)),
        "pending_rollout": pending_record,
        "aux_state_complete": True,
        "publication_version": rollout_id + 1,
        "completed_optimizer_step_index": rollout_id,
        "scientific_commit_path": str(scientific_commit.relative_to(logical_root)),
        "scientific_commit_id": scientific_id,
        "parent_commit_id": parent_commit_id,
        "inventory": inventory,
    }
    if run_contract_sha256:
        boundary["run_contract_sha256"] = run_contract_sha256
    return boundary


def test_attempt_namespace_is_unique_and_owner_manifest_is_immutable(tmp_path) -> None:
    logical = tmp_path / "run-a"
    attempt_id = "a" * 32
    root = create_attempt_namespace(
        logical,
        run_id="run-a",
        attempt_id=attempt_id,
        attempt_count=1,
    )

    owner = json.loads((root / ATTEMPT_MANIFEST).read_text())
    assert owner["attempt_id"] == attempt_id
    assert owner["attempt_count"] == 1
    with pytest.raises(FileExistsError):
        create_attempt_namespace(
            logical,
            run_id="run-a",
            attempt_id=attempt_id,
            attempt_count=2,
        )


def test_attempt_namespace_rejects_symlink_without_writing_through_it(
    tmp_path,
) -> None:
    logical = tmp_path / "run-a"
    outside = tmp_path / "outside"
    logical.mkdir()
    outside.mkdir()
    (logical / "attempts").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="attempts namespace"):
        create_attempt_namespace(
            logical,
            run_id="run-a",
            attempt_id="a" * 32,
            attempt_count=1,
        )

    assert list(outside.iterdir()) == []


def test_attempt_namespace_collision_never_removes_existing_target(tmp_path) -> None:
    logical = tmp_path / "run-a"
    target = logical / "attempts" / ("a" * 32)
    target.mkdir(parents=True)

    with pytest.raises(FileExistsError):
        create_attempt_namespace(
            logical,
            run_id="run-a",
            attempt_id="a" * 32,
            attempt_count=1,
        )

    assert target.is_dir()


def test_attempt_namespace_does_not_follow_attempt_id_symlink(tmp_path) -> None:
    logical = tmp_path / "run-a"
    outside = tmp_path / "outside"
    attempts = logical / "attempts"
    attempts.mkdir(parents=True)
    outside.mkdir()
    target = attempts / ("a" * 32)
    target.symlink_to(outside, target_is_directory=True)

    with pytest.raises(FileExistsError):
        create_attempt_namespace(
            logical,
            run_id="run-a",
            attempt_id="a" * 32,
            attempt_count=1,
        )

    assert target.is_symlink()
    assert list(outside.iterdir()) == []


def test_boundary_pointer_is_published_last_and_verified(tmp_path) -> None:
    logical = tmp_path / "run-a"
    attempt_id = "a" * 32
    create_attempt_namespace(
        logical,
        run_id="run-a",
        attempt_id=attempt_id,
        attempt_count=1,
    )
    pointer = publish_committed_boundary(
        logical,
        _boundary(logical, attempt_id=attempt_id, rollout_id=7),
    )

    loaded = load_latest_committed_boundary(logical)
    assert loaded is not None
    assert loaded["attempt_id"] == attempt_id
    assert loaded["rollout_id"] == 7
    assert pointer["boundary_sha256"] == loaded["boundary_sha256"]

    pending_path = logical / loaded["pending_rollout"]["path"]
    pending_path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="size mismatch|digest mismatch"):
        load_latest_committed_boundary(logical)


def test_contract_bound_boundary_rejects_incompatible_relaunch(tmp_path) -> None:
    logical = tmp_path / "run-a"
    attempt_id = "a" * 32
    contract = "1" * 64
    create_attempt_namespace(
        logical,
        run_id="run-a",
        attempt_id=attempt_id,
        attempt_count=1,
        run_contract_sha256=contract,
    )
    publish_committed_boundary(
        logical,
        _boundary(
            logical,
            attempt_id=attempt_id,
            rollout_id=7,
            run_contract_sha256=contract,
        ),
    )

    loaded = load_latest_committed_boundary(
        logical,
        expected_run_contract_sha256=contract,
    )
    assert loaded is not None
    assert loaded["run_contract_sha256"] == contract
    with pytest.raises(ValueError, match="different scientific run contract"):
        load_latest_committed_boundary(
            logical,
            expected_run_contract_sha256="2" * 64,
        )


def test_contract_bound_resume_rejects_mutable_pointer_without_journal(
    tmp_path,
) -> None:
    logical = tmp_path / "run-a"
    attempt_id = "a" * 32
    contract = "1" * 64
    create_attempt_namespace(
        logical,
        run_id="run-a",
        attempt_id=attempt_id,
        attempt_count=1,
        run_contract_sha256=contract,
    )
    publish_committed_boundary(
        logical,
        _boundary(
            logical,
            attempt_id=attempt_id,
            rollout_id=7,
            run_contract_sha256=contract,
        ),
    )
    publications = logical / BOUNDARY_PUBLICATIONS_DIRNAME
    receipt = next(publications.iterdir())
    receipt.unlink()
    publications.rmdir()

    with pytest.raises(ValueError, match="immutable.*receipt|explicit opt-in"):
        load_latest_committed_boundary(
            logical,
            expected_run_contract_sha256=contract,
        )
    loaded = load_latest_committed_boundary(
        logical,
        expected_run_contract_sha256=contract,
        allow_legacy_pointer=True,
    )
    assert loaded is not None
    assert loaded["rollout_id"] == 7


def test_contract_bound_attempt_cannot_descend_from_other_contract(tmp_path) -> None:
    logical = tmp_path / "run-a"
    first = "a" * 32
    contract = "1" * 64
    create_attempt_namespace(
        logical,
        run_id="run-a",
        attempt_id=first,
        attempt_count=1,
        run_contract_sha256=contract,
    )
    publish_committed_boundary(
        logical,
        _boundary(
            logical,
            attempt_id=first,
            rollout_id=7,
            run_contract_sha256=contract,
        ),
    )
    parent = load_latest_committed_boundary(
        logical,
        expected_run_contract_sha256=contract,
    )
    assert parent is not None

    with pytest.raises(ValueError, match="different scientific run contract"):
        create_attempt_namespace(
            logical,
            run_id="run-a",
            attempt_id="b" * 32,
            attempt_count=2,
            parent_boundary=parent,
            run_contract_sha256="2" * 64,
        )


def test_contract_binding_covers_scientific_commit_content(tmp_path) -> None:
    logical = tmp_path / "run-a"
    attempt_id = "a" * 32
    contract = "1" * 64
    create_attempt_namespace(
        logical,
        run_id="run-a",
        attempt_id=attempt_id,
        attempt_count=1,
        run_contract_sha256=contract,
    )
    payload = _boundary(
        logical,
        attempt_id=attempt_id,
        rollout_id=7,
        run_contract_sha256=contract,
    )
    payload["run_contract_sha256"] = "2" * 64

    with pytest.raises(
        ValueError,
        match="run contract does not match|content ID mismatch",
    ):
        publish_committed_boundary(logical, payload)


def test_append_only_publications_defeat_a_regressed_pointer_cache(tmp_path) -> None:
    logical = tmp_path / "run-a"
    attempt_id = "a" * 32
    create_attempt_namespace(
        logical,
        run_id="run-a",
        attempt_id=attempt_id,
        attempt_count=1,
    )
    stale = publish_committed_boundary(
        logical,
        _boundary(logical, attempt_id=attempt_id, rollout_id=7),
    )
    latest = publish_committed_boundary(
        logical,
        _boundary(logical, attempt_id=attempt_id, rollout_id=9),
    )
    assert latest["predecessor"] == {
        "attempt_id": attempt_id,
        "rollout_id": 7,
        "boundary_sha256": stale["boundary_sha256"],
        "scientific_commit_id": stale["scientific_commit_id"],
    }

    # This is the cross-container last-writer-wins failure mode: an older
    # writer commits its mutable pointer after the newer writer.
    (logical / LATEST_BOUNDARY_POINTER).write_text(
        json.dumps(stale, indent=2, sort_keys=True) + "\n"
    )

    loaded = load_latest_committed_boundary(logical)
    assert loaded is not None
    assert loaded["rollout_id"] == 9
    assert loaded["boundary_sha256"] == latest["boundary_sha256"]
    assert len(list((logical / BOUNDARY_PUBLICATIONS_DIRNAME).iterdir())) == 2


def test_publication_journal_rejects_a_missing_predecessor_receipt(tmp_path) -> None:
    logical = tmp_path / "run-a"
    attempt_id = "a" * 32
    create_attempt_namespace(
        logical,
        run_id="run-a",
        attempt_id=attempt_id,
        attempt_count=1,
    )
    first = publish_committed_boundary(
        logical,
        _boundary(logical, attempt_id=attempt_id, rollout_id=3),
    )
    publish_committed_boundary(
        logical,
        _boundary(logical, attempt_id=attempt_id, rollout_id=7),
    )
    publications = logical / BOUNDARY_PUBLICATIONS_DIRNAME
    first_receipt = next(
        path
        for path in publications.iterdir()
        if json.loads(path.read_text())["boundary_sha256"] == first["boundary_sha256"]
    )
    first_receipt.unlink()

    with pytest.raises(ValueError, match="predecessor receipt is missing"):
        load_latest_committed_boundary(logical)


def test_concurrent_sibling_attempt_publications_fail_closed(tmp_path) -> None:
    logical = tmp_path / "run-a"
    first = "a" * 32
    second = "b" * 32
    sibling = "c" * 32
    create_attempt_namespace(
        logical,
        run_id="run-a",
        attempt_id=first,
        attempt_count=1,
    )
    publish_committed_boundary(
        logical,
        _boundary(logical, attempt_id=first, rollout_id=7),
    )
    parent = load_latest_committed_boundary(logical)
    assert parent is not None
    for attempt_count, attempt_id in enumerate((second, sibling), start=2):
        create_attempt_namespace(
            logical,
            run_id="run-a",
            attempt_id=attempt_id,
            attempt_count=attempt_count,
            parent_boundary=parent,
        )

    publish_committed_boundary(
        logical,
        _boundary(
            logical,
            attempt_id=second,
            rollout_id=8,
            parent_commit_id=str(parent["scientific_commit_id"]),
        ),
    )
    # Simulate another Modal container's stale mounted snapshot: it can see the
    # common parent but not the first sibling's just-committed receipt.
    with patch(
        "modal_training_gym.common.attempts.load_latest_committed_boundary",
        return_value=parent,
    ):
        publish_committed_boundary(
            logical,
            _boundary(
                logical,
                attempt_id=sibling,
                rollout_id=9,
                parent_commit_id=str(parent["scientific_commit_id"]),
            ),
        )

    with pytest.raises(ValueError, match="publication history forked"):
        load_latest_committed_boundary(logical)


def test_boundary_rejects_files_outside_the_attempt_namespace(tmp_path) -> None:
    logical = tmp_path / "run-a"
    attempt_id = "a" * 32
    create_attempt_namespace(
        logical,
        run_id="run-a",
        attempt_id=attempt_id,
        attempt_count=1,
    )
    payload = _boundary(logical, attempt_id=attempt_id, rollout_id=7)
    payload["checkpoint_path"] = "other/iter_0000007"

    with pytest.raises(ValueError, match="outside its attempt namespace"):
        publish_committed_boundary(logical, payload)


def test_boundary_rejects_forged_scientific_commit_id(tmp_path) -> None:
    logical = tmp_path / "run-a"
    attempt_id = "a" * 32
    create_attempt_namespace(
        logical,
        run_id="run-a",
        attempt_id=attempt_id,
        attempt_count=1,
    )
    payload = _boundary(logical, attempt_id=attempt_id, rollout_id=7)
    payload["scientific_commit_id"] = f"boundary-sha256:{'f' * 64}"

    with pytest.raises(ValueError, match="content ID mismatch"):
        publish_committed_boundary(logical, payload)


def test_boundary_pointer_cannot_regress(tmp_path) -> None:
    logical = tmp_path / "run-a"
    attempt_id = "a" * 32
    create_attempt_namespace(
        logical,
        run_id="run-a",
        attempt_id=attempt_id,
        attempt_count=1,
    )
    publish_committed_boundary(
        logical,
        _boundary(logical, attempt_id=attempt_id, rollout_id=7),
    )

    with pytest.raises(ValueError, match="only advance"):
        publish_committed_boundary(
            logical,
            _boundary(logical, attempt_id=attempt_id, rollout_id=6),
        )


def test_successful_retry_freezes_only_committed_parent_prefix(tmp_path) -> None:
    logical = tmp_path / "run-a"
    first = "a" * 32
    second = "b" * 32
    create_attempt_namespace(
        logical,
        run_id="run-a",
        attempt_id=first,
        attempt_count=1,
    )
    publish_committed_boundary(
        logical,
        _boundary(logical, attempt_id=first, rollout_id=7),
    )
    parent = load_latest_committed_boundary(logical)
    assert parent is not None
    create_attempt_namespace(
        logical,
        run_id="run-a",
        attempt_id=second,
        attempt_count=2,
        parent_boundary=parent,
    )
    publish_committed_boundary(
        logical,
        _boundary(
            logical,
            attempt_id=second,
            rollout_id=19,
            pending=False,
            parent_commit_id=str(parent["scientific_commit_id"]),
            terminal=True,
        ),
    )

    path = write_accepted_lineage(
        logical,
        final_attempt_id=second,
        final_rollout_id=19,
    )
    assert path.name == ACCEPTED_LINEAGE
    payload = json.loads(path.read_text())
    assert payload["segments"] == [
        {
            "attempt_id": first,
            "attempt_root": f"attempts/{first}",
            "first_train_rollout_id": 0,
            "last_train_rollout_id": 7,
            "first_generation_rollout_id": 0,
            "last_generation_rollout_id": 8,
        },
        {
            "attempt_id": second,
            "attempt_root": f"attempts/{second}",
            "first_train_rollout_id": 8,
            "last_train_rollout_id": 19,
            "first_generation_rollout_id": 9,
            "last_generation_rollout_id": 19,
        },
    ]
    assert payload["pending_rollout_handoffs"] == [
        {
            "rollout_id": 8,
            "generation_attempt_id": first,
            "training_attempt_id": second,
        }
    ]
    assert (
        write_accepted_lineage(
            logical,
            final_attempt_id=second,
            final_rollout_id=19,
        )
        == path
    )


def test_attempt_ledger_marks_unreported_preemption_and_uses_fresh_ids() -> None:
    run = SimpleNamespace(
        metadata={},
        status=None,
        error_message="stale terminal error",
        ended_at=None,
        completed_at=None,
        duration_seconds=None,
    )
    assert mark_training_attempt_started(run, started_at=10) == 1
    first_id = run.metadata["active_attempt_id"]

    assert mark_training_attempt_started(run, started_at=20) == 2
    second_id = run.metadata["active_attempt_id"]
    assert first_id != second_id
    assert run.error_message is None
    assert run.metadata["attempts"][0]["status"] == (
        "interrupted_without_terminal_report"
    )
    assert run.metadata["attempts"][0]["ended_at"] == 20

    mark_training_attempt_finished(run, status="completed", ended_at=30)
    assert run.metadata["attempts"][1]["status"] == "completed"
    assert run.metadata["attempts"][1]["ended_at"] == 30


def test_terminal_retry_reports_the_parent_wandb_attempt_as_accepted() -> None:
    parent = "a" * 32
    retry = "b" * 32
    run = SimpleNamespace(
        metadata={
            "attempts": [
                {"attempt": 1, "attempt_id": parent},
                {"attempt": 2, "attempt_id": retry},
            ],
            "wandb_attempts": [
                {"attempt": 1, "run_id": "run-a1"},
                {"attempt": 2, "run_id": "run-a2"},
            ],
            "wandb_latest_run_id": "run-a2",
        }
    )

    selected = select_accepted_wandb_attempt(
        run,
        accepted_attempt_id=parent,
        skipped_attempt_count=2,
    )

    assert selected == "run-a1"
    assert run.metadata["wandb_accepted_run_id"] == "run-a1"
    assert run.metadata["wandb_latest_run_id"] == "run-a1"
    assert run.metadata["wandb_attempts"] == [
        {"attempt": 1, "run_id": "run-a1", "status": "accepted"},
        {
            "attempt": 2,
            "run_id": "run-a2",
            "status": "not_started_terminal_parent",
            "accepted_attempt_id": parent,
        },
    ]
