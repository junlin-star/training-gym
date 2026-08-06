from __future__ import annotations

import time
from enum import Enum
from typing import Any, Awaitable

from pydantic import BaseModel, Field

from modal_training_gym.common.config import modal_proxy_auth_headers
from modal_training_gym.common.status import SlimeStatus
from modal_training_gym.utils.metadata import MetadataStore, vol_list, vol_put


PROTOCOL = "training-gym-substep-timing"


class Role(str, Enum):
    DRIVER = "driver"
    ROLLOUT = "rollout"
    ACTOR = "actor"
    CRITIC = "critic"


class PhaseTiming(BaseModel):
    """How long one phase took, for the times it ran in one rollout.

    Offsets are seconds since the lane opened. ``total_duration_s`` is time spent
    in the phase: less than ``last_end_s - first_start_s`` when the phase ran
    repeatedly, more when its runs overlapped.

    ``invocations`` holds each run as ``[start_s, end_s]``, so a phase that ran a
    few times draws as those runs rather than one bar over all of them. It is
    empty for sampled phases, which are represented by aggregate statistics.
    """

    count: int
    total_duration_s: float
    longest_duration_s: float
    first_start_s: float
    last_end_s: float
    invocations: list[tuple[float, float]] = Field(default_factory=list)

    @property
    def average_duration_s(self) -> float:
        return self.total_duration_s / self.count if self.count else 0.0


class RoleTimingRecord(BaseModel):
    """One role's measured timing for one rollout.

    Single writer per key, whole-file overwrite, last write wins.
    """

    # A path component of the record's key, so it must not be "." or ".."
    training_run_id: str = Field(pattern=r"^[A-Za-z0-9_-][A-Za-z0-9._-]*$")
    rollout_id: int | None = Field(default=None, ge=0)
    role: Role
    created_at: int = 0
    lane_start_unix_s: float | None = None
    phases: dict[str, PhaseTiming] = Field(default_factory=dict)

    @property
    def storage_key(self) -> str:
        rollout = "bootstrap" if self.rollout_id is None else f"{self.rollout_id:08d}"
        return f"{rollout}__{self.role.value}"

    @staticmethod
    def store(training_run_id: str) -> str:
        return f"{MetadataStore.SUBSTEP_TIMING.value}/{training_run_id}"

    def _touch_created_at(self) -> None:
        if not self.created_at:
            self.created_at = int(time.time())

    def save(self, *, is_async: bool = False) -> None | Awaitable[None]:
        self._touch_created_at()
        return vol_put(
            self.store(self.training_run_id),
            self.storage_key,
            self.model_dump(mode="json"),
            is_async=is_async,
        )


class Substep(str, Enum):
    # Included in legacy substep times
    EVAL_BEFORE = SlimeStatus.EVAL_ROLLOUT_LOGGING.value
    GENERATE_ROLLOUTS = SlimeStatus.ROLLOUT_LOGGING.value
    OFFLOAD_ROLLOUT = SlimeStatus.OFFLOAD_ROLLOUT.value
    COMPUTE_LOG_PROBS = SlimeStatus.COMPUTE_LOG_PROBS.value
    OPTIMIZER_STEP = SlimeStatus.OPTIMIZER_STEP.value
    CHECKPOINT_SAVE = SlimeStatus.CHECKPOINT_SAVE.value
    OFFLOAD_TRAIN = SlimeStatus.OFFLOAD_TRAIN.value
    WEIGHT_SYNC = SlimeStatus.WEIGHT_SYNC.value
    EVAL_AFTER = f"{SlimeStatus.EVAL_ROLLOUT_LOGGING.value}_end"

    WAIT_FOR_ROLLOUT = "wait_for_rollout"  # driver, on this rollout's generation
    WAIT_FOR_NEXT_ROLLOUT = "wait_for_next_rollout"  # driver, on the next one's
    TRAIN_MODELS = "train_models"  # driver
    GENERATE_SAMPLES = "generate_samples"  # rollout worker
    SAMPLE_GENERATION = (
        "sample_generation"  # rollout worker, one run per generated sample
    )
    REWARD = "reward"  # rollout worker, one run per sample
    REWARD_BATCH = "reward_batch"  # rollout worker, one run per scored batch
    REWARD_POST_PROCESS = "reward_post_process"  # rollout worker
    FORWARD_BACKWARD = "forward_backward"  # actor / critic


def rollout_lanes(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Returns which roles need to be drawn for UI for a run.

    Keyed by role so the frontend indexes lanes by name rather than by position.
    """
    lanes = {
        record["role"]: {
            "role": record["role"],
            "lane_start_unix_s": record["lane_start_unix_s"],
            "phases": record["phases"],
        }
        for record in records
    }
    return {"roles": lanes}


def measured_run_times(
    training_run_id: str,
) -> tuple[
    dict[str, dict[str, int | None]], dict[str, dict[str, dict[str, float | None]]]
]:
    """How long each step of a run took, and each of its substeps.

    Both are keyed by step number from one, the way a stored baseline of a
    model's timings is, and a substep measured off the driver keeps its role in
    its key: the actor and the critic record the same phase names, and their
    times are not one substep.

    A step is the driver's substeps of a rollout added up: the driver runs them
    one after another, and the checkpoint or eval that landed on the rollout is
    left out rather than making the step read many times slower than its
    neighbours. The lanes are placed on the same wall clock
    (``lane_start_unix_s`` plus each phase's offsets), since they are measured in
    separate processes.
    """
    beside_the_step = (
        Substep.CHECKPOINT_SAVE.value,
        Substep.EVAL_BEFORE.value,
        Substep.EVAL_AFTER.value,
    )
    step_times: dict[str, dict[str, int | None]] = {}
    substep_times: dict[str, dict[str, dict[str, float | None]]] = {}
    for rollout_id, records in sorted(
        load_run(training_run_id).items(),
        key=lambda item: (item[0] is None, item[0] or 0),
    ):
        if rollout_id is None:
            continue
        substeps: dict[str, dict[str, float | None]] = {}
        step_duration = 0.0
        for record in records:
            lane_start = record["lane_start_unix_s"]
            role = record["role"]
            for name, phase in record["phases"].items():
                if role == Role.DRIVER.value and name not in beside_the_step:
                    step_duration += phase["total_duration_s"]
                key = name if role == Role.DRIVER.value else f"{name} ({role})"
                substeps[key] = {
                    "start": lane_start + phase["first_start_s"],
                    "duration_s": phase["total_duration_s"],
                }
        if not substeps:
            continue
        step = str(rollout_id + 1)
        step_times[step] = {"duration_s": round(step_duration)}
        substep_times[step] = substeps
    return step_times, substep_times


def load_run(training_run_id: str) -> dict[int | None, list[dict[str, Any]]]:
    """Every role record of a run, keyed by rollout id.

    Read whole, the way a run's rollouts are: the volume rate limits listings,
    so one listing per rollout fails as soon as a page wants a few rows, and one
    listing for the run is retried when it is the listing that is limited. A
    rollout with no records is left out; the run may predate measured timing, in
    which case the caller reads its legacy blob.
    """
    return _by_rollout(vol_list(RoleTimingRecord.store(training_run_id)))


async def load_run_async(
    training_run_id: str,
) -> dict[int | None, list[dict[str, Any]]]:
    """:func:`load_run` on the event loop, which reads the records together."""
    return _by_rollout(
        await vol_list(RoleTimingRecord.store(training_run_id), is_async=True)
    )


def _by_rollout(
    records: list[dict[str, Any]],
) -> dict[int | None, list[dict[str, Any]]]:
    steps: dict[int | None, list[dict[str, Any]]] = {}
    for record in records:
        steps.setdefault(record["rollout_id"], []).append(record)
    return steps


# ---------- Legacy handling --------------


def probe_substep_timing(framework_status_url: str, mode: str = "auto") -> bool:
    """Check the dashboard can accept timing records; return whether to enable.

    Called on the host before the training app is spawned, so ``"require"``
    fails while nothing is allocated rather than 40 GPUs in. ``"auto"`` warns
    and disables instead: a run is worth more than its telemetry.
    """
    import json
    import urllib.error
    import urllib.request

    if mode == "off":
        return False
    if not framework_status_url:
        if mode == "require":
            raise RuntimeError(
                "substep_timing='require' but this run has no dashboard URL to "
                "report timing to. Deploy the dashboard "
                "(`modal deploy dashboards/app.py`) or set substep_timing='auto'."
            )
        return False

    base = framework_status_url.rstrip("/")
    suffix = "/api/framework-status"
    if base.endswith(suffix):
        base = base[: -len(suffix)]
    url = f"{base}/api/timing-events"

    # A proxy-authenticated dashboard rejects an unsigned request at the proxy,
    # which would read here as a dashboard that cannot accept timing at all.
    request = urllib.request.Request(url, headers=modal_proxy_auth_headers())
    problem = ""
    try:
        with urllib.request.urlopen(request, timeout=5.0) as response:
            payload = json.loads(response.read() or b"{}")
        if payload.get("protocol") != PROTOCOL:
            problem = f"{url} did not identify as {PROTOCOL!r}"
    except urllib.error.HTTPError as exc:
        # 404 means an older dashboard without the route; anything else is a
        # live dashboard failing, for which "redeploy" is the wrong advice.
        problem = (
            f"{url} returned 404 -- the deployed dashboard predates substep timing"
            if exc.code == 404
            else f"{url} returned HTTP {exc.code}"
        )
    except (OSError, ValueError) as exc:
        problem = f"could not reach {url}: {exc}"

    if not problem:
        return True
    if mode == "require":
        raise RuntimeError(
            f"substep_timing='require' but {problem}. Redeploy the dashboard "
            "(`modal deploy dashboards/app.py`) or set substep_timing='auto'."
        )
    print(f"WARNING: substep timing disabled for this run -- {problem}")
    return False


# Old step timing method grouped general training step i.e. forward/backward
# under "optimizer step". New view more accurately reflects this.
_LEGACY_RENAMES = {Substep.OPTIMIZER_STEP.value: Substep.TRAIN_MODELS.value}


def legacy_run_to_records(
    substep_times: dict[str, dict[str, dict[str, float | None]]] | None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for step_key, subs in (substep_times or {}).items():
        if not step_key.isdigit():
            continue
        lane_start = min(
            (sub["start"] for sub in subs.values() if sub.get("start") is not None),
            default=None,
        )
        if lane_start is None:
            continue
        phases: dict[str, dict[str, float]] = {}
        for name, sub in subs.items():
            start, duration = sub.get("start"), sub.get("duration_s")
            if start is None or duration is None:
                continue
            rel = start - lane_start
            phases[_LEGACY_RENAMES.get(name, name)] = {
                "count": 1,
                "total_duration_s": round(duration, 6),
                "longest_duration_s": round(duration, 6),
                "first_start_s": round(rel, 6),
                "last_end_s": round(rel + duration, 6),
            }
        records.append(
            {
                "rollout_id": int(step_key) - 1,
                "role": Role.DRIVER.value,
                "lane_start_unix_s": lane_start,
                "phases": phases,
            }
        )
    return records
