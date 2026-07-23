from collections.abc import Collection, Mapping, Sequence

from modal_training_gym.common.timing_types import StepTimes, Substep, SubstepTimes


def aggregate_sync_step_times(
    recorded_timestamps: Mapping[str, float | None],
    training_run_id: str,
    num_steps: int,
    substep_order: Sequence[str],
    optional_substeps: Collection[str],
) -> tuple[StepTimes, SubstepTimes]:
    """Organize step and substep timings from the Modal dict at the end of a run.

    Records each step's start/end/duration; missing timestamps become None.
    Each substep's duration is the gap from its start to the next recorded
    substep's start (or the step's end for the last one). Duration is None
    if a mandatory substep in between is missing. Step duration is computed
    independently, since substeps include work outside the step (evals,
    checkpointing).
    """
    step_times: StepTimes = {}
    substep_times: SubstepTimes = {}

    for current_step_num in range(1, num_steps + 1):
        start_key = f"{training_run_id}:{current_step_num}:start"
        finish_key = f"{training_run_id}:{current_step_num}:finish"

        raw_start_time = recorded_timestamps.get(start_key)
        raw_end_time = recorded_timestamps.get(finish_key)
        precise_start_time = (
            float(raw_start_time) if raw_start_time is not None else None
        )
        precise_end_time = float(raw_end_time) if raw_end_time is not None else None
        start_time = int(precise_start_time) if precise_start_time is not None else None
        end_time = int(precise_end_time) if precise_end_time is not None else None

        duration = None
        if start_time is not None and end_time is not None:
            duration = end_time - start_time

        step_times[str(current_step_num)] = {
            "start": start_time,
            "end": end_time,
            "duration_s": duration,
        }

        raw_step_window_start = recorded_timestamps.get(
            f"{training_run_id}:{current_step_num}:substep_start"
        )
        step_window_start = (
            float(raw_step_window_start) if raw_step_window_start is not None else None
        )
        full_step_start_time = (
            step_window_start if step_window_start is not None else precise_start_time
        )
        full_step_end_time = recorded_timestamps.get(
            f"{training_run_id}:{current_step_num}:substep_finish"
        )
        if full_step_end_time is not None:
            full_step_end_time = float(full_step_end_time)
            if step_window_start is not None and full_step_end_time < step_window_start:
                full_step_end_time = precise_end_time
        else:
            full_step_end_time = precise_end_time

        step_key = str(current_step_num)
        substep_times[step_key] = {}
        eval_before = Substep.EVAL_BEFORE.value
        present: set[str] = set()
        recorded: list[tuple[float, int, str]] = []
        for order_idx, substep in enumerate(substep_order):
            substep_start = recorded_timestamps.get(
                f"{training_run_id}:{current_step_num}:substep:{substep}"
            )
            if substep_start is None:
                continue
            substep_start = float(substep_start)
            if (
                step_window_start is not None
                and substep_start < step_window_start
                and substep != eval_before
            ):
                continue
            if full_step_start_time is not None and substep != eval_before:
                substep_start = max(substep_start, full_step_start_time)
            if full_step_end_time is not None:
                substep_start = min(substep_start, full_step_end_time)
            present.add(substep)
            recorded.append((substep_start, order_idx, substep))
        recorded.sort()

        for idx, (substep_start, order_idx, substep) in enumerate(recorded):
            if idx + 1 < len(recorded):
                next_start, next_idx = recorded[idx + 1][0], recorded[idx + 1][1]
            else:
                next_start, next_idx = full_step_end_time, len(substep_order)

            gap = substep_order[order_idx + 1 : next_idx]
            dropped_mandatory = any(
                name not in optional_substeps and name not in present for name in gap
            )
            if next_start is None or dropped_mandatory:
                substep_duration = None
            else:
                substep_duration = round(max(next_start - substep_start, 0.0), 3)

            substep_times[step_key][substep] = {
                "start": round(substep_start, 3),
                "duration_s": substep_duration,
            }

    return step_times, substep_times
