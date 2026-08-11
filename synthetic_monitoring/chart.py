"""Timing history storage and PNG chart for synmon Slack reports."""

import io
import re
from dataclasses import dataclass
from datetime import datetime, timezone

import modal

_STEP_SUFFIX = re.compile(r" \(step \d+\)$")
_TOTAL_KEYS = ("Total", "Total duration")

HISTORY_DICT_NAME = "gym-synmon-timing-baselines"

_GRAY_00 = "#181818"
_GRAY_08 = "#222222"
_GRAY_15 = "#272727"
_GRAY_20 = "#2f2f2f"
_GRAY_30 = "#464646"
_GRAY_40 = "#747474"
_GRAY_50 = "#a3a3a3"
_GRAY_70 = "#d1d1d1"
_GREEN_50 = "#6ac345"
_GREEN_70 = "#63cd93"
_GREEN_80 = "#7fee64"
_RED_75 = "#f87171"
_YELLOW_60 = "#d1c05f"
_BLUE_60 = "#79a4c4"
_PHASE_COLORS = (
    _BLUE_60,
    _GREEN_50,
    _YELLOW_60,
    _RED_75,
    _GREEN_70,
    _GRAY_50,
    _GRAY_40,
    "#c4a27a",
)

_history_store: dict[str, modal.Dict] = {}


@dataclass(frozen=True)
class RunPoint:
    ts: float
    timings: dict[str, float]
    training_run_id: str = ""
    succeeded: bool = True
    total_duration_s: float = 0.0


def _history(environment_name: str) -> modal.Dict:
    cached = _history_store.get(environment_name)
    if cached is None:
        cached = modal.Dict.from_name(
            HISTORY_DICT_NAME,
            create_if_missing=True,
            environment_name=environment_name,
        )
        _history_store[environment_name] = cached
    return cached


def load_history(model_name: str, *, environment_name: str) -> list[RunPoint]:
    raw = _history(environment_name).get(model_name)
    if not isinstance(raw, list):
        return []
    points: list[RunPoint] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        timings_raw = item.get("timings")
        if not isinstance(timings_raw, dict):
            continue
        timings: dict[str, float] = {}
        for key, value in timings_raw.items():
            try:
                timings[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
        if not timings:
            continue
        try:
            ts = float(item.get("ts") or 0.0)
        except (TypeError, ValueError):
            ts = 0.0
        total = next(
            (timings[key] for key in _TOTAL_KEYS if key in timings),
            None,
        )
        try:
            total_duration_s = float(
                item.get("total_duration_s")
                if item.get("total_duration_s") is not None
                else (total or 0.0)
            )
        except (TypeError, ValueError):
            total_duration_s = float(total or 0.0)
        points.append(
            RunPoint(
                ts=ts,
                timings=timings,
                training_run_id=str(item.get("training_run_id") or ""),
                succeeded=bool(item.get("succeeded", True)),
                total_duration_s=total_duration_s,
            )
        )
    return points


def append_history(
    model_name: str, point: RunPoint, *, environment_name: str
) -> list[RunPoint]:
    points = load_history(model_name, environment_name=environment_name)
    if point.training_run_id and any(
        p.training_run_id == point.training_run_id for p in points
    ):
        return points
    points.append(point)
    points.sort(key=lambda p: (p.ts, p.training_run_id))
    _history(environment_name)[model_name] = [
        {
            "ts": p.ts,
            "timings": dict(p.timings),
            "training_run_id": p.training_run_id,
            "succeeded": p.succeeded,
            "total_duration_s": p.total_duration_s,
        }
        for p in points
    ]
    return points


def _legend_label(label: str) -> str:
    return _STEP_SUFFIX.sub("", label)


def _is_total_label(label: str) -> bool:
    return label in _TOTAL_KEYS


def _total_from_point(point: RunPoint) -> float | None:
    for key in _TOTAL_KEYS:
        if key in point.timings:
            return float(point.timings[key])
    if point.total_duration_s:
        return float(point.total_duration_s)
    return None


def _style_history_axes(ax, *, xlabel: str, use_run_index: bool, xs_num) -> None:
    from matplotlib.dates import DateFormatter

    ax.set_facecolor(_GRAY_08)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=_GRAY_20, linewidth=0.8)
    ax.xaxis.grid(False)
    ax.tick_params(colors=_GRAY_50, length=3, width=0.8)
    ax.xaxis.label.set_color(_GRAY_50)
    ax.yaxis.label.set_color(_GRAY_50)
    ax.title.set_color(_GRAY_70)
    for spine in ax.spines.values():
        spine.set_color(_GRAY_30)
        spine.set_linewidth(0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel(xlabel)
    if use_run_index:
        ax.set_xticks(xs_num)
    else:
        ax.xaxis.set_major_formatter(DateFormatter("%m-%d %H:%M"))


def render_timing_history_chart(
    history: list[RunPoint],
    *,
    model_name: str,
) -> bytes:
    from matplotlib import pyplot

    labels: list[str] = []
    seen: set[str] = set()
    for point in history:
        for label in point.timings:
            if _is_total_label(label) or label in seen:
                continue
            seen.add(label)
            labels.append(label)
    phase_labels = [label for label in labels if not label.startswith("Step ")]
    step_labels = [label for label in labels if label.startswith("Step ")]

    totals = [_total_from_point(point) for point in history]
    has_phases = bool(phase_labels)
    has_steps = bool(step_labels)
    has_totals = any(v is not None for v in totals)
    if not has_phases and not has_steps and not has_totals:
        raise ValueError(f"no chartable timings for {model_name}")

    xs = [
        datetime.fromtimestamp(p.ts, tz=timezone.utc)
        if p.ts > 0
        else datetime.fromtimestamp(i + 1, tz=timezone.utc)
        for i, p in enumerate(history)
    ]
    use_run_index = all(p.ts <= 0 for p in history)
    xs_num = list(range(1, len(history) + 1)) if use_run_index else None
    x_plot = xs_num if use_run_index else xs
    xlabel = "run" if use_run_index else "time (UTC)"

    nrows = int(has_phases) + int(has_steps) + int(has_totals)
    height_ratios = (
        [2.2] * int(has_phases) + [1.0] * (nrows - int(has_phases)) if nrows else [1.0]
    )
    fig, axes = pyplot.subplots(
        nrows,
        1,
        figsize=(10.5, 4.2 * nrows + 0.6),
        sharex=True,
        gridspec_kw={"height_ratios": height_ratios},
    )
    if nrows == 1:
        axes = [axes]
    fig.patch.set_facecolor(_GRAY_00)
    fig.suptitle(
        f"{model_name} (n={len(history)} runs)",
        color=_GRAY_70,
        fontsize=13,
        fontweight="medium",
    )
    ax_i = 0

    if has_phases:
        ax = axes[ax_i]
        ax_i += 1
        for color_i, label in enumerate(phase_labels):
            ys = [point.timings.get(label) for point in history]
            if all(v is None for v in ys):
                continue
            y_vals = [float("nan") if v is None else float(v) for v in ys]
            ax.plot(
                x_plot,
                y_vals,
                color=_PHASE_COLORS[color_i % len(_PHASE_COLORS)],
                linewidth=1.5,
                marker="o",
                markersize=5,
                label=_legend_label(label),
                zorder=2,
            )
        ax.set_title("Substeps")
        ax.set_ylabel("seconds")
        _style_history_axes(
            ax, xlabel=xlabel, use_run_index=use_run_index, xs_num=xs_num
        )
        ax.legend(
            loc="upper left",
            framealpha=0.95,
            fontsize=8,
            facecolor=_GRAY_15,
            edgecolor=_GRAY_30,
            labelcolor=_GRAY_70,
            ncol=2,
        )

    if has_steps:
        ax = axes[ax_i]
        ax_i += 1
        for label in step_labels:
            ys = [point.timings.get(label) for point in history]
            if all(v is None for v in ys):
                continue
            y_vals = [float("nan") if v is None else float(v) for v in ys]
            ax.plot(
                x_plot,
                y_vals,
                color=_GREEN_80,
                linewidth=2.4,
                marker="o",
                markersize=5,
            )
        ax.set_title("Step")
        ax.set_ylabel("seconds")
        _style_history_axes(
            ax, xlabel=xlabel, use_run_index=use_run_index, xs_num=xs_num
        )

    if has_totals:
        ax = axes[ax_i]
        y_vals = [float("nan") if v is None else float(v) for v in totals]
        ax.plot(
            x_plot,
            y_vals,
            color=_GREEN_80,
            linewidth=2.4,
            marker="o",
            markersize=6,
        )
        ax.fill_between(x_plot, y_vals, color=_GREEN_80, alpha=0.12)
        ax.set_title("Total")
        ax.set_ylabel("seconds")
        _style_history_axes(
            ax, xlabel=xlabel, use_run_index=use_run_index, xs_num=xs_num
        )

    if not use_run_index:
        fig.autofmt_xdate(rotation=20, ha="right")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight", facecolor=_GRAY_00)
    pyplot.close(fig)
    return buf.getvalue()
