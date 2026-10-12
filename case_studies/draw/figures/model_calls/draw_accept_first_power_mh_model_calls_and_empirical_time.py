#!/usr/bin/env python3
"""Draw the combined SubTreeMH-versus-PowerMH call/time figure.

Run with:

    MPLCONFIGDIR=/private/tmp/power-sharpening-mplconfig \
    XDG_CACHE_HOME=/private/tmp/power-sharpening-xdg-cache \
    /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/.venv/bin/python \
    /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/draw/draw_accept_first_power_mh_model_calls_and_empirical_time.py \
    --directory /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/logs/math500/2026-08-13/vllm

The script reads the raw ``accept_first`` SubTreeMH logs for every available prefetch budget and the matching PowerMH
timing log. It independently retains the highest-terminal-mean budget for MH transitions per model call and the
lowest-terminal-mean budgets for cumulative empirical time and empirical time per call, then writes one three-panel PDF.

For a single budget or another rank policy, pass ``--subtree-log /absolute/path/to/run.log`` and
``--power-mh-log /absolute/path/to/baseline.log``. The rank and budget are read from the subtree filename.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil, floor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.cbook import pts_to_poststep
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea, HPacker, TextArea
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator

from case_studies import paths
from case_studies.draw.common.figure_io import save_figure
from case_studies.extract.logs.model_call_traces import (
    MODEL_CALL_OUTPUT_SUFFIX,
    POWER_MH_LOG_GLOB,
    AnalysisRows,
    BatchComparison,
    ExperimentGroup,
    NoModelCallTracesError,
    Row,
    TraceKey,
    build_analysis_rows,
    build_batch_comparisons,
    build_power_mh_time_per_call_summary_rows,
    build_power_mh_time_summary_rows,
    discover_experiment_groups,
    experiment_figure_title,
    extract_logs,
    group_rows_by_rank,
    group_rows_by_trace,
    map_trace_ranks,
    power_mh_time_per_call_trajectories,
    power_mh_trace_times,
)
from case_studies.extract.run_naming import (
    RANK_ORDER,
    backend_of,
    canonical_rank,
    dataset_model_prefix,
    dataset_of,
    model_display_name,
    prefetch_budget_of,
    rank_of,
    shared_run_config_of,
)
from case_studies.plot_config import (
    BLACK,
    DARK_GRAY,
    POWER_METHOD_LABEL,
    SUBTREE_METHOD_LABEL,
    TRANSITION_RANK_COLORS,
    TRANSPARENT,
    apply_plot_style,
)
from case_studies.plot_config import (
    BLUE as POWER_COLOR,
)
from case_studies.plot_config import (
    GREEN as SUBTREE_COLOR,
)

apply_plot_style()

DEFAULT_DIRECTORY = (
    paths.LOGS_DIR / "math500" / "2026-08-13" / "vllm"
)
DEFAULT_TRACES_PER_RANK = 10
RANK_NAME = "accept_first"
FIGURE_SUFFIX = (
    ".all-prefetch-budget.accept-first-power-mh."
    "model-calls-and-empirical-time.pdf"
)

FIGURE_FONT_SIZE = 12.0
LEGEND_FONT_SIZE = 11.0
SUBTREE_LABEL_FONT_SIZE = FIGURE_FONT_SIZE
PANEL_WIDTH = 3.7
FIGURE_HEIGHT = 3.60
LEFT_MARGIN_INCHES = 0.82
BOTTOM_MARGIN_INCHES = 0.55
PANEL_WSPACE = 0.26
PLOT_TOP = 0.775
TITLE_Y = 0.945
LEGEND_Y = 0.896
PANEL_LABEL_Y = 1.04
Y_LABEL_Y = 0.45

RAW_TRACE_LINE_WIDTH = 0.85
RAW_TRACE_ALPHA = 0.48
MEAN_LINE_WIDTH = 2.2
REFERENCE_LINE_WIDTH = 1.35
SD_BAND_ALPHA = 0.18
STATISTIC_LINE_SPACING = 1.35

TRANSITION_AXIS_LABEL = r"total MH transitions ($K$)"
CALL_AXIS_LABEL = "cumulative number of model calls"
TIME_AXIS_LABEL = "cumulative model-call time (min)"
TIME_PER_CALL_AXIS_LABEL = "model-call time (sec/call)"
TRANSITIONS_PER_CALL_AXIS_LABEL = "MH transitions per model call"
EFFICIENCY_Y_PADDING = 0.5
# Fixed upper y-limit of the transitions-per-call panel (--transitions-per-call-ymax), so that figures stacked from
# several runs share one scale; None keeps the data-driven limit.
TRANSITIONS_PER_CALL_YMAX: float | None = None
TIME_PER_CALL_Y_HEADROOM = 1.20
SUBTREE_TOP_TEXT_X = 0.50
SUBTREE_TOP_TEXT_Y = 0.95
POWER_EFFICIENCY_TEXT_Y = 0.13
SECONDS_PER_MINUTE = 60.0
# In-panel statistics sit on a translucent white box so no trace strikes through them, and are moved (from the
# candidate grid below) to the spot with the fewest crossing line segments; ties keep the default position.
STATISTIC_BOX = {"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.85}
STATISTIC_CANDIDATE_X = ((0.03, "left"), (0.50, "center"), (0.97, "right"))
STATISTIC_CANDIDATE_Y = tuple(0.95 - 0.075 * index for index in range(13))
# Clearance (points) kept between a panel letter and the y tick labels it would otherwise touch.
PANEL_LABEL_CLEARANCE_POINTS = 3.0


def seconds_to_minutes(value: float) -> float:
    """Convert a duration from seconds to minutes."""
    return value / SECONDS_PER_MINUTE


def convert_time_rows_to_minutes(
    rows: Sequence[Row],
    *value_keys: str,
) -> list[Row]:
    """Copy empirical-time rows while converting selected values to minutes."""
    converted_rows: list[Row] = []
    for row in rows:
        converted_row = row.copy()
        for value_key in value_keys:
            converted_row[value_key] = seconds_to_minutes(
                float(row[value_key])
            )
        converted_rows.append(converted_row)
    return converted_rows


@dataclass(frozen=True)
class TrajectoryKind:
    """Column mapping and axis properties for one cumulative cost."""

    trajectory_attribute: str
    summary_attribute: str
    value_key: str
    mean_key: str
    sd_key: str
    axis_label: str
    include_power_mh: bool
    integer_cost_ticks: bool

    def trajectory_rows(self, analysis: AnalysisRows) -> list[Row]:
        """Return the per-trace rows for this cost."""
        return getattr(analysis, self.trajectory_attribute)

    def summary_rows(self, analysis: AnalysisRows) -> list[Row]:
        """Return the mean/SD rows for this cost."""
        return getattr(analysis, self.summary_attribute)


@dataclass(frozen=True)
class PowerMHTimeData:
    """Raw and summarized timing trajectories from one PowerMH log."""

    log_path: Path
    trajectories: list[tuple[str, list[float]]]
    summary: list[Row]
    time_per_call_trajectories: list[tuple[str, list[float]]]
    time_per_call_summary: list[Row]


MODEL_CALLS = TrajectoryKind(
    trajectory_attribute="cumulative",
    summary_attribute="summary",
    value_key="subtree_cumulative_calls",
    mean_key="subtree_mean_calls",
    sd_key="subtree_std_calls",
    axis_label=CALL_AXIS_LABEL,
    include_power_mh=True,
    integer_cost_ticks=True,
)
EMPIRICAL_TIME = TrajectoryKind(
    trajectory_attribute="empirical_time",
    summary_attribute="empirical_time_summary",
    value_key="cumulative_empirical_seconds",
    mean_key="mean_cumulative_seconds",
    sd_key="std_cumulative_seconds",
    axis_label=TIME_AXIS_LABEL,
    include_power_mh=False,
    integer_cost_ticks=False,
)
TIME_PER_CALL = TrajectoryKind(
    trajectory_attribute="time_per_call",
    summary_attribute="time_per_call_summary",
    value_key="empirical_seconds_per_call",
    mean_key="mean_empirical_seconds_per_call",
    sd_key="std_empirical_seconds_per_call",
    axis_label=TIME_PER_CALL_AXIS_LABEL,
    include_power_mh=False,
    integer_cost_ticks=False,
)
TRANSITIONS_PER_CALL = TrajectoryKind(
    trajectory_attribute="transitions_per_call",
    summary_attribute="transitions_per_call_summary",
    value_key="mh_transitions_per_call",
    mean_key="mean_mh_transitions_per_call",
    sd_key="std_mh_transitions_per_call",
    axis_label=TRANSITIONS_PER_CALL_AXIS_LABEL,
    include_power_mh=False,
    integer_cost_ticks=False,
)


def load_power_mh_time_data(log_path: Path) -> PowerMHTimeData:
    """Parse complete PowerMH timing traces and their sample statistics."""
    trajectories = power_mh_trace_times(log_path)
    return PowerMHTimeData(
        log_path=log_path,
        trajectories=trajectories,
        summary=build_power_mh_time_summary_rows(trajectories),
        time_per_call_trajectories=(
            power_mh_time_per_call_trajectories(trajectories)
        ),
        time_per_call_summary=(
            build_power_mh_time_per_call_summary_rows(trajectories)
        ),
    )


def best_analysis(
    batch_analyses: Sequence[tuple[int, AnalysisRows]],
    kind: TrajectoryKind,
    *,
    maximize: bool = False,
    rank_name: str = RANK_NAME,
) -> tuple[int, AnalysisRows]:
    """Choose the best terminal mean, then SD and budget."""
    candidates: list[tuple[float, float, int, AnalysisRows]] = []
    for prefetch_budget, analysis in batch_analyses:
        rank_rows = [
            row
            for row in kind.summary_rows(analysis)
            if str(row["rank"]) == rank_name
        ]
        if not rank_rows:
            continue
        final_iteration = max(int(row["mh_iteration"]) for row in rank_rows)
        terminal_rows = [
            row
            for row in rank_rows
            if int(row["mh_iteration"]) == final_iteration
        ]
        if len(terminal_rows) != 1:
            raise ValueError(
                f"Expected one terminal summary at prefetch budget "
                f"{prefetch_budget}, got {len(terminal_rows)}"
            )
        terminal = terminal_rows[0]
        candidates.append(
            (
                (
                    -float(terminal[kind.mean_key])
                    if maximize
                    else float(terminal[kind.mean_key])
                ),
                float(terminal[kind.sd_key]),
                prefetch_budget,
                analysis,
            )
        )
    if not candidates:
        raise ValueError(f"No {rank_name!r} terminal summary is available")
    _, _, prefetch_budget, analysis = min(
        candidates, key=lambda item: (item[0], item[1], item[2])
    )
    return prefetch_budget, analysis


def trajectory_limits(
    trajectory_rows: Sequence[Row],
    summary_rows: Sequence[Row],
    kind: TrajectoryKind,
) -> tuple[tuple[float, float], tuple[float, float], int]:
    """Return value and iteration limits that include traces and SD bands."""
    if not trajectory_rows or not summary_rows:
        raise ValueError("Trajectory limits require non-empty rows")
    final_iteration = max(int(row["mh_iteration"]) for row in trajectory_rows)
    maximum_cost = max(
        max(float(row[kind.value_key]) for row in trajectory_rows),
        max(
            float(row[kind.mean_key]) + float(row[kind.sd_key])
            for row in summary_rows
        ),
    )
    if kind.include_power_mh:
        maximum_cost = max(maximum_cost, float(final_iteration))
    cost_limits = (0.0, max(maximum_cost * 1.03, 1.0))
    if kind is TRANSITIONS_PER_CALL:
        minimum_cost = min(
            1.0,
            min(float(row[kind.value_key]) for row in trajectory_rows),
            min(
                float(row[kind.mean_key]) - float(row[kind.sd_key])
                for row in summary_rows
            ),
        )
        cost_limits = (
            max(0.0, floor(minimum_cost * 2) / 2 - EFFICIENCY_Y_PADDING),
            ceil(max(maximum_cost, 1.0) * 2) / 2 + EFFICIENCY_Y_PADDING,
        )
    return (
        cost_limits,
        (0.0, float(final_iteration)),
        final_iteration,
    )


def style_panel(
    ax: Axes,
    kind: TrajectoryKind,
    *,
    cost_limits: tuple[float, float],
    iteration_limits: tuple[float, float],
) -> None:
    """Apply the shared linear scales and major-tick treatment."""
    ax.set_xlim(*iteration_limits)
    ax.set_ylim(*cost_limits)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    ax.yaxis.set_major_locator(
        MaxNLocator(nbins=5, integer=kind.integer_cost_ticks)
    )
    ax.tick_params(
        axis="both",
        which="both",
        top=False,
        right=False,
        labeltop=False,
        labelright=False,
        length=3,
        width=0.8,
    )
    ax.set_xlabel(
        TRANSITION_AXIS_LABEL,
        fontsize=FIGURE_FONT_SIZE,
        labelpad=0,
    )
    ax.set_ylabel(kind.axis_label, fontsize=FIGURE_FONT_SIZE, y=Y_LABEL_Y)


def draw_subtree_trajectories(
    ax: Axes,
    rows_by_trace: dict[TraceKey, list[Row]],
    rank_by_trace: dict[TraceKey, str],
    summary_by_rank: dict[str, list[Row]],
    kind: TrajectoryKind,
    *,
    rank_name: str = RANK_NAME,
) -> tuple[list[int], list[float], list[float]]:
    """Draw SubTreeMH traces, their mean, and a sample-SD band."""
    for trace_key in sorted(
        key for key, trace_rank in rank_by_trace.items() if trace_rank == rank_name
    ):
        trace_rows = rows_by_trace[trace_key]
        ax.step(
            [int(row["mh_iteration"]) for row in trace_rows],
            [float(row[kind.value_key]) for row in trace_rows],
            where="post",
            color=SUBTREE_COLOR,
            linewidth=RAW_TRACE_LINE_WIDTH,
            alpha=RAW_TRACE_ALPHA,
            zorder=2,
        )

    rank_summary = summary_by_rank[rank_name]
    iterations = [int(row["mh_iteration"]) for row in rank_summary]
    means = [float(row[kind.mean_key]) for row in rank_summary]
    deviations = [float(row[kind.sd_key]) for row in rank_summary]
    lower = [
        max(mean - deviation, 0.0)
        for mean, deviation in zip(means, deviations, strict=True)
    ]
    upper = [
        mean + deviation
        for mean, deviation in zip(means, deviations, strict=True)
    ]
    ax.fill_between(
        iterations,
        lower,
        upper,
        step="post",
        color=SUBTREE_COLOR,
        alpha=SD_BAND_ALPHA,
        linewidth=0,
        zorder=1,
    )
    ax.step(
        iterations,
        means,
        where="post",
        color=SUBTREE_COLOR,
        linewidth=MEAN_LINE_WIDTH,
        zorder=4,
    )
    return iterations, means, deviations


def draw_power_mh_call_reference(ax: Axes, final_iteration: int) -> list[int]:
    """Draw PowerMH's exact one-call-per-transition trajectory."""
    steps = list(range(final_iteration + 1))
    ax.step(
        steps,
        steps,
        where="post",
        color=POWER_COLOR,
        linewidth=REFERENCE_LINE_WIDTH,
        linestyle="--",
        zorder=5,
    )
    return steps


def draw_power_mh_transitions_per_call_reference(
    ax: Axes, final_iteration: int
) -> list[float]:
    """Draw PowerMH's exact one-transition-per-call trajectory."""
    iterations = list(range(1, final_iteration + 1))
    values = [1.0] * final_iteration
    ax.step(
        iterations,
        values,
        where="post",
        color=POWER_COLOR,
        linewidth=REFERENCE_LINE_WIDTH,
        linestyle="--",
        zorder=5,
    )
    return values


def draw_power_mh_cost_trajectories(
    ax: Axes,
    trajectories: Sequence[tuple[str, list[float]]],
    iterations: Sequence[int],
    means: Sequence[float],
    deviations: Sequence[float],
) -> tuple[list[int], list[float], list[float]]:
    """Draw PowerMH cost traces, their mean, and sample-SD band."""
    for _, values in trajectories:
        trace_iterations = list(range(1, len(values) + 1))
        ax.step(
            trace_iterations,
            values,
            where="post",
            color=POWER_COLOR,
            linewidth=RAW_TRACE_LINE_WIDTH,
            alpha=RAW_TRACE_ALPHA,
            zorder=2,
        )

    lower = [
        max(mean - deviation, 0.0)
        for mean, deviation in zip(means, deviations, strict=True)
    ]
    upper = [
        mean + deviation
        for mean, deviation in zip(means, deviations, strict=True)
    ]
    ax.fill_between(
        iterations,
        lower,
        upper,
        step="post",
        color=POWER_COLOR,
        alpha=SD_BAND_ALPHA,
        linewidth=0,
        zorder=1,
    )
    ax.step(
        iterations,
        means,
        where="post",
        color=POWER_COLOR,
        linewidth=MEAN_LINE_WIDTH,
        linestyle="--",
        zorder=4,
    )
    return list(iterations), list(means), list(deviations)


def draw_power_mh_time_trajectories(
    ax: Axes,
    data: PowerMHTimeData,
) -> tuple[list[int], list[float], list[float]]:
    """Draw PowerMH cumulative timing trajectories in minutes."""
    return draw_power_mh_cost_trajectories(
        ax,
        [
            (
                trace_id,
                [seconds_to_minutes(value) for value in cumulative],
            )
            for trace_id, cumulative in data.trajectories
        ],
        [int(row["mh_iteration"]) for row in data.summary],
        [
            seconds_to_minutes(float(row["mean_cumulative_seconds"]))
            for row in data.summary
        ],
        [
            seconds_to_minutes(float(row["std_cumulative_seconds"]))
            for row in data.summary
        ],
    )


def draw_power_mh_time_per_call_trajectories(
    ax: Axes,
    data: PowerMHTimeData,
) -> tuple[list[int], list[float], list[float]]:
    """Draw PowerMH running empirical seconds per proposal-model call."""
    return draw_power_mh_cost_trajectories(
        ax,
        data.time_per_call_trajectories,
        [int(row["mh_iteration"]) for row in data.time_per_call_summary],
        [
            float(row["mean_empirical_seconds_per_call"])
            for row in data.time_per_call_summary
        ],
        [
            float(row["std_empirical_seconds_per_call"])
            for row in data.time_per_call_summary
        ],
    )


def statistic_text(
    ax: Axes,
    x: float,
    y: float,
    method: str,
    mean: float,
    sd: float,
    **kwargs: object,
) -> None:
    """Write the terminal mean with its plus/minus sample SD as a subscript."""
    ax.text(
        x,
        y,
        rf"{method} ${mean:.2f}_{{\pm {sd:.2f}}}$",
        transform=ax.transAxes,
        fontsize=(SUBTREE_LABEL_FONT_SIZE if method == SUBTREE_METHOD_LABEL else FIGURE_FONT_SIZE),
        fontweight="semibold",
        linespacing=STATISTIC_LINE_SPACING,
        zorder=6,
        **kwargs,
    )


def add_method_legends(
    figure: Figure,
    *,
    left: float,
    right: float,
    legend_y: float = LEGEND_Y,
    methods: Sequence[str] | None = None,
) -> None:
    """Add one single-row, color-matched legend box for each method.

    Args:
        figure: Figure receiving the two legend boxes.
        left: Left edge of the plotting area in figure coordinates.
        right: Right edge of the plotting area in figure coordinates.
        legend_y: Top of both legend boxes in figure coordinates.
        methods: Optional subset of method names; a single legend is centered.

    Returns:
        None. The legend artists are added to the figure.
    """
    plot_span = right - left
    entries_to_draw = [entry for entry in (
        (POWER_METHOD_LABEL, POWER_COLOR, "--", 0.28),
        (SUBTREE_METHOD_LABEL, SUBTREE_COLOR, "-", 0.72),
    ) if methods is None or entry[0] in methods]
    for method, color, mean_style, center_fraction in entries_to_draw:
        if len(entries_to_draw) == 1:
            center_fraction = 0.5
        entries = [
            TextArea(
                method,
                textprops={
                    "color": color,
                    "fontsize": LEGEND_FONT_SIZE,
                },
            )
        ]
        handle_width = 1.35 * LEGEND_FONT_SIZE
        for label, artist in (
            ("trace", Line2D(
                [0, handle_width], [6, 6], color=color,
                linewidth=RAW_TRACE_LINE_WIDTH, alpha=RAW_TRACE_ALPHA,
            )),
            ("mean", Line2D(
                [0, handle_width], [6, 6], color=color,
                linewidth=MEAN_LINE_WIDTH, linestyle=mean_style,
            )),
            (r"$\pm 1$ SD", Rectangle(
                (0, 1), handle_width, 10, facecolor=color,
                edgecolor=TRANSPARENT, alpha=SD_BAND_ALPHA,
            )),
        ):
            handle = DrawingArea(handle_width, 12)
            handle.add_artist(artist)
            entries.append(HPacker(
                children=[handle, TextArea(label, textprops={
                    "color": color, "fontsize": LEGEND_FONT_SIZE,
                })],
                align="center", pad=0, sep=0.35 * LEGEND_FONT_SIZE,
            ))
        legend = AnchoredOffsetbox(
            child=HPacker(
                children=entries, align="center", pad=0, sep=LEGEND_FONT_SIZE,
            ),
            loc="upper center",
            bbox_to_anchor=(left + center_fraction * plot_span, legend_y),
            bbox_transform=figure.transFigure,
            frameon=True,
            prop={"size": LEGEND_FONT_SIZE},
            pad=0.4,
            borderpad=0.0,
        )
        legend.patch.set_edgecolor(color)
        legend.patch.set_facecolor(TRANSPARENT)
        legend.patch.set_linewidth(0.8)
        figure.add_artist(legend)


def draw_comparison_panels(
    axes: Sequence[Axes],
    comparison: BatchComparison,
    power_mh_time: PowerMHTimeData,
    *,
    rank_name: str = RANK_NAME,
    run_label: str = "",
    panel_labels: Sequence[str] = ("(a)", "(b)", "(c)"),
) -> tuple[int, int, int]:
    """Draw the time, transitions-per-call, and time-per-call panels into three axes.

    Returns:
        The prefetch budgets selected for the transitions-per-call, empirical-time, and time-per-call panels.
    """
    efficiency_budget, efficiency_analysis = best_analysis(
        comparison.batch_analyses,
        TRANSITIONS_PER_CALL,
        maximize=True,
        rank_name=rank_name,
    )
    time_budget, time_analysis = best_analysis(
        comparison.batch_analyses, EMPIRICAL_TIME, rank_name=rank_name
    )
    time_per_call_budget, time_per_call_analysis = best_analysis(
        comparison.batch_analyses, TIME_PER_CALL, rank_name=rank_name
    )

    efficiency_rows = [
        row
        for row in TRANSITIONS_PER_CALL.trajectory_rows(efficiency_analysis)
        if str(row["rank"]) == rank_name
    ]
    efficiency_summary = [
        row
        for row in TRANSITIONS_PER_CALL.summary_rows(efficiency_analysis)
        if str(row["rank"]) == rank_name
    ]
    efficiency_rows_by_trace = group_rows_by_trace(efficiency_rows)
    efficiency_rank_by_trace = map_trace_ranks(efficiency_rows_by_trace)
    efficiency_summary_by_rank = group_rows_by_rank(efficiency_summary)
    (
        efficiency_cost_limits,
        efficiency_iteration_limits,
        final_iteration,
    ) = trajectory_limits(
        efficiency_rows,
        efficiency_summary,
        TRANSITIONS_PER_CALL,
    )
    power_efficiency_text_y = POWER_EFFICIENCY_TEXT_Y
    if run_label:
        # Single-run datasets can have early spikes or a high relative baseline.
        # Keep both statistics clear of the curves and the one-transition line.
        lower, upper = efficiency_cost_limits
        upper += max(1.0, 0.1 * (upper - lower))
        efficiency_cost_limits = (0.0, upper)
        power_efficiency_text_y = 0.15 / upper
    if TRANSITIONS_PER_CALL_YMAX is not None:
        efficiency_cost_limits = (0.0, TRANSITIONS_PER_CALL_YMAX)
        power_efficiency_text_y = 0.15 / TRANSITIONS_PER_CALL_YMAX

    time_rows = convert_time_rows_to_minutes(
        [
            row
            for row in EMPIRICAL_TIME.trajectory_rows(time_analysis)
            if str(row["rank"]) == rank_name
        ],
        EMPIRICAL_TIME.value_key,
    )
    time_summary = convert_time_rows_to_minutes(
        [
            row
            for row in EMPIRICAL_TIME.summary_rows(time_analysis)
            if str(row["rank"]) == rank_name
        ],
        EMPIRICAL_TIME.mean_key,
        EMPIRICAL_TIME.sd_key,
    )
    time_rows_by_trace = group_rows_by_trace(time_rows)
    time_rank_by_trace = map_trace_ranks(time_rows_by_trace)
    time_summary_by_rank = group_rows_by_rank(time_summary)
    time_cost_limits, time_iteration_limits, time_final_iteration = (
        trajectory_limits(time_rows, time_summary, EMPIRICAL_TIME)
    )
    if time_final_iteration != final_iteration:
        raise ValueError(
            "Call and time trajectories must reach the same final iteration: "
            f"{final_iteration} versus {time_final_iteration}"
        )

    time_per_call_rows = [
        row
        for row in TIME_PER_CALL.trajectory_rows(time_per_call_analysis)
        if str(row["rank"]) == rank_name
    ]
    time_per_call_summary = [
        row
        for row in TIME_PER_CALL.summary_rows(time_per_call_analysis)
        if str(row["rank"]) == rank_name
    ]
    time_per_call_rows_by_trace = group_rows_by_trace(time_per_call_rows)
    time_per_call_rank_by_trace = map_trace_ranks(
        time_per_call_rows_by_trace
    )
    time_per_call_summary_by_rank = group_rows_by_rank(
        time_per_call_summary
    )
    (
        time_per_call_cost_limits,
        time_per_call_iteration_limits,
        time_per_call_final_iteration,
    ) = trajectory_limits(
        time_per_call_rows, time_per_call_summary, TIME_PER_CALL
    )
    if time_per_call_final_iteration != final_iteration:
        raise ValueError(
            "Call and time-per-call trajectories must reach the same final "
            f"iteration: {final_iteration} versus "
            f"{time_per_call_final_iteration}"
        )

    power_iterations = [
        int(row["mh_iteration"]) for row in power_mh_time.summary
    ]
    power_means = [
        seconds_to_minutes(float(row["mean_cumulative_seconds"]))
        for row in power_mh_time.summary
    ]
    power_deviations = [
        seconds_to_minutes(float(row["std_cumulative_seconds"]))
        for row in power_mh_time.summary
    ]
    if power_iterations[-1] != final_iteration:
        raise ValueError(
            "PowerMH and SubTreeMH must reach the same final MH iteration: "
            f"{power_iterations[-1]} versus {final_iteration}"
        )
    power_maximum_time = max(
        max(
            seconds_to_minutes(cumulative[-1])
            for _, cumulative in power_mh_time.trajectories
        ),
        max(
            mean + deviation
            for mean, deviation in zip(
                power_means, power_deviations, strict=True
            )
        ),
    )
    time_cost_limits = (
        time_cost_limits[0],
        max(time_cost_limits[1], power_maximum_time * 1.03),
    )

    power_time_per_call_iterations = [
        int(row["mh_iteration"])
        for row in power_mh_time.time_per_call_summary
    ]
    power_time_per_call_means = [
        float(row["mean_empirical_seconds_per_call"])
        for row in power_mh_time.time_per_call_summary
    ]
    power_time_per_call_deviations = [
        float(row["std_empirical_seconds_per_call"])
        for row in power_mh_time.time_per_call_summary
    ]
    if power_time_per_call_iterations[-1] != final_iteration:
        raise ValueError(
            "PowerMH and SubTreeMH time-per-call trajectories must reach the "
            f"same final iteration: {power_time_per_call_iterations[-1]} "
            f"versus {final_iteration}"
        )
    power_maximum_time_per_call = max(
        max(
            values
            for _, trajectory in power_mh_time.time_per_call_trajectories
            for values in trajectory
        ),
        max(
            mean + deviation
            for mean, deviation in zip(
                power_time_per_call_means,
                power_time_per_call_deviations,
                strict=True,
            )
        ),
    )
    time_per_call_cost_limits = (
        time_per_call_cost_limits[0],
        max(
            time_per_call_cost_limits[1],
            power_maximum_time_per_call * 1.03,
        ) * TIME_PER_CALL_Y_HEADROOM,
    )

    time_ax, efficiency_ax, time_per_call_ax = axes

    _, efficiency_means, efficiency_deviations = draw_subtree_trajectories(
        efficiency_ax,
        efficiency_rows_by_trace,
        efficiency_rank_by_trace,
        efficiency_summary_by_rank,
        TRANSITIONS_PER_CALL,
        rank_name=rank_name,
    )
    efficiency_reference = draw_power_mh_transitions_per_call_reference(
        efficiency_ax, final_iteration
    )
    style_panel(
        efficiency_ax,
        TRANSITIONS_PER_CALL,
        cost_limits=efficiency_cost_limits,
        iteration_limits=efficiency_iteration_limits,
    )
    statistic_text(
        efficiency_ax,
        SUBTREE_TOP_TEXT_X,
        SUBTREE_TOP_TEXT_Y,
        SUBTREE_METHOD_LABEL,
        efficiency_means[-1],
        efficiency_deviations[-1],
        color=SUBTREE_COLOR,
        ha="center",
        va="top",
    )
    statistic_text(
        efficiency_ax,
        0.50,
        power_efficiency_text_y,
        POWER_METHOD_LABEL,
        efficiency_reference[-1],
        0.0,
        color=POWER_COLOR,
        ha="center",
        va="bottom",
    )

    _, time_means, time_deviations = draw_subtree_trajectories(
        time_ax,
        time_rows_by_trace,
        time_rank_by_trace,
        time_summary_by_rank,
        EMPIRICAL_TIME,
        rank_name=rank_name,
    )
    draw_power_mh_time_trajectories(time_ax, power_mh_time)
    style_panel(
        time_ax,
        EMPIRICAL_TIME,
        cost_limits=time_cost_limits,
        iteration_limits=time_iteration_limits,
    )
    statistic_text(
        time_ax,
        0.98,
        0.035,
        SUBTREE_METHOD_LABEL,
        time_means[-1],
        time_deviations[-1],
        color=SUBTREE_COLOR,
        ha="right",
        va="bottom",
    )
    statistic_text(
        time_ax,
        0.50,
        0.95,
        POWER_METHOD_LABEL,
        power_means[-1],
        power_deviations[-1],
        color=POWER_COLOR,
        ha="center",
        va="top",
    )

    _, time_per_call_means, time_per_call_deviations = (
        draw_subtree_trajectories(
            time_per_call_ax,
            time_per_call_rows_by_trace,
            time_per_call_rank_by_trace,
            time_per_call_summary_by_rank,
            TIME_PER_CALL,
            rank_name=rank_name,
        )
    )
    draw_power_mh_time_per_call_trajectories(
        time_per_call_ax, power_mh_time
    )
    style_panel(
        time_per_call_ax,
        TIME_PER_CALL,
        cost_limits=time_per_call_cost_limits,
        iteration_limits=time_per_call_iteration_limits,
    )
    statistic_text(
        time_per_call_ax,
        SUBTREE_TOP_TEXT_X,
        SUBTREE_TOP_TEXT_Y,
        SUBTREE_METHOD_LABEL,
        time_per_call_means[-1],
        time_per_call_deviations[-1],
        color=SUBTREE_COLOR,
        ha="center",
        va="top",
    )
    statistic_text(
        time_per_call_ax,
        0.50,
        0.035,
        POWER_METHOD_LABEL,
        power_time_per_call_means[-1],
        power_time_per_call_deviations[-1],
        color=POWER_COLOR,
        ha="center",
        va="bottom",
    )

    for ax, panel_label in zip(
        (time_ax, efficiency_ax, time_per_call_ax),
        panel_labels,
        strict=True,
    ):
        ax.text(
            -0.12,
            PANEL_LABEL_Y,
            rf"\textbf{{{panel_label}}}",
            transform=ax.transAxes,
            color=BLACK,
            fontsize=FIGURE_FONT_SIZE,
            ha="right" if run_label else "left",
            va="top",
            clip_on=False,
            zorder=7,
        )
        ax.minorticks_off()
        ax.tick_params(axis="both", labelsize=FIGURE_FONT_SIZE)
    return efficiency_budget, time_budget, time_per_call_budget


def _line_samples(ax: Axes, spacing_px: float = 1.0) -> np.ndarray:
    """Return display-space points sampled every ``spacing_px`` along every drawn line (steps expanded)."""
    samples = []
    for line in ax.lines:
        x, y = (np.asarray(values, dtype=float) for values in line.get_data())
        if len(x) < 2:
            continue
        if line.get_drawstyle() == "steps-post":
            x, y = pts_to_poststep(x, y)
        points = line.get_transform().transform(np.column_stack([x, y]))
        segments = np.diff(points, axis=0)
        counts = np.maximum(1, np.ceil(np.hypot(*segments.T) / spacing_px).astype(int))
        index = np.repeat(np.arange(len(segments)), counts)
        fraction = (np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)) / counts[index]
        samples.append(points[index] + segments[index] * fraction[:, None])
    return np.concatenate(samples) if samples else np.empty((0, 2))


def place_statistics(ax: Axes, renderer) -> None:
    """Back each in-panel statistic with a white box and move it to the candidate spot crossed by fewest lines."""
    texts = [text for text in ax.texts if r"_{\pm" in text.get_text()]
    if not texts:
        return
    samples = _line_samples(ax)
    axes_box = ax.get_window_extent(renderer)
    pad = 2.0
    placed = []
    for text in texts:
        original = (*text.get_position(), text.get_horizontalalignment(), text.get_verticalalignment())
        candidates = [original] + [
            (x, y, ha, "center") for y in STATISTIC_CANDIDATE_Y for x, ha in STATISTIC_CANDIDATE_X
        ]
        best = None
        for x, y, ha, va in candidates:
            text.set_position((x, y))
            text.set_horizontalalignment(ha)
            text.set_verticalalignment(va)
            box = text.get_window_extent(renderer)
            x0, y0, x1, y1 = box.x0 - pad, box.y0 - pad, box.x1 + pad, box.y1 + pad
            if x0 < axes_box.x0 or x1 > axes_box.x1 or y0 < axes_box.y0 or y1 > axes_box.y1:
                continue
            if any(x0 < p.x1 and p.x0 < x1 and y0 < p.y1 and p.y0 < y1 for p in placed):
                continue
            inside = (
                (samples[:, 0] >= x0) & (samples[:, 0] <= x1) & (samples[:, 1] >= y0) & (samples[:, 1] <= y1)
            )
            # Crossings dominate; distance from the default spot (in axes units) only breaks near-ties.
            score = int(inside.sum()) + 3.0 * float(np.hypot(x - original[0], y - original[1]))
            if best is None or score < best[0]:
                best = (score, (x, y, ha, va), box)
        if best is None:
            best = (0, original, None)
        x, y, ha, va = best[1]
        text.set_position((x, y))
        text.set_horizontalalignment(ha)
        text.set_verticalalignment(va)
        text.set_bbox(dict(STATISTIC_BOX))
        placed.append(text.get_window_extent(renderer))


def clear_panel_label(ax: Axes, renderer) -> None:
    """Right-align the panel letter just left of the y tick labels it would otherwise overlap."""
    labels = [text for text in ax.texts if text.get_text().startswith(r"\textbf{(")]
    if not labels:
        return
    label = labels[0]
    axes_box = ax.get_window_extent(renderer)
    if label.get_horizontalalignment() == "left":
        # A left-aligned letter starts at its anchor and runs over the tick labels; keep its anchor as right edge.
        label.set_horizontalalignment("right")
    label_box = label.get_window_extent(renderer)
    tick_boxes = [
        tick.get_window_extent(renderer) for tick in ax.get_yticklabels()
        if tick.get_visible() and tick.get_text()
    ]
    clearance = PANEL_LABEL_CLEARANCE_POINTS * ax.figure.dpi / 72.0
    right = min(label_box.x1, axes_box.x0 - clearance)
    for box in tick_boxes:
        if box.y1 > label_box.y0 - clearance and box.y0 < label_box.y1 + clearance:
            right = min(right, box.x0 - clearance)
    label.set_position(((right - axes_box.x0) / axes_box.width, label.get_position()[1]))


def plot_figure(
    comparison: BatchComparison,
    power_mh_time: PowerMHTimeData,
    output_path: Path,
    *,
    rank_name: str = RANK_NAME,
    run_label: str = "",
) -> tuple[Path, int, int, int]:
    """Render the three-panel figure and return each selected budget."""
    figure_width = PANEL_WIDTH * 3
    figure, (time_ax, efficiency_ax, time_per_call_ax) = plt.subplots(
        1,
        3,
        figsize=(figure_width, FIGURE_HEIGHT),
        sharex=True,
    )
    efficiency_budget, time_budget, time_per_call_budget = draw_comparison_panels(
        (time_ax, efficiency_ax, time_per_call_ax),
        comparison,
        power_mh_time,
        rank_name=rank_name,
        run_label=run_label,
    )

    figure.subplots_adjust(
        left=LEFT_MARGIN_INCHES / figure_width,
        right=0.995,
        bottom=BOTTOM_MARGIN_INCHES / FIGURE_HEIGHT,
        top=PLOT_TOP,
        wspace=PANEL_WSPACE,
    )
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    for ax in (time_ax, efficiency_ax, time_per_call_ax):
        place_statistics(ax, renderer)
        clear_panel_label(ax, renderer)
    first_position = time_ax.get_position()
    time_per_call_position = time_per_call_ax.get_position()
    dataset_name, separator, model_name = comparison.figure_title.partition(
        ", "
    )
    if not separator:
        raise ValueError(
            "Expected '<dataset>, <model>' figure title, got "
            f"{comparison.figure_title!r}"
        )
    first_log = comparison.directory / str(
        comparison.batch_analyses[0][1].events[0]["source_log"]
    )
    model_name = model_display_name(first_log, model_name)
    if dataset_of(first_log) == "lcb_v6":
        dataset_name = "LCB V6"
    figure.text(
        (first_position.x0 + time_per_call_position.x1) / 2.0,
        TITLE_Y,
        (
            f"dataset: {dataset_name}, "
            f"base LLM: {model_name}, "
            f"inference engine: {backend_of(first_log)}"
            + (f", {run_label}" if run_label else "")
        ),
        color=DARK_GRAY,
        fontsize=FIGURE_FONT_SIZE,
        fontweight="semibold",
        ha="center",
        va="top",
    )
    add_method_legends(
        figure, left=first_position.x0, right=time_per_call_position.x1,
    )
    return (
        save_figure(figure, output_path, pad_inches=0.01),
        efficiency_budget,
        time_budget,
        time_per_call_budget,
    )


def single_budget_comparison(
    log_path: Path,
    *,
    traces_per_rank: int,
    include_incomplete: bool = False,
) -> BatchComparison:
    """Build a comparison from one explicitly selected subtree log."""
    extracted = extract_logs(
        [log_path],
        include_incomplete=include_incomplete,
        known_ranks=tuple(RANK_ORDER),
    )
    analysis = build_analysis_rows(extracted, traces_per_rank)
    return BatchComparison(
        directory=log_path.parent,
        signature=log_path.name,
        output_prefix=log_path.parent / dataset_model_prefix(log_path),
        figure_title=experiment_figure_title(log_path),
        batch_analyses=((prefetch_budget_of(log_path), analysis),),
    )


def accept_first_analysis(
    group: ExperimentGroup,
    *,
    traces_per_rank: int,
    include_incomplete: bool,
) -> AnalysisRows | None:
    """Parse only the accept-first log from one prefetch-budget group."""
    accept_logs = [
        log_path
        for log_path in group.log_paths
        if canonical_rank(rank_of(log_path)) == RANK_NAME
    ]
    if not accept_logs:
        return None
    if len(accept_logs) != 1:
        raise ValueError(
            f"Expected one {RANK_NAME!r} log for {group.signature}, "
            f"got {len(accept_logs)}"
        )
    extracted = extract_logs(
        accept_logs,
        include_incomplete=include_incomplete,
        known_ranks=tuple(TRANSITION_RANK_COLORS),
    )
    return build_analysis_rows(extracted, traces_per_rank)


def find_comparison(
    directory: Path,
    *,
    traces_per_rank: int,
    include_incomplete: bool,
) -> BatchComparison:
    """Build the single cross-budget accept-first comparison under a root."""
    completed: list[tuple[ExperimentGroup, AnalysisRows]] = []
    for group in discover_experiment_groups(
        directory, MODEL_CALL_OUTPUT_SUFFIX
    ):
        analysis = accept_first_analysis(
            group,
            traces_per_rank=traces_per_rank,
            include_incomplete=include_incomplete,
        )
        if analysis is not None:
            completed.append((group, analysis))

    comparisons = build_batch_comparisons(completed)
    if len(comparisons) != 1:
        raise ValueError(
            "Expected one cross-prefetch-budget accept-first comparison under "
            f"{directory}, found {len(comparisons)}"
        )
    return comparisons[0]


def comparison_source_log(comparison: BatchComparison) -> Path:
    """Return one analyzed SubTreeMH log for identity matching."""
    return comparison.directory / str(
        comparison.batch_analyses[0][1].events[0]["source_log"]
    )


def _power_mh_config_matches(subtree_log: Path, power_log: Path) -> bool:
    """Match the full shared run configuration of a SubTreeMH log and its PowerMH baseline."""
    return shared_run_config_of(subtree_log) == shared_run_config_of(power_log)


def find_power_mh_log(
    directory: Path,
    subtree_log: Path,
    explicit_log: Path | None,
) -> Path:
    """Resolve exactly one PowerMH log with the same full run configuration."""
    if explicit_log is not None:
        candidate = explicit_log.resolve()
        if not candidate.is_file():
            raise ValueError(f"PowerMH log does not exist: {candidate}")
        candidates = [candidate]
    else:
        candidates = [
            candidate
            for candidate in sorted(directory.rglob(POWER_MH_LOG_GLOB))
            if _power_mh_config_matches(subtree_log, candidate)
        ]
    if len(candidates) != 1:
        raise ValueError(
            "Expected one PowerMH log matching the SubTreeMH run, found "
            f"{len(candidates)}; pass --power-mh-log explicitly"
        )
    power_log = candidates[0]
    if not _power_mh_config_matches(subtree_log, power_log):
        raise ValueError(
            "PowerMH and SubTreeMH run configurations differ: "
            f"{power_log.name} versus {subtree_log.name}"
        )
    if backend_of(power_log) != backend_of(subtree_log):
        raise ValueError("PowerMH and SubTreeMH must use the same serving backend")
    return power_log


def default_output_path(comparison: BatchComparison) -> Path:
    """Build the dataset/model-qualified output filename."""
    source_log = comparison_source_log(comparison)
    if len(comparison.batch_analyses) == 1:
        budget = comparison.batch_analyses[0][0]
        rank = canonical_rank(rank_of(source_log)).replace("_", "-")
        return comparison.directory / (
            f"{dataset_model_prefix(source_log)}.prefetch-budget-{budget}."
            f"{rank}-power-mh.model-calls-and-empirical-time.pdf"
        )
    return comparison.directory / f"{dataset_model_prefix(source_log)}{FIGURE_SUFFIX}"


def set_time_unit(unit: str) -> None:
    """Draw cumulative empirical time in minutes ("min", the default) or seconds ("sec")."""
    global SECONDS_PER_MINUTE, TIME_AXIS_LABEL
    SECONDS_PER_MINUTE = {"min": 60.0, "sec": 1.0}[unit]
    TIME_AXIS_LABEL = f"cumulative model-call time ({unit})"
    # Relabel in place: other scripts hold EMPIRICAL_TIME in tuples and compare it by identity.
    object.__setattr__(EMPIRICAL_TIME, "axis_label", TIME_AXIS_LABEL)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the dedicated figure command-line interface."""
    parser = argparse.ArgumentParser(
        description=(
            "Draw the three-panel accept-first SubTreeMH-versus-PowerMH "
            "proposal-call, empirical-time, and time-per-call comparison."
        )
    )
    parser.add_argument(
        "--directory",
        type=Path,
        default=DEFAULT_DIRECTORY,
        help="Directory containing the raw SubTreeMH and PowerMH logs.",
    )
    parser.add_argument(
        "--subtree-log",
        type=Path,
        default=None,
        help="Plot one subtree log, retaining its rank policy and single budget.",
    )
    parser.add_argument(
        "--power-mh-log",
        type=Path,
        default=None,
        help="Explicit matching PowerMH log; otherwise discovered by config.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PDF path; defaults to the dataset/model-qualified name.",
    )
    parser.add_argument(
        "--traces-per-rank",
        type=int,
        default=DEFAULT_TRACES_PER_RANK,
        help="Maximum deterministic accept-first traces per prefetch budget.",
    )
    parser.add_argument(
        "--time-unit",
        choices=("min", "sec"),
        default="min",
        help="Unit of cumulative empirical time (panel a and its statistics).",
    )
    parser.add_argument(
        "--include-incomplete",
        action="store_true",
        help="Include observed prefixes from accept-first traces open at EOF.",
    )
    parser.add_argument(
        "--transitions-per-call-ymax",
        type=float,
        default=None,
        help="Fixed upper y-limit of the transitions-per-call panel (b), to share one scale across stacked figures.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Parse the raw logs and render only the dedicated combined figure."""
    args = build_arg_parser().parse_args(argv)
    if args.traces_per_rank <= 0:
        raise SystemExit("--traces-per-rank must be positive")
    set_time_unit(args.time_unit)
    global TRANSITIONS_PER_CALL_YMAX
    TRANSITIONS_PER_CALL_YMAX = args.transitions_per_call_ymax

    directory = args.directory.resolve()
    try:
        if args.subtree_log is None:
            comparison = find_comparison(
                directory,
                traces_per_rank=args.traces_per_rank,
                include_incomplete=args.include_incomplete,
            )
        else:
            comparison = single_budget_comparison(
                args.subtree_log.resolve(),
                traces_per_rank=args.traces_per_rank,
                include_incomplete=args.include_incomplete,
            )
        subtree_log = comparison_source_log(comparison)
        rank_name = canonical_rank(rank_of(subtree_log))
        run_label = ""
        if args.subtree_log is not None:
            rank_label = rank_name.replace("bfs_", "BFS ").replace("_", "-")
            run_label = f"{rank_label}, $B={prefetch_budget_of(subtree_log)}$"
            trace_count = comparison.batch_analyses[0][1].traces_per_rank
            if trace_count < DEFAULT_TRACES_PER_RANK:
                plural = "s" if trace_count != 1 else ""
                run_label += f", {trace_count} subtree trace{plural}"
        power_log = find_power_mh_log(
            directory, subtree_log, args.power_mh_log
        )
        power_mh_time = load_power_mh_time_data(power_log)
        output_path = (
            args.output.resolve()
            if args.output is not None
            else default_output_path(comparison)
        )
        if output_path.suffix.lower() != ".pdf":
            raise ValueError(f"Output path must end in .pdf: {output_path}")
        (
            written_path,
            efficiency_budget,
            time_budget,
            time_per_call_budget,
        ) = plot_figure(
            comparison, power_mh_time, output_path,
            rank_name=rank_name, run_label=run_label,
        )
    except (NoModelCallTracesError, OSError, ValueError) as error:
        raise SystemExit(str(error)) from error

    budgets = ", ".join(
        str(prefetch_budget)
        for prefetch_budget, _ in comparison.batch_analyses
    )
    print(f"prefetch budgets: {budgets}")
    print(f"best transitions-per-call budget: {efficiency_budget}")
    print(f"best empirical-time budget: {time_budget}")
    print(f"best time-per-call budget: {time_per_call_budget}")
    print(f"PowerMH log: {power_log}")
    print(f"wrote {written_path}")


if __name__ == "__main__":
    main()
