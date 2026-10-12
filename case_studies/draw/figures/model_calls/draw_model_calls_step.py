#!/usr/bin/env python3
"""Plot MH-step yield per target-model call from subtree-prefetch logs.

The input directory is searched recursively. Logs are grouped by their full run configuration after removing only the
``rank-...`` filename token, so different dates, backends, and prefetch budgets are never pooled. The model-call
comparison gives PowerMH's one-call-per-step reference its own panel. Empirical-time figures use the matching measured
PowerMH run: the single-budget figures draw its mean behind each SubTreeMH panel, while the cross-budget figure gives
its full timing distribution a panel of its own. Compatible batch-size groups are also combined into comparisons. The
aggregate empirical-time figure retains the fastest terminal-mean result for each SubTreeMH traversal rank while its
source-data tables preserve every budget.

Run with:

    src/.venv/bin/python \
        case_studies/draw/draw_model_calls_step.py \
        --directory case_studies/logs/math500
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import LogLocator, MaxNLocator

from case_studies import paths
from case_studies.draw.common.figure_io import save_figure, write_dict_rows
from case_studies.draw.common.plot_helpers import panel_letter
from case_studies.draw.common.rank_plotting import (
    MEAN_LINE_WIDTH,
    RAW_TRACE_ALPHA,
    RAW_TRACE_LINE_WIDTH,
    SD_BAND_ALPHA,
    TERMINAL_MEAN_FONT_SIZE,
    TERMINAL_MEAN_LINE_SPACING,
    TERMINAL_MEAN_POSITION,
    TIME_AXIS_LABEL,
    TRANSITION_AXIS_LABEL,
    add_panel_label,
    annotate_terminal_statistics,
    draw_time_trajectories,
    prefetch_size_label,
    style_quantitative_axis,
)
from case_studies.draw.figures.model_calls import (
    draw_accept_first_power_mh_model_calls_and_empirical_time as combined_style,
)
from case_studies.extract.logs.model_call_traces import (
    MODEL_CALL_OUTPUT_SUFFIX,
    POWER_MH_LOG_GLOB,
    AnalysisRows,
    ExperimentGroup,
    ExtractedLogs,
    NoModelCallTracesError,
    Row,
    TraceKey,
    build_analysis_rows,
    build_batch_comparisons,
    build_power_mh_time_per_call_summary_rows,
    build_power_mh_time_rows,
    build_power_mh_time_summary_rows,
    discover_experiment_groups,
    extract_logs,
    group_rows_by_rank,
    group_rows_by_trace,
    map_trace_ranks,
    power_mh_time_per_call_trajectories,
    power_mh_trace_times,
)
from case_studies.extract.run_naming import (
    backend_of,
    dataset_of,
    model_of,
    output_path,
    rank_label,
    shared_run_config_of,
    sort_rank_names,
)
from case_studies.plot_config import (
    BLUE as POWER_COLOR,
)
from case_studies.plot_config import (
    DARK_GRAY,
    GRAY,
    POWER_METHOD_LABEL,
    SUBTREE_METHOD_LABEL,
    TRANSPARENT,
    apply_plot_style,
)
from case_studies.plot_config import (
    GREEN as SUBTREE_COMPARISON_COLOR,
)
from case_studies.plot_config import (
    TRANSITION_RANK_COLORS as RANK_COLORS,
)

apply_plot_style()

DEFAULT_DIRECTORY = paths.LOGS_DIR / "math500"
DEFAULT_TRACES_PER_RANK = 10

PANEL_HEIGHT = 2.15
# The six traversal ranks plus the PowerMH reference that follows them.
PANEL_COLUMNS = 7
PANEL_WIDTH = 3.8
BATCH_COMPARISON_PANEL_WIDTH = 2.65
BATCH_COMPARISON_ROW_HEIGHT = 2.10
BATCH_COMPARISON_BEST_ONLY_HEIGHT = 3.30
BATCH_COMPARISON_BEST_ONLY_SIDEBAR_WIDTH = 2.10
BATCH_COMPARISON_BEST_ONLY_SIDEBAR_GAP = 0.18
BATCH_COMPARISON_WSPACE = 0.08
BATCH_COMPARISON_HSPACE = 0.20
BATCH_COMPARISON_TITLE_FONT_SIZE = 13.0
BATCH_COMPARISON_HEADER_FONT_SIZE = 13.0
# One axis label serves the whole figure, so it can carry a larger size.
AXIS_LABEL_FONT_SIZE = 17.0
COMBINED_FIGURE_FONT_SIZE = combined_style.FIGURE_FONT_SIZE
COMBINED_PANEL_WIDTH = combined_style.PANEL_WIDTH
COMBINED_FIGURE_HEIGHT = combined_style.FIGURE_HEIGHT
COMBINED_PLOT_TOP = combined_style.PLOT_TOP
COMBINED_TITLE_Y = combined_style.TITLE_Y
COMBINED_LEGEND_Y = combined_style.LEGEND_Y
COMBINED_TIME_AXIS_LABEL = "cumulative model-call time (min)"
COMBINED_TIME_PER_CALL_AXIS_LABEL = "model-call time (sec/call)"
COMBINED_TRANSITIONS_PER_CALL_AXIS_LABEL = "MH transitions per model call"
COMBINED_EFFICIENCY_Y_LIMITS = (0.5, 6.5)
COMBINED_POWER_EFFICIENCY_TEXT_Y = combined_style.POWER_EFFICIENCY_TEXT_Y
SECONDS_PER_MINUTE = 60.0
# Held in inches so the axis labels keep their room as panels are added: the left margin also carries the per-row budget
# labels, the bottom the tick row.
BATCH_COMPARISON_LEFT_MARGIN_INCHES = 1.05
BATCH_COMPARISON_BOTTOM_MARGIN_INCHES = 0.62
BATCH_COMPARISON_BEST_ONLY_LEFT_MARGIN_INCHES = 0.82
BATCH_COMPARISON_BEST_ONLY_BOTTOM_MARGIN_INCHES = 0.50
BATCH_COMPARISON_BEST_ONLY_YLABEL_X_INCHES = 0.29
# The small-multiple grid plots the cost quantity up the y axis and the MH transitions it bought along x, matching the
# combined SubTreeMH-versus-PowerMH figures, which have read that way since they were added. Both per-budget grids
# follow it so the pair stays comparable panel for panel.
GRID_SWAP_AXES = True
# Swapping moves the longer name ("cumulative number of model calls") onto the y axis, where it is set rotated against
# the figure edge. The gutter widens to
# match: at 0.72 the rotated name ran under the panel letter of the first
# column, which is placed relative to its own axes.
GRID_LEFT_MARGIN_INCHES = 1.00 if GRID_SWAP_AXES else 0.72
LEGEND_FONT_SIZE = 12.0
PANEL_LABEL_POSITION = (-0.16, 0.98)
PANEL_LABEL_X_WITHOUT_Y_LABEL = -0.08
PANEL_LEGEND_LEFT_ANCHOR = 0.52
# Both in-panel legends clear the bottom strip the mean annotation now uses.
PANEL_LEGEND_BOTTOM_ANCHOR = 0.13
# Swapping the axes reflects each trajectory about the diagonal, so the empty corner moves with it: cost against
# transitions climbs shallowly from the origin and leaves the upper left free, where the unswapped figure left the lower
# right. The terminal mean keeps the lower right, still below the curve.
SWAPPED_PANEL_LEGEND_ANCHOR = (0.02, 0.98)
POWER_LEGEND_RIGHT_ANCHOR = 0.98
POWER_LEGEND_BOTTOM_ANCHOR = 0.13
# The trajectory encoding itself -- trace width, mean width, SD band, and where the terminal mean sits -- comes from
# rank_plotting, which the standalone and side-by-side timing figures draw with too. Only this file's departure from it
# is stated here: a log cost axis pulls the trajectories down to the right, freeing the opposite corner for the
# annotation.
LOG_AXIS_MEAN_ANNOTATION_POSITION = (0.025, 0.955)
LOG_AXIS_MEAN_ANNOTATION_ALIGNMENT = ("left", "top")
MEAN_ANNOTATION_ALIGNMENT = ("right", "bottom")
REFERENCE_LINE_WIDTH = 1.35
# The measured baseline is drawn behind every panel, so it cannot reuse
# POWER_COLOR: that is BLUE, which is also accept_first's rank colour. Grey
# clashes with no rank and reads as an annotation rather than a seventh series.
POWER_TIME_REFERENCE_COLOR = GRAY
POWER_TIME_REFERENCE_LABEL = f"{POWER_METHOD_LABEL} baseline"
POWER_METHOD_DESCRIPTION = "one call per step"

LEGEND_OPTIONS = {
    "alignment": "left",
    "frameon": False,
    "fontsize": LEGEND_FONT_SIZE,
    "borderaxespad": 0.45,
    "handlelength": 1.5,
    "handletextpad": 0.35,
    "labelspacing": 0.22,
}


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
    """One measured cost axis for the MH-yield trajectories.

    Both figures plot realized MH transitions against a cumulative cost; they differ only in which cost, how it is
    scaled, and what it is called.
    """

    trajectory_attribute: str
    summary_attribute: str
    value_key: str
    mean_key: str
    sd_key: str
    x_label: str
    include_power_mh: bool
    integer_x_ticks: bool
    log_x_axis: bool
    figure_suffix: str
    data_suffix: str
    summary_suffix: str

    def trajectory_rows(self, analysis: AnalysisRows) -> list[Row]:
        return getattr(analysis, self.trajectory_attribute)

    def summary_rows(self, analysis: AnalysisRows) -> list[Row]:
        return getattr(analysis, self.summary_attribute)


@dataclass(frozen=True)
class PowerMHTimeData:
    """One matched PowerMH log's raw and summarized timing trajectories."""

    log_path: Path
    trajectories: list[tuple[str, list[float]]]
    summary: list[Row]
    time_per_call_trajectories: list[tuple[str, list[float]]]
    time_per_call_summary: list[Row]

    @property
    def mean_reference(self) -> list[tuple[int, float]]:
        """Return the measured mean trajectory, including its zero origin."""
        return [(0, 0.0)] + [
            (
                int(row["mh_iteration"]),
                float(row["mean_cumulative_seconds"]),
            )
            for row in self.summary
        ]


def load_power_mh_time_data(log_path: Path) -> PowerMHTimeData:
    """Parse one complete PowerMH timing log once for every figure consumer."""
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


MODEL_CALLS = TrajectoryKind(
    trajectory_attribute="cumulative",
    summary_attribute="summary",
    value_key="subtree_cumulative_calls",
    mean_key="subtree_mean_calls",
    sd_key="subtree_std_calls",
    x_label="cumulative number of model calls",
    # PowerMH spends one call per step by construction, so its model-call panel is the line y = x with no spread -- it
    # restates the definition rather than measuring anything, and it cost a panel and a wider shared x axis. The
    # empirical-time figures still carry a measured PowerMH reference.
    include_power_mh=False,
    integer_x_ticks=True,
    log_x_axis=False,
    figure_suffix=".pdf",
    data_suffix=".cumulative.csv",
    summary_suffix=".summary.csv",
)
EMPIRICAL_TIME = TrajectoryKind(
    trajectory_attribute="empirical_time",
    summary_attribute="empirical_time_summary",
    value_key="cumulative_empirical_seconds",
    mean_key="mean_cumulative_seconds",
    sd_key="std_cumulative_seconds",
    x_label=TIME_AXIS_LABEL,
    include_power_mh=False,
    integer_x_ticks=False,
    log_x_axis=False,
    figure_suffix=".empirical-time.pdf",
    data_suffix=".empirical-time.csv",
    summary_suffix=".empirical-time.summary.csv",
)
TIME_PER_CALL = TrajectoryKind(
    trajectory_attribute="time_per_call",
    summary_attribute="time_per_call_summary",
    value_key="empirical_seconds_per_call",
    mean_key="mean_empirical_seconds_per_call",
    sd_key="std_empirical_seconds_per_call",
    x_label=COMBINED_TIME_PER_CALL_AXIS_LABEL,
    include_power_mh=False,
    integer_x_ticks=False,
    log_x_axis=False,
    figure_suffix=".time-per-call.pdf",
    data_suffix=".time-per-call.csv",
    summary_suffix=".time-per-call.summary.csv",
)
TRANSITIONS_PER_CALL = TrajectoryKind(
    trajectory_attribute="transitions_per_call",
    summary_attribute="transitions_per_call_summary",
    value_key="mh_transitions_per_call",
    mean_key="mean_mh_transitions_per_call",
    sd_key="std_mh_transitions_per_call",
    x_label=COMBINED_TRANSITIONS_PER_CALL_AXIS_LABEL,
    include_power_mh=False,
    integer_x_ticks=False,
    log_x_axis=False,
    figure_suffix=".transitions-per-call.pdf",
    data_suffix=".transitions-per-call.csv",
    summary_suffix=".transitions-per-call.summary.csv",
)


# --------------------------------------------------------------------------
# Legends and per-panel styling
# --------------------------------------------------------------------------


def power_mh_legend_handle(label: str) -> Line2D:
    """Return the PowerMH reference-line legend handle."""
    return Line2D(
        [0], [0], color=POWER_COLOR, linewidth=REFERENCE_LINE_WIDTH, label=label
    )


def shared_legend_handles() -> list[Line2D | Patch]:
    """Build the trajectory-encoding legend shared across method panels."""
    return [
        Line2D(
            [0],
            [0],
            color=GRAY,
            linewidth=RAW_TRACE_LINE_WIDTH,
            alpha=RAW_TRACE_ALPHA,
            label="individual trace",
        ),
        Line2D([0], [0], color=DARK_GRAY, linewidth=MEAN_LINE_WIDTH, label="mean"),
        Patch(
            facecolor=GRAY,
            edgecolor=TRANSPARENT,
            alpha=SD_BAND_ALPHA,
            label=r"$\pm 1$ sample SD",
        ),
    ]


def add_power_mh_panel_legend(ax: Axes) -> None:
    """Name the reference line inside the panel that holds it alone."""
    ax.legend(
        handles=[power_mh_legend_handle(POWER_METHOD_DESCRIPTION)],
        title=POWER_METHOD_LABEL,
        title_fontsize=LEGEND_FONT_SIZE,
        loc="lower right",
        bbox_to_anchor=(POWER_LEGEND_RIGHT_ANCHOR, POWER_LEGEND_BOTTOM_ANCHOR),
        **LEGEND_OPTIONS,
    )


def add_panel_legend(
    ax: Axes, rank_name: str, *, with_power_mh_reference: bool = False
) -> None:
    """Add the SubTreeMH legend group for one traversal rank."""
    power_mh_handles = (
        [
            Line2D(
                [0],
                [0],
                color=POWER_TIME_REFERENCE_COLOR,
                linewidth=REFERENCE_LINE_WIDTH,
                linestyle="--",
                label=POWER_TIME_REFERENCE_LABEL,
            )
        ]
        if with_power_mh_reference
        else []
    )
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=RANK_COLORS[rank_name],
                linewidth=MEAN_LINE_WIDTH,
                label="mean",
            ),
            Line2D(
                [0],
                [0],
                color=RANK_COLORS[rank_name],
                linewidth=RAW_TRACE_LINE_WIDTH,
                alpha=RAW_TRACE_ALPHA,
                label="individual trace",
            ),
            Patch(
                facecolor=RANK_COLORS[rank_name],
                edgecolor=TRANSPARENT,
                alpha=SD_BAND_ALPHA,
                label=r"$\pm 1$ sample SD",
            ),
            *power_mh_handles,
        ],
        loc="upper left" if GRID_SWAP_AXES else "lower left",
        bbox_to_anchor=(
            SWAPPED_PANEL_LEGEND_ANCHOR
            if GRID_SWAP_AXES
            else (PANEL_LEGEND_LEFT_ANCHOR, PANEL_LEGEND_BOTTOM_ANCHOR)
        ),
        **LEGEND_OPTIONS,
    )


def add_terminal_statistics_annotation(
    ax: Axes,
    iterations: Sequence[int],
    means: Sequence[float],
    deviations: Sequence[float],
    color: str,
    *,
    log_x_axis: bool = False,
) -> None:
    """Show the final-iteration mean and sample SD inside one axes box."""
    if not iterations or not (
        len(iterations) == len(means) == len(deviations)
    ):
        raise ValueError(
            "statistic annotations require aligned, non-empty trajectories"
        )
    position, alignment = (
        (LOG_AXIS_MEAN_ANNOTATION_POSITION, LOG_AXIS_MEAN_ANNOTATION_ALIGNMENT)
        if log_x_axis
        else (TERMINAL_MEAN_POSITION, MEAN_ANNOTATION_ALIGNMENT)
    )
    horizontal_alignment, vertical_alignment = alignment
    ax.text(
        *position,
        f"mean = {means[-1]:.1f}\nstd = {deviations[-1]:.1f}",
        transform=ax.transAxes,
        color=color,
        fontsize=TERMINAL_MEAN_FONT_SIZE,
        fontweight="semibold",
        linespacing=TERMINAL_MEAN_LINE_SPACING,
        ha=horizontal_alignment,
        va=vertical_alignment,
        zorder=6,
    )


def style_trajectory_axes(
    ax: Axes,
    kind: TrajectoryKind,
    *,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    show_x_label: bool,
    show_y_label: bool,
    log_x_axis: bool = False,
    swap_axes: bool = False,
) -> None:
    """Apply the shared trajectory-axis scale, ticks, and labels.

    Callers can request logarithmic scaling, but the model-call and empirical- time figures both use linear axes so
    equal distances mean equal costs.
    """
    if swap_axes:
        ax.set_xlim(*y_limits)
        ax.set_ylim(*x_limits)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        ax.yaxis.set_major_locator(
            MaxNLocator(nbins=5, integer=kind.integer_x_ticks)
        )
        if show_x_label:
            ax.set_xlabel(TRANSITION_AXIS_LABEL)
        if show_y_label:
            ax.set_ylabel(kind.x_label)
        style_quantitative_axis(ax)
        return

    if log_x_axis:
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=5))
    else:
        ax.xaxis.set_major_locator(
            MaxNLocator(nbins=5, integer=kind.integer_x_ticks)
        )
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    if show_x_label:
        ax.set_xlabel(kind.x_label)
    if show_y_label:
        ax.set_ylabel(TRANSITION_AXIS_LABEL)
    style_quantitative_axis(ax)


# --------------------------------------------------------------------------
# Trajectory drawing
# --------------------------------------------------------------------------


def trajectory_limits(
    trajectory_rows: Sequence[Row],
    summary_rows: Sequence[Row],
    kind: TrajectoryKind,
) -> tuple[tuple[float, float], tuple[float, float], int]:
    """Return shared x/y limits and the final MH iteration for trajectories."""
    if not trajectory_rows or not summary_rows:
        raise ValueError("Trajectory limits require non-empty data and summaries")
    final_iteration = max(int(row["mh_iteration"]) for row in trajectory_rows)
    maximum_x = max(
        max(float(row[kind.value_key]) for row in trajectory_rows),
        max(
            float(row[kind.mean_key]) + float(row[kind.sd_key])
            for row in summary_rows
        ),
    )
    if kind.include_power_mh:
        maximum_x = max(maximum_x, float(final_iteration))
    return (
        (0.0, max(maximum_x * 1.03, 1.0)),
        (0.0, float(final_iteration)),
        final_iteration,
    )


def positive_log_axis_floor(rows: Sequence[Row], value_key: str) -> float:
    """Return the power-of-ten floor of the smallest positive axis value."""
    positive_values = [
        value for row in rows if (value := float(row[value_key])) > 0.0
    ]
    if not positive_values:
        raise ValueError(f"No positive values are available for {value_key}")
    return 10.0 ** math.floor(math.log10(min(positive_values)))


def draw_rank_trajectory(
    ax: Axes,
    rows_by_trace: dict[TraceKey, list[Row]],
    rank_by_trace: dict[TraceKey, str],
    summary_by_rank: dict[str, list[Row]],
    rank_name: str,
    kind: TrajectoryKind,
    *,
    swap_axes: bool = False,
    color: str | None = None,
) -> tuple[list[int], list[float], list[float]]:
    """Draw one method's raw traces, mean, and sample SD band."""
    trajectory_color = color or RANK_COLORS[rank_name]
    for trace_key in sorted(
        key for key, rank in rank_by_trace.items() if rank == rank_name
    ):
        trace_rows = rows_by_trace[trace_key]
        trace_values = [float(row[kind.value_key]) for row in trace_rows]
        trace_iterations = [int(row["mh_iteration"]) for row in trace_rows]
        ax.step(
            trace_iterations if swap_axes else trace_values,
            trace_values if swap_axes else trace_iterations,
            where="post" if swap_axes else "pre",
            color=trajectory_color,
            linewidth=RAW_TRACE_LINE_WIDTH,
            alpha=RAW_TRACE_ALPHA,
            zorder=2,
        )

    rank_summary = summary_by_rank[rank_name]
    iterations = [int(row["mh_iteration"]) for row in rank_summary]
    means = [float(row[kind.mean_key]) for row in rank_summary]
    deviations = [float(row[kind.sd_key]) for row in rank_summary]
    highlight_color = trajectory_color
    lower = [
        max(mean - sd, 0.0)
        for mean, sd in zip(means, deviations, strict=True)
    ]
    upper = [mean + sd for mean, sd in zip(means, deviations, strict=True)]
    fill_between = ax.fill_between if swap_axes else ax.fill_betweenx
    fill_between(
        iterations,
        lower,
        upper,
        step="post" if swap_axes else "pre",
        color=highlight_color,
        alpha=SD_BAND_ALPHA,
        linewidth=0,
        zorder=1,
    )
    ax.step(
        iterations if swap_axes else means,
        means if swap_axes else iterations,
        where="post" if swap_axes else "pre",
        color=highlight_color,
        linewidth=MEAN_LINE_WIDTH,
        zorder=4,
    )
    return iterations, means, deviations


def draw_power_mh_time_reference(
    ax: Axes,
    reference: Sequence[tuple[int, float]],
    *,
    swap_axes: bool = False,
) -> None:
    """Draw the measured PowerMH cumulative-time trajectory behind a panel.

    Unlike the model-call reference, this one cannot be derived from the axes:
    PowerMH's one call per step says nothing about how long that call takes, so the curve comes from a measured baseline
    log. It is repeated behind every panel so each prefetch budget is read against the baseline in place.
    """
    seconds = [point[1] for point in reference]
    iterations = [point[0] for point in reference]
    ax.plot(
        iterations if swap_axes else seconds,
        seconds if swap_axes else iterations,
        color=POWER_TIME_REFERENCE_COLOR,
        linewidth=REFERENCE_LINE_WIDTH,
        linestyle="--",
        zorder=3,
    )


def draw_power_mh_reference(
    ax: Axes,
    final_iteration: int,
    *,
    color: str = POWER_COLOR,
    linestyle: str = "-",
    swap_axes: bool = False,
) -> list[int]:
    """Draw PowerMH's one-call-per-step trajectory and return its steps.

    The reference is the same at every prefetch budget, so it is drawn once in its own panel rather than repeated behind
    each SubTreeMH panel.
    """
    reference_steps = list(range(final_iteration + 1))
    ax.step(
        reference_steps,
        reference_steps,
        where="post" if swap_axes else "pre",
        color=color,
        linewidth=REFERENCE_LINE_WIDTH,
        linestyle=linestyle,
        zorder=5,
    )
    return reference_steps


def draw_power_mh_transitions_per_call_reference(
    ax: Axes,
    final_iteration: int,
    *,
    color: str = POWER_COLOR,
    linestyle: str = "--",
    swap_axes: bool = False,
) -> list[float]:
    """Draw PowerMH's exact one-transition-per-call trajectory."""
    iterations = list(range(1, final_iteration + 1))
    values = [1.0] * final_iteration
    ax.step(
        iterations if swap_axes else values,
        values if swap_axes else iterations,
        where="post" if swap_axes else "pre",
        color=color,
        linewidth=REFERENCE_LINE_WIDTH,
        linestyle=linestyle,
        zorder=5,
    )
    return values


def style_grid_panel(
    ax: Axes,
    kind: TrajectoryKind,
    panel_index: int,
    *,
    columns: int,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    swap_axes: bool = False,
) -> None:
    """Scale, letter, and label one slot of the small-multiple grid.

    The axis names are figure-wide, so no panel carries one of its own.
    """
    _, column_index = divmod(panel_index, columns)
    style_trajectory_axes(
        ax,
        kind,
        x_limits=x_limits,
        y_limits=y_limits,
        show_x_label=False,
        show_y_label=False,
        swap_axes=swap_axes,
    )
    add_panel_label(
        ax,
        panel_letter(panel_index),
        x=(
            PANEL_LABEL_POSITION[0]
            if column_index == 0
            else PANEL_LABEL_X_WITHOUT_Y_LABEL
        ),
        y=PANEL_LABEL_POSITION[1],
    )


def plot_trajectory_grid(
    analysis: AnalysisRows,
    kind: TrajectoryKind,
    output_prefix: Path,
    power_mh_reference: Sequence[tuple[int, float]] | None = None,
) -> Path:
    """Draw raw traces, mean, and sample SD as rank-specific small multiples."""
    trajectory_rows = kind.trajectory_rows(analysis)
    summary_rows = kind.summary_rows(analysis)
    rows_by_trace = group_rows_by_trace(trajectory_rows)
    if not rows_by_trace:
        raise ValueError("No trajectories are available to plot")
    rank_by_trace = map_trace_ranks(rows_by_trace)
    summary_by_rank = group_rows_by_rank(summary_rows)
    rank_names = sort_rank_names(set(rank_by_trace.values()))

    panel_count = len(rank_names) + (1 if kind.include_power_mh else 0)
    columns = min(PANEL_COLUMNS, panel_count)
    rows = math.ceil(panel_count / columns)
    figure, axes_grid = plt.subplots(
        rows,
        columns,
        figsize=(PANEL_WIDTH * columns, PANEL_HEIGHT * rows + 0.65),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    panel_axes = list(axes_grid.flat)
    x_limits, y_limits, final_iteration = trajectory_limits(
        trajectory_rows, summary_rows, kind
    )

    for panel_index, rank_name in enumerate(rank_names):
        ax = panel_axes[panel_index]
        if power_mh_reference:
            draw_power_mh_time_reference(
                ax, power_mh_reference, swap_axes=GRID_SWAP_AXES
            )
        iterations, means, deviations = draw_rank_trajectory(
            ax,
            rows_by_trace,
            rank_by_trace,
            summary_by_rank,
            rank_name,
            kind,
            swap_axes=GRID_SWAP_AXES,
        )
        style_grid_panel(
            ax,
            kind,
            panel_index,
            columns=columns,
            x_limits=x_limits,
            y_limits=y_limits,
            swap_axes=GRID_SWAP_AXES,
        )
        # The rule each panel plots used to title its in-panel legend, which buried the one thing telling the panels
        # apart inside the busiest part of the axes. It heads the panel instead, and the legend keeps only the three
        # handles that are identical across panels.
        ax.set_title(
            f"{SUBTREE_METHOD_LABEL} + {rank_label(rank_name)}",
            fontsize=LEGEND_FONT_SIZE,
        )
        add_terminal_statistics_annotation(
            ax, iterations, means, deviations, RANK_COLORS[rank_name]
        )
        add_panel_legend(
            ax, rank_name, with_power_mh_reference=bool(power_mh_reference)
        )

    if kind.include_power_mh:
        panel_index = len(rank_names)
        ax = panel_axes[panel_index]
        reference_steps = draw_power_mh_reference(
            ax, final_iteration, swap_axes=GRID_SWAP_AXES
        )
        style_grid_panel(
            ax,
            kind,
            panel_index,
            columns=columns,
            x_limits=x_limits,
            y_limits=y_limits,
            swap_axes=GRID_SWAP_AXES,
        )
        add_terminal_statistics_annotation(
            ax,
            reference_steps,
            [float(step) for step in reference_steps],
            [0.0 for _ in reference_steps],
            POWER_COLOR,
        )
        add_power_mh_panel_legend(ax)

    for ax in panel_axes[panel_count:]:
        ax.set_visible(False)
    figure_width = PANEL_WIDTH * columns
    figure.subplots_adjust(
        left=GRID_LEFT_MARGIN_INCHES / figure_width,
        right=0.995,
        bottom=0.19,
        top=0.94,
        wspace=0.17,
        hspace=0.10,
    )
    x_name, y_name = (
        (TRANSITION_AXIS_LABEL, kind.x_label)
        if GRID_SWAP_AXES
        else (kind.x_label, TRANSITION_AXIS_LABEL)
    )
    figure.supxlabel(x_name, fontsize=AXIS_LABEL_FONT_SIZE, y=0.015)
    figure.supylabel(y_name, fontsize=AXIS_LABEL_FONT_SIZE, x=0.004)
    return save_figure(
        figure, output_path(output_prefix, kind.figure_suffix), pad_inches=0.02
    )


# --------------------------------------------------------------------------
# Cross-batch trajectory comparison
# --------------------------------------------------------------------------


def rows_with_batch_size(rows: Sequence[Row], batch_size: int) -> list[Row]:
    """Annotate source-data rows for a cross-batch-size comparison."""
    return [{"prefetch_budget": batch_size, **row} for row in rows]


def power_mh_batch_trajectory_rows(data: PowerMHTimeData) -> list[Row]:
    """Align PowerMH raw timing rows with the aggregate empirical-time CSV."""
    return [
        {
            "prefetch_budget": "",
            "source_log": str(row["source_log"]),
            "rank": "power_mh",
            "trace_id": str(row["trace_id"]),
            "sample_idx": "",
            "block_idx": "",
            "trace_complete": True,
            "mh_iteration": int(row["mh_iteration"]),
            "cumulative_empirical_seconds": float(
                row["cumulative_empirical_seconds"]
            ),
        }
        for row in build_power_mh_time_rows(data.log_path, data.trajectories)
    ]


def power_mh_batch_summary_rows(data: PowerMHTimeData) -> list[Row]:
    """Align PowerMH summary rows with the aggregate empirical-time CSV."""
    return [
        {
            "prefetch_budget": "",
            "rank": "power_mh",
            "mh_iteration": int(row["mh_iteration"]),
            "trace_count": int(row["trace_count"]),
            "mean_cumulative_seconds": float(row["mean_cumulative_seconds"]),
            "std_cumulative_seconds": float(row["std_cumulative_seconds"]),
        }
        for row in data.summary
    ]


def best_batch_analysis_by_rank(
    batch_analyses: Sequence[tuple[int, AnalysisRows]],
    rank_names: Sequence[str],
    kind: TrajectoryKind,
    *,
    maximize: bool = False,
) -> dict[str, tuple[int, AnalysisRows]]:
    """Select each rank's best terminal mean, then SD and budget."""
    best_by_rank: dict[str, tuple[int, AnalysisRows]] = {}
    for rank_name in rank_names:
        candidates: list[tuple[float, float, int, AnalysisRows]] = []
        for batch_size, analysis in batch_analyses:
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
                    f"Expected one terminal summary for {rank_name!r} at "
                    f"prefetch budget {batch_size}, got {len(terminal_rows)}"
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
                    batch_size,
                    analysis,
                )
            )
        if not candidates:
            raise ValueError(f"No terminal summary is available for {rank_name!r}")
        _, _, batch_size, analysis = min(
            candidates, key=lambda item: (item[0], item[1], item[2])
        )
        best_by_rank[rank_name] = (batch_size, analysis)
    return best_by_rank


def plot_batch_trajectory_comparison(
    batch_analyses: Sequence[tuple[int, AnalysisRows]],
    kind: TrajectoryKind,
    output_prefix: Path,
    *,
    figure_title: str,
    model_title_label: str = "model",
    backend_title: str | None = None,
    power_mh_time: PowerMHTimeData | None = None,
    preserve_best_per_rank: bool = False,
) -> list[Path]:
    """Draw trajectories with traversal ranks as columns."""
    if not batch_analyses:
        raise ValueError("No batch-size analyses are available to compare")
    ordered = sorted(batch_analyses, key=lambda item: item[0])
    batch_sizes = [batch_size for batch_size, _ in ordered]
    if len(batch_sizes) != len(set(batch_sizes)):
        raise ValueError(f"Repeated prefetch budgets: {batch_sizes}")

    trajectory_rows = [
        row
        for batch_size, analysis in ordered
        for row in rows_with_batch_size(kind.trajectory_rows(analysis), batch_size)
    ]
    summary_rows = [
        row
        for batch_size, analysis in ordered
        for row in rows_with_batch_size(kind.summary_rows(analysis), batch_size)
    ]
    rank_names = sort_rank_names({str(row["rank"]) for row in trajectory_rows})
    if not rank_names:
        raise ValueError("No traversal ranks are available for comparison")
    best_by_rank = (
        best_batch_analysis_by_rank(ordered, rank_names, kind)
        if preserve_best_per_rank
        else {}
    )

    displayed_trajectory_rows = trajectory_rows
    displayed_summary_rows = summary_rows
    if preserve_best_per_rank:
        displayed_trajectory_rows = [
            row
            for rank_name, (batch_size, analysis) in best_by_rank.items()
            for row in rows_with_batch_size(
                kind.trajectory_rows(analysis), batch_size
            )
            if str(row["rank"]) == rank_name
        ]
        displayed_summary_rows = [
            row
            for rank_name, (batch_size, analysis) in best_by_rank.items()
            for row in rows_with_batch_size(kind.summary_rows(analysis), batch_size)
            if str(row["rank"]) == rank_name
        ]

    # PowerMH does not vary with the prefetch budget, so either reference takes one panel of its own instead of being
    # repeated in every rank panel.
    reference_columns = 1 if kind.include_power_mh or power_mh_time else 0
    separate_reference_x_axis = (
        preserve_best_per_rank
        and kind.include_power_mh
        and power_mh_time is None
    )
    columns = len(rank_names) + reference_columns
    display_row_count = 1 if preserve_best_per_rank else len(ordered)
    plot_width = BATCH_COMPARISON_PANEL_WIDTH * columns
    sidebar_width = (
        BATCH_COMPARISON_BEST_ONLY_SIDEBAR_WIDTH
        if preserve_best_per_rank
        else 0.0
    )
    sidebar_gap = (
        BATCH_COMPARISON_BEST_ONLY_SIDEBAR_GAP
        if preserve_best_per_rank
        else 0.0
    )
    figure_width = plot_width + sidebar_gap + sidebar_width
    figure_height = (
        BATCH_COMPARISON_BEST_ONLY_HEIGHT
        if preserve_best_per_rank
        else BATCH_COMPARISON_ROW_HEIGHT * display_row_count
    )
    figure, axes = plt.subplots(
        display_row_count,
        columns,
        figsize=(figure_width, figure_height),
        sharex=not separate_reference_x_axis,
        sharey=True,
        squeeze=False,
    )
    x_limits, y_limits, final_iteration = trajectory_limits(
        displayed_trajectory_rows, displayed_summary_rows, kind
    )
    if separate_reference_x_axis:
        subtree_maximum_x = max(
            max(
                float(row[kind.value_key])
                for row in displayed_trajectory_rows
            ),
            max(
                float(row[kind.mean_key]) + float(row[kind.sd_key])
                for row in displayed_summary_rows
            ),
        )
        x_limits = (0.0, max(subtree_maximum_x * 1.03, 1.0))
    if power_mh_time is not None:
        power_final_iteration = int(power_mh_time.summary[-1]["mh_iteration"])
        if power_final_iteration != final_iteration:
            raise ValueError(
                "PowerMH and SubTreeMH must reach the same final MH iteration: "
                f"{power_final_iteration} versus {final_iteration}"
            )
        power_maximum_x = max(
            max(cumulative[-1] for _, cumulative in power_mh_time.trajectories),
            max(
                float(row["mean_cumulative_seconds"])
                + float(row["std_cumulative_seconds"])
                for row in power_mh_time.summary
            ),
        )
        x_limits = (x_limits[0], max(x_limits[1], power_maximum_x * 1.03))
    if kind.log_x_axis:
        log_floor_rows = list(trajectory_rows)
        if power_mh_time is not None:
            log_floor_rows.extend(power_mh_batch_trajectory_rows(power_mh_time))
        x_limits = (
            positive_log_axis_floor(log_floor_rows, kind.value_key),
            x_limits[1],
        )

    panel_entries = (
        [
            (0, column_index, rank_name, *best_by_rank[rank_name])
            for column_index, rank_name in enumerate(rank_names)
        ]
        if preserve_best_per_rank
        else [
            (row_index, column_index, rank_name, batch_size, analysis)
            for row_index, (batch_size, analysis) in enumerate(ordered)
            for column_index, rank_name in enumerate(rank_names)
        ]
    )
    for row_index, column_index, rank_name, batch_size, analysis in panel_entries:
        rows_by_trace = group_rows_by_trace(kind.trajectory_rows(analysis))
        rank_by_trace = map_trace_ranks(rows_by_trace)
        summary_by_rank = group_rows_by_rank(kind.summary_rows(analysis))
        present_ranks = set(rank_by_trace.values())

        ax = axes[row_index, column_index]
        style_trajectory_axes(
            ax,
            kind,
            x_limits=x_limits,
            y_limits=y_limits,
            show_x_label=False,
            show_y_label=False,
            log_x_axis=kind.log_x_axis,
        )
        # A rank missing at this batch size gets a placeholder rather than an empty panel that could read as a measured
        # zero.
        if rank_name not in present_ranks:
            ax.text(
                0.5,
                0.5,
                "log not available",
                transform=ax.transAxes,
                color=GRAY,
                ha="center",
                va="center",
            )
            continue

        iterations, means, deviations = draw_rank_trajectory(
            ax,
            rows_by_trace,
            rank_by_trace,
            summary_by_rank,
            rank_name,
            kind,
        )
        add_terminal_statistics_annotation(
            ax,
            iterations,
            means,
            deviations,
            RANK_COLORS[rank_name],
            log_x_axis=kind.log_x_axis,
        )

    if not preserve_best_per_rank:
        for row_index, (batch_size, _) in enumerate(ordered):
            axes[row_index, 0].annotate(
                prefetch_size_label(batch_size),
                xy=(-0.27, 0.5),
                xycoords="axes fraction",
                ha="center",
                va="center",
                rotation=90,
            )

    if power_mh_time is not None:
        reference_ax = axes[0, -1]
        power_iterations = [
            int(row["mh_iteration"]) for row in power_mh_time.summary
        ]
        power_means = [
            float(row["mean_cumulative_seconds"])
            for row in power_mh_time.summary
        ]
        power_deviations = [
            float(row["std_cumulative_seconds"])
            for row in power_mh_time.summary
        ]
        draw_time_trajectories(
            reference_ax,
            [cumulative for _, cumulative in power_mh_time.trajectories],
            power_iterations,
            power_means,
            power_deviations,
            color=POWER_COLOR,
        )
        style_trajectory_axes(
            reference_ax,
            kind,
            x_limits=x_limits,
            y_limits=y_limits,
            show_x_label=False,
            show_y_label=False,
            log_x_axis=kind.log_x_axis,
        )
        reference_ax.tick_params(labelbottom=True)
        annotate_terminal_statistics(
            reference_ax,
            float(power_mh_time.summary[-1]["mean_cumulative_seconds"]),
            float(power_mh_time.summary[-1]["std_cumulative_seconds"]),
            color=POWER_COLOR,
        )
        for ax in axes[1:, -1]:
            ax.set_visible(False)
    elif kind.include_power_mh:
        reference_ax = axes[0, -1]
        reference_steps = draw_power_mh_reference(reference_ax, final_iteration)
        reference_x_limits = (
            (0.0, float(final_iteration) * 1.03)
            if separate_reference_x_axis
            else x_limits
        )
        style_trajectory_axes(
            reference_ax,
            kind,
            x_limits=reference_x_limits,
            y_limits=y_limits,
            show_x_label=False,
            show_y_label=False,
            log_x_axis=kind.log_x_axis,
        )
        # Nothing sits below this panel, so the x tick labels the shared axes keep for the bottom row never reach it; it
        # carries its own instead. The y scale stays labelled once per row, at the left of the figure.
        reference_ax.tick_params(labelbottom=True)
        add_terminal_statistics_annotation(
            reference_ax,
            reference_steps,
            [float(step) for step in reference_steps],
            [0.0 for _ in reference_steps],
            POWER_COLOR,
            log_x_axis=kind.log_x_axis,
        )
        for ax in axes[1:, -1]:
            ax.set_visible(False)

    left_margin_inches = (
        BATCH_COMPARISON_BEST_ONLY_LEFT_MARGIN_INCHES
        if preserve_best_per_rank
        else BATCH_COMPARISON_LEFT_MARGIN_INCHES
    )
    bottom_margin_inches = (
        BATCH_COMPARISON_BEST_ONLY_BOTTOM_MARGIN_INCHES
        if preserve_best_per_rank
        else BATCH_COMPARISON_BOTTOM_MARGIN_INCHES
    )
    plot_right = plot_width / figure_width if preserve_best_per_rank else 0.995
    sidebar_left = (
        (plot_width + sidebar_gap) / figure_width
        if preserve_best_per_rank
        else None
    )
    figure.subplots_adjust(
        left=left_margin_inches / figure_width,
        right=plot_right,
        bottom=bottom_margin_inches / figure_height,
        top=0.83 if preserve_best_per_rank else 0.875,
        wspace=BATCH_COMPARISON_WSPACE,
        hspace=BATCH_COMPARISON_HSPACE,
    )
    plot_center_x = (
        (
            axes[0, 0].get_position().x0
            + axes[0, -1].get_position().x1
        )
        / 2.0
        if preserve_best_per_rank
        else 0.5
    )
    figure.supxlabel(
        kind.x_label,
        fontsize=AXIS_LABEL_FONT_SIZE,
        x=plot_center_x,
        y=0.008,
    )
    figure.supylabel(
        TRANSITION_AXIS_LABEL,
        fontsize=AXIS_LABEL_FONT_SIZE,
        x=(
            BATCH_COMPARISON_BEST_ONLY_YLABEL_X_INCHES / figure_width
            if preserve_best_per_rank
            else 0.004
        ),
    )
    dataset_name, separator, model_name = figure_title.partition(", ")
    if not separator:
        raise ValueError(
            f"Expected '<dataset>, <model>' figure title, got {figure_title!r}"
        )
    if preserve_best_per_rank:
        if sidebar_left is None:
            raise ValueError("The best-result layout requires a sidebar")
        if backend_title is None:
            raise ValueError("The best-result layout requires a backend title")
        figure.text(
            sidebar_left,
            0.83,
            (
                f"dataset: {dataset_name}\n"
                f"{model_title_label}: {model_name}\n"
                f"backend: {backend_title}"
            ),
            color=DARK_GRAY,
            fontsize=BATCH_COMPARISON_TITLE_FONT_SIZE,
            fontweight="semibold",
            ha="left",
            va="top",
            linespacing=1.25,
        )
    else:
        figure.text(
            0.5,
            0.995,
            f"Dataset: {dataset_name}    Model: {model_name}",
            color=DARK_GRAY,
            fontsize=BATCH_COMPARISON_TITLE_FONT_SIZE,
            fontweight="semibold",
            ha="center",
            va="top",
        )
    # Column headers are placed after the layout settles so each sits over the centre of the column it names.
    headers = []
    for rank_name in rank_names:
        rank_description = rank_label(rank_name)
        if preserve_best_per_rank:
            best_batch_size, _ = best_by_rank[rank_name]
            rank_description += (
                rf"; best $N_{{\mathrm{{pf}}}}={best_batch_size}$"
            )
        headers.append(
            (
                RANK_COLORS[rank_name],
                f"{SUBTREE_METHOD_LABEL}\n({rank_description})",
            )
        )
    if power_mh_time is not None:
        headers.append((POWER_COLOR, POWER_METHOD_LABEL))
    elif kind.include_power_mh:
        headers.append(
            (POWER_COLOR, f"{POWER_METHOD_LABEL}\n({POWER_METHOD_DESCRIPTION})")
        )
    for column_index, (color, header) in enumerate(headers):
        position = axes[0, column_index].get_position()
        figure.text(
            (position.x0 + position.x1) / 2.0,
            0.97 if preserve_best_per_rank else 0.955,
            f"{panel_letter(column_index)} {header}",
            color=color,
            fontsize=BATCH_COMPARISON_HEADER_FONT_SIZE,
            ha="center",
            va="top",
        )
    legend_options = {
        "handles": shared_legend_handles(),
        "frameon": False,
        "fontsize": LEGEND_FONT_SIZE,
        "borderaxespad": 0.0,
        "handlelength": 1.8,
        "handletextpad": 0.4,
        "columnspacing": 1.0,
    }
    if preserve_best_per_rank:
        if sidebar_left is None:
            raise ValueError("The best-result layout requires a sidebar")
        figure.legend(
            loc="upper left",
            bbox_to_anchor=(sidebar_left, 0.50),
            ncol=1,
            labelspacing=0.55,
            **legend_options,
        )
    else:
        figure.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 0.905),
            ncol=3,
            **legend_options,
        )

    output_trajectory_rows = list(trajectory_rows)
    output_summary_rows = list(summary_rows)
    if power_mh_time is not None:
        output_trajectory_rows.extend(power_mh_batch_trajectory_rows(power_mh_time))
        output_summary_rows.extend(power_mh_batch_summary_rows(power_mh_time))

    figure_path = save_figure(
        figure,
        output_path(output_prefix, kind.figure_suffix),
        pad_inches=0.01 if preserve_best_per_rank else 0.02,
    )
    return [
        write_dict_rows(
            output_path(output_prefix, kind.data_suffix), output_trajectory_rows
        ),
        write_dict_rows(
            output_path(output_prefix, kind.summary_suffix), output_summary_rows
        ),
        figure_path,
    ]


def plot_accept_first_power_mh_overlay(
    batch_analyses: Sequence[tuple[int, AnalysisRows]],
    output_prefix: Path,
    *,
    figure_title: str,
    model_title_label: str,
    backend_title: str,
) -> Path:
    """Overlay the best accept-first SubTreeMH result and PowerMH in one axes."""
    rank_name = "accept_first"
    best_by_rank = best_batch_analysis_by_rank(
        batch_analyses, (rank_name,), MODEL_CALLS
    )
    best_batch_size, analysis = best_by_rank[rank_name]
    trajectory_rows = [
        row
        for row in MODEL_CALLS.trajectory_rows(analysis)
        if str(row["rank"]) == rank_name
    ]
    summary_rows = [
        row
        for row in MODEL_CALLS.summary_rows(analysis)
        if str(row["rank"]) == rank_name
    ]
    rows_by_trace = group_rows_by_trace(trajectory_rows)
    rank_by_trace = map_trace_ranks(rows_by_trace)
    summary_by_rank = group_rows_by_rank(summary_rows)
    x_limits, y_limits, final_iteration = trajectory_limits(
        trajectory_rows, summary_rows, MODEL_CALLS
    )

    plot_width = BATCH_COMPARISON_PANEL_WIDTH
    sidebar_width = BATCH_COMPARISON_BEST_ONLY_SIDEBAR_WIDTH
    sidebar_gap = BATCH_COMPARISON_BEST_ONLY_SIDEBAR_GAP
    figure_width = plot_width + sidebar_gap + sidebar_width
    figure_height = BATCH_COMPARISON_BEST_ONLY_HEIGHT
    figure, ax = plt.subplots(figsize=(figure_width, figure_height))

    iterations, means, deviations = draw_rank_trajectory(
        ax,
        rows_by_trace,
        rank_by_trace,
        summary_by_rank,
        rank_name,
        MODEL_CALLS,
    )
    reference_steps = draw_power_mh_reference(
        ax,
        final_iteration,
        color=DARK_GRAY,
        linestyle="--",
    )
    style_trajectory_axes(
        ax,
        MODEL_CALLS,
        x_limits=x_limits,
        y_limits=y_limits,
        show_x_label=False,
        show_y_label=False,
    )

    ax.text(
        0.28,
        0.95,
        f"mean = {means[-1]:.1f}\nstd = {deviations[-1]:.1f}",
        transform=ax.transAxes,
        color=RANK_COLORS[rank_name],
        fontsize=TERMINAL_MEAN_FONT_SIZE,
        fontweight="semibold",
        linespacing=TERMINAL_MEAN_LINE_SPACING,
        ha="left",
        va="top",
        zorder=6,
    )
    ax.text(
        0.98,
        0.035,
        f"mean = {float(reference_steps[-1]):.1f}\nstd = 0.0",
        transform=ax.transAxes,
        color=DARK_GRAY,
        fontsize=TERMINAL_MEAN_FONT_SIZE,
        fontweight="semibold",
        linespacing=TERMINAL_MEAN_LINE_SPACING,
        ha="right",
        va="bottom",
        zorder=6,
    )

    plot_right = plot_width / figure_width
    sidebar_left = (plot_width + sidebar_gap) / figure_width
    figure.subplots_adjust(
        left=BATCH_COMPARISON_BEST_ONLY_LEFT_MARGIN_INCHES / figure_width,
        right=plot_right,
        bottom=BATCH_COMPARISON_BEST_ONLY_BOTTOM_MARGIN_INCHES / figure_height,
        top=0.83,
    )
    axis_position = ax.get_position()
    plot_center_x = (axis_position.x0 + axis_position.x1) / 2.0
    figure.supxlabel(
        MODEL_CALLS.x_label,
        fontsize=AXIS_LABEL_FONT_SIZE,
        x=plot_center_x,
        y=0.008,
    )
    figure.supylabel(
        TRANSITION_AXIS_LABEL,
        fontsize=AXIS_LABEL_FONT_SIZE,
        x=BATCH_COMPARISON_BEST_ONLY_YLABEL_X_INCHES / figure_width,
    )
    figure.text(
        plot_center_x,
        0.97,
        (
            rf"{SUBTREE_METHOD_LABEL} (accept first; best "
            rf"$N_{{\mathrm{{pf}}}}={best_batch_size}$)"
            f"\nvs. {POWER_METHOD_LABEL}"
        ),
        color=DARK_GRAY,
        fontsize=BATCH_COMPARISON_HEADER_FONT_SIZE,
        ha="center",
        va="top",
    )

    dataset_name, separator, model_name = figure_title.partition(", ")
    if not separator:
        raise ValueError(
            f"Expected '<dataset>, <model>' figure title, got {figure_title!r}"
        )
    figure.text(
        sidebar_left,
        0.83,
        (
            f"dataset: {dataset_name}\n"
            f"{model_title_label}: {model_name}\n"
            f"backend: {backend_title}"
        ),
        color=DARK_GRAY,
        fontsize=BATCH_COMPARISON_TITLE_FONT_SIZE,
        fontweight="semibold",
        ha="left",
        va="top",
        linespacing=1.25,
    )
    figure.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=RANK_COLORS[rank_name],
                linewidth=RAW_TRACE_LINE_WIDTH,
                alpha=RAW_TRACE_ALPHA,
                label=f"{SUBTREE_METHOD_LABEL} trace",
            ),
            Line2D(
                [0],
                [0],
                color=RANK_COLORS[rank_name],
                linewidth=MEAN_LINE_WIDTH,
                label=f"{SUBTREE_METHOD_LABEL} mean",
            ),
            Patch(
                facecolor=RANK_COLORS[rank_name],
                edgecolor=TRANSPARENT,
                alpha=SD_BAND_ALPHA,
                label=r"$\pm 1$ sample SD",
            ),
            Line2D(
                [0],
                [0],
                color=DARK_GRAY,
                linewidth=REFERENCE_LINE_WIDTH,
                linestyle="--",
                label=POWER_METHOD_LABEL,
            ),
        ],
        loc="upper left",
        bbox_to_anchor=(sidebar_left, 0.50),
        ncol=1,
        labelspacing=0.55,
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
        borderaxespad=0.0,
        handlelength=1.8,
        handletextpad=0.4,
    )
    return save_figure(
        figure,
        output_path(output_prefix, MODEL_CALLS.figure_suffix),
        pad_inches=0.01,
    )


def plot_accept_first_power_mh_time_overlay(
    batch_analyses: Sequence[tuple[int, AnalysisRows]],
    power_mh_time: PowerMHTimeData,
    output_prefix: Path,
    *,
    figure_title: str,
    model_title_label: str,
    backend_title: str,
) -> Path:
    """Overlay best accept-first and measured PowerMH timing in one axes."""
    rank_name = "accept_first"
    best_by_rank = best_batch_analysis_by_rank(
        batch_analyses, (rank_name,), EMPIRICAL_TIME
    )
    best_batch_size, analysis = best_by_rank[rank_name]
    trajectory_rows = [
        row
        for row in EMPIRICAL_TIME.trajectory_rows(analysis)
        if str(row["rank"]) == rank_name
    ]
    summary_rows = [
        row
        for row in EMPIRICAL_TIME.summary_rows(analysis)
        if str(row["rank"]) == rank_name
    ]
    rows_by_trace = group_rows_by_trace(trajectory_rows)
    rank_by_trace = map_trace_ranks(rows_by_trace)
    summary_by_rank = group_rows_by_rank(summary_rows)
    x_limits, y_limits, final_iteration = trajectory_limits(
        trajectory_rows, summary_rows, EMPIRICAL_TIME
    )

    power_iterations = [
        int(row["mh_iteration"]) for row in power_mh_time.summary
    ]
    power_means = [
        float(row["mean_cumulative_seconds"]) for row in power_mh_time.summary
    ]
    power_deviations = [
        float(row["std_cumulative_seconds"]) for row in power_mh_time.summary
    ]
    if power_iterations[-1] != final_iteration:
        raise ValueError(
            "PowerMH and SubTreeMH must reach the same final MH iteration: "
            f"{power_iterations[-1]} versus {final_iteration}"
        )
    power_maximum_x = max(
        max(cumulative[-1] for _, cumulative in power_mh_time.trajectories),
        max(
            mean + deviation
            for mean, deviation in zip(
                power_means, power_deviations, strict=True
            )
        ),
    )
    x_limits = (x_limits[0], max(x_limits[1], power_maximum_x * 1.03))

    plot_width = BATCH_COMPARISON_PANEL_WIDTH
    sidebar_width = BATCH_COMPARISON_BEST_ONLY_SIDEBAR_WIDTH
    sidebar_gap = BATCH_COMPARISON_BEST_ONLY_SIDEBAR_GAP
    figure_width = plot_width + sidebar_gap + sidebar_width
    figure_height = BATCH_COMPARISON_BEST_ONLY_HEIGHT
    figure, ax = plt.subplots(figsize=(figure_width, figure_height))

    _, subtree_means, subtree_deviations = draw_rank_trajectory(
        ax,
        rows_by_trace,
        rank_by_trace,
        summary_by_rank,
        rank_name,
        EMPIRICAL_TIME,
    )
    draw_time_trajectories(
        ax,
        [cumulative for _, cumulative in power_mh_time.trajectories],
        power_iterations,
        power_means,
        power_deviations,
        color=DARK_GRAY,
        mean_linestyle="--",
    )
    style_trajectory_axes(
        ax,
        EMPIRICAL_TIME,
        x_limits=x_limits,
        y_limits=y_limits,
        show_x_label=False,
        show_y_label=False,
    )

    ax.text(
        0.38,
        0.22,
        (
            f"mean = {subtree_means[-1]:.1f}\n"
            f"std = {subtree_deviations[-1]:.1f}"
        ),
        transform=ax.transAxes,
        color=RANK_COLORS[rank_name],
        fontsize=TERMINAL_MEAN_FONT_SIZE,
        fontweight="semibold",
        linespacing=TERMINAL_MEAN_LINE_SPACING,
        ha="left",
        va="bottom",
        zorder=6,
    )
    ax.text(
        0.98,
        0.035,
        f"mean = {power_means[-1]:.1f}\nstd = {power_deviations[-1]:.1f}",
        transform=ax.transAxes,
        color=DARK_GRAY,
        fontsize=TERMINAL_MEAN_FONT_SIZE,
        fontweight="semibold",
        linespacing=TERMINAL_MEAN_LINE_SPACING,
        ha="right",
        va="bottom",
        zorder=6,
    )

    plot_right = plot_width / figure_width
    sidebar_left = (plot_width + sidebar_gap) / figure_width
    figure.subplots_adjust(
        left=BATCH_COMPARISON_BEST_ONLY_LEFT_MARGIN_INCHES / figure_width,
        right=plot_right,
        bottom=BATCH_COMPARISON_BEST_ONLY_BOTTOM_MARGIN_INCHES / figure_height,
        top=0.83,
    )
    axis_position = ax.get_position()
    plot_center_x = (axis_position.x0 + axis_position.x1) / 2.0
    figure.supxlabel(
        EMPIRICAL_TIME.x_label,
        fontsize=AXIS_LABEL_FONT_SIZE,
        x=plot_center_x,
        y=0.008,
    )
    figure.supylabel(
        TRANSITION_AXIS_LABEL,
        fontsize=AXIS_LABEL_FONT_SIZE,
        x=BATCH_COMPARISON_BEST_ONLY_YLABEL_X_INCHES / figure_width,
    )
    figure.text(
        plot_center_x,
        0.97,
        (
            rf"{SUBTREE_METHOD_LABEL} (accept first; best "
            rf"$N_{{\mathrm{{pf}}}}={best_batch_size}$)"
            f"\nvs. {POWER_METHOD_LABEL}"
        ),
        color=DARK_GRAY,
        fontsize=BATCH_COMPARISON_HEADER_FONT_SIZE,
        ha="center",
        va="top",
    )

    dataset_name, separator, model_name = figure_title.partition(", ")
    if not separator:
        raise ValueError(
            f"Expected '<dataset>, <model>' figure title, got {figure_title!r}"
        )
    figure.text(
        sidebar_left,
        0.83,
        (
            f"dataset: {dataset_name}\n"
            f"{model_title_label}: {model_name}\n"
            f"backend: {backend_title}"
        ),
        color=DARK_GRAY,
        fontsize=BATCH_COMPARISON_TITLE_FONT_SIZE,
        fontweight="semibold",
        ha="left",
        va="top",
        linespacing=1.25,
    )
    figure.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=RANK_COLORS[rank_name],
                linewidth=RAW_TRACE_LINE_WIDTH,
                alpha=RAW_TRACE_ALPHA,
                label=f"{SUBTREE_METHOD_LABEL} trace",
            ),
            Line2D(
                [0],
                [0],
                color=RANK_COLORS[rank_name],
                linewidth=MEAN_LINE_WIDTH,
                label=f"{SUBTREE_METHOD_LABEL} mean",
            ),
            Patch(
                facecolor=RANK_COLORS[rank_name],
                edgecolor=TRANSPARENT,
                alpha=SD_BAND_ALPHA,
                label=rf"{SUBTREE_METHOD_LABEL} $\pm 1$ SD",
            ),
            Line2D(
                [0],
                [0],
                color=GRAY,
                linewidth=RAW_TRACE_LINE_WIDTH,
                alpha=RAW_TRACE_ALPHA,
                label=f"{POWER_METHOD_LABEL} trace",
            ),
            Line2D(
                [0],
                [0],
                color=DARK_GRAY,
                linewidth=MEAN_LINE_WIDTH,
                linestyle="--",
                label=f"{POWER_METHOD_LABEL} mean",
            ),
            Patch(
                facecolor=DARK_GRAY,
                edgecolor=TRANSPARENT,
                alpha=SD_BAND_ALPHA,
                label=rf"{POWER_METHOD_LABEL} $\pm 1$ SD",
            ),
        ],
        loc="upper left",
        bbox_to_anchor=(sidebar_left, 0.50),
        ncol=1,
        labelspacing=0.42,
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
        borderaxespad=0.0,
        handlelength=1.8,
        handletextpad=0.4,
    )
    return save_figure(
        figure,
        output_path(output_prefix, ".pdf"),
        pad_inches=0.01,
    )


def plot_accept_first_power_mh_combined(
    batch_analyses: Sequence[tuple[int, AnalysisRows]],
    power_mh_time: PowerMHTimeData,
    output_prefix: Path,
    *,
    figure_title: str,
    model_title_label: str,
    backend_title: str,
) -> Path:
    """Place efficiency, cumulative-time, and time-per-call panels in one row."""
    rank_name = "accept_first"
    _, efficiency_analysis = best_batch_analysis_by_rank(
        batch_analyses,
        (rank_name,),
        TRANSITIONS_PER_CALL,
        maximize=True,
    )[rank_name]
    _, time_analysis = best_batch_analysis_by_rank(
        batch_analyses, (rank_name,), EMPIRICAL_TIME
    )[rank_name]
    _, time_per_call_analysis = best_batch_analysis_by_rank(
        batch_analyses, (rank_name,), TIME_PER_CALL
    )[rank_name]

    efficiency_rows = [
        row
        for row in TRANSITIONS_PER_CALL.trajectory_rows(efficiency_analysis)
        if str(row["rank"]) == rank_name
    ]
    efficiency_summary_rows = [
        row
        for row in TRANSITIONS_PER_CALL.summary_rows(efficiency_analysis)
        if str(row["rank"]) == rank_name
    ]
    efficiency_rows_by_trace = group_rows_by_trace(efficiency_rows)
    efficiency_rank_by_trace = map_trace_ranks(efficiency_rows_by_trace)
    efficiency_summary_by_rank = group_rows_by_rank(
        efficiency_summary_rows
    )
    (
        efficiency_x_limits,
        efficiency_y_limits,
        final_iteration,
    ) = trajectory_limits(
        efficiency_rows,
        efficiency_summary_rows,
        TRANSITIONS_PER_CALL,
    )

    time_rows = convert_time_rows_to_minutes(
        [
            row
            for row in EMPIRICAL_TIME.trajectory_rows(time_analysis)
            if str(row["rank"]) == rank_name
        ],
        EMPIRICAL_TIME.value_key,
    )
    time_summary_rows = convert_time_rows_to_minutes(
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
    time_summary_by_rank = group_rows_by_rank(time_summary_rows)
    time_x_limits, time_y_limits, time_final_iteration = trajectory_limits(
        time_rows, time_summary_rows, EMPIRICAL_TIME
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
    time_per_call_summary_rows = [
        row
        for row in TIME_PER_CALL.summary_rows(time_per_call_analysis)
        if str(row["rank"]) == rank_name
    ]
    time_per_call_rows_by_trace = group_rows_by_trace(time_per_call_rows)
    time_per_call_rank_by_trace = map_trace_ranks(
        time_per_call_rows_by_trace
    )
    time_per_call_summary_by_rank = group_rows_by_rank(
        time_per_call_summary_rows
    )
    (
        time_per_call_x_limits,
        time_per_call_y_limits,
        time_per_call_final_iteration,
    ) = trajectory_limits(
        time_per_call_rows,
        time_per_call_summary_rows,
        TIME_PER_CALL,
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
    power_maximum_x = max(
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
    time_x_limits = (
        time_x_limits[0],
        max(time_x_limits[1], power_maximum_x * 1.03),
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
            value
            for _, trajectory in power_mh_time.time_per_call_trajectories
            for value in trajectory
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
    time_per_call_x_limits = (
        time_per_call_x_limits[0],
        max(
            time_per_call_x_limits[1],
            power_maximum_time_per_call * 1.03,
        ) * combined_style.TIME_PER_CALL_Y_HEADROOM,
    )

    panel_count = 3
    plot_width = COMBINED_PANEL_WIDTH * panel_count
    figure_width = plot_width
    figure_height = COMBINED_FIGURE_HEIGHT
    figure, axes = plt.subplots(
        1,
        panel_count,
        figsize=(figure_width, figure_height),
        sharex=True,
    )
    time_ax, efficiency_ax, time_per_call_ax = axes

    _, efficiency_means, efficiency_deviations = draw_rank_trajectory(
        efficiency_ax,
        efficiency_rows_by_trace,
        efficiency_rank_by_trace,
        efficiency_summary_by_rank,
        rank_name,
        TRANSITIONS_PER_CALL,
        swap_axes=True,
        color=SUBTREE_COMPARISON_COLOR,
    )
    efficiency_reference = draw_power_mh_transitions_per_call_reference(
        efficiency_ax,
        final_iteration,
        color=POWER_COLOR,
        linestyle="--",
        swap_axes=True,
    )
    style_trajectory_axes(
        efficiency_ax,
        TRANSITIONS_PER_CALL,
        x_limits=efficiency_x_limits,
        y_limits=efficiency_y_limits,
        show_x_label=False,
        show_y_label=False,
        swap_axes=True,
    )
    efficiency_ax.set_ylim(*COMBINED_EFFICIENCY_Y_LIMITS)
    efficiency_ax.set_ylabel(
        TRANSITIONS_PER_CALL.x_label,
        fontsize=COMBINED_FIGURE_FONT_SIZE,
        y=combined_style.Y_LABEL_Y,
    )
    efficiency_ax.set_xlabel(
        TRANSITION_AXIS_LABEL,
        fontsize=COMBINED_FIGURE_FONT_SIZE,
        labelpad=0,
    )
    combined_style.statistic_text(
        efficiency_ax,
        combined_style.SUBTREE_TOP_TEXT_X,
        combined_style.SUBTREE_TOP_TEXT_Y,
        SUBTREE_METHOD_LABEL,
        efficiency_means[-1],
        efficiency_deviations[-1],
        color=SUBTREE_COMPARISON_COLOR,
        ha="center",
        va="top",
    )
    combined_style.statistic_text(
        efficiency_ax,
        0.50,
        COMBINED_POWER_EFFICIENCY_TEXT_Y,
        POWER_METHOD_LABEL,
        efficiency_reference[-1],
        0.0,
        color=POWER_COLOR,
        ha="center",
        va="bottom",
    )

    _, time_means, time_deviations = draw_rank_trajectory(
        time_ax,
        time_rows_by_trace,
        time_rank_by_trace,
        time_summary_by_rank,
        rank_name,
        EMPIRICAL_TIME,
        swap_axes=True,
        color=SUBTREE_COMPARISON_COLOR,
    )
    draw_time_trajectories(
        time_ax,
        [
            [seconds_to_minutes(value) for value in cumulative]
            for _, cumulative in power_mh_time.trajectories
        ],
        power_iterations,
        power_means,
        power_deviations,
        color=POWER_COLOR,
        trace_color=POWER_COLOR,
        mean_linestyle="--",
        swap_axes=True,
    )
    style_trajectory_axes(
        time_ax,
        EMPIRICAL_TIME,
        x_limits=time_x_limits,
        y_limits=time_y_limits,
        show_x_label=False,
        show_y_label=False,
        swap_axes=True,
    )
    time_ax.set_ylabel(
        COMBINED_TIME_AXIS_LABEL,
        fontsize=COMBINED_FIGURE_FONT_SIZE,
        y=combined_style.Y_LABEL_Y,
    )
    time_ax.set_xlabel(
        TRANSITION_AXIS_LABEL,
        fontsize=COMBINED_FIGURE_FONT_SIZE,
        labelpad=0,
    )
    combined_style.statistic_text(
        time_ax,
        0.98,
        0.035,
        SUBTREE_METHOD_LABEL,
        time_means[-1],
        time_deviations[-1],
        color=SUBTREE_COMPARISON_COLOR,
        ha="right",
        va="bottom",
    )
    combined_style.statistic_text(
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

    _, time_per_call_means, time_per_call_deviations = draw_rank_trajectory(
        time_per_call_ax,
        time_per_call_rows_by_trace,
        time_per_call_rank_by_trace,
        time_per_call_summary_by_rank,
        rank_name,
        TIME_PER_CALL,
        swap_axes=True,
        color=SUBTREE_COMPARISON_COLOR,
    )
    draw_time_trajectories(
        time_per_call_ax,
        [
            values
            for _, values in power_mh_time.time_per_call_trajectories
        ],
        power_time_per_call_iterations,
        power_time_per_call_means,
        power_time_per_call_deviations,
        color=POWER_COLOR,
        trace_color=POWER_COLOR,
        mean_linestyle="--",
        swap_axes=True,
    )
    style_trajectory_axes(
        time_per_call_ax,
        TIME_PER_CALL,
        x_limits=time_per_call_x_limits,
        y_limits=time_per_call_y_limits,
        show_x_label=False,
        show_y_label=False,
        swap_axes=True,
    )
    time_per_call_ax.set_ylabel(
        COMBINED_TIME_PER_CALL_AXIS_LABEL,
        fontsize=COMBINED_FIGURE_FONT_SIZE,
        y=combined_style.Y_LABEL_Y,
    )
    time_per_call_ax.set_xlabel(
        TRANSITION_AXIS_LABEL,
        fontsize=COMBINED_FIGURE_FONT_SIZE,
        labelpad=0,
    )
    combined_style.statistic_text(
        time_per_call_ax,
        combined_style.SUBTREE_TOP_TEXT_X,
        combined_style.SUBTREE_TOP_TEXT_Y,
        SUBTREE_METHOD_LABEL,
        time_per_call_means[-1],
        time_per_call_deviations[-1],
        color=SUBTREE_COMPARISON_COLOR,
        ha="center",
        va="top",
    )
    combined_style.statistic_text(
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
        axes,
        ("(a)", "(b)", "(c)"),
        strict=True,
    ):
        ax.text(
            -0.12,
            combined_style.PANEL_LABEL_Y,
            rf"\textbf{{{panel_label}}}",
            transform=ax.transAxes,
            color=combined_style.BLACK,
            fontsize=COMBINED_FIGURE_FONT_SIZE,
            fontweight="bold",
            ha="left",
            va="top",
            clip_on=False,
            zorder=7,
        )
        ax.minorticks_off()
        ax.tick_params(axis="both", labelsize=COMBINED_FIGURE_FONT_SIZE)

    figure.subplots_adjust(
        left=BATCH_COMPARISON_BEST_ONLY_LEFT_MARGIN_INCHES / figure_width,
        right=0.995,
        bottom=0.55 / figure_height,
        top=COMBINED_PLOT_TOP,
        wspace=0.26,
    )
    first_position = time_ax.get_position()
    time_per_call_position = time_per_call_ax.get_position()
    dataset_name, separator, model_name = figure_title.partition(", ")
    if not separator:
        raise ValueError(
            f"Expected '<dataset>, <model>' figure title, got {figure_title!r}"
        )
    if dataset_name in {"lcb_v6", "LiveCodeBench-release_v6"}:
        dataset_name = "LCB V6"
    figure.text(
        (first_position.x0 + time_per_call_position.x1) / 2.0,
        COMBINED_TITLE_Y,
        (
            f"dataset: {dataset_name}, "
            f"{model_title_label}: {model_name}, "
            f"inference engine: {backend_title}"
        ),
        color=DARK_GRAY,
        fontsize=COMBINED_FIGURE_FONT_SIZE,
        fontweight="semibold",
        ha="center",
        va="top",
    )
    combined_style.add_method_legends(
        figure,
        left=first_position.x0,
        right=time_per_call_position.x1,
        legend_y=COMBINED_LEGEND_Y,
    )
    return save_figure(
        figure,
        output_path(output_prefix, ".pdf"),
        pad_inches=0.01,
    )


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------


def render_batch_comparisons(
    completed: Sequence[tuple[ExperimentGroup, AnalysisRows]],
    power_mh_data_by_config: dict[str, PowerMHTimeData],
) -> list[Path]:
    """Render model-call and time comparisons across compatible batch sizes."""
    outputs: list[Path] = []
    for comparison in build_batch_comparisons(completed):
        entries = comparison.batch_analyses
        first_source_log = str(entries[0][1].events[0]["source_log"])
        subtree_log = comparison.directory / first_source_log
        power_mh_time = power_mh_data_by_config.get(
            shared_run_config_of(subtree_log)
        )
        if power_mh_time is not None:
            validate_power_mh_match(subtree_log, power_mh_time.log_path)
        # A cross-budget prefix already names the dataset and base model it holds fixed -- as the identity itself when
        # the directory holds one comparison, and as the head of the full run configuration when it holds several -- so
        # the empirical-time figures share it directly.
        empirical_comparison_prefix = comparison.output_prefix
        model_call_output_prefix = output_path(
            comparison.output_prefix, MODEL_CALL_OUTPUT_SUFFIX
        )
        accept_first_power_mh_output_prefix = output_path(
            model_call_output_prefix, ".accept-first-power-mh"
        )
        empirical_output_prefix = output_path(
            empirical_comparison_prefix, MODEL_CALL_OUTPUT_SUFFIX
        )
        empirical_overlay_outputs = (
            [
                plot_accept_first_power_mh_time_overlay(
                    entries,
                    power_mh_time,
                    output_path(
                        empirical_output_prefix,
                        ".empirical-time.accept-first-power-mh",
                    ),
                    figure_title=comparison.figure_title,
                    model_title_label="base LLM",
                    backend_title=backend_of(subtree_log),
                )
            ]
            if power_mh_time is not None
            else []
        )
        combined_overlay_outputs = (
            [
                plot_accept_first_power_mh_combined(
                    entries,
                    power_mh_time,
                    comparison.directory
                    / (
                        f"{comparison.output_prefix.name}."
                        "accept-first-power-mh."
                        "model-calls-and-empirical-time"
                    ),
                    figure_title=comparison.figure_title,
                    model_title_label="base LLM",
                    backend_title=backend_of(subtree_log),
                )
            ]
            if power_mh_time is not None
            else []
        )
        comparison_outputs = [
            *plot_batch_trajectory_comparison(
                entries,
                MODEL_CALLS,
                model_call_output_prefix,
                figure_title=comparison.figure_title,
                model_title_label="base LLM",
                backend_title=backend_of(subtree_log),
                preserve_best_per_rank=True,
            ),
            plot_accept_first_power_mh_overlay(
                entries,
                accept_first_power_mh_output_prefix,
                figure_title=comparison.figure_title,
                model_title_label="base LLM",
                backend_title=backend_of(subtree_log),
            ),
            *plot_batch_trajectory_comparison(
                entries,
                EMPIRICAL_TIME,
                empirical_output_prefix,
                figure_title=comparison.figure_title,
                model_title_label="base LLM",
                backend_title=backend_of(subtree_log),
                power_mh_time=power_mh_time,
                preserve_best_per_rank=True,
            ),
            *empirical_overlay_outputs,
            *combined_overlay_outputs,
        ]
        outputs.extend(comparison_outputs)
        print(
            f"\n{comparison.directory}\n"
            f"  combined prefetch budgets: "
            f"{', '.join(str(size) for size, _ in sorted(entries))}"
        )
        for path in comparison_outputs:
            print(f"  wrote {path}")
    return outputs


def write_analysis_csvs(analysis: AnalysisRows, output_prefix: Path) -> list[Path]:
    """Write source-data and summary tables in display order."""
    tables = (
        (".events.csv", analysis.events),
        (".cumulative.csv", analysis.cumulative),
        (".summary.csv", analysis.summary),
        (".empirical-time.csv", analysis.empirical_time),
        (".empirical-time.summary.csv", analysis.empirical_time_summary),
        (".time-per-call.csv", analysis.time_per_call),
        (".time-per-call.summary.csv", analysis.time_per_call_summary),
        (".transitions-per-call.csv", analysis.transitions_per_call),
        (
            ".transitions-per-call.summary.csv",
            analysis.transitions_per_call_summary,
        ),
    )
    return [
        write_dict_rows(output_path(output_prefix, suffix), rows)
        for suffix, rows in tables
    ]


def print_run_report(
    group: ExperimentGroup,
    extracted: ExtractedLogs,
    analysis: AnalysisRows,
    output_paths: Sequence[Path],
    *,
    include_incomplete: bool,
) -> None:
    """Print provenance, selection, incomplete-data, and output details."""
    call_traces = [trace for trace in extracted.traces if trace.calls]
    empty_trace_count = len(extracted.traces) - len(call_traces)
    complete_count = sum(trace.complete for trace in call_traces)
    incomplete_count = len(call_traces) - complete_count
    rank_count = len({str(row["rank"]) for row in analysis.cumulative})
    mh_step_counts = sorted(
        {
            total_mh_steps
            for trace in extracted.traces
            if (total_mh_steps := trace.total_mh_steps) is not None
        }
    )

    print(
        f"\n{group.directory}\n"
        f"  logs: {len(group.log_paths)}; data logs: {len(extracted.data_log_paths)}; "
        f"ranks: {rank_count}"
    )
    if extracted.empty_log_paths:
        print(
            "  logs without model-call traces: "
            + ", ".join(path.name for path in extracted.empty_log_paths)
        )
    print(
        f"  traces with calls: {len(call_traces)} total, {complete_count} complete, "
        f"{incomplete_count} incomplete; selected {analysis.traces_per_rank} per rank"
    )
    if empty_trace_count:
        print(f"  blocks without a target-model call: {empty_trace_count} omitted")
    if incomplete_count:
        action = "included as observed prefixes" if include_incomplete else "excluded"
        print(f"  incomplete traces: {action}")
    print(
        f"  target-model calls: {len(analysis.events)}; configured MH steps: "
        f"{mh_step_counts}; uncertainty: mean ± sample SD"
    )
    for path in output_paths:
        print(f"  wrote {path}")


def validate_power_mh_match(subtree_log: Path, power_mh_log: Path) -> None:
    """Require a timing baseline from the same dataset and model."""
    subtree_identity = (dataset_of(subtree_log), model_of(subtree_log))
    power_identity = (dataset_of(power_mh_log), model_of(power_mh_log))
    if subtree_identity != power_identity:
        raise ValueError(
            "PowerMH empirical time must use the same dataset and model as "
            f"SubTreeMH: {power_identity} versus {subtree_identity}"
        )


def resolve_power_mh_time_data(
    directory: Path, explicit_log: Path | None
) -> dict[str, PowerMHTimeData]:
    """Map each shared run configuration to its parsed PowerMH timing run.

    Dataset and model are both part of the key, along with the other settings common to PowerMH and SubTreeMH. This
    prevents a cumulative-time curve from another workload from entering a visually plausible comparison. An explicit
    log must exist and parse; discovery is best-effort, since a directory holds a matching baseline only sometimes.
    """
    if explicit_log is not None:
        log_path = explicit_log.resolve()
        return {shared_run_config_of(log_path): load_power_mh_time_data(log_path)}

    references: dict[str, PowerMHTimeData] = {}
    seen: dict[str, Path] = {}
    for candidate in sorted(directory.rglob(POWER_MH_LOG_GLOB)):
        shared_config = shared_run_config_of(candidate)
        if shared_config in seen:
            raise ValueError(
                "Two PowerMH baselines share the same dataset, model, and run "
                f"settings under {directory}: {seen[shared_config].name} and "
                f"{candidate.name}; choose one with "
                "--power-mh-log"
            )
        seen[shared_config] = candidate
        try:
            references[shared_config] = load_power_mh_time_data(candidate)
        except NoModelCallTracesError:
            continue
    return references


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the recursive extraction and plotting command-line interface."""
    parser = argparse.ArgumentParser(
        description=(
            "Recursively group subtree-prefetch logs by run configuration, "
            "extract MH-step yield per target-model call, write source-data "
            "CSVs, and draw rank-specific model-call and time figures."
        )
    )
    parser.add_argument(
        "--directory",
        "--log-dir",
        dest="directory",
        type=Path,
        default=DEFAULT_DIRECTORY,
        help="Root directory searched recursively for sampler logs.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=None,
        help="Custom output prefix; valid only when the input has one group.",
    )
    parser.add_argument(
        "--include-incomplete",
        action="store_true",
        help="Include only the observed prefix of an open trace at EOF.",
    )
    parser.add_argument(
        "--power-mh-log",
        type=Path,
        default=None,
        help=(
            "PowerMH baseline log used by the empirical-time figures. Its "
            "measured mean is drawn behind the single-budget panels, and its "
            "full timing distribution is shown in the cross-budget figure. "
            "Defaults to the one *.powerMH.*.log found beside the subtree "
            "logs, if any."
        ),
    )
    parser.add_argument(
        "--traces-per-rank",
        type=int,
        default=DEFAULT_TRACES_PER_RANK,
        help=(
            "Maximum number of deterministic traces per rank; the common "
            "count is reduced to the smallest available rank."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Render every configuration-safe experiment group under a directory."""
    args = build_arg_parser().parse_args(argv)
    if args.traces_per_rank <= 0:
        raise SystemExit("--traces-per-rank must be positive")

    directory = args.directory.resolve()
    try:
        groups = discover_experiment_groups(directory, MODEL_CALL_OUTPUT_SUFFIX)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if args.output_prefix is not None and len(groups) != 1:
        raise SystemExit(
            "--output-prefix requires exactly one experiment group; "
            f"discovered {len(groups)}"
        )

    try:
        power_mh_data_by_config = resolve_power_mh_time_data(
            directory, args.power_mh_log
        )
    except (ValueError, NoModelCallTracesError) as error:
        raise SystemExit(str(error)) from error

    output_count = 0
    completed: list[tuple[ExperimentGroup, AnalysisRows]] = []
    for group in groups:
        output_prefix = (
            args.output_prefix.resolve()
            if args.output_prefix is not None
            else group.output_prefix
        )
        try:
            extracted = extract_logs(
                group.log_paths, args.include_incomplete, tuple(RANK_COLORS)
            )
            analysis = build_analysis_rows(extracted, args.traces_per_rank)
            group_log = group.log_paths[0]
            group_power_mh = power_mh_data_by_config.get(
                shared_run_config_of(group_log)
            )
            if group_power_mh is not None:
                validate_power_mh_match(group_log, group_power_mh.log_path)
            output_paths = [
                *write_analysis_csvs(analysis, output_prefix),
                plot_trajectory_grid(analysis, MODEL_CALLS, output_prefix),
                plot_trajectory_grid(
                    analysis,
                    EMPIRICAL_TIME,
                    output_prefix,
                    power_mh_reference=(
                        group_power_mh.mean_reference
                        if group_power_mh is not None
                        else None
                    ),
                ),
            ]
        except (OSError, ValueError) as error:
            raise SystemExit(
                f"Failed experiment group {group.directory}/{group.signature}: {error}"
            ) from error
        output_count += len(output_paths)
        print_run_report(
            group,
            extracted,
            analysis,
            output_paths,
            include_incomplete=args.include_incomplete,
        )
        completed.append((group, analysis))

    try:
        batch_comparison_paths = render_batch_comparisons(
            completed, power_mh_data_by_config
        )
    except (OSError, ValueError) as error:
        raise SystemExit(f"Failed cross-batch-size comparison: {error}") from error
    output_count += len(batch_comparison_paths)

    print(
        f"\nRendered {len(groups)} experiment groups from "
        f"{sum(len(group.log_paths) for group in groups)} logs; "
        f"wrote {output_count} files."
    )


if __name__ == "__main__":
    main()
