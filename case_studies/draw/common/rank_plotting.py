"""Shared drawing helpers for traversal-rank comparison figures.

This library is not run directly; invoke a sibling drawing CLI instead.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import MaxNLocator

from case_studies.draw.common.plot_helpers import hide_secondary_ticks
from case_studies.extract.analysis.rank_analysis import RankComparison
from case_studies.extract.logs.model_call_traces import Row, experiment_figure_title
from case_studies.plot_config import (
    COLOR_PALETTE as FALLBACK_COLORS,
)
from case_studies.plot_config import (
    DISTRIBUTION_MEAN_LINE_WIDTH as MEDIAN_LINE_WIDTH,
)
from case_studies.plot_config import (
    GRAY as RAW_TRACE_COLOR,
)
from case_studies.plot_config import (
    LARGE_FONT_SIZE as BASE_FONT_SIZE,
)
from case_studies.plot_config import (
    PANEL_LABEL_FONT_WEIGHT,
    TRANSITION_RANK_COLORS,
    TRANSPARENT,
)

BOX_WIDTH = 0.62
BOX_LINE_WIDTH = 0.7
ANNOTATION_SIZE = int(BASE_FONT_SIZE * 0.92)
FLIER_ALPHA = 0.35
PREFETCH_BUDGET_AXIS_LABEL = r"prefetch budget $N_{\mathrm{pf}}$"
TIME_AXIS_LABEL = "cumulative model-call time (sec)"
TRANSITION_AXIS_LABEL = r"total MH transitions ($K$)"
# The one encoding every transitions-against-cost figure draws with: the standalone PowerMH figure, the
# PowerMH-versus-SubTreeMH comparison, and the empirical-time panels of the rank sweep, which import these rather than
# keeping the copies they used to.
RAW_TRACE_LINE_WIDTH = 0.85
RAW_TRACE_ALPHA = 0.48
MEAN_LINE_WIDTH = 2.2
SD_BAND_ALPHA = 0.18
# The trajectories climb away from the x axis, leaving the bottom-right corner free for the terminal mean.
TERMINAL_MEAN_POSITION = (0.975, 0.035)
TERMINAL_MEAN_FONT_SIZE = 14.0
TERMINAL_MEAN_LINE_SPACING = 1.35
TIME_PANEL_TICK_BINS = 5


def assign_rank_colors(rank_fns: Sequence[str]) -> dict[str, str]:
    """Assign stable rank colors with a fallback for new methods."""
    return {
        rank_fn: TRANSITION_RANK_COLORS.get(
            rank_fn,
            FALLBACK_COLORS[index % len(FALLBACK_COLORS)],
        )
        for index, rank_fn in enumerate(rank_fns)
    }


def prefetch_size_label(batch_size: int) -> str:
    """Format the prefetch-node budget using the manuscript notation."""
    return rf"$N_{{\mathrm{{pf}}}} = {batch_size}$"


def format_figure_title(title: str) -> str:
    """Format a validated dataset and base-LLM title."""
    dataset_name, separator, base_llm_name = title.partition(", ")
    if not separator:
        raise ValueError(f"Figure title must use '<dataset>, <base LLM>': {title!r}")
    return f"dataset: {dataset_name}, base LLM: {base_llm_name}"


def comparison_figure_title(comparisons: Sequence[RankComparison]) -> str:
    """Format one validated dataset and base-LLM title for a rank sweep."""
    titles = {
        experiment_figure_title(comparison.collected[0].log_path)
        for comparison in comparisons
        if comparison.collected
    }
    if len(titles) != 1:
        raise ValueError(
            "All prefetch budgets must share one dataset and base LLM; "
            f"found {sorted(titles)}"
        )
    return format_figure_title(next(iter(titles)))


def add_panel_label(
    ax: Axes,
    label: str,
    *,
    x: float = -0.1,
    y: float = 1.0,
) -> None:
    """Add a compact panel label at a configurable axes-relative position."""
    if plt.rcParams["text.usetex"]:
        label = rf"\textbf{{{label}}}"
        fontweight = None
    else:
        fontweight = PANEL_LABEL_FONT_WEIGHT
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=ANNOTATION_SIZE,
        fontweight=fontweight,
        ha="left",
        va="bottom",
    )


def style_quantitative_axis(ax: Axes) -> None:
    """Apply consistent quantitative-axis styling."""
    hide_secondary_ticks(ax, length=3, width=0.8)


def draw_time_trajectories(
    ax: Axes,
    trajectories: Sequence[Sequence[float]],
    iterations: Sequence[int],
    means: Sequence[float],
    deviations: Sequence[float],
    *,
    color: str,
    mean_linestyle: str = "-",
    swap_axes: bool = False,
    trace_color: str | None = None,
) -> None:
    """Draw per-sample cumulative-time traces, their mean, and a sample SD band.

    By default, time runs along x and realized MH transitions along y. Set ``swap_axes`` to place transitions along x
    and cumulative time along y. Traces are stepped because a model call advances the transition count all at once.
    """
    for cumulative in trajectories:
        trace_iterations = list(range(1, len(cumulative) + 1))
        ax.step(
            trace_iterations if swap_axes else list(cumulative),
            list(cumulative) if swap_axes else trace_iterations,
            where="post" if swap_axes else "pre",
            color=trace_color or RAW_TRACE_COLOR,
            linewidth=RAW_TRACE_LINE_WIDTH,
            alpha=RAW_TRACE_ALPHA,
            zorder=2,
        )

    lower = [
        max(mean - sd, 0.0)
        for mean, sd in zip(means, deviations, strict=True)
    ]
    upper = [mean + sd for mean, sd in zip(means, deviations, strict=True)]
    fill_between = ax.fill_between if swap_axes else ax.fill_betweenx
    fill_between(
        list(iterations),
        lower,
        upper,
        step="post" if swap_axes else "pre",
        color=color,
        alpha=SD_BAND_ALPHA,
        linewidth=0,
        zorder=1,
    )
    ax.step(
        list(iterations) if swap_axes else list(means),
        list(means) if swap_axes else list(iterations),
        where="post" if swap_axes else "pre",
        color=color,
        linewidth=MEAN_LINE_WIDTH,
        linestyle=mean_linestyle,
        zorder=4,
    )


def annotate_terminal_statistics(
    ax: Axes, mean: float, sample_sd: float, *, color: str
) -> None:
    """Label a trajectory's terminal mean and spread in the corner it leaves empty.

    Both are read at the final MH iteration, so they describe the cost of a whole run: how long it took on average, and
    how widely the samples varied around that.
    """
    ax.text(
        *TERMINAL_MEAN_POSITION,
        f"mean = {mean:.1f}\nstd = {sample_sd:.1f}",
        transform=ax.transAxes,
        color=color,
        fontsize=TERMINAL_MEAN_FONT_SIZE,
        fontweight="semibold",
        linespacing=TERMINAL_MEAN_LINE_SPACING,
        ha="right",
        va="bottom",
        zorder=6,
    )


def draw_cumulative_time_panel(
    ax: Axes,
    trajectories: Sequence[Sequence[float]],
    summary: Sequence[Row],
    *,
    color: str,
    show_y_label: bool = True,
) -> None:
    """Draw one method's transitions-against-time panel, labelled and styled.

    Composes the three pieces above into the whole panel: the traces and their mean, the terminal statistics, and the
    axis treatment. The standalone PowerMH figure and each half of the side-by-side comparison are all this panel, which
    is what lets them be read against one another.

    ``show_y_label`` is off for a panel that shares its y axis with the one to its left, where a second copy of the
    label would only repeat it.
    """
    means = [float(row["mean_cumulative_seconds"]) for row in summary]
    draw_time_trajectories(
        ax,
        trajectories,
        [int(row["mh_iteration"]) for row in summary],
        means,
        [float(row["std_cumulative_seconds"]) for row in summary],
        color=color,
    )
    annotate_terminal_statistics(
        ax,
        means[-1],
        float(summary[-1]["std_cumulative_seconds"]),
        color=color,
    )

    ax.set_xlabel(TIME_AXIS_LABEL)
    if show_y_label:
        ax.set_ylabel(TRANSITION_AXIS_LABEL)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=TIME_PANEL_TICK_BINS, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=TIME_PANEL_TICK_BINS, integer=True))
    style_quantitative_axis(ax)


def _style_boxplot(artists: dict[str, list], colors: Sequence[str]) -> None:
    """Color each hollow boxplot distribution by traversal rank."""
    for index, color in enumerate(colors):
        artists["boxes"][index].set(
            facecolor=TRANSPARENT,
            edgecolor=color,
            linewidth=BOX_LINE_WIDTH,
        )
        artists["medians"][index].set(color=color, linewidth=MEDIAN_LINE_WIDTH)
        for line in (
            *artists["whiskers"][2 * index : 2 * index + 2],
            *artists["caps"][2 * index : 2 * index + 2],
        ):
            line.set(color=color, linewidth=BOX_LINE_WIDTH)
        artists["fliers"][index].set(
            marker=".",
            markersize=2.4,
            markeredgecolor=color,
            markerfacecolor=color,
            alpha=FLIER_ALPHA,
        )


def draw_vertical_boxplots(
    ax: Axes,
    values: Sequence[Sequence[float]],
    *,
    positions: Sequence[float],
    tick_labels: Sequence[str],
    colors: Sequence[str],
) -> None:
    """Draw styled vertical boxplots at caller-specified categories."""
    artists = ax.boxplot(
        [list(series) for series in values],
        positions=list(positions),
        widths=BOX_WIDTH,
        orientation="vertical",
        patch_artist=True,
        tick_labels=list(tick_labels),
    )
    _style_boxplot(artists, colors)
