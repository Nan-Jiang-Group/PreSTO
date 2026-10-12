#!/usr/bin/env python3
"""Plot realized MH transitions across prefetch sizes and traversal rules.

Run with:

    /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/.venv/bin/python \
        /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/draw/draw_rank_realized_mh_transitions.py \
        --directory /path/to/vllm/logs
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import NullLocator

from case_studies import paths
from case_studies.draw.common.figure_io import save_figure, write_dict_rows
from case_studies.draw.common.plot_helpers import panel_letter
from case_studies.draw.common.rank_plotting import (
    PREFETCH_BUDGET_AXIS_LABEL,
    assign_rank_colors,
    comparison_figure_title,
    draw_vertical_boxplots,
    style_quantitative_axis,
)
from case_studies.extract.analysis.rank_analysis import (
    RankComparison,
    all_batch_rank_names,
    collect_rank_comparisons,
    max_realized_transition,
    rank_observation,
)
from case_studies.extract.run_naming import dataset_model_prefix, rank_label
from case_studies.plot_config import SUBTREE_METHOD_LABEL, apply_plot_style, scale_font_sizes

apply_plot_style()

FONT_SCALE = 1.18
scale_font_sizes(
    (
        "font.size",
        "axes.titlesize",
        "axes.labelsize",
        "xtick.labelsize",
        "ytick.labelsize",
        "legend.fontsize",
        "figure.titlesize",
    ),
    FONT_SCALE,
)

DEFAULT_DIRECTORY = paths.LOGS_DIR / "math500" / "2026-08-08" / "vllm"
FIGURE_SUFFIX = ".all-rank-realized-mh-transitions.pdf"
CSV_SUFFIX = ".all-rank-realized-mh-transitions.csv"

PANEL_WIDTH = 2.65
FIGURE_HEIGHT = 3.60
TOP_MARGIN = 0.75
BOTTOM_MARGIN = 0.20
LEFT_MARGIN = 0.12
RIGHT_MARGIN = 0.995
COLUMN_SPACE = 0.10
# How far left of the axes the y label sits, holding it just clear of panel (a)'s tick labels. Measured in inches rather
# than figure fractions because the figure grows one PANEL_WIDTH per traversal rule: a fraction tuned to the five-panel
# case would drift into the ticks on a two-panel one. Pinning it at the figure edge instead left an inch of blank
# between label and ticks, and the tight bounding box crops whatever stays empty to the left.
Y_LABEL_INSET_INCHES = 0.46
TRANSITION_TICK_TARGET = 8


def draw_transition_panel(
    ax: Axes,
    comparisons: Sequence[RankComparison],
    rank_name: str,
    max_transition: int,
    *,
    panel_label: str,
    show_y_ticks: bool,
) -> None:
    """Draw vertical transition distributions across prefetch sizes."""
    batch_sizes = [comparison.batch_size for comparison in comparisons]
    observations = [
        (position, observation)
        for position, comparison in enumerate(comparisons, start=1)
        if (observation := rank_observation(comparison, rank_name)) is not None
    ]
    positions = [float(position) for position, _ in observations]
    values = [
        [float(call.path_nodes - 1) for call in observation[0].calls]
        for _, observation in observations
    ]
    rank_color = assign_rank_colors([rank_name])[rank_name]
    draw_vertical_boxplots(
        ax,
        values,
        positions=positions,
        tick_labels=[str(batch_sizes[int(position) - 1]) for position in positions],
        colors=[rank_color] * len(values),
    )

    available_positions = {int(position) for position in positions}
    for position in range(1, len(batch_sizes) + 1):
        if position not in available_positions:
            ax.text(
                position,
                max_transition / 2.0,
                "log not available",
                color="0.55",
                ha="center",
                va="center",
                rotation=90,
            )

    tick_step = max(1, int(np.ceil(max_transition / TRANSITION_TICK_TARGET)))
    ax.set_xlim(0.5, len(batch_sizes) + 0.5)
    ax.set_xticks(range(1, len(batch_sizes) + 1), batch_sizes)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_ylim(0.5, max_transition + 0.5)
    ax.set_yticks(np.arange(tick_step, max_transition + 1, tick_step))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis="y", labelleft=show_y_ticks)
    ax.set_title(
        f"{panel_label} {SUBTREE_METHOD_LABEL}\n({rank_label(rank_name)})",
        color=rank_color,
        loc="center",
        pad=3.0,
    )
    style_quantitative_axis(ax)


def draw_figure(comparisons: Sequence[RankComparison]) -> Figure:
    """Draw all traversal-rule transition panels in one row."""
    rank_names = all_batch_rank_names(comparisons)
    if not comparisons or not rank_names:
        raise ValueError("at least one prefetch budget and traversal rule is required")
    max_transition = max_realized_transition(comparisons)

    figure, axes = plt.subplots(
        1,
        len(rank_names),
        figsize=(PANEL_WIDTH * len(rank_names), FIGURE_HEIGHT),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for column_index, rank_name in enumerate(rank_names):
        draw_transition_panel(
            axes[0, column_index],
            comparisons,
            rank_name,
            max_transition,
            panel_label=panel_letter(column_index),
            show_y_ticks=column_index == 0,
        )

    figure.subplots_adjust(
        left=LEFT_MARGIN,
        right=RIGHT_MARGIN,
        bottom=BOTTOM_MARGIN,
        top=TOP_MARGIN,
        wspace=COLUMN_SPACE,
    )
    figure.suptitle(comparison_figure_title(comparisons), y=0.995)
    figure.supxlabel(PREFETCH_BUDGET_AXIS_LABEL, y=0.025)
    figure.supylabel(
        r"\# realized MH transitions",
        x=LEFT_MARGIN - Y_LABEL_INSET_INCHES / figure.get_figwidth(),
    )
    return figure


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the realized-transition plotting CLI."""
    parser = argparse.ArgumentParser(
        description="Plot all-batch realized MH transitions by traversal rule."
    )
    parser.add_argument("--directory", type=Path, default=DEFAULT_DIRECTORY)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--csv-output", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    directory = args.directory.resolve()
    if not directory.is_dir():
        raise SystemExit(f"missing directory {directory}")
    try:
        comparisons = collect_rank_comparisons(directory)
        figure = draw_figure(comparisons)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from error

    # Named after the run it pools, so the stem waits on a parsed log.
    stem = dataset_model_prefix(comparisons[0].collected[0].log_path)
    output = args.output.resolve() if args.output else directory / f"{stem}{FIGURE_SUFFIX}"
    csv_output = (
        args.csv_output.resolve()
        if args.csv_output
        else directory / f"{stem}{CSV_SUFFIX}"
    )

    save_figure(figure, output, pad_inches=0.02)
    write_dict_rows(
        csv_output,
        [
            row
            for comparison in comparisons
            for row in comparison.transition_summary_rows()
        ],
    )
    print(f"wrote {output}")
    print(f"wrote {csv_output}")


if __name__ == "__main__":
    main()
