#!/usr/bin/env python3
"""Compare one PowerMH baseline run against one SubTreeMH run, side by side.

Both panels plot realized MH transitions against cumulative logged MH-step time and share both axes, so the horizontal
distance between the two mean curves at a given height is the wall-clock cost of reaching that many transitions each
way.

The two methods reach that height differently, which is what the shared axes make comparable: PowerMH buys exactly one
transition per model call, while SubTreeMH buys a variable number from each prefetched subtree.

Run with:

    src/.venv/bin/python \
        case_studies/draw/draw_power_mh_vs_subtree_time.py \
        --power-mh-log path/to/....powerMH.vllm.log \
        --subtree-log path/to/....subtreePrefetch.vllm.log \
        --include-incomplete
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from case_studies.draw.common.figure_io import save_figure, write_dict_rows
from case_studies.draw.common.plot_helpers import panel_letter
from case_studies.draw.common.rank_plotting import (
    MEAN_LINE_WIDTH,
    RAW_TRACE_ALPHA,
    RAW_TRACE_COLOR,
    RAW_TRACE_LINE_WIDTH,
    SD_BAND_ALPHA,
    draw_cumulative_time_panel,
)
from case_studies.extract.logs.model_call_traces import (
    AnalysisRows,
    Row,
    build_power_mh_time_rows,
    build_power_mh_time_summary_rows,
    experiment_figure_title,
    is_power_mh_log,
    power_mh_trace_times,
    row_trace_key,
    subtree_mh_trace_times,
)
from case_studies.extract.run_naming import (
    canonical_rank,
    config_output_prefix,
    latex_text,
    output_path,
    prefetch_budget_of,
    rank_label,
    rank_of,
)
from case_studies.plot_config import (
    BLUE as POWER_COLOR,
)
from case_studies.plot_config import (
    DARK_GRAY,
    POWER_METHOD_LABEL,
    SUBTREE_METHOD_LABEL,
    apply_plot_style,
)
from case_studies.plot_config import (
    TRANSITION_RANK_COLORS as RANK_COLORS,
)

apply_plot_style()

OUTPUT_SUFFIX = ".power-mh-vs-subtree-time"
FIGURE_SUFFIX = f"{OUTPUT_SUFFIX}.pdf"
# One comparison, so one pair of source tables; the method column separates them.
DATA_SUFFIX = f"{OUTPUT_SUFFIX}.csv"
SUMMARY_SUFFIX = f"{OUTPUT_SUFFIX}.summary.csv"

DEFAULT_TRACES_PER_RANK = 10
PANEL_SIZE = (4.3, 2.6)
PANEL_LABEL_POSITION = (-0.02, 1.01)
PANEL_TITLE_PAD = 9.0
X_LIMIT_HEADROOM = 1.02
TRACE_NOTE_POSITION = (0.03, 0.97)
TRACE_NOTE_FONT_SIZE = 9.0
# Both panels encode the same three things, so one legend outside the right-hand panel serves the figure without
# repeating itself or covering any trajectory. The caption and the legend share that column and are both hung from the
# top of the axes, so shortening the figure cannot walk them into each other the way anchoring one to the middle did.
SIDE_COLUMN_X = 1.02
SIDE_FONT_SIZE = 12.0
CAPTION_ANCHOR = (SIDE_COLUMN_X, 1.0)
CAPTION_LINE_SPACING = 1.4
LEGEND_ANCHOR = (SIDE_COLUMN_X, 0.62)
LEGEND_OPTIONS = {
    "loc": "upper left",
    "alignment": "left",
    "frameon": False,
    "fontsize": SIDE_FONT_SIZE,
    "handlelength": 1.6,
    "handletextpad": 0.45,
    "labelspacing": 0.5,
    "borderaxespad": 0.0,
}


def shared_limits(
    *trajectory_groups: Sequence[Sequence[float]],
) -> tuple[float, int]:
    """Return x and y limits covering every run, so the panels share a scale.

    Taken from both runs rather than from whichever is drawn last: the panels are a comparison only if the same distance
    means the same thing in each.
    """
    all_trajectories = [values for group in trajectory_groups for values in group]
    return (
        max(values[-1] for values in all_trajectories) * X_LIMIT_HEADROOM,
        max(len(values) for values in all_trajectories),
    )


def draw_panel(
    ax: Axes,
    trajectories: Sequence[Sequence[float]],
    summary: Sequence[Row],
    *,
    color: str,
    title: str,
    label: str,
    show_y_label: bool,
    note: str | None = None,
) -> None:
    """Draw one method's panel: the shared time panel, named and lettered."""
    draw_cumulative_time_panel(
        ax, trajectories, summary, color=color, show_y_label=show_y_label
    )
    ax.set_title(title, pad=PANEL_TITLE_PAD)
    ax.text(
        *PANEL_LABEL_POSITION,
        label,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
    )
    if note is not None:
        ax.text(
            *TRACE_NOTE_POSITION,
            note,
            transform=ax.transAxes,
            color=DARK_GRAY,
            fontsize=TRACE_NOTE_FONT_SIZE,
            ha="left",
            va="top",
        )


def incomplete_trace_note(analysis: AnalysisRows) -> str | None:
    """Describe included open traces so the panel cannot imply completion."""
    incomplete_keys = {
        row_trace_key(row)
        for row in analysis.empirical_time
        if not bool(row["trace_complete"])
    }
    if not incomplete_keys:
        return None

    observed_iterations = max(
        int(row["mh_iteration"])
        for row in analysis.empirical_time
        if row_trace_key(row) in incomplete_keys
    )
    configured_totals = {
        int(row["total_mh_steps"])
        for row in analysis.events
        if row_trace_key(row) in incomplete_keys
    }
    observed_label = str(observed_iterations)
    if len(configured_totals) == 1:
        observed_label += f"/{configured_totals.pop()}"
    trace_label = "trace" if len(incomplete_keys) == 1 else "traces"
    return (
        f"partial log: {len(incomplete_keys)} {trace_label}; "
        f"up to {observed_label} MH iterations"
    )


def add_encoding_legend(ax: Axes, color: str) -> None:
    """Name what a panel draws, placed clear of the axes on its right.

    The swatches take the panel's own colour, but the encoding is the one both panels use, so the labels stay
    method-neutral.
    """
    ax.legend(
        handles=[
            Line2D([0], [0], color=color, linewidth=MEAN_LINE_WIDTH, label="mean"),
            Patch(
                facecolor=color,
                alpha=SD_BAND_ALPHA,
                linewidth=0,
                label=r"mean $\pm$ 1 SD",
            ),
            Line2D(
                [0],
                [0],
                color=RAW_TRACE_COLOR,
                linewidth=RAW_TRACE_LINE_WIDTH,
                alpha=RAW_TRACE_ALPHA,
                label="per-sample trace",
            ),
        ],
        bbox_to_anchor=LEGEND_ANCHOR,
        **LEGEND_OPTIONS,
    )


def add_run_caption(ax: Axes, subtree_log: Path) -> None:
    """Name the dataset and base model above the legend.

    Read from the subtree log rather than the baseline one: the PowerMH launcher does not echo the config block these
    come from, and the two runs are on the same model anyway, which is what makes them comparable.
    """
    title = experiment_figure_title(subtree_log)
    dataset_name, separator, base_llm_name = title.partition(", ")
    if not separator:
        raise ValueError(f"Figure title must use '<dataset>, <base LLM>': {title!r}")
    ax.text(
        *CAPTION_ANCHOR,
        f"dataset: {latex_text(dataset_name)}\n"
        f"base LLM: {latex_text(base_llm_name)}",
        transform=ax.transAxes,
        fontsize=SIDE_FONT_SIZE,
        linespacing=CAPTION_LINE_SPACING,
        ha="left",
        va="top",
    )


def subtree_panel_title(subtree_log: Path) -> str:
    """Name the subtree run by the two settings that distinguish it."""
    budget = prefetch_budget_of(subtree_log)
    rank_name = canonical_rank(rank_of(subtree_log))
    return (
        rf"{SUBTREE_METHOD_LABEL}, $N_{{\mathrm{{pf}}}} = {budget}$, "
        f"{rank_label(rank_name)}"
    )


def draw_figure(
    power_trajectories: Sequence[Sequence[float]],
    power_summary: Sequence[Row],
    subtree_trajectory_values: Sequence[Sequence[float]],
    subtree_summary: Sequence[Row],
    subtree_log: Path,
    subtree_note: str | None,
) -> plt.Figure:
    """Render the two methods as one left-and-right figure on shared axes."""
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(PANEL_SIZE[0] * 2, PANEL_SIZE[1]),
        sharex=True,
        sharey=True,
    )
    draw_panel(
        axes[0],
        power_trajectories,
        power_summary,
        color=POWER_COLOR,
        title=POWER_METHOD_LABEL,
        label=panel_letter(0),
        show_y_label=True,
    )
    subtree_color = RANK_COLORS[canonical_rank(rank_of(subtree_log))]
    draw_panel(
        axes[1],
        subtree_trajectory_values,
        subtree_summary,
        color=subtree_color,
        title=subtree_panel_title(subtree_log),
        label=panel_letter(1),
        show_y_label=False,
        note=subtree_note,
    )
    add_encoding_legend(axes[1], subtree_color)
    add_run_caption(axes[1], subtree_log)

    x_limit, y_limit = shared_limits(power_trajectories, subtree_trajectory_values)
    axes[0].set_xlim(0, x_limit)
    axes[0].set_ylim(0, y_limit)
    figure.tight_layout()
    return figure


def summary_rows_with_method(
    rows: Sequence[Row],
    method: str,
    *,
    includes_incomplete: bool,
) -> list[Row]:
    """Tag summary rows with the method they came from for the source CSV."""
    return [
        {
            "method": method,
            "includes_incomplete": includes_incomplete,
            **row,
        }
        for row in rows
    ]


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the PowerMH-versus-SubTreeMH timing CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare cumulative logged MH-step time for one PowerMH baseline "
            "run and one SubTreeMH run."
        )
    )
    parser.add_argument("--power-mh-log", type=Path, required=True)
    parser.add_argument("--subtree-log", type=Path, required=True)
    parser.add_argument("--traces-per-rank", type=int, default=DEFAULT_TRACES_PER_RANK)
    parser.add_argument(
        "--include-incomplete",
        action="store_true",
        help=(
            "Include the observed MH-step prefix of a subtree trace interrupted "
            "before completion; the figure and source data mark it as partial."
        ),
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=None,
        help="Output prefix; defaults to the subtree run's configuration.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    power_log = args.power_mh_log.resolve()
    subtree_log = args.subtree_log.resolve()

    if not is_power_mh_log(power_log):
        raise SystemExit(f"--power-mh-log is not a PowerMH log: {power_log}")
    if is_power_mh_log(subtree_log):
        raise SystemExit(f"--subtree-log is not a subtree log: {subtree_log}")

    try:
        power_trajectories = power_mh_trace_times(power_log)
        power_summary = build_power_mh_time_summary_rows(power_trajectories)
        subtree_trajectories, subtree_analysis = subtree_mh_trace_times(
            subtree_log,
            args.traces_per_rank,
            tuple(RANK_COLORS),
            include_incomplete=args.include_incomplete,
        )
        subtree_values = [cumulative for _, cumulative in subtree_trajectories]
        subtree_rows = subtree_analysis.empirical_time
        subtree_summary = subtree_analysis.empirical_time_summary
        subtree_note = incomplete_trace_note(subtree_analysis)
        figure = draw_figure(
            [cumulative for _, cumulative in power_trajectories],
            power_summary,
            subtree_values,
            subtree_summary,
            subtree_log,
            subtree_note,
        )
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error

    output_prefix = args.output_prefix or config_output_prefix(subtree_log)
    outputs = [
        save_figure(
            figure, output_path(output_prefix, FIGURE_SUFFIX), pad_inches=0.02
        ),
        write_dict_rows(
            output_path(output_prefix, DATA_SUFFIX),
            [
                {
                    "method": POWER_METHOD_LABEL,
                    "trace_complete": True,
                    **row,
                }
                for row in build_power_mh_time_rows(power_log, power_trajectories)
            ]
            + [
                {
                    "method": SUBTREE_METHOD_LABEL,
                    "source_log": str(row["source_log"]),
                    "trace_id": str(row["trace_id"]),
                    "trace_complete": bool(row["trace_complete"]),
                    "mh_iteration": int(row["mh_iteration"]),
                    "cumulative_empirical_seconds": float(
                        row["cumulative_empirical_seconds"]
                    ),
                }
                for row in subtree_rows
            ],
        ),
        write_dict_rows(
            output_path(output_prefix, SUMMARY_SUFFIX),
            summary_rows_with_method(
                power_summary,
                POWER_METHOD_LABEL,
                includes_incomplete=False,
            )
            + summary_rows_with_method(
                subtree_summary,
                SUBTREE_METHOD_LABEL,
                includes_incomplete=subtree_note is not None,
            ),
            fieldnames=(
                "method",
                "includes_incomplete",
                "rank",
                "mh_iteration",
                "trace_count",
                "mean_cumulative_seconds",
                "std_cumulative_seconds",
            ),
        ),
    ]
    for path in outputs:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
