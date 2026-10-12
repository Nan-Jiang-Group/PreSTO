#!/usr/bin/env python3
"""Plot acceptance mass across prefetch sizes and traversal rules.

Run with:

    /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/draw/.venv/bin/python \
        /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/draw/draw_rank_proposal_certainty.py \
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
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MultipleLocator, NullLocator

from case_studies import paths
from case_studies.draw.common.figure_io import save_figure, write_dict_rows
from case_studies.draw.common.plot_helpers import panel_letter
from case_studies.draw.common.rank_plotting import (
    comparison_figure_title,
    style_quantitative_axis,
)
from case_studies.extract.analysis.rank_analysis import (
    RankComparison,
    all_batch_rank_names,
    collect_rank_comparisons,
    rank_observation,
)
from case_studies.extract.logs.proposal_log import (
    BUCKET_NAMES,
    EFFECTIVELY_ZERO_ACCEPT_PROB,
    EFFECTIVELY_ZERO_NATS,
    NEAR_CERTAIN_ACCEPT_PROB,
    NEAR_CERTAIN_NATS,
    RATIO_HIST_CLIP,
    TreeNode,
    parse_log,
)
from case_studies.extract.run_naming import (
    LOG_GLOB,
    dataset_model_prefix,
    prefetch_budget_token,
    rank_label,
)
from case_studies.plot_config import (
    SUBTREE_BUCKET_COLORS,
    SUBTREE_BUCKET_FILL_ALPHA,
    SUBTREE_BUCKET_HATCHES,
    SUBTREE_BUCKET_TEXT_COLORS,
    SUBTREE_METHOD_LABEL,
    apply_plot_style,
    scale_font_sizes,
)

apply_plot_style()

plt.rcParams["text.usetex"] = True

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
FIGURE_SUFFIX = ".all-rank-proposal-certainty.pdf"
CSV_SUFFIX = ".all-rank-proposal-certainty.csv"

GRID_PANEL_WIDTH = 2.40
GRID_PANEL_HEIGHT = 2.20
SINGLE_ROW_MIN_WIDTH = 8.0
SINGLE_ROW_HEIGHT = 4.20
SINGLE_ROW_TOP_MARGIN = 0.70
SINGLE_ROW_BOTTOM_MARGIN = 0.17
SINGLE_ROW_SUPTITLE_Y = 0.95
SINGLE_ROW_LEGEND_Y = 0.88
SINGLE_ROW_XLABEL_Y = 0.045
MULTI_ROW_SUPTITLE_Y = 0.98
TOP_MARGIN = 0.91
BOTTOM_MARGIN = 0.07
LEFT_MARGIN = 0.055
RIGHT_MARGIN = 0.995
ROW_SPACE = 0.30
COLUMN_SPACE = 0.13
TITLE_FONT_SIZE = 16
MULTI_ROW_PANEL_FONT_SIZE = float(plt.rcParams["axes.labelsize"])
LEGEND_FONT_SIZE = TITLE_FONT_SIZE
LEGEND_HANDLE_LENGTH = 1.0
LEGEND_HANDLE_HEIGHT = 0.8
CROWDED_TICK_SIZE = 0.85 * float(plt.rcParams["xtick.labelsize"])
ACCEPTANCE_XLABEL = r"MH acceptance ratio $A$"
ACCEPTANCE_XLABEL_Y = 0.035
ACCEPTANCE_YLABEL = "proposal share"
MASS_BIN_COUNTS_ASC = (2, 7, 2)
MASS_BUCKET_DISPLAY_WIDTHS = tuple(float(count) for count in MASS_BIN_COUNTS_ASC)
LOG_ACCEPTANCE_MIN = float(np.exp(-RATIO_HIST_CLIP))
MASS_BUCKET_EDGES_ASC = (
    LOG_ACCEPTANCE_MIN,
    EFFECTIVELY_ZERO_ACCEPT_PROB,
    NEAR_CERTAIN_ACCEPT_PROB,
    1.0,
)
BIN_WIDTH_FRACTION = 0.7
BAR_LINE_WIDTH = 0.55
SEPARATOR_LINE_WIDTH = 0.8
PERCENTAGE_ANNOTATION_Y = 0.95
PERCENTAGE_FONT_SIZE = plt.rcParams["axes.labelsize"]
Y_AXIS_MIN = 0.0
Y_LIMIT_HEADROOM = 1.2
Y_TICK_INTERVAL = 10.0

# Shared bucket colors, ordered from low to high acceptance probability.
MASS_BUCKET_FILL_COLORS_ASC = tuple(reversed(SUBTREE_BUCKET_COLORS))
MASS_BUCKET_TEXT_COLORS_ASC = tuple(reversed(SUBTREE_BUCKET_TEXT_COLORS))
MASS_BUCKET_HATCHES_ASC = tuple(reversed(SUBTREE_BUCKET_HATCHES))
BAR_EDGE_COLOR = "#FFFFFF"
GRID_COLOR = "#E9E9E9"
AXIS_COLOR = "#262626"
SEPARATOR_COLOR = "#525252"

DISPLAY_BUCKET_NAMES = {
    "effectively rejected": r"$\epsilon$-certain reject",
    "genuinely stochastic": r"$\epsilon$-uncertain",
    "effectively accepted": r"$\epsilon$-certain accept",
}
EDGE_BUCKET_NAMES = {
    "effectively rejected": "effectively not taken",
    "genuinely stochastic": "uncertain",
    "effectively accepted": "effectively taken",
}
DISPLAY_EDGE_BUCKET_NAMES = {
    "effectively rejected": r"$\epsilon$-certain not taken",
    "genuinely stochastic": r"$\epsilon$-uncertain",
    "effectively accepted": r"$\epsilon$-certain taken",
}


def percentage_tick_label(value: float, _position: float) -> str:
    """Format the linear proposal-share axis as whole percentages."""
    return f"{value:g}\\%"


def certainty_legend_handles(
    *, hatched: bool = False, all_edges: bool = False
) -> list[Patch]:
    """Build legend handles using solid fills or unfilled colored hatches."""
    labels = DISPLAY_EDGE_BUCKET_NAMES if all_edges else DISPLAY_BUCKET_NAMES
    return [
        Patch(
            facecolor="none" if hatched else fill_color,
            edgecolor=fill_color if hatched else "none",
            hatch=hatch if hatched else None,
            linewidth=BAR_LINE_WIDTH,
            alpha=SUBTREE_BUCKET_FILL_ALPHA,
            label=labels[bucket_name],
        )
        for bucket_name, fill_color, hatch in zip(
            reversed(BUCKET_NAMES),
            MASS_BUCKET_FILL_COLORS_ASC,
            MASS_BUCKET_HATCHES_ASC,
            strict=True,
        )
    ]


def log_acceptance_buckets(
    nodes: list[TreeNode],
    *,
    bin_counts: tuple[int, int, int] = MASS_BIN_COUNTS_ASC,
    all_edges: bool = False,
) -> list[tuple[str, np.ndarray, np.ndarray, float]]:
    """Bin acceptance probabilities, optionally including rejection edges.

    Args:
        nodes: Scored proposals, each representing one binary MH decision.
        bin_counts: Bin counts for the low, middle, and high probability buckets.
        all_edges: Count both A and 1-A once per proposal, including zero-mass edges.

    Returns:
        Bucket names, bin boundaries, counts, and shares of all counted observations.
        Exact zero probabilities are clipped only for display on the log axis.
    """
    if not nodes:
        raise ValueError("cannot compute certainty shares without scored proposals")
    log_probabilities = np.asarray([node.log_a for node in nodes], dtype=float)
    if all_edges:
        # expm1 preserves small rejection probabilities when A is close to one.
        with np.errstate(divide="ignore"):
            log_rejection = np.log(-np.expm1(log_probabilities))
        log_probabilities = np.concatenate((log_probabilities, log_rejection))
    masks = {
        "effectively rejected": log_probabilities <= EFFECTIVELY_ZERO_NATS,
        "genuinely stochastic": (
            (log_probabilities > EFFECTIVELY_ZERO_NATS)
            & (log_probabilities <= NEAR_CERTAIN_NATS)
        ),
        "effectively accepted": log_probabilities > NEAR_CERTAIN_NATS,
    }
    buckets: list[tuple[str, np.ndarray, np.ndarray, float]] = []
    for bucket_index, bucket_name in enumerate(reversed(BUCKET_NAMES)):
        left, right = MASS_BUCKET_EDGES_ASC[bucket_index : bucket_index + 2]
        bin_edges = np.geomspace(
            left,
            right,
            bin_counts[bucket_index] + 1,
        )
        values = np.exp(log_probabilities[masks[bucket_name]])
        counts, _ = np.histogram(np.clip(values, left, right), bins=bin_edges)
        buckets.append(
            (bucket_name, bin_edges, counts, len(values) / len(log_probabilities))
        )
    return buckets


def acceptance_tick_position(
    value: float,
    *,
    bucket_display_widths: tuple[float, float, float] = MASS_BUCKET_DISPLAY_WIDTHS,
) -> float:
    """Map an acceptance probability to its piecewise-log display position."""
    clipped_value = float(np.clip(value, LOG_ACCEPTANCE_MIN, 1.0))
    bucket_offset = 0.0
    for bucket_index, display_width in enumerate(bucket_display_widths):
        left, right = MASS_BUCKET_EDGES_ASC[bucket_index : bucket_index + 2]
        if clipped_value <= right or bucket_index == len(bucket_display_widths) - 1:
            log_fraction = (np.log(clipped_value) - np.log(left)) / (
                np.log(right) - np.log(left)
            )
            return bucket_offset + display_width * float(log_fraction)
        bucket_offset += display_width
    raise AssertionError("acceptance probability did not map to a certainty bucket")


def style_piecewise_log_acceptance_axis(
    ax: Axes,
    *,
    show_separators: bool = True,
    bucket_display_widths: tuple[float, float, float] = MASS_BUCKET_DISPLAY_WIDTHS,
) -> None:
    """Style the piecewise-log acceptance axis and linear percentage scale."""
    bucket_bounds = np.cumsum((0.0, *bucket_display_widths))
    ax.set_xlim(0.0, float(bucket_bounds[-1]))
    tick_values = (0.0, 0.01, 0.99, 1.0)
    ax.set_xticks(
        [
            acceptance_tick_position(
                value, bucket_display_widths=bucket_display_widths
            )
            for value in tick_values
        ],
        ["0", "0.01", "0.99", "1"],
        fontsize=CROWDED_TICK_SIZE,
    )
    major_ticks = ax.xaxis.get_major_ticks()
    major_ticks[0].label1.set_ha("left")
    major_ticks[2].label1.set_ha("center")
    major_ticks[-1].label1.set_ha("right")
    if show_separators:
        for boundary in bucket_bounds[1:-1]:
            ax.axvline(
                boundary,
                color=SEPARATOR_COLOR,
                ls=(0, (4, 3)),
                lw=SEPARATOR_LINE_WIDTH,
                zorder=4,
            )
    ax.set_yscale("linear")
    ax.yaxis.set_major_locator(MultipleLocator(Y_TICK_INTERVAL))
    ax.yaxis.set_major_formatter(FuncFormatter(percentage_tick_label))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_locator(NullLocator())
    ax.grid(axis="y", color=GRID_COLOR, lw=0.8)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(AXIS_COLOR)
    ax.tick_params(
        axis="both",
        which="both",
        top=False,
        right=False,
        colors=AXIS_COLOR,
    )


def plot_piecewise_log_acceptance_mass(
    ax: Axes,
    nodes: list[TreeNode],
    *,
    bin_counts: tuple[int, int, int] = MASS_BIN_COUNTS_ASC,
    bin_width_fraction: float = BIN_WIDTH_FRACTION,
    max_bar_display_width: float | None = None,
    inset_annotations: bool = False,
    show_separators: bool = True,
    bucket_display_widths: tuple[float, float, float] = MASS_BUCKET_DISPLAY_WIDTHS,
    hatched: bool = False,
    all_edges: bool = False,
) -> float:
    """Draw acceptance mass with solid fills or unfilled colored hatches."""
    buckets = log_acceptance_buckets(
        nodes, bin_counts=bin_counts, all_edges=all_edges
    )
    observation_count = len(nodes) * (2 if all_edges else 1)
    total_display_width = sum(bucket_display_widths)
    low_width, middle_width, high_width = bucket_display_widths
    annotation_positions = (
        (
            low_width / (2.0 * total_display_width),
            PERCENTAGE_ANNOTATION_Y,
            "center",
        ),
        (
            (low_width + middle_width / 2.0) / total_display_width,
            PERCENTAGE_ANNOTATION_Y,
            "center",
        ),
        (
            (low_width + middle_width + high_width / 2.0) / total_display_width,
            PERCENTAGE_ANNOTATION_Y,
            "center",
        ),
    )
    if inset_annotations:
        annotation_positions = (
            (0.02, PERCENTAGE_ANNOTATION_Y, "left"),
            annotation_positions[1],
            (0.98, PERCENTAGE_ANNOTATION_Y, "right"),
        )
    bucket_offset = 0.0
    maximum_bin_share = 0.0
    styled_buckets = zip(
        buckets,
        MASS_BUCKET_FILL_COLORS_ASC,
        MASS_BUCKET_TEXT_COLORS_ASC,
        strict=True,
    )
    for bucket_index, ((_, _edges, counts, share), fill_color, text_color) in enumerate(
        styled_buckets
    ):
        bin_shares = 100.0 * counts / observation_count
        maximum_bin_share = max(maximum_bin_share, float(bin_shares.max()))
        bin_width = bucket_display_widths[bucket_index] / len(counts)
        bar_width = bin_width_fraction * bin_width
        if max_bar_display_width is not None:
            bar_width = min(bar_width, max_bar_display_width)
        bin_lefts = bucket_offset + bin_width * np.arange(len(counts), dtype=float)
        ax.bar(
            bin_lefts + (bin_width - bar_width) / 2.0,
            bin_shares,
            width=bar_width,
            align="edge",
            facecolor="none" if hatched else fill_color,
            edgecolor=fill_color if hatched else BAR_EDGE_COLOR,
            hatch=MASS_BUCKET_HATCHES_ASC[bucket_index] if hatched else None,
            alpha=SUBTREE_BUCKET_FILL_ALPHA,
            linewidth=BAR_LINE_WIDTH,
            zorder=2,
        )

        annotation_x, annotation_y, alignment = annotation_positions[bucket_index]
        ax.text(
            annotation_x,
            annotation_y,
            rf"{share * 100:.0f}\%",
            transform=ax.transAxes,
            color=text_color,
            fontsize=PERCENTAGE_FONT_SIZE,
            fontweight="bold",
            ha=alignment,
            va="top",
            zorder=5,
        )
        bucket_offset += bucket_display_widths[bucket_index]
    style_piecewise_log_acceptance_axis(
        ax,
        show_separators=show_separators,
        bucket_display_widths=bucket_display_widths,
    )
    return maximum_bin_share


def draw_acceptance_mass_panel(
    ax: Axes,
    comparison: RankComparison,
    rank_name: str,
) -> float:
    """Draw one panel and return its largest bin share in percent."""
    observation = rank_observation(comparison, rank_name)
    if observation is None:
        maximum_bin_share = 0.0
        style_piecewise_log_acceptance_axis(ax)
        ax.text(
            0.5,
            0.5,
            "log not available",
            transform=ax.transAxes,
            color="0.55",
            ha="center",
            va="center",
        )
    else:
        entry, _, _ = observation
        nodes = parse_log(entry.log_path).nodes
        maximum_bin_share = plot_piecewise_log_acceptance_mass(ax, nodes)

    ax.set_xlabel("")
    ax.set_ylabel("")
    style_quantitative_axis(ax)
    return maximum_bin_share


def draw_figure(comparisons: Sequence[RankComparison]) -> Figure:
    """Draw one acceptance-mass panel per prefetch-size/rule pair."""
    rank_names = all_batch_rank_names(comparisons)
    if not comparisons or not rank_names:
        raise ValueError("at least one prefetch budget and traversal rule is required")

    rows = len(comparisons)
    columns = len(rank_names)
    single_row = rows == 1
    panel_font_size = (
        TITLE_FONT_SIZE if single_row else MULTI_ROW_PANEL_FONT_SIZE
    )
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(
            max(GRID_PANEL_WIDTH * columns, SINGLE_ROW_MIN_WIDTH)
            if single_row
            else GRID_PANEL_WIDTH * columns,
            SINGLE_ROW_HEIGHT if single_row else GRID_PANEL_HEIGHT * rows,
        ),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    maximum_bin_share = 0.0
    for row_index, comparison in enumerate(comparisons):
        for column_index, rank_name in enumerate(rank_names):
            ax = axes[row_index, column_index]
            maximum_bin_share = max(
                maximum_bin_share,
                draw_acceptance_mass_panel(ax, comparison, rank_name),
            )
            ax.tick_params(axis="x", labelbottom=True)
            if column_index > 0:
                ax.tick_params(axis="y", left=False)
            if row_index == 0:
                ax.set_title(
                    f"{panel_letter(column_index)} {rank_label(rank_name)}",
                    color="black",
                    fontsize=panel_font_size,
                    pad=4.0,
                )
            if column_index == 0:
                ax.set_ylabel(
                    rf"{ACCEPTANCE_YLABEL} "
                    rf"($N_{{\mathrm{{pf}}}}={comparison.batch_size}$)",
                    fontsize=panel_font_size,
                )

    y_axis_max = max(10.0, Y_LIMIT_HEADROOM * maximum_bin_share)
    axes[0, 0].set_ylim(Y_AXIS_MIN, y_axis_max)

    figure.subplots_adjust(
        left=LEFT_MARGIN,
        right=RIGHT_MARGIN,
        bottom=SINGLE_ROW_BOTTOM_MARGIN if single_row else BOTTOM_MARGIN,
        top=SINGLE_ROW_TOP_MARGIN if single_row else TOP_MARGIN,
        hspace=ROW_SPACE,
        wspace=COLUMN_SPACE,
    )
    figure.suptitle(
        f"{comparison_figure_title(comparisons)}, "
        f"Method: {SUBTREE_METHOD_LABEL}",
        fontsize=TITLE_FONT_SIZE,
        y=SINGLE_ROW_SUPTITLE_Y if single_row else MULTI_ROW_SUPTITLE_Y,
    )
    figure.supxlabel(
        ACCEPTANCE_XLABEL,
        y=SINGLE_ROW_XLABEL_Y if single_row else ACCEPTANCE_XLABEL_Y,
        fontsize=panel_font_size,
    )
    figure.legend(
        handles=certainty_legend_handles(),
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, SINGLE_ROW_LEGEND_Y if single_row else 0.965),
        borderaxespad=0.0,
        ncol=3,
        fontsize=LEGEND_FONT_SIZE,
        handlelength=LEGEND_HANDLE_LENGTH,
        handleheight=LEGEND_HANDLE_HEIGHT,
        handletextpad=0.3,
        columnspacing=0.55,
    )
    return figure


def acceptance_mass_rows(
    comparisons: Sequence[RankComparison],
) -> list[dict[str, object]]:
    """Return one source-data row per acceptance-mass histogram bin."""
    rows: list[dict[str, object]] = []
    for comparison in comparisons:
        for entry in comparison.collected:
            nodes = parse_log(entry.log_path).nodes
            for bucket_name, bin_edges, counts, bucket_share in (
                log_acceptance_buckets(nodes)
            ):
                for local_index, count_value in enumerate(counts):
                    count = int(count_value)
                    rows.append(
                        {
                            "prefetch_budget": comparison.batch_size,
                            "rank_fn": entry.rank_fn,
                            "proposal_nodes": len(nodes),
                            "acceptance_bucket": bucket_name,
                            "bucket_share": f"{bucket_share:.6f}",
                            "bin_in_bucket": local_index,
                            "acceptance_left": f"{bin_edges[local_index]:.8e}",
                            "acceptance_right": f"{bin_edges[local_index + 1]:.8e}",
                            "proposal_count": count,
                            "proposal_share": f"{count / len(nodes):.6f}",
                        }
                    )
    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the proposal-certainty plotting CLI."""
    parser = argparse.ArgumentParser(
        description="Plot all-batch acceptance mass by traversal rule."
    )
    parser.add_argument("--directory", type=Path, default=DEFAULT_DIRECTORY)
    parser.add_argument(
        "--log-glob",
        default=LOG_GLOB,
        help="glob selecting compatible direct-child logs inside --directory",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--csv-output", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    directory = args.directory.resolve()
    if not directory.is_dir():
        raise SystemExit(f"missing directory {directory}")
    try:
        # This artifact is explicitly an all-rank comparison. Other aggregate figures may omit expensive traversal
        # rules, but proposal certainty should retain the complete panel set used by the reference figure.
        comparisons = collect_rank_comparisons(
            directory,
            include_excluded_ranks=True,
            log_glob=args.log_glob,
        )
        figure = draw_figure(comparisons)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from error

    # Named after the run it pools, so the stem waits on a parsed log.
    stem = dataset_model_prefix(comparisons[0].collected[0].log_path)
    if len(comparisons) == 1:
        stem = f"{stem}.{prefetch_budget_token(comparisons[0].batch_size)}"
    output = args.output.resolve() if args.output else directory / f"{stem}{FIGURE_SUFFIX}"
    csv_output = (
        args.csv_output.resolve()
        if args.csv_output
        else directory / f"{stem}{CSV_SUFFIX}"
    )

    save_figure(figure, output, pad_inches=0.02)
    write_dict_rows(
        csv_output,
        acceptance_mass_rows(comparisons),
    )
    print(f"wrote {output}")
    print(f"wrote {csv_output}")


if __name__ == "__main__":
    main()
