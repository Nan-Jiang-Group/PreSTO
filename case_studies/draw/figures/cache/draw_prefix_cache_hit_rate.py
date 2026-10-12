#!/usr/bin/env python3
"""Draw the prefix-cache token hit rate alone, without the other cache panels.

Run with:

    /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/.venv/bin/python \
        /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/draw/draw_prefix_cache_hit_rate.py \
        --baseline-resource-json /absolute/path/to/run.resources.json

This is panel (b) of draw_combined_cache_metrics.py as a standalone figure, and
reads the same measurements through the same loaders: 100 * prefix_cache_hit_rate
from each run's resources.json, checked against its queried and hit token counts.
One line per traversal rank runs across prefetch budgets, and the baseline run,
when given, is the dashed horizontal line.

Without --baseline-resource-json the values come from a resource-usage CSV
(--resource-csv) and no baseline line is drawn. The method labels follow the
sampler the baseline summary records, so EntropyCut runs are named as such
without configuring the renderer; --baseline-label and --subtree-label override
them. The plotted values and their source paths are written to a CSV beside the
PDF.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, MultipleLocator, NullLocator

from case_studies import paths
from case_studies.draw.common.figure_io import save_figure, write_dict_rows
from case_studies.draw.common.rank_plotting import PREFETCH_BUDGET_AXIS_LABEL, style_quantitative_axis
from case_studies.draw.figures.cache import draw_combined_cache_metrics as combined
from case_studies.draw.figures.cache.draw_combined_cache_metrics import (
    DEFAULT_RESOURCE_CSV,
    FIGURE_FONT_SIZE,
    HIT_RATE_RANGE_STEP,
    LINE_WIDTH,
    MARKER_EDGE_WIDTH,
    MARKER_SIZE,
    RANK_MARKERS,
    banded_limits,
    draw_lines,
    numeric_series,
    rank_offsets,
    read_csv,
)
from case_studies.extract.run_naming import RANK_ORDER, dataset_of, model_of, rank_label, sort_rank_names
from case_studies.plot_config import DARK_GRAY, PURPLE
from case_studies.plot_config import TRANSITION_RANK_COLORS as RANK_COLORS

FIGURE_SUFFIX = ".prefix-cache-hit-rate.pdf"
HIT_RATE_KEY = "prefix_cache_hit_rate_percent"
Y_LABEL = r"prefix-cache token hit rate (\%)"
FIGURE_SIZE = (5.8, 3.7)
LEGEND_COLUMNS = 3
LEGEND_ROW_HEIGHT = 0.34
DEFAULT_OUTPUT = (
    paths.REPO_ROOT
    / "output/pdf/math500-qwen-prefix-cache-hit-rate.pdf"
)
# The samplers whose runs are labeled after the entropy cut rather than PowerMH.
ENTROPYCUT_METHODS = frozenset({"entropycut_power_mh"})
ENTROPYCUT_LABELS = ("EntropyCut", "PreSTO-EntropyCut")
# Ranks the shared palette has no entry for yet, given the color the EntropyCut
# sweep draws them in (draw_entropycut_analysis.configure_renderers), so this
# figure matches the ones beside it.
SWEEP_RANK_COLORS = {"bfs_accept_first": PURPLE}


def dataset_label_of(name: str) -> str:
    """Name the benchmark the way the other cache figures title it."""
    dataset = dataset_of(name)
    return "LCB V6" if dataset == "lcb_v6" else dataset


def method_labels(baseline_resource_json: Path | None) -> tuple[str, str]:
    """Name the methods after the sampler the baseline run recorded."""
    if baseline_resource_json is not None:
        method = json.loads(baseline_resource_json.read_text(encoding="utf-8")).get("method")
        if method in ENTROPYCUT_METHODS:
            return ENTROPYCUT_LABELS
    return combined.POWER_METHOD_LABEL, combined.SUBTREE_METHOD_LABEL


def rank_colors(rank_names: list[str]) -> None:
    """Give every plotted rank the color its sibling figures already use."""
    for rank_name in rank_names:
        if rank_name not in RANK_COLORS and rank_name in SWEEP_RANK_COLORS:
            RANK_COLORS[rank_name] = SWEEP_RANK_COLORS[rank_name]
    unknown = sorted(set(rank_names) - RANK_COLORS.keys())
    if unknown:
        raise ValueError("No shared rank color for: " + ", ".join(unknown))


def legend_handles(rank_names: list[str], baseline_label: str | None) -> list[Line2D]:
    """Name the traversal ranks alone; the title carries the method they ran under."""
    handles = [
        Line2D(
            [0], [0], color=RANK_COLORS[rank_name], linewidth=LINE_WIDTH,
            marker=RANK_MARKERS[index % len(RANK_MARKERS)], markersize=MARKER_SIZE,
            markerfacecolor="white", markeredgewidth=MARKER_EDGE_WIDTH,
            label=rank_label(rank_name),
        )
        for index, rank_name in enumerate(rank_names)
    ]
    if baseline_label is not None:
        handles.append(
            Line2D(
                [0], [0], color=DARK_GRAY, linewidth=LINE_WIDTH,
                linestyle="--", label=baseline_label,
            )
        )
    return handles


def style_axis(
    ax: Axes,
    *,
    budgets: list[int],
    y_limits: tuple[float, float],
    tick_step: float | None,
) -> None:
    """Apply the quantitative styling the cache panels share."""
    ax.set_ylabel(Y_LABEL)
    ax.set_xlabel(PREFETCH_BUDGET_AXIS_LABEL)
    ax.set_xticks(budgets)
    ax.set_xlim(min(budgets) - 0.75, max(budgets) + 0.75)
    ax.set_ylim(*y_limits)
    # A hit-rate range set by the runs themselves can be a few tenths of a
    # percent wide or tens of percent wide, so pick the step from the data
    # unless the caller states one.
    ax.yaxis.set_major_locator(
        MaxNLocator(nbins=5, steps=(1, 2, 2.5, 5, 10), min_n_ticks=3)
        if tick_step is None else MultipleLocator(tick_step)
    )
    ax.grid(False)
    ax.set_axisbelow(True)
    style_quantitative_axis(ax)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())


def draw_figure(
    output: Path,
    *,
    resource_csv: Path = DEFAULT_RESOURCE_CSV,
    baseline_resource_json: Path | None = None,
    dataset_label: str = "MATH500",
    model_label: str = "Qwen2.5-7B",
    hit_rate_tick_step: float | None = None,
    include_excluded_ranks: bool = False,
    baseline_label: str | None = None,
    subtree_label: str | None = None,
) -> Path:
    """Render prefix-cache token hit rate against prefetch budget."""
    if hit_rate_tick_step is not None and not hit_rate_tick_step > 0:
        raise ValueError("hit_rate_tick_step must be positive")
    # The shared loader and legend read these method names from their own
    # module when they run, so set them before either is called.
    recorded_labels = method_labels(baseline_resource_json)
    combined.POWER_METHOD_LABEL = baseline_label or recorded_labels[0]
    combined.SUBTREE_METHOD_LABEL = subtree_label or recorded_labels[1]
    baseline_row = None
    if baseline_resource_json is None:
        resource_rows = read_csv(resource_csv)
    else:
        resource_rows, baseline_row = combined.read_resource_comparison(
            baseline_resource_json, include_excluded_ranks=include_excluded_ranks
        )
    source_rows = resource_rows if baseline_row is None else [baseline_row, *resource_rows]
    series = numeric_series(resource_rows, rank_key="rank_fn", value_key=HIT_RATE_KEY)
    rank_names = (
        sorted(
            {rank for rank, _ in series},
            key=lambda rank: (RANK_ORDER.get(rank, len(RANK_ORDER)), rank),
        )
        if include_excluded_ranks else sort_rank_names(rank for rank, _ in series)
    )
    budgets = sorted({budget for _, budget in series})
    rank_colors(rank_names)
    handles = legend_handles(
        rank_names,
        combined.POWER_METHOD_LABEL if baseline_row is not None else None,
    )
    legend_columns = min(LEGEND_COLUMNS, len(handles))
    legend_rows = -(-len(handles) // legend_columns)

    figure = plt.figure(
        figsize=(FIGURE_SIZE[0], FIGURE_SIZE[1] + LEGEND_ROW_HEIGHT * (legend_rows - 1))
    )
    ax = figure.add_subplot(1, 1, 1)
    values = draw_lines(ax, series, rank_names, rank_offsets(rank_names))
    if baseline_row is not None:
        baseline_value = float(baseline_row[HIT_RATE_KEY])
        ax.axhline(
            baseline_value, color=DARK_GRAY, linestyle="--",
            linewidth=LINE_WIDTH, zorder=2,
        )
        values.append(baseline_value)
    style_axis(
        ax,
        budgets=budgets,
        # One panel has room to keep the extreme runs off the frame, which the
        # narrower combined panels leave to their shared tick step.
        y_limits=banded_limits(
            values,
            padding=max(HIT_RATE_RANGE_STEP, 0.06 * (max(values) - min(values))),
        ),
        tick_step=hit_rate_tick_step,
    )

    title = (
        f"dataset: {dataset_label}, base LLM: {model_label}, "
        f"Method: {combined.SUBTREE_METHOD_LABEL}"
    )
    figure.suptitle(title, y=0.985)
    legend = figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        fontsize=FIGURE_FONT_SIZE,
        ncol=legend_columns,
        frameon=False,
        handlelength=1.7,
        handletextpad=0.4,
        columnspacing=0.9,
    )
    # The legend height depends on how many ranks a sweep recorded, so measure
    # it before placing the axes rather than reserving a fixed top margin.
    figure.canvas.draw()
    legend_bottom = legend.get_window_extent().transformed(
        figure.transFigure.inverted()
    ).y0
    figure.subplots_adjust(
        left=0.155, right=0.985, bottom=0.145, top=min(0.86, legend_bottom - 0.025)
    )
    result = save_figure(figure, output, pad_inches=0.02)
    write_dict_rows(output.with_suffix(".csv"), source_rows)
    return result


def build_parser() -> argparse.ArgumentParser:
    """Build the standalone hit-rate CLI."""
    parser = argparse.ArgumentParser(
        description="Draw the prefix-cache token hit rate as a single-panel figure."
    )
    parser.add_argument("--resource-csv", type=Path, default=DEFAULT_RESOURCE_CSV)
    parser.add_argument(
        "--baseline-resource-json", type=Path,
        help="Load this baseline summary and compatible sibling PreSTO resources instead of a CSV.",
    )
    parser.add_argument("--dataset-label")
    parser.add_argument("--model-label")
    parser.add_argument(
        "--hit-rate-tick-step", type=float,
        help="Fix the y-axis step; by default it is chosen from the plotted range.",
    )
    parser.add_argument(
        "--include-excluded-ranks", action="store_true",
        help="Include available ranks normally omitted by the shared comparison policy.",
    )
    parser.add_argument("--baseline-label", help="Legend name for the baseline run.")
    parser.add_argument("--subtree-label", help="Legend name for the prefetching runs.")
    parser.add_argument(
        "--output", type=Path,
        help="Defaults beside a baseline summary as <run>" + FIGURE_SUFFIX + ".",
    )
    return parser


def main() -> None:
    """Render the standalone prefix-cache hit-rate figure."""
    args = build_parser().parse_args()
    baseline = (
        args.baseline_resource_json.resolve() if args.baseline_resource_json else None
    )
    if args.output is not None:
        output = args.output.resolve()
    elif baseline is not None:
        output = baseline.with_name(
            baseline.name.removesuffix(".resources.json") + FIGURE_SUFFIX
        )
    else:
        output = DEFAULT_OUTPUT
    name = baseline.name if baseline is not None else args.resource_csv.name
    output = draw_figure(
        output,
        resource_csv=args.resource_csv.resolve(),
        baseline_resource_json=baseline,
        dataset_label=args.dataset_label or dataset_label_of(name),
        model_label=args.model_label or model_of(name),
        hit_rate_tick_step=args.hit_rate_tick_step,
        include_excluded_ranks=args.include_excluded_ranks,
        baseline_label=args.baseline_label,
        subtree_label=args.subtree_label,
    )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
