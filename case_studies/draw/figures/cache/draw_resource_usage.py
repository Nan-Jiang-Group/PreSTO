#!/usr/bin/env python3
"""Compare run-level metrics from ``*.resources.json`` files.

The script groups compatible resource summaries, writes a source-data CSV, and
draws peak KV-cache occupancy and prefix-cache hit rate:

    case_studies/draw/.venv/bin/python \
        case_studies/draw/draw_resource_usage.py \
        --directory /absolute/path/to/vllm/results
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from case_studies import paths
from case_studies.draw.common.figure_io import save_figure, write_dict_rows
from case_studies.draw.common.plot_helpers import panel_letter
from case_studies.draw.common.rank_plotting import (
    PREFETCH_BUDGET_AXIS_LABEL,
    add_panel_label,
    style_quantitative_axis,
)
from case_studies.extract.logs.resource_summary import (
    ResourceGroup,
    ResourceSummary,
    discover_resource_groups,
)
from case_studies.extract.run_naming import output_path, rank_label, sort_rank_names
from case_studies.plot_config import (
    DARK_GRAY,
    GRAY,
    SUBTREE_METHOD_LABEL,
    apply_plot_style,
)
from case_studies.plot_config import (
    TRANSITION_RANK_COLORS as RANK_COLORS,
)

apply_plot_style()

DEFAULT_DIRECTORY = paths.LOGS_DIR / "math500"
FIGURE_SIZE = (8.4, 5.0)
LINE_WIDTH = 1.35
MARKER_SIZE = 5.2
MARKER_EDGE_WIDTH = 1.0
PANEL_LABEL_POSITION = (-0.15, 1.02)
RANK_MARKERS = ("o", "s", "^", "D", "v", "P")
X_DODGE_WIDTH = 0.58

MetricAccessor = Callable[[ResourceSummary], float | None]

# Columns of the source-data table, and how each is read from one summary. A percentage column is derived from the
# fraction beside it.
SOURCE_COLUMNS: tuple[str, ...] = (
    "source_json",
    "available",
    "run",
    "rank_fn",
    "prefetch_budget",
    "num_blocks",
    "mcmc_steps",
    "prefix_cache",
    "gpu_memory_utilization",
    "peak_gpu_memory_gib",
    "device_peak_gpu_memory_gib",
    "baseline_gpu_memory_gib",
    "peak_kv_cache_occupancy",
    "peak_kv_cache_occupancy_percent",
    "prefix_cache_hit_rate",
    "prefix_cache_hit_rate_percent",
    "prefix_cache_queried_tokens",
    "prefix_cache_hit_tokens",
    "engine_steps_sampled",
    "kv_source",
    "prefix_source",
    "nvml_error",
)


def _zero_based_limits(values: Sequence[float], headroom: float) -> tuple[float, float]:
    """Read a magnitude against zero, with room above the largest run."""
    return 0.0, max(values) * headroom


def _banded_limits(values: Sequence[float]) -> tuple[float, float]:
    """Frame a rate that clusters in a narrow band, where zero would flatten it."""
    return (
        max(math.floor(min(values)) - 1.0, 0.0),
        min(math.ceil(max(values)) + 1.0, 100.0),
    )


@dataclass(frozen=True)
class MetricPanel:
    """One resource metric, its axis text, and how it is read and scaled."""

    panel_label: str
    title: str
    y_label: str
    accessor: MetricAccessor
    missing_message: str
    limits: Callable[[Sequence[float]], tuple[float, float]]
    scale: float = 1.0


PANELS: tuple[MetricPanel, ...] = (
    MetricPanel(
        panel_label=panel_letter(0),
        title="Peak KV-cache occupancy (polled)",
        y_label=r"occupied KV-block pool (\%)",
        accessor=lambda summary: summary.peak_kv_cache_occupancy,
        missing_message="No peak KV-cache occupancy values are available",
        limits=lambda values: _zero_based_limits(values, 1.10),
        scale=100.0,
    ),
    MetricPanel(
        panel_label=panel_letter(1),
        title="Prefix-cache hit rate",
        y_label=r"cumulative token hit rate (\%)",
        accessor=lambda summary: summary.prefix_cache_hit_rate,
        missing_message="No prefix-cache hit-rate values are available",
        limits=_banded_limits,
        scale=100.0,
    ),
)


def ordered_rank_names(summaries: Sequence[ResourceSummary]) -> list[str]:
    """Return traversal ranks in the shared case-study order."""
    return sort_rank_names(summary.rank_fn for summary in summaries)


def validate_group(group: ResourceGroup) -> None:
    """Reject duplicate points and incompatible metadata in one comparison."""
    point_keys = [
        (summary.prefetch_budget, summary.rank_fn) for summary in group.summaries
    ]
    if len(point_keys) != len(set(point_keys)):
        raise ValueError(f"Duplicate rank/batch resource point in {group.directory}")
    invariants = {
        (
            summary.num_blocks,
            summary.mcmc_steps,
            summary.prefix_cache,
            summary.gpu_memory_utilization,
        )
        for summary in group.summaries
    }
    if len(invariants) != 1:
        raise ValueError(
            f"Incompatible resource settings in {group.directory}/{group.signature}"
        )
    unknown_ranks = sorted(
        {summary.rank_fn for summary in group.summaries} - RANK_COLORS.keys()
    )
    if unknown_ranks:
        raise ValueError(
            "No shared traversal color for rank(s): " + ", ".join(unknown_ranks)
        )


def batch_sizes_of(summaries: Sequence[ResourceSummary]) -> list[int]:
    """Return the prefetch budgets present, in ascending order."""
    return sorted({summary.prefetch_budget for summary in summaries})


def design_grid(summaries: Sequence[ResourceSummary]) -> list[tuple[int, str]]:
    """Return every batch-size by rank point the observed design implies."""
    return [
        (batch_size, rank_name)
        for batch_size in batch_sizes_of(summaries)
        for rank_name in ordered_rank_names(summaries)
    ]


def missing_combinations(
    summaries: Sequence[ResourceSummary],
) -> list[tuple[int, str]]:
    """Return absent points in the observed batch-size by rank grid."""
    present = {
        (summary.prefetch_budget, summary.rank_fn) for summary in summaries
    }
    return [point for point in design_grid(summaries) if point not in present]


def _as_percent(fraction: float | None) -> float | None:
    return None if fraction is None else 100.0 * fraction


def _source_row(
    summary: ResourceSummary | None, batch_size: int, rank_name: str
) -> dict[str, object]:
    """Create one available or explicitly missing source-data record.

    A missing resource file still gets a row, so a gap in the design shows up in the table rather than only as an absent
    marker in the figure.
    """
    if summary is None:
        return {
            column: {
                "available": False,
                "rank_fn": rank_name,
                "prefetch_budget": batch_size,
            }.get(column, "")
            for column in SOURCE_COLUMNS
        }
    values: dict[str, object] = {
        "source_json": summary.source_path.name,
        "available": True,
        "peak_kv_cache_occupancy_percent": _as_percent(
            summary.peak_kv_cache_occupancy
        ),
        "prefix_cache_hit_rate_percent": _as_percent(summary.prefix_cache_hit_rate),
    }
    # Every remaining column is named after the field it reads, so a typo in SOURCE_COLUMNS raises here rather than
    # silently writing a blank column.
    return {
        column: values[column] if column in values else getattr(summary, column)
        for column in SOURCE_COLUMNS
    }


def build_source_rows(group: ResourceGroup) -> list[dict[str, object]]:
    """Expand the observed design grid so missing resource files stay visible."""
    summaries_by_key = {
        (summary.prefetch_budget, summary.rank_fn): summary
        for summary in group.summaries
    }
    return [
        _source_row(summaries_by_key.get(point), *point)
        for point in design_grid(group.summaries)
    ]


def _rank_offsets(rank_names: Sequence[str]) -> dict[str, float]:
    """Dodge method points slightly so identical measurements remain visible."""
    if len(rank_names) == 1:
        return {rank_names[0]: 0.0}
    step = X_DODGE_WIDTH / (len(rank_names) - 1)
    return {
        rank_name: -X_DODGE_WIDTH / 2.0 + index * step
        for index, rank_name in enumerate(rank_names)
    }


def _draw_metric_lines(
    ax: Axes,
    summaries: Sequence[ResourceSummary],
    rank_names: Sequence[str],
    rank_offsets: dict[str, float],
    panel: MetricPanel,
) -> list[float]:
    """Draw one marker-line series per traversal rank without imputation."""
    plotted_values: list[float] = []
    for rank_index, rank_name in enumerate(rank_names):
        rank_summaries = sorted(
            (
                summary
                for summary in summaries
                if summary.rank_fn == rank_name
                and panel.accessor(summary) is not None
            ),
            key=lambda summary: summary.prefetch_budget,
        )
        if not rank_summaries:
            continue
        values = [
            float(panel.accessor(summary)) * panel.scale for summary in rank_summaries
        ]
        plotted_values.extend(values)
        ax.plot(
            [
                summary.prefetch_budget + rank_offsets[rank_name]
                for summary in rank_summaries
            ],
            values,
            color=RANK_COLORS[rank_name],
            linewidth=LINE_WIDTH,
            marker=RANK_MARKERS[rank_index % len(RANK_MARKERS)],
            markersize=MARKER_SIZE,
            markerfacecolor="white",
            markeredgewidth=MARKER_EDGE_WIDTH,
            zorder=3,
        )
    if not plotted_values:
        raise ValueError(panel.missing_message)
    return plotted_values


def _style_panel(ax: Axes, panel: MetricPanel, batch_sizes: Sequence[int]) -> None:
    """Apply common quantitative styling to one resource panel."""
    ax.set_title(panel.title, loc="left", pad=5.0)
    ax.set_ylabel(panel.y_label)
    ax.set_xticks(batch_sizes)
    ax.set_xlim(min(batch_sizes) - 0.75, max(batch_sizes) + 0.75)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.grid(axis="y", color=GRAY, linewidth=0.55)
    ax.set_axisbelow(True)
    add_panel_label(
        ax,
        panel.panel_label,
        x=PANEL_LABEL_POSITION[0],
        y=PANEL_LABEL_POSITION[1],
    )
    style_quantitative_axis(ax)


def _legend_handles(rank_names: Sequence[str]) -> list[Line2D]:
    """Build one shared traversal-rank legend with color and marker encoding."""
    return [
        Line2D(
            [0],
            [0],
            color=RANK_COLORS[rank_name],
            linewidth=LINE_WIDTH,
            marker=RANK_MARKERS[index % len(RANK_MARKERS)],
            markersize=MARKER_SIZE,
            markerfacecolor="white",
            markeredgewidth=MARKER_EDGE_WIDTH,
            label=rank_label(rank_name),
        )
        for index, rank_name in enumerate(rank_names)
    ]


def _figure_note(summaries: Sequence[ResourceSummary]) -> str:
    """State what one point means and what the figure could not show."""
    parts = ["each point is one run"]
    kv_sources = sorted(
        {summary.kv_source for summary in summaries if summary.kv_source}
    )
    if kv_sources == ["metrics-poll"]:
        parts.append("polled KV peaks may miss brief spikes")
    if missing := missing_combinations(summaries):
        parts.append(f"{len(missing)} missing resource files are omitted")
    return "; ".join(parts)


def draw_resource_figure(group: ResourceGroup, figure_path: Path) -> Path:
    """Draw the two cache-related run-level resource metrics."""
    validate_group(group)
    summaries = group.summaries
    rank_names = ordered_rank_names(summaries)
    batch_sizes = batch_sizes_of(summaries)
    offsets = _rank_offsets(rank_names)

    figure, axes_array = plt.subplots(
        1, len(PANELS), figsize=FIGURE_SIZE, sharex=True, squeeze=False
    )
    axes = list(axes_array.flat)
    for ax, panel in zip(axes, PANELS, strict=True):
        values = _draw_metric_lines(ax, summaries, rank_names, offsets, panel)
        ax.set_ylim(*panel.limits(values))
        _style_panel(ax, panel, batch_sizes)

    figure.suptitle(f"{SUBTREE_METHOD_LABEL} resource usage", y=0.985)
    figure.supxlabel(PREFETCH_BUDGET_AXIS_LABEL, y=0.225)
    figure.text(
        0.5,
        0.155,
        _figure_note(summaries),
        ha="center",
        va="center",
        color=DARK_GRAY,
        fontsize=plt.rcParams["legend.fontsize"],
    )
    figure.legend(
        handles=_legend_handles(rank_names),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),
        ncol=len(rank_names),
        frameon=False,
        handlelength=1.7,
        handletextpad=0.4,
        columnspacing=0.9,
    )
    figure.subplots_adjust(
        left=0.07, right=0.99, bottom=0.31, top=0.85, wspace=0.27
    )
    return save_figure(figure, figure_path, pad_inches=0.02)


def print_group_report(group: ResourceGroup, csv_path: Path, pdf_path: Path) -> None:
    """Report coverage, provenance, missing points, and written outputs."""
    ranks = ordered_rank_names(group.summaries)
    missing = missing_combinations(group.summaries)
    print(
        f"\n{group.directory}\n"
        f"  resources: {len(group.summaries)}; ranks: {len(ranks)}; "
        f"prefetch budgets: {batch_sizes_of(group.summaries)}"
    )
    if missing:
        missing_text = ", ".join(
            f"batch {batch_size}/{rank_name}" for batch_size, rank_name in missing
        )
        print(f"  missing: {missing_text}")
    print(
        "  metrics: peak KV-cache occupancy, cumulative prefix-cache hit rate"
    )
    print(f"  wrote {csv_path}\n  wrote {pdf_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the recursive resource-comparison command-line interface."""
    parser = argparse.ArgumentParser(
        description=(
            "Group compatible resources.json files, write source data, and "
            "draw run-level resource comparisons."
        )
    )
    parser.add_argument(
        "--directory",
        type=Path,
        default=DEFAULT_DIRECTORY,
        help="Root directory searched recursively for *resources.json files.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=None,
        help="Custom output prefix; valid only when one group is discovered.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Render every configuration-safe resource group below one directory."""
    args = build_arg_parser().parse_args(argv)
    directory = args.directory.resolve()
    try:
        groups = discover_resource_groups(directory)
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    if args.output_prefix is not None and len(groups) != 1:
        raise SystemExit(
            "--output-prefix requires exactly one resource group; "
            f"discovered {len(groups)}"
        )

    for group in groups:
        prefix = (
            args.output_prefix.resolve()
            if args.output_prefix is not None
            else group.output_prefix
        )
        try:
            validate_group(group)
            csv_path = write_dict_rows(
                output_path(prefix, ".csv"),
                build_source_rows(group),
                fieldnames=SOURCE_COLUMNS,
            )
            pdf_path = draw_resource_figure(group, output_path(prefix, ".pdf"))
        except (OSError, ValueError) as error:
            raise SystemExit(
                f"Failed resource group {group.directory}/{group.signature}: {error}"
            ) from error
        print_group_report(group, csv_path, pdf_path)

    print(
        f"\nRendered {len(groups)} resource group(s) from "
        f"{sum(len(group.summaries) for group in groups)} JSON files."
    )


if __name__ == "__main__":
    main()
