"""Build and draw shared endpoint metrics for subtree-prefetch comparisons.

This library is not run directly. Invoke one of the sibling figure CLIs:

    /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/.venv/bin/python \
        /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/draw/draw_mean_mh_transitions_and_empirical_time.py \
        --directory /path/to/vllm/logs

    /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/.venv/bin/python \
        /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/draw/draw_peak_kv_cache.py \
        --directory /path/to/vllm/logs
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from case_studies import paths
from case_studies.draw.common.figure_io import save_figure, write_dict_rows
from case_studies.draw.common.plot_helpers import panel_letter
from case_studies.draw.common.rank_plotting import (
    PREFETCH_BUDGET_AXIS_LABEL,
    add_panel_label,
    format_figure_title,
    style_quantitative_axis,
)
from case_studies.extract.logs.model_call_traces import (
    MODEL_CALL_OUTPUT_SUFFIX,
    AnalysisRows,
    BatchComparison,
    ExperimentGroup,
    Row,
    TraceKey,
    build_analysis_rows,
    build_batch_comparisons,
    discover_experiment_groups,
    extract_logs,
    group_rows_by_trace,
    map_trace_ranks,
    mean_and_sample_sd,
)
from case_studies.extract.logs.resource_summary import ResourceSummary, discover_resource_groups
from case_studies.extract.run_naming import (
    output_path,
    rank_label,
    sort_rank_names,
    strip_backend_suffix,
)
from case_studies.plot_config import (
    TRANSITION_RANK_COLORS as RANK_COLORS,
)
from case_studies.plot_config import (
    apply_plot_style,
)

apply_plot_style()

DEFAULT_DIRECTORY = paths.LOGS_DIR / "math500" / "2026-08-08" / "vllm"
DEFAULT_TRACES_PER_RANK = 10
ENDPOINT_CSV_SUFFIX = ".endpoint-metrics.csv"

PANEL_WIDTH = 6.2
FIGURE_HEIGHT = 3.2
PANEL_WSPACE = 0.12
TITLE_Y = 0.95
LEGEND_Y = 0.88
AXES_TOP = 0.72
BAR_WIDTH = 0.13
BAR_GAP = 0.012
BAR_EDGE_WIDTH = 0.4
BAR_HATCH_LINE_WIDTH = 0.4
BAR_HATCHES = ("///", "|||", "xxx", "...", "ooo", r"\\\\")
AXIS_LABEL_FONT_SIZE = 12.0
PANEL_LABEL_POSITION = (-0.05, 1.01)
LEGEND_TITLE_FONT_SIZE = 13.0
LEGEND_FONT_SIZE = 12.0


@dataclass(frozen=True)
class EndpointPanel:
    """One bar panel of an endpoint summary figure."""

    value_key: str
    y_label: str
    y_limits: tuple[float, float] | None
    major_y_ticks: tuple[float, ...] | None = None
    major_y_tick_labels: tuple[str, ...] | None = None
    log_y_base: float | None = None
    dynamic_y_step: float | None = None
    dynamic_y_padding: float = 0.0
    dynamic_y_min: float | None = None


@dataclass(frozen=True)
class EndpointFigureSpec:
    """Configuration owned by one dedicated endpoint figure CLI."""

    figure_suffix: str
    panels: tuple[EndpointPanel, ...]
    description: str
    require_resource_summaries: bool = False


def collect_endpoint_comparisons(
    directory: Path,
    *,
    include_incomplete: bool = False,
    traces_per_rank: int = DEFAULT_TRACES_PER_RANK,
) -> list[BatchComparison]:
    """Parse logs and return every compatible cross-batch comparison."""
    if traces_per_rank <= 0:
        raise ValueError("traces_per_rank must be positive")

    completed: list[tuple[ExperimentGroup, AnalysisRows]] = []
    for group in discover_experiment_groups(directory, MODEL_CALL_OUTPUT_SUFFIX):
        analysis = build_analysis_rows(
            extract_logs(
                group.log_paths,
                include_incomplete,
                tuple(RANK_COLORS),
            ),
            traces_per_rank,
        )
        completed.append((group, analysis))

    comparisons = build_batch_comparisons(completed)
    if not comparisons:
        raise ValueError(
            "No compatible experiment has more than one prefetch budget"
        )
    return comparisons


def _endpoint_rank_row(
    batch_size: int,
    rank_name: str,
    trace_keys: Sequence[TraceKey],
    calls_by_trace: dict[TraceKey, list[Row]],
    events_by_trace: dict[TraceKey, list[Row]],
) -> Row:
    """Summarize final call efficiency and directly logged MH-step times."""
    source_logs = {trace_key[0] for trace_key in trace_keys}
    if len(source_logs) != 1:
        raise ValueError(
            "Matched endpoint rank must come from one source log, found "
            f"{sorted(source_logs)}"
        )
    call_counts: list[float] = []
    transition_efficiencies: list[float] = []
    logged_mh_step_seconds: list[float] = []
    final_iterations: set[int] = set()

    for trace_key in trace_keys:
        call_row = calls_by_trace[trace_key][-1]
        call_iteration = int(call_row["mh_iteration"])
        trace_events = events_by_trace.get(trace_key)
        if not trace_events:
            raise ValueError(
                "Selected trace has no directly logged MH-step times for "
                f"batch size {batch_size}, {rank_name}, {trace_key}"
            )
        call_count = float(call_row["subtree_cumulative_calls"])
        if call_count <= 0.0:
            raise ValueError(
                "Final target-model-call count must be positive for "
                f"batch size {batch_size}, {rank_name}, {trace_key}"
            )
        if len(trace_events) != int(call_count):
            raise ValueError(
                "Logged MH-step times and target-model calls disagree for "
                f"batch size {batch_size}, {rank_name}, {trace_key}: "
                f"{len(trace_events)} times versus {int(call_count)} calls"
            )
        final_iterations.add(call_iteration)
        call_counts.append(call_count)
        transition_efficiencies.append(call_iteration / call_count)
        logged_mh_step_seconds.extend(
            float(event["duration_seconds"]) for event in trace_events
        )

    if len(final_iterations) != 1:
        raise ValueError(
            "Matched endpoint comparison requires a common final MH "
            f"iteration for batch size {batch_size}, {rank_name}: "
            f"{sorted(final_iterations)}"
        )
    mean_calls, sd_calls = mean_and_sample_sd(call_counts)
    mean_efficiency, sd_efficiency = mean_and_sample_sd(transition_efficiencies)
    mean_seconds, sd_seconds = mean_and_sample_sd(logged_mh_step_seconds)
    return {
        "source_log": source_logs.pop(),
        "prefetch_budget": batch_size,
        "rank": rank_name,
        "trace_count": len(trace_keys),
        "configured_mh_transitions": final_iterations.pop(),
        "mean_target_model_calls": mean_calls,
        "sample_sd_target_model_calls": sd_calls,
        "mean_mh_transitions_per_target_model_call": mean_efficiency,
        "sample_sd_mh_transitions_per_target_model_call": sd_efficiency,
        "timed_mh_step_count": len(logged_mh_step_seconds),
        "mean_logged_mh_step_seconds": mean_seconds,
        "sample_sd_logged_mh_step_seconds": sd_seconds,
    }


def build_endpoint_rows(
    batch_analyses: Sequence[tuple[int, AnalysisRows]],
) -> list[Row]:
    """Summarize matched call efficiency and direct logged MH-step time."""
    rows: list[Row] = []
    for batch_size, analysis in sorted(batch_analyses):
        calls_by_trace = group_rows_by_trace(analysis.cumulative)
        events_by_trace = group_rows_by_trace(analysis.events)
        if missing_traces := calls_by_trace.keys() - events_by_trace.keys():
            raise ValueError(
                f"Batch size {batch_size} has traces without logged MH-step "
                f"times: {sorted(missing_traces)}"
            )

        rank_by_trace = map_trace_ranks(calls_by_trace)
        for rank_name in sort_rank_names(set(rank_by_trace.values())):
            trace_keys = sorted(
                key for key, rank in rank_by_trace.items() if rank == rank_name
            )
            rows.append(
                _endpoint_rank_row(
                    batch_size,
                    rank_name,
                    trace_keys,
                    calls_by_trace,
                    events_by_trace,
                )
            )

    endpoint_counts = {int(row["configured_mh_transitions"]) for row in rows}
    if len(endpoint_counts) != 1:
        raise ValueError(
            "Cross-batch endpoint comparison requires one configured MH "
            f"transition count, found {sorted(endpoint_counts)}"
        )
    return rows


def matching_resource_summaries(
    directory: Path, comparison_signature: str
) -> tuple[ResourceSummary, ...]:
    """Return resource data matching one cross-batch log comparison."""
    try:
        resource_groups = discover_resource_groups(directory)
    except ValueError as error:
        if "No *resources.json files found" in str(error):
            return ()
        raise
    expected_signature = strip_backend_suffix(comparison_signature)
    matches = [
        group for group in resource_groups if group.signature == expected_signature
    ]
    if len(matches) > 1:
        raise ValueError(
            "More than one resource group matches comparison signature "
            f"{comparison_signature}"
        )
    return () if not matches else matches[0].summaries


def attach_peak_kv_cache_occupancy(
    rows: Sequence[Row], resource_summaries: Sequence[ResourceSummary]
) -> None:
    """Attach polled peak KV-cache occupancy to matched endpoint rows."""
    summaries_by_key: dict[tuple[int, str], ResourceSummary] = {}
    for summary in resource_summaries:
        key = summary.prefetch_budget, summary.rank_fn
        if key in summaries_by_key:
            raise ValueError(
                f"Duplicate resource summary for prefetch budget {key[0]}, {key[1]}"
            )
        summaries_by_key[key] = summary

    endpoint_keys = {
        (int(row["prefetch_budget"]), str(row["rank"])) for row in rows
    }
    if unexpected_keys := sorted(summaries_by_key.keys() - endpoint_keys):
        raise ValueError(
            f"Resource summaries do not match endpoint rows: {unexpected_keys}"
        )

    for row in rows:
        key = int(row["prefetch_budget"]), str(row["rank"])
        summary = summaries_by_key.get(key)
        occupancy = None if summary is None else summary.peak_kv_cache_occupancy
        if (
            occupancy is not None
            and summary.mcmc_steps != int(row["configured_mh_transitions"])
        ):
            raise ValueError(
                "Resource and endpoint MH-transition counts disagree for "
                f"prefetch budget {key[0]}, {key[1]}"
            )
        row["resource_source_json"] = (
            "" if summary is None else summary.source_path.name
        )
        row["peak_kv_cache_occupancy"] = occupancy
        row["peak_kv_cache_occupancy_percent"] = (
            None if occupancy is None else 100.0 * occupancy
        )
        row["kv_source"] = "" if summary is None else summary.kv_source


def prepare_endpoint_rows(
    comparison: BatchComparison,
    *,
    require_resource_summaries: bool = False,
) -> list[Row]:
    """Build all endpoint metrics needed by either dedicated figure."""
    rows = build_endpoint_rows(comparison.batch_analyses)
    resource_summaries = matching_resource_summaries(
        comparison.directory, comparison.signature
    )
    if resource_summaries:
        attach_peak_kv_cache_occupancy(rows, resource_summaries)
    elif require_resource_summaries:
        raise ValueError(
            "No matching *resources.json files for "
            f"{comparison.directory}/{comparison.signature}"
        )
    return rows


def _endpoint_y_axis(
    panel: EndpointPanel,
    rows: Sequence[Row],
) -> tuple[tuple[float, float], tuple[float, ...] | None]:
    """Resolve fixed or data-adaptive endpoint-panel y-axis settings."""
    if panel.y_limits is not None:
        return panel.y_limits, panel.major_y_ticks
    if panel.dynamic_y_step is None:
        raise ValueError(f"Panel {panel.value_key} has no y-axis specification")
    values = [
        float(row[panel.value_key])
        for row in rows
        if row.get(panel.value_key) is not None
    ]
    if not values:
        raise ValueError(f"Panel {panel.value_key} has no values")
    step = panel.dynamic_y_step
    lower = (
        panel.dynamic_y_min
        if panel.dynamic_y_min is not None
        else max(
            0.0,
            step * math.floor((min(values) - panel.dynamic_y_padding) / step),
        )
    )
    upper = step * math.ceil((max(values) + panel.dynamic_y_padding) / step)
    if upper <= lower:
        upper = lower + step
    ticks = tuple(
        lower + index * step
        for index in range(int(round((upper - lower) / step)))
    )
    return (lower, upper), ticks


def _endpoint_bar_positions(
    panel: EndpointPanel,
    rank_names: Sequence[str],
    batch_sizes: Sequence[int],
    rows_by_key: dict[tuple[str, int], Row],
) -> dict[tuple[str, int], float]:
    """Centre each batch's available bars over its x-axis tick."""
    positions: dict[tuple[str, int], float] = {}
    for batch_index, batch_size in enumerate(batch_sizes):
        available = [
            rank_name
            for rank_name in rank_names
            if (row := rows_by_key.get((rank_name, batch_size))) is not None
            and row.get(panel.value_key) is not None
        ]
        bar_step = BAR_WIDTH + BAR_GAP
        for rank_index, rank_name in enumerate(available):
            offset = (rank_index - (len(available) - 1) / 2.0) * bar_step
            positions[(rank_name, batch_size)] = batch_index + offset
    return positions


def draw_endpoint_figure(
    rows: Sequence[Row],
    panels: Sequence[EndpointPanel],
    *,
    figure_title: str,
) -> Figure:
    """Draw one horizontal endpoint figure with a shared top legend."""
    rank_names = sort_rank_names({str(row["rank"]) for row in rows})
    batch_sizes = sorted({int(row["prefetch_budget"]) for row in rows})
    rows_by_key = {
        (str(row["rank"]), int(row["prefetch_budget"])): row for row in rows
    }
    figure, axes = plt.subplots(
        1,
        len(panels),
        figsize=(PANEL_WIDTH * len(panels), FIGURE_HEIGHT),
        sharex=True,
        squeeze=False,
    )
    panel_axes = list(axes[0])
    legend_handles: list[Patch] = []
    for rank_index, rank_name in enumerate(rank_names):
        hatch = BAR_HATCHES[rank_index % len(BAR_HATCHES)]
        for ax, panel in zip(panel_axes, panels, strict=True):
            positions = _endpoint_bar_positions(
                panel, rank_names, batch_sizes, rows_by_key
            )
            drawn = [
                (positions[(rank_name, batch_size)], float(row[panel.value_key]))
                for batch_size in batch_sizes
                if (row := rows_by_key.get((rank_name, batch_size))) is not None
                and row.get(panel.value_key) is not None
            ]
            bars = ax.bar(
                [position for position, _ in drawn],
                [value for _, value in drawn],
                width=BAR_WIDTH,
                facecolor="none",
                edgecolor=RANK_COLORS[rank_name],
                hatch=hatch,
                linewidth=BAR_EDGE_WIDTH,
                zorder=3,
            )
            for bar in bars:
                bar.set_hatch_linewidth(BAR_HATCH_LINE_WIDTH)
        legend_handle = Patch(
            facecolor="none",
            edgecolor=RANK_COLORS[rank_name],
            hatch=hatch,
            linewidth=BAR_EDGE_WIDTH,
            label=rank_label(rank_name),
        )
        legend_handle.set_hatch_linewidth(BAR_HATCH_LINE_WIDTH)
        legend_handles.append(legend_handle)

    for panel_index, (ax, panel) in enumerate(
        zip(panel_axes, panels, strict=True)
    ):
        add_panel_label(
            ax,
            panel_letter(panel_index),
            x=PANEL_LABEL_POSITION[0],
            y=PANEL_LABEL_POSITION[1],
        )
        ax.set_xlabel(
            PREFETCH_BUDGET_AXIS_LABEL,
            fontsize=AXIS_LABEL_FONT_SIZE,
        )
        ax.set_ylabel(panel.y_label, fontsize=AXIS_LABEL_FONT_SIZE)
        ax.set_xticks(range(len(batch_sizes)), batch_sizes)
        ax.set_xlim(-0.43, len(batch_sizes) - 0.57)
        if panel.log_y_base is not None:
            ax.set_yscale("log", base=panel.log_y_base)
        y_limits, major_y_ticks = _endpoint_y_axis(panel, rows)
        ax.set_ylim(*y_limits)
        if major_y_ticks is None:
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        else:
            ax.set_yticks(major_y_ticks, panel.major_y_tick_labels)
        ax.minorticks_off()
        style_quantitative_axis(ax)

    figure.suptitle(
        format_figure_title(figure_title),
        x=0.5,
        y=TITLE_Y,
        fontsize=LEGEND_TITLE_FONT_SIZE,
    )
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, LEGEND_Y),
        ncol=len(rank_names),
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
        borderaxespad=0.0,
        handlelength=1.5,
        handletextpad=0.35,
        columnspacing=0.8,
    )
    figure.subplots_adjust(
        left=0.065,
        right=0.995,
        bottom=0.18,
        top=AXES_TOP,
        wspace=PANEL_WSPACE,
    )
    return figure


def render_endpoint_comparison(
    comparison: BatchComparison,
    spec: EndpointFigureSpec,
    *,
    output: Path | None = None,
    csv_output: Path | None = None,
) -> tuple[Path, Path]:
    """Render one endpoint figure and write its shared source-data table."""
    rows = prepare_endpoint_rows(
        comparison,
        require_resource_summaries=spec.require_resource_summaries,
    )
    figure = draw_endpoint_figure(
        rows,
        spec.panels,
        figure_title=comparison.figure_title,
    )
    figure_path = output or output_path(
        comparison.output_prefix,
        spec.figure_suffix,
    )
    csv_path = csv_output or output_path(
        comparison.output_prefix,
        ENDPOINT_CSV_SUFFIX,
    )
    write_dict_rows(csv_path, rows)
    save_figure(figure, figure_path, pad_inches=0.02)
    return figure_path, csv_path


def build_endpoint_arg_parser(spec: EndpointFigureSpec) -> argparse.ArgumentParser:
    """Build the common endpoint-figure command-line interface."""
    parser = argparse.ArgumentParser(description=spec.description)
    parser.add_argument("--directory", type=Path, default=DEFAULT_DIRECTORY)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--csv-output", type=Path, default=None)
    parser.add_argument("--include-incomplete", action="store_true")
    parser.add_argument(
        "--traces-per-rank",
        type=int,
        default=DEFAULT_TRACES_PER_RANK,
    )
    return parser


def run_endpoint_figure_cli(
    spec: EndpointFigureSpec,
    argv: Sequence[str] | None = None,
) -> None:
    """Collect, render, and report every compatible endpoint comparison."""
    args = build_endpoint_arg_parser(spec).parse_args(argv)
    try:
        comparisons = collect_endpoint_comparisons(
            args.directory.resolve(),
            include_incomplete=args.include_incomplete,
            traces_per_rank=args.traces_per_rank,
        )
        if (args.output is not None or args.csv_output is not None) and len(
            comparisons
        ) != 1:
            raise ValueError(
                "--output and --csv-output require exactly one comparison; "
                f"found {len(comparisons)}"
            )
        for comparison in comparisons:
            figure_path, csv_path = render_endpoint_comparison(
                comparison,
                spec,
                output=args.output.resolve() if args.output else None,
                csv_output=args.csv_output.resolve() if args.csv_output else None,
            )
            print(f"wrote {figure_path}")
            print(f"wrote {csv_path}")
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
