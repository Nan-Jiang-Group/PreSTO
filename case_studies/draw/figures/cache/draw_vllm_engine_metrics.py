#!/usr/bin/env python3
"""Draw periodic vLLM engine metrics from subtree-prefetch logs.

For one run, write ``.engine-metrics.csv`` and a horizontal four-panel
``.engine-metrics.pdf`` beside the log:

    /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/.venv/bin/python \
        /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/draw/draw_vllm_engine_metrics.py \
        --log /absolute/path/to/run.subtreePrefetch.vllm.log

For a directory, combine compatible runs into one three-column figure per
traversal rule, with one row for every available prefetch budget:

    /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/.venv/bin/python \
        /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/draw/draw_vllm_engine_metrics.py \
        --directory /absolute/path/to/vllm/results

For a pasted logger entry saved to a text file, use ``--snapshot-log`` instead of ``--log``. This draws labelled bars
for exactly one sample, without assuming a dataset, method, prefetch budget, or a time trajectory.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from operator import attrgetter
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import MaxNLocator

from case_studies.draw.common.figure_io import save_figure, write_dict_rows
from case_studies.draw.common.plot_helpers import panel_letter
from case_studies.draw.common.rank_plotting import (
    add_panel_label,
    prefetch_size_label,
    style_quantitative_axis,
)
from case_studies.extract.logs.vllm_engine_metrics import (
    EngineMetricSample,
    elapsed_seconds,
    parse_engine_metrics,
)
from case_studies.extract.run_naming import (
    ALL_PREFETCH_BUDGETS_TAG,
    VLLM_LOG_GLOB,
    config_output_prefix,
    output_path,
    prefetch_budget_of,
    rank_label,
    rank_of,
    strip_tokens,
)
from case_studies.plot_config import (
    BLUE,
    DARK_GRAY,
    GREEN,
    ORANGE,
    PURPLE,
    RED,
    SUBTREE_METHOD_LABEL,
    apply_plot_style,
)

apply_plot_style()

SINGLE_FIGURE_SIZE = (16.4, 3.3)
COMBINED_COLUMN_WIDTH = 4.0
COMBINED_ROW_HEIGHT = 2.45
COMBINED_TITLE_Y = 0.985
COMBINED_TOP_MARGIN = 0.95
COMBINED_COLUMN_SPACE = 0.22
COMBINED_ROW_SPACE = 0.12
COMBINED_AXIS_LABEL_FONT_SIZE = 12.0
LINE_WIDTH = 1.25
PANEL_LABEL_POSITION = (-0.15, 1.01)
LEGEND_OPTIONS = {
    "loc": "lower right",
    "frameon": False,
    "borderaxespad": 0.35,
    "handlelength": 1.5,
    "handletextpad": 0.4,
    "labelspacing": 0.25,
}
SOURCE_DATA_FIELDS = [
    "source_log",
    "source_line",
    "timestamp",
    "elapsed_seconds",
    "prompt_throughput_tokens_s",
    "generation_throughput_tokens_s",
    "running_requests",
    "waiting_requests",
    "gpu_kv_cache_usage_percent",
    "prefix_cache_hit_rate_percent",
]
COMBINED_SOURCE_DATA_FIELDS = [
    "source_log",
    "rank",
    "prefetch_budget",
    *SOURCE_DATA_FIELDS[1:],
]


def _headroom_limits(values: Sequence[float], factor: float) -> tuple[float, float]:
    """Return a zero-based axis with a little room above the tallest sample."""
    return 0.0, max(max(values) * factor, 1.0)


def _percent_limits(
    values: Sequence[float], *, include_zero: bool
) -> tuple[float, float]:
    """Return readable bounded limits for one percentage time series."""
    lower, upper = min(values), max(values)
    if include_zero:
        return 0.0, min(max(upper * 1.08, 0.2), 100.0)
    padding = max((upper - lower) * 0.08, 0.25)
    return max(lower - padding, 0.0), min(upper + padding, 100.0)


@dataclass(frozen=True)
class MetricSeries:
    """One line drawn on an engine-metric panel."""

    color: str
    value: Callable[[EngineMetricSample], float]
    # Only a panel drawing more than one series needs a legend key.
    label: str | None = None


@dataclass(frozen=True)
class MetricPanel:
    """One engine-metric column: what it draws, names itself, and scales to."""

    title: str
    y_label: str
    series: tuple[MetricSeries, ...]
    limits: Callable[[Sequence[float]], tuple[float, float]]
    as_steps: bool = False
    integer_y: bool = False

    def values(self, samples: Iterable[EngineMetricSample]) -> list[float]:
        """Every value this panel plots, for scaling its axis."""
        sample_list = list(samples)
        return [
            series.value(sample) for series in self.series for sample in sample_list
        ]

    def draw(
        self,
        ax: Axes,
        elapsed_minutes: Sequence[float],
        samples: Sequence[EngineMetricSample],
        *,
        show_legend: bool,
    ) -> None:
        """Draw every series of this panel onto one axis."""
        draw_line = ax.step if self.as_steps else ax.plot
        options = {"where": "post"} if self.as_steps else {}
        for series in self.series:
            if series.label is not None:
                options["label"] = series.label
            draw_line(
                elapsed_minutes,
                [series.value(sample) for sample in samples],
                color=series.color,
                linewidth=LINE_WIDTH,
                **options,
            )
        # A single-series panel names itself in its y label; only the paired panels need a key telling the two lines
        # apart.
        if show_legend and len(self.series) > 1:
            ax.legend(**LEGEND_OPTIONS)


PANELS: tuple[MetricPanel, ...] = (
    MetricPanel(
        title="Token throughput",
        y_label=r"throughput (tokens s$^{-1}$)",
        series=(
            MetricSeries(
                BLUE,
                attrgetter("generation_throughput_tokens_s"),
                label="generation",
            ),
            MetricSeries(
                ORANGE,
                attrgetter("prompt_throughput_tokens_s"),
                label="prompt",
            ),
        ),
        limits=lambda values: _headroom_limits(values, 1.06),
    ),
    MetricPanel(
        title="Request load",
        y_label="requests",
        series=(
            MetricSeries(GREEN, attrgetter("running_requests"), label="running"),
            MetricSeries(RED, attrgetter("waiting_requests"), label="waiting"),
        ),
        limits=lambda values: _headroom_limits(values, 1.08),
        as_steps=True,
        integer_y=True,
    ),
    MetricPanel(
        title="KV-cache occupancy",
        y_label=r"GPU KV cache usage (\%)",
        series=(MetricSeries(PURPLE, attrgetter("gpu_kv_cache_usage_percent")),),
        limits=lambda values: _percent_limits(values, include_zero=True),
    ),
    MetricPanel(
        title="Prefix-cache hits",
        y_label=r"prefix cache hit rate (\%)",
        series=(
            MetricSeries(DARK_GRAY, attrgetter("prefix_cache_hit_rate_percent")),
        ),
        limits=lambda values: _percent_limits(values, include_zero=False),
    ),
)
COMBINED_PANELS = PANELS[:3]


@dataclass(frozen=True)
class EngineMetricRun:
    """Parsed engine metrics and identifying configuration for one log."""

    log_path: Path
    rank: str
    batch_size: int
    samples: tuple[EngineMetricSample, ...]
    elapsed_values: tuple[float, ...]

    @classmethod
    def from_log(cls, log_path: Path) -> EngineMetricRun:
        """Parse one engine-metric log and its configuration identity."""
        samples = tuple(parse_engine_metrics(log_path))
        return cls(
            log_path=log_path,
            rank=rank_of(log_path),
            batch_size=prefetch_budget_of(log_path),
            samples=samples,
            elapsed_values=tuple(elapsed_seconds(list(samples))),
        )

    @property
    def elapsed_minutes(self) -> list[float]:
        return [value / 60.0 for value in self.elapsed_values]

    @property
    def title(self) -> str:
        """A concise title from the traversal rank and prefetch budget."""
        return (
            f"{SUBTREE_METHOD_LABEL} + {rank_label(self.rank)}; "
            f"{prefetch_size_label(self.batch_size)}"
        )

    def source_rows(self, *, identify_run: bool) -> list[dict[str, object]]:
        """One record per logger snapshot, for the figure's source-data CSV."""
        rows: list[dict[str, object]] = []
        for sample, elapsed_value in zip(
            self.samples, self.elapsed_values, strict=True
        ):
            row: dict[str, object] = {"source_log": self.log_path.name}
            if identify_run:
                row |= {"rank": self.rank, "prefetch_budget": self.batch_size}
            row |= {
                "source_line": sample.source_line,
                "timestamp": sample.timestamp.isoformat(sep=" "),
                "elapsed_seconds": f"{elapsed_value:.1f}",
                "prompt_throughput_tokens_s": sample.prompt_throughput_tokens_s,
                "generation_throughput_tokens_s": (
                    sample.generation_throughput_tokens_s
                ),
                "running_requests": sample.running_requests,
                "waiting_requests": sample.waiting_requests,
                "gpu_kv_cache_usage_percent": sample.gpu_kv_cache_usage_percent,
                "prefix_cache_hit_rate_percent": sample.prefix_cache_hit_rate_percent,
            }
            rows.append(row)
        return rows


def write_source_data(
    runs: Sequence[EngineMetricRun],
    csv_path: Path,
    *,
    identify_run: bool,
) -> Path:
    """Write the logger snapshots behind one figure to a CSV table.

    A comparison pools several runs, so its rows carry the rank and batch size that tell them apart; a single-run table
    does not repeat its own identity.
    """
    return write_dict_rows(
        csv_path,
        [row for run in runs for row in run.source_rows(identify_run=identify_run)],
        fieldnames=(
            COMBINED_SOURCE_DATA_FIELDS if identify_run else SOURCE_DATA_FIELDS
        ),
    )


def _style_panel(
    ax: Axes,
    panel: MetricPanel,
    *,
    panel_label: str,
    show_title: bool,
) -> None:
    """Apply the shared case-study panel style."""
    if show_title:
        ax.set_title(panel.title, loc="left", pad=5.0)
    ax.set_ylabel(panel.y_label)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=panel.integer_y))
    if panel_label:
        add_panel_label(
            ax,
            panel_label,
            x=PANEL_LABEL_POSITION[0],
            y=PANEL_LABEL_POSITION[1],
        )
    style_quantitative_axis(ax)


def draw_run_row(
    axes: Sequence[Axes],
    run: EngineMetricRun,
    *,
    is_first_row: bool,
    show_legends: bool = True,
    panels: Sequence[MetricPanel] = PANELS,
) -> None:
    """Draw one run across the selected engine-metric columns."""
    for index, (ax, panel) in enumerate(zip(axes, panels, strict=True)):
        panel.draw(
            ax,
            run.elapsed_minutes,
            run.samples,
            show_legend=is_first_row and show_legends,
        )
        _style_panel(
            ax,
            panel,
            panel_label=panel_letter(index) if is_first_row else "",
            show_title=is_first_row,
        )


def apply_row_limits(
    axes: Sequence[Axes],
    runs: Sequence[EngineMetricRun],
    *,
    x_limits: tuple[float, float] | None = None,
    panels: Sequence[MetricPanel] = PANELS,
) -> None:
    """Scale each column to every run it must display."""
    samples = [sample for run in runs for sample in run.samples]
    for ax, panel in zip(axes, panels, strict=True):
        if x_limits is not None:
            ax.set_xlim(*x_limits)
        ax.set_ylim(*panel.limits(panel.values(samples)))


def draw_engine_metrics(run: EngineMetricRun, figure_path: Path) -> Path:
    """Draw one run as four panels arranged in a single horizontal row."""
    figure, axes_grid = plt.subplots(
        1,
        len(PANELS),
        figsize=SINGLE_FIGURE_SIZE,
        sharex=True,
        squeeze=False,
    )
    axes = list(axes_grid[0])
    draw_run_row(axes, run, is_first_row=True)
    apply_row_limits(axes, [run])
    for ax in axes:
        ax.set_xlabel("elapsed time (min)")

    figure.suptitle(run.title, y=0.985)
    figure.subplots_adjust(
        left=0.06,
        right=0.992,
        bottom=0.20,
        top=0.80,
        wspace=0.34,
    )
    return save_figure(figure, figure_path, pad_inches=0.02)


def render_snapshot(log_path: Path, output_prefix: Path) -> tuple[Path, Path]:
    """Draw the six reported values from a single entry without a time axis."""
    samples = parse_engine_metrics(log_path)
    if len(samples) != 1:
        raise ValueError(
            f"Snapshot mode requires exactly one engine-metric record; "
            f"found {len(samples)} in {log_path}. Use --log for a full run."
        )
    sample = samples[0]
    # The logger entry supplies no year. Do not export the parser's fallback year as though it were part of the
    # measurement.
    timestamp = sample.timestamp.strftime("%m-%d %H:%M:%S")
    source_row = {
        "source_log": str(log_path),
        **asdict(sample),
        "timestamp": timestamp,
        "elapsed_seconds": 0.0,
    }
    csv_path = write_dict_rows(
        output_path(output_prefix, ".engine-metrics.csv"),
        [source_row],
        fieldnames=SOURCE_DATA_FIELDS,
    )
    with plt.rc_context({"font.size": 13, "axes.labelsize": 14}):
        figure, axes = plt.subplots(1, len(PANELS), figsize=(12.6, 3.5))
        for index, (ax, panel) in enumerate(zip(axes, PANELS, strict=True)):
            values = panel.values(samples)
            labels = [series.label or "reported value" for series in panel.series]
            ax.bar(
                range(len(values)), values, width=0.55,
                color=[series.color for series in panel.series],
                linewidth=0,
            )
            for position, value in enumerate(values):
                label = str(int(value)) if panel.integer_y else f"{value:.1f}"
                ax.annotate(
                    label, (position, value), xytext=(0, 5),
                    textcoords="offset points", ha="center", va="bottom",
                )
            ax.set_xticks(range(len(values)), labels, fontsize=15)
            ax.set_xlim(-0.65, len(values) - 0.35)
            upper = max(max(values) * 1.25, 1.0)
            ax.set_ylim(0, min(upper, 100) if index >= 2 else upper)
            ax.set_ylabel(panel.y_label)
            ax.set_title(f"({chr(97 + index)}) {panel.title}", fontsize=13, pad=9)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=panel.integer_y))
            ax.minorticks_off()
            ax.grid(False)
        figure.suptitle(f"vLLM logger snapshot: {timestamp}", fontsize=15, y=0.98)
        figure.text(
            0.5, 0.02, "Single reported sample; throughput values are logger averages.",
            ha="center", fontsize=11,
        )
        figure.subplots_adjust(left=0.055, right=0.995, bottom=0.19, top=0.78, wspace=0.47)
        figure_path = save_figure(
            figure, output_path(output_prefix, ".engine-metrics.pdf"), pad_inches=0.04
        )
    return csv_path, figure_path


def draw_batch_size_comparison(
    runs: Sequence[EngineMetricRun], figure_path: Path
) -> Path:
    """Draw one traversal rule with one row per available prefetch budget."""
    if not runs:
        raise ValueError("Cannot draw an empty engine-metric comparison")
    ranks = {run.rank for run in runs}
    batch_sizes = [run.batch_size for run in runs]
    if len(ranks) != 1:
        raise ValueError(f"Comparison contains multiple traversal ranks: {ranks}")
    if len(batch_sizes) != len(set(batch_sizes)):
        raise ValueError(f"Comparison repeats prefetch budgets: {batch_sizes}")

    ordered_runs = sorted(runs, key=lambda run: run.batch_size)
    figure, axes_grid = plt.subplots(
        len(ordered_runs),
        len(COMBINED_PANELS),
        figsize=(
            COMBINED_COLUMN_WIDTH * len(COMBINED_PANELS),
            COMBINED_ROW_HEIGHT * len(ordered_runs) + 0.8,
        ),
        sharex=True,
        squeeze=False,
    )
    for row_index, run in enumerate(ordered_runs):
        row_axes = list(axes_grid[row_index])
        draw_run_row(
            row_axes,
            run,
            is_first_row=row_index == 0,
            show_legends=False,
            panels=COMBINED_PANELS,
        )
        for ax in row_axes:
            ax.yaxis.label.set_size(COMBINED_AXIS_LABEL_FONT_SIZE)
            ax.minorticks_off()
        row_axes[0].annotate(
            prefetch_size_label(run.batch_size),
            xy=(-0.33, 0.5),
            xycoords="axes fraction",
            ha="center",
            va="center",
            rotation=90,
        )
        if row_index == len(ordered_runs) - 1:
            for ax in row_axes:
                ax.set_xlabel(
                    "elapsed time (min)",
                    fontsize=COMBINED_AXIS_LABEL_FONT_SIZE,
                )

    # Every row shares one scale so the batch sizes can be read against another.
    max_minutes = max(run.elapsed_values[-1] for run in ordered_runs) / 60.0
    for row_axes in axes_grid:
        apply_row_limits(
            list(row_axes),
            ordered_runs,
            x_limits=(0.0, max(max_minutes, 1.0)),
            panels=COMBINED_PANELS,
        )

    figure.suptitle(
        f"{SUBTREE_METHOD_LABEL} + {rank_label(ordered_runs[0].rank)}; "
        "prefetch budget comparison",
        y=COMBINED_TITLE_Y,
    )
    figure.subplots_adjust(
        left=0.075,
        right=0.993,
        bottom=0.075,
        top=COMBINED_TOP_MARGIN,
        wspace=COMBINED_COLUMN_SPACE,
        hspace=COMBINED_ROW_SPACE,
    )
    return save_figure(figure, figure_path, pad_inches=0.02)


def _comparison_groups(
    log_paths: Iterable[Path],
) -> dict[tuple[Path, str, str], list[EngineMetricRun]]:
    """Group compatible logs by directory, configuration, and traversal rank."""
    groups: dict[tuple[Path, str, str], list[EngineMetricRun]] = defaultdict(list)
    for log_path in log_paths:
        run = EngineMetricRun.from_log(log_path)
        signature = strip_tokens(config_output_prefix(log_path).name)
        groups[(log_path.parent, signature, run.rank)].append(run)
    return groups


def render_log(
    log_path: Path, output_prefix: Path
) -> tuple[Path, Path, EngineMetricRun]:
    """Write the source CSV and horizontal four-panel figure for one log."""
    run = EngineMetricRun.from_log(log_path)
    csv_path = write_source_data(
        [run],
        output_path(output_prefix, ".engine-metrics.csv"),
        identify_run=False,
    )
    figure_path = draw_engine_metrics(
        run, output_path(output_prefix, ".engine-metrics.pdf")
    )
    return csv_path, figure_path, run


def render_directory(directory: Path) -> list[tuple[Path, Path, int]]:
    """Write one rank-specific batch-size comparison for every compatible set."""
    log_paths = sorted(directory.rglob(VLLM_LOG_GLOB))
    if not log_paths:
        raise ValueError(f"No {VLLM_LOG_GLOB} files found under {directory}")

    outputs: list[tuple[Path, Path, int]] = []
    for (_, signature, rank), runs in sorted(_comparison_groups(log_paths).items()):
        ordered_runs = sorted(runs, key=lambda run: run.batch_size)
        prefix = (
            ordered_runs[0].log_path.parent
            / f"{signature}.rank-{rank}.{ALL_PREFETCH_BUDGETS_TAG}"
        )
        csv_path = write_source_data(
            ordered_runs,
            output_path(prefix, ".engine-metrics.csv"),
            identify_run=True,
        )
        figure_path = draw_batch_size_comparison(
            ordered_runs, output_path(prefix, ".engine-metrics.pdf")
        )
        outputs.append((csv_path, figure_path, len(ordered_runs)))
    return outputs


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the single-log and directory-comparison command-line interface."""
    parser = argparse.ArgumentParser(
        description=(
            "Parse periodic vLLM engine metrics, write source data, and draw "
            "horizontal single-run or multi-batch-size figures."
        )
    )
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--log", type=Path, help="One vLLM log file.")
    inputs.add_argument(
        "--snapshot-log", type=Path,
        help="Text file with exactly one engine-metric entry; draw labelled bars.",
    )
    inputs.add_argument(
        "--directory",
        type=Path,
        help="Directory searched recursively for compatible vLLM logs.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=None,
        help="Single-log output prefix; defaults to the run configuration.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Render engine-metric outputs for one log or one directory tree."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.snapshot_log is not None:
        log_path = args.snapshot_log.resolve()
        prefix = (
            args.output_prefix.resolve()
            if args.output_prefix is not None else log_path.with_suffix("")
        )
        try:
            csv_path, figure_path = render_snapshot(log_path, prefix)
        except (OSError, ValueError) as error:
            raise SystemExit(f"Failed to plot snapshot {log_path}: {error}") from error
        print(f"parsed 1 engine-metric sample\nwrote {csv_path}\nwrote {figure_path}")
        return
    if args.directory is not None:
        if args.output_prefix is not None:
            parser.error("--output-prefix can only be used with --log")
        directory = args.directory.resolve()
        if not directory.is_dir():
            raise SystemExit(f"No such directory: {directory}")
        try:
            outputs = render_directory(directory)
        except (OSError, ValueError) as error:
            raise SystemExit(f"Failed to plot {directory}: {error}") from error
        for csv_path, figure_path, row_count in outputs:
            print(
                f"combined {row_count} prefetch budget(s)\n"
                f"wrote {csv_path}\n"
                f"wrote {figure_path}"
            )
        return

    log_path = args.log.resolve()
    if not log_path.is_file():
        raise SystemExit(f"No such log file: {log_path}")
    resolved_prefix = (
        args.output_prefix.resolve()
        if args.output_prefix is not None
        else config_output_prefix(log_path)
    )
    try:
        csv_path, figure_path, run = render_log(log_path, resolved_prefix)
    except (OSError, ValueError) as error:
        raise SystemExit(f"Failed to plot {log_path}: {error}") from error

    print(
        f"parsed {len(run.samples)} engine-metric samples over "
        f"{run.elapsed_values[-1] / 60.0:.1f} min\n"
        f"wrote {csv_path}\n"
        f"wrote {figure_path}"
    )


if __name__ == "__main__":
    main()
