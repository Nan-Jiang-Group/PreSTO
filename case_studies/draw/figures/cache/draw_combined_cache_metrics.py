#!/usr/bin/env python3
"""Draw GPU/cache usage, prefix-cache hit rate, and a selected third cache metric.

Run with:

    /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/.venv/bin/python \
        /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/draw/draw_combined_cache_metrics.py

For LCB V6 with a PowerMH baseline, run the same script with:

    --baseline-resource-json /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/logs/lcb_v6/2026-09-07/vllm/dataset-lcb_v6.model-qwen3.5-9b.alpha4.0.steps100.blocks-1.samples20.maxnew1024.seed10086.resources.json \
    --dataset-label 'LCB V6' --model-label 'Qwen3.5-9B' \
    --hit-rate-tick-step 2 --first-metric peak-gpu-memory-delta \
    --include-excluded-ranks \
    --third-metric eviction-rate \
    --output /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/output/pdf/lcb-v6-qwen3.5-9b-resource-usage-and-peak-kv-cache.pdf

Baseline mode reads compatible sibling resource JSONs; add --subtree-resource-dir DIR ... when the matched PreSTO runs
were written under another date directory. With --first-metric peak-gpu-memory-delta, panel (a) converts
peak_llm_generate_gpu_memory_delta_gib to decimal MB using 2**30 / 1_000_000. This is the sampled peak during generation
minus post-load GPU memory, clipped at zero. The default first metric remains peak KV-block occupancy for older MATH500
data.

With --third-metric eviction-rate, panel (c) plots 100 * kv_cache_eviction_rate directly from those JSONs, checking it
against cached blocks evicted / physical blocks allocated. Missing eviction measurements are rejected rather than
estimated.

The default panel (c), miss-tokens-per-answer, divides the run's queried tokens minus cache-hit tokens by the number of
completed answers in its result CSV, and displays thousands of tokens per answer. Every internal prefix-cache query
contributes, including queries for unused proposals. This measures prefix-cache miss workload, not decoding work,
runtime, or correctness. The derived values and source paths are written to a CSV beside the PDF.

Use --third-metric peak-occupancy to reproduce the former duplicated panel (c).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, MultipleLocator, NullLocator

from case_studies import paths
from case_studies.draw.common.figure_io import save_figure, write_dict_rows
from case_studies.draw.common.plot_helpers import panel_letter
from case_studies.draw.common.rank_plotting import PREFETCH_BUDGET_AXIS_LABEL, style_quantitative_axis
from case_studies.extract.logs.resource_summary import parse_resource_summary
from case_studies.extract.run_naming import RANK_ORDER, is_excluded_rank, rank_label, sort_rank_names, strip_tokens
from case_studies.plot_config import (
    DARK_GRAY,
    GRAY,
    POWER_METHOD_LABEL,
    PURPLE,
    SUBTREE_METHOD_LABEL,
    apply_plot_style,
)
from case_studies.plot_config import (
    TRANSITION_RANK_COLORS as RANK_COLORS,
)

apply_plot_style()

REPO_ROOT = paths.REPO_ROOT
LOG_DIRECTORY = (
    REPO_ROOT / "case_studies/logs/math500/2026-08-08/vllm"
)
DEFAULT_RESOURCE_CSV = LOG_DIRECTORY / (
    "dataset-math500.model-qwen.alpha4.0.steps100.blocks-8.samples10."
    "maxnew3072.seed10086.resource-usage.csv"
)
DEFAULT_ENDPOINT_CSV = LOG_DIRECTORY / "all-prefetch-budget.endpoint-metrics.csv"
DEFAULT_OUTPUT = (
    REPO_ROOT / "output/pdf/math500-qwen-resource-usage-and-peak-kv-cache.pdf"
)

FIGURE_SIZE = (11.7, 3.6)
LINE_WIDTH = 1.35
MARKER_SIZE = 5.2
MARKER_EDGE_WIDTH = 1.0
RANK_MARKERS = ("o", "s", "^", "D", "v", "P")
X_DODGE_WIDTH = 0.58
HIT_RATE_RANGE_STEP = 0.2
FIGURE_FONT_SIZE = 14

plt.rcParams.update(
    {
        "font.size": FIGURE_FONT_SIZE,
        "axes.titlesize": FIGURE_FONT_SIZE,
        "axes.labelsize": FIGURE_FONT_SIZE,
        "xtick.labelsize": FIGURE_FONT_SIZE,
        "ytick.labelsize": FIGURE_FONT_SIZE,
        "legend.fontsize": FIGURE_FONT_SIZE,
        "figure.titlesize": FIGURE_FONT_SIZE,
        "figure.labelsize": FIGURE_FONT_SIZE,
    }
)


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read one source-data CSV without changing its values."""
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def cache_source_row(path: Path, payload: dict, *, baseline: bool) -> dict[str, str]:
    """Read measured cache values with their completed-run provenance."""
    run = payload["run"]
    if path.name != f"{run}.resources.json" or payload.get("enabled") is not True:
        raise ValueError(f"Invalid or disabled resource summary: {path}")
    log_suffix = ".powerMH.vllm.log" if baseline else ".subtreePrefetch.vllm.log"
    log_path = path.with_name(run + log_suffix)
    if "INFO: output saved to" not in log_path.read_text(encoding="utf-8"):
        raise ValueError(f"Run is incomplete: {log_path}")
    row = {
        "source_json": str(path),
        "source_log": str(log_path),
        "method": POWER_METHOD_LABEL if baseline else SUBTREE_METHOD_LABEL,
        "run": run,
        "rank_fn": str(payload.get("rank_fn", "")),
        "prefetch_budget": str(payload.get("prefetch_budget", "")),
    }
    for key in ("peak_kv_cache_occupancy", "prefix_cache_hit_rate"):
        value = payload.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Missing numeric {key} in {path}")
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{key} outside [0, 1] in {path}")
        row[key] = str(value)
        row[f"{key}_percent"] = str(100.0 * value)
    queries = payload["prefix_cache_queried_tokens"]
    hits = payload["prefix_cache_hit_tokens"]
    if not queries or not 0 <= hits <= queries or not math.isclose(
        payload["prefix_cache_hit_rate"], hits / queries, rel_tol=1e-12
    ):
        raise ValueError(f"Prefix-cache hit rate disagrees with token counts: {path}")
    for key in (
        "mcmc_steps", "num_blocks", "prefix_cache", "gpu_memory_utilization",
        "num_gpu_blocks", "prefix_cache_queried_tokens", "prefix_cache_hit_tokens",
        "kv_source", "prefix_source",
    ):
        row[key] = str(payload[key])
    return row


def read_resource_comparison(
    baseline_path: Path,
    *, include_excluded_ranks: bool = False,
    subtree_resource_dirs: tuple[Path, ...] = (),
) -> tuple[list[dict[str, str]], dict[str, str]]:
    """Match PreSTO resources to one PowerMH run's full configuration.

    PreSTO runs are read beside the baseline and from ``subtree_resource_dirs``,
    for matched runs that were written under a different date directory.
    """
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if baseline.get("method") not in {"power_mh", "entropycut_power_mh"}:
        raise ValueError(f"Expected a PowerMH resource summary: {baseline_path}")
    baseline_row = cache_source_row(baseline_path, baseline, baseline=True)
    resource_rows: list[dict[str, str]] = []
    directories = dict.fromkeys((baseline_path.parent, *subtree_resource_dirs))
    for path in sorted(p for d in directories for p in d.glob("*.resources.json")):
        if path == baseline_path:
            continue
        if strip_tokens(path.name.removesuffix(".resources.json")) != baseline["run"]:
            continue
        summary = parse_resource_summary(path)
        if is_excluded_rank(summary.rank_fn) and not include_excluded_ranks:
            continue
        for key in (
            "mcmc_steps", "num_blocks", "prefix_cache", "gpu_memory_utilization",
            "num_gpu_blocks", "kv_source", "prefix_source",
        ):
            if getattr(summary, key) != baseline[key]:
                raise ValueError(f"PowerMH and PreSTO differ in {key}: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        resource_rows.append(cache_source_row(path, payload, baseline=False))
    if not resource_rows:
        raise ValueError(f"No compatible PreSTO resources beside {baseline_path}")
    resource_rows.sort(key=lambda row: (row["rank_fn"], int(row["prefetch_budget"])))
    return resource_rows, baseline_row


def add_prefix_misses_per_answer(
    rows: list[dict[str, str]], *, resource_directory: Path
) -> None:
    """Normalize run-level cache misses by actual completed result rows."""
    for row in rows:
        source_json = Path(row["source_json"])
        if not source_json.is_absolute():
            source_json = resource_directory / source_json
        results_path = source_json.with_name(row["run"] + ".csv").resolve()
        answers = read_csv(results_path)
        if not answers or any(not answer.get("completion", "").strip() for answer in answers):
            raise ValueError(f"Expected nonempty completed answers in {results_path}")
        queries = int(row["prefix_cache_queried_tokens"])
        hits = int(row["prefix_cache_hit_tokens"])
        if not 0 <= hits <= queries:
            raise ValueError(f"Invalid prefix-cache token counts for {source_json}")
        misses = queries - hits
        row.update(
            source_results_csv=str(results_path),
            completed_answers=str(len(answers)),
            prefix_cache_miss_tokens=str(misses),
            prefix_cache_miss_tokens_per_answer=str(misses / len(answers)),
            prefix_cache_miss_k_tokens_per_answer=str(misses / len(answers) / 1000.0),
        )


def add_eviction_rates(
    rows: list[dict[str, str]], *, resource_directory: Path
) -> None:
    """Read recorded eviction rates and validate their physical-block counts."""
    for row in rows:
        path = Path(row["source_json"])
        if not path.is_absolute():
            path = resource_directory / path
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("run") != row["run"]:
            raise ValueError(f"Resource run does not match source row: {path}")
        rate = payload.get("kv_cache_eviction_rate")
        if (
            isinstance(rate, bool) or not isinstance(rate, (int, float))
            or not 0.0 <= rate <= 1.0
            or payload.get("eviction_source") != "block-pool-hook"
        ):
            raise ValueError(f"No valid measured KV-cache eviction rate in {path}")
        evicted = payload.get("kv_cache_evicted_blocks")
        allocated = payload.get("kv_cache_allocated_blocks")
        if (
            type(evicted) is not int or type(allocated) is not int
            or allocated <= 0 or not 0 <= evicted <= allocated
            or not math.isclose(rate, evicted / allocated, rel_tol=1e-12)
        ):
            raise ValueError(f"Eviction rate disagrees with physical-block counts: {path}")
        row.update(
            kv_cache_eviction_rate=str(rate),
            kv_cache_eviction_rate_percent=str(100.0 * rate),
            kv_cache_evicted_blocks=str(evicted),
            kv_cache_allocated_blocks=str(allocated),
            eviction_source=payload["eviction_source"],
        )


def add_peak_gpu_memory_deltas(
    rows: list[dict[str, str]], *, resource_directory: Path
) -> None:
    """Read generation-memory deltas measured above the post-load baseline."""
    for row in rows:
        path = Path(row["source_json"])
        if not path.is_absolute():
            path = resource_directory / path
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("run") != row["run"]:
            raise ValueError(f"Resource run does not match source row: {path}")
        keys = (
            "loaded_gpu_memory_gib", "peak_llm_generate_gpu_memory_gib",
            "peak_llm_generate_gpu_memory_delta_gib",
        )
        values = [payload.get(key) for key in keys]
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0
            for value in values
        ) or payload.get("generation_gpu_source") in (None, "unavailable"):
            raise ValueError(f"No valid measured peak GPU memory delta in {path}")
        loaded, peak, delta = values
        if not math.isclose(delta, max(0.0, peak - loaded), rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"GPU memory delta disagrees with peak and loaded memory: {path}")
        row.update({key: str(value) for key, value in zip(keys, values, strict=True)})
        row["peak_llm_generate_gpu_memory_delta_mb"] = str(delta * 2**30 / 1_000_000)
        row["generation_gpu_source"] = payload["generation_gpu_source"]


def numeric_series(
    rows: list[dict[str, str]],
    *,
    rank_key: str,
    value_key: str,
) -> dict[tuple[str, int], float]:
    """Index available numeric values by traversal rank and prefetch budget."""
    series: dict[tuple[str, int], float] = {}
    for row in rows:
        value = row.get(value_key, "")
        if value in ("", None):
            continue
        key = (row[rank_key], int(row["prefetch_budget"]))
        if key in series:
            raise ValueError(f"Duplicate source-data point: {key}")
        series[key] = float(value)
    if not series:
        raise ValueError(f"No values found for {value_key} in source table")
    return series


def rank_offsets(rank_names: list[str]) -> dict[str, float]:
    """Dodge coincident markers while preserving the quantitative x scale."""
    if len(rank_names) == 1:
        return {rank_names[0]: 0.0}
    step = X_DODGE_WIDTH / (len(rank_names) - 1)
    return {
        rank_name: -X_DODGE_WIDTH / 2.0 + index * step
        for index, rank_name in enumerate(rank_names)
    }


def draw_lines(
    ax: Axes,
    series: dict[tuple[str, int], float],
    rank_names: list[str],
    offsets: dict[str, float],
) -> list[float]:
    """Draw every traversal rank with the shared marker-and-line encoding."""
    values: list[float] = []
    for rank_index, rank_name in enumerate(rank_names):
        points = sorted(
            (budget, value)
            for (rank, budget), value in series.items()
            if rank == rank_name
        )
        if not points:
            continue
        values.extend(value for _, value in points)
        ax.plot(
            [budget + offsets[rank_name] for budget, _ in points],
            [value for _, value in points],
            color=RANK_COLORS[rank_name],
            linewidth=LINE_WIDTH,
            marker=RANK_MARKERS[rank_index % len(RANK_MARKERS)],
            markersize=MARKER_SIZE,
            markerfacecolor="white",
            markeredgewidth=MARKER_EDGE_WIDTH,
            zorder=3,
        )
    return values


def draw_panel_label(ax: Axes, label: str, *, x: float, y: float) -> None:
    """Draw one bold panel tag at the figure-wide text size."""
    if plt.rcParams["text.usetex"]:
        label = rf"\textbf{{{label}}}"
        fontweight = None
    else:
        fontweight = "bold"
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=FIGURE_FONT_SIZE,
        fontweight=fontweight,
        ha="left",
        va="bottom",
    )


def style_panel(
    ax: Axes,
    *,
    index: int,
    title: str,
    y_label: str,
    budgets: list[int],
    y_limits: tuple[float, float],
    hit_rate_tick_step: float = 0.5,
    panel_label_x: float | None = None,
) -> None:
    """Apply the same quantitative styling to each horizontal panel."""
    ax.set_title(title, loc="left", pad=5.0)
    ax.set_ylabel(y_label)
    ax.set_xticks(budgets)
    ax.set_xlim(min(budgets) - 0.75, max(budgets) + 0.75)
    ax.set_ylim(*y_limits)
    # A fixed hit-rate step can leave a narrow range (e.g. 98.3-98.6%) without
    # any tick; fall back to automatic ticks when fewer than two would show.
    low, high = y_limits
    fixed_ticks = math.floor(high / hit_rate_tick_step) - math.ceil(low / hit_rate_tick_step) + 1
    ax.yaxis.set_major_locator(
        MultipleLocator(hit_rate_tick_step)
        if index == 1 and fixed_ticks >= 2
        else MaxNLocator(nbins=5)
    )
    ax.grid(False)
    ax.set_axisbelow(True)
    if panel_label_x is None:
        panel_label_x = -0.08 if index in (0, 2) else -0.10
    draw_panel_label(ax, panel_letter(index), x=panel_label_x, y=1.02)
    style_quantitative_axis(ax)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())


def occupancy_limits(values: list[float]) -> tuple[float, float]:
    """Show occupancy from 0.2 percent with modest headroom."""
    return 0.2, max(values) * 1.10


def workload_limits(values: list[float]) -> tuple[float, float]:
    """Show nonnegative token workload from zero with modest headroom."""
    return 0.0, max(max(values) * 1.10, 1.0)


def gpu_memory_delta_limits(values: list[float]) -> tuple[float, float]:
    """Frame MB deltas with padding and bounds rounded to half an MB."""
    return (
        max(0.0, math.floor((min(values) - 0.2) / 0.5) * 0.5),
        math.ceil((max(values) + 0.2) / 0.5) * 0.5,
    )


def eviction_rate_limits(values: list[float]) -> tuple[float, float]:
    """Show eviction percentages from zero to a round bound within 100 percent."""
    return 0.0, min(100.0, max(10.0, math.ceil(max(values) / 10.0) * 10.0))


def banded_limits(values: list[float], *, padding: float = 0.0) -> tuple[float, float]:
    """Bound hit rates to the nearest compact axis-step interval."""
    return (
        max(
            math.floor((min(values) - padding) / HIT_RATE_RANGE_STEP) * HIT_RATE_RANGE_STEP,
            0.0,
        ),
        min(
            math.ceil((max(values) + padding) / HIT_RATE_RANGE_STEP) * HIT_RATE_RANGE_STEP,
            100.0,
        ),
    )


def legend_handles(
    rank_names: list[str], *, include_baseline: bool = False
) -> list[Line2D]:
    """Build the one shared line-style legend."""
    handles = [
        Line2D(
            [0],
            [0],
            color=RANK_COLORS[rank_name],
            linewidth=LINE_WIDTH,
            marker=RANK_MARKERS[index % len(RANK_MARKERS)],
            markersize=MARKER_SIZE,
            markerfacecolor="white",
            markeredgewidth=MARKER_EDGE_WIDTH,
            label=(
                f"{SUBTREE_METHOD_LABEL} ({rank_label(rank_name)})"
                if include_baseline else rank_label(rank_name)
            ),
        )
        for index, rank_name in enumerate(rank_names)
    ]
    if include_baseline:
        handles.append(
            Line2D(
                [0], [0], color=DARK_GRAY, linewidth=LINE_WIDTH,
                linestyle="--", label=POWER_METHOD_LABEL,
            )
        )
    return handles


def draw_figure(
    resource_csv: Path,
    endpoint_csv: Path,
    output: Path,
    *,
    baseline_resource_json: Path | None = None,
    dataset_label: str = "MATH500",
    model_label: str = "Qwen2.5-7B",
    hit_rate_tick_step: float = 0.5,
    first_metric: str = "peak-occupancy",
    third_metric: str = "miss-tokens-per-answer",
    include_excluded_ranks: bool = False,
    subtree_resource_dirs: tuple[Path, ...] = (),
) -> Path:
    """Render cache capacity, reuse, and the requested third cache metric."""
    if not math.isfinite(hit_rate_tick_step) or hit_rate_tick_step <= 0:
        raise ValueError("hit_rate_tick_step must be finite and positive")
    baseline_row = None
    if baseline_resource_json is None:
        resource_rows = read_csv(resource_csv)
    else:
        resource_rows, baseline_row = read_resource_comparison(
            baseline_resource_json, include_excluded_ranks=include_excluded_ranks,
            subtree_resource_dirs=subtree_resource_dirs,
        )
    source_rows = resource_rows if baseline_row is None else [baseline_row, *resource_rows]
    if first_metric == "peak-gpu-memory-delta":
        add_peak_gpu_memory_deltas(source_rows, resource_directory=resource_csv.parent)
        first_key = "peak_llm_generate_gpu_memory_delta_mb"
        first_panel = (
            "", "peak GPU memory delta (MB)",
            numeric_series(resource_rows, rank_key="rank_fn", value_key=first_key),
            gpu_memory_delta_limits,
        )
    elif first_metric == "peak-occupancy":
        first_key = "peak_kv_cache_occupancy_percent"
        first_panel = (
            "", r"occupied KV-block pool (\%)",
            numeric_series(resource_rows, rank_key="rank_fn", value_key=first_key),
            occupancy_limits,
        )
    else:
        raise ValueError(f"Unknown first metric: {first_metric}")
    if third_metric == "eviction-rate":
        add_eviction_rates(source_rows, resource_directory=resource_csv.parent)
        third_key = "kv_cache_eviction_rate_percent"
        third_panel = (
            "", r"KV-cache eviction rate (\%)",
            numeric_series(resource_rows, rank_key="rank_fn", value_key=third_key),
            eviction_rate_limits,
        )
    elif third_metric == "miss-tokens-per-answer":
        add_prefix_misses_per_answer(source_rows, resource_directory=resource_csv.parent)
        third_key = "prefix_cache_miss_k_tokens_per_answer"
        third_panel = (
            "",
            "prefix-cache miss tokens\n" + r"per answer ($10^3$)",
            numeric_series(resource_rows, rank_key="rank_fn", value_key=third_key),
            workload_limits,
        )
    elif third_metric == "peak-occupancy":
        endpoint_rows = (
            read_csv(endpoint_csv) if baseline_row is None else
            [{**row, "rank": row["rank_fn"]} for row in resource_rows]
        )
        third_key = "peak_kv_cache_occupancy_percent"
        third_panel = (
            "", r"peak KV-cache occupancy (\%)",
            numeric_series(endpoint_rows, rank_key="rank", value_key=third_key),
            occupancy_limits,
        )
    else:
        raise ValueError(f"Unknown third metric: {third_metric}")
    baseline_values = (
        (None, None, None) if baseline_row is None else (
            float(baseline_row[first_key]),
            float(baseline_row["prefix_cache_hit_rate_percent"]),
            float(baseline_row[third_key]),
        )
    )
    panels = (
        first_panel,
        (
            "",
            r"prefix-cache token hit rate (\%)",
            numeric_series(
                resource_rows,
                rank_key="rank_fn",
                value_key="prefix_cache_hit_rate_percent",
            ),
            lambda values: banded_limits(
                values, padding=HIT_RATE_RANGE_STEP if baseline_row is not None else 0.0,
            ),
        ),
        third_panel,
    )
    all_keys = [set(series) for _, _, series, _ in panels]
    if any(keys != all_keys[0] for keys in all_keys[1:]):
        raise ValueError("The two source tables do not contain the same design grid")
    rank_names = (
        sorted(
            {rank for rank, _ in all_keys[0]},
            key=lambda rank: (RANK_ORDER.get(rank, len(RANK_ORDER)), rank),
        )
        if include_excluded_ranks else sort_rank_names(rank for rank, _ in all_keys[0])
    )
    # The shared palette has no bfs_accept_first entry; draw it in the purple the
    # EntropyCut sweep uses, or gray when longest_path_first already holds purple.
    if "bfs_accept_first" in rank_names and "bfs_accept_first" not in RANK_COLORS:
        taken = {RANK_COLORS[rank] for rank in rank_names if rank in RANK_COLORS}
        RANK_COLORS["bfs_accept_first"] = GRAY if PURPLE in taken else PURPLE
    unknown_ranks = sorted(set(rank_names) - RANK_COLORS.keys())
    if unknown_ranks:
        raise ValueError("No shared rank color for: " + ", ".join(unknown_ranks))
    budgets = sorted({budget for _, budget in all_keys[0]})
    offsets = rank_offsets(rank_names)

    multirow_legend = baseline_row is not None and len(rank_names) > 2
    figure = plt.figure(
        figsize=(FIGURE_SIZE[0], 4.2) if multirow_legend else FIGURE_SIZE,
    )
    grid = figure.add_gridspec(
        1, 5, width_ratios=(1, 0.30, 1, 0.30, 1), wspace=0
    )
    axes = [figure.add_subplot(grid[0, 0])]
    axes.extend(
        figure.add_subplot(grid[0, column], sharex=axes[0]) for column in (2, 4)
    )
    for index, (ax, (title, y_label, series, limit_fn)) in enumerate(
        zip(axes, panels, strict=True)
    ):
        values = draw_lines(ax, series, rank_names, offsets)
        if (baseline_value := baseline_values[index]) is not None:
            ax.axhline(
                baseline_value, color=DARK_GRAY, linestyle="--",
                linewidth=LINE_WIDTH, zorder=2,
            )
            values.append(baseline_value)
        style_panel(
            ax,
            index=index,
            title=title,
            y_label=y_label,
            budgets=budgets,
            y_limits=limit_fn(values),
            hit_rate_tick_step=hit_rate_tick_step,
            panel_label_x=(
                0.0 if index == 0 and first_metric == "peak-gpu-memory-delta" else None
            ),
        )
        if index == 0 and first_metric == "peak-gpu-memory-delta":
            ax.yaxis.set_major_locator(
                MaxNLocator(nbins=4, steps=(1, 2, 5, 10), min_n_ticks=3)
            )

    title = f"dataset: {dataset_label}, base LLM: {model_label}"
    if baseline_row is None:
        title += f", Method: {SUBTREE_METHOD_LABEL}"
    figure.suptitle(title, y=0.985)
    figure.supxlabel(PREFETCH_BUDGET_AXIS_LABEL, y=0.055, fontsize=15)
    handles = legend_handles(rank_names, include_baseline=baseline_row is not None)
    legend = figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        fontsize=FIGURE_FONT_SIZE,
        ncol=2 if multirow_legend else len(handles),
        frameon=False,
        handlelength=1.7,
        handletextpad=0.4,
        columnspacing=0.9,
    )
    # Long method labels and a fourth legend row need more room than the
    # original fixed top margin. Measure the actual legend before placing axes.
    figure.canvas.draw()
    legend_bottom = legend.get_window_extent().transformed(figure.transFigure.inverted()).y0
    figure.subplots_adjust(
        left=0.055,
        right=0.995,
        bottom=0.17,
        top=min(0.70 if multirow_legend else 0.74, legend_bottom - 0.065),
        wspace=0,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    result = save_figure(figure, output, pad_inches=0.02)
    if (
        baseline_row is not None or third_metric != "peak-occupancy"
        or first_metric == "peak-gpu-memory-delta"
    ):
        write_dict_rows(output.with_suffix(".csv"), source_rows)
    return result


def build_parser() -> argparse.ArgumentParser:
    """Build the standalone combined-figure CLI."""
    parser = argparse.ArgumentParser(
        description="Draw resource and endpoint cache metrics in one line row."
    )
    parser.add_argument("--resource-csv", type=Path, default=DEFAULT_RESOURCE_CSV)
    parser.add_argument("--endpoint-csv", type=Path, default=DEFAULT_ENDPOINT_CSV)
    parser.add_argument(
        "--baseline-resource-json", type=Path,
        help="Load this PowerMH summary and compatible sibling PreSTO resources instead of CSVs.",
    )
    parser.add_argument(
        "--subtree-resource-dir", type=Path, nargs="+", default=[],
        help="Also read matching PreSTO resource JSONs from these directories.",
    )
    parser.add_argument("--dataset-label")
    parser.add_argument("--model-label")
    parser.add_argument("--hit-rate-tick-step", type=float, default=0.5)
    parser.add_argument(
        "--include-excluded-ranks", action="store_true",
        help="Include available ranks normally omitted by the shared comparison policy.",
    )
    parser.add_argument(
        "--first-metric", choices=("peak-gpu-memory-delta", "peak-occupancy"),
        default="peak-occupancy",
        help="Panel (a): measured generation-memory delta in MB, or KV-block occupancy.",
    )
    parser.add_argument(
        "--third-metric",
        choices=("eviction-rate", "miss-tokens-per-answer", "peak-occupancy"),
        default="miss-tokens-per-answer",
        help="Panel (c): measured eviction rate, per-answer cache misses, or legacy occupancy.",
    )
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    """Render the combined line-style cache figure."""
    parser = build_parser()
    args = parser.parse_args()
    if args.baseline_resource_json is not None and not all(
        (args.dataset_label, args.model_label, args.output)
    ):
        parser.error("Baseline comparisons require --dataset-label, --model-label, and --output")
    output = draw_figure(
        args.resource_csv.resolve(),
        args.endpoint_csv.resolve(),
        (args.output or DEFAULT_OUTPUT).resolve(),
        baseline_resource_json=(
            args.baseline_resource_json.resolve() if args.baseline_resource_json else None
        ),
        dataset_label=args.dataset_label or "MATH500",
        model_label=args.model_label or "Qwen2.5-7B",
        hit_rate_tick_step=args.hit_rate_tick_step,
        first_metric=args.first_metric,
        third_metric=args.third_metric,
        include_excluded_ranks=args.include_excluded_ranks,
        subtree_resource_dirs=tuple(path.resolve() for path in args.subtree_resource_dir),
    )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
