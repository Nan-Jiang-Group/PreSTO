#!/usr/bin/env python3
"""Plot PreSTO-PowerMH KV-cache usage during MH iterations against the prefetch budget.

Run with:

    /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/.venv/bin/python \
        /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/draw/draw_kv_cache_over_budget.py \
        --directory /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/logs/lcb_v6/2026-09-07/vllm \
        --run-config dataset-lcb_v6.model-qwen3.5-9b.alpha4.0.steps100.blocks-1.samples20.maxnew1024.seed10086 \
        --power-mh-log /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/logs/lcb_v6/2026-09-07/vllm/dataset-lcb_v6.model-qwen3.5-9b.alpha4.0.steps100.blocks-1.samples20.maxnew1024.seed10086.powerMH.vllm.log \
        --dataset-label 'LCB V6' --model-label Qwen3.5-9B --output /absolute/path/to/figure.pdf

Every completed PreSTO log in ``--directory`` whose shared settings equal
``--run-config`` becomes one point per traversal rule and budget. Omit
``--power-mh-log`` when no PowerMH run logged engine samples. Add
``--include-excluded-ranks`` to keep longest_first, which the comparison figures
drop by default.

Each run's values come from vLLM's periodic engine lines assigned to MH
iterations as in ``draw_kv_cache_over_iterations.py``, not from the run-level
resource summary. Samples are first averaged within each MH iteration, so every
iteration counts once regardless of how long it ran.

  (a) mean occupied KV cache over MH iterations (logged usage x the log's KV
      pool size in tokens); bars span the 25th-75th percentile of iterations.
  (b) peak occupied KV cache over all engine samples of the run.
  (c) eviction rate (evicted / allocated blocks over the run), only when the
      logs record it.

A CSV with one row per run is written beside the PDF.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from case_studies.draw.common.figure_io import save_figure, write_dict_rows
from case_studies.draw.common.rank_plotting import PREFETCH_BUDGET_AXIS_LABEL, style_quantitative_axis
from case_studies.draw.figures.cache.draw_combined_cache_metrics import (
    LINE_WIDTH,
    MARKER_EDGE_WIDTH,
    MARKER_SIZE,
    RANK_MARKERS,
    draw_panel_label,
    rank_offsets,
)
from case_studies.draw.figures.cache.draw_kv_cache_over_iterations import COMPLETION_MARKER, RunTrace, parse_run
from case_studies.extract.run_naming import (
    RANK_ORDER,
    is_excluded_rank,
    prefetch_budget_of,
    rank_label,
    shared_run_config_of,
)
from case_studies.plot_config import (
    DARK_GRAY,
    GRAY,
    POWER_METHOD_LABEL,
    PURPLE,
    SUBTREE_METHOD_LABEL,
    TRANSITION_RANK_COLORS,
)

FIGURE_HEIGHT = 4.6
PANEL_WIDTH = 4.4


def iteration_means(trace: RunTrace) -> np.ndarray:
    """Mean occupied KV cache (10^3 tokens) of each MH iteration that has samples."""
    by_iteration: dict[int, list[float]] = defaultdict(list)
    for sample in trace.samples:
        by_iteration[sample.iteration].append(sample.usage_percent)
    return np.array(
        [np.mean(values) / 100 * trace.kv_cache_tokens / 1000 for values in by_iteration.values()]
    )


def run_row(trace: RunTrace, method: str, rank: str, budget: int | str) -> dict[str, object]:
    """Summarize one run over its MH iterations."""
    means = iteration_means(trace)
    evicted = sum(s.evicted_blocks for s in trace.samples if s.allocated_blocks is not None)
    allocated = sum(s.allocated_blocks for s in trace.samples if s.allocated_blocks is not None)
    return {
        "method": method,
        "rank_fn": rank,
        "prefetch_budget": budget,
        "log_path": str(trace.log_path),
        "kv_cache_tokens": trace.kv_cache_tokens,
        "engine_samples": len(trace.samples),
        "iterations_with_samples": len(means),
        "mean_occupied_k_tokens": float(means.mean()),
        "q25_occupied_k_tokens": float(np.percentile(means, 25)),
        "q75_occupied_k_tokens": float(np.percentile(means, 75)),
        "peak_occupied_k_tokens": max(s.usage_percent for s in trace.samples) / 100 * trace.kv_cache_tokens / 1000,
        "evicted_blocks": evicted if trace.has_eviction else "",
        "allocated_blocks": allocated if trace.has_eviction else "",
        "eviction_rate_percent": 100 * evicted / allocated if trace.has_eviction and allocated else "",
    }


def rank_colors(ranks: list[str]) -> dict[str, str]:
    """Shared rank colors, with bfs_accept_first in purple unless purple is taken."""
    colors = {rank: TRANSITION_RANK_COLORS[rank] for rank in ranks if rank in TRANSITION_RANK_COLORS}
    if "bfs_accept_first" in ranks:
        colors["bfs_accept_first"] = GRAY if PURPLE in colors.values() else PURPLE
    missing = sorted(set(ranks) - colors.keys())
    if missing:
        raise ValueError("No shared rank color for: " + ", ".join(missing))
    return colors


def draw_figure(
    rows: list[dict[str, object]],
    baseline: dict[str, object] | None,
    output: Path,
    *,
    dataset_label: str,
    model_label: str,
) -> Path:
    """Draw one line per traversal rule against the prefetch budget."""
    ranks = sorted({row["rank_fn"] for row in rows}, key=lambda r: (RANK_ORDER.get(r, len(RANK_ORDER)), r))
    colors = rank_colors(ranks)
    offsets = rank_offsets(ranks)
    budgets = sorted({row["prefetch_budget"] for row in rows})
    panels = [
        ("mean_occupied_k_tokens", "mean occupied KV cache\n" + r"over MH iterations ($10^3$ tokens)"),
        ("peak_occupied_k_tokens", "peak occupied KV cache\n" + r"($10^3$ tokens)"),
    ]
    if any(row["eviction_rate_percent"] != "" for row in rows):
        panels.append(("eviction_rate_percent", r"KV-cache eviction rate (\%)"))
    figure, axes = plt.subplots(
        1, len(panels), figsize=(PANEL_WIDTH * len(panels), FIGURE_HEIGHT), sharex=True, squeeze=False
    )
    for index, (ax, (key, label)) in enumerate(zip(axes[0], panels, strict=True)):
        for rank_index, rank in enumerate(ranks):
            points = sorted(
                (row for row in rows if row["rank_fn"] == rank and row[key] != ""),
                key=lambda row: row["prefetch_budget"],
            )
            x = [row["prefetch_budget"] + offsets[rank] for row in points]
            y = [row[key] for row in points]
            if key == "mean_occupied_k_tokens":
                ax.errorbar(
                    x, y,
                    yerr=[[row[key] - row["q25_occupied_k_tokens"] for row in points],
                          [row["q75_occupied_k_tokens"] - row[key] for row in points]],
                    fmt="none", ecolor=colors[rank], elinewidth=0.8, capsize=1.5, alpha=0.7, zorder=2,
                )
            ax.plot(
                x, y, color=colors[rank], linewidth=LINE_WIDTH,
                marker=RANK_MARKERS[rank_index % len(RANK_MARKERS)], markersize=MARKER_SIZE,
                markerfacecolor="white", markeredgewidth=MARKER_EDGE_WIDTH, zorder=3,
            )
        if baseline is not None and baseline[key] != "":
            ax.axhline(baseline[key], color=DARK_GRAY, linestyle="--", linewidth=LINE_WIDTH, zorder=2)
        ax.set_ylabel(label)
        ax.set_xticks(budgets)
        ax.set_xlim(min(budgets) - 0.75, max(budgets) + 0.75)
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        style_quantitative_axis(ax)
        draw_panel_label(ax, f"({chr(ord('a') + index)})", x=0.0, y=1.02)
    figure.suptitle(f"dataset: {dataset_label}, base LLM: {model_label}", y=0.985)
    figure.supxlabel(PREFETCH_BUDGET_AXIS_LABEL, y=0.01, fontsize=15)
    handles = [
        Line2D([0], [0], color=colors[rank], linewidth=LINE_WIDTH,
               marker=RANK_MARKERS[i % len(RANK_MARKERS)], markersize=MARKER_SIZE,
               markerfacecolor="white", markeredgewidth=MARKER_EDGE_WIDTH,
               label=f"{SUBTREE_METHOD_LABEL} ({rank_label(rank)})")
        for i, rank in enumerate(ranks)
    ]
    if baseline is not None:
        handles.append(Line2D([0], [0], color=DARK_GRAY, linestyle="--", linewidth=LINE_WIDTH, label=POWER_METHOD_LABEL))
    legend = figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.945), ncol=2,
                           frameon=False, handlelength=1.7, handletextpad=0.4, columnspacing=0.9)
    figure.canvas.draw()
    legend_bottom = legend.get_window_extent().transformed(figure.transFigure.inverted()).y0
    figure.subplots_adjust(left=0.11, right=0.995, bottom=0.19, top=legend_bottom - 0.08, wspace=0.36)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_figure(figure, output, pad_inches=0.02)
    write_dict_rows(output.with_suffix(".csv"), [*([baseline] if baseline else []), *rows])
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot KV-cache usage during MH iterations against the prefetch budget.")
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--run-config", required=True, help="Shared settings the PreSTO logs must match.")
    parser.add_argument("--power-mh-log", type=Path)
    parser.add_argument("--dataset-label", required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--include-excluded-ranks", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for log in sorted(args.directory.resolve().glob("*.subtreePrefetch.vllm.log")):
        if shared_run_config_of(log) != args.run_config:
            continue
        rank = log.name.split(".rank-")[1].split(".")[0]
        if is_excluded_rank(rank) and not args.include_excluded_ranks:
            continue
        if COMPLETION_MARKER not in log.read_text(encoding="utf-8", errors="replace"):
            print(f"skip incomplete {log.name}")
            continue
        rows.append(run_row(parse_run(log), SUBTREE_METHOD_LABEL, rank, prefetch_budget_of(log)))
    if not rows:
        parser.error("no completed PreSTO logs match --run-config")
    baseline = None
    if args.power_mh_log is not None:
        if shared_run_config_of(args.power_mh_log) != args.run_config:
            parser.error("--power-mh-log does not match --run-config")
        baseline = run_row(parse_run(args.power_mh_log.resolve()), POWER_METHOD_LABEL, "", "")
    output = draw_figure(rows, baseline, args.output.resolve(),
                         dataset_label=args.dataset_label, model_label=args.model_label)
    print(f"{len(rows)} PreSTO runs; wrote {output}")


if __name__ == "__main__":
    main()
