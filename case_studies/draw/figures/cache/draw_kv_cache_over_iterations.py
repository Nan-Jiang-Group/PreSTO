#!/usr/bin/env python3
"""Plot KV-cache usage over MH iterations for a PowerMH run and its PreSTO-PowerMH run.

Run with:

    /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/.venv/bin/python \
        /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/draw/draw_kv_cache_over_iterations.py \
        --power-mh-log /absolute/path/to/run.powerMH.vllm.log \
        --subtree-log /absolute/path/to/run.prefetch-budget-20.rank-bfs_accept_first....subtreePrefetch.vllm.log \
        --dataset-label MATH500 --model-label Qwen3-4B \
        --output /absolute/path/to/figure.pdf

Add ``--baseline-label EntropyCut --subtree-label PreSTO-EntropyCut`` for EntropyCut logs.

The data are vLLM's periodic engine lines (about every 10 s), not the run-level
resource summary. Each engine line is assigned to the MH iteration in progress
when it was printed: the next ``MH step k/K`` line of the same question. Lines
printed after a question's last step are dropped. Iterations are pooled into
``--bin-width`` windows across all questions of the run.

  (a) occupied KV cache = logged ``GPU KV cache usage`` x the log's
      ``GPU KV cache size`` in tokens, so runs with different pool sizes compare
      directly; mean with the interquartile band.
  (b) KV-cache eviction rate = evicted / allocated physical blocks summed over
      the engine intervals in the window. Older logs without the eviction field
      show the panel as not logged.

vLLM's logged ``Prefix cache hit rate`` is not plotted: in these runs it stays at
0% for every PowerMH run and every Gemma run while the resource probe measures
up to 94% token hits, so it does not track the run's prefix reuse. It is kept in
the CSV as ``vllm_logged_hit_rate_percent_mean`` for reference only.

A CSV with the per-window values and sample counts is written beside the PDF.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from case_studies.draw.common.figure_io import save_figure, write_dict_rows
from case_studies.draw.common.plot_helpers import panel_letter
from case_studies.draw.common.rank_plotting import style_quantitative_axis
from case_studies.plot_config import (
    DARK_GRAY,
    POWER_METHOD_LABEL,
    PURPLE,
    SUBTREE_METHOD_LABEL,
    apply_plot_style,
)

apply_plot_style()

FIGURE_SIZE = (8.2, 3.6)
FIGURE_FONT_SIZE = 14
LINE_WIDTH = 1.35
MARKER_SIZE = 5.2
MARKER_EDGE_WIDTH = 1.0
BAND_ALPHA = 0.18
DEFAULT_BIN_WIDTH = 5
plt.rcParams.update(
    {
        key: FIGURE_FONT_SIZE
        for key in (
            "font.size", "axes.titlesize", "axes.labelsize", "xtick.labelsize",
            "ytick.labelsize", "legend.fontsize", "figure.titlesize", "figure.labelsize",
        )
    }
)

ENGINE_PATTERN = re.compile(
    r"GPU KV cache usage: (?P<usage>[\d.]+)%, Prefix cache hit rate: (?P<hit>[\d.]+)%"
    r"(?:, KV cache eviction rate: (?:[\d.]+% \((?P<evicted>\d+)/(?P<allocated>\d+) blocks\)|n/a))?"
)
MH_STEP_PATTERN = re.compile(r"MH step (?P<step>\d+)/(?P<steps>\d+) took")
KV_SIZE_PATTERN = re.compile(r"GPU KV cache size: (?P<tokens>[\d,]+) tokens")
# PreSTO prints an IDX header per question; PowerMH prints its prompt settings.
QUESTION_START_PATTERN = re.compile(r"^(?:IDX \d+ .*QUESTION:|\[power MH\] INFO: prompts: )")
COMPLETION_MARKER = "INFO: output saved to"


@dataclass(frozen=True)
class EngineSample:
    """One periodic engine line and the MH iteration it was logged during."""

    question: int
    iteration: int
    usage_percent: float
    hit_rate_percent: float
    evicted_blocks: int | None
    allocated_blocks: int | None


@dataclass(frozen=True)
class RunTrace:
    """All engine samples of one completed run with its KV pool size."""

    log_path: Path
    kv_cache_tokens: int
    mcmc_steps: int
    samples: tuple[EngineSample, ...]
    dropped_samples: int
    has_eviction: bool


def parse_run(log_path: Path) -> RunTrace:
    """Assign each engine line of a completed run to its MH iteration."""
    text = log_path.read_text(encoding="utf-8", errors="replace")
    if COMPLETION_MARKER not in text:
        raise ValueError(f"Run is incomplete: {log_path}")
    sizes = {int(m.group("tokens").replace(",", "")) for m in KV_SIZE_PATTERN.finditer(text)}
    if len(sizes) != 1:
        raise ValueError(f"Expected one GPU KV cache size in {log_path}, found {sorted(sizes)}")
    samples: list[EngineSample] = []
    pending: list[re.Match[str]] = []
    question = -1
    dropped = 0
    steps_seen: set[int] = set()
    for line in text.splitlines():
        if QUESTION_START_PATTERN.match(line):
            dropped += len(pending)
            pending = []
            question += 1
            continue
        if question < 0:
            continue
        if engine := ENGINE_PATTERN.search(line):
            pending.append(engine)
            continue
        if step := MH_STEP_PATTERN.search(line):
            steps_seen.add(int(step.group("steps")))
            for match in pending:
                evicted = match.group("evicted")
                allocated = match.group("allocated")
                samples.append(
                    EngineSample(
                        question=question,
                        iteration=int(step.group("step")),
                        usage_percent=float(match.group("usage")),
                        hit_rate_percent=float(match.group("hit")),
                        evicted_blocks=int(evicted) if evicted is not None else None,
                        allocated_blocks=int(allocated) if allocated is not None else None,
                    )
                )
            pending = []
    dropped += len(pending)
    if len(steps_seen) != 1 or not samples:
        raise ValueError(f"No MH-step engine samples in {log_path}")
    return RunTrace(
        log_path=log_path,
        kv_cache_tokens=sizes.pop(),
        mcmc_steps=steps_seen.pop(),
        samples=tuple(samples),
        dropped_samples=dropped,
        has_eviction=any(s.allocated_blocks is not None for s in samples),
    )


def window_rows(trace: RunTrace, method: str, bin_width: int) -> list[dict[str, object]]:
    """Summarize one run per iteration window."""
    rows: list[dict[str, object]] = []
    for start in range(0, trace.mcmc_steps, bin_width):
        window = [s for s in trace.samples if start <= s.iteration < start + bin_width]
        if not window:
            continue
        tokens = np.array([s.usage_percent / 100 * trace.kv_cache_tokens / 1000 for s in window])
        evicted = sum(s.evicted_blocks for s in window if s.allocated_blocks is not None)
        allocated = sum(s.allocated_blocks for s in window if s.allocated_blocks is not None)
        rows.append(
            {
                "method": method,
                "log_path": str(trace.log_path),
                "kv_cache_tokens": trace.kv_cache_tokens,
                "iteration_start": start,
                "iteration_end": min(start + bin_width, trace.mcmc_steps) - 1,
                "iteration_center": start + (min(bin_width, trace.mcmc_steps - start) - 1) / 2,
                "engine_samples": len(window),
                "questions": len({s.question for s in window}),
                "occupied_k_tokens_mean": float(tokens.mean()),
                "occupied_k_tokens_q25": float(np.percentile(tokens, 25)),
                "occupied_k_tokens_q75": float(np.percentile(tokens, 75)),
                "usage_percent_mean": float(np.mean([s.usage_percent for s in window])),
                "vllm_logged_hit_rate_percent_mean": float(np.mean([s.hit_rate_percent for s in window])),
                "evicted_blocks": evicted if trace.has_eviction else "",
                "allocated_blocks": allocated if trace.has_eviction else "",
                "eviction_rate_percent": (
                    100 * evicted / allocated if trace.has_eviction and allocated else ""
                ),
            }
        )
    return rows


def draw_series(ax: Axes, rows, key: str, *, color: str, marker: str | None, linestyle: str, band: bool) -> None:
    """Draw one method's windowed mean, optionally with its interquartile band."""
    points = [row for row in rows if row[key + ("_mean" if band else "")] != ""]
    x = [row["iteration_center"] for row in points]
    y = [row[key + ("_mean" if band else "")] for row in points]
    if band:
        ax.fill_between(
            x, [row[key + "_q25"] for row in points], [row[key + "_q75"] for row in points],
            color=color, alpha=BAND_ALPHA, linewidth=0, zorder=1,
        )
    ax.plot(
        x, y, color=color, linestyle=linestyle, linewidth=LINE_WIDTH, marker=marker,
        markersize=MARKER_SIZE, markerfacecolor="white", markeredgewidth=MARKER_EDGE_WIDTH, zorder=3,
    )


def draw_panel_label(ax: Axes, label: str) -> None:
    """Draw one bold panel tag above the axes' top-left corner."""
    text = rf"\textbf{{{label}}}" if plt.rcParams["text.usetex"] else label
    ax.text(0.0, 1.03, text, transform=ax.transAxes, ha="left", va="bottom",
            fontweight=None if plt.rcParams["text.usetex"] else "bold")


def draw_figure(
    baseline: RunTrace,
    subtree: RunTrace,
    output: Path,
    *,
    dataset_label: str,
    model_label: str,
    baseline_label: str,
    subtree_label: str,
    bin_width: int,
) -> Path:
    """Render the two iteration panels and their source CSV."""
    if baseline.mcmc_steps != subtree.mcmc_steps:
        raise ValueError("PowerMH and PreSTO runs use different MH step counts")
    rows = {
        baseline_label: window_rows(baseline, baseline_label, bin_width),
        subtree_label: window_rows(subtree, subtree_label, bin_width),
    }
    styles = {
        baseline_label: {"color": DARK_GRAY, "marker": None, "linestyle": "--"},
        subtree_label: {"color": PURPLE, "marker": "o", "linestyle": "-"},
    }
    figure, axes = plt.subplots(1, 2, figsize=FIGURE_SIZE, sharex=True)
    panels = (
        ("occupied_k_tokens", r"occupied KV cache ($10^3$ tokens)", True),
        ("eviction_rate_percent", r"KV-cache eviction rate (\%)", False),
    )
    for index, (ax, (key, label, band)) in enumerate(zip(axes, panels, strict=True)):
        for method, method_rows in rows.items():
            trace = baseline if method == baseline_label else subtree
            if key == "eviction_rate_percent" and not trace.has_eviction:
                continue
            draw_series(ax, method_rows, key, band=band, **styles[method])
        if key == "eviction_rate_percent" and not (baseline.has_eviction or subtree.has_eviction):
            ax.text(0.5, 0.5, "not logged", transform=ax.transAxes, ha="center", va="center", color="0.45")
            ax.set_yticks([])
        else:
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            # Leave a sliver below zero so a run that stays at exactly 0% remains visible.
            top = max(ax.get_ylim()[1], 1.0)
            ax.set_ylim(-0.03 * top, top)
        ax.set_ylabel(label)
        ax.set_xlim(-2, baseline.mcmc_steps + 1)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        style_quantitative_axis(ax)
        draw_panel_label(ax, panel_letter(index))
    figure.suptitle(f"dataset: {dataset_label}, base LLM: {model_label}", y=0.985)
    figure.supxlabel("MH iteration", y=0.055, fontsize=15)
    handles = [
        Line2D([0], [0], linewidth=LINE_WIDTH, markersize=MARKER_SIZE, markerfacecolor="white",
               markeredgewidth=MARKER_EDGE_WIDTH, label=method, **styles[method])
        for method in (subtree_label, baseline_label)
    ]
    figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.945), ncol=2,
                  frameon=False, handlelength=1.7, handletextpad=0.4, columnspacing=0.9)
    figure.subplots_adjust(left=0.09, right=0.995, bottom=0.19, top=0.72, wspace=0.30)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_figure(figure, output, pad_inches=0.02)
    write_dict_rows(output.with_suffix(".csv"), [*rows[baseline_label], *rows[subtree_label]])
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot KV-cache metrics over MH iterations.")
    parser.add_argument("--power-mh-log", type=Path, required=True)
    parser.add_argument("--subtree-log", type=Path, required=True)
    parser.add_argument("--dataset-label", required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--baseline-label", default=POWER_METHOD_LABEL)
    parser.add_argument("--subtree-label", default=f"{SUBTREE_METHOD_LABEL} (bfs accept first)")
    parser.add_argument("--bin-width", type=int, default=DEFAULT_BIN_WIDTH)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.bin_width < 1:
        parser.error("--bin-width must be positive")
    baseline = parse_run(args.power_mh_log.resolve())
    subtree = parse_run(args.subtree_log.resolve())
    output = draw_figure(
        baseline, subtree, args.output.resolve(),
        dataset_label=args.dataset_label, model_label=args.model_label,
        baseline_label=args.baseline_label, subtree_label=args.subtree_label,
        bin_width=args.bin_width,
    )
    for trace in (baseline, subtree):
        print(f"{trace.log_path.name}: {len(trace.samples)} samples, "
              f"{trace.dropped_samples} after a question's last step dropped")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
