#!/usr/bin/env python3
"""Draw full-page likelihood/confidence figures per sampler: PowerMH vs PreSTO-PowerMH, EntropyCut vs PreSTO-EntropyCut.

Uses the pairs that draw_rescored_likelihood_and_confidence_summary.py keeps (same launcher, same rescoring,
prompt, and exact-prompt checks). Each sampler gets its own set of pages, laid out like the model-call figures in
exps/mh-transitions-vs-call-time, without the header line: one legend box, and one row per dataset and base LLM,
at most --rows-per-page rows per page. The two columns are density histograms of token log-likelihood and confidence
under p_0 for the baseline and one PreSTO run (N_pf=20, or N_pf=10 where 20 was not run); each panel prints both
methods' mean magnitudes (both statistics are never positive) with their standard deviations, and dashed lines mark
the means.

Writes likelihood-and-confidence.<uniform|entropy>-cut.pdf to --output-dir (.part<k> when a sampler spans pages).

Run:

    /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/.venv/bin/python \
        /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/draw/draw_rescored_likelihood_and_confidence_by_sampler.py
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from statistics import fmean, stdev

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from case_studies.draw.common.figure_io import save_figure
from case_studies.draw.figures.likelihood.draw_rescored_likelihood_and_confidence import budget_of, one_presto_budget
from case_studies.draw.figures.likelihood.draw_rescored_likelihood_and_confidence_summary import (
    DATASET_DISPLAY,
    DEFAULT_OUTPUT_DIR,
    LAUNCHER,
    MODEL_DISPLAY,
    kept_pairs,
    launcher_jobs,
    rescoring_agreement,
)
from case_studies.draw.figures.likelihood.plot_likelihood_and_confidence import (
    HIST_ALPHA,
    HIST_LINE_WIDTH,
    MEAN_LINE_WIDTH,
    draw_histogram,
    shared_bin_edges,
)
from case_studies.plot_config import BLACK, BLUE, COLOR_PALETTE, GRAY, GREEN, ORANGE, apply_plot_style

DATASET_ORDER = ("math500", "aime", "gpqa", "mmlu", "lcb_v6", "mbpp", "human_eval")
MODEL_ORDER = ("qwen3-4b", "qwen3-8b", "qwen3.5-4b", "qwen3.5-9b", "gemma-12b-it", "qwen")
FAMILIES = (
    ("uniform", "PowerMH", "PreSTO-PowerMH", "uniform cut"),
    ("entropy", "EntropyCut", "PreSTO-EntropyCut", "entropy cut"),
)
METRICS = (("log_likelihood", r"token log-likelihood under $p_0$"),
           ("confidence", r"token confidence under $p_0$"))
FIGURE_FONT_SIZE = 12.0
LEGEND_FONT_SIZE = 11.0
STAT_FONT_SIZE = 10.5
FIGURE_WIDTH = 11.0
ROW_HEIGHT = 1.55
TITLE_HEIGHT = 0.08
LEGEND_HEIGHT = 0.42
BOTTOM_HEIGHT = 0.55
BINS = 12
# The top of each panel is left to the per-method statistics; bars and mean lines stay below it.
# Bars fill the axes up to BAR_TOP (axes fraction); the method statistics sit above it, so the y range is the tallest bar
# divided by BAR_TOP, and mean lines stop there.
BAR_TOP = 0.55
MEAN_LINE_TOP = BAR_TOP
# Vertical step between the per-method statistics, in axes fractions.
STAT_LINE_STEP = 0.2


def method_color(label: str) -> str:
    """Baseline blue and PreSTO green, as in the model-call figures."""
    return BLUE if budget_of(label) is None else GREEN


def stat_label(label: str) -> str:
    """'PowerMH' or 'PreSTO-PowerMH ($N_pf=10$)' become the in-panel method names."""
    budget = budget_of(label)
    return label if budget is None else label.split(",")[0] + rf" ($N_{{\mathrm{{pf}}}}={budget}$)"


def draw_pair_panel(ax: Axes, groups: dict[str, list[dict]], key: str, x_label: str, panel: str) -> None:
    """Density histograms of one statistic for one pair, with each method's mean and SD printed on top."""
    labels = list(groups)
    colors = {label: method_color(label) for label in labels}
    edges = shared_bin_edges([float(row[key]) for rows in groups.values() for row in rows], BINS)
    draw_histogram(ax, groups, labels, colors, value_key=key, x_label=x_label, edges=edges, show_y_label=True)
    # Tight x range: the pooled bin edges plus half a bin on each side, instead of matplotlib's default margins.
    half_bin = (edges[1] - edges[0]) / 2
    ax.set_xlim(edges[0] - half_bin, edges[-1] + half_bin)
    ax.set_ylim(0, max(patch.get_height() for patch in ax.patches) / BAR_TOP)
    for line in ax.lines:
        line.set_ydata([0, MEAN_LINE_TOP])
    for index, label in enumerate(labels):
        values = [float(row[key]) for row in groups[label]]
        ax.text(0.03, 0.95 - STAT_LINE_STEP * index,
                # Both statistics are never positive, so the panel prints the magnitude of the mean.
                rf"{stat_label(label)} ${abs(fmean(values)):.3f}_{{\pm {stdev(values):.3f}}}$",
                transform=ax.transAxes, ha="left", va="top", color=colors[label], fontsize=STAT_FONT_SIZE, zorder=6)
    if panel:
        ax.text(-0.1, 1.03, rf"\textbf{{({panel})}}", transform=ax.transAxes, ha="left", va="bottom")


def family_legend(figure, family: tuple[str, str, str, str], x: float, y: float) -> None:
    """A single-row legend box, framed in the baseline colour, for one sampler and its PreSTO version."""
    _, baseline, presto, cut = family
    patch = dict(edgecolor=BLACK, linewidth=HIST_LINE_WIDTH, alpha=HIST_ALPHA)
    handles = [Patch(facecolor=BLUE, label=f"{baseline} ({cut})", **patch),
               Patch(facecolor=GREEN, label=presto, **patch),
               Line2D([0], [0], color=GRAY, linestyle="--", linewidth=MEAN_LINE_WIDTH, label="mean")]
    legend = figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(x, y), ncols=len(handles),
                           frameon=True, fontsize=LEGEND_FONT_SIZE, handlelength=1.4, columnspacing=1.2,
                           borderpad=0.35, fancybox=False)
    legend.get_frame().set_edgecolor(BLUE)
    legend.get_frame().set_linewidth(0.8)


def draw_page(family: tuple[str, str, str, str], rows: list[tuple[str, str]],
              pairs: dict[tuple[str, str, str], dict[str, list[dict]]], path: Path) -> Path:
    """One row per (dataset, base LLM) of this sampler; columns are log-likelihood and confidence."""
    header = TITLE_HEIGHT + LEGEND_HEIGHT
    height = header + ROW_HEIGHT * len(rows) + BOTTOM_HEIGHT
    figure = plt.figure(figsize=(FIGURE_WIDTH, height))
    grid = figure.add_gridspec(len(rows), len(METRICS), left=0.08, right=0.98, bottom=BOTTOM_HEIGHT / height,
                               top=1 - (header + 0.32) / height, hspace=0.5, wspace=0.2)
    for row, (dataset, model) in enumerate(rows):
        groups = one_presto_budget(pairs[(family[0], dataset, model)])
        axes = []
        for column, (metric, x_label) in enumerate(METRICS):
            ax = figure.add_subplot(grid[row, column])
            # Column labels only on the top row; below it the gap between rows holds the row titles.
            draw_pair_panel(ax, groups, metric, x_label, "ab"[column] if row == 0 else "")
            if row < len(rows) - 1:
                ax.set_xlabel("")
            axes.append(ax)
        n = len(next(iter(groups.values())))
        figure.text(axes[0].get_position().x0 + 0.01, axes[0].get_position().y1 + 0.05 / height,
                    rf"\textbf{{{DATASET_DISPLAY.get(dataset, dataset)} / {MODEL_DISPLAY.get(model, model)}}}"
                    rf"\quad ($n={n}$ prompts)", ha="left", va="bottom", fontsize=FIGURE_FONT_SIZE)
    family_legend(figure, family, x=0.5, y=1 - TITLE_HEIGHT / height)
    return save_figure(figure, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--launcher", type=Path, default=LAUNCHER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "by-sampler")
    parser.add_argument("--rows-per-page", type=int, default=9)
    parser.add_argument("--min-exact", type=int, default=10)
    parser.add_argument("--agreement-tol", type=float, default=0.01)
    parser.add_argument("--max-disagreement", type=float, default=0.05)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    apply_plot_style()
    mpl.rcParams.update({"font.size": FIGURE_FONT_SIZE, "axes.labelsize": FIGURE_FONT_SIZE,
                         "xtick.labelsize": FIGURE_FONT_SIZE - 1, "ytick.labelsize": FIGURE_FONT_SIZE - 1})
    jobs = launcher_jobs(args.launcher)
    agreement = rescoring_agreement(jobs, args.agreement_tol)
    kept = kept_pairs(jobs, agreement, args.min_exact, args.max_disagreement)
    pairs = {(family, dataset, model): groups for family, dataset, model, groups, _ in kept}
    order = {d: i for i, d in enumerate(DATASET_ORDER)}
    for family in FAMILIES:
        rows = sorted(((d, m) for (f, d, m) in pairs if f == family[0]),
                      key=lambda dm: (order.get(dm[0], len(order)),
                                      MODEL_ORDER.index(dm[1]) if dm[1] in MODEL_ORDER else len(MODEL_ORDER)))
        pages = [rows[i:i + args.rows_per_page] for i in range(0, len(rows), args.rows_per_page)]
        for part, page_rows in enumerate(pages, 1):
            suffix = f".part{part}" if len(pages) > 1 else ""
            path = args.output_dir / f"likelihood-and-confidence.{family[0]}-cut{suffix}.pdf"
            print(draw_page(family, page_rows, pairs, path))

if __name__ == "__main__":
    main()
