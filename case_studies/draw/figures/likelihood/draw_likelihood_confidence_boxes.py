#!/usr/bin/env python3
"""Box plots of rescored likelihood and confidence, baseline vs. PreSTO, one pair of boxes per dataset.

Selects the same pairs and prompts as draw_rescored_likelihood_and_confidence_summary.py (its launcher jobs,
rescoring check, exact-prompt restriction, and single PreSTO budget) and reads the paired TOST p-value from the
likelihood-confidence-summary.csv that script writes; the p-value is printed right of each pair, bold when equivalent
at 0.05. With several --families, each family is one row of panels under its own "baseline vs. PreSTO" heading,
the metric columns share an x-range, and one legend serves all rows.

Run:

    /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/.venv/bin/python \
        /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/draw/draw_likelihood_confidence_boxes.py
"""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from case_studies import paths
from case_studies.draw.common.figure_io import save_figure
from case_studies.draw.figures.likelihood.draw_rescored_likelihood_and_confidence import one_presto_budget
from case_studies.draw.figures.likelihood.draw_rescored_likelihood_and_confidence_summary import (
    LAUNCHER,
    kept_pairs,
    launcher_jobs,
    rescoring_agreement,
)
from case_studies.draw.figures.likelihood.plot_likelihood_and_confidence import PANEL_SPECS
from case_studies.draw.figures.proposals.draw_dataset_proposal_certainty import DATASET_LABELS, MODEL_LABELS
from case_studies.plot_config import BLUE, GRAY, GREEN, PANEL_LABEL_FONT_SIZE, PANEL_LABEL_FONT_WEIGHT, apply_plot_style

CASE_STUDIES_DIR = paths.CASE_STUDIES_DIR
DEFAULT_OUTPUT_DIR = CASE_STUDIES_DIR / "score_log_likelihood_and_confidence/figures"
DATASET_DISPLAY = {**DATASET_LABELS, "aime": "AIME"}
FAMILY_NAMES = {"uniform": ("PowerMH", "PreSTO-PowerMH"), "entropy": ("EntropyCut", "PreSTO-EntropyCut")}
PANEL_TITLES = {"log_likelihood": "Token log-likelihood", "confidence": "Token confidence"}
P_EQ_FONT_SIZE = 6.5
BOX_WIDTH = 0.34
# Opaque, so the boxes print in the paper's sglBlue (plot_config.BLUE) and sglGreen (plot_config.GREEN).
BOX_FILL_ALPHA = 1.0
EQUIVALENT_SHADE_ALPHA = 0.22


def load_pairs(family: str, models: list[str], jobs, agreement, p_equiv: dict) -> list[tuple]:
    """((dataset, model), groups) for one family, the equivalent datasets first."""
    every_model = models == ["all"]
    pairs = [((dataset, model), one_presto_budget(groups)) for fam, dataset, model, groups, _
             in kept_pairs(jobs, agreement, 10, 0.05)
             if fam == family and (every_model or model in models)]
    # By how many of the two statistics the paired TOST finds equivalent (stable sort).
    pairs.sort(key=lambda pair: -sum(p_equiv[(family, pair[0], metric)] < 0.05 for metric, _ in PANEL_SPECS))
    for key, groups in pairs:
        print(family, key, {label[:22]: len(rows) for label, rows in groups.items()})
    return pairs


def draw_boxes(ax, pairs: list[tuple], metric: str) -> None:
    """Datasets run top to bottom, the baseline box above the PreSTO box."""
    for offset, index, color in ((-BOX_WIDTH / 2 - 0.02, 0, BLUE), (BOX_WIDTH / 2 + 0.02, 1, GREEN)):
        values = [[float(row[metric]) for row in list(groups.values())[index]] for _, groups in pairs]
        ax.boxplot(values, positions=[i + offset for i in range(len(pairs))], widths=BOX_WIDTH,
                   orientation="horizontal", patch_artist=True, showfliers=True, manage_ticks=False,
                   boxprops={"facecolor": color, "alpha": BOX_FILL_ALPHA, "linewidth": 0.5},
                   medianprops={"color": "black", "linewidth": 0.9},
                   whiskerprops={"linewidth": 0.5}, capprops={"linewidth": 0.5},
                   flierprops={"marker": "o", "markersize": 1.8, "markerfacecolor": color, "markeredgewidth": 0})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR / "likelihood-confidence-boxes.pdf")
    parser.add_argument("--summary", type=Path, default=DEFAULT_OUTPUT_DIR / "likelihood-confidence-summary.csv")
    parser.add_argument("--families", nargs="+", default=["uniform"], choices=list(FAMILY_NAMES),
                        help="Cut laws to draw, one row of panels each, top to bottom.")
    parser.add_argument("--models", nargs="+", default=["qwen3.5-4b"],
                        help="Base LLMs to draw (summary-CSV names); 'all' draws every kept pair. With one model the rows"
                             " name only the dataset.")
    parser.add_argument("--no-p-values", dest="p_values", action="store_false",
                        help="Omit the TOST p-value printed right of each pair (the shading still marks equivalence).")
    args = parser.parse_args()

    # Same selection as the summary script, with its default thresholds.
    jobs = launcher_jobs(LAUNCHER)
    agreement = rescoring_agreement(jobs, 0.01)
    with args.summary.open() as handle:
        p_equiv = {(r["family"], (r["dataset"], r["model"]), r["metric"]): float(r["p_equiv"])
                   for r in csv.DictReader(handle) if r["p_equiv"]}
    rows = [(family, load_pairs(family, args.models, jobs, agreement, p_equiv)) for family in args.families]
    single_model = len({model for _, pairs in rows for (_, model), _ in pairs}) == 1
    headed = len(rows) > 1

    apply_plot_style()
    heights = [len(pairs) for _, pairs in rows]
    figure = plt.figure(figsize=(6.8, 0.2 * sum(heights) + 0.5 * len(rows) + (0.1 if headed else 0)),
                        layout="constrained")
    subfigures = figure.subfigures(len(rows), 1, height_ratios=[h + 2.5 for h in heights], squeeze=False)[:, 0]
    grid = [subfigure.subplots(1, len(PANEL_SPECS), sharey=True) for subfigure in subfigures]
    for axes, (_, pairs) in zip(grid, rows):
        for ax, (metric, _) in zip(axes, PANEL_SPECS):
            draw_boxes(ax, pairs, metric)
    # One x-range per metric column, with room on the right for the p-values.
    for column, (metric, _) in enumerate(PANEL_SPECS):
        low = min(axes[column].get_xlim()[0] for axes in grid)
        high = max(axes[column].get_xlim()[1] for axes in grid)
        for axes in grid:
            axes[column].set_xlim(low, high + (0.3 if args.p_values else 0.05) * (high - low))
        for axes, (family, pairs) in zip(grid, rows):
            ax = axes[column]
            for i, (key, _) in enumerate(pairs):
                p_eq = p_equiv[(family, key, metric)]
                # Shade the pairs the paired TOST finds equivalent at 0.05, behind the boxes.
                if p_eq < 0.05:
                    ax.axhspan(i - 0.5, i + 0.5, color=GRAY, alpha=EQUIVALENT_SHADE_ALPHA, linewidth=0, zorder=0)
                if not args.p_values:
                    continue
                text = r"$<$0.001" if p_eq < 0.001 else f"{p_eq:.3f}"
                ax.text(high + 0.28 * (high - low), i, text, ha="right", va="center", fontsize=P_EQ_FONT_SIZE,
                        fontweight="bold" if p_eq < 0.05 else "normal")
    panel = 0
    for row, (subfigure, axes, (family, pairs)) in enumerate(zip(subfigures, grid, rows)):
        if headed:
            baseline_name, presto_name = FAMILY_NAMES[family]
            subfigure.suptitle(f"{baseline_name} vs. {presto_name}", fontsize=plt.rcParams["axes.labelsize"])
        for column, (ax, (metric, _)) in enumerate(zip(axes, PANEL_SPECS)):
            ax.tick_params(axis="y", which="both", left=False, right=False)
            ax.tick_params(axis="x", which="both", top=False)
            if row == len(rows) - 1:
                ax.set_xlabel(PANEL_TITLES[metric] + r" under $p_0$")
            # Panel label level with the top of the frame: in the first column as a y-label, so it sits left of the
            # dataset names; otherwise just left of the frame.
            label = f"({chr(ord('a') + panel)})"
            panel += 1
            if column == 0:
                # The long "dataset / model" names need more room than the dataset-only names.
                ax.set_ylabel(label, rotation=0, loc="top", va="top", labelpad=-5 if single_model else 4,
                              fontsize=PANEL_LABEL_FONT_SIZE,
                              fontweight=PANEL_LABEL_FONT_WEIGHT)
            else:
                ax.text(-0.01, 1.0, label, transform=ax.transAxes, ha="right", va="top",
                        fontsize=PANEL_LABEL_FONT_SIZE, fontweight=PANEL_LABEL_FONT_WEIGHT)
        axes[0].set_yticks(range(len(pairs)), [DATASET_DISPLAY.get(d, d) if single_model
                                               else f"{DATASET_DISPLAY.get(d, d)} / {MODEL_LABELS.get(m, m)}"
                                               for (d, m), _ in pairs])
        axes[0].set_ylim(len(pairs) - 0.5, -0.5)
    if headed:
        baseline_name, presto_name = "baseline", "PreSTO variant"
    else:
        baseline_name, presto_name = FAMILY_NAMES[rows[0][0]]
        presto_name += "\n(Ours)"
    handles = [Patch(facecolor=BLUE, alpha=BOX_FILL_ALPHA, edgecolor="black", linewidth=0.5, label=baseline_name),
               Patch(facecolor=GREEN, alpha=BOX_FILL_ALPHA, edgecolor="black", linewidth=0.5, label=presto_name),
               Patch(facecolor=GRAY, alpha=EQUIVALENT_SHADE_ALPHA, linewidth=0,
                     label=("equivalent " if headed else "equivalent\n") + r"(TOST $p_{\mathrm{eq}}<0.05$)")]
    # Stacked rows: one legend row across the top; a single row of panels: legend stacked on the right.
    figure.legend(handles=handles, loc="outside upper center" if headed else "outside right center",
                  ncols=len(handles) if headed else 1, frameon=False, handlelength=1.0, handletextpad=0.4, columnspacing=1.5, labelspacing=1.4,
                  borderpad=0, borderaxespad=0.2)
    path = save_figure(figure, args.output, pad_inches=0)
    # Crop to the ink so the PDF carries no white margin (the legend's text box leaves a few points otherwise).
    subprocess.run(["pdfcrop", "--margins", "0", str(path), str(path)], check=True, capture_output=True)
    print(path)


if __name__ == "__main__":
    main()
