#!/usr/bin/env python3
"""Grouped bars of rescored likelihood and confidence, baseline vs. PreSTO, one pair of bars per dataset.

Reads likelihood-confidence-summary.csv written by draw_rescored_likelihood_and_confidence_summary.py. Bars are the
across-prompt means with 95% intervals (1.96 standard errors); the paired TOST p-value is printed above each pair,
bold when equivalent at 0.05.

Run:

    /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/.venv/bin/python \
        /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/draw/draw_likelihood_confidence_bars.py
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from case_studies import paths
from case_studies.draw.common.figure_io import save_figure
from case_studies.draw.figures.likelihood.plot_likelihood_and_confidence import PANEL_SPECS, add_panel_label
from case_studies.draw.figures.proposals.draw_dataset_proposal_certainty import DATASET_LABELS
from case_studies.plot_config import BLUE, GREEN, apply_plot_style

CASE_STUDIES_DIR = paths.CASE_STUDIES_DIR
DEFAULT_INPUT_DIR = CASE_STUDIES_DIR / "score_log_likelihood_and_confidence/figures"
DATASET_DISPLAY = {**DATASET_LABELS, "aime": "AIME"}
FAMILY_NAMES = {"uniform": ("PowerMH", "PreSTO-PowerMH"), "entropy": ("EntropyCut", "PreSTO-EntropyCut")}
PANEL_TITLES = {"log_likelihood": "Token log-likelihood", "confidence": "Token confidence"}
P_EQ_FONT_SIZE = 6.5
BAR_WIDTH = 0.38


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_INPUT_DIR / "likelihood-confidence-bars.pdf")
    parser.add_argument("--family", default="uniform", choices=list(FAMILY_NAMES))
    parser.add_argument("--model", default="qwen3.5-4b")
    args = parser.parse_args()

    with (args.input_dir / "likelihood-confidence-summary.csv").open() as handle:
        rows = [r for r in csv.DictReader(handle) if r["family"] == args.family and r["model"] == args.model]
    datasets = list(dict.fromkeys(r["dataset"] for r in rows))
    baseline_name, presto_name = FAMILY_NAMES[args.family]

    apply_plot_style()
    figure, axes = plt.subplots(1, len(PANEL_SPECS), figsize=(6.8, 1.75), layout="constrained")
    x = np.arange(len(datasets))
    for column, (ax, (metric, _)) in enumerate(zip(axes, PANEL_SPECS)):
        for offset, is_presto, color, label in ((-BAR_WIDTH / 2, False, BLUE, baseline_name),
                                                 (BAR_WIDTH / 2, True, GREEN, presto_name)):
            picked = [next(r for r in rows if r["dataset"] == d and r["metric"] == metric
                           and bool(r["p_equiv"]) == is_presto) for d in datasets]
            means = [float(r["mean"]) for r in picked]
            errors = [1.96 * float(r["sd"]) / math.sqrt(int(r["n"])) for r in picked]
            ax.bar(x + offset, means, BAR_WIDTH, yerr=errors, color=color, edgecolor="black", linewidth=0.4,
                   error_kw={"elinewidth": 0.7, "capsize": 1.5}, label=label if column == 0 else None)
        low = min(float(r["mean"]) - 1.96 * float(r["sd"]) / math.sqrt(int(r["n"]))
                  for r in rows if r["metric"] == metric)
        ax.set_ylim(low * 1.18, 0)
        for i, dataset in enumerate(datasets):
            r = next(r for r in rows if r["dataset"] == dataset and r["metric"] == metric and r["p_equiv"])
            p_eq = float(r["p_equiv"])
            text = r"$<$0.001" if p_eq < 0.001 else f"{p_eq:.3f}"
            ax.text(i, low * 1.16, text, ha="center", va="bottom", fontsize=P_EQ_FONT_SIZE,
                    fontweight="bold" if p_eq < 0.05 else "normal")
        ax.set_xticks(x, [DATASET_DISPLAY.get(d, d) for d in datasets], fontsize=6.5)
        ax.tick_params(axis="x", which="both", top=False, bottom=False)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title(PANEL_TITLES[metric])
        add_panel_label(ax, f"({chr(ord('a') + column)})")
    axes[0].set_ylabel(r"mean under $p_0$")
    figure.legend(loc="outside right center", ncols=1, frameon=False, handletextpad=0.4, borderpad=0,
                  borderaxespad=0.2)
    print(save_figure(figure, args.output, pad_inches=0))


if __name__ == "__main__":
    main()
