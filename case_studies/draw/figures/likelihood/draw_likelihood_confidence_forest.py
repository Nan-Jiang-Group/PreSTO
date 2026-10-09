#!/usr/bin/env python3
"""Forest plot of the standardized paired likelihood/confidence differences, PreSTO minus baseline.

Reads likelihood-confidence-summary.csv and likelihood-confidence-pooled-equivalence.csv written by
draw_rescored_likelihood_and_confidence_summary.py. Each pair's mean difference and 90% TOST interval are divided by
the baseline's across-prompt standard deviation (margin / equivalence margin), so every row shares one axis with the
equivalence band [-margin, +margin]. Filled markers are equivalent at 0.05; open markers are not. Diamonds give each
sampler's fixed-effect pooled difference with its 90% interval.

Run:

    /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/.venv/bin/python \
        /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/draw/draw_likelihood_confidence_forest.py
"""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory

from case_studies import paths
from case_studies.draw.common.figure_io import save_figure
from case_studies.draw.figures.likelihood.draw_rescored_likelihood_and_confidence_summary import normal_cdf, t_quantile
from case_studies.draw.figures.likelihood.plot_likelihood_and_confidence import PANEL_SPECS, add_panel_label
from case_studies.draw.figures.proposals.draw_dataset_proposal_certainty import DATASET_LABELS, MODEL_LABELS
from case_studies.plot_config import GRAY, GREEN, ORANGE, apply_plot_style

CASE_STUDIES_DIR = paths.CASE_STUDIES_DIR
DEFAULT_INPUT_DIR = CASE_STUDIES_DIR / "score_log_likelihood_and_confidence/figures"
DATASET_DISPLAY = {**DATASET_LABELS, "aime": "AIME"}
FAMILIES = (("uniform", "PreSTO-PowerMH", GREEN), ("entropy", "PreSTO-EntropyCut", ORANGE))
P_DIFF_FONT_SIZE = 6.5
PANEL_TITLES = {"log_likelihood": "Token log-likelihood", "confidence": "Token confidence"}


def read_rows(path: Path) -> list[dict]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def pooled_difference(rows: list[dict], margin_sd: float) -> tuple[float, float, float, float]:
    """Fixed-effect pool of the drawn pairs' standardized differences: (mean, 90% low, high, TOST p-value).

    Matches pooled_equivalence in draw_rescored_likelihood_and_confidence_summary.py; each pair's standard error is
    recovered from its 90% t-interval.
    """
    z, weights = [], []
    for r in rows:
        sd = float(r["margin"]) / margin_sd
        se = (float(r["ci90_high"]) - float(r["ci90_low"])) / (2 * t_quantile(0.95, int(r["n"]) - 1))
        z.append(float(r["mean_diff"]) / sd)
        weights.append((sd / se) ** 2)
    mean = sum(w * v for w, v in zip(weights, z)) / sum(weights)
    se = sum(weights) ** -0.5
    p_equiv = max(1 - normal_cdf((mean + margin_sd) / se), normal_cdf((mean - margin_sd) / se))
    return mean, mean - 1.645 * se, mean + 1.645 * se, p_equiv


def build_layout(summary: list[dict], families: list[str], models: list[str] | None,
                 pooled: bool) -> list[tuple]:
    """Rows top to bottom: ('header', family) / ('pair', family, dataset, model) / ('pooled', family).

    The header row is drawn only when more than one family is shown.
    """
    layout = []
    for family, _, _ in FAMILIES:
        if family not in families:
            continue
        pairs = list(dict.fromkeys((r["dataset"], r["model"]) for r in summary
                                   if r["family"] == family and r["p_equiv"]
                                   and (models is None or r["model"] in models)))
        if not pairs:
            continue
        if len(families) > 1:
            layout.append(("header", family))
        layout += [("pair", family, d, m) for d, m in pairs]
        if pooled:
            layout.append(("pooled", family))
    return layout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_INPUT_DIR / "likelihood-confidence-forest.pdf")
    parser.add_argument("--margin-sd", type=float, default=0.2)
    parser.add_argument("--families", nargs="+", default=["uniform"], choices=[f for f, _, _ in FAMILIES],
                        help="Cut laws to draw, top to bottom.")
    parser.add_argument("--models", nargs="+", default=["qwen3.5-4b"],
                        help="Base LLMs to draw (summary-CSV names); the pooled row pools only these pairs.")
    parser.add_argument("--p-diff", action="store_true",
                        help="Add a column of Holm-adjusted difference-test p-values right of each panel.")
    parser.add_argument("--pooled", action="store_true", help="Add each family's fixed-effect pooled row.")
    args = parser.parse_args()

    summary = read_rows(args.input_dir / "likelihood-confidence-summary.csv")
    tested = {(r["family"], r["dataset"], r["model"], r["metric"]): r for r in summary if r["p_equiv"]}
    layout = build_layout(summary, args.families, args.models, args.pooled)
    single_model = args.models is not None and len(args.models) == 1
    colors = {family: color for family, _, color in FAMILIES}
    names = {family: name for family, name, _ in FAMILIES}

    apply_plot_style()
    figure, axes = plt.subplots(1, len(PANEL_SPECS), figsize=(6.8, 0.13 * len(layout) + 0.9), sharey=True,
                                layout="constrained")
    for column, (ax, (metric, _)) in enumerate(zip(axes, PANEL_SPECS)):
        ax.axvspan(-args.margin_sd, args.margin_sd, color=GRAY, alpha=0.18, linewidth=0)
        ax.axvline(0, color=GRAY, linewidth=0.6, linestyle="--")
        p_column = blended_transform_factory(ax.transAxes, ax.transData)
        if args.p_diff:
            ax.text(1.03, -0.6, r"$p_{\mathrm{diff}}$", transform=p_column, ha="left", va="bottom",
                    fontsize=P_DIFF_FONT_SIZE)
        for y, entry in enumerate(layout):
            kind, family = entry[0], entry[1]
            color = colors[family]
            if kind == "pair":
                r = tested[(family, entry[2], entry[3], metric)]
                sd = float(r["margin"]) / args.margin_sd
                mean, low, high = (float(r[k]) / sd for k in ("mean_diff", "ci90_low", "ci90_high"))
                equivalent = float(r["p_equiv"]) < 0.05
                ax.errorbar(mean, y, xerr=[[mean - low], [high - mean]], fmt="o", color=color, markersize=3.2,
                            markerfacecolor=color if equivalent else "white", elinewidth=0.9, capsize=1.5)
                if args.p_diff:
                    p_holm = float(r["p_holm"])
                    ax.text(1.03, y, f"{p_holm:.2f}", transform=p_column, ha="left", va="center",
                            fontsize=P_DIFF_FONT_SIZE, fontweight="bold" if p_holm < 0.05 else "normal")
            elif kind == "pooled":
                rows = [tested[(family, e[2], e[3], metric)] for e in layout if e[0] == "pair" and e[1] == family]
                mean, low, high, p_equiv = pooled_difference(rows, args.margin_sd)
                print(f"pooled {family} {metric}: {mean:+.3f} [{low:+.3f}, {high:+.3f}] p_eq={p_equiv:.2g}")
                ax.errorbar(mean, y, xerr=[[mean - low], [high - mean]], fmt="D", color=color, markersize=4.5,
                            markeredgecolor="black", markeredgewidth=0.5, elinewidth=1.2, capsize=2)
        for y, entry in enumerate(layout):
            if entry[0] == "header" and y > 0:
                ax.axhline(y - 0.5, color=GRAY, linewidth=0.4)
        ax.set_title(PANEL_TITLES[metric])
        ax.set_xlim(-0.6, 0.6)
        add_panel_label(ax, f"({chr(ord('a') + column)})")
        ax.tick_params(axis="y", which="both", left=False, right=False)

    labels = []
    for entry in layout:
        if entry[0] == "header":
            labels.append(names[entry[1]])
        elif entry[0] == "pair":
            dataset = DATASET_DISPLAY.get(entry[2], entry[2])
            labels.append(dataset if single_model else f"{dataset} / {MODEL_LABELS.get(entry[3], entry[3])}")
        else:
            labels.append("Pooled")
    axes[0].set_yticks(range(len(layout)), labels)
    axes[0].set_ylim(len(layout) - 0.5, -0.5)
    for tick, entry in zip(axes[0].get_yticklabels(), layout):
        if entry[0] != "pair":
            tick.set_fontweight("bold")
    legend_color = colors[args.families[0]]
    handles = [
        Line2D([], [], marker="o", color=legend_color, markersize=3.2, linestyle="none",
               label="equivalent\n(TOST $p<0.05$)"),
        Line2D([], [], marker="o", color=legend_color, markerfacecolor="white", markersize=3.2, linestyle="none",
               label="not equivalent"),
        *([Line2D([], [], marker="D", color=legend_color, markeredgecolor="black", markeredgewidth=0.5,
                  markersize=4.5, linestyle="none", label="pooled")] if args.pooled else []),
        Patch(facecolor=GRAY, alpha=0.18, label=rf"$\pm{args.margin_sd:g}$ SD" "\nmargin"),
    ]
    figure.legend(handles=handles, loc="outside right center", ncols=1, frameon=False, handletextpad=0.4,
                  borderpad=0, borderaxespad=0.2)
    figure.supxlabel(r"Standardized difference, PreSTO $-$ baseline (baseline SD)", fontsize=plt.rcParams["axes.labelsize"])
    path = save_figure(figure, args.output, pad_inches=0)
    # Crop to the ink so the PDF carries no white margin (the legend's text box leaves a few points otherwise).
    subprocess.run(["pdfcrop", "--margins", "0", str(path), str(path)], check=True, capture_output=True)
    print(path)


if __name__ == "__main__":
    main()
