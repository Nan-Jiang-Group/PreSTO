#!/usr/bin/env python3
"""Summarize one predictive-prefetching (PreSTO) run: edge acceptance, prunable budget, and per-question yield.

Companion to ``draw_acceptance_predictors.py``, which scores each pre-proposal feature. This figure asks what the
scores buy. Every scored prefetch edge carries its acceptance probability A (``extract/acceptance_features.py``); the
run's ``.stats.json`` carries the per-question model calls, prefetched nodes, and MH transitions. Four panels:

    (a) distribution of A over all prefetch edges, split into certain reject (A <= epsilon), uncertain, and certain
        accept (A >= 1 - epsilon);
    (b) the budget a cut-only rule could prune: skipping every edge whose cut fraction is below t, the share of
        certain-reject edges removed and the share of acceptance mass (sum of A) lost, against the share skipped;
    (c) MH transitions per model call for each question, marked by whether its final answer is correct;
    (d) how many of the prefetched proposals one model call ends up accepting.

Beside the figure, ``.predictive-prefetching.questions.csv`` holds one row per question plus a pooled row, and
``.predictive-prefetching.pruning.csv`` holds panel (b)'s curve.

Run from the repository root:

    uv run --project case_studies/draw \
        python case_studies/draw/draw_predictive_prefetching_summary.py \
        --log /absolute/path/to/run.subtreePrefetch.vllm.log \
        --edges /absolute/path/to/pooled.acceptance-edges.csv \
        --figure-format png --no-usetex

``--edges`` reuses a CSV written by ``extract/acceptance_features.py``; without it the log is parsed again. The run's
``.stats.json`` and ``.csv`` are found beside the log.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from case_studies.draw.common.figure_io import save_figure, write_dict_rows
from case_studies.draw.common.plot_helpers import panel_letter
from case_studies.extract.analysis.acceptance_features import DEFAULT_EPSILON, edge_table
from case_studies.extract.run_naming import output_path
from case_studies.plot_config import BLUE, DARK_GRAY, GRAY, ORANGE, RED, apply_plot_style

apply_plot_style()

OUTPUT_TAG = ".predictive-prefetching"
LOG_SUFFIX = ".subtreePrefetch.vllm.log"
PRUNE_THRESHOLDS = np.linspace(0.0, 1.0, 101)
MARKED_THRESHOLDS = (0.25, 0.5)
MAX_ACCEPT_BIN = 3


def percent(value: float) -> str:
    """Format a share as a percentage, escaping the sign LaTeX would read as a comment."""
    sign = r"\%" if plt.rcParams["text.usetex"] else "%"
    return f"{100 * value:.0f}{sign}"


def run_prefix(log_path: Path) -> Path:
    """Return the path prefix the runner shares between the log, ``.csv``, and ``.stats.json``."""
    name = log_path.name
    if not name.endswith(LOG_SUFFIX):
        raise SystemExit(f"expected a *{LOG_SUFFIX} file, got {log_path}")
    return log_path.parent / name.removesuffix(LOG_SUFFIX)


def question_rows(prefix: Path) -> list[dict[str, object]]:
    """One row per question from ``.stats.json``, with correctness from the results ``.csv`` when present."""
    stats = json.loads(output_path(prefix, ".stats.json").read_text())
    correct: dict[int, bool] = {}
    results_path = output_path(prefix, ".csv")
    if results_path.exists():
        csv.field_size_limit(1 << 30)
        with results_path.open(newline="") as handle:
            for index, row in enumerate(csv.DictReader(handle)):
                correct[index] = row["is_correct"].strip().lower() == "true"
    rows = []
    for entry in stats:
        calls = entry["total_nfe"]
        steps = entry["total_walked_steps"]
        rows.append({
            "question": entry["idx"],
            "is_correct": correct.get(entry["idx"]),
            "model_calls": calls,
            "prefetched_nodes": entry["total_workload"],
            "mh_transitions": steps,
            "accepted_transitions": entry["total_acceptances"],
            "acceptance_rate": entry["acceptance_rate"],
            "transitions_per_call": steps / calls,
            "prefetched_per_call": entry["total_workload"] / calls,
            "budget_utilization": steps / entry["total_workload"],
            "accepts_per_call": entry["acceptances"],
        })
    return rows


def pooled_row(rows: list[dict[str, object]]) -> dict[str, object]:
    """Pool the question rows as sums, so ratios are ratios of totals."""
    calls = sum(row["model_calls"] for row in rows)
    nodes = sum(row["prefetched_nodes"] for row in rows)
    steps = sum(row["mh_transitions"] for row in rows)
    accepted = sum(row["accepted_transitions"] for row in rows)
    known = [row["is_correct"] for row in rows if row["is_correct"] is not None]
    return {
        "question": "pooled",
        "is_correct": f"{sum(known)}/{len(known)}" if known else None,
        "model_calls": calls,
        "prefetched_nodes": nodes,
        "mh_transitions": steps,
        "accepted_transitions": accepted,
        "acceptance_rate": accepted / steps,
        "transitions_per_call": steps / calls,
        "prefetched_per_call": nodes / calls,
        "budget_utilization": steps / nodes,
    }


def pruning_curve(edges: pd.DataFrame) -> pd.DataFrame:
    """Skip edges with ``cut_frac < t``: share skipped, certain rejects removed, and acceptance mass lost."""
    cut_frac = edges["cut_frac"].to_numpy()
    accept_prob = edges["accept_prob"].to_numpy()
    reject = edges["certain_reject"].to_numpy(dtype=bool)
    keep_useful = ~reject
    rows = []
    for threshold in PRUNE_THRESHOLDS:
        skipped = cut_frac < threshold
        rows.append({
            "cut_frac_threshold": threshold,
            "skipped_share": skipped.mean(),
            "certain_reject_removed": (skipped & reject).sum() / max(reject.sum(), 1),
            "non_reject_lost": (skipped & keep_useful).sum() / max(keep_useful.sum(), 1),
            "acceptance_mass_lost": accept_prob[skipped].sum() / accept_prob.sum(),
        })
    return pd.DataFrame(rows)


def draw_accept_histogram(ax, edges: pd.DataFrame, epsilon: float) -> None:
    accept_prob = edges["accept_prob"].to_numpy()
    inner = np.linspace(epsilon, 1.0 - epsilon, 13)
    bins = np.concatenate(([0.0], inner, [1.0]))
    counts, _ = np.histogram(accept_prob, bins=bins)
    shares = counts / counts.sum()
    colors = [BLUE] + [RED] * (len(bins) - 3) + [ORANGE]
    centers = 0.5 * (bins[:-1] + bins[1:])
    widths = np.diff(bins)
    # The two end bins are only epsilon wide; draw them at the inner width so they read as bars, not lines.
    widths[0] = widths[-1] = widths[1]
    centers[0], centers[-1] = -widths[1] / 2, 1.0 + widths[1] / 2
    ax.bar(centers, shares, width=widths * 0.92, color=colors, edgecolor="none")
    labels = (
        (centers[0], shares[0], percent(shares[0]), BLUE),
        (0.5, shares[1:-1].max(), f"{percent(shares[1:-1].sum())} total", RED),
        (centers[-1], shares[-1], percent(shares[-1]), ORANGE),
    )
    for x, y, text, color in labels:
        ax.annotate(text, (x, y), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom",
                    color=color, fontsize=7)
    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(0, shares.max() * 1.18)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("acceptance probability $A$ of a prefetch edge")
    ax.set_ylabel("share of edges")
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in (BLUE, RED, ORANGE)]
    ax.legend(handles, [f"$A \\le {epsilon:g}$", "uncertain", f"$A \\ge {1 - epsilon:g}$"],
              loc="upper center", fontsize=6.5, frameon=False)


def draw_pruning(ax, curve: pd.DataFrame) -> None:
    x = curve["skipped_share"]
    ax.plot([0, 1], [0, 1], color=GRAY, lw=0.8, ls=":", label="random skip")
    ax.plot(x, curve["certain_reject_removed"], color=BLUE, lw=1.4, label="certain rejects removed")
    ax.plot(x, curve["acceptance_mass_lost"], color=ORANGE, lw=1.4, label="acceptance mass lost")
    for threshold in MARKED_THRESHOLDS:
        row = curve.iloc[(curve["cut_frac_threshold"] - threshold).abs().argmin()]
        for column, color in (("certain_reject_removed", BLUE), ("acceptance_mass_lost", ORANGE)):
            ax.plot(row["skipped_share"], row[column], "o", color=color, ms=3)
        ax.annotate(f"$t={threshold:g}$", (row["skipped_share"], row["acceptance_mass_lost"]),
                    xytext=(4, -9), textcoords="offset points", ha="left", fontsize=6.5, color=DARK_GRAY)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("share of edges skipped (cut fraction $< t$)")
    ax.set_ylabel("share")
    ax.legend(loc="upper left", fontsize=6.5, frameon=False)


def draw_per_question(ax, rows: list[dict[str, object]]) -> None:
    ordered = sorted(rows, key=lambda row: row["transitions_per_call"], reverse=True)
    values = [row["transitions_per_call"] for row in ordered]
    colors = [
        GRAY if row["is_correct"] is None else (BLUE if row["is_correct"] else RED)
        for row in ordered
    ]
    positions = np.arange(len(ordered))
    ax.bar(positions, values, color=colors, width=0.8, edgecolor="none")
    pooled = pooled_row(rows)["transitions_per_call"]
    ax.axhline(pooled, color=DARK_GRAY, lw=0.9, ls="--")
    ax.axhline(1.0, color=GRAY, lw=0.8, ls=":")
    ax.set_xticks(positions, [str(row["question"]) for row in ordered], fontsize=5.5)
    ax.tick_params(axis="x", which="both", length=0)
    ax.set_xlim(-0.6, len(ordered) - 0.4)
    ax.set_ylim(0, max(values) * 1.4)
    ax.set_xlabel("question (sorted)")
    ax.set_ylabel("MH transitions per model call")
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in (BLUE, RED)] + [
        Line2D([], [], color=DARK_GRAY, lw=0.9, ls="--"),
        Line2D([], [], color=GRAY, lw=0.8, ls=":"),
    ]
    labels = ["correct", "incorrect", f"pooled ({pooled:.2f})", "sequential MH (1)"]
    ax.legend(handles, labels, loc="upper center", fontsize=6.5, frameon=False, ncol=2)


def draw_accepts_per_call(ax, rows: list[dict[str, object]]) -> None:
    accepts = np.concatenate([np.asarray(row["accepts_per_call"]) for row in rows])
    clipped = np.minimum(accepts, MAX_ACCEPT_BIN)
    counts = np.bincount(clipped, minlength=MAX_ACCEPT_BIN + 1)
    shares = counts / counts.sum()
    positions = np.arange(MAX_ACCEPT_BIN + 1)
    ax.bar(positions, shares, color=DARK_GRAY, width=0.7, edgecolor="none")
    for x, share in zip(positions, shares):
        ax.annotate(percent(share), (x, share), xytext=(0, 2), textcoords="offset points", ha="center",
                    va="bottom", fontsize=7)
    labels = [str(value) for value in positions]
    labels[-1] = f"{MAX_ACCEPT_BIN}+"
    ax.set_xticks(positions, labels)
    ax.tick_params(axis="x", which="both", length=0)
    ax.set_ylim(0, shares.max() * 1.18)
    ax.set_xlabel("accepted proposals per model call")
    ax.set_ylabel("share of model calls")
    ax.text(0.97, 0.95, f"mean {accepts.mean():.2f}\n{len(accepts)} calls", transform=ax.transAxes,
            ha="right", va="top", fontsize=6.5, color=DARK_GRAY)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--log", type=Path, required=True, help="*.subtreePrefetch.vllm.log of one run")
    parser.add_argument("--edges", type=Path, default=None, help="edge CSV from extract/acceptance_features.py")
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    parser.add_argument("--output-prefix", type=Path, default=None, help="defaults to the run prefix beside the log")
    parser.add_argument("--figure-format", choices=("pdf", "png"), default="pdf", help="PDF export needs qpdf.")
    parser.add_argument("--no-usetex", action="store_true", help="use Matplotlib's own text rendering")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.no_usetex:
        plt.rcParams["text.usetex"] = False
    prefix = run_prefix(args.log)
    output_prefix = args.output_prefix or prefix
    edges = pd.read_csv(args.edges) if args.edges else edge_table(args.log, epsilon=args.epsilon)
    rows = question_rows(prefix)
    curve = pruning_curve(edges)

    # A 2x2 grid at text width, so the figure is placed at its drawn size and the labels stay legible in the paper.
    figure, axes = plt.subplots(2, 2, figsize=(6.2, 4.4))
    axes = axes.ravel()
    draw_accept_histogram(axes[0], edges, args.epsilon)
    draw_pruning(axes[1], curve)
    draw_per_question(axes[2], rows)
    draw_accepts_per_call(axes[3], rows)
    for index, ax in enumerate(axes):
        ax.set_title(panel_letter(index), loc="left", fontsize=9, fontweight="bold")
    figure.tight_layout(w_pad=1.2, h_pad=0.8)
    figure_path = save_figure(figure, output_path(output_prefix, f"{OUTPUT_TAG}.{args.figure_format}"))
    print(f"wrote {figure_path}")

    table = [{key: value for key, value in row.items() if key != "accepts_per_call"} for row in rows]
    table.append(pooled_row(rows))
    print(f"wrote {write_dict_rows(output_path(output_prefix, f'{OUTPUT_TAG}.questions.csv'), table)}")
    curve_path = output_path(output_prefix, f"{OUTPUT_TAG}.pruning.csv")
    curve.to_csv(curve_path, index=False)
    print(f"wrote {curve_path}")
    pd.set_option("display.width", 200)
    print(pd.DataFrame(table).round(3).to_string(index=False))
    for threshold in MARKED_THRESHOLDS:
        row = curve.iloc[(curve["cut_frac_threshold"] - threshold).abs().argmin()]
        print(f"cut_frac < {threshold:g}: skip {row['skipped_share']:.1%} of edges, remove "
              f"{row['certain_reject_removed']:.1%} of certain rejects, lose {row['acceptance_mass_lost']:.1%} "
              f"of acceptance mass")


if __name__ == "__main__":
    main()
