#!/usr/bin/env python3
"""Plot sampled-response log-likelihood, confidence, and length from PreSTO result CSVs.

Each subtree-prefetching run writes one result CSV beside its log, one row per sampled response, with the response
length ``num_response_tokens``, the length-normalized log-likelihood (1/n) log p_0(x | x_0), and the confidence
(1/n) sum_t sum_u p_0 log p_0. The EntropyCut baseline (powerMH) CSVs do not record these, so only PreSTO runs appear.

Only the matched sweep is kept (--steps, --samples, --max-new-tokens, --rank, --budgets), and when a configuration was re-run on
several dates the latest date wins. Runs whose median response is at most --min-median-tokens long are failed
generations (the model emitted EOS immediately); they are dropped from the figures, listed on stdout, and kept in the
CSV with ``excluded=True``.

Outputs, written to --output-dir:
    <stem>.pdf          rows = datasets; columns = sequence log-likelihood, token log-likelihood, confidence, length
    <stem>.scatter.pdf  token log-likelihood vs confidence per dataset, colored by response length
    <stem>.csv          one row per sampled response behind both figures

Run:

    /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/.venv/bin/python \
        /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/draw/draw_likelihood_confidence_length.py
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from case_studies import paths
from case_studies.draw.common.figure_io import save_figure, write_dict_rows
from case_studies.draw.common.plot_helpers import panel_letter
from case_studies.extract.run_naming import MODEL_DISPLAY_NAMES, dataset_of, model_of, prefetch_budget_of, rank_of
from case_studies.plot_config import (
    BLACK,
    BLUE,
    BOXPLOT_FILL_ALPHA,
    GRAY,
    ORANGE,
    SUBTREE_METHOD_LABEL,
    apply_plot_style,
)

CASE_STUDIES_DIR = paths.CASE_STUDIES_DIR
DEFAULT_LOG_ROOT = CASE_STUDIES_DIR / "logs-entropycut"
DEFAULT_OUTPUT_DIR = DEFAULT_LOG_ROOT / "all-datasets"
RESULT_CSV = re.compile(r"\.seed\d+\.csv$")
DATE_DIR = re.compile(r"^\d{4}-\d{2}-\d{2}$")

MODEL_ORDER = ("qwen3-4b", "qwen3-8b", "qwen3.5-4b", "qwen3.5-9b", "gemma-12b-it")
DATASET_ORDER = ("math500", "aime", "gpqa", "mmlu", "human_eval", "mbpp", "lcb_v6")
BUDGET_COLORS = (BLUE, ORANGE)
METRICS = (
    ("logprob_sum", r"Sequence log-likelihood $\log p_0(x)$"),
    ("log_likelihood", r"Token log-likelihood $\frac{1}{n}\log p_0(x)$"),
    ("confidence", r"Confidence $\frac{1}{n}\sum_t\sum_u p_0\log p_0$"),
    ("num_response_tokens", "Response length (tokens)"),
)
BOX_WIDTH = 0.34
POINT_SIZE = 4.0
POINT_ALPHA = 0.55
PANEL_WIDTH = 3.3
PANEL_HEIGHT = 1.9
csv.field_size_limit(1 << 30)


def run_date(path: Path) -> str:
    """Return the date-stamped directory a run sits under, or '' if there is none."""
    return next((part for part in reversed(path.parts) if DATE_DIR.match(part)), "")


def read_rows(log_root: Path, args: argparse.Namespace) -> list[dict]:
    """Read every matched-sweep response, keeping only the latest date per configuration."""
    tag = f".steps{args.steps}."
    latest: dict[tuple, Path] = {}
    for path in sorted(log_root.rglob("*prefetch-budget-*.csv")):
        name = path.name
        if not RESULT_CSV.search(name) or tag not in name or f".samples{args.samples}." not in name:
            continue
        if f".maxnew{args.max_new_tokens}." not in name or rank_of(name) != args.rank:
            continue
        if prefetch_budget_of(name) not in args.budgets:
            continue
        key = (dataset_of(name), model_of(name), prefetch_budget_of(name))
        if key not in latest or run_date(path) > run_date(latest[key]):
            latest[key] = path

    rows = []
    for (dataset, model, budget), path in sorted(latest.items()):
        with path.open(encoding="utf-8") as handle:
            records = [r for r in csv.DictReader(handle) if r.get("log_likelihood") not in (None, "")]
        if not records:
            continue
        excluded = median(int(r["num_response_tokens"]) for r in records) <= args.min_median_tokens
        for sample_idx, record in enumerate(records):
            tokens = int(record["num_response_tokens"])
            log_likelihood = float(record["log_likelihood"])
            rows.append({
                "dataset": dataset,
                "model": model,
                "prefetch_budget": budget,
                "run_date": run_date(path),
                "sample_idx": sample_idx,
                "num_response_tokens": tokens,
                "log_likelihood": log_likelihood,
                "logprob_sum": log_likelihood * tokens,
                "confidence": float(record["confidence"]),
                "is_correct": record.get("is_correct") == "True",
                "excluded": excluded,
                "source_csv": str(path),
            })
    return rows


def ordered(values: set[str], order: tuple[str, ...]) -> list[str]:
    """Sort by a preferred order, with unknown names appended alphabetically."""
    return sorted(values, key=lambda v: (order.index(v) if v in order else len(order), v))


def model_label(model: str) -> str:
    """Return the figure label for a model alias."""
    return MODEL_DISPLAY_NAMES.get(model, model.replace("qwen", "Qwen"))


def draw_grid(rows: list[dict], budgets: list[int], max_new_tokens: int, path: Path) -> None:
    """Draw one row per dataset and one column per metric, models on x, budgets side by side."""
    datasets = ordered({r["dataset"] for r in rows}, DATASET_ORDER)
    figure, axes = plt.subplots(
        len(datasets), len(METRICS),
        figsize=(PANEL_WIDTH * len(METRICS), PANEL_HEIGHT * len(datasets)),
        squeeze=False, layout="constrained",
    )
    rng = np.random.default_rng(0)
    offsets = (np.arange(len(budgets)) - (len(budgets) - 1) / 2) * BOX_WIDTH * 1.1
    for row_index, dataset in enumerate(datasets):
        subset = [r for r in rows if r["dataset"] == dataset]
        models = ordered({r["model"] for r in subset}, MODEL_ORDER)
        for column_index, (key, label) in enumerate(METRICS):
            ax = axes[row_index, column_index]
            for budget_index, budget in enumerate(budgets):
                color = BUDGET_COLORS[budget_index % len(BUDGET_COLORS)]
                for model_index, model in enumerate(models):
                    values = [r[key] for r in subset if r["model"] == model and r["prefetch_budget"] == budget]
                    if not values:
                        continue
                    x = model_index + offsets[budget_index]
                    ax.boxplot(
                        [values], positions=[x], widths=BOX_WIDTH, patch_artist=True, showfliers=False,
                        boxprops={"facecolor": color, "alpha": BOXPLOT_FILL_ALPHA, "edgecolor": color},
                        medianprops={"color": color, "linewidth": 1.2},
                        whiskerprops={"color": color}, capprops={"color": color},
                    )
                    jitter = rng.uniform(-BOX_WIDTH / 3, BOX_WIDTH / 3, len(values))
                    ax.scatter(x + jitter, values, s=POINT_SIZE, color=color, alpha=POINT_ALPHA,
                               linewidths=0, zorder=3)
            if key == "num_response_tokens":
                ax.axhline(max_new_tokens, color=GRAY, linestyle="--", linewidth=0.8, zorder=1)
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels([model_label(m) for m in models], rotation=30, ha="right", fontsize=6)
            ax.tick_params(axis="x", which="minor", bottom=False, top=False)
            ax.set_xlim(-0.6, len(models) - 0.4)
            if row_index == 0:
                ax.set_title(label, fontsize=8)
            if column_index == 0:
                ax.set_ylabel(dataset, fontsize=8)
            ax.text(-0.02, 1.02, panel_letter(row_index * len(METRICS) + column_index),
                    transform=ax.transAxes, fontsize=8, fontweight="bold", ha="right", va="bottom")

    handles = [Patch(facecolor=BUDGET_COLORS[i % len(BUDGET_COLORS)], alpha=0.6,
                     label=rf"{SUBTREE_METHOD_LABEL}, prefetch budget $N_{{\mathrm{{pf}}}}={b}$")
               for i, b in enumerate(budgets)]
    handles.append(Line2D([0], [0], color=GRAY, linestyle="--", linewidth=0.8,
                          label=f"max new tokens = {max_new_tokens}"))
    figure.legend(handles=handles, loc="outside upper center", ncols=len(handles), frameon=False, fontsize=7)
    save_figure(figure, path)


def draw_scatter(rows: list[dict], max_new_tokens: int, path: Path) -> None:
    """Draw token log-likelihood against confidence per dataset, colored by response length."""
    datasets = ordered({r["dataset"] for r in rows}, DATASET_ORDER)
    columns = min(4, len(datasets))
    grid_rows = -(-len(datasets) // columns)
    figure, axes = plt.subplots(grid_rows, columns, figsize=(2.8 * columns, 2.5 * grid_rows),
                                squeeze=False, layout="constrained")
    scatter = None
    for index, dataset in enumerate(datasets):
        ax = axes.flat[index]
        subset = [r for r in rows if r["dataset"] == dataset]
        correct = np.array([r["is_correct"] for r in subset])
        x = np.array([r["log_likelihood"] for r in subset])
        y = np.array([r["confidence"] for r in subset])
        c = np.array([r["num_response_tokens"] for r in subset])
        for mask, marker in ((correct, "o"), (~correct, "x")):
            scatter = ax.scatter(x[mask], y[mask], c=c[mask], cmap="viridis", vmin=0, vmax=max_new_tokens,
                                 s=8, marker=marker, linewidths=0.6 if marker == "x" else 0, alpha=0.8)
        ax.set_title(dataset, fontsize=8)
        ax.set_xlabel("Token log-likelihood", fontsize=7)
        ax.set_ylabel("Confidence", fontsize=7)
        ax.text(-0.02, 1.02, panel_letter(index), transform=ax.transAxes, fontsize=8, fontweight="bold",
                ha="right", va="bottom")
    for ax in axes.flat[len(datasets):]:
        ax.set_axis_off()
    figure.colorbar(scatter, ax=axes, label="Response length (tokens)", shrink=0.8)
    figure.legend(handles=[
        Line2D([0], [0], marker="o", linestyle="", color=BLACK, markersize=4, label="correct"),
        Line2D([0], [0], marker="x", linestyle="", color=BLACK, markersize=4, label="incorrect"),
    ], loc="outside upper center", ncols=2, frameon=False, fontsize=7)
    save_figure(figure, path)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT, help="Folder scanned for result CSVs.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--steps", type=int, default=100, help="MH steps of the matched sweep.")
    parser.add_argument("--samples", type=int, default=20, help="Samples per run of the matched sweep.")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--rank", default="accept_first", help="Traversal rank function to keep.")
    parser.add_argument("--budgets", type=int, nargs="+", default=[10, 20], help="Prefetch budgets to compare.")
    parser.add_argument("--min-median-tokens", type=int, default=2,
                        help="Drop runs whose median response length is at most this (failed generations).")
    return parser


def main() -> None:
    """Collect responses, write the source CSV, and draw both figures."""
    args = build_parser().parse_args()
    rows = read_rows(args.log_root, args)
    if not rows:
        raise SystemExit(f"No matching PreSTO result CSVs with log-likelihood under {args.log_root}")
    kept = [r for r in rows if not r["excluded"]]
    budgets = sorted({r["prefetch_budget"] for r in kept})

    apply_plot_style()
    stem = (f"entropycut-presto.rank-{args.rank}.steps{args.steps}.samples{args.samples}"
            f".maxnew{args.max_new_tokens}.likelihood-confidence-length")
    grid_path = args.output_dir / f"{stem}.pdf"
    scatter_path = args.output_dir / f"{stem}.scatter.pdf"
    draw_grid(kept, budgets, args.max_new_tokens, grid_path)
    draw_scatter(kept, args.max_new_tokens, scatter_path)
    outputs = [write_dict_rows(args.output_dir / f"{stem}.csv", rows), grid_path, scatter_path]

    runs = sorted({(r["dataset"], r["model"], r["prefetch_budget"], r["run_date"], r["excluded"]) for r in rows})
    print(f"{len(runs)} runs, {len(rows)} responses ({len(kept)} plotted)")
    for dataset, model, budget, date, excluded in runs:
        if excluded:
            print(f"  excluded (median length <= {args.min_median_tokens}): {dataset} {model} budget {budget} {date}")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
