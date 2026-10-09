"""Draw the accuracy scaling figures from ``accuracy_scaling.csv``.

Run from the repository root after ``collect_accuracy``:
    PYTHONPATH=/Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening \
        python -m case_studies.scaling_baselines.draw_accuracy [--table FILE] [--output-dir DIR]

Writes PDFs to ``figures/``:
    low_temp.temperature.pdf     accuracy against temperature, one panel per dataset, one line per base LLM
    best_of_n.samples.temp<T>.pdf accuracy against N, same layout (one figure per sampling temperature)
    power_smc.particles.pdf      accuracy against the particle count, same layout
    methods.pdf                  macro-average accuracy over datasets, one panel per base LLM: best-of-N and Power-SMC
                                 against N, with the best low-temperature accuracy as a reference line
Error bars are one binomial standard error.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from case_studies.plot_config import COLOR_PALETTE, apply_plot_style

SCALING_DIR = Path(__file__).resolve().parent
DATASET_LABELS = {
    "math500": "MATH500", "aime": "AIME 24&25", "mbpp": "MBPP", "gpqa": "GPQA", "human_eval": "HumanEval",
    "mmlu": "MMLU", "lcb_v6": "LCB V6",
}
MODEL_LABELS = {
    "qwen3-4b": "Qwen3-4B", "qwen3-8b": "Qwen3-8B", "qwen3.5-4b": "Qwen3.5-4B", "qwen3.5-9b": "Qwen3.5-9B",
    "gemma-12b-it": "Gemma4-12B-it",
}
AXIS_LABELS = {"temperature": "Temperature", "best_of_n": "Samples $N$", "n_particles": "Particles $N$"}
LOG_AXES = {"best_of_n", "n_particles"}


def _color(model: str, models: list[str]) -> str:
    return COLOR_PALETTE[models.index(model) % len(COLOR_PALETTE)]


def _set_axis(ax, axis: str, values) -> None:
    if axis in LOG_AXES:
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(values))
        ax.set_xticklabels([str(int(v)) for v in sorted(values)])
        ax.minorticks_off()


def draw_by_dataset(table: pd.DataFrame, method: str, output: Path, title: str) -> None:
    """One panel per dataset; one line per base LLM."""
    data = table[table["method"] == method]
    if data.empty:
        return
    axis = data["axis"].iloc[0]
    datasets = [d for d in DATASET_LABELS if d in set(data["dataset"])]
    models = [m for m in MODEL_LABELS if m in set(data["model"])]
    columns = min(4, len(datasets))
    rows = -(-len(datasets) // columns)
    fig, axes = plt.subplots(rows, columns, figsize=(3.0 * columns, 2.4 * rows), squeeze=False, sharey=False)
    for ax in axes.flat[len(datasets):]:
        ax.set_visible(False)
    for ax, dataset in zip(axes.flat, datasets):
        for model in models:
            cell = data[(data["dataset"] == dataset) & (data["model"] == model)].sort_values("value")
            if cell.empty:
                continue
            ax.errorbar(cell["value"], 100 * cell["accuracy"], yerr=100 * cell["stderr"], marker="o", ms=3,
                        lw=1.2, capsize=2, color=_color(model, models), label=MODEL_LABELS[model])
        _set_axis(ax, axis, data["value"].unique())
        ax.set_title(DATASET_LABELS[dataset])
        ax.set_xlabel(AXIS_LABELS[axis])
        ax.set_ylabel("Accuracy (\\%)")
    axes.flat[0].legend(fontsize=6, loc="best")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    print(f"wrote {output}")


def draw_methods(table: pd.DataFrame, output: Path) -> None:
    """Macro-average accuracy over datasets, one panel per base LLM, methods compared on the same N axis."""
    models = [m for m in MODEL_LABELS if m in set(table["model"])]
    if not models:
        return
    fig, axes = plt.subplots(1, len(models), figsize=(2.8 * len(models), 2.6), squeeze=False)
    for ax, model in zip(axes[0], models):
        part = table[table["model"] == model]
        low = part[part["method"] == "low_temp"].groupby(["value", "dataset"])["accuracy"].mean().groupby("value").mean()
        if not low.empty:
            ax.axhline(100 * low.max(), color=COLOR_PALETTE[5], ls="--", lw=1, label="Low-temp (best temp.)")
        for method, label, color in (("best_of_n", "Best-of-$N$", COLOR_PALETTE[0]),
                                     ("power_smc", "Power-SMC", COLOR_PALETTE[3])):
            cell = part[part["method"] == method]
            if method == "best_of_n" and not cell.empty:
                # Several sampling temperatures: plot the best one at each N.
                cell = cell.groupby(["setting", "value", "dataset"])["accuracy"].mean().groupby(
                    ["setting", "value"]).mean().groupby("value").max().reset_index()
            elif not cell.empty:
                cell = cell.groupby(["value", "dataset"])["accuracy"].mean().groupby("value").mean().reset_index()
            if cell.empty:
                continue
            ax.plot(cell["value"], 100 * cell["accuracy"], marker="o", ms=3, lw=1.2, color=color, label=label)
        ax.set_xscale("log", base=2)
        ax.set_title(MODEL_LABELS[model])
        ax.set_xlabel("$N$ (samples or particles)")
        ax.set_ylabel("Mean accuracy over datasets (\\%)")
    axes[0][0].legend(fontsize=6, loc="best")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    print(f"wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--table", type=Path, default=SCALING_DIR / "accuracy_scaling.csv")
    parser.add_argument("--output-dir", type=Path, default=SCALING_DIR / "figures")
    args = parser.parse_args()
    table = pd.read_csv(args.table)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    apply_plot_style()
    draw_by_dataset(table, "low_temp", args.output_dir / "low_temp.temperature.pdf",
                    "Low-temperature sampling: accuracy against temperature")
    best = table[table["method"] == "best_of_n"]
    for temperature in sorted(best["setting"].dropna().unique()):
        draw_by_dataset(table[(table["method"] != "best_of_n") | (table["setting"] == temperature)], "best_of_n",
                        args.output_dir / f"best_of_n.samples.temp{temperature:g}.pdf",
                        f"Best-of-$N$ (temperature {temperature:g}): accuracy against $N$")
    draw_by_dataset(table, "power_smc", args.output_dir / "power_smc.particles.pdf",
                    "Power-SMC: accuracy against the number of particles")
    draw_methods(table, args.output_dir / "methods.pdf")


if __name__ == "__main__":
    main()
