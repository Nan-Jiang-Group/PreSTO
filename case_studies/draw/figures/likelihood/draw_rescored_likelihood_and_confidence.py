#!/usr/bin/env python3
"""Plot rescored token log-likelihood and confidence under p_0: a sampler with and without PreSTO.

Reads the ``<run>.rescored.csv`` files written by ``power_sharpening.runners.hf.score_log_likelihood_and_confidence``
for one baseline run (PowerMH or EntropyCut) and its PreSTO runs, and draws the two-panel density histogram of
``plot_likelihood_and_confidence.py``: (a) token log-likelihood (1/n) log p_0(x | x_0), (b) token confidence
(1/n) sum_t sum_u p_t(u) log p_t(u), one colour per method, dashed lines at the group means.

A rescored row is exact when the re-encoded completion is the token sequence the sampler produced: the fixed horizon
for the baseline, which ignores EOS, and the logged response length for PreSTO (from the CSV, or from the run log's
"response tokens" lines). The samplers decoded with skip_special_tokens=True, so a completion that contained special
tokens re-encodes shorter. By default only the prompts exact in every group are plotted, so all methods are compared on
the same prompts; --all-samples keeps every row. The figure and its source CSV are written beside the baseline CSV, or
to --output-dir.

Run:

    /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/.venv/bin/python \
        /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/draw/draw_rescored_likelihood_and_confidence.py \
        --budgets 10
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from case_studies import paths
from case_studies.draw.common.figure_io import save_figure, write_dict_rows
from case_studies.draw.figures.likelihood.plot_likelihood_and_confidence import (
    HIST_ALPHA,
    HIST_LINE_WIDTH,
    MEAN_LINE_WIDTH,
    PANEL_HEIGHT,
    PANEL_SPECS,
    PANEL_WIDTH,
    add_panel_label,
    draw_histogram,
    shared_bin_edges,
)
from case_studies.extract.run_naming import prefetch_budget_of, strip_batch_size
from case_studies.plot_config import (
    BLACK,
    BLUE,
    COLOR_PALETTE,
    GRAY,
    GREEN,
    POWER_METHOD_LABEL,
)

CASE_STUDIES_DIR = paths.CASE_STUDIES_DIR
DEFAULT_RUN_DIR = CASE_STUDIES_DIR / "logs/lcb_v6/2026-09-23/vllm"
DEFAULT_GLOB = "dataset-lcb_v6.model-qwen3.5-4b.*.rescored.csv"
DEFAULT_BINS = 12
OUTPUT_SUFFIX = ".rescored-likelihood-and-confidence"
# Baseline runner methods and their figure labels; the PreSTO variant of each is "PreSTO-<label>".
BASELINE_LABELS = {
    "power_sampling_mcmc": POWER_METHOD_LABEL,
    "entropycut_power_sampling_mcmc": "EntropyCut",
}
# vLLM PreSTO prints this line bare; SGLang PreSTO logs it with the logger prefix.
RESPONSE_TOKENS = re.compile(r"^(?:\[[^\]\n]+\] INFO: )?response length: \d+, response tokens: (\d+)", re.M)
csv.field_size_limit(1 << 30)


def budget_of(label: str) -> int | None:
    match = re.search(r"=(\d+)", label)
    return int(match.group(1)) if match else None


# One PreSTO run per pair: N_pf=20 when it was run, otherwise N_pf=10.
PREFERRED_BUDGETS = (20, 10)


def one_presto_budget(groups: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """Keep the baseline and a single PreSTO budget, the first of PREFERRED_BUDGETS that the pair has."""
    budgets = {budget_of(label): label for label in groups if budget_of(label) is not None}
    chosen = next((budgets[b] for b in PREFERRED_BUDGETS if b in budgets), budgets[min(budgets)])
    return {label: rows for label, rows in groups.items() if budget_of(label) is None or label == chosen}


def presto_label(baseline_label: str, budget: int) -> str:
    return rf"PreSTO-{baseline_label}, $N_{{\mathrm{{pf}}}}={budget}$"


def logged_response_lengths(rescored: Path, rows: list[dict]) -> list[int | None]:
    """Return the sampler's own response length per row: from the CSV, else from the run log beside it."""
    if all(row.get("logged_num_response_tokens") for row in rows):
        return [int(row["logged_num_response_tokens"]) for row in rows]
    stem = rescored.name.removesuffix(".rescored.csv")
    for log in sorted(rescored.parent.glob(f"{stem}.*.log")):
        lengths = [int(n) for n in RESPONSE_TOKENS.findall(log.read_text(errors="replace"))]
        if len(lengths) >= len(rows):
            return lengths[:len(rows)]
    return [None] * len(rows)


def read_run(path: Path) -> list[dict]:
    """Read one rescored CSV and mark each row ``exact`` when its tokens are the ones the sampler produced."""
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    is_baseline = bool(rows) and rows[0]["method"] in BASELINE_LABELS
    logged = [None] * len(rows) if is_baseline else logged_response_lengths(path, rows)
    for row, length in zip(rows, logged):
        row["source_csv"] = str(path)
        if is_baseline:
            row["exact"] = row["round_trip_exact"] == "True"
        else:
            row["exact"] = length is not None and int(row["num_response_tokens"]) == length
    return rows


def read_groups(paths: list[Path], budgets: list[int] | None = None) -> dict[str, list[dict]]:
    """Return rows keyed by figure label: the baseline first, then PreSTO budgets in ascending order.

    ``budgets=None`` keeps every PreSTO budget among the inputs.
    """
    baseline: dict[str, list[dict]] = {}
    presto: dict[int, list[dict]] = {}
    for path in paths:
        rows = read_run(path)
        if not rows:
            continue
        if rows[0]["method"] in BASELINE_LABELS:
            if baseline:
                raise SystemExit(f"more than one baseline among the inputs: {path}")
            baseline[BASELINE_LABELS[rows[0]["method"]]] = rows
        elif budgets is None or prefetch_budget_of(path.name) in budgets:
            presto[prefetch_budget_of(path.name)] = rows
    if not baseline:
        raise SystemExit("no PowerMH or EntropyCut rescored CSV among the inputs")
    missing = sorted(set(budgets or ()) - set(presto))
    if missing or not presto:
        raise SystemExit(f"no PreSTO rescored CSV for prefetch budget(s) {missing or budgets}")
    (label, rows), = baseline.items()
    groups = {label: rows}
    for budget in sorted(presto):
        groups[presto_label(label, budget)] = presto[budget]
    return groups


def restrict_to_exact(groups: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """Keep only the prompts whose rows are exact in every group."""
    exact = set.intersection(*({row["sample_idx"] for row in rows if row["exact"]} for rows in groups.values()))
    return {label: [row for row in rows if row["sample_idx"] in exact] for label, rows in groups.items()}


def group_colors(labels: list[str]) -> dict[str, str]:
    """The baseline in blue and PreSTO in green, as in the other method-comparison figures; extra budgets cycle on."""
    extra = [color for color in COLOR_PALETTE if color not in (BLUE, GREEN)]
    colors = {}
    for index, label in enumerate(labels):
        colors[label] = BLUE if index == 0 else GREEN if index == 1 else extra[(index - 2) % len(extra)]
    return colors


def plot(groups: dict[str, list[dict]], bins: int, path: Path, title: str | None = None,
         figsize: tuple[float, float] | None = None, font_size: float | None = None) -> Path:
    """Draw the log-likelihood and confidence panels side by side."""
    labels = list(groups)
    colors = group_colors(labels)
    if font_size is not None:
        # Every text element at one size, for a figure drawn near the width it is printed at.
        mpl.rcParams.update({key: font_size for key in ("font.size", "axes.labelsize", "xtick.labelsize",
                                                          "ytick.labelsize", "legend.fontsize")})
    figure, axes = plt.subplots(1, len(PANEL_SPECS), figsize=figsize or (PANEL_WIDTH * 2.3, PANEL_HEIGHT * 1.3),
                                layout="constrained")
    for index, (ax, (value_key, x_label)) in enumerate(zip(axes, PANEL_SPECS)):
        edges = shared_bin_edges([float(row[value_key]) for rows in groups.values() for row in rows], bins)
        draw_histogram(ax, groups, labels, colors, value_key=value_key, x_label=x_label, edges=edges,
                       show_y_label=True)
        add_panel_label(ax, f"({chr(ord('a') + index)})")
    handles: list[Patch | Line2D] = [
        Patch(facecolor=colors[label], edgecolor=BLACK, linewidth=HIST_LINE_WIDTH, alpha=HIST_ALPHA,
              label=f"{label} ($n$={len(groups[label])})")
        for label in labels
    ]
    handles.append(Line2D([0], [0], color=GRAY, linestyle="--", linewidth=MEAN_LINE_WIDTH, label="Group mean"))
    if title:
        # The title takes the top edge, so the legend moves below the panels.
        figure.suptitle(title)
        figure.legend(handles=handles, loc="outside lower center", ncols=min(len(handles), 2), frameon=False)
    else:
        # A narrow figure cannot hold every entry on one line; two columns keep the legend inside its width.
        columns = len(handles) if figsize is None else 2
        figure.legend(handles=handles, loc="outside upper center", ncols=columns, frameon=False)
    return save_figure(figure, path)


def output_prefix(groups: dict[str, list[dict]], all_samples: bool, output_dir: Path | None) -> Path:
    """Name the outputs after the baseline run, its PreSTO budgets, and the sample subset."""
    baseline_csv = Path(next(iter(groups.values()))[0]["source_csv"])
    stem = strip_batch_size(baseline_csv.name.removesuffix(".rescored.csv"))
    budgets = "-".join(re.search(r"=(\d+)", label).group(1) for label in list(groups)[1:])
    subset = "all-samples" if all_samples else "exact-samples"
    return (output_dir or baseline_csv.parent) / f"{stem}.prefetch-budget-{budgets}.{subset}{OUTPUT_SUFFIX}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--csv", type=Path, nargs="+",
                        help=f"Rescored CSVs of one baseline and its PreSTO runs; default: {DEFAULT_GLOB} under "
                             f"{DEFAULT_RUN_DIR}.")
    parser.add_argument("--budgets", type=int, nargs="+",
                        help="PreSTO prefetch budgets to overlay on the baseline; default: all among the inputs.")
    parser.add_argument("--all-samples", action="store_true",
                        help="Plot every row instead of only the prompts exact in every group.")
    parser.add_argument("--bins", type=int, default=DEFAULT_BINS)
    parser.add_argument("--output-dir", type=Path, help="Where to write the figure; default: beside the inputs.")
    parser.add_argument("--figsize", type=float, nargs=2, metavar=("WIDTH", "HEIGHT"),
                        help="Figure size in inches; draw near the printed width so fonts keep their size.")
    parser.add_argument("--font-size", type=float, help="Font size in points for every text element.")
    parser.add_argument("--output", type=Path, help="Exact PDF path, overriding the derived name.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    paths = args.csv or sorted(DEFAULT_RUN_DIR.glob(DEFAULT_GLOB))
    groups = read_groups(paths, args.budgets)
    if not args.all_samples:
        groups = restrict_to_exact(groups)
    if any(not rows for rows in groups.values()):
        raise SystemExit("a group has no samples left to plot")

    prefix = output_prefix(groups, args.all_samples, args.output_dir)
    figure_path = plot(groups, args.bins, args.output or prefix.with_name(prefix.name + ".pdf"),
                       figsize=tuple(args.figsize) if args.figsize else None, font_size=args.font_size)
    table_path = write_dict_rows(prefix.with_name(prefix.name + ".csv"),
                                 [{"group": label, **row} for label, rows in groups.items() for row in rows])
    for label, rows in groups.items():
        print(f"{label}: n={len(rows)}, samples {[int(row['sample_idx']) for row in rows]}")
    print(figure_path)
    print(table_path)


if __name__ == "__main__":
    main()
