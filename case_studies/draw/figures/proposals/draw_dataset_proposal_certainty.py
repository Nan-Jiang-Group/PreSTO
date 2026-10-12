#!/usr/bin/env python3
"""Plot proposal certainty with one panel per benchmark dataset.

Run the current Qwen3.5-9B, budget-20 BFS-accept comparison with:

    MPLCONFIGDIR=/private/tmp/proposal-certainty-mpl \
      /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/.venv/bin/python \
      /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/draw/draw_dataset_proposal_certainty.py

Add ``--hatch`` to write a separately named variant with unfilled colored hatches.
Add ``--all-edges`` to count acceptance and rejection edges with probabilities A
and 1-A, including zero-probability edges. The default counts A once per proposal.
Use ``--run-date 2026-09-16 --model MODEL`` for the Qwen3-4B, Qwen3.5-4B,
Qwen3-8B, and Gemma4-12B-it runs (keys: qwen3-4b, qwen3.5-4b, qwen3-8b, gemma-12b-it).
Use explicit ``--output`` and ``--csv-output`` paths to replace an existing figure.

For the EntropyCut logs, whose datasets ran on different dates and whose names
carry a ``cut-entropy4.0`` tag after the rank, use:

    --log-root /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/logs-entropycut \
      --run-date latest --method-tag cut-entropy4.0 --rank-fn accept_first \
      --method-label PreSTO-EntropyCut --model qwen3.5-9b --hatch --all-edges \
      --datasets math500 aime mbpp gpqa lcb_v6

``--run-date latest`` takes each dataset's newest date directory holding a
matching log; the figure is written under the newest date used.

The default comparison uses the September 5 GPQA rerun. Until it finishes, only completed questions are included and its
panel is marked as partial.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import NullLocator

from case_studies import paths
from case_studies.draw.common.figure_io import save_figure, write_dict_rows
from case_studies.draw.common.plot_helpers import panel_letter
from case_studies.draw.common.rank_plotting import style_quantitative_axis
from case_studies.draw.figures.proposals.draw_rank_proposal_certainty import (
    ACCEPTANCE_XLABEL,
    EDGE_BUCKET_NAMES,
    LEGEND_HANDLE_HEIGHT,
    LEGEND_HANDLE_LENGTH,
    Y_AXIS_MIN,
    Y_LIMIT_HEADROOM,
    acceptance_tick_position,
    certainty_legend_handles,
    log_acceptance_buckets,
    plot_piecewise_log_acceptance_mass,
    style_piecewise_log_acceptance_axis,
)
from case_studies.extract.logs.proposal_log import TreeNode, parse_log
from case_studies.plot_config import SUBTREE_METHOD_LABEL

DEFAULT_DATASETS = (
    "math500",
    "aime",
    "mbpp",
    "gpqa",
    "human_eval",
    "lcb_v6",
    "mmlu",
)
DATASET_LABELS = {
    "math500": "MATH500",
    "aime": r"AIME 24\&25",
    "mbpp": "MBPP",
    "gpqa": "GPQA",
    "human_eval": "HumanEval",
    "lcb_v6": "LCB V6",
    "mmlu": "MMLU",
}
MODEL_LABELS = {
    "qwen3-4b": "Qwen3-4B",
    "qwen3.5-4b": "Qwen3.5-4B",
    "qwen3-8b": "Qwen3-8B",
    "qwen3.5-9b": "Qwen3.5-9B",
    "gemma-12b-it": "Gemma4-12B-it",
}

DEFAULT_RUN_DATE = "2026-09-04"
# Select each dataset's newest date directory that holds a matching log.
LATEST_RUN_DATE = "latest"
# Historical partial reruns for the default comparison. Other runs require a
# completion marker unless explicitly selected with --allow-partial.
DEFAULT_PARTIAL_RERUN_DATES = {"gpqa": "2026-09-05"}
DEFAULT_MODEL = "qwen3.5-9b"
DEFAULT_PREFETCH_BUDGET = 20
DEFAULT_RANK_FN = "bfs_accept_first"
DEFAULT_ALPHA = "4.0"
DEFAULT_MCMC_STEPS = 100
DEFAULT_NUM_BLOCKS = 1
DEFAULT_MAX_SAMPLES = 20
DEFAULT_MAX_NEW_TOKENS = 1024
DEFAULT_SEED = 10086
BACKEND = "vllm"
COMPLETION_MARKER = "INFO: output saved to"
FIGURE_SUFFIX = ".proposal-certainty.pdf"
CSV_SUFFIX = ".proposal-certainty.csv"

# Narrow panels share one row; bucket boundaries and mass are unchanged.
DATASET_PANEL_WIDTH = 1.50
DATASET_PANEL_HEIGHT = 1.75
DATASET_HEADER_HEIGHT = 1.15
DATASET_TITLE_FONT_SIZE = 12
DATASET_SUPTITLE_FONT_SIZE = 14
DATASET_YLABEL_FONT_SIZE = 14
DATASET_XLABEL_FONT_SIZE = 16
DATASET_TICK_FONT_SIZE = 11
DATASET_LEGEND_FONT_SIZE = 14
DATASET_BIN_COUNTS = (1, 3, 1)
# Give the outer percentages room without changing bins or total panel width.
DATASET_BUCKET_DISPLAY_WIDTHS = (3.25, 4.50, 3.25)
DATASET_BAR_WIDTH_FRACTION = 0.65
# Keep the outer bars at their original width while widening their groups.
DATASET_MAX_BAR_DISPLAY_WIDTH = 1.30
DATASET_LINE_COLOR = "#B0B0B0"


@dataclass(frozen=True)
class DatasetRun:
    """One dataset's proposal nodes and complete, partial, or missing status."""

    dataset: str
    log_path: Path
    status: str
    nodes: tuple[TreeNode, ...] = ()


def run_log_path(
    log_root: Path,
    dataset: str,
    run_date: str,
    model: str,
    prefetch_budget: int,
    rank_fn: str,
    alpha: str,
    mcmc_steps: int,
    num_blocks: int,
    max_samples: int,
    max_new_tokens: int,
    seed: int,
    method_tag: str = "",
) -> Path:
    """Build the exact log path for one dataset under a shared run setting.

    ``run_date="latest"`` resolves to the newest date directory holding the log.
    """
    tag = f"{method_tag}." if method_tag else ""
    name = (
        f"dataset-{dataset}.model-{model}.alpha{alpha}.steps{mcmc_steps}."
        f"blocks-{num_blocks}.prefetch-budget-{prefetch_budget}.rank-{rank_fn}."
        f"{tag}samples{max_samples}.maxnew{max_new_tokens}.seed{seed}."
        f"subtreePrefetch.{BACKEND}.log"
    )
    if run_date == LATEST_RUN_DATE:
        candidates = sorted((log_root / dataset).glob(f"*/{BACKEND}/{name}"))
        if candidates:
            return candidates[-1]
        return log_root / dataset / LATEST_RUN_DATE / BACKEND / name
    return log_root / dataset / run_date / BACKEND / name


def log_completed(log_path: Path) -> bool:
    """Report whether the runner reached its output-save marker."""
    if not log_path.is_file() or log_path.stat().st_size == 0:
        return False
    with log_path.open("r", encoding="utf-8", errors="replace") as log_file:
        return any(COMPLETION_MARKER in line for line in log_file)


def completed_question_prefix(log_path: Path) -> tuple[int, int]:
    """Return the completed-question count and its last included line number."""
    completed_questions = 0
    last_completed_line = 0
    in_question = False
    with log_path.open("r", encoding="utf-8", errors="replace") as log_file:
        for line_number, line in enumerate(log_file, start=1):
            if line.startswith("IDX ") and "QUESTION:" in line:
                in_question = True
            elif in_question and line.startswith("used time: each "):
                completed_questions += 1
                last_completed_line = line_number
                in_question = False
    return completed_questions, last_completed_line


def collect_dataset_runs(args: argparse.Namespace) -> list[DatasetRun]:
    """Load complete runs and explicitly approved completed-question prefixes."""
    runs: list[DatasetRun] = []
    for dataset in args.datasets:
        default_partial_rerun = (
            args.run_date == DEFAULT_RUN_DATE
            and dataset in DEFAULT_PARTIAL_RERUN_DATES
        )
        allow_partial = default_partial_rerun or dataset in args.allow_partial
        run_date = (
            DEFAULT_PARTIAL_RERUN_DATES[dataset]
            if default_partial_rerun
            else args.run_date
        )
        log_path = run_log_path(
            args.log_root,
            dataset,
            run_date,
            args.model,
            args.prefetch_budget,
            args.rank_fn,
            args.alpha,
            args.mcmc_steps,
            args.num_blocks,
            args.max_samples,
            args.max_new_tokens,
            args.seed,
            args.method_tag,
        )
        if not log_path.is_file():
            runs.append(DatasetRun(dataset, log_path, "log not available"))
            continue
        max_lines = None
        status = "complete"
        if not log_completed(log_path):
            if not allow_partial:
                runs.append(DatasetRun(dataset, log_path, "rerun required"))
                continue
            completed_questions, max_lines = completed_question_prefix(log_path)
            if not completed_questions:
                runs.append(DatasetRun(dataset, log_path, "rerun required"))
                continue
            status = f"partial: {completed_questions}/{args.max_samples} questions"

        parsed = parse_log(log_path, max_lines=max_lines)
        nodes = tuple(parsed.nodes)
        if not nodes:
            runs.append(DatasetRun(dataset, log_path, "rerun required"))
            continue
        if args.all_edges:
            printed_edges = sum(len(ids) - 1 for ids in parsed.tree_node_ids.values())
            complete_pairs = all(
                {2 * node.node_id + 1, 2 * node.node_id + 2}.issubset(
                    parsed.tree_node_ids[node.tree_index]
                )
                for node in nodes
            )
            if not complete_pairs or printed_edges != 2 * len(nodes):
                raise ValueError(
                    f"cannot count all edges in {log_path}: "
                    "each scored proposal must have both children and every "
                    "printed edge must have a scored parent"
                )
        runs.append(DatasetRun(dataset, log_path, status, nodes))

    if not any(run.nodes for run in runs):
        raise ValueError("no proposal data available for the requested dataset runs")
    return runs


def draw_dataset_panel(
    ax, run: DatasetRun, *, hatched: bool = False, all_edges: bool = False
) -> float:
    """Draw one dataset panel and return its largest proposal-bin share."""
    if run.nodes:
        maximum_bin_share = plot_piecewise_log_acceptance_mass(
            ax,
            list(run.nodes),
            bin_counts=DATASET_BIN_COUNTS,
            bin_width_fraction=DATASET_BAR_WIDTH_FRACTION,
            max_bar_display_width=DATASET_MAX_BAR_DISPLAY_WIDTH,
            inset_annotations=False,
            show_separators=True,
            bucket_display_widths=DATASET_BUCKET_DISPLAY_WIDTHS,
            hatched=hatched,
            all_edges=all_edges,
        )
        for annotation in ax.texts:
            annotation.set_bbox(
                {
                    "facecolor": "white",
                    "edgecolor": "none",
                    "boxstyle": "square,pad=0.10",
                    "alpha": 1.0,
                }
            )
        if run.status.startswith("partial:"):
            ax.text(
                0.5,
                0.78,
                run.status.replace(": ", ":\n").replace(
                    " questions", "\nquestions"
                ),
                transform=ax.transAxes,
                color="0.45",
                fontsize=9,
                ha="center",
                va="top",
            )
    else:
        maximum_bin_share = 0.0
        style_piecewise_log_acceptance_axis(
            ax,
            show_separators=True,
            bucket_display_widths=DATASET_BUCKET_DISPLAY_WIDTHS,
        )
        ax.text(
            0.5,
            0.5,
            run.status,
            transform=ax.transAxes,
            color="0.55",
            fontsize=11,
            ha="center",
            va="center",
        )
    ax.set_xlabel("")
    ax.set_ylabel("")
    style_quantitative_axis(ax)
    ax.yaxis.set_major_locator(NullLocator())
    ax.grid(axis="y", visible=False)
    for separator in ax.lines:
        separator.set_color(DATASET_LINE_COLOR)
    ax.spines["bottom"].set_color(DATASET_LINE_COLOR)
    ax.tick_params(axis="both", labelsize=DATASET_TICK_FONT_SIZE)
    ax.tick_params(
        axis="y",
        which="both",
        left=False,
        right=False,
        labelleft=False,
        labelright=False,
    )
    ax.set_xticks(
        [
            acceptance_tick_position(
                value, bucket_display_widths=DATASET_BUCKET_DISPLAY_WIDTHS
            )
            for value in (0.01, 0.99)
        ],
        ["0.01", "0.99"],
    )
    # Center each boundary label directly below its group-separating line.
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_ha("center")
    return maximum_bin_share


def draw_figure(
    runs: Sequence[DatasetRun],
    *,
    model: str,
    prefetch_budget: int,
    rank_fn: str,
    hatched: bool = False,
    all_edges: bool = False,
    method_label: str = SUBTREE_METHOD_LABEL,
) -> Figure:
    """Draw proposal certainty in one compact row of dataset panels."""
    columns = len(runs)
    figure, axes = plt.subplots(
        1,
        columns,
        figsize=(
            DATASET_PANEL_WIDTH * columns + 0.55,
            DATASET_PANEL_HEIGHT + DATASET_HEADER_HEIGHT,
        ),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    maximum_bin_share = 0.0
    for index, run in enumerate(runs):
        ax = axes[0, index]
        maximum_bin_share = max(
            maximum_bin_share,
            draw_dataset_panel(ax, run, hatched=hatched, all_edges=all_edges),
        )
        ax.tick_params(axis="x", labelbottom=True)
        ax.set_title(
            f"{panel_letter(index)} "
            f"{DATASET_LABELS.get(run.dataset, run.dataset)}",
            color="black",
            fontsize=DATASET_TITLE_FONT_SIZE,
            pad=4.0,
        )

    figure.supylabel(
        r"edge share (\%)" if all_edges else r"edge category (\%)",
        fontsize=DATASET_YLABEL_FONT_SIZE,
        x=0.30 / figure.get_figwidth(),
        y=0.465,
    )
    axes[0, 0].set_ylim(
        Y_AXIS_MIN,
        max(10.0, Y_LIMIT_HEADROOM * maximum_bin_share),
    )

    figure.subplots_adjust(
        left=0.62 / figure.get_figwidth(),
        right=0.995,
        bottom=0.50 / figure.get_figheight(),
        top=1.0 - 0.98 / figure.get_figheight(),
        wspace=0.18,
    )
    figure.suptitle(
        f"base LLM: {MODEL_LABELS.get(model, model)}, "
        f"Method: {method_label}, inference engine: vLLM",
        fontsize=DATASET_SUPTITLE_FONT_SIZE,
        y=1.0 - 0.28 / figure.get_figheight(),
    )
    figure.supxlabel(
        "Edge transition probability" if all_edges else ACCEPTANCE_XLABEL,
        y=0.01,
        fontsize=DATASET_XLABEL_FONT_SIZE,
    )
    figure.legend(
        handles=certainty_legend_handles(hatched=hatched),
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.53, 1.0 - 0.52 / figure.get_figheight()),
        borderaxespad=0.0,
        borderpad=0.0,
        ncol=3,
        fontsize=DATASET_LEGEND_FONT_SIZE,
        handlelength=LEGEND_HANDLE_LENGTH,
        handleheight=LEGEND_HANDLE_HEIGHT,
        handletextpad=0.3,
        columnspacing=0.55,
    )
    return figure


SOURCE_FIELDS = (
    "dataset",
    "status",
    "log_path",
    "model",
    "prefetch_budget",
    "rank_fn",
    "proposal_nodes",
    "acceptance_bucket",
    "bucket_share",
    "bin_in_bucket",
    "acceptance_left",
    "acceptance_right",
    "proposal_count",
    "proposal_share",
)
EDGE_SOURCE_FIELDS = (
    "dataset",
    "status",
    "log_path",
    "model",
    "prefetch_budget",
    "rank_fn",
    "proposal_nodes",
    "tree_edges",
    "edge_bucket",
    "bucket_share",
    "bin_in_bucket",
    "probability_left",
    "probability_right",
    "edge_count",
    "edge_share",
)


def source_rows(
    runs: Sequence[DatasetRun],
    *,
    model: str,
    prefetch_budget: int,
    rank_fn: str,
    all_edges: bool = False,
) -> list[dict[str, object]]:
    """Return histogram source rows plus an explicit row for each missing run."""
    rows: list[dict[str, object]] = []
    for run in runs:
        common: dict[str, object] = {
            "dataset": run.dataset,
            "status": run.status,
            "log_path": str(run.log_path),
            "model": model,
            "prefetch_budget": prefetch_budget,
            "rank_fn": rank_fn,
            "proposal_nodes": len(run.nodes),
        }
        observation_count = len(run.nodes) * (2 if all_edges else 1)
        if all_edges:
            common["tree_edges"] = observation_count
        if not run.nodes:
            rows.append(common)
            continue

        for bucket_name, bin_edges, counts, bucket_share in (
            log_acceptance_buckets(
                list(run.nodes),
                bin_counts=DATASET_BIN_COUNTS,
                all_edges=all_edges,
            )
        ):
            for local_index, count_value in enumerate(counts):
                count = int(count_value)
                rows.append(
                    {
                        **common,
                        ("edge_bucket" if all_edges else "acceptance_bucket"): (
                            EDGE_BUCKET_NAMES[bucket_name] if all_edges else bucket_name
                        ),
                        "bucket_share": f"{bucket_share:.6f}",
                        "bin_in_bucket": local_index,
                        ("probability_left" if all_edges else "acceptance_left"): (
                            f"{bin_edges[local_index]:.8e}"
                        ),
                        ("probability_right" if all_edges else "acceptance_right"): (
                            f"{bin_edges[local_index + 1]:.8e}"
                        ),
                        ("edge_count" if all_edges else "proposal_count"): count,
                        ("edge_share" if all_edges else "proposal_share"): (
                            f"{count / observation_count:.6f}"
                        ),
                    }
                )
    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the cross-dataset proposal-certainty CLI."""
    parser = argparse.ArgumentParser(
        description="Plot proposal certainty with one panel per dataset."
    )
    parser.add_argument(
        "--log-root",
        type=Path,
        default=paths.LOGS_DIR,
    )
    parser.add_argument(
        "--run-date",
        default=DEFAULT_RUN_DATE,
        help=f"Date directory, or '{LATEST_RUN_DATE}' for each dataset's newest match.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--method-tag",
        default="",
        help="Log-name component after the rank, e.g. cut-entropy4.0 for EntropyCut.",
    )
    parser.add_argument("--method-label", default=SUBTREE_METHOD_LABEL)
    parser.add_argument(
        "--prefetch-budget",
        type=int,
        default=DEFAULT_PREFETCH_BUDGET,
    )
    parser.add_argument("--rank-fn", default=DEFAULT_RANK_FN)
    parser.add_argument("--alpha", default=DEFAULT_ALPHA)
    parser.add_argument("--mcmc-steps", type=int, default=DEFAULT_MCMC_STEPS)
    parser.add_argument("--num-blocks", type=int, default=DEFAULT_NUM_BLOCKS)
    parser.add_argument("--max-samples", type=int, default=DEFAULT_MAX_SAMPLES)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
    )
    parser.add_argument(
        "--allow-partial",
        nargs="+",
        default=[],
        metavar="DATASET",
        help="Include only completed-question prefixes for these unfinished dataset logs.",
    )
    parser.add_argument(
        "--all-edges",
        action="store_true",
        help="Count both acceptance (A) and rejection (1-A) edges in each logged tree.",
    )
    parser.add_argument(
        "--hatch",
        action="store_true",
        help="Use unfilled colored hatches for bars and matching legend handles.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--csv-output", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    args.log_root = args.log_root.resolve()
    if len(set(args.datasets)) != len(args.datasets):
        raise SystemExit("--datasets contains a duplicate name")
    try:
        runs = collect_dataset_runs(args)
        figure = draw_figure(
            runs,
            model=args.model,
            prefetch_budget=args.prefetch_budget,
            rank_fn=args.rank_fn,
            hatched=args.hatch,
            all_edges=args.all_edges,
            method_label=args.method_label,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error

    output_date = args.run_date
    if output_date == LATEST_RUN_DATE:
        # The date component of each found log: <dataset>/<date>/<backend>/<log>.
        output_date = max(
            run.log_path.parent.parent.name for run in runs if run.nodes
        )
    output_directory = (
        args.log_root / "all-datasets" / output_date / BACKEND
    )
    stem = (
        f"dataset-all.model-{args.model}."
        f"prefetch-budget-{args.prefetch_budget}.rank-{args.rank_fn}"
    )
    if args.method_tag:
        stem += f".{args.method_tag}"
    if args.all_edges:
        stem += ".all-edges"
    figure_suffix = (
        FIGURE_SUFFIX.replace(".pdf", ".hatched.pdf") if args.hatch else FIGURE_SUFFIX
    )
    csv_suffix = (
        CSV_SUFFIX.replace(".csv", ".hatched.csv") if args.hatch else CSV_SUFFIX
    )
    output = (
        args.output.resolve()
        if args.output
        else output_directory / f"{stem}{figure_suffix}"
    )
    csv_output = (
        args.csv_output.resolve()
        if args.csv_output
        else output_directory / f"{stem}{csv_suffix}"
    )

    save_figure(figure, output, pad_inches=0.02)
    write_dict_rows(
        csv_output,
        source_rows(
            runs,
            model=args.model,
            prefetch_budget=args.prefetch_budget,
            rank_fn=args.rank_fn,
            all_edges=args.all_edges,
        ),
        fieldnames=EDGE_SOURCE_FIELDS if args.all_edges else SOURCE_FIELDS,
    )
    complete = sum(run.status == "complete" for run in runs)
    partial = sum(run.status.startswith("partial:") for run in runs)
    print(f"wrote {output}")
    print(f"wrote {csv_output}")
    print(f"complete dataset panels: {complete}/{len(runs)}")
    if partial:
        print(f"partial dataset panels: {partial}/{len(runs)}")


if __name__ == "__main__":
    main()
