#!/usr/bin/env python3
"""Plot proposal certainty for several models in one grid (rows: models, columns: datasets).

Each row reuses the dataset panels of ``draw_dataset_proposal_certainty.py``; the
legend and x-axis label are drawn once. Render the EntropyCut comparison with:

    MPLCONFIGDIR=/private/tmp/proposal-certainty-mpl \
      /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/.venv/bin/python \
      /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/draw/draw_multi_model_proposal_certainty.py \
      --log-root /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/logs-entropycut \
      --run-date latest --method-tag cut-entropy4.0 --rank-fn accept_first \
      --method-label PreSTO-EntropyCut --hatch --all-edges \
      --models gemma-12b-it qwen3-8b qwen3.5-4b qwen3-4b qwen3.5-9b \
      --datasets math500 aime mbpp lcb_v6 gpqa

Render the PreSTO (subtree prefetching) comparison, whose models ran on
different dates, with ``--model-run-date MODEL=DATE`` overriding ``--run-date``:

    ... draw_multi_model_proposal_certainty.py --run-date 2026-09-16 \
      --hatch --all-edges \
      --models gemma-12b-it qwen3-8b qwen3.5-4b qwen3-4b \
      --datasets math500 aime mbpp gpqa human_eval mmlu lcb_v6

All other options match ``draw_dataset_proposal_certainty.py``. A dataset whose
log is missing for a model leaves that grid cell empty, so list datasets that
only some models ran last.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from case_studies.draw.common.figure_io import save_figure, write_dict_rows
from case_studies.draw.common.plot_helpers import panel_letter
from case_studies.draw.figures.proposals.draw_dataset_proposal_certainty import (
    BACKEND,
    CSV_SUFFIX,
    DATASET_LABELS,
    DATASET_LEGEND_FONT_SIZE,
    DATASET_SUPTITLE_FONT_SIZE,
    DATASET_TITLE_FONT_SIZE,
    DATASET_XLABEL_FONT_SIZE,
    DATASET_YLABEL_FONT_SIZE,
    EDGE_SOURCE_FIELDS,
    FIGURE_SUFFIX,
    LATEST_RUN_DATE,
    MODEL_LABELS,
    SOURCE_FIELDS,
    DatasetRun,
    build_arg_parser,
    collect_dataset_runs,
    draw_dataset_panel,
    source_rows,
)
from case_studies.draw.figures.proposals.draw_rank_proposal_certainty import (
    ACCEPTANCE_XLABEL,
    LEGEND_HANDLE_HEIGHT,
    LEGEND_HANDLE_LENGTH,
    Y_AXIS_MIN,
    Y_LIMIT_HEADROOM,
    certainty_legend_handles,
)

MISSING_LOG_STATUS = "log not available"

# Vertical bands in inches, from the top of the figure down.
SUPTITLE_BAND = 0.34
LEGEND_BAND = 0.36
ROW_HEADER_BAND = 0.30
COLUMN_TITLE_BAND = 0.26
TICK_LABEL_BAND = 0.30
# Narrower and taller than a single-model panel so the grid fits a text column.
GRID_PANEL_WIDTH = 1.25
GRID_PANEL_HEIGHT = 1.60
XLABEL_BAND = 0.40
LEFT_MARGIN = 0.45
RIGHT_MARGIN = 0.05
COLUMN_GAP = 0.18 * GRID_PANEL_WIDTH
ROW_HEADER_FONT_SIZE = 13


def draw_figure(
    rows: Sequence[tuple[str, Sequence[DatasetRun]]],
    datasets: Sequence[str],
    *,
    hatched: bool,
    all_edges: bool,
    method_label: str,
) -> Figure:
    """Draw one row of dataset panels per model with a shared legend and x label."""
    columns = len(datasets)
    width = (
        LEFT_MARGIN
        + columns * GRID_PANEL_WIDTH
        + (columns - 1) * COLUMN_GAP
        + RIGHT_MARGIN
    )
    row_height = ROW_HEADER_BAND + GRID_PANEL_HEIGHT + TICK_LABEL_BAND
    height = (
        SUPTITLE_BAND
        + LEGEND_BAND
        + COLUMN_TITLE_BAND
        + len(rows) * row_height
        + XLABEL_BAND
    )
    figure = plt.figure(figsize=(width, height))
    # Column titles sit once above the grid, so a dataset missing from the first
    # model still gets its title.
    for column, dataset in enumerate(datasets):
        figure.text(
            (LEFT_MARGIN + column * (GRID_PANEL_WIDTH + COLUMN_GAP) + GRID_PANEL_WIDTH / 2) / width,
            1.0 - (SUPTITLE_BAND + LEGEND_BAND + COLUMN_TITLE_BAND - 0.04) / height,
            DATASET_LABELS.get(dataset, dataset),
            fontsize=DATASET_TITLE_FONT_SIZE,
            ha="center",
            va="bottom",
        )

    axes = []
    maximum_bin_share = 0.0
    for row_index, (model, runs) in enumerate(rows):
        panel_top = (
            SUPTITLE_BAND
            + LEGEND_BAND
            + COLUMN_TITLE_BAND
            + row_index * row_height
            + ROW_HEADER_BAND
        )
        bottom = 1.0 - (panel_top + GRID_PANEL_HEIGHT) / height
        by_dataset = {run.dataset: run for run in runs}
        for column, dataset in enumerate(datasets):
            left = LEFT_MARGIN + column * (GRID_PANEL_WIDTH + COLUMN_GAP)
            ax = figure.add_axes(
                (
                    left / width,
                    bottom,
                    GRID_PANEL_WIDTH / width,
                    GRID_PANEL_HEIGHT / height,
                )
            )
            run = by_dataset[dataset]
            if run.status == MISSING_LOG_STATUS:
                ax.set_visible(False)
                continue
            maximum_bin_share = max(
                maximum_bin_share,
                draw_dataset_panel(ax, run, hatched=hatched, all_edges=all_edges),
            )
            axes.append(ax)
        figure.text(
            (LEFT_MARGIN + (width - RIGHT_MARGIN)) / 2 / width,
            1.0 - (panel_top - ROW_HEADER_BAND + 0.04) / height,
            rf"\textbf{{{panel_letter(row_index)}}} {MODEL_LABELS.get(model, model)}",
            fontsize=ROW_HEADER_FONT_SIZE,
            ha="center",
            va="top",
        )

    y_max = max(10.0, Y_LIMIT_HEADROOM * maximum_bin_share)
    for ax in axes:
        ax.set_ylim(Y_AXIS_MIN, y_max)

    figure.suptitle(
        f"Method: {method_label}, inference engine: vLLM",
        fontsize=DATASET_SUPTITLE_FONT_SIZE,
        y=1.0 - 0.05 / height,
        va="top",
    )
    figure.legend(
        handles=certainty_legend_handles(hatched=hatched),
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0 - SUPTITLE_BAND / height),
        borderaxespad=0.0,
        borderpad=0.0,
        ncol=3,
        fontsize=DATASET_LEGEND_FONT_SIZE,
        handlelength=LEGEND_HANDLE_LENGTH,
        handleheight=LEGEND_HANDLE_HEIGHT,
        handletextpad=0.3,
        columnspacing=0.55,
    )
    figure.supylabel(
        r"edge share (\%)" if all_edges else r"edge category (\%)",
        fontsize=DATASET_YLABEL_FONT_SIZE,
        x=0.05 / width,
    )
    figure.supxlabel(
        "Edge transition probability" if all_edges else ACCEPTANCE_XLABEL,
        y=0.06 / height,
        fontsize=DATASET_XLABEL_FONT_SIZE,
        va="bottom",
    )
    return figure


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_arg_parser()
    parser.description = "Plot proposal certainty with one row per model."
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument(
        "--model-run-date",
        nargs="+",
        default=[],
        metavar="MODEL=DATE",
        help="Per-model run date overriding --run-date.",
    )
    args = parser.parse_args(argv)
    model_run_dates = dict(item.split("=", 1) for item in args.model_run_date)
    unknown = set(model_run_dates) - set(args.models)
    if unknown:
        raise SystemExit(f"--model-run-date names models not in --models: {sorted(unknown)}")
    default_run_date = args.run_date
    args.log_root = args.log_root.resolve()
    if len(set(args.datasets)) != len(args.datasets):
        raise SystemExit("--datasets contains a duplicate name")

    rows: list[tuple[str, list[DatasetRun]]] = []
    csv_rows: list[dict[str, object]] = []
    try:
        for model in args.models:
            args.model = model
            args.run_date = model_run_dates.get(model, default_run_date)
            runs = collect_dataset_runs(args)
            rows.append((model, runs))
            csv_rows.extend(
                source_rows(
                    runs,
                    model=model,
                    prefetch_budget=args.prefetch_budget,
                    rank_fn=args.rank_fn,
                    all_edges=args.all_edges,
                )
            )
        figure = draw_figure(
            rows,
            args.datasets,
            hatched=args.hatch,
            all_edges=args.all_edges,
            method_label=args.method_label,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error

    output_date = default_run_date
    if output_date == LATEST_RUN_DATE or model_run_dates:
        output_date = max(
            run.log_path.parent.parent.name
            for _, runs in rows
            for run in runs
            if run.nodes
        )
    stem = (
        f"dataset-all.model-all.prefetch-budget-{args.prefetch_budget}."
        f"rank-{args.rank_fn}"
    )
    if args.method_tag:
        stem += f".{args.method_tag}"
    if args.all_edges:
        stem += ".all-edges"
    figure_suffix = FIGURE_SUFFIX.replace(".pdf", ".hatched.pdf") if args.hatch else FIGURE_SUFFIX
    csv_suffix = CSV_SUFFIX.replace(".csv", ".hatched.csv") if args.hatch else CSV_SUFFIX
    output_directory = args.log_root / "all-datasets" / output_date / BACKEND
    output = args.output.resolve() if args.output else output_directory / f"{stem}{figure_suffix}"
    csv_output = (
        args.csv_output.resolve() if args.csv_output else output_directory / f"{stem}{csv_suffix}"
    )

    save_figure(figure, output, pad_inches=0.02)
    write_dict_rows(
        csv_output,
        csv_rows,
        fieldnames=EDGE_SOURCE_FIELDS if args.all_edges else SOURCE_FIELDS,
    )
    print(f"wrote {output}")
    print(f"wrote {csv_output}")


if __name__ == "__main__":
    main()
