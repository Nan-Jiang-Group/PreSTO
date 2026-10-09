#!/usr/bin/env python3
"""Stack the PowerMH and EntropyCut accept-first call/time comparisons as two rows of one figure.

Run with:

    MPLCONFIGDIR=/private/tmp/power-sharpening-mplconfig \
    /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/.venv/bin/python \
    /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/draw/draw_power_mh_and_entropycut_model_calls_and_empirical_time.py

Row 1 reproduces ``draw_accept_first_power_mh_model_calls_and_empirical_time.py --directory <power-mh-directory>``:
the accept-first PreSTO-PowerMH runs of every prefetch budget, each panel keeping its best budget. Row 2 draws one
PreSTO-EntropyCut subtree log (``--entropycut-subtree-log``) against the EntropyCut baseline of identical settings
in its directory. The two rows share one title line and one legend; only the bottom row carries x-axis labels.
Pass ``--output /absolute/path.pdf`` to write elsewhere. ``--rows power-mh`` (or ``entropycut``) draws that row alone
as a standalone figure with the same styling, keeping its own x-axis labels and a legend of its methods only.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from case_studies import paths
from case_studies.draw.figures.model_calls import draw_accept_first_power_mh_model_calls_and_empirical_time as combined
from case_studies.extract.logs.model_call_traces import extract_model_call_traces
from case_studies.extract.run_naming import backend_of, dataset_of, model_display_name
from case_studies.plot_config import BLUE, DARK_GRAY, GRAY, GREEN, TRANSPARENT

LOG_ROOT = paths.CASE_STUDIES_DIR
DEFAULT_POWER_MH_DIRECTORY = LOG_ROOT / "logs" / "lcb_v6" / "2026-09-07" / "vllm"
DEFAULT_ENTROPYCUT_SUBTREE_LOG = (
    LOG_ROOT / "logs-entropycut" / "lcb_v6" / "2026-09-15" / "vllm"
    / "dataset-lcb_v6.model-qwen3.5-9b.alpha4.0.steps100.blocks-1.prefetch-budget-20.rank-accept_first"
    ".cut-entropy4.0.samples20.maxnew1024.seed10086.subtreePrefetch.vllm.log"
)
DEFAULT_OUTPUT_NAME = (
    "dataset-lcb_v6.model-qwen3.5-9b.power-mh-and-entropycut.accept-first.model-calls-and-empirical-time.pdf"
)
# --rows choice -> indices into ROWS, and the tag replacing "power-mh-and-entropycut" in the default output name.
ROW_CHOICES = {"both": ((0, 1), "power-mh-and-entropycut"), "power-mh": ((0,), "power-mh"),
               "entropycut": ((1,), "entropycut")}

# Baselines share the blue family and PreSTO the green family. Within each family the EntropyCut row shifts both hue
# and lightness (cyan-blue; deep emerald) so the two shades stay separable in thin traces.
CYAN_BLUE = "#00A3D9"
EMERALD = "#1B6B2F"
# (baseline label, subtree label, baseline color, subtree color, subtree statistic font size) for each row.
ROWS = (
    ("PowerMH", "PreSTO-PowerMH", BLUE, GREEN, combined.FIGURE_FONT_SIZE),
    ("EntropyCut", "PreSTO-EntropyCut", CYAN_BLUE, EMERALD, combined.FIGURE_FONT_SIZE),
)
# Fixed y-axis maxima, keyed by (row, column); other panels keep the renderer's automatic limits.
Y_MAXIMA = {(0, 1): 6.1, (0, 2): 10.1, (1, 1): 8.1, (1, 2): 14.0}
# Panels whose y maximum is exactly the largest plotted value (traces, means, and SD bands), keyed by (row, column).
Y_DATA_MAXIMUM_PANELS = ((1, 0),)
# Fixed y ticks, keyed by (row, column); other panels keep automatic tick locations.
Y_TICKS = {(0, 1): (1, 3, 5), (0, 2): (2, 4, 6, 8), (1, 1): (1, 3, 5, 7)}
# Wider, shorter panels than the single-row figure (3.7 in wide), so the stacked pair keeps a paper-friendly aspect.
PANEL_WIDTH = 3.0
ROW_HEIGHT = 2
# Panel letters hang from the top of each axes, beside (not above) the top y tick label; with the current limits
# and ticks no letter meets a tick label, which frees the space above each row.
PANEL_LABEL_Y = 1.0
# Letters' left edge, in inches left of the axes, so they clear the tick labels whatever the panel width.
PANEL_LABEL_LEFT_INCHES = 0.48
# Single-line axis labels are taller than a short row, so they are broken onto two centered lines.
TWO_LINE_Y_LABELS = {
    combined.TIME_AXIS_LABEL: "cumulative time (min)",
    combined.TRANSITIONS_PER_CALL_AXIS_LABEL: "transitions per call",
    # Short enough to fit the row height on one line.
    combined.TIME_PER_CALL_AXIS_LABEL: "time per call (sec)",
}
HEADER_INCHES = 0.68
ROW_GAP_INCHES = 0.18
# Horizontal gap between panels as a fraction of one panel's width (single-row figure: 0.26). With 3.4 in panels,
# 0.225 leaves ~0.11 in between each column-2 panel and the next panel's y label ((f)'s two-digit ticks are widest).
PANEL_WSPACE = 0.225
# Column 1 moves right by this much, narrowing only the column 1-2 gap (column 2's y label then sits ~0.08 in from it).
COLUMN_1_SHIFT_INCHES = 0.05
# Column 3 moves left by this much, keeping its width: its one-line y label needs less room than the gap gives.
COLUMN_3_SHIFT_INCHES = 0.14
# Y-label padding (points) from the tick labels, keyed by (row, column); others keep the style default (4 pt).
Y_LABEL_PADS = {(1, 0): 0.0}
# In-panel statistics given an opaque white background, keyed by (row, column) -> method label.
OPAQUE_STATISTICS = {(1, 0): "PreSTO-EntropyCut"}
# Row-2 panels whose y label is pinned to the same offset as the row-1 label above it, keyed by column.
ALIGNED_Y_LABEL_COLUMNS = (2,)
# Distances below the figure top (inches) of the title line's top and the legend's top.
TITLE_TOP_INCHES = 0.13
LEGEND_TOP_INCHES = 0.30
BOTTOM_INCHES = combined.BOTTOM_MARGIN_INCHES


def plotted_maximum(ax) -> float:
    """Return the largest y value among an axes' lines and filled bands."""
    values = [float(np.nanmax(line.get_ydata())) for line in ax.lines if len(line.get_ydata())]
    values += [float(np.nanmax(path.vertices[:, 1])) for band in ax.collections for path in band.get_paths()
               if len(path.vertices)]
    return max(values)


def align_y_label(ax, reference, renderer) -> None:
    """Place ``ax``'s y label at the same horizontal offset from its axes as ``reference``'s label."""
    reference_box = reference.get_window_extent(renderer)
    label_right = reference.yaxis.label.get_window_extent(renderer).x1
    ax.yaxis.set_label_coords((label_right - reference_box.x0) / reference_box.width, 0.5)


def draw_row(
    axes: Sequence, comparison: combined.BatchComparison, baseline: combined.PowerMHTimeData,
    row: tuple[str, str, str, str, float], panel_labels: Sequence[str],
) -> tuple[int, int, int]:
    """Draw one three-panel row after pointing the shared renderer at this row's labels and colors."""
    baseline_label, subtree_label, baseline_color, subtree_color, subtree_font_size = row
    combined.POWER_METHOD_LABEL, combined.SUBTREE_METHOD_LABEL = baseline_label, subtree_label
    combined.POWER_COLOR, combined.SUBTREE_COLOR = baseline_color, subtree_color
    combined.SUBTREE_LABEL_FONT_SIZE = subtree_font_size
    return combined.draw_comparison_panels(axes, comparison, baseline, panel_labels=panel_labels)


def add_shared_legend(figure: plt.Figure, center_x: float, top_y: float, rows: Sequence[int] = (0, 1),
                      frame: bool = True) -> None:
    """One legend, framed unless ``frame`` is off: a color per method (dashed baselines, solid PreSTO) and a gray key
    for the line roles."""
    handles = [
        Line2D([], [], color=color, linewidth=combined.MEAN_LINE_WIDTH, linestyle=style, label=label)
        for label, color, style in (
            [(ROWS[row][0], ROWS[row][2], "--") for row in rows]
            + [(f"{ROWS[row][1]} (Ours)", ROWS[row][3], "-") for row in rows]
        )
    ]
    handles += [
        Line2D([], [], color=GRAY, linewidth=combined.RAW_TRACE_LINE_WIDTH, alpha=combined.RAW_TRACE_ALPHA + 0.2,
               label="trace"),
        Line2D([], [], color=GRAY, linewidth=combined.MEAN_LINE_WIDTH, label="mean"),
        Rectangle((0, 0), 1, 1, facecolor=GRAY, alpha=0.35, edgecolor=TRANSPARENT, label=r"$\pm 1$ SD"),
    ]
    legend = figure.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(center_x, top_y), ncol=len(handles),
        fontsize=combined.LEGEND_FONT_SIZE, frameon=frame, handlelength=1.8, columnspacing=1.3,
        handletextpad=0.5, borderpad=0.35 if frame else 0.0,
    )
    legend.get_frame().set_edgecolor(DARK_GRAY)
    legend.get_frame().set_linewidth(0.8)


def main(argv: Sequence[str] | None = None) -> None:
    """Build both comparisons, render the two-row figure, and report the budgets each panel used."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--power-mh-directory", type=Path, default=DEFAULT_POWER_MH_DIRECTORY)
    parser.add_argument("--entropycut-subtree-log", type=Path, default=DEFAULT_ENTROPYCUT_SUBTREE_LOG)
    parser.add_argument("--traces-per-rank", type=int, default=combined.DEFAULT_TRACES_PER_RANK,
                        help="Traces per budget for the PowerMH row, as in the single-row script.")
    parser.add_argument("--rows", choices=sorted(ROW_CHOICES), default="both",
                        help="Draw both rows, or one row alone as a standalone figure.")
    parser.add_argument("--font-size", type=float, default=None,
                        help=f"Font size of all text, legend included (default {combined.FIGURE_FONT_SIZE:g}, "
                             f"legend {combined.LEGEND_FONT_SIZE:g}).")
    parser.add_argument("--time-unit", choices=("min", "sec"), default="min",
                        help="Unit of cumulative model-call time in the first column.")
    parser.add_argument("--legend-font-size", type=float, default=None,
                        help="Legend font size, applied after --font-size (default: the legend size it sets).")
    parser.add_argument("--no-title", dest="title", action="store_false",
                        help="Omit the dataset/base-LLM/engine line above the legend; the header shrinks to fit.")
    parser.add_argument("--no-legend-frame", dest="legend_frame", action="store_false",
                        help="Draw the legend without its box.")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    rows, row_tag = ROW_CHOICES[args.rows]
    global ROWS, ROW_HEIGHT, HEADER_INCHES, TITLE_TOP_INCHES, LEGEND_TOP_INCHES, BOTTOM_INCHES
    global PANEL_LABEL_LEFT_INCHES, PANEL_WSPACE
    if args.font_size is not None:
        # Larger text needs proportionally larger margins, header, and gaps; the rows grow by the same factor.
        scale = args.font_size / combined.FIGURE_FONT_SIZE
        combined.FIGURE_FONT_SIZE = combined.LEGEND_FONT_SIZE = args.font_size
        ROWS = tuple((*row[:4], args.font_size) for row in ROWS)
        ROW_HEIGHT *= scale
        HEADER_INCHES *= scale
        TITLE_TOP_INCHES *= scale
        LEGEND_TOP_INCHES *= scale
        BOTTOM_INCHES *= scale
        PANEL_LABEL_LEFT_INCHES *= scale
        PANEL_WSPACE *= scale
        combined.LEFT_MARGIN_INCHES *= scale
    if args.time_unit == "sec":
        # The renderer divides seconds by SECONDS_PER_MINUTE; 1 keeps them in seconds.
        combined.SECONDS_PER_MINUTE = 1.0
        TWO_LINE_Y_LABELS[combined.TIME_AXIS_LABEL] = "cumulative time (sec)"
    if args.legend_font_size is not None:
        combined.LEGEND_FONT_SIZE = args.legend_font_size
    if not args.title:
        # The legend moves up into the title's place and the header loses the title line.
        HEADER_INCHES -= LEGEND_TOP_INCHES - 0.02
        LEGEND_TOP_INCHES = 0.02

    power_directory = args.power_mh_directory.resolve()
    power_comparison = combined.find_comparison(
        power_directory, traces_per_rank=args.traces_per_rank, include_incomplete=False
    )
    power_baseline = combined.load_power_mh_time_data(combined.find_power_mh_log(
        power_directory, combined.comparison_source_log(power_comparison), None
    ))
    entropy_log = args.entropycut_subtree_log.resolve()
    entropy_comparison = combined.single_budget_comparison(
        entropy_log, traces_per_rank=len(extract_model_call_traces(entropy_log))
    )
    entropy_baseline_log = combined.find_power_mh_log(entropy_log.parent, entropy_log, None)
    entropy_baseline = combined.load_power_mh_time_data(entropy_baseline_log)

    width = PANEL_WIDTH * 3
    height = HEADER_INCHES + len(rows) * ROW_HEIGHT + (len(rows) - 1) * ROW_GAP_INCHES + BOTTOM_INCHES
    figure, axes = plt.subplots(len(rows), 3, figsize=(width, height), squeeze=False)
    sources = ((power_comparison, power_baseline), (entropy_comparison, entropy_baseline))
    letters = iter(("(a)", "(b)", "(c)", "(d)", "(e)", "(f)"))
    budgets = [
        draw_row(axes[index], *sources[row], ROWS[row], tuple(next(letters) for _ in range(3)))
        for index, row in enumerate(rows)
    ]
    # Settings below are keyed by the row's position in the full two-row figure.
    position = {index: row for index, row in enumerate(rows)}
    for ax in axes[0] if len(rows) > 1 else ():
        ax.set_xlabel("")
        ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    for index, row_axes in enumerate(axes):
        row = position[index]
        for column, ax in enumerate(row_axes):
            ax.set_ylabel(TWO_LINE_Y_LABELS[ax.get_ylabel()], fontsize=combined.FIGURE_FONT_SIZE, y=0.5)
            if (row, column) in Y_LABEL_PADS:
                ax.yaxis.labelpad = Y_LABEL_PADS[(row, column)]
            for text in ax.texts:
                if text.get_text().startswith(r"\textbf{("):
                    text.set_position((-PANEL_LABEL_LEFT_INCHES / PANEL_WIDTH, PANEL_LABEL_Y))
                    text.set_verticalalignment("top")
    drawn = {row: index for index, row in position.items()}
    for (row, column), y_maximum in Y_MAXIMA.items():
        if row in drawn:
            axes[drawn[row]][column].set_ylim(top=y_maximum)
    # Fixed maxima and the short rows leave little headroom for the in-panel statistics; back them so they stay
    # legible where they cross traces.
    for ax in axes.flat:
        for text in ax.texts:
            if r"_{\pm" in text.get_text():
                text.set_bbox({"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 1.0})
    for (row, column), label in OPAQUE_STATISTICS.items():
        for text in (axes[drawn[row]][column].texts if row in drawn else ()):
            if text.get_text().startswith(label + " "):
                text.get_bbox_patch().set_alpha(1.0)
    for row, column in Y_DATA_MAXIMUM_PANELS:
        if row in drawn:
            axes[drawn[row]][column].set_ylim(top=plotted_maximum(axes[drawn[row]][column]))
    for (row, column), ticks in Y_TICKS.items():
        if row in drawn:
            axes[drawn[row]][column].set_yticks(ticks)
    figure.subplots_adjust(
        left=combined.LEFT_MARGIN_INCHES / width, right=0.995,
        bottom=BOTTOM_INCHES / height, top=1 - HEADER_INCHES / height,
        wspace=PANEL_WSPACE, hspace=ROW_GAP_INCHES / ROW_HEIGHT,
    )
    for ax in axes[:, 0]:
        position = ax.get_position()
        ax.set_position([position.x0 + COLUMN_1_SHIFT_INCHES / width, position.y0, position.width, position.height])
    for ax in axes[:, 2]:
        position = ax.get_position()
        ax.set_position([position.x0 - COLUMN_3_SHIFT_INCHES / width, position.y0, position.width, position.height])
    figure.canvas.draw()
    for column in ALIGNED_Y_LABEL_COLUMNS if len(rows) > 1 else ():
        align_y_label(axes[1][column], axes[0][column], figure.canvas.get_renderer())

    first, last = axes[0][0].get_position(), axes[0][2].get_position()
    center_x = (first.x0 + last.x1) / 2
    source_log = combined.comparison_source_log(power_comparison)
    dataset_name, _, model_name = power_comparison.figure_title.partition(", ")
    dataset_name = "LCB V6" if dataset_of(source_log) == "lcb_v6" else dataset_name
    if args.title:
        figure.text(
            center_x, 1 - TITLE_TOP_INCHES / height,
            f"dataset: {dataset_name}, base LLM: {model_display_name(source_log, model_name)}, "
            f"inference engine: {backend_of(source_log)}",
            color=DARK_GRAY, fontsize=combined.FIGURE_FONT_SIZE, fontweight="semibold", ha="center", va="top",
        )
    add_shared_legend(figure, center_x, 1 - LEGEND_TOP_INCHES / height, rows, frame=args.legend_frame)

    default_name = DEFAULT_OUTPUT_NAME.replace("power-mh-and-entropycut", row_tag)
    output = args.output.resolve() if args.output else entropy_log.parent.parent.parent / default_name
    written = combined.save_figure(figure, output, pad_inches=0.01)
    for (baseline_label, subtree_label, *_), (efficiency, time, per_call) in zip(
        [ROWS[row] for row in rows], budgets, strict=True
    ):
        print(f"{subtree_label} vs {baseline_label}: budgets (time, transitions/call, sec/call) = "
              f"{time}, {efficiency}, {per_call}")
    print(f"EntropyCut baseline: {entropy_baseline_log}")
    print(f"wrote {written}")


if __name__ == "__main__":
    main()
