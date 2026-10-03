#!/usr/bin/env python3
"""Draw the acceptance-probability distribution of subtree-prefetching runs.

The single-log figure shows where proposals land across the three acceptance categories. The rank-sweep mode combines
the same acceptance panel across compatible logs.

Run a single log from the repository root:

    case_studies/draw/.venv/bin/python \
        case_studies/draw/draw_acceptance_panels.py \
        --log case_studies/logs/math500/<date>/<run>.subtreePrefetch.hf.log

Or combine the acceptance panels from a rank sweep:

    case_studies/draw/.venv/bin/python \
        case_studies/draw/draw_acceptance_panels.py \
        --log-root case_studies/logs/math500/<date>/vllm
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import FuncFormatter

from case_studies.draw.common.figure_io import save_figure
from case_studies.draw.common.plot_helpers import panel_letter, percentage_tick_label
from case_studies.draw.common.rank_plotting import prefetch_size_label
from case_studies.draw.figures.proposals.acceptance_mass import (
    AcceptanceBars,
    draw_acceptance_bars,
    label_acceptance_axis,
)
from case_studies.extract.logs.proposal_log import (
    EFFECTIVELY_ZERO_ACCEPT_PROB,
    NEAR_CERTAIN_ACCEPT_PROB,
    TreeNode,
    parse_log,
)
from case_studies.extract.run_naming import (
    LOG_GLOB,
    batch_size_in,
    config_output_prefix,
    latex_text,
    rank_of,
)
from case_studies.plot_config import (
    LARGE_FONT_SIZE as BASE_FONT_SIZE,
)
from case_studies.plot_config import (
    apply_plot_style,
)

apply_plot_style()

ANNOTATION_SIZE = int(BASE_FONT_SIZE * 0.92)
CROWDED_TICK_SIZE = int(BASE_FONT_SIZE * 0.78)

# Wide enough that the banner's three bucket labels fit beside each other.
SINGLE_FIGURE_SIZE = (5.0, 3.0)
GRID_FIGURE_WIDTH = 10.0
GRID_COLUMNS = 2
GRID_PANEL_HEIGHT = 2.8

# Descending in A, matching the palette; reversed for drawing.
BUCKET_LABELS = (
    "$\\epsilon$-certain\naccept\n"
    f"{NEAR_CERTAIN_ACCEPT_PROB:g}$<A\\leq 1$",
    "$\\epsilon$-uncertain\n"
    f"{EFFECTIVELY_ZERO_ACCEPT_PROB:g}$<A\\leq${NEAR_CERTAIN_ACCEPT_PROB:g}",
    "$\\epsilon$-certain\nreject\n"
    f"$0 < A \\leq {EFFECTIVELY_ZERO_ACCEPT_PROB:g}$",
)
BIN_GAP = 0.08  # fraction of each unit-width bin left as whitespace
BUCKET_LABEL_Y = 1.02  # puts the category row in the former title position
COMPACT_LABEL_Y = 0.95  # percentage-only labels inside comparison panels
REJECT_LABEL_OFFSET = -1.1  # shifts the long reject label past the left spine
ACCEPT_LABEL_INSET = 0  # keeps the accept label inside the right spine
Y_MARGIN_FRACTION = 0.06  # autoscale padding around the observed bars
PERCENT_AXIS_MAX = 100.0
ACCEPTANCE_XLABEL = "acceptance probability $A$ (equal-width bins in each bucket)"


class BucketAnnotation(NamedTuple):
    """One bucket caption, positioned in axis-x / axes-y coordinates."""

    x: float
    y: float
    text: str
    ha: str
    multialignment: str
    va: str


def _bucket_annotation(
    bars: AcceptanceBars,
    index: int,
    label: str,
    *,
    compact: bool,
    last_index: int,
) -> BucketAnnotation:
    """Place and word one bucket caption.

    The banner captions sit above the axes and must not collide, so the outer two are anchored to their own bucket edge
    rather than centred. The compact form used inside a comparison grid is a centred percentage instead.
    """
    left, right = bars.bucket_span(index)
    middle = (left + right) / 2
    share = bars.shares[index] * 100
    if compact:
        return BucketAnnotation(
            middle, COMPACT_LABEL_Y, f"${share:.0f}\\%$", "center", "center", "top"
        )

    # The label's last line is the interval; the rest is the category name, which carries the share and is joined back
    # onto one line.
    *name_lines, interval = label.split("\n")
    text = f"{' '.join(name_lines)} ({share:.0f}\\%)\n{interval}"
    if index == 0:
        # The reject caption is the widest, so it hangs past the left spine and its two lines stay right-aligned against
        # the bucket it names.
        return BucketAnnotation(
            left + REJECT_LABEL_OFFSET, BUCKET_LABEL_Y, text, "left", "right", "bottom"
        )
    if index == last_index:
        return BucketAnnotation(
            right - ACCEPT_LABEL_INSET, BUCKET_LABEL_Y, text, "right", "right", "bottom"
        )
    return BucketAnnotation(middle, BUCKET_LABEL_Y, text, "center", "center", "bottom")


def plot_acceptance_mass(
    ax: Axes,
    nodes: list[TreeNode],
    *,
    fix_percentage_ceiling: bool = True,
    compact_labels: bool = False,
) -> None:
    """Acceptance-probability histogram cut into the three categories."""
    bars = draw_acceptance_bars(ax, nodes, as_percentage=True, bin_gap=BIN_GAP)
    ax.set_yscale("log")
    ax.margins(y=Y_MARGIN_FRACTION)
    ax.yaxis.set_major_formatter(FuncFormatter(percentage_tick_label))
    if fix_percentage_ceiling:
        ax.set_ylim(top=PERCENT_AXIS_MAX)

    labels = list(reversed(BUCKET_LABELS))
    spans = ax.get_xaxis_transform()
    for index, label in enumerate(labels):
        caption = _bucket_annotation(
            bars,
            index,
            label,
            compact=compact_labels,
            last_index=len(labels) - 1,
        )
        ax.text(
            caption.x,
            caption.y,
            caption.text,
            transform=spans,
            ha=caption.ha,
            multialignment=caption.multialignment,
            va=caption.va,
            fontsize=ANNOTATION_SIZE,
            color=bars.text_colors[index],
            clip_on=False,
        )

    label_acceptance_axis(ax, bars, tick_size=CROWDED_TICK_SIZE)
    ax.set_xlabel(ACCEPTANCE_XLABEL)
    ax.set_ylabel("percentage of proposals")


def draw_acceptance_panel(nodes: list[TreeNode], output_path: Path) -> Path:
    """Draw the single-log acceptance panel and save it."""
    figure, ax = plt.subplots(figsize=SINGLE_FIGURE_SIZE, layout="constrained")
    plot_acceptance_mass(ax, nodes)
    return save_figure(figure, output_path)


def rank_title(log_path: Path) -> str:
    """Return the ``rank-...`` field encoded in a sampler-log filename."""
    return f"rank-{rank_of(log_path)}"


def _panel_sort_key(log_path: Path) -> tuple[str, int]:
    """Order panels by traversal rank, then by ascending batch size."""
    batch_size = batch_size_in(log_path.name)
    return rank_title(log_path), -1 if batch_size is None else batch_size


def _panel_title(log_path: Path, index: int, *, show_batch_size: bool) -> str:
    """Name one panel, adding the batch size only when it disambiguates."""
    title = f"{panel_letter(index)} {latex_text(rank_title(log_path))}"
    batch_size = batch_size_in(log_path.name) if show_batch_size else None
    if batch_size is None:
        return title
    return f"{title}, {prefetch_size_label(batch_size)}"


def draw_acceptance_mass_grid(log_paths: list[Path], output_path: Path) -> Path:
    """Combine acceptance-mass panels from a rank sweep in one figure."""
    if not log_paths:
        raise ValueError("no sampler logs were provided")
    ranked_logs = sorted(log_paths, key=_panel_sort_key)
    # A recursive sweep can hold the same rank at several batch sizes, which the rank name alone would not tell apart.
    show_batch_size = (
        len({batch_size_in(log_path.name) for log_path in ranked_logs}) > 1
    )

    columns = min(GRID_COLUMNS, len(ranked_logs))
    rows = math.ceil(len(ranked_logs) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(GRID_FIGURE_WIDTH, GRID_PANEL_HEIGHT * rows),
        layout="constrained",
        sharey=True,
        squeeze=False,
    )

    for index, log_path in enumerate(ranked_logs):
        parsed = parse_log(log_path)
        if not parsed.nodes:
            raise ValueError(
                f"no Node(...) lines with a log ratio found in {log_path}"
            )

        row, column = divmod(index, columns)
        ax = axes[row, column]
        plot_acceptance_mass(
            ax,
            parsed.nodes,
            fix_percentage_ceiling=False,
            compact_labels=True,
        )
        panel_title = _panel_title(
            log_path, index, show_batch_size=show_batch_size
        )
        # Only the bottom row carries the axis name; the rest name themselves.
        ax.set_xlabel(
            f"{ACCEPTANCE_XLABEL}\n{panel_title}"
            if row == rows - 1
            else panel_title
        )
        if column > 0:
            ax.set_ylabel("")

    axes[0, 0].set_ylim(top=PERCENT_AXIS_MAX)
    for index in range(len(ranked_logs), rows * columns):
        row, column = divmod(index, columns)
        axes[row, column].set_visible(False)

    return save_figure(figure, output_path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot one run's acceptance distribution or combine a rank "
            "sweep's acceptance panels in one figure."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--log",
        type=Path,
        default=None,
        help="Sampler log to parse for the acceptance figure.",
    )
    source.add_argument(
        "--log-root",
        type=Path,
        default=None,
        help="Directory searched recursively for logs to combine by rank.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output figure. Defaults beside the log for --log, or to "
            "'acceptance-mass-by-rank.pdf' under --log-root."
        ),
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.log_root is not None:
        if not args.log_root.is_dir():
            raise SystemExit(f"missing log directory {args.log_root}")
        log_paths = sorted(args.log_root.rglob(LOG_GLOB))
        if not log_paths:
            raise SystemExit(f"no {LOG_GLOB} files under {args.log_root}")
        output_path = args.output or args.log_root / "acceptance-mass-by-rank.pdf"
        try:
            written = draw_acceptance_mass_grid(log_paths, output_path)
        except ValueError as error:
            raise SystemExit(str(error)) from error
        print(f"wrote {written}")
        return

    log_path = args.log
    if not log_path.is_file():
        raise SystemExit(f"missing log {log_path}")

    parsed = parse_log(log_path)
    if not parsed.nodes:
        raise SystemExit(f"no Node(...) lines with a log ratio found in {log_path}")

    output_path = args.output or Path(
        f"{config_output_prefix(log_path)}.acceptance-panels.pdf"
    )
    print(f"wrote {draw_acceptance_panel(parsed.nodes, output_path)}")


if __name__ == "__main__":
    main()
