#!/usr/bin/env python3
"""Plot the log Hastings ratio against the resample length that earned it.

One scatter panel per subtree-prefetching MH log: every scored proposal node placed at the number of tokens resampled
after its cut. Nodes whose log records no ``cut_idx`` cannot be placed, so a log without one draws a notice instead.

Run from the repository root:

    src/.venv/bin/python \
        case_studies/draw/draw_resample_vs_ratio.py \
        --log /absolute/path/to/run.subtreePrefetch.vllm.log
"""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from case_studies.draw.common.figure_io import save_figure
from case_studies.draw.common.plot_helpers import rank_correlation, show_notice
from case_studies.extract.logs.proposal_log import TreeNode, parse_log
from case_studies.extract.run_naming import config_output_prefix, output_path
from case_studies.plot_config import (
    DARK_GRAY,
    TRANSPARENT,
    apply_plot_style,
)
from case_studies.plot_config import (
    GRAY as NEUTRAL_COLOR,
)
from case_studies.plot_config import (
    LARGE_FONT_SIZE as BASE_FONT_SIZE,
)
from case_studies.plot_config import (
    RED as HIGHLIGHT_COLOR,
)

apply_plot_style()

FIGURE_SUFFIX = ".resample-vs-ratio.pdf"
FIGURE_SIZE = (7.0, 5.0)
NOTICE_SIZE = BASE_FONT_SIZE * 0.80

_show_notice = partial(show_notice, fontsize=NOTICE_SIZE, color=NEUTRAL_COLOR)


def plot_resample_vs_ratio(ax: Axes, nodes: list[TreeNode]) -> None:
    """Log ratio against the cut of the proposal it scores."""
    with_cut = [
        node
        for node in nodes
        if node.proposed_suffix_len is not None and node.proposed_suffix_len >= 0
    ]
    if not with_cut:
        _show_notice(
            ax,
            "Log ratio vs. cut index",
            "cut_idx absent from this log\n\n"
            "Node.__repr__ did not emit it when\nthis run was produced; reruns\n"
            "now include it and this panel\nwill populate itself.",
        )
        return

    suffix = np.array([node.proposed_suffix_len for node in with_cut], dtype=float)
    ratios = np.array([node.log_ratio for node in with_cut], dtype=float)
    ax.scatter(
        suffix,
        ratios,
        s=12,
        color=HIGHLIGHT_COLOR,
        alpha=0.55,
        edgecolors=TRANSPARENT,
    )
    ax.axhline(0.0, color=DARK_GRAY, ls="--", lw=1.0)
    if suffix.min() > 0:
        ax.set_xscale("log")
    ax.set_xlabel("Number of resampled tokens")
    ax.set_ylabel("log Hastings ratio")
    spearman = rank_correlation(suffix, ratios)
    ax.set_title(f"Longer resamples score worse (Spearman {spearman:+.2f})")


def draw_figure(nodes: list[TreeNode], output_prefix: Path) -> Path:
    """Draw the single panel and save it beside the log it came from."""
    figure, ax = plt.subplots(figsize=FIGURE_SIZE)
    plot_resample_vs_ratio(ax, nodes)
    figure.tight_layout()
    return save_figure(figure, output_path(output_prefix, FIGURE_SUFFIX))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the log Hastings ratio against resample length over "
            "prefetched PreSTO Power-MH subtrees."
        )
    )
    parser.add_argument(
        "--log", type=Path, required=True, help="Sampler log to parse."
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=None,
        help=(
            f"Path prefix for the {FIGURE_SUFFIX} output. Defaults to the "
            "log's own run configuration, beside the log (see "
            "config_output_prefix)."
        ),
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if not args.log.is_file():
        raise SystemExit(f"missing log {args.log}")

    nodes = parse_log(args.log).nodes
    if not nodes:
        raise SystemExit(f"no Node(...) lines with a log ratio found in {args.log}")

    output_prefix = args.output_prefix or config_output_prefix(args.log)
    print(f"wrote {draw_figure(nodes, output_prefix)}")


if __name__ == "__main__":
    main()
