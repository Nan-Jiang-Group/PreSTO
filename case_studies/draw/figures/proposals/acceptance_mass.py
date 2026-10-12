"""Draw the acceptance-probability histogram shared by the case-study figures.

A node's bucket depends only on ``log A = min(0, log ratio)``, which is what the axis runs over, so every bucket is one
contiguous interval of it and the partition is exact. Each bucket is binned over its own range and every bin is drawn
the same width, which makes the axis piecewise linear -- it spends screen space per bin, not per nat. That is
deliberate: the stochastic band is 4.6 nats wide against 20 below it, so equal nats per bin would squeeze the only
bucket that decides anything into a couple of bars. A below the clip folds into the first bin, so every node is still
counted.

Callers supply their own bucket annotations, which is the only part that differs between the single-run panel and the
method-comparison grids.

This library is not run directly; invoke a sibling drawing CLI instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.axes import Axes

from case_studies.draw.common.plot_helpers import exp_tick_label
from case_studies.extract.logs.proposal_log import (
    BUCKET_EDGES_ASC,
    RATIO_HIST_CLIP,
    TreeNode,
    acceptance_histogram,
)
from case_studies.plot_config import (
    DARK_GRAY,
    TRANSPARENT,
)
from case_studies.plot_config import (
    SUBTREE_BUCKET_COLORS as BUCKET_COLORS,
)
from case_studies.plot_config import (
    SUBTREE_BUCKET_HATCHES as BUCKET_HATCHES,
)
from case_studies.plot_config import (
    SUBTREE_BUCKET_TEXT_COLORS as BUCKET_TEXT_COLORS,
)

BAR_LINE_WIDTH = 0.7
SEPARATOR_LINE_WIDTH = 0.9

@dataclass(frozen=True)
class AcceptanceBars:
    """Drawn bar geometry and per-bucket shares, in ascending-``A`` order."""

    bounds: tuple[int, ...]
    heights: np.ndarray
    shares: list[float]
    text_colors: list[str]

    def bucket_span(self, index: int) -> tuple[int, int]:
        """Return the first and last bin edge of one bucket."""
        return self.bounds[index], self.bounds[index + 1]


def draw_acceptance_bars(
    ax: Axes,
    nodes: list[TreeNode],
    *,
    as_percentage: bool,
    bin_gap: float = 0.0,
) -> AcceptanceBars:
    """Draw the three hatched bucket histograms and return their geometry.

    ``bin_gap`` is the fraction of each unit-width bin left as whitespace.
    """
    histogram = acceptance_histogram(nodes)
    heights = (
        100.0 * histogram.counts / len(nodes) if as_percentage else histogram.counts
    )
    bounds = histogram.bounds

    # ``acceptance_histogram`` bins in ascending A; the palette is declared in descending A, so it is reversed once here
    # rather than at each use site.
    colors = list(reversed(BUCKET_COLORS))
    hatches = list(reversed(BUCKET_HATCHES))
    for index, (color, hatch) in enumerate(zip(colors, hatches, strict=True)):
        start, stop = bounds[index], bounds[index + 1]
        ax.bar(
            np.arange(start, stop, dtype=float) + bin_gap / 2,
            heights[start:stop],
            width=1.0 - bin_gap,
            align="edge",
            facecolor=TRANSPARENT,
            edgecolor=color,
            hatch=hatch,
            linewidth=BAR_LINE_WIDTH,
        )
    return AcceptanceBars(
        bounds=bounds,
        heights=heights,
        shares=list(reversed(histogram.shares)),
        text_colors=list(reversed(BUCKET_TEXT_COLORS)),
    )


def label_acceptance_axis(ax: Axes, bars: AcceptanceBars, *, tick_size: float) -> None:
    """Separate the buckets and tick the axis in acceptance probability."""
    for bound in bars.bounds[1:-1]:
        ax.axvline(bound, color=DARK_GRAY, ls="--", lw=SEPARATOR_LINE_WIDTH)
    ax.set_xlim(0, len(bars.heights))
    ax.set_xticks(bars.bounds)
    ax.set_xticklabels(
        [f"$\\leq${exp_tick_label(-RATIO_HIST_CLIP)}"]
        + [exp_tick_label(edge) for edge in BUCKET_EDGES_ASC[1:-1]]
        + [exp_tick_label(0.0)],
        fontsize=tick_size,
    )
