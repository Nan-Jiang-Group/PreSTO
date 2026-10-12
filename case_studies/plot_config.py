"""Shared colors and the SciencePlots style for ``case_studies`` figures.

Figure modules import it as ``case_studies.plot_config`` and call ``apply_plot_style()`` before creating figures.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import matplotlib as mpl
import scienceplots  # noqa: F401  (registers the SciencePlots styles)
from matplotlib import style as mpl_style

# The two methods every comparison figure names. Kept together so a panel title, a legend entry, and a CSV "method"
# column cannot drift apart.
SUBTREE_METHOD_LABEL: Final = "PreSTO-PowerMH"
POWER_METHOD_LABEL: Final = "PowerMH"

# Structural colors are kept outside the data palette.
WHITE: Final = "#FFFFFF"
BLACK: Final = "#000000"
TRANSPARENT: Final = "none"

# Shared publication palette.
BLUE: Final = "#0066CC"
GREEN: Final = "#76CD26"
sglOrange: Final = "#FCB827"  # RGB (252, 184, 39)
ORANGE: Final = sglOrange
RED: Final = "#DE4444"
PURPLE: Final = "#845B97"
DARK_GRAY: Final = "#474747"
GRAY: Final = "#9e9e9e"
COLOR_PALETTE: Final = (
    BLUE,
    GREEN,
    ORANGE,
    RED,
    PURPLE,
    DARK_GRAY,
    GRAY,
)
PANEL_LABEL_FONT_SIZE: Final = 9.0
PANEL_LABEL_FONT_WEIGHT: Final = "bold"
DISTRIBUTION_HISTOGRAM_ALPHA: Final = 0.82
DISTRIBUTION_HISTOGRAM_LINE_WIDTH: Final = 0.55
DISTRIBUTION_HISTOGRAM_BAR_WIDTH: Final = 0.92
DISTRIBUTION_MEAN_LINE_WIDTH: Final = 1.2
# Boxes are drawn over a grid and over each other's fliers, so they sit far lighter than a histogram bar; the median
# line, not the fill, carries the read.
BOXPLOT_FILL_ALPHA: Final = 0.30

CUT_DISTRIBUTION_COLORS: Final = {
    "uniform": BLUE,
    "local_geometric": GREEN,
    "local_zipf": ORANGE,
    "truncated_normal": RED,
    "beta": PURPLE,
    "global_geometric": DARK_GRAY,
    "entropy": GRAY,
}

TRANSITION_RANK_COLORS: Final = {
    "accept_first": BLUE,
    "reject_first": GREEN,
    "longest_first": ORANGE,
    "smallest_cut_diff": RED,
    "longest_path_first": PURPLE,
    "bfs_reject_first": DARK_GRAY,
}

# Descending in acceptance probability: effectively accepted, genuinely stochastic, effectively rejected. Warm at the
# accept end, cool at the reject end, with the one bucket that decides anything in the signal colour between them.
SUBTREE_BUCKET_COLORS: Final = (
    ORANGE,
    RED,
    BLUE,
)
SUBTREE_BUCKET_FILL_ALPHA: Final = 1.0
SUBTREE_BUCKET_TEXT_COLORS: Final = (
    ORANGE,
    RED,
    BLUE,
)
# Forward diagonals for accept and reject, backward diagonals for uncertain.
SUBTREE_BUCKET_HATCHES: Final = ("////", r"\\\\", "////")

PASSK_MARKERS: Final = ("o", "^", "<", "d", "*", "2")

MULTI_TRY_CMAP: Final = "viridis"
MULTI_TRY_BEST_COLOR: Final = RED

# Notebook HTML colors.
NOTEBOOK_HEADING_COLOR: Final = DARK_GRAY
NOTEBOOK_ACCENT_COLOR: Final = BLUE
NOTEBOOK_CODE_BACKGROUND: Final = GRAY
NOTEBOOK_ROW_EVEN: Final = GRAY
NOTEBOOK_ROW_ODD: Final = WHITE
NOTEBOOK_BORDER_COLOR: Final = GRAY

LARGE_FONT_SIZE: Final = 13.0


def apply_plot_style() -> None:
    """Reset Matplotlib and apply the SciencePlots ``science`` style."""
    mpl.rcdefaults()
    mpl_style.use("science")


def scale_font_sizes(parameters: Sequence[str], scale: float) -> None:
    """Scale numeric or named Matplotlib font-size parameters."""
    resolved_sizes = {
        parameter: mpl.font_manager.FontProperties(
            size=mpl.rcParams[parameter]
        ).get_size_in_points()
        for parameter in parameters
    }
    for parameter, size in resolved_sizes.items():
        mpl.rcParams[parameter] = size * scale
