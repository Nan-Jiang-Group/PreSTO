"""Shared formatting helpers for subtree-proposal figures.

This library is not run directly; invoke a sibling drawing CLI instead.
"""

from __future__ import annotations

import math

import numpy as np
from matplotlib.axes import Axes


def panel_letter(index: int) -> str:
    """Return the ``(a)``, ``(b)``, ... label for one panel position.

    Past the 26th panel the labels continue ``(aa)``, ``(ab)``, ... rather than running off the end of the alphabet. A
    rank sweep can hold more panels than there are letters, and the characters just past ``z`` include ``{`` and ``}``,
    which LaTeX reads as group delimiters and refuses to typeset.
    """
    if index < 0:
        raise ValueError(f"panel index must be non-negative, got {index}")
    letters = ""
    position = index + 1
    while position:
        position, remainder = divmod(position - 1, 26)
        letters = chr(ord("a") + remainder) + letters
    return f"({letters})"


def hide_secondary_ticks(ax: Axes, **overrides: object) -> None:
    """Keep ticks and their labels on the bottom and left spines only."""
    ax.tick_params(
        axis="both",
        which="both",
        top=False,
        right=False,
        labeltop=False,
        labelright=False,
        **overrides,
    )


def exp_tick_label(nats: float) -> str:
    """Render a log-probability tick as the value it exponentiates to."""
    value = math.exp(nats)
    exponent = math.log10(value) if value > 0 else 0.0
    if abs(exponent) >= 3:
        return f"$10^{{{round(exponent, 1):g}}}$"
    return f"{value:g}"


def percentage_tick_label(value: float, _position: float) -> str:
    """Format percentage ticks without rounding 0.1 percent to zero."""
    return f"${value:g}\\%$"


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Rank values with tied entries sharing their average rank."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=float)
    ranks[order] = np.arange(values.size, dtype=float)
    sorted_values = values[order]
    start = 0
    for stop in range(1, values.size + 1):
        if stop == values.size or sorted_values[stop] != sorted_values[start]:
            ranks[order[start:stop]] = ranks[order[start:stop]].mean()
            start = stop
    return ranks


def rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Return tie-aware Spearman rank correlation for equally sized arrays."""
    return float(np.corrcoef(average_ranks(x), average_ranks(y))[0, 1])


def show_notice(
    ax: Axes,
    title: str,
    message: str,
    *,
    fontsize: float,
    color: str,
) -> None:
    """Render a consistent missing-data notice on an otherwise empty axis."""
    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=color,
        transform=ax.transAxes,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
