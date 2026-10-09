"""Collect per-call subtree metrics across traversal rank functions.

This library is not run directly; invoke a sibling drawing CLI instead.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from case_studies.extract.logs.proposal_log import ParsedLog, bucket_shares, parse_log
from case_studies.extract.run_naming import (
    LOG_GLOB,
    RANK_ORDER,
    excluded_rank_log,
    prefetch_budget_of,
    rank_of,
)


@dataclass(frozen=True)
class PrefetchCall:
    """One realized walk through a prefetched tree."""

    path_nodes: int


@dataclass(frozen=True)
class RankCalls:
    """Every prefetch call logged by one traversal rank function."""

    rank_fn: str
    log_path: Path
    calls: tuple[PrefetchCall, ...]
    proposal_shares: np.ndarray
    proposal_nodes: int

    def sort_key(self) -> tuple[int, str]:
        """Stable order shared by rank-comparison figures."""
        return (RANK_ORDER.get(self.rank_fn, len(RANK_ORDER)), self.rank_fn)


@dataclass(frozen=True)
class RankComparison:
    """One batch size's rank sweep and its pooled proposal statistics."""

    collected: tuple[RankCalls, ...]
    batch_size: int

    def realized_transitions(self) -> list[list[float]]:
        """Return transitions walked per prefetch call, one series per rank."""
        return [
            [float(call.path_nodes - 1) for call in entry.calls]
            for entry in self.collected
        ]

    def transition_summary_rows(self) -> list[dict[str, object]]:
        """Return realized-transition source rows in visual traversal order."""
        return [
            {
                "prefetch_budget": self.batch_size,
                "rank_fn": entry.rank_fn,
                "prefetch_calls": len(entry.calls),
                "median_mh_transitions": (
                    f"{np.median([call.path_nodes - 1 for call in entry.calls]):.1f}"
                ),
            }
            for entry in self.collected
        ]


def display_rank_order(collected: Sequence[RankCalls]) -> list[RankCalls]:
    """Return the established visual order used by rank-comparison figures."""
    return sorted(collected, key=RankCalls.sort_key)


def collect_calls(parsed: ParsedLog) -> tuple[PrefetchCall, ...]:
    """Pair every sampled walk with the tree it was drawn on."""
    calls: list[PrefetchCall] = []
    for walk in parsed.paths:
        node_ids = parsed.tree_node_ids.get(walk.tree_index)
        if not parsed.tree_sizes.get(walk.tree_index) or not node_ids:
            continue
        calls.append(PrefetchCall(path_nodes=len(walk.node_ids)))
    return tuple(calls)


def collect_rank_calls(log_paths: list[Path]) -> list[RankCalls]:
    """Parse and validate one supplied log per traversal rank function."""
    collected: list[RankCalls] = []
    for log_path in log_paths:
        parsed = parse_log(log_path)
        calls = collect_calls(parsed)
        if not calls:
            raise ValueError(
                f"no sampled walk with a printed tree size in {log_path}; "
                "the walk is only logged under print_tree=true"
            )
        if not parsed.nodes:
            raise ValueError(f"no proposal nodes in {log_path}")
        collected.append(
            RankCalls(
                rank_fn=rank_of(log_path),
                log_path=log_path,
                calls=calls,
                proposal_shares=100.0 * bucket_shares(parsed.nodes),
                proposal_nodes=len(parsed.nodes),
            )
        )

    counts = Counter(entry.rank_fn for entry in collected)
    duplicates = sorted(rank_fn for rank_fn, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(
            "more than one log per rank function: "
            f"{', '.join(duplicates)}"
        )
    return sorted(collected, key=RankCalls.sort_key)


def discover_rank_calls_by_batch_size(
    directory: Path,
    *,
    include_excluded_ranks: bool = False,
    log_glob: str = LOG_GLOB,
) -> list[tuple[int, list[RankCalls]]]:
    """Parse direct child logs and group rank comparisons by batch size."""
    log_paths = sorted(
        path
        for path in directory.glob(log_glob)
        if include_excluded_ranks or not excluded_rank_log(path)
    )
    if not log_paths:
        raise FileNotFoundError(f"no {log_glob} files directly under {directory}")

    paths_by_batch_size: dict[int, list[Path]] = defaultdict(list)
    for log_path in log_paths:
        paths_by_batch_size[prefetch_budget_of(log_path)].append(log_path)

    grouped: list[tuple[int, list[RankCalls]]] = []
    for batch_size, paths in sorted(paths_by_batch_size.items()):
        try:
            collected = collect_rank_calls(paths)
        except ValueError as error:
            raise ValueError(
                f"{error} for prefetch budget {batch_size} in {directory}"
            ) from error
        grouped.append((batch_size, collected))
    return grouped


def collect_rank_comparisons(
    directory: Path,
    *,
    include_excluded_ranks: bool = False,
    log_glob: str = LOG_GLOB,
) -> list[RankComparison]:
    """Collect one pooled comparison for every discovered prefetch size."""
    return [
        RankComparison(
            collected=tuple(display_rank_order(entries)),
            batch_size=batch_size,
        )
        for batch_size, entries in discover_rank_calls_by_batch_size(
            directory,
            include_excluded_ranks=include_excluded_ranks,
            log_glob=log_glob,
        )
    ]


def all_batch_rank_names(comparisons: Sequence[RankComparison]) -> list[str]:
    """Return traversal rules in the first-seen display order."""
    return list(
        dict.fromkeys(
            entry.rank_fn
            for comparison in comparisons
            for entry in comparison.collected
        )
    )


def rank_observation(
    comparison: RankComparison,
    rank_name: str,
) -> tuple[RankCalls, np.ndarray, int] | None:
    """Return one traversal rule's calls and proposal summary."""
    for entry in comparison.collected:
        if entry.rank_fn == rank_name:
            return entry, entry.proposal_shares, entry.proposal_nodes
    return None


def max_realized_transition(comparisons: Sequence[RankComparison]) -> int:
    """Return the longest realized walk across all comparisons."""
    return int(
        max(
            value
            for comparison in comparisons
            for series in comparison.realized_transitions()
            for value in series
        )
    )
