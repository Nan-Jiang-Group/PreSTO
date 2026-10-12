"""Frontier ranking functions shared by binary and multi-try prefetching.

Run the CPU checks with:
    python -m pytest /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/tests/test_proposal_tree.py
"""

from typing import Any, Callable

# Built-in rank functions for subtree prefetching.
# Signature: (node_idx, cut, parent_cut, depth, num_accepts) -> score;
# lower score pops first. Scores only need to be comparable, so a rank function may also return a tuple for
# lexicographic ordering.
RANK_FNS: dict[str, Callable[[int, int, int, int, int], Any]] = {
    # score = #rejects on the path: acceptance-heavy paths expand first,
    # plunging down deep accept chains.
    "accept_first": lambda node_idx, cut, parent_cut, depth, num_accepts: depth - num_accepts,
    # score = #accepts on the path: rejection-heavy paths expand first,
    # chasing down reject chains.
    "reject_first": lambda node_idx, cut, parent_cut, depth, num_accepts: num_accepts,
    # score = cut index: proposals that resample the longest suffix
    # (smallest cut) expand first.
    "longest_first": lambda node_idx, cut, parent_cut, depth, num_accepts: cut,
    # score = parent_cut - cut: nodes whose cut index is closest to their
    # parent's expand first (reject children have diff 0, accept children the sampled gap to the parent's cut).
    "smallest_cut_diff": lambda node_idx, cut, parent_cut, depth, num_accepts: parent_cut - cut,
    # score = depth: the longest feasible path from a node is the remaining
    # tree depth (mcmc_steps - depth) — reject edges are always feasible, so maximizing it means minimizing depth (plain
    # BFS; accept children win ties via FIFO order).
    "longest_path_first": lambda node_idx, cut, parent_cut, depth, num_accepts: depth,
    # score = (depth, node_idx): plain BFS but accept (left) children pop
    # first within each level (within a level, smaller node index places accept branches earlier in lexicographic path
    # order).
    "bfs_accept_first": lambda node_idx, cut, parent_cut, depth, num_accepts: (depth, node_idx),
    # score = (depth, -node_idx): plain BFS but reject (right) children pop
    # first within each level (within a level, larger node index = more rejects on the path).
    "bfs_reject_first": lambda node_idx, cut, parent_cut, depth, num_accepts: (depth, -node_idx),
}
