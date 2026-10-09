"""Data structures and Rich tree display for subtree-prefetched Multi-Try MH.

Run the CPU checks with:
    python -m pytest /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/tests/test_prefetch_multi_try_tree.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from power_sharpening.common.multi_try import mtm_edge_probabilities, suffix_log_weights


@dataclass
class _Node:
    """K+1-way MTM node: K candidate children followed by one rejection child.

    children[j] accepts candidate j for j < K; children[K] keeps this proposal. Every new subtree starts at node_id=0
    and depth=0. Child j has ID (K+1) * node_id + j + 1; leaves have no children. log_accept_ratio[j] is candidate j's
    log MH ratio against this proposal. cut_idx is the cut that produced this proposal, as in prefetch_subtree.Node.
    """

    proposal: Sequence[int]
    cut_idx: int = -1
    node_id: int = -1
    children: list[_Node] = field(default_factory=list)
    log_accept_ratio: np.ndarray | None = None
    depth: int = 0
    candidate_pool_id: int | None = None

    def is_leaf(self) -> bool:
        return not self.children

    def __repr__(self) -> str:
        ratios = (
            "[" + ", ".join(f"{value:.4f}" for value in self.log_accept_ratio) + "]"
            if self.log_accept_ratio is not None else "None"
        )
        return (
            f"Node(id={self.node_id}, log_acc_ratio={ratios}, "
            f"cut_idx={self.cut_idx}, seq_len={len(self.proposal)})"
        )

    def tree_str(self) -> str:
        """Return the weighted K+1-way tree as text without printing it."""
        import io
        from rich.console import Console
        from power_sharpening.common.prefetch_subtree import PRINT_MAX_WIDTH

        output = io.StringIO()
        Console(width=PRINT_MAX_WIDTH, file=output).print(_as_rich_tree(self))
        return output.getvalue().rstrip("\n")


@dataclass(frozen=True)
class _PrefetchNode:
    """Cut and retained-prefix bound for a node in the K+1-way prefetch tree.

    The initial root has ID 0; child j has ID (K+1) * node_id + j + 1. IDs persist across batches so a deferred node
    keeps its sampled cut.
    """

    node_id: int
    depth: int
    cut_idx: int
    prefix_bound: int
    num_accepts: int


def build_subtree(
    subtree_root_proposal: tuple,
    sampled_proposals: list[tuple],
    node_to_batch_idx: dict[int, int],
    prefetched_nodes: dict[int, _PrefetchNode],
    num_tries: int,
    alpha: float,
) -> _Node | None:
    """Build a new subtree rooted at node_id=0, depth=0 for this batch.

    Each proposal is (token_ids, base_logprobs, logprobs_by_temperature). node_to_batch_idx maps each prefetched node to
    its first candidate in the batch; every K consecutive suffixes form one candidate pool. Accepted states keep
    distinct nodes and MH ratios even when sharing a pool. The sampler's rank_fn chooses prefetched_nodes before
    generation; this function reconstructs that selection with its MH edge weights.
    """
    root = prefetched_nodes[min(prefetched_nodes)]
    root_tokens, _, _ = subtree_root_proposal
    subtree_root = _Node(
        proposal=list(root_tokens),
        cut_idx=len(root_tokens),
        node_id=0,
        depth=0,
        candidate_pool_id=node_to_batch_idx[root.node_id] // num_tries,
    )
    stack = [(root, subtree_root, subtree_root_proposal)]
    while stack:
        prefetch_node, node, state = stack.pop()
        tokens, base, logprobs_by_temperature = state
        start = node_to_batch_idx[prefetch_node.node_id]
        proposals = sampled_proposals[start:start + num_tries]
        node.log_accept_ratio = suffix_log_weights(state, proposals, prefetch_node.cut_idx, alpha)
        for outcome in range(num_tries + 1):
            accepted = outcome < num_tries
            child_state = state
            if accepted:
                suffix, suffix_base, suffix_logprobs_by_temperature = proposals[outcome]
                child_state = (
                    tokens[:prefetch_node.cut_idx] + suffix,
                    base[:prefetch_node.cut_idx] + suffix_base,
                    np.concatenate([
                        logprobs_by_temperature[:, :prefetch_node.cut_idx], suffix_logprobs_by_temperature,
                    ], axis=1),
                )
            child_tokens, _, _ = child_state
            child = _Node(
                proposal=list(child_tokens),
                cut_idx=prefetch_node.cut_idx if accepted else node.cut_idx,
                node_id=(num_tries + 1) * node.node_id + outcome + 1,
                depth=node.depth + 1,
            )
            node.children.append(child)
            prefetch_child = prefetched_nodes.get(
                (num_tries + 1) * prefetch_node.node_id + outcome + 1
            )
            if prefetch_child is not None:
                child.candidate_pool_id = node_to_batch_idx[prefetch_child.node_id] // num_tries
                stack.append((prefetch_child, child, child_state))
    return subtree_root


def _as_rich_tree(subtree_root: _Node | None, max_depth: int | None = None):
    """Render existing proposal nodes; edge mass is p_select * A_mtm."""
    from rich.text import Text
    from rich.tree import Tree

    def node_label(node):
        return Text(
            f"Node(id={node.node_id}, cut_idx={node.cut_idx}, seq_len={len(node.proposal)})"
        )

    if subtree_root is None:
        return Tree(Text("Empty MTM subtree"))
    tree = Tree(node_label(subtree_root))
    stack = [(subtree_root, tree, 0)]
    while stack:
        node, branch, depth = stack.pop()
        if node.is_leaf():
            continue
        if max_depth is not None and depth >= max_depth:
            branch.label.append(" ... display depth limit", style="dim")
            continue
        if node.log_accept_ratio is not None:
            selection, acceptance, edges = mtm_edge_probabilities(node.log_accept_ratio)
        num_tries = len(node.children) - 1
        for outcome, child in enumerate(node.children):
            accepted = outcome < num_tries
            label = Text(
                f"[accept cand = {outcome}" if accepted else "reject",
                style="green" if accepted else "red",
            )
            if node.log_accept_ratio is not None:
                label.append(
                    f", p_edge={edges[outcome]:.6g}" if accepted else f" (p_edge={edges[outcome]:.6g})"
                )
                if accepted:
                    label.append(
                        f", p_select={selection[outcome]:.6g}, "
                        f"A_mtm={acceptance[outcome]:.6g}"
                    )
            label.append("] " if accepted else " -> ")
            label.append_text(node_label(child))
            stack.append((child, branch.add(label), depth + 1))
    return tree


def subtree_to_rich_tree(
    subtree_root: _Node | None,
    saveto: str | None = None,
    *,
    max_depth: int | None = None,
):
    """Print the K candidate children first, then rejection; return the Rich tree.

    Candidate edges show p_edge = p_select * A_mtm; rejection has probability 1 - sum_j p_edge(j). Node IDs and scores
    come from the built tree. saveto exports text, SVG, or HTML; max_depth limits the displayed steps. Rendering does
    not modify nodes or consume sampling randomness.
    """
    import io
    from pathlib import Path
    from rich.console import Console
    from power_sharpening.common.prefetch_subtree import PRINT_MAX_WIDTH

    tree = _as_rich_tree(subtree_root, max_depth=max_depth)
    Console(width=PRINT_MAX_WIDTH).print(tree)
    if saveto is not None:
        path = Path(saveto)
        width = Console(width=1 << 16, file=io.StringIO()).measure(tree).maximum + 1
        console = Console(record=True, width=width, file=io.StringIO())
        console.print(tree)
        if path.suffix.lower() == ".svg":
            console.save_svg(str(path))
        elif path.suffix.lower() == ".html":
            console.save_html(str(path))
        else:
            console.save_text(str(path))
    return tree
