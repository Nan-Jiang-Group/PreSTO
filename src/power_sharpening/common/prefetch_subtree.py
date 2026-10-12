"""Binary MH proposal tree construction, sampling, and Rich display.

Run the CPU tests with:
    uv run --project /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src \
      pytest /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/tests/test_proposal_tree.py
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from itertools import count
from typing import Any, Callable, Optional, Sequence
import heapq
import logging
import math
import numpy as np

from power_sharpening.common.cut_distribution import (
    compute_entropy_cut_policy,
    draw_uniform_cut_index,
    summarize_cut_policy,
)
from power_sharpening.common.prefetch_rank import RANK_FNS


logger = logging.getLogger("[subtree prefetching MH]")


# Columns to print a tree at. rich sizes its default console to the terminal, or to 80 when stdout is a file -- and a
# redirected run is the usual case, so deep trees soft-wrapped mid-node. That is not only ugly: the case-study parsers
# match a whole ``Node(...)`` per line, so every wrapped node silently vanished
# from the figures. Wide enough that no realistic depth reaches it.
PRINT_MAX_WIDTH = 1000


@dataclass
class Node(object):
    """MH proposal subtree node.

    Binary tree where left = accept (new proposal), right = reject (keep parent's). BFS-indexed: node k -> left child
    2k+1, right child 2k+2. Only left children appear in node_to_batch_idx.
    """

    proposal: Sequence[int]  # sampled token sequence
    cut_idx: int = -1  # cut the suffix of the sequence
    node_id: int = -1
    left: Optional[Node] = None  # accept branch (None at leaves)
    right: Optional[Node] = None  # reject branch (None at leaves)
    # log(p_target/p_proposal); None at leaves
    log_accept_ratio: Optional[float] = None

    def is_leaf(self) -> bool:
        return self.left is None and self.right is None

    def __repr__(self) -> str:
        acc_text = (
            f"{self.log_accept_ratio:.4f}"
            if self.log_accept_ratio is not None
            else "None"
        )
        return (
            f"Node(id={self.node_id}, log_acc_prob={acc_text}, "
            f"cut_idx={self.cut_idx}, seq_len={len(self.proposal)})"
        )

    def tree_str(self) -> str:
        """Return a multi-line string showing the full tree rooted at this node.

        The reported ``log_acc_prob`` is the unclamped MH log Hastings ratio, so it may be positive; the acceptance
        *probability* it implies is ``min(1, exp(log_acc_prob))``, i.e. a positive value means "always
        accept".  Example output::

            Node(id=0, log_acc_prob=-49.7306, cut_idx=489, seq_len=489)
            ├── [accept] Node(id=1, log_acc_prob=7.9701, cut_idx=412, seq_len=489)
            │   ├── [accept] Node(id=3, log_acc_prob=None, cut_idx=350, seq_len=489)
            │   └── [reject] Node(id=4, log_acc_prob=None, cut_idx=412, seq_len=489)
            └── [reject] Node(id=2, log_acc_prob=-57.0037, cut_idx=489, seq_len=489)
                ├── [accept] Node(id=5, log_acc_prob=None, cut_idx=401, seq_len=489)
                └── [reject] Node(id=6, log_acc_prob=None, cut_idx=489, seq_len=489)
        """
        lines: list[str] = []
        self._tree_str_helper(lines, prefix="", child_prefix="")
        return "\n".join(lines)

    def _tree_str_helper(self, lines: list[str], prefix: str, child_prefix: str) -> None:
        lines.append(prefix + repr(self))
        # left (accept) first, then right (reject)
        children = []
        if self.left is not None:
            children.append((self.left, "[accept] "))
        if self.right is not None:
            children.append((self.right, "[reject] "))
        for i, (child, tag) in enumerate(children):
            is_last = (i == len(children) - 1)
            connector = "└── " if is_last else "├── "
            extension = "    " if is_last else "│   "
            child._tree_str_helper(
                lines,
                prefix=child_prefix + connector + tag,
                child_prefix=child_prefix + extension,
            )


def subtree_to_rich_tree(
    subtree_root: Node,
    saveto: str = None,
    epsilon: Optional[float] = None,
):
    """Print and return the proposal subtree as a ``rich.tree.Tree``.

    The tree is printed to the terminal and also returned.  Labels are built here rather than from ``Node.__repr__`` so
    the MH quantity sits on the edge
    it actually governs:

    - ``log_acc_prob`` is the *unclamped* log Hastings ratio and may be positive, since it is a difference of log
      densities.
    - a node's ratio decides the move to its left child, so it annotates that accept edge rather than a node.  Reject
      edges carry no value; their mass is the complement, ``log(1 - min(1, exp(log_acc_prob)))``.  A node with no accept
      edge below it is a leaf, whose ratio would have been ``None``.

    Example output::

        Node(id=0, cut_idx=12, seq_len=12)
        ├── [accept, log_acc_prob=0.0000] Node(id=1, cut_idx=9, seq_len=12)
        │   ├── [accept, log_acc_prob=-0.1800] Node(id=3, cut_idx=1, seq_len=12)
        │   └── [reject] Node(id=4, cut_idx=9, seq_len=12)
        │       ├── [accept, log_acc_prob=-40.0000] Node(id=9, cut_idx=2, seq_len=12)
        │       └── [reject] Node(id=10, cut_idx=9, seq_len=12)
        └── [reject] Node(id=2, cut_idx=12, seq_len=12)
            ├── [accept, log_acc_prob=-9.2000] Node(id=5, cut_idx=2, seq_len=12)
            └── [reject] Node(id=6, cut_idx=12, seq_len=12)

    ``[accept]`` tags render green and ``[reject]`` red; negative ratios render yellow, while a certain accept renders
    bold green.  Labels are built as ``rich.text.Text`` so the literal square brackets are not parsed as markup.

    When ``epsilon`` is provided, branches made negligible by an epsilon-certain decision are omitted from the display.
    For acceptance probability ``A = min(1, exp(log_accept_ratio))``, the reject child and its subtree are hidden when
    ``A > 1 - epsilon``; the accept child and its subtree are hidden when ``A <= epsilon``.  This only changes the
    rendered tree and does not mutate ``subtree_root``.  With the default ``None``, the full tree is shown.

    When ``saveto`` is given the tree is also written to that file; the format follows the extension — ``.svg`` and
    ``.html`` keep the colors, anything else is plain text.
    """
    import io
    from rich.text import Text
    from rich.tree import Tree
    from rich.console import Console

    def label(
        node: Node,
        tag: str = None,
        style: str = None,
        edge_log_accept_ratio: float = None,
    ) -> Text:
        text = Text()
        if tag is not None:
            # A node's ratio governs the move to its left child, so it annotates the accept edge rather than either
            # endpoint: it is the number the walk tossed against to arrive at this child.
            if edge_log_accept_ratio is None:
                text.append(f"[{tag}] ", style=style)
            else:
                log_ratio = float(edge_log_accept_ratio)
                text.append(f"[{tag}, ", style=style)
                text.append(
                    f"log_acc_prob={log_ratio:.4f}",
                    style="bold green" if log_ratio == 0.0 else "yellow",
                )
                text.append("] ", style=style)
        text.append(
            f"Node(id={node.node_id}, cut_idx={node.cut_idx}, "
            f"seq_len={len(node.proposal)})"
        )
        return text

    def displayed_children(node: Node) -> tuple[bool, bool]:
        """Return whether to display the accept and reject children."""
        show_accept = node.left is not None
        show_reject = node.right is not None
        if epsilon is None or node.log_accept_ratio is None:
            return show_accept, show_reject

        log_ratio = float(node.log_accept_ratio)
        if math.isnan(log_ratio):
            return show_accept, show_reject

        log_acceptance_probability = min(0.0, log_ratio)
        if log_acceptance_probability > math.log(1.0 - epsilon):
            show_reject = False
        elif epsilon > 0.0 and log_acceptance_probability <= math.log(epsilon):
            show_accept = False
        return show_accept, show_reject

    tree = Tree(label(subtree_root))
    stack = [(subtree_root, tree)]
    while stack:
        node, branch = stack.pop()
        show_accept, show_reject = displayed_children(node)
        # add accept before reject so display order matches tree_str
        if show_accept:
            stack.append((
                node.left,
                branch.add(
                    label(node.left, "accept", "green", node.log_accept_ratio)
                ),
            ))
        if show_reject:
            stack.append((node.right, branch.add(label(node.right, "reject", "red"))))

    if saveto is not None:

        # a width wide enough for the longest line, so nothing soft-wraps. measured off the rendered tree, whose labels
        # are wider than tree_str's
        width = Console(width=1 << 16, file=io.StringIO()).measure(tree).maximum + 1
        console = Console(record=True, width=width, file=io.StringIO())
        console.print(tree)
        if saveto.endswith(".svg"):
            console.save_svg(saveto)
        else:
            console.save_text(saveto)

    # An explicit width rather than rich.print, whose console would size itself to the terminal (or to 80 under
    # redirection) and wrap nodes across lines.
    Console(width=PRINT_MAX_WIDTH).print(tree)
    return tree


def collect_batch_cut_indicesv3(
    max_batch_size: int,
    prompt_len: int,
    seq_len: int,
    mcmc_steps: int,
    rank: str | Callable[[int, int, int, int, int], Any] = "accept_first",
    rng: np.random.Generator | None = None,
    cut_dist_type: str = "uniform",
    cut_dist_param: float = None,
    cut_entropies: np.ndarray = None,
) -> tuple[dict[int, int], dict[int, int], list[int]]:
    """Pruned best-first search over the subtree using a weighted (priority) queue.

    Frontier nodes are ranked by a score (lower = expanded sooner; ties broken FIFO, i.e. BFS order). The built-in modes
    — "accept_first", "reject_first", "longest_first", "smallest_cut_diff", "longest_path_first", "bfs_accept_first",
    "bfs_reject_first" — are documented alongside their entries in `RANK_FNS`.

    `rank` may also be a custom callable ``(node_idx, cut, parent_cut, depth, num_accepts) -> score`` returning any
    comparable score.

    `cut_dist_type` selects the law for each accept-child's cut index. Uniform cuts span ``[prompt_len, seq_len - 1]``;
    entropy cuts map adjacent entropy jumps to ``[prompt_len, seq_len - 2]``. `cut_dist_param` is the entropy cut power
    and defaults to 4.0.

    "entropy" additionally requires `cut_entropies`, one base-model predictive entropy per candidate cut (length
    ``seq_len - prompt_len``). It is state-dependent, so callers MUST add the matching `entropy_cut_log_ratio` term to
    each acceptance probability.

    Returns:
        parent_to_child:   parent -> accepted left-child node index.
        node_to_batch_idx: accept-child (left) node index -> batch position.
        batch_cut_indices:  cut indices in expansion (pop) order.
    """
    num_nodes = 2 ** (mcmc_steps + 1) - 1
    parent_to_child: dict[int, int] = {}
    node_to_batch_idx: dict[int, int] = {}
    batch_cut_indices: list[int] = []
    entropy_cut_probabilities = None
    if rng is None:
        rng = np.random.default_rng()

    if cut_dist_type == "entropy":
        if cut_dist_param is None:
            cut_dist_param = 4
        entropy_cut_probabilities = compute_entropy_cut_policy(
            cut_entropies,
            beta=cut_dist_param,
        )
        # lambda_beta is the cut-position distribution each accept-child's cut index is drawn from: lambda_beta[j] ~
        # max(0, H_{j+1} - H_j)^beta.
        logger.info(
            "  cut law: %s",
            summarize_cut_policy(
                cut_entropies,
                entropy_cut_probabilities,
                float(cut_dist_param),
                low=prompt_len,
            ),
        )

    if isinstance(rank, str):
        rank_fn = RANK_FNS[rank]
    else:
        rank_fn = rank

    tie_break = count()  # FIFO tie-break keeps equal-score nodes in BFS order
    root_idx = 0
    # heap entries: (score, tie_break, node_idx, cut, depth, num_accepts) the root has no parent; its own cut doubles as
    # parent_cut (diff 0)
    heap: list[tuple[Any, int, int, int, int, int]] = [
        (rank_fn(root_idx, seq_len, seq_len, 0, 0), next(tie_break), root_idx, seq_len, 0, 0)
    ]

    while heap and len(batch_cut_indices) < max_batch_size:
        _, _, node_idx, parent_cut, depth, num_accepts = heapq.heappop(heap)

        accept = 2 * node_idx + 1
        reject = 2 * node_idx + 2

        # Acceptance branch: claims a batch slot; include only if cut index <= parent's
        if accept < num_nodes:
            if cut_dist_type == "uniform":
                accept_cut = draw_uniform_cut_index(rng, prompt_len, seq_len - 1)
            elif cut_dist_type == "entropy":
                accept_cut = prompt_len + int(
                    rng.choice(
                        entropy_cut_probabilities.size,
                        p=entropy_cut_probabilities,
                    )
                )
            else:
                raise ValueError(f"unknown cut_dist_type {cut_dist_type!r}; expected 'uniform' or 'entropy'")
            if accept_cut <= parent_cut and len(batch_cut_indices) < max_batch_size:
                node_to_batch_idx[accept] = len(batch_cut_indices)
                parent_to_child[node_idx] = accept
                batch_cut_indices.append(accept_cut)
                heapq.heappush(
                    heap,
                    (
                        rank_fn(accept, accept_cut, parent_cut, depth + 1, num_accepts + 1),
                        next(tie_break), accept, accept_cut, depth + 1, num_accepts + 1,
                    ),
                )

        # Rejection branch: inherits the parent's cut index
        if reject < num_nodes and len(batch_cut_indices) < max_batch_size:
            heapq.heappush(
                heap,
                (
                    rank_fn(reject, parent_cut, parent_cut, depth + 1, num_accepts),
                    next(tie_break), reject, parent_cut, depth + 1, num_accepts,
                ),
            )

    if cut_dist_type == "entropy":
        logger.info(
            "  drawn cut indices (batch, pop order): %s; "
            "sampling probabilities (same order): %s",
            batch_cut_indices,
            [
                float(entropy_cut_probabilities[cut - prompt_len])
                for cut in batch_cut_indices
            ],
        )
    return parent_to_child, node_to_batch_idx, batch_cut_indices


def build_subtree(
        subtree_root_proposal: Sequence[int],  # token sequence at the subtree root
        sampled_proposals: Sequence[Sequence[int]],  # batched proposals from the LLM, one per batch slot
        log_acceptance_ratios: np.ndarray,  # MH log acceptance ratio for each proposal
        node_to_batch_idx: dict[int, int],  # tree node index -> batch position
        parent_to_accept_child: dict[int, int],  # parent node index -> accepted left-child node index
        batch_cut_indices: Sequence[int] = None,  # cut index for each batch slot
        verbose=False,
) -> Node:
    """Construct the binary decision subtree from batched proposal results.

    Left child = acceptance (adopts new proposal), right child = rejection (keeps parent's proposal).  Returns the root
    node.
    """
    if verbose:
        print("build the subtree...")
    # Build the subtree root representing the initial (unconditional) proposal.
    root = Node(
        node_id=0,
        proposal=subtree_root_proposal,
        log_accept_ratio=None,
        cut_idx=len(subtree_root_proposal),
    )
    if verbose:
        print("root", root)

    # BFS queue of nodes to expand. Each dequeued node produces a reject-chain for its children; accepted nodes are
    # enqueued for further expansion, ensuring parents are built before children.
    queue: deque[Node] = deque([root])

    while queue:
        cursor = queue.popleft()
        cursor_node_id = cursor.node_id

        child_idx = parent_to_accept_child.get(cursor_node_id, None)
        if child_idx is not None:
            # The acceptance ratio belongs to the child proposal being considered.
            child_batch_pos = node_to_batch_idx[child_idx]
            cursor.log_accept_ratio = log_acceptance_ratios[child_batch_pos]

            # Resolve the cut index for this left child from the batch.
            child_cut = batch_cut_indices[child_batch_pos] if batch_cut_indices is not None else -1

            # Left (accept): adopts the newly sampled proposal and its cut index.
            cursor.left = Node(
                node_id=child_idx,
                proposal=sampled_proposals[child_batch_pos],
                cut_idx=child_cut,
                log_accept_ratio=None,
            )

            # Right (reject): inherits the parent proposal and cut index.
            cursor.right = Node(
                node_id=child_idx + 1,
                proposal=cursor.proposal,
                cut_idx=cursor.cut_idx,
                log_accept_ratio=None,
            )
            # Enqueue the accept node so its descendants can be expanded next.
            queue.append(cursor.left)
            queue.append(cursor.right)
    return root


def build_and_sample_subtree(
        subtree_root_proposal: Sequence[int],  # token sequence at the subtree root
        sampled_proposals: Sequence[Sequence[int]],  # batched proposals from the LLM, one per batch slot
        log_acceptance_ratios: np.ndarray,  # MH log acceptance ratio for each proposal
        node_to_batch_idx: dict[int, int],  # tree node index -> batch position
        parent_to_accept_child: dict[int, int],  # parent node index -> accepted left-child node index
        batch_cut_indices: Sequence[int] = None,  # cut index for each batch slot
        rng: np.random.Generator | None = None,
        print_tree: bool = False,
        verbose=False,
) -> tuple[Node, Node, list[list[Any]], int]:
    """Construct a binary decision subtree, then walk one root-to-leaf path."""
    root = build_subtree(
        subtree_root_proposal=subtree_root_proposal,
        sampled_proposals=sampled_proposals,
        log_acceptance_ratios=log_acceptance_ratios,
        node_to_batch_idx=node_to_batch_idx,
        parent_to_accept_child=parent_to_accept_child,
        batch_cut_indices=batch_cut_indices,
        verbose=verbose,
    )
    return sample_mh_tansition_pathwise(
        root,
        rng=rng,
        print_tree=print_tree,
    )


def _log_transition_probabilities(node: Node) -> float:
    """Return the log acceptance probability of an internal node's accept branch.

    ``A_v = min(1, exp(log_accept_ratio))``; the reject branch carries the complementary probability ``1 - A_v``.
    """
    return min(0.0, float(node.log_accept_ratio))


def sample_mh_tansition_pathwise(
    subtree_root: Node,
    rng: np.random.Generator | None = None,
    print_tree: bool = False,
) -> tuple[Node, Node, list[list[Any]], int]:
    """Randomly walk one root-to-leaf path through a proposal subtree.

    At each internal node:
    - accept with the node's MH acceptance probability and move to `left`
    - reject with the complementary probability and move to `right`

    Returns:
        - leaf_node: the reached leaf node
        - last_accepted_node: the last accepted proposal, or the subtree root
        - trajectory: proposal, log-ratio, and decision at each step
        - number_of_accepts: the number of accept decisions
    """
    if rng is None:
        rng = np.random.default_rng()
    if print_tree:
        # Rich prints the tree itself.
        subtree_to_rich_tree(subtree_root, saveto=None)

    cursor = subtree_root
    trajectory = []
    number_of_accepts = 0
    last_accepted_node = subtree_root  # fallback: root proposal if all rejected
    while not cursor.is_leaf():
        log_accept_probability = _log_transition_probabilities(cursor)
        # -Exp(1) is distributed as log U, so  -e <= log A  <=>  u < A;
        # <= keeps forced accepts (log A = 0) exact even when e == 0.0
        if -rng.exponential(1.0) <= log_accept_probability:
            trajectory.append(["accept", cursor.left.proposal, cursor.log_accept_ratio])
            number_of_accepts += 1
            cursor = cursor.left
            last_accepted_node = cursor  # this node is in node_to_batch_idx
        else:
            trajectory.append(["reject", cursor.proposal, cursor.log_accept_ratio])
            cursor = cursor.right

    if print_tree:
        cursor = subtree_root
        parts = [str(cursor.node_id)]
        for decision, _, log_accept_ratio in trajectory:
            # The departed node's unclamped MH log Hastings ratio, under the same name and precision Node.__repr__ gives
            # it, so a path edge and the tree printed above it carry the same number. It may be positive; the acceptance
            # probability it implies is min(1, exp(.)). Printed on reject edges too, so the log shows the odds the walk
            # decided against rather than only the outcome.
            cursor = cursor.left if decision == "accept" else cursor.right
            parts.append(
                f"-[{decision}, log_acc_prob={float(log_accept_ratio):.4f}]-> "
                f"{cursor.node_id}"
            )
        print("sampled path in the prefetched subtree: " + " ".join(parts))
        print("-" * 60)

    return cursor, last_accepted_node, trajectory, number_of_accepts
