"""Tests for proposal-node tree rendering.

Run with:
    /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/.venv/bin/python \
      -m pytest -q \
      /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/tests/test_prefetch_node.py
"""

import math

import pytest

from power_sharpening.common.prefetch_subtree import Node, subtree_to_rich_tree


def _epsilon_certain_subtree():
    """Build a tree containing certain-accept and certain-reject decisions."""
    root = Node(node_id=0, proposal=[0], log_accept_ratio=math.log(0.999))
    accepted = Node(node_id=1, proposal=[1], log_accept_ratio=math.log(0.001))
    rejected = Node(node_id=2, proposal=[0], log_accept_ratio=math.log(0.5))
    accepted.left = Node(node_id=3, proposal=[3])
    accepted.right = Node(node_id=4, proposal=[1])
    rejected.left = Node(node_id=5, proposal=[5])
    rejected.right = Node(node_id=6, proposal=[0])
    root.left = accepted
    root.right = rejected
    return root


def test_subtree_to_rich_tree_prunes_epsilon_certain_branches(tmp_path):
    """Epsilon pruning hides the negligible branch and its whole subtree."""
    root = _epsilon_certain_subtree()
    rejected = root.right
    full_path = tmp_path / "full-tree.txt"
    pruned_path = tmp_path / "pruned-tree.txt"

    subtree_to_rich_tree(root, saveto=str(full_path))
    subtree_to_rich_tree(root, saveto=str(pruned_path), epsilon=0.01)

    full_tree = full_path.read_text()
    pruned_tree = pruned_path.read_text()
    assert all(f"Node(id={node_id}," in full_tree for node_id in range(7))
    assert all(f"Node(id={node_id}," in pruned_tree for node_id in (0, 1, 4))
    assert all(
        f"Node(id={node_id}," not in pruned_tree for node_id in (2, 3, 5, 6)
    )

    # Rendering is display-only: neither hidden branch is removed from the tree.
    assert root.right is rejected
    assert root.left.left.node_id == 3


@pytest.mark.parametrize("epsilon", [-0.01, 0.5, 1.0, math.inf, math.nan])
def test_subtree_to_rich_tree_rejects_invalid_epsilon(epsilon):
    """Epsilon must define disjoint certain-reject/accept regions."""
    with pytest.raises(ValueError, match="0 <= epsilon < 0.5"):
        subtree_to_rich_tree(_epsilon_certain_subtree(), epsilon=epsilon)
