"""CPU tests for pathwise and leaf-categorical MH transition sampling.

Run with:
    uv run --project /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src \
      pytest /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/tests/test_proposal_tree.py
"""

import math

import numpy as np
import pytest

from power_sharpening.common.prefetch_rank import RANK_FNS
from power_sharpening.common.prefetch_subtree import (
    Node,
    build_and_sample_subtree,
    collect_batch_cut_indicesv3,
    sample_mh_tansition_pathwise,
)


def test_rank_modes_use_accept_reject_branch_names():
    """RANK_FNS keys follow accept/reject naming, not the old left/right."""
    assert "accept_first" in RANK_FNS
    assert "reject_first" in RANK_FNS
    assert "bfs_accept_first" in RANK_FNS
    assert "bfs_reject_first" in RANK_FNS
    assert "left_first" not in RANK_FNS
    assert "right_first" not in RANK_FNS


@pytest.mark.parametrize(
    ("rank", "expected_accept_nodes"),
    [
        ("bfs_accept_first", [1, 3, 5, 7, 9, 11, 13]),
        ("bfs_reject_first", [1, 5, 3, 13, 11, 9, 7]),
    ],
)
def test_bfs_rank_modes_break_level_ties_by_branch_direction(
    rank,
    expected_accept_nodes,
):
    """BFS modes visit the requested branch first within one tree level."""
    _, node_to_batch_idx, _ = collect_batch_cut_indicesv3(
        max_batch_size=7,
        prompt_len=3,
        seq_len=4,
        mcmc_steps=3,
        rank=rank,
    )
    accept_nodes = [
        node_idx
        for node_idx, _ in sorted(
            node_to_batch_idx.items(),
            key=lambda item: item[1],
        )
    ]

    assert accept_nodes == expected_accept_nodes


def test_build_and_sample_subtree_forces_accept_on_infinite_log_ratio():
    """End-to-end build_and_sample_subtree: a +inf log ratio forces the single
    accept leaf."""
    leaf, last_accepted, trajectory, accepts = build_and_sample_subtree(
        subtree_root_proposal=[0],
        sampled_proposals=[[1]],
        log_acceptance_ratios=[math.inf],
        node_to_batch_idx={1: 0},
        parent_to_accept_child={0: 1},
        batch_cut_indices=[0],
    )

    assert leaf.node_id == 1
    assert last_accepted.node_id == 1
    assert [step[0] for step in trajectory] == ["accept"]
    assert accepts == 1


@pytest.mark.parametrize(
    ("log_accept_ratio", "expected_leaf_id"),
    [
        (math.inf, 1),
        (-math.inf, 2),
    ],
)
def test_pathwise_handles_deterministic_transition_probabilities(
    log_accept_ratio,
    expected_leaf_id,
):
    """±inf log ratios yield accept probability exactly 1 or 0 without NaNs:
    the forced branch's leaf is always selected."""
    root = Node(
        node_id=0,
        proposal=[0],
        log_accept_ratio=log_accept_ratio,
        left=Node(node_id=1, proposal=[1]),
        right=Node(node_id=2, proposal=[0]),
    )

    leaf, _, _, _ = sample_mh_tansition_pathwise(
        root,
        rng=np.random.default_rng(0),
    )

    assert leaf.node_id == expected_leaf_id
