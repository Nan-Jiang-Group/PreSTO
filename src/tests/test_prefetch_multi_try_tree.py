"""CPU checks for the K+1-way MTM Rich tree and its sampler integration.

Run with:
    python -m pytest /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/tests/test_prefetch_multi_try_tree.py
"""

import copy
import re

import numpy as np
import pytest
from rich.tree import Tree

from power_sharpening.common import prefetch_multi_try as mtm
from power_sharpening.common.multi_try import mtm_edge_probabilities
from power_sharpening.common.prefetch_multi_try import sample_subtree_multi_try
from power_sharpening.common.prefetch_multi_try_tree import (
    _Node,
    _PrefetchNode,
    build_subtree,
    subtree_to_rich_tree,
)
from test_prefetch_multi_try import _Oracle, _logp


def _subtree(num_tries=3):
    root = _Node(
        proposal=list(range(10)), cut_idx=10, node_id=0,
        depth=0, candidate_pool_id=0,
    )
    root.children = [
        _Node(
            proposal=root.proposal[:6] + [candidate] * 4, cut_idx=6,
            node_id=candidate + 1,
            depth=1, candidate_pool_id=1,
        )
        for candidate in range(num_tries)
    ]
    root.children.append(_Node(
        proposal=root.proposal, cut_idx=root.cut_idx,
        node_id=num_tries + 1, depth=1, candidate_pool_id=2,
    ))
    for node in root.children:
        cut_idx = 8 if node is root.children[-1] else 3
        node.children = [
            _Node(
                proposal=node.proposal[:cut_idx] + [candidate] * (10 - cut_idx),
                cut_idx=cut_idx, node_id=(num_tries + 1) * node.node_id + candidate + 1,
                depth=2,
            )
            for candidate in range(num_tries)
        ]
        node.children.append(_Node(
            proposal=node.proposal, cut_idx=node.cut_idx,
            node_id=(num_tries + 1) * node.node_id + num_tries + 1, depth=2,
        ))
    return root


def test_node_matches_proposal_constructor_and_tree_methods(capsys):
    node = _Node(proposal=[7, 8], cut_idx=1, node_id=0)
    assert node.is_leaf()
    node.children = [
        _Node(proposal=[7, 1], cut_idx=1, node_id=1),
        _Node(proposal=[7, 2], cut_idx=1, node_id=2),
        _Node(proposal=node.proposal, cut_idx=node.cut_idx, node_id=3),
    ]
    node.log_accept_ratio = np.array([0.0, 0.0])
    assert not node.is_leaf()
    assert all(child.is_leaf() for child in node.children)
    assert 'log_acc_ratio=[0.0000, 0.0000]' in repr(node)
    assert 'cut_idx=1, seq_len=2' in repr(node)
    text = node.tree_str()
    assert capsys.readouterr().out == ''
    assert 'reject (p_edge=0)' in text
    assert '[accept cand = 1, p_edge=0.5, p_select=0.5, A_mtm=1] Node(' in text


@pytest.mark.parametrize("num_tries", [1, 2, 3])
def test_each_step_has_rejection_and_every_candidate(num_tries, capsys):
    subtree = _subtree(num_tries=num_tries)
    before = copy.deepcopy(subtree)
    tree = subtree_to_rich_tree(subtree_root=subtree)
    assert isinstance(tree, Tree)
    assert tree.label.plain == "Node(id=0, cut_idx=10, seq_len=10)"
    assert len(tree.children) == num_tries + 1
    first_child = 1
    assert tree.children[-1].label.plain.startswith(f"reject -> Node(id={first_child + num_tries},")
    for candidate, branch in enumerate(tree.children[:-1]):
        assert branch.label.plain.startswith(
            f"[accept cand = {candidate}] Node(id={first_child + candidate},"
        )
    # Every accepted outcome gets its own displayed continuation, even though those continuations refer to the same
    # generated candidate pool.
    assert len({id(branch) for branch in tree.children}) == num_tries + 1
    for branch in tree.children:
        assert len(branch.children) == num_tries + 1
        assert branch.children[-1].label.plain.startswith("reject ->")
        for candidate, leaf in enumerate(branch.children[:-1]):
            assert leaf.label.plain.startswith(f"[accept cand = {candidate}] Node(")
        assert all(not leaf.children for leaf in branch.children)
    seen = set()

    def check_indices(branch, expected_index):
        assert _node_index(branch) == expected_index
        assert expected_index not in seen
        seen.add(expected_index)
        for outcome, child in enumerate(branch.children):
            check_indices(child, (num_tries + 1) * expected_index + outcome + 1)

    check_indices(tree, 0)
    assert len(seen) == 1 + (num_tries + 1) + (num_tries + 1) ** 2
    assert subtree == before
    output = capsys.readouterr().out
    assert f"[accept cand = {num_tries - 1}]" in output
    assert "no prefetched continuation" not in output
    assert "depth=" not in output
    assert "candidate_pool_id=" not in output


@pytest.mark.parametrize("suffix, marker", [("txt", "[accept cand = 2]"), ("svg", "<svg"), ("html", "<html")])
def test_tree_exports(suffix, marker, tmp_path, capsys):
    path = tmp_path / f"mtm-tree.{suffix}"
    subtree_to_rich_tree(_subtree(), saveto=str(path))
    assert marker in path.read_text()
    assert "Node(id=0, cut_idx=10, seq_len=10)" in capsys.readouterr().out


def test_depth_limit_is_explicit_and_preserves_root_outcomes(capsys):
    tree = subtree_to_rich_tree(_subtree(), max_depth=1)
    assert len(tree.children) == 4
    assert all(not branch.children for branch in tree.children)
    assert all("display depth limit" in branch.label.plain for branch in tree.children)
    assert "display depth limit" in capsys.readouterr().out


def _edge_value(branch, name):
    return float(re.search(rf"\b{name}=([0-9.eE+-]+)", branch.label.plain).group(1))


def _node_index(branch):
    return int(re.search(r"Node\(id=(\d+),", branch.label.plain).group(1))


@pytest.mark.parametrize("weights, expected_acceptance, expected_edges", [
    ([0.2], [0.2], [0.2, 0.8]),
    ([0.25, 0.5], [0.5, 0.6], [1 / 6, 2 / 5, 13 / 30]),
    ([2, 4], [1, 1], [1 / 3, 2 / 3, 0]),
])
def test_edge_mass_includes_candidate_selection(weights, expected_acceptance, expected_edges):
    selection, acceptance, edges = mtm_edge_probabilities(np.log(weights))
    np.testing.assert_allclose(selection, np.array(weights) / sum(weights))
    np.testing.assert_allclose(acceptance, expected_acceptance)
    np.testing.assert_allclose(edges, expected_edges, atol=1e-15)


@pytest.mark.parametrize("start_step", [0, 2])
@pytest.mark.parametrize("shared_accept_pool", [True, False])
def test_descendant_weights_use_each_candidate_state(start_step, shared_accept_pool, capsys):
    prompt = [2]
    temperatures = [0.5, 1.0]
    alpha = 2.0

    def score(prefix, tokens):
        contexts = [prefix + tokens[:index] for index in range(len(tokens))]
        base = [_logp(context)[token] for context, token in zip(contexts, tokens)]
        components = np.array([
            [_logp(context, temperature)[token] for context, token in zip(contexts, tokens)]
            for temperature in temperatures
        ])
        return tokens, base, components

    # A later batch can start below the persistent root, but its display still starts at zero. This persistent path
    # takes candidate zero at each step.
    root_id = (3 ** start_step - 1) // 2
    root = _PrefetchNode(node_id=root_id, depth=start_step, cut_idx=1, prefix_bound=3, num_accepts=0)
    accepted = [
        _PrefetchNode(
            node_id=3 * root.node_id + outcome + 1, depth=start_step + 1,
            cut_idx=0 if outcome == 0 or shared_accept_pool else 1,
            prefix_bound=1, num_accepts=1,
        )
        for outcome in range(2)
    ]
    rejected = _PrefetchNode(node_id=3 * root.node_id + 3, depth=start_step + 1, cut_idx=2, prefix_bound=3, num_accepts=0)
    prefetched_nodes = {node.node_id: node for node in (root, *accepted, rejected)}
    state = score(prompt, [2, 2, 2])
    root_tokens, _, _ = state
    pools = {
        (root.depth, 1): [score(prompt + [2], [0, 0]), score(prompt + [2], [1, 1])],
        (accepted[0].depth, 0): [score(prompt, [0, 0, 0]), score(prompt, [0, 0, 1])],
        (rejected.depth, 2): [score(prompt + [2, 2], [0]), score(prompt + [2, 2], [1])],
    }
    if not shared_accept_pool:
        pools[(accepted[1].depth, 1)] = [score(prompt + [2], [0, 1]), score(prompt + [2], [1, 0])]
    pool_to_batch_idx = {key: 2 * index for index, key in enumerate(pools)}
    node_to_batch_idx = {
        node.node_id: pool_to_batch_idx[(node.depth, node.cut_idx)]
        for node in prefetched_nodes.values()
    }
    before = copy.deepcopy((state, pools))
    subtree_root = build_subtree(
        subtree_root_proposal=state,
        sampled_proposals=[proposal for pool in pools.values() for proposal in pool],
        prefetched_nodes=prefetched_nodes,
        node_to_batch_idx=node_to_batch_idx,
        num_tries=2, alpha=alpha,
    )
    assert subtree_root.node_id == 0
    assert subtree_root.depth == 0
    assert subtree_root.cut_idx == len(root_tokens)
    assert subtree_root.children[-1].proposal == root_tokens
    assert subtree_root.children[-1].cut_idx == subtree_root.cut_idx
    assert subtree_root.children[0].proposal != subtree_root.children[1].proposal
    assert subtree_root.children[0].node_id != subtree_root.children[1].node_id
    assert (
        subtree_root.children[0].candidate_pool_id == subtree_root.children[1].candidate_pool_id
    ) == shared_accept_pool
    assert all(child.cut_idx == root.cut_idx for child in subtree_root.children[:-1])
    assert not np.allclose(
        subtree_root.children[0].log_accept_ratio, subtree_root.children[1].log_accept_ratio,
    )
    tree = subtree_to_rich_tree(subtree_root=subtree_root)

    def check(branch, node, tokens, proposal_node):
        assert proposal_node.proposal == tokens
        assert proposal_node.node_id == _node_index(branch)
        assert proposal_node.depth == node.depth - start_step
        _, current_base, current_components = score(prompt, tokens)
        proposals = pools[(node.depth, node.cut_idx)]
        current_q = np.exp(current_components[:, node.cut_idx:].sum(axis=1)).mean()
        weights = np.array([
            np.exp(alpha * (sum(base) - sum(current_base[node.cut_idx:])))
            * current_q / np.exp(components.sum(axis=1)).mean()
            for _, base, components in proposals
        ])
        np.testing.assert_allclose(proposal_node.log_accept_ratio, np.log(weights), atol=1e-14)
        selection = weights / weights.sum()
        acceptance = np.array([
            min(1, weights.sum() / (1 + sum(weights[i] for i in range(2) if i != j)))
            for j in range(2)
        ])
        expected = [*(selection * acceptance), 1 - np.dot(selection, acceptance)]
        np.testing.assert_allclose(
            [_edge_value(child, 'p_edge') for child in branch.children],
            expected, rtol=5e-6, atol=1e-6,
        )
        for j, child in enumerate(branch.children[:-1]):
            assert _edge_value(child, 'p_select') == pytest.approx(selection[j], rel=5e-6)
            assert _edge_value(child, 'A_mtm') == pytest.approx(acceptance[j], rel=5e-6)
        for outcome, child_branch in enumerate(branch.children):
            accepted = outcome < len(proposals)
            child = prefetched_nodes.get(3 * node.node_id + outcome + 1)
            child_tokens = tokens
            if accepted:
                suffix, _, _ = proposals[outcome]
                child_tokens = tokens[:node.cut_idx] + suffix
            child_node = proposal_node.children[outcome]
            assert child_node.proposal == child_tokens
            assert child_node.cut_idx == (node.cut_idx if accepted else proposal_node.cut_idx)
            if child is not None:
                check(child_branch, child, child_tokens, child_node)
            else:
                assert child_node.is_leaf()

    check(tree, root, root_tokens, subtree_root)
    assert _edge_value(tree.children[0].children[-1], 'p_edge') != pytest.approx(
        _edge_value(tree.children[1].children[-1], 'p_edge'),
    )
    for (original_tokens, original_base, original_components), (after_tokens, after_base, after_components) in zip(
        [before[0], *(proposal for pool in before[1].values() for proposal in pool)],
        [state, *(proposal for pool in pools.values() for proposal in pool)],
    ):
        assert original_tokens == after_tokens
        np.testing.assert_array_equal(original_base, after_base)
        np.testing.assert_array_equal(original_components, after_components)


@pytest.mark.parametrize("num_tries", [1, 2, 3])
def test_print_tree_does_not_change_sampling(num_tries, capsys, monkeypatch):
    settings = dict(
        alpha=3, proposal_temperatures=[0.25, 1.0], num_tries=num_tries,
        prefetch_budget=2 * num_tries, mcmc_steps=4, num_of_blocks=2, max_new_tokens=4,
    )
    rng = np.random.default_rng(17)
    (tokens, _, components), stats = sample_subtree_multi_try([2], _Oracle(), rng=rng, **settings)
    assert capsys.readouterr().out == ""
    rendered_trees = []

    def capture_tree(*args, **kwargs):
        tree = subtree_to_rich_tree(*args, **kwargs)
        rendered_trees.append(tree)
        return tree

    monkeypatch.setattr(mtm, "subtree_to_rich_tree", capture_tree)
    printed_rng = np.random.default_rng(17)
    (printed_tokens, _, printed_components), printed_stats = sample_subtree_multi_try(
        [2], _Oracle(), rng=printed_rng, print_tree=True, **settings,
    )
    output = capsys.readouterr().out
    assert "reject (p_edge=" in output
    assert "[accept cand = 0, p_edge=" in output
    assert f"[accept cand = {num_tries - 1}, p_edge=" in output
    assert "p_select=" in output and "A_mtm=" in output
    assert "prefetched MTM bundles" not in output
    assert printed_tokens == tokens
    np.testing.assert_array_equal(printed_components, components)
    assert printed_stats.to_json() == stats.to_json()
    assert printed_rng.bit_generator.state == rng.bit_generator.state
    transitions = iter(stats.transitions)
    assert len(rendered_trees) == len(stats.walked_steps)
    assert len(rendered_trees) > 1
    path_prefix = "sampled path in the prefetched subtree: "
    path_lines = [line for line in output.splitlines() if line.startswith(path_prefix)]
    assert len(path_lines) == len(rendered_trees)
    for tree, walked, path_line in zip(rendered_trees, stats.walked_steps, path_lines):
        expected_index = 0
        branch = tree
        assert _node_index(branch) == expected_index
        path_parts = [str(expected_index)]
        for _ in range(walked):
            event = next(transitions)
            assert _node_index(branch) == event['tree_index'] == expected_index
            np.testing.assert_allclose(
                [_edge_value(child, 'p_edge') for child in branch.children],
                event['candidate_outcome_probabilities'], rtol=5e-6, atol=1e-6,
            )
            selected = branch.children[event['selected']]
            assert _edge_value(selected, 'A_mtm') == pytest.approx(
                np.exp(event['log_acceptance']), rel=5e-6,
            )
            branch = selected if event['accepted'] else branch.children[-1]
            outcome = event['selected'] if event['accepted'] else num_tries
            expected_index = (num_tries + 1) * expected_index + outcome + 1
            assert _node_index(branch) == event['next_tree_index'] == expected_index
            decision = 'accept' if event['accepted'] else 'reject'
            path_parts.append(
                f"-[{decision}, log_acc_prob={event['log_acceptance']:.4f}]-> {expected_index}"
            )
        assert path_line == path_prefix + ' '.join(path_parts)
