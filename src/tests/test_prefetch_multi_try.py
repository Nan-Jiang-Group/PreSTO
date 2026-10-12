"""Finite-state and seeded-trajectory checks for shared-prefix MultiTryMH.

Run without a GPU:
    python -m pytest /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/tests/test_prefetch_multi_try.py

The oracle is a normalized context-dependent categorical autoregressive model. These tests establish CPU/reference
behavior, not real-backend timing or bitwise agreement across GPU batch layouts.
"""

import copy
import itertools
import math
from types import SimpleNamespace

import numpy as np
import pytest

from power_sharpening.common import prefetch_multi_try as mtm
from power_sharpening.common.prefetch_rank import RANK_FNS


def _logp(prefix, temperature=1.0):
    logits = np.array([-0.8, 0.3, 1.1])
    logits += ((np.arange(3) + sum(prefix)) % 3) * 0.17
    logits /= temperature
    return logits - np.logaddexp.reduce(logits)


class _Oracle:
    def __init__(self):
        self.calls = []

    def __call__(self, requests, temperatures):
        self.calls.append(requests)
        proposals = []
        for prefix, length, sample_temperature, seed in requests:
            rng = np.random.default_rng(seed)
            prefix = prefix.copy()
            tokens, base, components = [], [], [[] for _ in temperatures]
            for _ in range(length):
                token = int(rng.choice(3, p=np.exp(_logp(prefix, sample_temperature))))
                base.append(float(_logp(prefix)[token]))
                for scores, temperature in zip(components, temperatures):
                    scores.append(float(_logp(prefix, temperature)[token]))
                prefix.append(token)
                tokens.append(token)
            proposals.append((tokens, base, np.asarray(components)))
        return proposals, {}


def _independent_replay(prompt, records, alpha, temperatures):
    """Replay recorded proposal pools and RNG states with independent MTM math.

    Recompute each candidate's scores from the realized conditioning prefix; reconstruct accepted states and block
    extensions without calling the planner, mixture helper, or implementation's transition function.
    """
    tokens, base = [], []
    components = np.empty((len(temperatures), 0))
    trajectory = []

    def scores(prefix, suffix):
        logp, logq = [], [[] for _ in temperatures]
        for token in suffix:
            logp.append(float(_logp(prefix)[token]))
            for row, temperature in zip(logq, temperatures):
                row.append(float(_logp(prefix, temperature)[token]))
            prefix = prefix + [token]
        return logp, np.asarray(logq)

    for before, proposals, cut, rng_state, after in records:
        before_tokens, before_base, before_components = before
        after_tokens, after_base, after_components = after
        # A longer horizon identifies the next block's initial extension.
        assert before_tokens[:len(tokens)] == tokens
        np.testing.assert_allclose(before_base[:len(tokens)], base, atol=1e-14)
        np.testing.assert_allclose(before_components[:, :len(tokens)], components, atol=1e-14)
        if len(before_tokens) > len(tokens):
            extension = before_tokens[len(tokens):]
            extension_base, extension_components = scores(prompt + tokens, extension)
            tokens += extension
            base += extension_base
            components = np.concatenate([components, extension_components], axis=1)
        assert before_tokens == tokens
        np.testing.assert_allclose(before_base, base, atol=1e-14)
        np.testing.assert_allclose(before_components, components, atol=1e-14)
        candidate_scores = [scores(prompt + tokens[:cut], suffix) for suffix, _, _ in proposals]
        for (_, proposal_base, proposal_components), (candidate_base, candidate_components) in zip(proposals, candidate_scores):
            np.testing.assert_allclose(proposal_base, candidate_base, atol=1e-14)
            np.testing.assert_allclose(proposal_components, candidate_components, atol=1e-14)
        current_q = np.logaddexp.reduce(components[:, cut:].sum(axis=1)) - math.log(len(temperatures))
        current_p = math.fsum(base[cut:])
        log_weights = np.array([
            alpha * (math.fsum(candidate_base) - current_p) + current_q
            - (np.logaddexp.reduce(candidate_components.sum(axis=1)) - math.log(len(temperatures)))
            for candidate_base, candidate_components in candidate_scores
        ])
        weights = np.exp(log_weights)
        rng = np.random.default_rng()
        rng.bit_generator.state = rng_state
        selected = int(rng.choice(len(proposals), p=weights / weights.sum()))
        probability = min(1.0, weights.sum() / (1 + np.delete(weights, selected).sum()))
        accepted = bool(rng.random() < probability)
        if accepted:
            suffix, _, _ = proposals[selected]
            tokens[cut:] = suffix
            base[cut:], components[:, cut:] = candidate_scores[selected]
        assert after_tokens == tokens
        np.testing.assert_allclose(after_base, base, atol=1e-14)
        np.testing.assert_allclose(after_components, components, atol=1e-14)
        trajectory.append((cut, selected, accepted))
    return tokens, base, components, trajectory


@pytest.mark.parametrize("rank", sorted(RANK_FNS))
@pytest.mark.parametrize("tries", [1, 3])
@pytest.mark.parametrize("seed", [2, 19, 101])
def test_prefetch_matches_independent_replay_with_rank_name_or_callable(rank, tries, seed, monkeypatch):
    settings = dict(alpha=3.0, proposal_temperatures=[0.25, 0.7, 1.0], num_tries=tries,
                    mcmc_steps=12, num_of_blocks=2, max_new_tokens=6, seed=seed)
    records = []
    transition = mtm._transition

    def record_transition(state, proposals, cut, alpha, rng):
        before = (copy.deepcopy(state), copy.deepcopy(proposals), cut, copy.deepcopy(rng.bit_generator.state))
        after, event = transition(state, proposals, cut, alpha, rng)
        records.append((*before, copy.deepcopy(after)))
        return after, event

    monkeypatch.setattr(mtm, "_transition", record_transition)
    for capacity in [1, 4]:
        records.clear()
        oracle = _Oracle()
        budget = tries*capacity+int(tries>1)
        (tokens, base, components), stats = mtm.sample_subtree_multi_try(
            [2], oracle, prefetch_budget=budget, rank_fn=rank, **settings,
        )
        expected_tokens, expected_base, expected_components, expected_trajectory = _independent_replay(
            [2], records, settings["alpha"], settings["proposal_temperatures"],
        )
        assert tokens == expected_tokens
        np.testing.assert_allclose(base, expected_base, atol=1e-14)
        np.testing.assert_allclose(components, expected_components, atol=1e-14)
        assert [(t["cut"], t["selected"], t["accepted"]) for t in stats.transitions] == expected_trajectory
        assert stats.total_walked_steps == 24
        assert all(size % tries == 0 and size <= budget for size in stats.batch_sizes)
        assert stats.total_workload == tries * stats.total_bundles
        assert stats.total_nfe == len(oracle.calls)
        assert stats.to_json()["unused_suffixes"] >= 0
        repeated_oracle = _Oracle()
        (repeated_tokens, _, _), repeated_stats = mtm.sample_subtree_multi_try(
            [2], repeated_oracle, prefetch_budget=budget, rank_fn=RANK_FNS[rank], **settings,
        )
        assert repeated_tokens == tokens
        assert repeated_stats.to_json() == stats.to_json()
        assert repeated_oracle.calls == oracle.calls


@pytest.mark.parametrize("prefer_accept, expected_nodes", [(True, [0, 1]), (False, [0, 3])])
def test_custom_rank_controls_which_nodes_fit_the_budget(prefer_accept, expected_nodes):
    """Different priorities select different continuations from identical cuts."""
    calls = []

    def rank(node_idx, cut, parent_cut, depth, num_accepts):
        calls.append((node_idx, cut, parent_cut, depth, num_accepts))
        return depth, -num_accepts if prefer_accept else num_accepts

    nodes, pools, limit_hit = mtm._collect_prefetch_nodes(
        rng=np.random.default_rng(0), pending_cuts={0: 3, 1: 1, 2: 2, 3: 0},
        root_node_id=0, current_step=0, response_length=4,
        remaining_steps=2, candidate_pool_capacity=2, num_tries=2, rank_fn=rank,
    )
    assert list(nodes) == expected_nodes
    assert len(pools) == 2
    assert not limit_hit
    assert calls == [(0, 3, 4, 0, 0), (1, 1, 3, 1, 1), (2, 2, 3, 1, 1), (3, 0, 3, 1, 0)]


def test_each_candidate_has_its_own_child_and_rejection_keeps_the_prefix_bound():
    nodes, pools, limit_hit = mtm._collect_prefetch_nodes(
        rng=np.random.default_rng(0), pending_cuts={0: 2, 1: 0, 2: 1, 3: 2, 4: 4},
        root_node_id=0, current_step=0, response_length=5,
        remaining_steps=2, candidate_pool_capacity=5, num_tries=3,
        rank_fn=lambda node_idx, cut, parent_cut, depth, num_accepts: node_idx,
    )
    assert list(nodes) == [0, 1, 2, 3, 4]
    assert [nodes[index].cut_idx for index in [1, 2, 3, 4]] == [0, 1, 2, 4]
    assert [nodes[index].prefix_bound for index in [1, 2, 3, 4]] == [2, 2, 2, 5]
    assert [nodes[index].num_accepts for index in [1, 2, 3, 4]] == [1, 1, 1, 0]
    assert len(pools) == 5
    assert not limit_hit


def test_two_candidate_pools_cover_all_five_outcomes_at_the_next_depth():
    nodes, pools, limit_hit = mtm._collect_prefetch_nodes(
        rng=np.random.default_rng(0), pending_cuts=dict.fromkeys(range(6), 0),
        root_node_id=0, current_step=0, response_length=3,
        remaining_steps=2, candidate_pool_capacity=2, num_tries=4,
        rank_fn=lambda node_idx, cut, parent_cut, depth, num_accepts: node_idx,
    )
    assert list(nodes) == [0, 1, 2, 3, 4, 5]
    assert [[node.node_id for node in pool] for pool in pools.values()] == [[0], [1, 2, 3, 4, 5]]
    assert not limit_hit


@pytest.mark.parametrize("last_accept_cut, rejection_cut, expected_children", [
    (0, 0, [4, 5]), (1, 1, [5]),
])
def test_infeasible_early_siblings_do_not_hide_later_feasible_candidate_pools(
    last_accept_cut, rejection_cut, expected_children,
):
    pending_cuts = {0: 0, 1: 1, 2: 1, 3: 1, 4: last_accept_cut, 5: rejection_cut}
    nodes, pools, limit_hit = mtm._collect_prefetch_nodes(
        rng=np.random.default_rng(0), pending_cuts=pending_cuts,
        root_node_id=0, current_step=0, response_length=2,
        remaining_steps=2, candidate_pool_capacity=2, num_tries=4,
        rank_fn=lambda node_idx, cut, parent_cut, depth, num_accepts: node_idx,
    )
    assert list(nodes) == [0] + expected_children
    assert len(pools) == 2
    assert [nodes[index].cut_idx for index in expected_children] == [rejection_cut] * len(expected_children)
    assert not limit_hit


def test_rank_indices_restart_at_zero_when_a_nonzero_node_becomes_root():
    calls = []

    def rank(node_idx, cut, parent_cut, depth, num_accepts):
        calls.append((node_idx, depth))
        return node_idx

    persistent_ids = [7, 22, 23, 24, 67, 68, 69, 70, 71, 72, 73, 74, 75]
    nodes, pools, limit_hit = mtm._collect_prefetch_nodes(
        rng=np.random.default_rng(0), pending_cuts=dict.fromkeys(persistent_ids, 0),
        root_node_id=7, current_step=5, response_length=4,
        remaining_steps=3, candidate_pool_capacity=13, num_tries=2, rank_fn=rank,
    )
    assert list(nodes) == persistent_ids
    assert calls == [(0, 0)] + [(index, 1) for index in [1, 2, 3]] + [
        (index, 2) for index in range(4, 13)
    ]
    assert len(pools) == 3
    assert not limit_hit


@pytest.mark.parametrize("selected, accepted, child_id, child_cut", [
    (0, True, 1, 0), (1, True, 2, 1), (2, True, 3, 2), (1, False, 4, 3),
])
def test_sampling_follows_the_selected_candidate_or_rejection_child(
    selected, accepted, child_id, child_cut, monkeypatch,
):
    cuts = {0: 3, 1: 0, 2: 1, 3: 2, 4: 3}
    monkeypatch.setattr(mtm, "_draw_cut", lambda rng, node_id, horizon, pending_cuts: cuts[node_id])
    rng = SimpleNamespace(
        integers=np.random.default_rng(7).integers,
        choice=lambda n, p: selected,
        random=lambda: 0.0 if accepted else 1.0,
    )
    _, stats = mtm.sample_subtree_multi_try(
        [1], _Oracle(), alpha=1.0, proposal_temperatures=[1.0], num_tries=3,
        mcmc_steps=2, num_of_blocks=1, max_new_tokens=4, prefetch_budget=15, rng=rng,
    )
    assert [(event["node_id"], event["cut"]) for event in stats.transitions] == [
        (0, 3), (child_id, child_cut),
    ]
    assert stats.transitions[0]["next_tree_index"] == child_id
    assert stats.transitions[1]["tree_index"] == child_id
    assert stats.walked_steps == [2]


@pytest.mark.parametrize("tries", [1, 2, 3])
def test_exact_enumerated_kernel_is_normalized_and_reversible(tries):
    p = np.array([0.12, 0.31, 0.57])
    alpha = 2.7
    pi = p**alpha
    pi /= pi.sum()
    temperatures = [0.5, 1.3]
    components = np.array([p**(1/t) / np.sum(p**(1/t)) for t in temperatures])
    q = components.mean(axis=0)
    transition = np.zeros((3, 3))
    rng = SimpleNamespace(choice=lambda n, p: 0, random=lambda: 1.0)
    for current in range(3):
        state = ([current], [math.log(p[current])], np.log(components[:, current:current+1]))
        for bundle in itertools.product(range(3), repeat=tries):
            proposals = [(
                [candidate], [math.log(p[candidate])], np.log(components[:, candidate:candidate+1]),
            ) for candidate in bundle]
            _, event = mtm._transition(state, proposals, 0, alpha, rng)
            probabilities = event["candidate_outcome_probabilities"]
            pool_probability = np.prod(q[list(bundle)])
            transition[current, current] += pool_probability * probabilities[-1]
            for candidate, probability in zip(bundle, probabilities[:-1]):
                transition[current, candidate] += pool_probability * probability
    np.testing.assert_allclose(transition.sum(axis=1), 1.0, atol=2e-14)
    flux = pi[:, None] * transition
    np.testing.assert_allclose(flux, flux.T, atol=2e-14)
    if tries == 1:
        standard_mh = np.zeros_like(transition)
        for x, y in itertools.product(range(3), repeat=2):
            standard_mh[x, y] = q[y] * min(1.0, pi[y]*q[x]/(pi[x]*q[y]))
        for x in range(3):
            standard_mh[x, x] += 1-standard_mh[x].sum()
        np.testing.assert_allclose(transition, standard_mh, atol=2e-14)


def test_same_depth_bundles_share_but_successive_steps_use_fresh_candidates(monkeypatch):
    monkeypatch.setattr(mtm, "_draw_cut", lambda rng, node_id, horizon, pending_cuts: pending_cuts.setdefault(node_id, 0))
    oracle = _Oracle()
    _, stats = mtm.sample_subtree_multi_try(
        [1], oracle, alpha=1.0, proposal_temperatures=[1.0], num_tries=4,
        mcmc_steps=4, num_of_blocks=1, max_new_tokens=3, prefetch_budget=16, seed=3,
    )
    assert stats.total_walked_steps == 4
    assert stats.total_bundles == 4
    assert stats.total_workload == 16
    assert stats.deduplicated_bundles > 0
    assert stats.walked_steps == [4]
    assert stats.planned_nodes == 156
    assert stats.planner_limit_hits == 0
    request_seeds = [seed for _, _, _, seed in oracle.calls[1]]
    assert len(request_seeds) == len(set(request_seeds)) == 16
    assert len(oracle.calls) == 2  # one extension, one prefetched candidate batch


def test_infeasible_pending_cut_is_retained_when_realized_state_becomes_root():
    class ScriptedRng:
        def __init__(self):
            self.rng = np.random.default_rng(3)
            self.cut_draws = []

        def integers(self, low, high=None):
            if (low if high is None else high) == 4:
                assert len(self.cut_draws) < 4, "reroot resampled an already drawn cut"
                cut = [1, 0, 3, 2][len(self.cut_draws)]
                self.cut_draws.append(cut)
                return cut
            return self.rng.integers(low, high)

        def choice(self, n, p):
            return 1  # Select the second candidate's distinct acceptance child.

        def random(self):
            return 0.0  # With alpha=temperature=1, both proposals are accepted.

    rng = ScriptedRng()
    _, stats = mtm.sample_subtree_multi_try(
        [1], _Oracle(), alpha=1.0, proposal_temperatures=[1.0], num_tries=2,
        mcmc_steps=2, num_of_blocks=1, max_new_tokens=4, prefetch_budget=8, rng=rng,
    )
    assert [(event["cut"], event["accepted"]) for event in stats.transitions] == [(1, True), (3, True)]
    assert rng.cut_draws == [1, 0, 3, 2]  # Root, two acceptance children, rejection child.
    assert [event["node_id"] for event in stats.transitions] == [0, 2]
    assert stats.walked_steps == [1, 1]


@pytest.mark.parametrize("stop_on_eos, expected_steps", [(True, 2), (False, 4)])
def test_eos_is_checked_after_full_block_refinement(stop_on_eos, expected_steps):
    def all_eos(requests, temperatures):
        proposals = [(
            [0]*length, [-math.log(3)]*length,
            np.full((len(temperatures), length), -math.log(3)),
        ) for _, length, _, _ in requests]
        return proposals, {}
    (tokens, _, components), stats = mtm.sample_subtree_multi_try(
        [0, 1], all_eos, alpha=1, proposal_temperatures=[1], num_tries=2,
        mcmc_steps=2, num_of_blocks=2, max_new_tokens=6, prefetch_budget=8,
        seed=3, eos_token_id=0, stop_on_eos=stop_on_eos,
    )
    assert tokens == [0]  # prompt EOS is not part of the response
    assert components.shape == (1, 1)
    assert stats.total_walked_steps == expected_steps
    assert stats.to_json()["acceptance_rate"] == 1.0
    assert len(stats.final_base_logprobs) == 1


def test_zero_steps_and_backend_work_accounting():
    oracle = _Oracle()
    def draw(requests, temperatures):
        proposals, _ = oracle(requests, temperatures)
        return proposals, {
            "generation_calls": 2, "scoring_calls": 3, "scoring_requests": 11,
            "generated_tokens": sum(length for _, length, _, _ in requests) + 5,
        }
    _, stats = mtm.sample_subtree_multi_try(
        [1], draw, alpha=2, proposal_temperatures=[0.5, 1], num_tries=2,
        mcmc_steps=0, num_of_blocks=2, max_new_tokens=6, prefetch_budget=4, seed=1,
    )
    summary = stats.to_json(mcmc_steps=0, num_blocks=2)
    assert summary["total_workload"] == summary["total_bundles"] == 0
    assert summary["total_nfe"] == 10
    assert summary["generation_calls"] == 4 and summary["scoring_calls"] == 6
    assert summary["scoring_requests"] == 22
    assert summary["generated_tokens"] == 16
    assert summary["acceptance_rate"] == 0.0
