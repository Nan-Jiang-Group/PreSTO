"""Shared-prefix subtree prefetching for the existing independent MTM kernel.

Run the CPU checks with:
    python -m pytest /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/tests/test_prefetch_multi_try.py

Ye and Lu's PMP-MCMC (https://doi.org/10.1109/TAI.2024.3385384) is the multiproposal-prefetching precedent. This module
retains this repository's MTM selection and acceptance rule, rather than PMP's cloud-resampling kernel. Every node
schedules a COMPLETE K-candidate update and has K accepted children plus one rejection child. Each child has its own
node ID and sampled cut. Accepted children retain the prefix before the parent's cut; the rejection child retains the
earlier prefix bound. Children with matching depth and cut can share a candidate pool while keeping their distinct
states and MH weights.

Cuts remain iid uniform along the realized path. An infeasible node keeps its sampled cut when it becomes the next root.
Equal-cut candidate pools are shared only between mutually exclusive nodes at the SAME transition depth in one batch.
Never reuse a candidate pool across successive transitions.
"""

from __future__ import annotations

import heapq
import itertools
import logging
import math
from typing import Any, Callable

import numpy as np

from power_sharpening.common.multi_try import (
    DEFAULT_PROPOSAL_TEMPERATURES,
    categorical_from_log_weights,
    log_acceptance_probability,
    mtm_edge_probabilities,
    suffix_log_weights,
)
from power_sharpening.common.prefetch_multi_try_stats import (
    MultiTryPrefetchStats,
    _record_draw,
)
from power_sharpening.common.prefetch_multi_try_tree import (
    _PrefetchNode,
    build_subtree,
    subtree_to_rich_tree,
)
from power_sharpening.common.prefetch_rank import RANK_FNS


logger = logging.getLogger("[subtree MultiTryMH]")


def _draw_cut(rng, node_id, horizon, pending_cuts):
    """Draw a node's cut once and retain it when the tree is rerooted."""
    if node_id not in pending_cuts:
        pending_cuts[node_id] = int(rng.integers(0, horizon))
    return pending_cuts[node_id]


def _collect_prefetch_nodes(
    rng, pending_cuts, root_node_id, current_step, response_length,
    remaining_steps, candidate_pool_capacity, num_tries,
    rank_fn: str | Callable[[int, int, int, int, int], Any],
):
    """Return (prefetched_nodes, candidate_pools, limit_hit) for the next batch.

    prefetched_nodes maps node IDs to their cut and retained-prefix bounds. candidate_pools groups nodes that can share
    the same K candidate suffixes. The visit bound includes every node through the remaining MH horizon;
    candidate_pool_capacity separately limits generated candidate pools. rank_fn is a RANK_FNS name or (node_idx, cut,
    parent_cut, depth, num_accepts) -> score; lower scores get priority in the prefetch budget. node_idx is relative to
    this subtree's root, which has index 0. Both persistent and relative IDs use child = (K+1) * parent + outcome + 1.
    """
    rank = RANK_FNS[rank_fn] if isinstance(rank_fn, str) else rank_fn
    root = _PrefetchNode(
        node_id=root_node_id, depth=current_step,
        cut_idx=_draw_cut(rng, root_node_id, response_length, pending_cuts),
        prefix_bound=response_length, num_accepts=0,
    )
    serial = itertools.count()
    heap = [(rank(0, root.cut_idx, response_length, 0, 0), next(serial), root, 0)]
    prefetched_nodes = {}
    candidate_pools = {}
    visited_nodes = 0
    # MH updates occupy depths 0 through remaining_steps - 1. Shared pools can serve many nodes, so the pool budget is
    # not a bound on node visits.
    node_limit = ((num_tries + 1) ** remaining_steps - 1) // num_tries
    while heap and visited_nodes < node_limit:
        _, _, node, relative_id = heapq.heappop(heap)
        visited_nodes += 1
        if node.cut_idx > node.prefix_bound:
            # Keep this cut pending instead of resampling until it is feasible.
            continue
        key = (node.depth, node.cut_idx)
        if key not in candidate_pools and len(candidate_pools) >= candidate_pool_capacity:
            """Existing candidate pools can serve other paths after the budget is full."""
            continue
        prefetched_nodes[node.node_id] = node
        candidate_pools.setdefault(key, []).append(node)
        if node.depth + 1 >= current_step + remaining_steps:
            continue
        for outcome in range(num_tries + 1):
            accepted = outcome < num_tries
            child_id = (num_tries + 1) * node.node_id + outcome + 1
            bound = node.cut_idx if accepted else node.prefix_bound
            child = _PrefetchNode(
                node_id=child_id, depth=node.depth + 1,
                cut_idx=_draw_cut(rng, child_id, response_length, pending_cuts),
                prefix_bound=bound, num_accepts=node.num_accepts + int(accepted),
            )
            relative_depth = child.depth - current_step
            child_relative_id = (num_tries + 1) * relative_id + outcome + 1
            score = rank(child_relative_id, child.cut_idx, node.cut_idx, relative_depth, child.num_accepts)
            heapq.heappush(heap, (score, next(serial), child, child_relative_id))
    limit_hit = bool(heap and visited_nodes >= node_limit)
    return prefetched_nodes, candidate_pools, limit_hit


def _transition(state, proposals, cut, alpha, rng):
    """Resolve one exact existing MTM update against the REALIZED state."""
    weights = suffix_log_weights(state, proposals, cut, alpha)
    selected = categorical_from_log_weights(weights, rng=rng)
    log_acceptance = log_acceptance_probability(weights, selected)
    accepted = rng.random() < math.exp(log_acceptance)
    candidate_selection_probabilities, candidate_acceptance_probabilities, outcome_probabilities = (
        mtm_edge_probabilities(weights)
    )
    logger.info(
        "MTM candidates: cut=%d p_select_cand=[%s] "
        "A_mtm=[%s] selected_index=%d "
        "mtm_acceptance_probability=%.6g accepted=%s",
        cut, ", ".join(f"{probability:.6g}" for probability in candidate_selection_probabilities),
        ", ".join(f"{probability:.6g}" for probability in candidate_acceptance_probabilities),
        selected, math.exp(log_acceptance), accepted,
    )
    positive = outcome_probabilities[outcome_probabilities > 0]
    entropy = -float(np.sum(positive * np.log(positive)))
    if accepted:
        tokens, base, logprobs_by_temperature = state
        suffix, suffix_base, suffix_logprobs_by_temperature = proposals[selected]
        state = (
            tokens[:cut] + suffix,
            base[:cut] + suffix_base,
            np.concatenate([
                logprobs_by_temperature[:, :cut], suffix_logprobs_by_temperature,
            ], axis=1),
        )
    return state, {
        "selected": selected,
        "accepted": bool(accepted),
        "log_weights": weights.tolist(),
        "log_acceptance": log_acceptance,
        "candidate_outcome_probabilities": outcome_probabilities.tolist(),
        "candidate_outcome_entropy": entropy,
        "effective_candidate_outcomes": math.exp(entropy),
    }


def sample_subtree_multi_try(
    prompt_ids: list[int],
    draw_batch: Callable,
    alpha: float,
    proposal_temperatures=DEFAULT_PROPOSAL_TEMPERATURES,
    num_tries: int = 4,
    mcmc_steps: int = 10,
    num_of_blocks: int = 8,
    max_new_tokens: int = 1024,
    prefetch_budget: int = 16,
    rank_fn: str | Callable[[int, int, int, int, int], Any] = "longest_path_first",
    seed: int | None = None,
    rng: np.random.Generator | None = None,
    eos_token_id: int | None = None,
    stop_on_eos: bool = True,
    print_tree: bool = False,
    verbose: bool = False,
) -> tuple[tuple, MultiTryPrefetchStats]:
    """Generate a response in blocks and refine it with prefetched Multi-Try MH.

    After each block extension, refine the accumulated response at its fixed length. Each Multi-Try Metropolis-Hastings
    (MTM) update considers ``num_tries`` candidate suffixes at one cut, selects a candidate by its weight, and accepts
    it or keeps the current response. Prefetching batches candidate pools for possible future tree nodes; updates are
    then committed sequentially along the realized path. Nodes can share a pool when their transition depth, cut, and
    retained prefix match.

    Args:
        prompt_ids: Prompt token IDs used to condition generation. The returned response excludes these tokens.
        draw_batch: Backend callback called as draw_batch(requests, temperatures). Each request is (prefix, length,
            temperature, seed), where prefix includes the prompt and length is the number of suffix tokens to generate.
            Returns (proposals, work_counts) in request order. Each proposal is (token_ids, base_logprobs,
            logprobs_by_temperature), with exactly length tokens, unscaled base-model token log probabilities, and
            per-temperature scores of shape (len(temperatures), length). work_counts optionally reports
            generation_calls, scoring_calls, scoring_requests, and generated_tokens.
        alpha: Exponent in the MH target proportional to
            p(response | prompt)^alpha at the current fixed response length.
        proposal_temperatures: Fixed temperature choices for the suffix-mixture proposal; defaults to (0.25, 0.5, 1.0).
            Each candidate independently chooses one entry uniformly for its whole suffix. MH weights score the mixture
            over all entries. A one-entry list uses one temperature.
        num_tries: Number of candidate suffixes per MTM update, independently of how many proposal temperatures are
            supplied.
        mcmc_steps: Number of MTM updates after each block extension. Both acceptance and rejection count as one
            completed update.
        num_of_blocks: Number of extension/refinement blocks. Each extension adds max_new_tokens // num_of_blocks tokens
            before refinement.
        max_new_tokens: Response-token budget, excluding the prompt. The remainder after division by num_of_blocks is
            unused; EOS handling can shorten the returned response.
        prefetch_budget: Maximum number of candidate suffix requests per prefetch batch. Only prefetch_budget //
            num_tries complete pools fit; any remainder is unused. Setting this to num_tries gives one MTM update per
            candidate-generation batch.
        rank_fn: Prefetch priority, given as a RANK_FNS name or a callable
            (node_idx, cut, parent_cut, depth, num_accepts) -> score. Node index
            and depth are relative to the current subtree root. Lower scores expand first; equal scores use FIFO order.
            This controls scheduling; candidate selection and acceptance use the MH weights.
        seed: Seed for the NumPy generator created when rng is omitted. A fixed seed reproduces the schedule for the
            same settings; changing the prefetch budget or rank can change the sampled path.
        rng: Optional NumPy generator for cuts, temperature choices, proposal seeds, candidate selection, and
            acceptance. Takes precedence over seed when supplied.
        eos_token_id: End-of-sequence token ID. None disables EOS detection and final truncation.
        stop_on_eos: Stop adding blocks when the refined response contains EOS. The current block always completes its
            mcmc_steps first. If False, all blocks run, but final truncation at EOS still applies.
        print_tree: Print each prefetched tree with its outcome probabilities and the path actually taken through that
            tree.
        verbose: Log candidate selection and acceptance details for each MTM update through the module logger.

    Returns:
        (state, stats), where state is (token_ids, base_logprobs, logprobs_by_temperature) for the final response,
        including its first EOS token when present. Per-temperature scores have shape (len(proposal_temperatures),
        len(token_ids)). stats is a MultiTryPrefetchStats object containing backend work counts, prefetch batch counts,
        and the recorded MTM transitions.
    """
    # Initialize the proposal mixture, RNG, statistics, and empty response.
    temperatures = np.asarray(proposal_temperatures, dtype=np.float64).tolist()
    rng = np.random.default_rng(seed) if rng is None else rng
    stats = MultiTryPrefetchStats(int(num_tries), int(prefetch_budget))
    state = ([], [], np.empty((len(temperatures), 0)))
    block_size = max_new_tokens // num_of_blocks
    candidate_pool_capacity = prefetch_budget // num_tries

    for block in range(num_of_blocks):
        logger.info("block [%d/%d] (subtree MultiTryMH)", block, num_of_blocks)

        # 1. Generate one block extension before refining the enlarged response.
        tokens, base, logprobs_by_temperature = state
        temperature = temperatures[int(rng.integers(len(temperatures)))]
        requests = [(
            prompt_ids + tokens, block_size, temperature,
            int(rng.integers(0, 2**63 - 1)),
        )]
        proposals, work = draw_batch(requests, temperatures)

        # Normalize the generated extension before updating the chain state.
        for index, (
            proposal_tokens, proposal_base, proposal_logprobs_by_temperature,
        ) in enumerate(proposals):
            proposals[index] = (
                list(proposal_tokens), np.asarray(proposal_base, dtype=np.float64).tolist(),
                np.asarray(proposal_logprobs_by_temperature, dtype=np.float64),
            )

        # Statistics: record extension backend work.
        _record_draw(stats, work, requests)

        # Append the generated suffix to the current chain state.
        suffix, suffix_base, suffix_logprobs_by_temperature = proposals[0]
        state = (
            tokens + suffix,
            base + suffix_base,
            np.concatenate([logprobs_by_temperature, suffix_logprobs_by_temperature], axis=1),
        )

        # Statistics: count the extension separately from MTM proposals.
        stats.extension_tokens += block_size
        stats.extension_suffixes += 1

        # Keep this response length fixed throughout the block's MH updates.
        horizon = len(tokens) + len(suffix)
        node_id, step = 0, 0
        # Retain sampled cuts across batches, including infeasible future cuts.
        pending_cuts = {}
        while step < mcmc_steps:
            # 2. Plan feasible future MTM nodes from the current chain state.
            # Rank determines prefetch priority; each pool contains K candidates.
            tree_index = 0
            root_state = state
            root_tokens, _, _ = root_state
            prefetched_nodes, candidate_pools, limit_hit = _collect_prefetch_nodes(
                rng=rng,
                pending_cuts=pending_cuts,
                root_node_id=node_id,
                current_step=step,
                response_length=horizon,
                remaining_steps=mcmc_steps - step,
                candidate_pool_capacity=candidate_pool_capacity,
                num_tries=num_tries,
                rank_fn=rank_fn,
            )

            # 3. Generate K suffixes per pool in one batch. Each candidate gets
            # its own temperature draw and seed; shared nodes reuse the K rows.
            requests = []
            node_to_batch_idx = {}
            for (_, cut_idx), nodes in candidate_pools.items():
                for node in nodes:
                    node_to_batch_idx[node.node_id] = len(requests)
                prefix = prompt_ids + root_tokens[:cut_idx]
                length = horizon - cut_idx
                for _ in range(num_tries):
                    temperature = temperatures[int(rng.integers(len(temperatures)))]
                    requests.append((
                        list(prefix), length, temperature,
                        int(rng.integers(0, 2**63 - 1)),
                    ))
            sampled_proposals, work = draw_batch(requests, temperatures)

            # Normalize candidate suffixes before computing MH weights.
            for index, (
                proposal_tokens, proposal_base, proposal_logprobs_by_temperature,
            ) in enumerate(sampled_proposals):
                sampled_proposals[index] = (
                    list(proposal_tokens), np.asarray(proposal_base, dtype=np.float64).tolist(),
                    np.asarray(proposal_logprobs_by_temperature, dtype=np.float64),
                )

            # Statistics: record batch work, pool sharing, and planned nodes.
            _record_draw(
                stats, work, requests,
                num_candidate_pools=len(candidate_pools),
                num_prefetched_nodes=len(prefetched_nodes),
                limit_hit=limit_hit,
            )

            # Statistics: accumulate the progress made by this batch.
            accepted_count, walked = 0, 0

            # 4. Resolve MTM updates sequentially along the realized path.
            # If its next pool is absent, start a new batch from that state.
            while node_id in prefetched_nodes and step < mcmc_steps:
                node = prefetched_nodes[node_id]
                start = node_to_batch_idx[node.node_id]
                proposals = sampled_proposals[start:start + num_tries]

                # Score K candidates against the realized state, select one, and apply MTM acceptance. Rejection keeps
                # the state unchanged.
                state, transition = _transition(state, proposals, node.cut_idx, alpha, rng)
                outcome = transition["selected"] if transition["accepted"] else num_tries
                next_tree_index = (num_tries + 1) * tree_index + outcome + 1

                # Statistics: record the decision and its location in the tree.
                transition.update(
                    block=block, step=step, node_id=node_id, cut=node.cut_idx,
                    tree_index=tree_index, next_tree_index=next_tree_index,
                )
                stats.transitions.append(transition)
                accepted_count += int(transition["accepted"])
                walked += 1

                # Every accepted or rejected decision completes one MH update.
                step += 1
                if verbose or print_tree:
                    logger.info(
                        "MTM step %d/%d node=%d -> node=%d cut=%d selected=%d accepted=%s log_acceptance=%.6f",
                        step, mcmc_steps, tree_index, next_tree_index, node.cut_idx,
                        transition["selected"], transition["accepted"], transition["log_acceptance"],
                    )

                # Follow the selected candidate child, or rejection child K.
                node_id = (num_tries + 1) * node_id + outcome + 1
                tree_index = next_tree_index

            # Optional display: print the tree and its sampled path together after resolving the batch.
            if print_tree:
                subtree_root = build_subtree(
                    subtree_root_proposal=root_state,
                    sampled_proposals=sampled_proposals,
                    node_to_batch_idx=node_to_batch_idx,
                    prefetched_nodes=prefetched_nodes,
                    num_tries=num_tries, alpha=alpha,
                )
                subtree_to_rich_tree(subtree_root=subtree_root)
                batch_transitions = stats.transitions[-walked:]
                parts = [str(batch_transitions[0]["tree_index"])]
                for transition in batch_transitions:
                    decision = "accept" if transition["accepted"] else "reject"
                    """log_acc_prob is log A_mtm of the selected candidate, even on rejection."""
                    parts.append(
                        f"-[{decision}, log_acc_prob={transition['log_acceptance']:.4f}]-> "
                        f"{transition['next_tree_index']}"
                    )
                print("sampled path in the prefetched subtree: " + " ".join(parts), flush=True)

            # Statistics: store the batch's realized acceptance and step counts.
            stats.acceptances.append(accepted_count)
            stats.walked_steps.append(walked)
            logger.info(
                "prefetch batch: bundles=%d suffixes=%d walked_steps=%d accepted=%d",
                len(candidate_pools), len(requests), walked, accepted_count,
            )

        # 5. Check EOS only after completing the block's fixed-horizon updates.
        tokens, base, logprobs_by_temperature = state
        if stop_on_eos and eos_token_id is not None and eos_token_id in tokens:
            break

    # Trim the final response through its first EOS, including that token.
    tokens, base, logprobs_by_temperature = state
    if eos_token_id is not None and eos_token_id in tokens:
        keep = tokens.index(eos_token_id) + 1
        tokens, base, logprobs_by_temperature = (
            tokens[:keep], base[:keep], logprobs_by_temperature[:, :keep],
        )
        state = (tokens, base, logprobs_by_temperature)

    # Statistics: retain base scores for the final likelihood summaries.
    stats.final_base_logprobs = base.copy()
    return state, stats
