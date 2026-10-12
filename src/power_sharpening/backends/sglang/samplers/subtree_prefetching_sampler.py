"""Run SGLang subtree MH via its runner; enable cleanup with --evict_subtree_cache."""

import logging
import time
from typing import List
import numpy as np

from power_sharpening.common.prefetch_subtree import (
    build_and_sample_subtree,
    collect_batch_cut_indicesv3,
)
from power_sharpening.common.sample_stats import SamplingStats
from power_sharpening.common.cache_eviction import evict_subtree_cache as _evict_subtree_cache
from power_sharpening.common.tree_print import print_node_features, print_root_sequence_scores
from power_sharpening.backends.sglang.samplers.low_temp_proposal_sampler import low_temp_proposal_sampling
from power_sharpening.backends.sglang.samplers.proposal_model_call import batched_proposal_callv2

logger = logging.getLogger("[subtree prefetching MH]")

def subtree_prefetching_sampling(
    sampler_wrapper,  # wraps the base LLM, proposal, and target distributions
    prompt: List[int],  # fixed prefix tokens (never resampled)
    mcmc_steps: int = 10,  # total MH steps per block (tree depth shrinks with remaining steps)
    max_batch_size: int = 4,  # B_\max: maximum proposals per batched tree traversal
    rank_fn: str = "accept_first",  # RANK_FNS mode ranking the proposal subtree frontier
    max_new_tokens: int = 1024,  # total new tokens across all blocks
    num_of_blocks: int = 16,  # number of autoregressive blocks
    stop_on_eos: bool = False,  # whether to stop generation upon encountering EOS
    print_tree: bool = False,  # print each prefetched proposal tree, in the vLLM sampler's format
    rng: np.random.Generator = None,  # random number generator
    verbose: bool = False,  # print debug information
    evict_subtree_cache: bool = False,  # prune discarded prefixes after scoring and MH
) -> tuple[list[int], SamplingStats]:
    """Sample from the target distribution p^alpha using subtree-prefetching MH.

    For each autoregressive block:
      1. Extend the current sequence with a low-temperature proposal.
      2. Build a binary proposal subtree and batch-generate resampling proposals.
      3. Walk the tree to determine accept/reject outcomes.
      4. Update the sequence and log-probability caches accordingly.
    """
    prompt_len = len(prompt)
    if rng is None:
        rng = np.random.default_rng()

    sequence = prompt.copy() if prompt is not None else []
    proposal_logprobs: List[float] = []
    target_logprobs: List[float] = []

    assert max_new_tokens % num_of_blocks == 0
    block_size = max_new_tokens // num_of_blocks

    stats = SamplingStats()

    for bi in range(num_of_blocks):
        block_start = time.perf_counter()
        logger.info("block [%d/%d]", bi, num_of_blocks)
        # Step 1: Extend the sequence by one block via low-temperature proposal

        extend_start = time.perf_counter()
        context_len = len(sequence)
        sequence, block_proposal_lp, block_target_lp, _ = low_temp_proposal_sampling(
            sampler_wrapper,
            context=sequence,
            seq_len=block_size + len(sequence),
            verbose=verbose,
            # Fixed-length states, as in the vLLM sampler: an EOS-shortened proposal would change the state space and
            # invalidate the MH ratio. EOS is handled after the block.
            ignore_eos=True,
        )
        logger.info(
            "  block extend took %.3f sec: %d prompts, context_len=%s, "
            "new_suffix_len=%s",
            time.perf_counter() - extend_start,
            1,
            [context_len],
            [len(sequence) - context_len],
        )
        if verbose:
            logger.debug(
                "sequence length: %d, block proposal logprobs: %d, block target logprobs: %d",
                len(sequence), len(block_proposal_lp), len(block_target_lp),
            )

        proposal_logprobs.extend(block_proposal_lp)
        target_logprobs.extend(block_target_lp)

        trajectory = []

        # Step 2: MCMC refinement via subtree-prefetching MH

        stop = False
        while len(trajectory) < mcmc_steps:
            step_index = len(trajectory)
            step_start = time.perf_counter()

            # target_logprobs hold alpha * log p; the printout reports base-model log p, as the vLLM sampler does.
            root_log_p = [score / sampler_wrapper.alpha for score in target_logprobs] if print_tree else None
            if print_tree:
                print_root_sequence_scores(root_log_p)
            # 2a: Collect a batch of proposals from a fresh subtree sized to remaining steps
            seq_len = len(sequence)
            parent_to_child, node_to_batch_idx, batch_cuts = collect_batch_cut_indicesv3(
                max_batch_size, prompt_len, seq_len, mcmc_steps - len(trajectory),
                rank=rank_fn, rng=rng,
            )
            if verbose:
                logger.debug("parent_to_child: %s", parent_to_child)
                logger.debug("node_to_batch_idx: %s", node_to_batch_idx)
                logger.debug("batch_cuts: %s", batch_cuts)

            # 2b: Generate proposals and compute acceptance ratios
            proposals, log_accept_ratios, cached_proposal_lp, cached_target_lp = (
                batched_proposal_callv2(
                    sampler_wrapper,
                    parent_proposal_seq=sequence,
                    batched_cut_indexes=batch_cuts,
                    prompt_len=prompt_len,
                    proposal_logprobs_seq=proposal_logprobs,
                    target_log_scores_seq=target_logprobs,
                    verbose=verbose,
                )
            )
            if verbose:
                logger.debug("num proposals: %d", len(proposals))
                logger.debug("accept_ratios: %s", np.exp(log_accept_ratios))

            stats.batch_sizes.append(len(proposals))

            # 2c: Walk the subtree to determine accept/reject outcomes
            # The tree only stores proposals, so walk it in response coordinates (prompt stripped, cuts shifted) like
            # the vLLM sampler; a printed tree's cut_idx and seq_len then index the root log_p printed above it.
            previous_seq = sequence
            response_cuts = [cut - prompt_len for cut in batch_cuts]
            leaf, last_accepted, path, num_accepted = build_and_sample_subtree(
                subtree_root_proposal=sequence[prompt_len:],
                sampled_proposals=[proposal[prompt_len:] for proposal in proposals],
                log_acceptance_ratios=log_accept_ratios,
                node_to_batch_idx=node_to_batch_idx,
                parent_to_accept_child=parent_to_child,
                batch_cut_indices=response_cuts,
                rng=rng,
                print_tree=print_tree,
                verbose=verbose,
            )
            if verbose:
                logger.debug(
                    "leaf: %s, trajectory: %s, accepted: %d",
                    leaf, [s[0] for s in path], num_accepted,
                )

            if print_tree:
                print_node_features(
                    root_log_p, None, node_to_batch_idx, response_cuts, log_accept_ratios, path,
                )

            trajectory.extend(path)
            stats.acceptances.append(num_accepted)
            stats.walked_steps.append(len(path))

            # 2d: Update sequence and log-prob caches on acceptance
            sequence = list(prompt) + list(leaf.proposal)
            if num_accepted > 0:
                batch_idx = node_to_batch_idx[last_accepted.node_id]
                accepted_cut = batch_cuts[batch_idx]
                offset = accepted_cut - prompt_len
                proposal_logprobs[offset:] = cached_proposal_lp[batch_idx].copy()
                target_logprobs[offset:] = cached_target_lp[batch_idx].copy()

            # EOS handling: truncate sequence and caches, update seq_len
            eos_id = sampler_wrapper.tokenizer.eos_token_id
            if eos_id is not None and eos_id in sequence:
                eos_pos = sequence.index(eos_id)
                sequence = sequence[: eos_pos + 1]

                completion_len = eos_pos + 1 - prompt_len
                proposal_logprobs = proposal_logprobs[:completion_len]
                target_logprobs = target_logprobs[:completion_len]
                if stop_on_eos:
                    stop = True
            if evict_subtree_cache:
                _evict_subtree_cache(sampler_wrapper, stats, [previous_seq, *proposals], sequence)

            # Same wording as the vLLM sampler: the case-study parser counts "MH step <i>/<n> took <s> seconds" as one
            # batched target-model call.
            logger.info(
                "  MH step %d/%d took %.3f seconds",
                step_index,
                mcmc_steps,
                time.perf_counter() - step_start,
            )
            if stop:
                break

        # Record the block duration even when EOS ends generation.
        logger.info(
            "block [%d/%d] took %.3f seconds",
            bi,
            num_of_blocks,
            time.perf_counter() - block_start,
        )
        if stop:
            logger.info("stop on eos after block %d", bi)
            break

    return sequence, stats
