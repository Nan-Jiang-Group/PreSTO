import logging
import time
from typing import Any, List
import numpy as np

from .low_temp_sampler import low_temp_sampling as low_temp_proposal_sampling
from power_sharpening.backends.hf.compute_log_likelihood_and_confidence import (
    compute_log_likelihood_and_confidence,
)
from power_sharpening.common.prefetch_subtree import (
    build_and_sample_subtree,
    collect_batch_cut_indicesv3,
)
from power_sharpening.common.sample_stats import SamplingStats
from power_sharpening.backends.hf.samplers.proposal_model_call import batched_proposal_callv2

logger = logging.getLogger("[HF subtree prefetching MH]")


def subtree_prefetching_sampling(
    sampler_wrapper,  # wraps the base LLM, proposal, and target distributions
    prompt: List[int],  # fixed prefix tokens (never resampled)
    mcmc_steps: int = 10,  # total MH steps per block (tree depth shrinks with remaining steps)
    max_batch_size: int = 4,  # B_\max: maximum proposals per batched tree traversal
    max_new_tokens: int = 1024,  # total new tokens across all blocks
    num_of_blocks: int = 16,  # number of autoregressive blocks
    rank_fn: str = "longest_path_first",  # RANK_FNS mode ranking the proposal subtree frontier
    cut_dist_type: str = "uniform",  # uniform or entropy cut-index law
    cut_dist_param: float = None,  # shape parameter for cut_dist_type (None = family default)
    print_tree: bool = False,  # print each prefetched proposal tree
    rng: Any = None,  # random number generator
    verbose: bool = False,  # print debug information
) -> tuple[list[int], SamplingStats]:
    """Sample from the target distribution p^alpha using subtree-prefetching MH.

    For each autoregressive block:
      1. Extend the current sequence with a low-temperature proposal.
      2. Build a binary proposal subtree and batch-generate resampling proposals.
      3. Walk the tree to determine accept/reject outcomes.
      4. Update the sequence and log-probability caches accordingly.
    """
    prompt_len = len(prompt)
    eos_id = sampler_wrapper.tokenizer.eos_token_id
    if rng is None:
        rng = np.random.default_rng()

    generated_output_seq = prompt.copy() if prompt is not None else []
    proposal_logprobs: List[float] = []
    target_logprobs: List[float] = []
    # EntropyCut needs a per-token predictive-entropy cache kept in lockstep with the log-prob caches; other cut laws
    # never touch it.
    use_entropy_cut = cut_dist_type == "entropy"
    entropies: List[float] = []
    if use_entropy_cut and cut_dist_param is None:
        cut_dist_param = 4

    assert max_new_tokens % num_of_blocks == 0
    block_size = max_new_tokens // num_of_blocks

    stats = SamplingStats()

    for mi in range(num_of_blocks):
        logger.info("block [%d/%d]", mi, num_of_blocks)
        # Step 1: Extend the sequence by one block via low-temperature proposal

        block_sampled = low_temp_proposal_sampling(
            sampler_wrapper,
            context=generated_output_seq,
            seq_len=block_size + len(generated_output_seq),
            verbose=verbose,
            ignore_eos=True,
            return_entropies=use_entropy_cut,
        )
        if use_entropy_cut:
            (
                generated_output_seq,
                block_proposal_logprobs,
                block_target_logprobs,
                block_entropies,
            ) = block_sampled
            entropies.extend(block_entropies)
        else:
            (
                generated_output_seq,
                block_proposal_logprobs,
                block_target_logprobs,
            ) = block_sampled
        logger.debug(
            "sequence length: %d, block proposal logprobs: %d, block target logprobs: %d",
            len(generated_output_seq), len(block_proposal_logprobs), len(block_target_logprobs),
        )

        proposal_logprobs.extend(block_proposal_logprobs)
        target_logprobs.extend(block_target_logprobs)

        trajectory = []

        # Step 2: MCMC refinement via subtree-prefetching MH

        while len(trajectory) < mcmc_steps:
            step_index = len(trajectory)
            step_start = time.perf_counter()
            # 2a: Collect a batch of proposals from a fresh subtree sized to remaining steps
            seq_len = len(generated_output_seq)
            parent_to_child, node_to_batch_idx, batch_cuts = (
                collect_batch_cut_indicesv3(
                    max_batch_size,
                    prompt_len,
                    seq_len,
                    mcmc_steps - len(trajectory),
                    rank=rank_fn,
                    rng=rng,
                    cut_dist_type=cut_dist_type,
                    cut_dist_param=cut_dist_param,
                    cut_entropies=entropies if use_entropy_cut else None,
                )
            )
            logger.debug("parent_to_child: %s", parent_to_child)
            logger.debug("node_to_batch_idx: %s", node_to_batch_idx)
            logger.debug("batch_cuts: %s", batch_cuts)

            # 2b: Generate proposals and compute acceptance ratios
            proposals, log_accept_ratios, cached_proposal_lp, cached_target_lp, cached_entropies = (
                batched_proposal_callv2(
                    sampler_wrapper,
                    parent_proposal_seq=generated_output_seq,
                    batched_cut_indexes=batch_cuts,
                    prompt_len=prompt_len,
                    proposal_logprobs_seq=proposal_logprobs,
                    target_log_scores_seq=target_logprobs,
                    verbose=verbose,
                    ignore_eos=True,
                    cut_entropies=entropies if use_entropy_cut else None,
                    cut_power=cut_dist_param if use_entropy_cut else None,
                )
            )

            logger.debug("num proposals: %d", len(proposals))
            logger.debug("accept_ratios: %s", np.exp(log_accept_ratios))

            stats.batch_sizes.append(len(proposals))

            # 2c: Walk the subtree to determine accept/reject outcomes
            leaf, last_accepted, path, num_accepted = build_and_sample_subtree(
                subtree_root_proposal=generated_output_seq,
                sampled_proposals=proposals,
                log_acceptance_ratios=log_accept_ratios,
                node_to_batch_idx=node_to_batch_idx,
                parent_to_accept_child=parent_to_child,
                batch_cut_indices=batch_cuts,
                rng=rng,
                print_tree=print_tree,
                verbose=verbose,
            )
            logger.debug(
                    "leaf: %s, trajectory: %s, accepted: %d",
                    leaf, [s[0] for s in path], num_accepted,
                )

            trajectory.extend(path)
            stats.acceptances.append(num_accepted)
            stats.walked_steps.append(len(path))

            # 2d: Update sequence and log-prob caches on acceptance
            generated_output_seq = leaf.proposal
            if num_accepted > 0:
                batch_idx = node_to_batch_idx[last_accepted.node_id]
                accepted_cut = batch_cuts[batch_idx]
                offset = accepted_cut - prompt_len
                proposal_logprobs[offset:] = cached_proposal_lp[batch_idx].copy()
                target_logprobs[offset:] = cached_target_lp[batch_idx].copy()
                if use_entropy_cut:
                    entropies[offset:] = cached_entropies[batch_idx].copy()

            logger.info(
                "  MH step %d/%d took %.3f seconds",
                step_index,
                mcmc_steps,
                time.perf_counter() - step_start,
            )

            # EOS handling: truncate sequence and caches, update seq_len

        if eos_id in generated_output_seq:
            eos_idx = generated_output_seq.index(eos_id)
            generated_output_seq = generated_output_seq[: eos_idx + 1]

            completion_len = eos_idx + 1 - prompt_len
            proposal_logprobs = proposal_logprobs[:completion_len]
            target_logprobs = target_logprobs[:completion_len]
            entropies = entropies[:completion_len]
            break

    expected_score_count = len(generated_output_seq) - prompt_len
    if len(target_logprobs) != expected_score_count:
        raise RuntimeError(
            "final target-log-score cache is misaligned with the generated "
            f"response: expected {expected_score_count}, got {len(target_logprobs)}"
        )

    # The cached scores are alpha * log p_0(x_t | x_{<t}) and carry no entropies,
    # so the terminal state is rescored under the base model p_0 to obtain the length-normalized log-likelihood and
    # confidence.
    stats.base_diagnostics = compute_log_likelihood_and_confidence(
        sampler_wrapper,
        generated_output_seq,
        prompt_len,
    )

    return generated_output_seq, stats
