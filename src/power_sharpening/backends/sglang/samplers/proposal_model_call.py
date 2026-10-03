"""SGLang proposal calls for subtree-prefetching MH.

Run repository scripts with:
    PYTHONPATH=/Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src python <script>
"""

from typing import List

import numpy as np

from power_sharpening.backends.sglang.wrapper import SGL_LLM_Wrapper
from power_sharpening.backends.sglang.samplers.low_temp_proposal_sampler import (
    batched_low_temp_proposal_sampling,
    low_temp_proposal_sampling,
)


def batched_proposal_callv1(
        sampler_wrapper: SGL_LLM_Wrapper,
        parent_proposal_seq: List[int],
        batched_cut_indexes: List[int],
        prompt_len: int,
        proposal_logprobs_seq: List[float],
        target_log_scores_seq: List[float],
        verbose=False,
):
    """Generate proposals and compute MH acceptance ratios for a batch of cut positions."""
    if verbose:
        print("batched proposal LLM...")
    max_seq_len = len(parent_proposal_seq)

    resampled_children_proposals: List[List[int]] = []
    log_acceptance_ratios: List[float] = []
    proposal_logprobs_cache: List[List[float]] = []
    target_logprobs_cache: List[List[float]] = []

    for cut_idx in batched_cut_indexes:
        if verbose:
            print("len context", len(parent_proposal_seq[:cut_idx]), max_seq_len)
            assert max_seq_len > len(parent_proposal_seq[:cut_idx])
        prop, suffix_proposal_logprob_x_prime, suffix_target_log_score_x_prime, _ = (
            low_temp_proposal_sampling(
                sampler_wrapper,
                context=parent_proposal_seq[:cut_idx],
                seq_len=max_seq_len,
                verbose=verbose,
            )
        )

        proposal_len = len(prop)
        if verbose:
            print("cut idx", cut_idx)
            print("init proposal, prop_len", proposal_len)
            print("proposal probability LOG P_{low-temp}(x')", len(suffix_proposal_logprob_x_prime))
            print("target score  P_0(x')^alpha", len(suffix_target_log_score_x_prime))
        assert len(suffix_proposal_logprob_x_prime) == proposal_len - cut_idx
        assert len(suffix_target_log_score_x_prime) == proposal_len - cut_idx

        proposal_logprob_x = proposal_logprobs_seq.copy()[cut_idx - prompt_len:]
        target_log_score_x = target_log_scores_seq.copy()[cut_idx - prompt_len:]

        part1 = sum(suffix_target_log_score_x_prime) - sum(target_log_score_x)
        part2 = sum(proposal_logprob_x) - sum(suffix_proposal_logprob_x_prime)

        log_acceptance_ratio = part1 + part2
        if verbose:
            print("log_acceptance_ratio", log_acceptance_ratio)

        resampled_children_proposals.append(prop)
        log_acceptance_ratios.append(log_acceptance_ratio)
        proposal_logprobs_cache.append(list(suffix_proposal_logprob_x_prime))
        target_logprobs_cache.append(list(suffix_target_log_score_x_prime))

    return (
        resampled_children_proposals,
        log_acceptance_ratios,
        proposal_logprobs_cache,
        target_logprobs_cache,
    )


def batched_proposal_callv2(
        sampler_wrapper: SGL_LLM_Wrapper,
        parent_proposal_seq: List[int],
        batched_cut_indexes: List[int],
        prompt_len: int,
        proposal_logprobs_seq: List[float],
        target_log_scores_seq: List[float],
        verbose=False,
):
    """Generate proposals in one SGLang batch and compute MH acceptance ratios."""
    if verbose:
        print("batched proposal LLM...")
    max_seq_len = len(parent_proposal_seq)

    contexts = [parent_proposal_seq[:cut_idx] for cut_idx in batched_cut_indexes]
    batch_proposals, batch_proposal_lps, batch_target_lss = batched_low_temp_proposal_sampling(
        sampler_wrapper,
        contexts=contexts,
        seq_len=max_seq_len,
        verbose=verbose,
    )

    log_acceptance_ratios: List[float] = []
    proposal_logprobs_cache: List[List[float]] = []
    target_logprobs_cache: List[List[float]] = []

    for i, cut_idx in enumerate(batched_cut_indexes):
        suffix_proposal_lp = batch_proposal_lps[i]
        suffix_target_ls = batch_target_lss[i]
        proposal_len = len(batch_proposals[i])

        assert len(suffix_proposal_lp) == proposal_len - cut_idx
        assert len(suffix_target_ls) == proposal_len - cut_idx

        offset = cut_idx - prompt_len
        proposal_lp_x = proposal_logprobs_seq[offset:]
        target_ls_x = target_log_scores_seq[offset:]

        log_acc = (
            sum(suffix_target_ls)
            - sum(target_ls_x)
            + sum(proposal_lp_x)
            - sum(suffix_proposal_lp)
        )

        if verbose:
            print(f"  proposal[{i}]: cut_idx={cut_idx}, acc_ratio={np.exp(log_acc):.4f}")

        log_acceptance_ratios.append(log_acc)
        proposal_logprobs_cache.append(list(suffix_proposal_lp))
        target_logprobs_cache.append(list(suffix_target_ls))

    return (
        batch_proposals,
        log_acceptance_ratios,
        proposal_logprobs_cache,
        target_logprobs_cache,
    )
