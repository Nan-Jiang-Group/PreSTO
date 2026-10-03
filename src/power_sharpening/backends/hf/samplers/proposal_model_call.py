"""HuggingFace proposal calls for subtree-prefetching MH.

Run repository scripts with:
    PYTHONPATH=/Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src python <script>
"""

import logging
import math
from typing import List

import numpy as np

from power_sharpening.common.cut_distribution import (
    compute_entropy_cut_policy,
    entropy_cut_log_ratio,
)
from power_sharpening.backends.hf.wrapper import HF_LLM_Wrapper
from power_sharpening.backends.hf.samplers.low_temp_sampler import batched_low_temp_proposal_sampling
from power_sharpening.backends.hf.samplers.low_temp_sampler import low_temp_sampling

logger = logging.getLogger("[HF proposal model call]")


def _finite_score_sum(
    values: List[float],
    *,
    score_name: str,
    proposal_index: int,
    cut_idx: int,
) -> float:
    """Return an accurate score sum or fail before an indeterminate MH ratio."""
    invalid = [
        (index, float(value))
        for index, value in enumerate(values)
        if not math.isfinite(float(value))
    ]
    if invalid:
        raise ValueError(
            f"proposal {proposal_index} at cut {cut_idx} has non-finite "
            f"{score_name} entries {invalid}"
        )
    return math.fsum(float(value) for value in values)


def batched_proposal_callv1(
        sampler_wrapper: HF_LLM_Wrapper,
        parent_proposal_seq: List[int],
        batched_cut_indexes: List[int],
        prompt_len: int,
        proposal_logprobs_seq: List[float],
        target_log_scores_seq: List[float],
        verbose=False,
):
    """Generate proposals and compute MH acceptance ratios for a batch of cut positions."""
    if verbose:
        logger.debug("batched proposal LLM...")
    max_seq_len = len(parent_proposal_seq)

    resampled_children_proposals: List[List[int]] = []
    log_acceptance_ratios: List[float] = []
    proposal_logprobs_cache: List[List[float]] = []
    target_logprobs_cache: List[List[float]] = []

    for cut_idx in batched_cut_indexes:
        if verbose:
            logger.debug("len context %d, max_seq_len %d", len(parent_proposal_seq[:cut_idx]), max_seq_len)
            assert max_seq_len > len(parent_proposal_seq[:cut_idx])
        prop, suffix_proposal_logprob_x_prime, suffix_target_log_score_x_prime = (
            low_temp_sampling(
                sampler_wrapper,
                context=parent_proposal_seq[:cut_idx],
                seq_len=max_seq_len,
                verbose=verbose,
            )
        )

        proposal_len = len(prop)
        if verbose:
            logger.debug("cut idx %d", cut_idx)
            logger.debug("init proposal, prop_len %d", proposal_len)
            logger.debug("proposal probability LOG P_{low-temp}(x') %d", len(suffix_proposal_logprob_x_prime))
            logger.debug("target score  P_0(x')^alpha %d", len(suffix_target_log_score_x_prime))
        assert len(suffix_proposal_logprob_x_prime) == proposal_len - cut_idx
        assert len(suffix_target_log_score_x_prime) == proposal_len - cut_idx

        proposal_logprob_x = proposal_logprobs_seq.copy()[cut_idx - prompt_len:]
        target_log_score_x = target_log_scores_seq.copy()[cut_idx - prompt_len:]

        proposal_index = len(log_acceptance_ratios)
        part1 = _finite_score_sum(
            suffix_target_log_score_x_prime,
            score_name="proposed target score",
            proposal_index=proposal_index,
            cut_idx=cut_idx,
        ) - _finite_score_sum(
            target_log_score_x,
            score_name="current target score",
            proposal_index=proposal_index,
            cut_idx=cut_idx,
        )
        part2 = _finite_score_sum(
            proposal_logprob_x,
            score_name="current proposal log-probability",
            proposal_index=proposal_index,
            cut_idx=cut_idx,
        ) - _finite_score_sum(
            suffix_proposal_logprob_x_prime,
            score_name="proposed proposal log-probability",
            proposal_index=proposal_index,
            cut_idx=cut_idx,
        )

        log_acceptance_ratio = part1 + part2
        if verbose:
            logger.debug("log_acceptance_ratio %s", log_acceptance_ratio)

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
        sampler_wrapper: HF_LLM_Wrapper,
        parent_proposal_seq: List[int],
        batched_cut_indexes: List[int],
        prompt_len: int,
        proposal_logprobs_seq: List[float],
        target_log_scores_seq: List[float],
        verbose=False,
        ignore_eos: bool = False,
        cut_entropies: List[float] = None,
        cut_power: float = None,
):
    """Generate proposals in one HuggingFace batch and compute MH acceptance ratios.

    When ``cut_entropies`` is supplied the cuts came from a state-dependent (EntropyCut) law, so each acceptance ratio
    gains the ``entropy_cut_log_ratio`` correction term that keeps the chain unbiased. ``cut_entropies`` holds one
    base-model predictive entropy per position of ``parent_proposal_seq`` from ``prompt_len`` onward.
    """
    if verbose:
        logger.debug("batched proposal LLM...")
    max_seq_len = len(parent_proposal_seq)

    use_entropy_cut = cut_entropies is not None
    if use_entropy_cut:
        expected_entropies = max_seq_len - prompt_len
        if len(cut_entropies) != expected_entropies:
            raise ValueError(
                "cut_entropies must cover every position from prompt_len "
                f"onward: expected {expected_entropies}, got {len(cut_entropies)}"
            )
        if cut_power is None:
            raise ValueError("cut_power is required when cut_entropies is given")

    contexts = [parent_proposal_seq[:cut_idx] for cut_idx in batched_cut_indexes]
    sampled = batched_low_temp_proposal_sampling(
        sampler_wrapper,
        contexts=contexts,
        seq_len=max_seq_len,
        verbose=verbose,
        ignore_eos=ignore_eos,
        return_entropies=use_entropy_cut,
    )
    if use_entropy_cut:
        batch_proposals, batch_proposal_lps, batch_target_lss, batch_entropies = sampled
        # The forward policy is shared by every row: all cuts were drawn from this same current-state law.
        current_cut_probabilities = compute_entropy_cut_policy(
            cut_entropies,
            beta=cut_power,
        )
    else:
        batch_proposals, batch_proposal_lps, batch_target_lss = sampled
        batch_entropies = None
        current_cut_probabilities = None

    log_acceptance_ratios: List[float] = []
    proposal_logprobs_cache: List[List[float]] = []
    target_logprobs_cache: List[List[float]] = []
    entropies_cache: List[List[float]] = []

    for i, cut_idx in enumerate(batched_cut_indexes):
        suffix_proposal_lp = batch_proposal_lps[i]
        suffix_target_ls = batch_target_lss[i]
        proposal_len = len(batch_proposals[i])

        assert len(suffix_proposal_lp) == proposal_len - cut_idx
        assert len(suffix_target_ls) == proposal_len - cut_idx

        offset = cut_idx - prompt_len
        proposal_lp_x = proposal_logprobs_seq[offset:]
        target_ls_x = target_log_scores_seq[offset:]

        proposed_target = _finite_score_sum(
            suffix_target_ls,
            score_name="proposed target score",
            proposal_index=i,
            cut_idx=cut_idx,
        )
        current_target = _finite_score_sum(
            target_ls_x,
            score_name="current target score",
            proposal_index=i,
            cut_idx=cut_idx,
        )
        current_proposal = _finite_score_sum(
            proposal_lp_x,
            score_name="current proposal log-probability",
            proposal_index=i,
            cut_idx=cut_idx,
        )
        proposed_proposal = _finite_score_sum(
            suffix_proposal_lp,
            score_name="proposed proposal log-probability",
            proposal_index=i,
            cut_idx=cut_idx,
        )
        # A state-dependent cut law does not cancel between the forward and reverse moves, so its ratio must enter the
        # acceptance probability or the chain is biased. State-independent laws contribute exactly 0.
        cut_term = 0.0
        if current_cut_probabilities is not None:
            proposed_entropies = (
                list(cut_entropies[:offset]) + list(batch_entropies[i])
            )
            proposed_cut_probabilities = compute_entropy_cut_policy(
                proposed_entropies,
                beta=cut_power,
            )
            cut_term = float(
                entropy_cut_log_ratio(
                    current_cut_probabilities,
                    proposed_cut_probabilities,
                    cut_offset=offset,
                )
            )

        log_acc = (
            proposed_target
            - current_target
            + current_proposal
            - proposed_proposal
            + cut_term
        )
        if math.isnan(log_acc) or log_acc == math.inf:
            raise ValueError(
                f"proposal {i} at cut {cut_idx} has a non-finite MH "
                "log acceptance ratio despite finite per-token scores: "
                f"proposed_target={proposed_target}, "
                f"current_target={current_target}, "
                f"current_proposal={current_proposal}, "
                f"proposed_proposal={proposed_proposal}, "
                f"cut_term={cut_term}"
            )

        if verbose:
            logger.debug("  proposal[%d]: cut_idx=%d, acc_ratio=%.4f", i, cut_idx, np.exp(log_acc))

        log_acceptance_ratios.append(log_acc)
        proposal_logprobs_cache.append(list(suffix_proposal_lp))
        target_logprobs_cache.append(list(suffix_target_ls))
        entropies_cache.append(
            list(batch_entropies[i]) if use_entropy_cut else None
        )

    return (
        batch_proposals,
        log_acceptance_ratios,
        proposal_logprobs_cache,
        target_logprobs_cache,
        entropies_cache,
    )
