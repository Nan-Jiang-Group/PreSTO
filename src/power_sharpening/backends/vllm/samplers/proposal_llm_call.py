"""Generate suffix proposals for subtree-prefetching Metropolis-Hastings with vLLM.

The target is proportional to p(x)^alpha. Each custom vLLM completion supplies two sets of token log-probabilities:

* ``out.logprobs``: log q(x_t | x_{<t}) under the proposal temperature.
* ``out.power_logprobs``: base-model log p(x_t | x_{<t}), before multiplication by alpha.

EntropyCut uses full-vocabulary base-model entropies computed by the engine alongside those log-probabilities.
Prefixes and cuts use token IDs throughout, so proposals retain the exact prefix without re-encoding text.

Run through ``python -m power_sharpening.runners.vllm.run_subtree_prefetching_mh`` on a GPU.
"""

import logging
import time
from typing import List

from power_sharpening.backends.vllm.samplers.power_sampling_mh import compute_acceptance_ratio, _extract_logprobs
from power_sharpening.backends.vllm.engine_patch.sampling_params import _copy_sampling_params
from power_sharpening.common.cut_distribution import (
    compute_entropy_cut_policy,
    entropy_cut_log_ratio,
)

logger = logging.getLogger("[vLLM proposal call]")


def _tokens_prompt(token_ids: List[int]) -> dict:
    """Pass ``token_ids`` to vLLM directly, without tokenization or added special tokens."""
    return {"prompt_token_ids": list(token_ids)}


def _suffix_entropies(output, expected_length: int) -> List[float]:
    """Read engine-computed entropies aligned with ``expected_length`` suffix tokens."""
    if output.entropies is None or len(output.entropies) != expected_length:
        raise RuntimeError("custom vLLM output must provide one entropy per suffix token")
    return list(output.entropies)


def vllm_proposal_sampling(
    mh_llm_model,
    context_ids: List[int],
    max_new_tokens: int,
    sampling_params,
    return_entropies: bool = False,
    verbose: bool = False,
):
    """Extend ``context_ids`` by up to ``max_new_tokens`` via one vLLM call.

    Args:
        mh_llm_model: wrapper providing the custom vLLM engine.
        context_ids: full prefix token IDs, including the prompt and any generated tokens.
        max_new_tokens: maximum number of suffix tokens to sample.
        sampling_params: generation settings, including proposal temperature and target alpha.
        return_entropies: append the engine-computed entropy for each new token.
        verbose: retained for caller compatibility; logging is controlled by the logger level.

    Returns:
        new_suffix: newly sampled suffix token IDs.
        suffix_proposal_lp: per-token log q(x_t | x_{<t}) for the new suffix.
        suffix_power_lp: per-token base-model log p(x_t | x_{<t}), without the alpha factor.
        suffix_entropies: per-token base-model entropies, returned only when ``return_entropies`` is true.
    """
    assert max_new_tokens > 0, "new suffixes must be non-negative"
    # Entropies are returned alongside token scores by the custom engine.
    sp = _copy_sampling_params(
        sampling_params,
        max_tokens=max_new_tokens,
        logprobs=1,
    )

    outputs = mh_llm_model.llm.generate(
        [_tokens_prompt(context_ids)], sampling_params=sp, use_tqdm=False
    )
    out = outputs[0].outputs[0]

    new_suffix = list(out.token_ids)
    suffix_proposal_lp = _extract_logprobs(out.logprobs)
    suffix_power_lp = _extract_logprobs(out.power_logprobs)
    suffix_entropies = (
        _suffix_entropies(out, len(new_suffix)) if return_entropies else None
    )

    logger.info(
        "vllm_proposal_sampling: context_len=%d, new_suffix_len=%d",
        len(context_ids), len(new_suffix)
    )
    if return_entropies:
        return (
            new_suffix,
            suffix_proposal_lp,
            suffix_power_lp,
            suffix_entropies,
        )
    return new_suffix, suffix_proposal_lp, suffix_power_lp


def batched_proposal_call(
    mh_llm_model,
    parent_proposal_seq: List[int],
    batched_cut_indexes: List[int],
    proposal_logprobs_seq: List[float],
    power_logprobs_seq: List[float],
    sampling_params,
    prompt_ids: List[int],
    verbose: bool = False,
):
    """Regenerate one suffix per uniform cut and return the standard MH log ratios.

    For current suffix x and proposed suffix x', both conditioned on the retained prefix:

        log A = alpha * [log p(x') - log p(x)] + log q(x) - log q(x')

    The uniform cut probabilities cancel. Returned log ratios are unclipped; acceptance uses min(1, exp(log A)).

    Args:
        mh_llm_model: wrapper providing the custom vLLM engine.
        parent_proposal_seq: current generated token sequence, excluding the prompt.
        batched_cut_indexes: first token to regenerate in ``parent_proposal_seq`` for each proposal.
        proposal_logprobs_seq: per-token log q for the current generated sequence.
        power_logprobs_seq: per-token base-model log p for the current generated sequence, without alpha.
        sampling_params: generation settings, including proposal temperature and target alpha.
        prompt_ids: fixed prompt token IDs prepended to each retained prefix.
        verbose: log batch size and cut positions at the debug level.

    Returns:
        resampled_children_proposals: new generated sequences, each containing the retained prefix and new suffix.
        log_acceptance_ratios: unclipped MH log ratio per proposal.
        proposal_logprobs_cache: per-token log q for each new suffix.
        power_logprobs_cache: per-token base-model log p for each new suffix, without alpha.
    """
    alpha = sampling_params.alpha
    gen_len = len(parent_proposal_seq)

    # Each request keeps the prompt and tokens before its cut, then regenerates the remaining suffix.
    build_start = time.perf_counter()
    token_prompts = []
    per_request_sp = []
    for cut_idx in batched_cut_indexes:
        context_ids = prompt_ids + parent_proposal_seq[:cut_idx]
        token_prompts.append(_tokens_prompt(context_ids))
        per_request_sp.append(
            _copy_sampling_params(
                sampling_params,
                max_tokens=gen_len - cut_idx,
                logprobs=1,
            )
        )
    build_seconds = time.perf_counter() - build_start

    if verbose:
        logger.debug(
            "batched vLLM proposal: %d requests, cuts=%s",
            len(token_prompts), batched_cut_indexes,
        )

    generate_start = time.perf_counter()
    outputs = mh_llm_model.llm.generate(
        token_prompts, sampling_params=per_request_sp, use_tqdm=False
    )
    generate_seconds = time.perf_counter() - generate_start

    score_start = time.perf_counter()
    resampled_children_proposals: List[List[int]] = []
    log_acceptance_ratios: List[float] = []
    proposal_logprobs_cache: List[List[float]] = []
    power_logprobs_cache: List[List[float]] = []

    for i, cut_idx in enumerate(batched_cut_indexes):
        out = outputs[i].outputs[0]
        new_suffix = list(out.token_ids)
        suffix_proposal_lp = _extract_logprobs(out.logprobs)
        suffix_power_lp = _extract_logprobs(out.power_logprobs)

        assert len(suffix_proposal_lp) == len(new_suffix)
        assert len(suffix_power_lp) == len(new_suffix)

        # Compare suffix scores under the same retained prefix.
        log_acc = compute_acceptance_ratio(
            proposed_logprobs=suffix_proposal_lp,
            curr_logprobs=proposal_logprobs_seq[cut_idx:],
            proposed_power_logprobs=suffix_power_lp,
            curr_power_logprobs=power_logprobs_seq[cut_idx:],
            alpha=alpha,
        )
        resampled_children_proposals.append(parent_proposal_seq[:cut_idx] + new_suffix)
        log_acceptance_ratios.append(log_acc)
        proposal_logprobs_cache.append(suffix_proposal_lp)
        power_logprobs_cache.append(suffix_power_lp)
    score_seconds = time.perf_counter() - score_start

    # Keep this timing breakdown distinct from the "MH step ... took" lines counted as model calls.
    logger.info(
        "      proposal call: %d requests, build %.3f, generate %.3f, "
        "score %.3f seconds",
        len(token_prompts),
        build_seconds,
        generate_seconds,
        score_seconds,
    )

    return (
        resampled_children_proposals,
        log_acceptance_ratios,
        proposal_logprobs_cache,
        power_logprobs_cache,
    )


def batched_proposal_call_with_entropy_cut(
    mh_llm_model,
    parent_proposal_seq: List[int],
    batched_cut_indexes: List[int],
    proposal_logprobs_seq: List[float],
    power_logprobs_seq: List[float],
    sampling_params,
    prompt_ids: List[int],
    cut_entropies: List[float],
    cut_power: float,
    verbose: bool = False,
):
    """Regenerate one suffix per entropy cut and return the corrected MH log ratios.

    For current suffix x and proposed suffix x', both conditioned on the retained prefix:

        log A = alpha * [log p(x') - log p(x)] + log q(x) - log q(x')

    Add log lambda(cut; proposed sequence) - log lambda(cut; current sequence) to account for the cut law.
    Returned log ratios are unclipped; acceptance uses min(1, exp(log A)).

    Args:
        mh_llm_model: wrapper providing the custom vLLM engine.
        parent_proposal_seq: current generated token sequence, excluding the prompt.
        batched_cut_indexes: first token to regenerate in ``parent_proposal_seq`` for each proposal.
        proposal_logprobs_seq: per-token log q for the current generated sequence.
        power_logprobs_seq: per-token base-model log p for the current generated sequence, without alpha.
        sampling_params: generation settings, including proposal temperature and target alpha.
        prompt_ids: fixed prompt token IDs prepended to each retained prefix.
        cut_entropies: per-token base-model entropies for the current generated sequence.
        cut_power: exponent used to compute entropy-cut probabilities.
        verbose: log batch size and cut positions at the debug level.

    Returns:
        resampled_children_proposals: new generated sequences, each containing the retained prefix and new suffix.
        log_acceptance_ratios: unclipped MH log ratio per proposal, including the cut-probability correction.
        proposal_logprobs_cache: per-token log q for each new suffix.
        power_logprobs_cache: per-token base-model log p for each new suffix, without alpha.
        entropies_cache: engine-computed entropies for each new suffix.
    """
    alpha = sampling_params.alpha
    gen_len = len(parent_proposal_seq)

    current_cut_probabilities = compute_entropy_cut_policy(
        cut_entropies,
        beta=cut_power,
    )

    # Each request keeps the prompt and tokens before its cut, then regenerates the remaining suffix.
    build_start = time.perf_counter()
    token_prompts = []
    per_request_sp = []
    for cut_idx in batched_cut_indexes:
        context_ids = prompt_ids + parent_proposal_seq[:cut_idx]
        token_prompts.append(_tokens_prompt(context_ids))
        per_request_sp.append(
            _copy_sampling_params(
                sampling_params,
                max_tokens=gen_len - cut_idx,
                logprobs=1,
            )
        )
    build_seconds = time.perf_counter() - build_start

    if verbose:
        logger.debug(
            "batched vLLM proposal: %d requests, cuts=%s",
            len(token_prompts), batched_cut_indexes,
        )

    generate_start = time.perf_counter()
    outputs = mh_llm_model.llm.generate(
        token_prompts, sampling_params=per_request_sp, use_tqdm=False
    )
    generate_seconds = time.perf_counter() - generate_start

    score_start = time.perf_counter()
    resampled_children_proposals: List[List[int]] = []
    log_acceptance_ratios: List[float] = []
    proposal_logprobs_cache: List[List[float]] = []
    power_logprobs_cache: List[List[float]] = []
    entropies_cache: List[List[float]] = []

    for i, cut_idx in enumerate(batched_cut_indexes):
        out = outputs[i].outputs[0]
        new_suffix = list(out.token_ids)
        suffix_proposal_lp = _extract_logprobs(out.logprobs)
        suffix_power_lp = _extract_logprobs(out.power_logprobs)
        suffix_entropies = _suffix_entropies(out, len(new_suffix))

        assert len(suffix_proposal_lp) == len(new_suffix)
        assert len(suffix_power_lp) == len(new_suffix)

        # Preserve the generated prefix and replace only the suffix.
        new_generated = parent_proposal_seq[:cut_idx] + new_suffix

        # Compare suffix scores under the same retained prefix.
        curr_proposal_lp = proposal_logprobs_seq[cut_idx:]
        curr_power_lp = power_logprobs_seq[cut_idx:]

        log_acc = compute_acceptance_ratio(
            proposed_logprobs=suffix_proposal_lp,
            curr_logprobs=curr_proposal_lp,
            proposed_power_logprobs=suffix_power_lp,
            curr_power_logprobs=curr_power_lp,
            alpha=alpha,
        )
        # Correct for the probability of selecting this cut in the proposed versus current sequence.
        proposed_entropies = list(cut_entropies[:cut_idx]) + list(suffix_entropies)
        proposed_cut_probabilities = compute_entropy_cut_policy(
            proposed_entropies,
            beta=cut_power,
        )
        log_acc += entropy_cut_log_ratio(
            current_cut_probabilities,
            proposed_cut_probabilities,
            cut_offset=cut_idx,
        )

        resampled_children_proposals.append(new_generated)
        log_acceptance_ratios.append(log_acc)
        proposal_logprobs_cache.append(suffix_proposal_lp)
        power_logprobs_cache.append(suffix_power_lp)
        entropies_cache.append(suffix_entropies)
    score_seconds = time.perf_counter() - score_start

    # Report request assembly, generation, and scoring separately. The case-study parser counts "MH step ... took"
    # messages as model calls, so this timing breakdown uses a distinct prefix.
    logger.info(
        "      proposal call: %d requests, build %.3f, generate %.3f, "
        "score %.3f seconds",
        len(token_prompts),
        build_seconds,
        generate_seconds,
        score_seconds,
    )

    return (
        resampled_children_proposals,
        log_acceptance_ratios,
        proposal_logprobs_cache,
        power_logprobs_cache,
        entropies_cache,
    )
