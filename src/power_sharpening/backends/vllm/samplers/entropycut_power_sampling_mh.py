"""Entropy-Cut stagewise Metropolis-Hastings sampling on custom vLLM.

Run the focused CPU tests with:
    uv run --project /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src \
      pytest /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/tests/test_vllm_entropycut_power_sampler.py

This implements Algorithm 2 of "Reasoning with Sampling: Cutting at Decision Points" (https://arxiv.org/abs/2605.30327)
against the repository's custom vLLM engine, which returns proposal log-probabilities and base-model log-probabilities.

Per-token predictive entropies are computed from the full base-model vocabulary inside the custom vLLM engine and
returned alongside proposal and base-model log-probabilities. The entropy shapes the MH proposal, and the
state-dependent cut-law correction preserves the ``p^alpha`` target.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any

import numpy as np
from vllm.logprobs import Logprob

from power_sharpening.backends.vllm.engine_patch import SamplingParams
from power_sharpening.backends.vllm.engine_patch.sampling_params import (
    _copy_sampling_params,
)
from power_sharpening.backends.vllm.engine_patch.entropy import suffix_log_acceptance
from power_sharpening.common.cut_distribution import (
    compute_entropy_cut_policy,
    draw_entropy_cut_index,
    entropy_cut_log_ratio,
    summarize_cut_policy,
)


logger = logging.getLogger("[entropy-cut MH]")


def _extract_logprobs(logprobs: list[dict[int, Logprob]]) -> list[float]:
    """Extract the sampled token's log-probability at each decoding step."""
    extracted_logprobs = []
    for step in logprobs:
        logprob_entries = list(step.values())
        sampled_token_logprob = logprob_entries[0]
        extracted_logprobs.append(sampled_token_logprob.logprob)
    return extracted_logprobs


def compute_acceptance_ratio(
    proposed_logprobs: np.ndarray,
    curr_logprobs: np.ndarray,
    proposed_power_logprobs: np.ndarray,
    curr_power_logprobs: np.ndarray,
    alpha: float,
    current_entropies: np.ndarray,
    proposed_suffix_entropies: np.ndarray,
    cut_index: int,
    beta: float,
) -> float:
    """Return Algorithm 2's complete Entropy-Cut MH log ratio.

    The first term is the power-target ratio and the reverse/forward suffix proposal ratio. The second is the
    state-dependent cut-law correction
    ``log lambda(cut | proposed) - log lambda(cut | current)``.
    """
    suffix_ratio = suffix_log_acceptance(
        proposed_logq=proposed_logprobs,
        current_logq=curr_logprobs,
        proposed_logp=proposed_power_logprobs,
        current_logp=curr_power_logprobs,
        alpha=alpha,
    )
    proposed_entropies = (
        list(current_entropies[:cut_index])
        + list(proposed_suffix_entropies)
    )
    current_probabilities = compute_entropy_cut_policy(
        current_entropies,
        beta=beta,
    )
    proposed_probabilities = compute_entropy_cut_policy(
        proposed_entropies,
        beta=beta,
    )
    cut_ratio = entropy_cut_log_ratio(
        current_probabilities,
        proposed_probabilities,
        cut_offset=cut_index - 1,
    )
    return suffix_ratio + cut_ratio


def _unpack_suffix_output(
    output: Any,
    expected_length: int,
):
    """Return one fixed-length suffix's tokens, log-probs, and engine-computed entropies."""
    token_ids = list(output.token_ids)
    proposal_logprobs = _extract_logprobs(output.logprobs)
    power_logprobs = _extract_logprobs(output.power_logprobs)
    if output.entropies is None:
        raise RuntimeError("custom vLLM output did not provide suffix entropies")
    entropies = list(output.entropies)
    lengths = (
        len(token_ids),
        len(proposal_logprobs),
        len(power_logprobs),
        len(entropies),
    )
    if lengths != (expected_length,) * 4:
        raise RuntimeError(
            "EntropyCut proposals must replace the complete fixed-length "
            f"suffix: expected {expected_length}, got tokens/proposal "
            f"logprobs/base logprobs/entropies={lengths}"
        )
    return token_ids, proposal_logprobs, power_logprobs, entropies


def mcmc_power_sampler_entropycut(
    mh_llm_model,
    prompts: str | list[str],
    sampling_params: SamplingParams,
    num_of_blocks: int = 8,
    max_new_tokens: int = 3_072,
    mcmc_steps: int = 10,
    beta: float = 4.0,
    temperature_scheduler=None,
    rng=None,
) -> list[str]:
    """Generate continuations with Entropy-Cut stagewise MH (Algorithm 2).

    Cut position ``m`` is sampled proportionally to the positive predictive- entropy jump at ``m`` raised to ``beta``.
    If every jump is zero, the shared cut-law implementation falls back to a uniform distribution. Every accept/reject
    ratio includes the reverse/forward cut-law ratio required for the state-dependent proposal.

    ``sampling_params.temperature`` defines the proposal model. The paper's experiments use ``temperature = 1 / alpha``
    and ``beta = 4``. The scheduler, when given, is stepped once to fix that temperature for the whole run: the stored
    proposal log-probs would otherwise refer to a different proposal kernel.

    Args:
        mh_llm_model: Custom ``vLLM_Wrapper`` instance.
        prompts: One prompt or a batch of prompts.
        sampling_params: Sampling configuration carrying ``alpha`` and the fixed proposal temperature.
        num_of_blocks: Number of stagewise generation blocks.
        max_new_tokens: Generated-token horizon; a prompt stops early once a block ends on the EOS token.
        mcmc_steps: MH transitions after each block extension.
        beta: Entropy-jump exponent of the cut law; beta=0 gives uniform cuts.
        temperature_scheduler: Optional scheduler stepped once for the fixed proposal temperature.
        rng: NumPy-compatible random-number generator.

    Returns:
        One decoded continuation per prompt.
    """
    if isinstance(prompts, str):
        prompts = [prompts]
    if not prompts:
        return []
    if rng is None:
        rng = np.random.default_rng()

    alpha = sampling_params.alpha

    proposal_temperature = float(sampling_params.temperature)
    if temperature_scheduler is not None:
        temperature_scheduler.step_count = 0
        proposal_temperature = float(temperature_scheduler.step())
        
    # The custom engine computes full-vocabulary entropy alongside token scores.
    sampling_params = _copy_sampling_params(
        sampling_params,
        n=1,
        logprobs=1,
        ignore_eos=True,
        temperature=proposal_temperature,
    )
    verbose = getattr(mh_llm_model, "verbose", False)

    num_prompts = len(prompts)
    generated_tokens: list[list[int]] = [[] for _ in prompts]
    proposal_logprobs: list[list[float]] = [[] for _ in prompts]
    power_logprobs: list[list[float]] = [[] for _ in prompts]
    entropies: list[list[float]] = [[] for _ in prompts]
    active = set(range(num_prompts))
    block_size = max_new_tokens // num_of_blocks

    logger.info(
        "prompts: %d, alpha: %.4f, beta: %.4f, block size: %d, "
        "max new tokens: %d, MH steps: %d, blocks: %d",
        num_prompts,
        alpha,
        beta,
        block_size,
        max_new_tokens,
        mcmc_steps,
        num_of_blocks,
    )

    for block_index in range(num_of_blocks):
        if not active:
            break

        block_start = time.perf_counter()
        active_list = sorted(active)
        logger.info("block [%d/%d]", block_index, num_of_blocks)
        logger.info(
            "active prompts: %s, sequence lengths: %s",
            active_list,
            [len(generated_tokens[i]) for i in active_list],
        )

        # Algorithm 2, lines 3-5: extend every previous-stage state by one block.
        extension_prompts = [
            mh_llm_model._build_prompt(prompts[i], generated_tokens[i])
            for i in active_list
        ]
        extension_params = _copy_sampling_params(
            sampling_params,
            max_tokens=block_size,
        )
        extension_start = time.perf_counter()
        extension_outputs = mh_llm_model.llm.generate(
            extension_prompts,
            sampling_params=extension_params,
            use_tqdm=False,
        )

        for batch_index, prompt_index in enumerate(active_list):
            output = extension_outputs[batch_index].outputs[0]
            tokens, logq, logp, token_entropies = _unpack_suffix_output(
                output,
                block_size,
            )
            generated_tokens[prompt_index].extend(tokens)
            proposal_logprobs[prompt_index].extend(logq)
            power_logprobs[prompt_index].extend(logp)
            entropies[prompt_index].extend(token_entropies)

        logger.info(
            "  block extend took %.3f sec: %d prompts, new_suffix_len=%d",
            time.perf_counter() - extension_start,
            len(active_list),
            block_size,
        )

        # Algorithm 2, lines 6-16: fixed-horizon Entropy-Cut MH transitions.
        for step_index in range(mcmc_steps):
            step_start = time.perf_counter()
            build_start = time.perf_counter()            

            cut_indices ={}
            proposal_prompts = []
            proposal_params = []
            for i in active_list:
                # The cut law favors positive predictive-entropy jumps:
                # lambda_beta[j] ~ max(0, H_{j+1} - H_j)^beta.
                cut_probabilities = compute_entropy_cut_policy(
                    entropies[i],
                    beta=beta,
                )
                logger.info(
                    "  [step %d, prompt %d] Cut summary: %s",
                    step_index,
                    i,
                    summarize_cut_policy(
                        entropies[i],
                        cut_probabilities,
                        beta,
                    ),
                )
                cut_indices[i] = draw_entropy_cut_index(
                    rng,
                    entropies[i],
                    beta=beta,
                )
                cut_offset = cut_indices[i] - 1
                entropy_before = entropies[i][cut_offset]
                entropy_after = entropies[i][cut_offset + 1]
                suffix_length = len(generated_tokens[i]) - cut_indices[i]
                kept_token_word = (
                    "token" if cut_indices[i] == 1 else "tokens"
                )
                logger.info(
                    "  [step %d, prompt %d] Selected cut %d: keep the first "
                    "%d %s and regenerate the remaining %d of %d tokens. "
                    "Entropy changed from %.4f to %.4f at this cut; the "
                    "positive increase used by the sampler was %.4f.",
                    step_index,
                    i,
                    cut_indices[i],
                    cut_indices[i],
                    kept_token_word,
                    suffix_length,
                    len(generated_tokens[i]),
                    entropy_before,
                    entropy_after,
                    max(0.0, entropy_after - entropy_before),
                )
                one_prompt=mh_llm_model._build_prompt(
                    prompts[i],
                    generated_tokens[i][: cut_indices[i]],
                )
                proposal_prompts.append(one_prompt)
                one_samp_param = _copy_sampling_params(
                    sampling_params,
                    max_tokens=len(generated_tokens[i]) - cut_indices[i],
                )
                proposal_params.append(one_samp_param)


                
            build_seconds = time.perf_counter() - build_start

            generate_start = time.perf_counter()
            outputs = mh_llm_model.llm.generate(
                proposal_prompts,
                sampling_params=proposal_params,
                use_tqdm=False,
            )
            generate_seconds = time.perf_counter() - generate_start
            

            score_start = time.perf_counter()
            step_log_acceptance_ratios: dict[int, float] = {}
            step_acceptances: dict[int, bool] = {}
            for batch_index, prompt_index in enumerate(active_list):
                cut_index = cut_indices[prompt_index]
                suffix_length = len(generated_tokens[prompt_index]) - cut_index
                output = outputs[batch_index].outputs[0]
                proposed_tokens, proposed_logq, proposed_logp, proposed_entropy = (
                    _unpack_suffix_output(output, suffix_length)
                )

                log_acceptance_ratio = compute_acceptance_ratio(
                    proposed_logprobs=proposed_logq,
                    curr_logprobs=proposal_logprobs[prompt_index][cut_index:],
                    proposed_power_logprobs=proposed_logp,
                    curr_power_logprobs=power_logprobs[prompt_index][cut_index:],
                    alpha=alpha,
                    current_entropies=entropies[prompt_index],
                    proposed_suffix_entropies=proposed_entropy,
                    cut_index=cut_index,
                    beta=beta,
                )
                acceptance_probability = math.exp(min(0.0, log_acceptance_ratio))
                accepted = rng.random() < acceptance_probability
                step_log_acceptance_ratios[prompt_index] = log_acceptance_ratio
                step_acceptances[prompt_index] = accepted

                if accepted:
                    generated_tokens[prompt_index][cut_index:] = proposed_tokens
                    proposal_logprobs[prompt_index][cut_index:] = proposed_logq
                    power_logprobs[prompt_index][cut_index:] = proposed_logp
                    entropies[prompt_index][cut_index:] = proposed_entropy

                if verbose:
                    logger.debug(
                        "prompt [%d], cut: %d, suffix length: %d, log "
                        "acceptance ratio: %.4f, acceptance probability: "
                        "%.4f, accepted: %s",
                        prompt_index,
                        cut_index,
                        suffix_length,
                        log_acceptance_ratio,
                        acceptance_probability,
                        accepted,
                    )

            score_seconds = time.perf_counter() - score_start
            logger.debug(
                "cut indices: %s, log acceptance ratios: %s, accepted: %s",
                cut_indices,
                step_log_acceptance_ratios,
                step_acceptances,
            )
            logger.info(
                "  MH step %d/%d took %.3f seconds: %d prompts",
                step_index,
                mcmc_steps,
                time.perf_counter() - step_start,
                len(active_list),
            )
            logger.info(
                "    step %d breakdown: build %.6f sec, generate %.6f sec, "
                "score %.6f sec",
                step_index,
                build_seconds,
                generate_seconds,
                score_seconds,
            )

        # A block that ends on EOS retires the prompt: later blocks would extend past the stop token, and its token
        # lists stay at the length the final MH step left them.
        eos_finished = {
            i
            for i in active_list
            if generated_tokens[i]
            and generated_tokens[i][-1] == mh_llm_model.tokenizer.eos_token_id
        }
        if eos_finished:
            logger.info(
                "stop on eos after block %d for prompts %s",
                block_index,
                sorted(eos_finished),
            )
        active -= eos_finished

        logger.info(
            "block [%d/%d] took %.3f seconds",
            block_index,
            num_of_blocks,
            time.perf_counter() - block_start,
        )

    generated_output_seq =  [
        mh_llm_model.tokenizer.decode(tokens, skip_special_tokens=True)
        for tokens in generated_tokens
    ]

    return generated_output_seq


__all__ = [
    "compute_acceptance_ratio",
    "mcmc_power_sampler_entropycut",
]
