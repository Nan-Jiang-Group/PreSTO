"""Blockwise power-MCMC (PowerMH) sampler on SGLang.

Run through ``python -m power_sharpening.runners.sglang.run_power_sample_mh --algorithm power_mcmc``; the log lines match
the vLLM sampler (``backends/vllm/samplers/power_sampling_mh.py``) so the case-study parsers read both backends.
"""
import logging
import time

import numpy as np

from power_sharpening.backends.sglang.samplers.low_temp_proposal_sampler import low_temp_proposal_sampling
from power_sharpening.common.cut_distribution import draw_uniform_cut_index

logger = logging.getLogger("[power MH]")


### DESCRIPTION ###
# power sampling to sample from p^{alpha}, where p is the base model takes in 1/alpha (temperature) as an argument
# (default 0.25), and mcmc_power_samp implements sampling from p^{alpha}
def mcmc_power_sampler(
        sampler_wrapper,  # AutoRegressiveLMWrapper (SGLang-backed)
        context,
        mcmc_steps,
        max_new_tokens=1024,
        num_of_blocks=8,
        given_cut_idx=-1,
        temperature_schedule_type='const',
        verbose=False,
        cut_dist_type: str = "uniform",
        rng=None,
):
    """
    Sample from p^alpha using Metropolis-Hastings MCMC.

    Args:
        sampler_wrapper: AR-Sampler instance.
        context: Initial context tokens

        mcmc_steps: Number of MCMC steps per block
        max_new_tokens: Maximum tokens to generate
        num_of_blocks: Number of blocks to generate
        cut_dist_type: State-independent cut-index distribution.
        rng: Random-number generator used for cut draws.

    procedure:
    1. Proposal generation: Uses low_temp_proposal_sampling() which samples from p^alpha via temperature scaling in
       model.generate() with temperature=temp
    2. MCMC acceptance: Metropolis-Hastings acceptance ratio ensures the stationary distribution is p^alpha
    3. The final output 'generated_output_seq' is a sample from p^alpha

    Returns:
        generated_output_seq: Generated sequence (sample from p^alpha)
        proposal_logprobs_seq: per-token log q(x_t | x_{<t}) under the proposal.
        target_log_scores_seq: per-token alpha * log p(x_t | x_{<t}), log-scores under p^alpha.
        acceptance_ratio: MCMC acceptance rate
    """
    if cut_dist_type == "entropy":
        raise ValueError(
            f"PowerMH does not support state-dependent cut distribution "
            f"{cut_dist_type!r}; its acceptance ratio would require the "
            "forward/reverse cut-probability correction"
        )
    if cut_dist_type != "uniform":
        raise ValueError(
            f"unknown cut_dist_type {cut_dist_type!r}; expected 'uniform'"
        )
    if rng is None:
        # draw_uniform_cut_index needs a NumPy Generator: it calls .integers, which the stdlib random module does not
        # have.
        rng = np.random.default_rng()

    context_len = len(context)

    generated_output_seq = []
    if context is not None:
        generated_output_seq = context.copy()
    proposal_logprobs_seq = []
    target_log_scores_seq = []

    assert max_new_tokens % num_of_blocks == 0
    one_block_size = int(max_new_tokens // num_of_blocks)
    logger.info(
        "prompts: %d, alpha: %.4f, block size: %d, max new tokens: %d, "
        "MH steps: %d, blocks: %d",
        1,
        sampler_wrapper.alpha,
        one_block_size,
        max_new_tokens,
        mcmc_steps,
        num_of_blocks,
    )

    total_attempt, num_acceptance = [], []

    for bi in range(num_of_blocks):
        block_start = time.perf_counter()
        logger.info("block [%d/%d]", bi, num_of_blocks)
        sampler_wrapper.init_schedule(
            temperature_schedule_type, mcmc_steps
        )
        logger.debug("initial temperature: %.6f", sampler_wrapper.temperature)
        logger.info(
            "active prompts: %s, sequence lengths: %s",
            [0],
            [len(generated_output_seq) - context_len],
        )
        # STEP 1: Generate intial proposal from p^alpha This is where the actual sampling from p^alpha happens (via
        # naive_temp)
        extend_start = time.perf_counter()
        block_context_len = len(generated_output_seq)
        generated_output_seq, suffix_proposal_logprob, suffix_target_log_score, _ = (
            low_temp_proposal_sampling(
                sampler_wrapper,
                context=generated_output_seq,
                seq_len=one_block_size + len(generated_output_seq),
                use_cache=True,
                # Fixed-length states, as in the vLLM sampler: an EOS-shortened proposal would change the state space and
                # invalidate the MH ratio. EOS is handled after the block.
                ignore_eos=True,
            )
        )
        logger.info(
            "  block extend took %.3f sec: %d prompts, context_len=%s, "
            "new_suffix_len=%s",
            time.perf_counter() - extend_start,
            1,
            [block_context_len],
            [len(generated_output_seq) - block_context_len],
        )
        proposal_logprobs_seq.extend(suffix_proposal_logprob)
        target_log_scores_seq.extend(suffix_target_log_score)

        # STEP 2: MCMC refinement to ensure convergence to p^alpha
        blcok_attempt = 0
        block_acceptance = 0
        for mi in range(mcmc_steps):
            step_start = time.perf_counter()
            build_start = time.perf_counter()
            old_temp = sampler_wrapper.temperature
            sampler_wrapper.temperature_step()

            blcok_attempt += 1
            t = len(generated_output_seq)
            if given_cut_idx > 0:
                cut_idx = given_cut_idx
            else:
                cut_idx = draw_uniform_cut_index(
                    rng,
                    context_len,
                    t - 1,
                )
            logger.debug("cut indices: %s", {0: cut_idx - context_len})
            logger.debug("temperature: %.6f -> %.6f", old_temp, sampler_wrapper.temperature)
            build_seconds = time.perf_counter() - build_start

            # Generate new proposal from p^alpha (sampling happens here)
            generate_start = time.perf_counter()
            prop, suffix_proposal_logprob_x_prime, suffix_target_log_score_x_prime, _ = low_temp_proposal_sampling(
                sampler_wrapper,
                context=generated_output_seq[:cut_idx],
                seq_len=t,
                use_cache=True,
                ignore_eos=True,
            )
            generate_seconds = time.perf_counter() - generate_start
            score_start = time.perf_counter()
            s = len(prop)
            assert (len(suffix_proposal_logprob_x_prime) == s - cut_idx)
            assert (len(suffix_target_log_score_x_prime) == s - cut_idx)
            # idx - c: start index in the log-prob arrays for token at position idx. s - c: end index in the log-prob
            # arrays for the last generated token (position s-1).
            proposal_logprob_x = proposal_logprobs_seq.copy()[cut_idx - context_len:s - context_len]
            target_log_score_x = target_log_scores_seq.copy()[cut_idx - context_len:s - context_len]

            # METROPOLIS-HASTINGS ACCEPTANCE RATIO:
            # log A = [alpha*log p(x') - alpha*log p(x)] + [log q(x|x') - log q(x'|x)]
            #
            # - suffix_target_log_score_x_prime, Log Pi(x'): alpha * log(p(x')
            # - target_log_score_x, Log Pi(x): alpha *log p(x) (target prob of current sequence)
            part1 = sum(suffix_target_log_score_x_prime) - sum(target_log_score_x)

            # - proposal_logprob_x: log q(x|x') (prob of generating current from proposed)
            # - suffix_proposal_logprob_x_prime = log q(x'|x) (prob of generating proposed from current)
            part2 = sum(proposal_logprob_x) - sum(suffix_proposal_logprob_x_prime)

            log_acceptance_ratio = part1 + part2

            # Same fraction-of-unmoved-proposals diagnostic as the vLLM sampler.
            logger.info(
                "    step %d proposal identical to current suffix: %s", mi, prop[cut_idx:] == generated_output_seq[cut_idx:]
            )
            accepted = rng.random() < np.exp(log_acceptance_ratio)
            logger.debug(
                "prompt [%d], cut: %d, suffix length: %d, proposed "
                "tokens: %d, log acceptance ratio: %.4f, acceptance "
                "probability: %.4f, accepted: %s",
                0,
                cut_idx - context_len,
                t - cut_idx,
                s - cut_idx,
                log_acceptance_ratio,
                min(np.exp(log_acceptance_ratio), 1.0),
                accepted,
            )
            if accepted:
                block_acceptance += 1
                generated_output_seq = prop.copy()
                proposal_logprobs_seq[cut_idx - context_len:] = suffix_proposal_logprob_x_prime.copy()
                target_log_scores_seq[cut_idx - context_len:] = suffix_target_log_score_x_prime.copy()

                del prop, proposal_logprob_x, target_log_score_x
            score_seconds = time.perf_counter() - score_start

            # Same wording as the vLLM sampler: the case-study parser counts "MH step <i>/<n> took <s> seconds" as one
            # target-model call.
            logger.info(
                "  MH step %d/%d took %.3f seconds: context_len=%s, "
                "new_suffix_len=%s",
                mi,
                mcmc_steps,
                time.perf_counter() - step_start,
                [cut_idx],
                [s - cut_idx],
            )
            logger.info(
                "    step %d breakdown: %d prompts, build %.6f sec, generate %.6f sec, "
                "score %.6f sec",
                mi,
                1,
                build_seconds,
                generate_seconds,
                score_seconds,
            )

        stop_on_eos = sampler_wrapper.tokenizer.eos_token_id in generated_output_seq
        if stop_on_eos:
            # contains End-of-sentence token
            eos_idx = generated_output_seq.index(sampler_wrapper.tokenizer.eos_token_id)
            generated_output_seq = generated_output_seq[:eos_idx + 1]
            proposal_logprobs_seq = proposal_logprobs_seq[:eos_idx + 1]
            target_log_scores_seq = target_log_scores_seq[:eos_idx + 1]
            logger.info("stop on eos after block %d for prompts %s", bi, [0])

        num_acceptance.append(block_acceptance)
        total_attempt.append(blcok_attempt)

        logger.info(
            "block [%d/%d] took %.3f seconds",
            bi,
            num_of_blocks,
            time.perf_counter() - block_start,
        )
        if stop_on_eos:
            break

    acceptance_ratio = np.sum(num_acceptance) / np.sum(total_attempt)

    logger.info("acceptances, attempts: %s / %s", num_acceptance, total_attempt)
    return generated_output_seq, proposal_logprobs_seq, target_log_scores_seq, acceptance_ratio
