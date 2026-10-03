"""
Metropolis-Hastings algorithm based on VLLM.
"""
import logging
import time

import numpy as np


from vllm.logprobs import Logprob
from power_sharpening.backends.vllm.engine_patch.entropy import suffix_log_acceptance
from power_sharpening.common.cut_distribution import draw_uniform_cut_index


from power_sharpening.backends.vllm.engine_patch import SamplingParams
from power_sharpening.backends.vllm.engine_patch.sampling_params import (
    _copy_sampling_params,
)


logger = logging.getLogger("[power MH]")

_ACCEPTANCE_THRESHOLD = 1 - 1e-5


def _extract_logprobs(logprobs: list[dict[int, Logprob]]) -> list[float]:
    """Extract the top-1 logprob value from each decoding step."""
    return [list(lp.values())[0].logprob for lp in logprobs]



def compute_acceptance_ratio(
    proposed_logprobs: list[float],
    curr_logprobs: list[float],
    proposed_power_logprobs: list[float],
    curr_power_logprobs: list[float],
    alpha: float,
) -> float:
    """Metropolis-Hastings accept/reject decision for a proposed suffix.

    We target the power distribution  pi(x) ~ p(x)^alpha  and use the base
    model p(x) as the proposal.  The MH acceptance ratio is:

        A = [pi(x') / pi(x)] * [q(x) / q(x')]

    where log p and log p_power are the base and power log-probs over the suffix tokens, respectively.  The proposal is
    accepted deterministically when A >= 1 (up to a small numerical tolerance), otherwise accepted with probability A.

    Args:
        proposed_logprobs:       Base-model log-probs for the proposed suffix.
        curr_logprobs:           Base-model log-probs for the current suffix.
        proposed_power_logprobs: Power-distribution log-probs for the proposed suffix.
        curr_power_logprobs:     Power-distribution log-probs for the current suffix.
        alpha:                   Exponent of the power distribution.

    Returns:
        The MH log-acceptance ratio.
    """
    return suffix_log_acceptance(
        proposed_logq=proposed_logprobs,
        current_logq=curr_logprobs,
        proposed_logp=proposed_power_logprobs,
        current_logp=curr_power_logprobs,
        alpha=alpha,
    )


def mcmc_power_sampler(
    mh_llm_model,
    prompts: str | list[str],
    sampling_params: SamplingParams,
    num_of_blocks: int = 8,
    max_new_tokens: int = 3_072,
    mcmc_steps: int = 10,
    temperature_scheduler=None,
    rng=None,
) -> list[str]:
    """Generate text using blockwise Metropolis-Hastings sampling.

    Accepts a single prompt or a batch.  The batch is processed jointly so that vLLM can parallelise the forward passes.

    Args:
        prompts: One or more input prompts.
        sampling_params: Sampling configuration (temperature, alpha, …).
        num_of_blocks: Number of blocks
        max_new_tokens: Hard cap on generated tokens per prompt.
        mcmc_steps: MH refinement steps after each block.
        cut_dist_type: State-independent cut-index distribution.
        rng: Random-number generator used for cut draws.

    Returns:
        A list of generated strings (one per prompt).
    """
    # Normalise to list so the rest of the method has a single code-path.
    if isinstance(prompts, str):
        prompts = [prompts]

    if rng is None:
        # draw_uniform_cut_index needs a NumPy Generator: it calls .integers, which the stdlib random module does not
        # have.
        rng = np.random.default_rng()

    sampling_params = _copy_sampling_params(
        sampling_params,
        n=1,
        logprobs=1,
        ignore_eos=True,
    )
    alpha = sampling_params.alpha
    verbose = getattr(mh_llm_model, "verbose", False)

    n = len(prompts)
    generated_output_seq = [[] for _ in range(n)]
    logprobs = [[] for _ in range(n)]
    power_logprobs = [[] for _ in range(n)]
    active = set(range(n))

    one_block_size = int(max_new_tokens // num_of_blocks)

    logger.info(
        "prompts: %d, alpha: %.4f, block size: %d, max new tokens: %d, "
        "MH steps: %d, blocks: %d",
        n,
        alpha,
        one_block_size,
        max_new_tokens,
        mcmc_steps,
        num_of_blocks,
    )

    for bi in range(num_of_blocks):
        if not active:
            break

        block_start = time.perf_counter()
        logger.info("block [%d/%d]", bi, num_of_blocks)

        temperature_scheduler.step_count = 0
        init_temp = temperature_scheduler.step()
        sampling_params = _copy_sampling_params(
            sampling_params,
            temperature=init_temp,
        )
        logger.debug("initial temperature: %.6f", init_temp)

        active_list = sorted(active)
        logger.info(
            "active prompts: %s, sequence lengths: %s",
            active_list,
            [len(generated_output_seq[i]) for i in active_list],
        )

        # --- Forward pass: extend each active sequence by one block ------
        extend_start = time.perf_counter()
        fwd_prompts = [
            mh_llm_model._build_prompt(prompts[i], generated_output_seq[i])
            for i in active_list
        ]
        fwd_sp = _copy_sampling_params(sampling_params, max_tokens=one_block_size)
        # print(">>>",fwd_sp)
        # print("\t>>> one_block_size:", one_block_size)
        fwd_outputs = mh_llm_model.llm.generate(
            fwd_prompts, sampling_params=fwd_sp, use_tqdm=False
        )
        # The batched twin of vllm_proposal_sampling's line, carried on the timing message so one read explains the
        # cost: how many sequences the engine decoded in lockstep, and how long each was going in and coming out. The
        # lengths are gathered before the loop below extends the sequences, so context_len is still the pre-extension
        # length.
        logger.info(
            "  block extend took %.3f sec: %d prompts, context_len=%s, "
            "new_suffix_len=%s",
            time.perf_counter() - extend_start,
            len(fwd_prompts),
            [len(output.prompt_token_ids) for output in fwd_outputs],
            [len(output.outputs[0].token_ids) for output in fwd_outputs],
        )

        for batch_idx, i in enumerate(active_list):
            out = fwd_outputs[batch_idx].outputs[0]
            block_proposal_lp = _extract_logprobs(out.logprobs)
            block_power_lp = _extract_logprobs(out.power_logprobs)
            generated_output_seq[i].extend(out.token_ids)
            logprobs[i].extend(block_proposal_lp)
            power_logprobs[i].extend(block_power_lp)

            logger.debug(
                "prompt [%d], sequence length: %d, block proposal logprobs: "
                "%d, block power logprobs: %d",
                i,
                len(generated_output_seq[i]),
                len(block_proposal_lp),
                len(block_power_lp),
            )

        # STEP 2: MCMC refinement to ensure convergence to p^alpha
        for mi in range(mcmc_steps):
            step_start = time.perf_counter()

            # Pick a random cut-point per active prompt (at least 1 token resampled).
            build_start = time.perf_counter()
            old_temp = sampling_params.temperature
            new_temp = temperature_scheduler.step()

            cut_ids = {}
            mcmc_prompts = []
            mcmc_samp_param = []
            for i in active_list:
                cut_id = draw_uniform_cut_index(
                    rng,
                    0,
                    max(len(generated_output_seq[i]) - 2, 0),
                )
                cut_ids[i] = cut_id
                one_prompt = mh_llm_model._build_prompt(
                        prompts[i],
                        generated_output_seq[i][:cut_id],
                    )
                mcmc_prompts.append(one_prompt)
                one_samp_param = _copy_sampling_params(
                        sampling_params,
                        max_tokens=len(generated_output_seq[i]) - cut_id,
                        temperature=new_temp,
                    )
                mcmc_samp_param.append(
                    one_samp_param
                )

            logger.debug("cut indices: %s", cut_ids)
            logger.debug("temperature: %.6f -> %.6f", old_temp, new_temp)
            build_seconds = time.perf_counter() - build_start

            generate_start = time.perf_counter()
            mcmc_outputs = mh_llm_model.llm.generate(
                mcmc_prompts, sampling_params=mcmc_samp_param, use_tqdm=False
            )
            generate_seconds = time.perf_counter() - generate_start

            logger.debug("num proposals: %d", len(mcmc_outputs))

            score_start = time.perf_counter()
            step_log_acceptance_ratios = {}
            step_acceptances = {}

            for batch_idx, i in enumerate(active_list):
                out = mcmc_outputs[batch_idx].outputs[0]
                proposed_logprobs = _extract_logprobs(out.logprobs)
                proposed_power_logprobs = _extract_logprobs(out.power_logprobs)
                idx = cut_ids[i]

                log_acceptance_ratio = compute_acceptance_ratio(
                    proposed_logprobs,
                    logprobs[i][idx:],
                    proposed_power_logprobs,
                    power_logprobs[i][idx:],
                    alpha,
                )

                accepted = rng.random() < np.exp(log_acceptance_ratio)
                step_log_acceptance_ratios[i] = log_acceptance_ratio
                step_acceptances[i] = bool(accepted)

                if verbose:
                    suffix_len = len(generated_output_seq[i]) - idx
                    logger.debug(
                        "prompt [%d], cut: %d, suffix length: %d, proposed "
                        "tokens: %d, log acceptance ratio: %.4f, acceptance "
                        "probability: %.4f, accepted: %s",
                        i,
                        idx,
                        suffix_len,
                        len(out.token_ids),
                        log_acceptance_ratio,
                        min(np.exp(log_acceptance_ratio), 1.0),
                        accepted,
                    )

                # Accept deterministically when A >= 1 (within tolerance), otherwise accept stochastically.
                if accepted:
                    generated_output_seq[i] = generated_output_seq[i][:idx] + list(
                        out.token_ids
                    )
                    logprobs[i] = logprobs[i][:idx] + proposed_logprobs
                    power_logprobs[i] = (
                        power_logprobs[i][:idx] + proposed_power_logprobs
                    )

            score_seconds = time.perf_counter() - score_start

            logger.debug(
                "log acceptance ratios: %s",
                step_log_acceptance_ratios,
            )
            logger.debug("accepted: %s", step_acceptances)
            logger.info(
                "  MH step %d/%d took %.3f seconds: context_len=%s, "
                "new_suffix_len=%s",
                mi,
                mcmc_steps,
                time.perf_counter() - step_start,
                [len(output.prompt_token_ids) for output in mcmc_outputs],
                [len(output.outputs[0].token_ids) for output in mcmc_outputs],
            )
            # Printed under the total it decomposes, matching the subtree sampler's breakdown so the two are read the
            # same way: assembling the cut prompts, the batched engine call, and forming the acceptance ratios and
            # splicing the caches. Deliberately avoids the "MH step <i>/<n> took <s> seconds" wording the case-study
            # parser counts as a model call, and the "mcmc_power_sampler took <s> seconds for <i>-th prompts" wording it
            # uses to close a sample.
            logger.info(
                "    step %d breakdown: %d prompts, build %.6f sec, generate %.6f sec, "
                "score %.6f sec",
                mi,
                len(active_list),
                build_seconds,
                generate_seconds,
                score_seconds,
            )

        # --- Remove prompts that hit EOS ---------------------------------
        eos_finished = {
            i
            for i in active_list
            if generated_output_seq[i]
            and generated_output_seq[i][-1] == mh_llm_model.tokenizer.eos_token_id
        }
        if eos_finished:
            logger.info(
                "stop on eos after block %d for prompts %s",
                bi,
                sorted(eos_finished),
            )
        active -= eos_finished

        logger.info(
            "block [%d/%d] took %.3f seconds",
            bi,
            num_of_blocks,
            time.perf_counter() - block_start,
        )

    logger.debug(
        "final sequence lengths: %s",
        [len(generated_output_seq[i]) for i in range(n)],
    )

    generated_output_seq = [
        mh_llm_model.tokenizer.decode(
            generated_output_seq[i],
            skip_special_tokens=True,
        )
        for i in range(n)
    ]
    return generated_output_seq


