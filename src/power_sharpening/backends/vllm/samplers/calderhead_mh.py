"""
 Calderhead's MH algorithm. The orginal code is from:
 https://github.com/Lev-Stambler/reasoning-with-samples-efficient/blob/master/src/benchmark_runner.py
"""
import logging
import random
import numpy as np

from vllm.logprobs import Logprob


from power_sharpening.backends.vllm.engine_patch import SamplingParams
from power_sharpening.backends.vllm.engine_patch.sampling_params import _copy_sampling_params

logger = logging.getLogger("[Calderhead MH]")

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
    """
    Parallel MCMC with multiple proposals per step.

    Implements the MH acceptance ratio from "Reasoning with Sampling" (Eq. 9):

    A(x, x') = min{1, [p(x')^α · q(x|x')] / [p(x)^α · q(x'|x)]}

    With independent proposal q(x) = p(x):

    log R = α·log P(x') + log P(x) - α·log P(x) - log P(x')
          = log_target' + log_p - log_target - log_p'

    Uses Calderhead's parallel structure: generate N proposals, build transition matrix, sample next state. This enables
    parallel generation while maintaining correct MH dynamics.

    Args:
        proposed_logprobs:       Base-model log-probs for the proposed suffix.
        curr_logprobs:           Base-model log-probs for the current suffix.
        proposed_power_logprobs: Power-distribution log-probs for the proposed suffix.
        curr_power_logprobs:     Power-distribution log-probs for the current suffix.
        alpha:                   Exponent of the power distribution.

    Returns:
        True if the proposal is accepted, False otherwise.
    """
    # log q(x) / q(x')  —  proposal (base model) ratio in log-space
    log_prob_ratio = sum(proposed_logprobs) - sum(curr_logprobs)

    # log pi(x') / pi(x)  —  target (power distribution) ratio in log-space
    if alpha == float("inf"):
        log_power_prob_ratio = 0.0  # TODO: check this
    else:
        log_power_prob_ratio = (
            sum(proposed_power_logprobs) - sum(curr_power_logprobs)
        ) * alpha

    # A = exp(log_power_prob_ratio + log_prob_ratio);  clamp overflow to 1.0
    # so that overflowing ratios are always accepted.
    log_acceptance_ratio = log_power_prob_ratio + log_prob_ratio
    return log_acceptance_ratio


def _categorical_from_log_weights(log_weights: list[float]) -> int:
    """Sample an index from a categorical distribution given unnormalized log weights.

    Uses the log-sum-exp trick for numerical stability.
    """
    arr = np.asarray(log_weights, dtype=np.float64)
    arr = arr - np.max(arr)
    probs = np.exp(arr)
    probs = probs / probs.sum()
    return int(np.random.choice(len(probs), p=probs))


def calderhead_mcmc_power_sampler(
    mh_llm_model,
    prompts: str | list[str],
    sampling_params: SamplingParams,
    num_of_blocks: int = 8,
    max_new_tokens: int = 3_072,
    mcmc_steps: int = 10,
    num_proposals: int = 4,
    temperature_scheduler=None,
) -> list[str]:
    """
    Parallel MCMC with multiple proposals per step.
    reference: https://www.pnas.org/doi/epdf/10.1073/pnas.1408184111

    Implements the MH acceptance ratio from "Reasoning with Sampling" (Eq. 9):

    A(x, x') = min{1, [p(x')^α · q(x|x')] / [p(x)^α · q(x'|x)]}

    With independent proposal q(x) = p(x):

    log R = α·log P(x') + log P(x) - α·log P(x) - log P(x')
          = log_target' + log_p - log_target - log_p'

    Uses Calderhead's parallel structure:
     generate N proposals, build transition matrix, sample next state. This enables parallel generation while
     maintaining
    correct MH dynamics.

    Accepts a single prompt or a batch.  The batch is processed jointly so that vLLM can parallelise the forward passes.

    Args:
        prompts: One or more input prompts.
        sampling_params: Sampling configuration (temperature, alpha, …).
        num_of_blocks: Number of blocks
        max_new_tokens: Hard cap on generated tokens per prompt.
        mcmc_steps: MH refinement steps after each block.
        num_proposals: Number of parallel proposals N per MH step (Calderhead). The transition picks among {current,
            proposal_1, ..., proposal_N} with probabilities proportional to the unnormalized target density (up to the
            proposal ratio). N=1 recovers standard single-proposal MH.

    Returns:
        A list of generated strings (one per prompt).
    """
    # Normalise to list so the rest of the method has a single code-path.
    if isinstance(prompts, str):
        prompts = [prompts]

    if num_proposals < 1:
        raise ValueError(f"num_proposals must be >= 1, got {num_proposals}")

    sampling_params = _copy_sampling_params(sampling_params, n=1, logprobs=1)
    alpha = sampling_params.alpha
    verbose = getattr(mh_llm_model, "verbose", False)

    n = len(prompts)
    generated_output_seq = [[] for _ in range(n)]
    logprobs = [[] for _ in range(n)]
    power_logprobs = [[] for _ in range(n)]
    active = set(range(n))

    one_block_size = int(max_new_tokens // num_of_blocks)

    logger.debug("n_prompts=%d, alpha=%.4f, block_size=%d, max_new_tokens=%d, "
                 "mcmc_steps=%d, num_of_blocks=%d",
                 n, alpha, one_block_size, max_new_tokens, mcmc_steps, num_of_blocks)

    for bi in range(num_of_blocks):
        if not active:
            break

        temperature_scheduler.step_count = 0
        init_temp = temperature_scheduler.step()
        sampling_params = _copy_sampling_params(sampling_params, temperature=init_temp)
        logger.info("set init temperature %.6f", init_temp)


        active_list = sorted(active)

        logger.info("block [%d/%d], active_prompts=%d, seq_lens=%s",
                     bi, num_of_blocks, len(active_list),
                     [len(generated_output_seq[i]) for i in active_list])

        # --- Forward pass: extend each active sequence by one block ------
        fwd_prompts = [
            mh_llm_model._build_prompt(prompts[i], generated_output_seq[i])
            for i in active_list
        ]
        fwd_sp = _copy_sampling_params(sampling_params, max_tokens=one_block_size)
        fwd_outputs = mh_llm_model.llm.generate(
            fwd_prompts, sampling_params=fwd_sp, use_tqdm=False
        )

        for batch_idx, i in enumerate(active_list):
            out = fwd_outputs[batch_idx].outputs[0]
            generated_output_seq[i].extend(out.token_ids)
            logprobs[i].extend(_extract_logprobs(out.logprobs))
            power_logprobs[i].extend(_extract_logprobs(out.power_logprobs))

            logger.info("  prompt[%d] fwd: +%d tokens, total_len=%d, "
                         "block_logprob_sum=%.4f, block_power_logprob_sum=%.4f",
                         i, len(out.token_ids), len(generated_output_seq[i]),
                         sum(_extract_logprobs(out.logprobs)),
                         sum(_extract_logprobs(out.power_logprobs)))

        # STEP 2: MCMC refinement to ensure convergence to p^alpha
        for mi in range(mcmc_steps):

            # Pick a random cut-point per active prompt (at least 1 token resampled).
            cut_ids = {
                i: random.randint(0, max(len(generated_output_seq[i]) - 2, 0))
                for i in active_list
            }

            logger.info("  MH step %d/%d, cut_ids=%s", mi, mcmc_steps, cut_ids)

            mcmc_prompts = [
                mh_llm_model._build_prompt(prompts[i], generated_output_seq[i][: cut_ids[i]])
                for i in active_list
            ]

            old_temp = sampling_params.temperature
            new_temp = temperature_scheduler.step()
            # Request N parallel proposals per prompt via vLLM's `n`.
            mcmc_samp_param = [
                _copy_sampling_params(
                    sampling_params,
                    max_tokens=len(generated_output_seq[i]) - cut_ids[i],
                    temperature=new_temp,
                    n=num_proposals,
                )
                for i in active_list
            ]
            logger.info("set temperature %.6f -> %.6f (N=%d proposals)",
                        old_temp, new_temp, num_proposals)
            mcmc_outputs = mh_llm_model.llm.generate(
                mcmc_prompts, sampling_params=mcmc_samp_param, use_tqdm=False
            )

            n_accepted = 0
            for batch_idx, i in enumerate(active_list):
                idx = cut_ids[i]
                curr_suffix_logprobs = logprobs[i][idx:]
                curr_suffix_power_logprobs = power_logprobs[i][idx:]

                # Collect per-proposal quantities.
                prop_token_ids: list[list[int]] = []
                prop_logprobs_list: list[list[float]] = []
                prop_power_logprobs_list: list[list[float]] = []
                # log_weights[0] corresponds to the current state (baseline = 0);
                # log_weights[1..N] are the log MH ratios of proposals vs current.
                log_weights: list[float] = [0.0]

                for prop_out in mcmc_outputs[batch_idx].outputs:
                    p_lp = _extract_logprobs(prop_out.logprobs)
                    p_plp = _extract_logprobs(prop_out.power_logprobs)
                    prop_token_ids.append(list(prop_out.token_ids))
                    prop_logprobs_list.append(p_lp)
                    prop_power_logprobs_list.append(p_plp)
                    log_weights.append(
                        compute_acceptance_ratio(
                            p_lp,
                            curr_suffix_logprobs,
                            p_plp,
                            curr_suffix_power_logprobs,
                            alpha,
                        )
                    )

                # Calderhead parallel MCMC: sample next state from categorical over {current, prop_1, ..., prop_N}
                # weighted by unnormalized target-over-proposal density. With independent proposals, this admits the
                # correct stationary distribution.
                selected = _categorical_from_log_weights(log_weights)

                suffix_len = len(generated_output_seq[i]) - idx
                probs_preview = np.exp(np.asarray(log_weights) - np.max(log_weights))
                probs_preview = probs_preview / probs_preview.sum()
                logger.info(
                    "    prompt[%d] MH step %d: cut=%d, suffix_len=%d, N=%d, "
                    "log_weights=%s, probs=%s, selected=%d (%s)",
                    i, mi, idx, suffix_len, num_proposals,
                    [f"{w:.3f}" for w in log_weights],
                    [f"{p:.3f}" for p in probs_preview],
                    selected,
                    "STAY" if selected == 0 else f"PROPOSAL_{selected}",
                )

                # selected == 0 means keep current state; otherwise adopt proposal.
                if selected > 0:
                    n_accepted += 1
                    j = selected - 1
                    generated_output_seq[i] = (
                        generated_output_seq[i][:idx] + prop_token_ids[j]
                    )
                    logprobs[i] = logprobs[i][:idx] + prop_logprobs_list[j]
                    power_logprobs[i] = (
                        power_logprobs[i][:idx] + prop_power_logprobs_list[j]
                    )

            logger.info(
                "  MH step %d/%d summary: accepted %d/%d prompts (rate=%.3f)",
                mi, mcmc_steps, n_accepted, len(active_list),
                n_accepted / max(len(active_list), 1),
            )

        # --- Remove prompts that hit EOS ---------------------------------
        eos_finished = {
            i
            for i in active_list
            if generated_output_seq[i]
            and generated_output_seq[i][-1] == mh_llm_model.tokenizer.eos_token_id
        }
        if eos_finished:
            logger.info("EOS reached for prompts %s (remaining active=%d)",
                        eos_finished, len(active) - len(eos_finished))
        active -= eos_finished

    logger.info("done, final seq_lens=%s",
                [len(generated_output_seq[i]) for i in range(n)])

    generated_output_seq = [
        mh_llm_model.tokenizer.decode(generated_output_seq[i], skip_special_tokens=True)
        for i in range(n)
    ]
    return generated_output_seq
