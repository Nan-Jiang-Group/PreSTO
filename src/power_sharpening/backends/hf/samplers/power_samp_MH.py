"""Sequential PowerMH sampling for the HuggingFace backend.

Run the CPU Entropy-Cut tests with:
    uv run --project /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src \
      pytest /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/tests/test_hf_power_mh_entropy_cut.py

Based on https://github.com/aakaran/reasoning-with-sampling/blob/main/llm_experiments/power_samp_utils.py
"""
import math

import numpy as np

from power_sharpening.common.cut_distribution import (
    compute_entropy_cut_policy,
    draw_entropy_cut_index,
    draw_uniform_cut_index,
    entropy_cut_log_ratio,
)

from .low_temp_sampler import low_temp_sampling as low_temp_proposal_sampling

def mcmc_power_sampler(
        sampler_wrapper,
        context,
        mcmc_steps,
        max_new_tokens=1024,
        num_of_blocks=8,
        given_cut_idx=-1,
        temperature_schedule_type='const',
        verbose=False,
        cut_dist_type: str = "uniform",
        cut_power: float = 4.0,
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
        cut_dist_type: ``"uniform"`` or state-dependent ``"entropy"`` cuts.
        cut_power: Entropy-jump exponent beta when using entropy cuts.
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
    if cut_dist_type not in {"uniform", "entropy"}:
        raise ValueError(
            f"unknown cut_dist_type {cut_dist_type!r}; expected 'uniform' "
            "or 'entropy'"
        )
    use_entropy_cut = cut_dist_type == "entropy"
    if use_entropy_cut and given_cut_idx > 0:
        raise ValueError(
            "given_cut_idx cannot be combined with cut_dist_type='entropy'"
        )
    if use_entropy_cut and temperature_schedule_type != "const":
        raise ValueError(
            "EntropyCut requires a fixed proposal distribution; use the "
            "constant temperature schedule"
        )
    if not math.isfinite(cut_power) or cut_power < 0.0:
        raise ValueError("cut_power must be a finite nonnegative value")
    if rng is None:
        rng = np.random.default_rng()

    context_len = len(context)

    generated_output_seq = []
    if context is not None:
        generated_output_seq = context.copy()
    proposal_logprobs_seq = []
    target_log_scores_seq = []
    entropies = []

    assert max_new_tokens % num_of_blocks == 0
    one_block_size = int(max_new_tokens // num_of_blocks)
    if verbose:
        print("one_block_size:", one_block_size, "num of blocks", num_of_blocks)

    total_attempt, num_acceptance = [], []

    for bi in range(num_of_blocks):

        print(f"current block {bi}/{num_of_blocks}......", flush=True)
        sampler_wrapper.init_schedule(
            temperature_schedule_type, mcmc_steps
        )
        # STEP 1: Generate intial proposal from p^alpha This is where the actual sampling from p^alpha happens (via
        # naive_temp)
        extension = low_temp_proposal_sampling(
            sampler_wrapper,
            context=generated_output_seq,
            seq_len=one_block_size + len(generated_output_seq),
            use_cache=True,
            ignore_eos=use_entropy_cut,
            return_entropies=use_entropy_cut,
        )
        if use_entropy_cut:
            (
                generated_output_seq,
                suffix_proposal_logprob,
                suffix_target_log_score,
                suffix_entropies,
            ) = extension
            entropies.extend(suffix_entropies)
        else:
            (
                generated_output_seq,
                suffix_proposal_logprob,
                suffix_target_log_score,
            ) = extension
        proposal_logprobs_seq.extend(suffix_proposal_logprob)
        target_log_scores_seq.extend(suffix_target_log_score)

        # STEP 2: MCMC refinement to ensure convergence to p^alpha
        block_attempt = 0
        block_acceptance = 0
        for mi in range(mcmc_steps):
            # print(mi,)

            print(f"MH step {mi}/{mcmc_steps}", end=", ", flush=True)
            print(f"set temperature {sampler_wrapper.temperature:.6f}", end=" -> ", flush=True)
            sampler_wrapper.temperature_step()
            print(
                f"{sampler_wrapper.temperature:.6f}",
                end="\t",
                flush=True,
            )

            block_attempt += 1
            t = len(generated_output_seq)
            if given_cut_idx > 0:
                cut_idx = given_cut_idx
            elif use_entropy_cut and len(entropies) >= 2:
                cut_idx = draw_entropy_cut_index(
                    rng,
                    entropies,
                    low=context_len + 1,
                    beta=cut_power,
                )
            else:
                cut_idx = draw_uniform_cut_index(
                    rng,
                    context_len,
                    t - 1,
                )
            if verbose:
                print(f"cut index is {cut_idx}, total seq_len is {t}")

            # Generate new proposal from p^alpha (sampling happens here)
            proposed = low_temp_proposal_sampling(
                sampler_wrapper,
                context=generated_output_seq[:cut_idx],
                seq_len=t,
                use_cache=True,
                ignore_eos=use_entropy_cut,
                return_entropies=use_entropy_cut,
            )
            if use_entropy_cut:
                (
                    prop,
                    suffix_proposal_logprob_x_prime,
                    suffix_target_log_score_x_prime,
                    suffix_entropies_x_prime,
                ) = proposed
            else:
                (
                    prop,
                    suffix_proposal_logprob_x_prime,
                    suffix_target_log_score_x_prime,
                ) = proposed
            s = len(prop)
            assert len(suffix_proposal_logprob_x_prime) == s - cut_idx
            assert len(suffix_target_log_score_x_prime) == s - cut_idx
            # idx - c: start index in the log-prob arrays for token at position idx. s - c: end index in the log-prob
            # arrays for the last generated token (position s-1).
            proposal_logprob_x = proposal_logprobs_seq[
                cut_idx - context_len:s - context_len
            ]
            target_log_score_x = target_log_scores_seq[
                cut_idx - context_len:s - context_len
            ]

            # METROPOLIS-HASTINGS ACCEPTANCE RATIO:
            # acceptance_ratio = log(p(x')^alpha / p(x)^alpha * q(x|x') / q(x'|x))
            # log_acceptance_ratio = alpha * log(p(x') + log q(x|x') - alpha log p(x) - log q(x'|x))
            # where p^alpha is target, q is proposal (also p^alpha)

            # - suffix_target_log_score_x_prime, Log Pi(x'): alpha * log(p(x')
            # - target_log_score_x, Log Pi(x): alpha *log p(x) (target prob of current sequence)
            part1 = (
                math.fsum(suffix_target_log_score_x_prime)
                - math.fsum(target_log_score_x)
            )

            # - proposal_logprob_x: log q(x|x') (prob of generating current from proposed)
            # - suffix_proposal_logprob_x_prime = log q(x'|x) (prob of generating proposed from current)
            part2 = (
                math.fsum(proposal_logprob_x)
                - math.fsum(suffix_proposal_logprob_x_prime)
            )

            cut_term = 0.0
            if use_entropy_cut and len(entropies) >= 2:
                retained_generated_tokens = cut_idx - context_len
                policy_offset = retained_generated_tokens - 1
                proposed_entropies = (
                    entropies[:retained_generated_tokens]
                    + suffix_entropies_x_prime
                )
                current_cut_probabilities = compute_entropy_cut_policy(
                    entropies,
                    beta=cut_power,
                )
                proposed_cut_probabilities = compute_entropy_cut_policy(
                    proposed_entropies,
                    beta=cut_power,
                )
                cut_term = entropy_cut_log_ratio(
                    current_cut_probabilities,
                    proposed_cut_probabilities,
                    cut_offset=policy_offset,
                )

            log_acceptance_ratio = part1 + part2 + cut_term

            # One draw, taken once and reused, so the trace below reports the decision that was actually made rather
            # than a second sample.
            acceptance_probability = math.exp(
                min(0.0, log_acceptance_ratio)
            )
            accepted = float(rng.random()) < acceptance_probability
            if verbose:
                # One line per realized MH transition. The baseline has no proposal tree, so this is the analogue of the
                # subtree run's Node(...) lines: it is what the branch-uncertainty analysis reads to recover A = min(1,
                # exp(ratio)) and the outcome Y.
                print(
                    f"MHStep(block={bi}, step={mi}, "
                    f"log_acc_ratio={log_acceptance_ratio:.4f}, "
                    f"accepted={int(accepted)}, cut_idx={cut_idx}, seq_len={t})",
                    flush=True,
                )
            if accepted:
                block_acceptance += 1
                generated_output_seq = prop.copy()
                proposal_logprobs_seq[cut_idx - context_len:] = (
                    suffix_proposal_logprob_x_prime.copy()
                )
                target_log_scores_seq[cut_idx - context_len:] = (
                    suffix_target_log_score_x_prime.copy()
                )
                if use_entropy_cut:
                    entropies[cut_idx - context_len:] = (
                        suffix_entropies_x_prime.copy()
                    )

                del prop, proposal_logprob_x, target_log_score_x

        if sampler_wrapper.tokenizer.eos_token_id in generated_output_seq:
            # contains End-of-sentence token
            eos_idx = generated_output_seq.index(
                sampler_wrapper.tokenizer.eos_token_id
            )
            generated_output_seq = generated_output_seq[:eos_idx + 1]
            cache_length = max(0, eos_idx - context_len + 1)
            proposal_logprobs_seq = proposal_logprobs_seq[:cache_length]
            target_log_scores_seq = target_log_scores_seq[:cache_length]
            if use_entropy_cut:
                entropies = entropies[:cache_length]
            break

        num_acceptance.append(block_acceptance)
        total_attempt.append(block_attempt)

    num_attempts = int(np.sum(total_attempt))
    acceptance_ratio = (
        float(np.sum(num_acceptance)) / num_attempts
        if num_attempts
        else 0.0
    )

    print(f"acceptances, attempts: {num_acceptance} / {total_attempt}")
    return (
        generated_output_seq,
        proposal_logprobs_seq,
        target_log_scores_seq,
        acceptance_ratio,
    )
