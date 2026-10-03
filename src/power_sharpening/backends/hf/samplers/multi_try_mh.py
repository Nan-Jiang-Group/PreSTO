"""Paper-style Multiple-Try Metropolis for the HuggingFace backend.

Run with:
    srun -p $PARTITION --quotatype=$QUOTATYPE --gres=gpu:1 \
      --cpus-per-task=24 --time=03:00:00 \
      python -m power_sharpening.runners.hf.run_power_mh --task math500 \
      --algorithm multi_try --num_tries 5 --proposal_temperatures 0.25 0.5 1.0

Implements the MTM steps in Algorithm 1 of https://arxiv.org/html/2606.09926v1. For fixed ``context``, the unnormalized
power target in Eq. (2) is ``np.exp(sum(target_log_scores_seq))``; the scores already include the fixed exponent
``sampler_wrapper.alpha``. This variant refines every block and uses
uniform cuts unless a fixed cut is supplied. Code names used below:
    generated_output_seq: current sequence, including the prompt;
    prop_seqs[j]: candidate sequence sharing generated_output_seq[:cut_idx];
    num_tries: candidates per refinement; one_block_size: tokens per block;
    trial_log_weights[j]: log MH ratio for prop_seqs[j] (Eq. (6));
    selected: candidate index sampled proportionally to np.exp(trial_log_weights);
    log_acceptance: log probability of accepting prop_seqs[selected] (Eq. (7)).
The unselected candidates are reused; no extra reference suffixes are drawn.

The temperature-list extension draws a temperature independently for each candidate and holds it fixed throughout that
suffix. All candidates therefore use the same proposal: the mean of the whole-suffix probabilities under
``temperatures``. Both directions of Eq. (6) use this mixture density. ``sampler_wrapper.alpha`` stays fixed when
proposal temperatures change.
"""

import math
import random

import numpy as np

from power_sharpening.common.multi_try import (
    DEFAULT_PROPOSAL_TEMPERATURES,
    categorical_from_log_weights as _categorical_from_log_weights,
    log_acceptance_probability as _log_acceptance_probability,
    mixture_logprobs as _mixture_logprobs,
)

from .low_temp_sampler import batched_low_temp_proposal_sampling


def _log_weight(
    prop_target_log_score,
    curr_target_log_score,
    prop_proposal_logprob,
    curr_proposal_logprob,
) -> float:
    """Return a log MH ratio using Eq. (6), expressed with these arguments:

        _log_weight(...) = sum(prop_target_log_score) - sum(curr_target_log_score)
                           + sum(curr_proposal_logprob) - sum(prop_proposal_logprob).

    ``prop_target_log_score`` and ``curr_target_log_score`` contain per-token base-model log probabilities already
    multiplied by ``sampler_wrapper.alpha``. ``prop_proposal_logprob`` scores the proposed suffix (forward direction);
    ``curr_proposal_logprob`` scores the current suffix (reverse direction). Both proposals condition on the same
    retained prefix, whose target terms cancel. Summing each argument therefore gives the required suffix score.
    """
    return (
        math.fsum(prop_target_log_score) - math.fsum(curr_target_log_score)
        + math.fsum(curr_proposal_logprob) - math.fsum(prop_proposal_logprob)
    )


def _draw_proposals(sampler_wrapper, prefix, seq_len, count, component_temperatures):
    """Draw candidates in one batch and score the temperature-list extension.

    ``row_temperatures[j]`` is drawn from ``component_temperatures`` once and held fixed for all tokens in
    ``sequences[j][len(prefix):]``. Each row of ``component_scores[j]`` stores per-token log probabilities at one entry
    of ``component_temperatures``. The mixture used in Eq. (6) satisfies:

        sum(_mixture_logprobs(component_scores[j])) =
            np.logaddexp.reduce(np.sum(component_scores[j], axis=1))
            - np.log(len(component_temperatures)).

    Sum token scores before mixing: the temperature is drawn once per suffix. Every candidate is scored at every entry
    of ``component_temperatures``.
    """
    row_temperatures = (
        [component_temperatures[0]] * count
        if len(component_temperatures) == 1
        else np.random.choice(component_temperatures, size=count).tolist()
    )
    # seq_len is the full sequence horizon. Ignoring EOS keeps all candidates at that horizon until the block's
    # refinement steps have finished.
    sequences, _, target_scores, component_scores = batched_low_temp_proposal_sampling(
        sampler_wrapper,
        contexts=[prefix] * count,
        seq_len=seq_len,
        use_cache=True,
        ignore_eos=True,
        row_temperatures=row_temperatures,
        component_temperatures=component_temperatures,
    )
    # sequences include prefix; target_scores and component_scores cover only new tokens. component_scores has shape
    # (count, len(component_temperatures), seq_len - len(prefix)).
    return sequences, target_scores, np.asarray(component_scores, dtype=np.float64)


def multi_try_mcmc_power_sampler(
        sampler_wrapper,
        context,
        mcmc_steps,
        max_new_tokens=1024,
        num_of_blocks=8,
        num_tries=4,
        given_cut_idx=-1,
        temperature_schedule_type='const',
        verbose=False,
        proposal_temperatures=DEFAULT_PROPOSAL_TEMPERATURES,
):
    """Refine generated blocks with the paper's Multiple-Try Metropolis rule.

    Args:
        sampler_wrapper: HF wrapper with fixed target exponent ``alpha``.
        context: Nonempty prompt token list.
        mcmc_steps: Refinement steps per block; zero gives proposal sampling.
        max_new_tokens: Total generation horizon, divisible by num_of_blocks.
        num_of_blocks: Number of equal-sized generation blocks.
        num_tries: Candidates per step; num_tries=1 reduces to ordinary MH.
        given_cut_idx: Positive absolute token index for a fixed cut; otherwise choose uniformly over all generated
            positions before the block end.
        temperature_schedule_type: Accepted for call compatibility with the runners. The proposal is the fixed
            ``proposal_temperatures`` mixture, so no schedule is applied here.
        verbose: Print candidate selection and acceptance diagnostics.
        proposal_temperatures: Nonempty list of finite positive temperatures, defaulting to
            ``DEFAULT_PROPOSAL_TEMPERATURES``. Independently select a uniform list entry per candidate suffix and use
            the full mixture density for MH weights. Repeated entries give their temperature proportionally more mixture
            mass. A one-entry list is the fixed scalar proposal.

    Returns:
        A four-tuple: full token sequence; per-token proposal log probabilities; per-token base log probabilities times
        sampler_wrapper.alpha; acceptance rate. Proposal scores describe the final mixture starting at the prompt (or
        the final scalar scheduled proposal), not the selected component or the historical generation law.
    """

    temperatures = np.asarray(proposal_temperatures, dtype=np.float64).tolist()

    context_len = len(context)
    one_block_size = max_new_tokens // num_of_blocks

    # generated_output_seq includes context; both score arrays exclude it. target_log_scores_seq stores base log
    # probabilities multiplied by sampler_wrapper.alpha. component_logprobs_seq stores proposal log probabilities with
    # shape (len(temperatures), len(generated_output_seq) - context_len).
    generated_output_seq = list(context)
    target_log_scores_seq = []
    component_logprobs_seq = np.empty((len(temperatures), 0), dtype=np.float64)
    total_attempt, num_acceptance = [], []

    for bi in range(num_of_blocks):
        print(f"current block {bi}/{num_of_blocks}......", flush=True)
        # 1. Extend generated_output_seq by one_block_size tokens
        # (Algorithm 1, line 5). Refinements keep this block endpoint fixed.
        sequences, target_scores, component_scores = _draw_proposals(
            sampler_wrapper, generated_output_seq,
            len(generated_output_seq) + one_block_size, 1, temperatures,
        )
        generated_output_seq = sequences[0]
        target_log_scores_seq.extend(target_scores[0])
        component_logprobs_seq = np.concatenate(
            [component_logprobs_seq, component_scores[0]], axis=1
        )

        block_acceptance = 0
        for mi in range(mcmc_steps):
            # 2. Choose cut_idx uniformly from [context_len, t), unless
            # given_cut_idx fixes it. Earlier blocks can change; the cut probability cancels between directions at this
            # fixed horizon t.
            t = len(generated_output_seq)
            cut_idx = (
                given_cut_idx if given_cut_idx > 0
                else random.randint(context_len, t - 1)
            )
            # cut_idx indexes the full sequence; offset indexes response-only scores. The token at cut_idx is the first
            # token to regenerate.
            offset = cut_idx - context_len
            # Slice component_logprobs_seq[:, offset:] BEFORE mixing: each proposal draws its temperature afresh at
            # cut_idx. Thus sum(curr_proposal_logprob) scores generated_output_seq[cut_idx:] conditional on
            # generated_output_seq[:cut_idx].
            curr_proposal_logprob = _mixture_logprobs(
                component_logprobs_seq[:, offset:]
            )
            curr_target_log_score = target_log_scores_seq[offset:]
            # 3. Draw num_tries independent prop_seqs in one generation batch
            # (Algorithm 1, line 12). All share generated_output_seq[:cut_idx] and have full length t.
            prop_seqs, target_ls_list, prop_components = _draw_proposals(
                sampler_wrapper, generated_output_seq[:cut_idx], t, num_tries, temperatures,
            )
            """
            4. Eq. (6): compute each candidate's weight relative to the current
            sequence. Both sequences retain the same prefix, so only their suffix probabilities enter the ratio. For
            this update, write
              c = generated_output_seq[:cut_idx]   (retained prefix),
              x = generated_output_seq[cut_idx:]   (current suffix),
              y = prop_seqs[j][cut_idx:]           (candidate j's suffix).
            With p the base model, alpha = sampler_wrapper.alpha, and q the full temperature-mixture proposal, the
            weight is

              exp(trial_log_weights[j]) =
                  (p(y | c)**alpha / p(x | c)**alpha) * (q(x | c) / q(y | c)).

            The first ratio favors suffixes with higher target probability. The second corrects for how often the
            proposal generates each
            suffix: q(y | c) is the forward probability of proposing y from x;
            q(x | c) is the reverse probability of proposing x from y. Both
            directions use c, so q depends on c and the suffix being proposed.

            _log_weight adds/subtracts the four summed token-score arrays:
              trial_log_weights[j] = (
                  sum(target_ls_list[j]) - sum(curr_target_log_score)
                  + sum(curr_proposal_logprob)
                  - sum(_mixture_logprobs(prop_components[j]))
              ).
            The two target arrays already contain alpha * log p per token.
            The two proposal arrays sum to log q(x | c) and log q(y | c),
            using the mixture over all temperatures (not just the sampled one). The shared prefix's target score and
            target normalizer cancel.

            Keep these log weights unclipped: step 5 selects a candidate in proportion to exp(trial_log_weights), then
            step 6 uses all trial weights to decide acceptance. Only with num_tries=1 does acceptance reduce to the
            ordinary MH rule min(1, exp(trial_log_weights[0])).
            """
            trial_log_weights = [
                _log_weight(
                    target_ls_list[j], curr_target_log_score,
                    _mixture_logprobs(prop_components[j]), curr_proposal_logprob,
                )
                for j in range(num_tries)
            ]
            # 5. Algorithm 1, lines 13-14: for j in range(num_tries),
            # P(selected == j) = np.exp(trial_log_weights[j])
            #                    / sum(np.exp(trial_log_weights)).
            selected = _categorical_from_log_weights(trial_log_weights)
            # 6. Eq. (7), expressed with the stored log weights:
            # np.exp(log_acceptance) = min(
            #     1, sum(np.exp(trial_log_weights))
            #        / (1 + sum(np.exp(np.delete(trial_log_weights, selected))))).
            # The +1 is generated_output_seq's weight relative to itself; the other terms reuse unselected prop_seqs.
            # The helper works in log
            # space. With num_tries=1, log_acceptance=min(0, trial_log_weights[0]).
            log_acceptance = _log_acceptance_probability(trial_log_weights, selected)
            if verbose:
                print(
                    f"MH step {mi}/{mcmc_steps}, cut={cut_idx}, K={num_tries}, "
                    f"selected={selected}, accept_prob={math.exp(log_acceptance):.4f}",
                    flush=True,
                )
            # Comparing with math.exp(log_acceptance) handles a zero uniform draw.
            if np.random.rand() < math.exp(log_acceptance):
                # Replace generated_output_seq with prop_seqs[selected] and update both suffix-score caches (Algorithm
                # 1, line 16). A rejection leaves generated_output_seq and its scores intact.
                block_acceptance += 1
                generated_output_seq = prop_seqs[selected]
                component_logprobs_seq[:, offset:] = prop_components[selected]
                target_log_scores_seq[offset:] = target_ls_list[selected]

        num_acceptance.append(block_acceptance)
        total_attempt.append(mcmc_steps)
        # Keep a fixed horizon during refinement, then truncate the response at EOS. Prompt EOS tokens do not terminate
        # generation.
        eos_token_id = sampler_wrapper.tokenizer.eos_token_id
        if eos_token_id in generated_output_seq[context_len:]:
            eos_idx = generated_output_seq.index(eos_token_id, context_len)
            generated_output_seq = generated_output_seq[:eos_idx + 1]
            response_len = eos_idx + 1 - context_len
            component_logprobs_seq = component_logprobs_seq[:, :response_len]
            target_log_scores_seq = target_log_scores_seq[:response_len]
            break

    # Count one attempted transition per MTM step, regardless of num_tries.
    acceptance_ratio = sum(num_acceptance) / max(sum(total_attempt), 1)
    # proposal_logprobs_seq scores the response from context under the final temperatures; curr_proposal_logprob above
    # restarts the mixture at cut_idx.
    proposal_logprobs_seq = _mixture_logprobs(component_logprobs_seq)
    print(f"acceptances, attempts: {num_acceptance} / {total_attempt}")
    return generated_output_seq, proposal_logprobs_seq, target_log_scores_seq, acceptance_ratio
