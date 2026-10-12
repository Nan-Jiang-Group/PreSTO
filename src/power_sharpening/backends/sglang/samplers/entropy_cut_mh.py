"""Run EntropyCut Metropolis-Hastings with an SGLang model wrapper.

Example:
    uv run --project /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src \
      python -m power_sharpening.runners.sglang.run_entropy_cut_mh \
      --dataset math500 --algorithm entropy_cut_mh

The sampler keeps a fixed-length continuation state, chooses suffix cuts from positive jumps in the base model's
predictive entropy, and includes the state-dependent cut-law ratio in every Metropolis-Hastings decision.
"""

from __future__ import annotations

import numpy as np

from power_sharpening.backends.sglang.samplers.low_temp_proposal_sampler import (
    low_temp_proposal_sampling,
    require_suffix_entropies,
)
from power_sharpening.common.cut_distribution import (
    compute_entropy_cut_policy,
    entropy_cut_log_ratio,
)


def _extract_entropies(meta) -> np.ndarray:
    """Return token-aligned entropies from the SGLang proposal metadata."""
    return np.asarray(require_suffix_entropies(meta), dtype=np.float64)


def _validate_fixed_length(full_sequence, expected_total_length):
    """Fail clearly when a backend terminates before the requested length."""
    if len(full_sequence) != expected_total_length:
        raise RuntimeError(
            "EntropyCut uses a fixed-length state space, but SGLang returned "
            f"{len(full_sequence)} tokens instead of {expected_total_length}. "
            "The proposal decode must honor ignore_eos=True."
        )


def entropy_cut_mh_sampler(
    sampler_wrapper,
    context,
    mcmc_steps,
    max_new_tokens=3072,
    num_of_blocks=16,
    cut_power=4.0,
    entropy_mode="topk",
    entropy_top_k=64,
    verbose=False,
    rng=None,
):
    """Sample a fixed-length continuation with stagewise EntropyCut MH.

    Args:
        sampler_wrapper: SGLang wrapper with a fixed ``alpha`` and proposal ``temperature=1/alpha``.
        context: prompt token ids. The returned sequence includes this prefix.
        mcmc_steps: MH transitions per stage.
        max_new_tokens: fixed final continuation length.
        num_of_blocks: number of equal stagewise growth blocks.
        cut_power: EntropyCut power beta.
        entropy_mode: ``topk`` for the public-SGLang approximation or ``exact`` for a future server scorer that returns
            exact entropy scalars.
        entropy_top_k: top-K size used by ``topk`` mode.
        verbose: print per-stage transition diagnostics.
        rng: NumPy random-number generator.

    Returns:
        ``(full_token_ids, stats)`` where ``stats`` is a JSON-serializable mapping containing aggregate and
        per-transition diagnostics.
    """
    context = list(context)
    if rng is None:
        rng = np.random.default_rng()

    context_len = len(context)
    block_size = max_new_tokens // num_of_blocks
    entropy_request_k = entropy_top_k if entropy_mode == "topk" else None

    generated_tokens = context.copy()
    proposal_logprobs = np.empty(0, dtype=np.float64)
    target_log_scores = np.empty(0, dtype=np.float64)
    entropies = np.empty(0, dtype=np.float64)

    attempts = 0
    acceptances = 0
    cut_positions = []
    cut_fractions = []
    transitions = []
    entropy_estimator = None

    for block_index in range(num_of_blocks):
        target_total_length = context_len + (block_index + 1) * block_size
        (
            generated_tokens,
            suffix_proposal_logprobs,
            suffix_target_log_scores,
            meta,
        ) = low_temp_proposal_sampling(
            sampler_wrapper,
            context=generated_tokens,
            seq_len=target_total_length,
            entropy_top_k=entropy_request_k,
            ignore_eos=True,
            verbose=verbose,
        )
        _validate_fixed_length(generated_tokens, target_total_length)
        suffix_proposal_logprobs = np.asarray(
            suffix_proposal_logprobs,
            dtype=np.float64,
        )
        suffix_target_log_scores = np.asarray(
            suffix_target_log_scores,
            dtype=np.float64,
        )
        suffix_entropies = _extract_entropies(meta)
        proposal_logprobs = np.concatenate(
            (proposal_logprobs, suffix_proposal_logprobs)
        )
        target_log_scores = np.concatenate(
            (target_log_scores, suffix_target_log_scores)
        )
        entropies = np.concatenate((entropies, suffix_entropies))
        entropy_estimator = meta.get("entropy_mode") or entropy_mode

        for step_index in range(mcmc_steps):
            current_cut_probabilities = compute_entropy_cut_policy(
                entropies,
                beta=cut_power,
            )
            policy_offset = int(
                rng.choice(
                    current_cut_probabilities.size,
                    p=current_cut_probabilities,
                )
            )
            cut_offset = policy_offset + 1
            cut_index = context_len + cut_offset
            cut_positions.append(cut_offset)
            cut_fractions.append(
                cut_offset / max(1, len(entropies) - 1)
            )

            (
                proposed_tokens,
                proposed_suffix_logprobs,
                proposed_suffix_target_scores,
                proposed_meta,
            ) = low_temp_proposal_sampling(
                sampler_wrapper,
                context=generated_tokens[:cut_index],
                seq_len=len(generated_tokens),
                entropy_top_k=entropy_request_k,
                ignore_eos=True,
                verbose=verbose,
            )
            _validate_fixed_length(
                proposed_tokens,
                len(generated_tokens),
            )
            proposed_suffix_logprobs = np.asarray(
                proposed_suffix_logprobs,
                dtype=np.float64,
            )
            proposed_suffix_target_scores = np.asarray(
                proposed_suffix_target_scores,
                dtype=np.float64,
            )
            proposed_suffix_entropies = _extract_entropies(proposed_meta)

            proposed_entropies = np.concatenate(
                (entropies[:cut_offset], proposed_suffix_entropies)
            )
            proposed_cut_probabilities = compute_entropy_cut_policy(
                proposed_entropies,
                beta=cut_power,
            )

            current_suffix_logprobs = proposal_logprobs[cut_offset:]
            current_suffix_target_scores = target_log_scores[cut_offset:]
            target_term = float(
                np.sum(proposed_suffix_target_scores)
                - np.sum(current_suffix_target_scores)
            )
            proposal_term = float(
                np.sum(current_suffix_logprobs)
                - np.sum(proposed_suffix_logprobs)
            )
            cut_term = float(
                entropy_cut_log_ratio(
                    current_cut_probabilities,
                    proposed_cut_probabilities,
                    cut_offset=policy_offset,
                )
            )
            log_acceptance = target_term + proposal_term + cut_term
            accepted = rng.random() < np.exp(min(0.0, log_acceptance))
            attempts += 1

            transition = {
                "block": block_index,
                "step": step_index,
                "cut_offset": cut_offset,
                "cut_fraction": cut_fractions[-1],
                "cut_token_index": cut_index,
                "current_log_cut_probability": float(
                    np.log(current_cut_probabilities[policy_offset])
                ),
                "proposed_log_cut_probability": float(
                    np.log(proposed_cut_probabilities[policy_offset])
                ),
                "target_term": target_term,
                "proposal_term": proposal_term,
                "cut_term": cut_term,
                "log_acceptance": float(log_acceptance),
                "accepted": bool(accepted),
            }
            transitions.append(transition)

            if verbose:
                print(
                    f"block={block_index} step={step_index} "
                    f"cut={cut_offset} logA={log_acceptance:.6f} "
                    f"accepted={accepted}",
                    flush=True,
                )

            if accepted:
                acceptances += 1
                generated_tokens = list(proposed_tokens)
                proposal_logprobs[cut_offset:] = proposed_suffix_logprobs
                target_log_scores[cut_offset:] = proposed_suffix_target_scores
                entropies[cut_offset:] = proposed_suffix_entropies

    stats = {
        "attempts": attempts,
        "acceptances": acceptances,
        "acceptance_rate": (
            float(acceptances / attempts) if attempts else 0.0
        ),
        "mean_cut_position": (
            float(np.mean(cut_positions)) if cut_positions else None
        ),
        "mean_cut_fraction": (
            float(np.mean(cut_fractions)) if cut_fractions else None
        ),
        "mean_token_entropy": (
            float(np.mean(entropies)) if entropies.size else None
        ),
        "output_tokens": int(entropies.size),
        "sequence_base_logprob": float(
            np.sum(target_log_scores) / sampler_wrapper.alpha
        ),
        "sequence_target_log_score": float(np.sum(target_log_scores)),
        "sequence_proposal_logprob": float(np.sum(proposal_logprobs)),
        "entropy_estimator": entropy_estimator,
        "entropy_top_k": entropy_top_k,
        "cut_power": float(cut_power),
        "num_blocks": int(num_of_blocks),
        "mcmc_steps": int(mcmc_steps),
        "max_new_tokens": int(max_new_tokens),
        "transitions": transitions,
    }
    return generated_tokens, stats


__all__ = ["entropy_cut_mh_sampler"]
