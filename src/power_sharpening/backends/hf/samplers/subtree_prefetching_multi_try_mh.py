"""Subtree-prefetched Multi-Try MH using the existing HuggingFace proposals.

Run on a GPU with:
    srun -p $PARTITION --quotatype=$QUOTATYPE --gres=gpu:1 \
      --cpus-per-task=24 --time=03:00:00 \
      python -m power_sharpening.runners.hf.run_subtree_prefetching_multi_try_mh \
      --dataset math500 --algorithm subtree_prefetching_multi_try_mh

The common engine schedules complete Multi-Try candidate bundles and evaluates the same selection and acceptance rule as
the sequential sampler. This adapter only generates and scores suffixes, using the wrapper's fixed target exponent. The
shared engine draws from an ordinary seeded NumPy stream. Generation uses HF's global torch random stream because HF
generate does not support a generator per batch row. The runner seeds both streams. Changing the prefetch budget or rank
can change draws and need not preserve the trajectory.
"""

from collections import defaultdict
from functools import partial
import math

import numpy as np
import torch

from power_sharpening.common.multi_try import DEFAULT_PROPOSAL_TEMPERATURES
from power_sharpening.common.prefetch_multi_try import sample_subtree_multi_try

from .low_temp_sampler import batched_low_temp_proposal_sampling


def _model_context_length(model, requested=None):
    """Honor both the loaded model's context window and a smaller caller cap."""
    config = getattr(model, "config", None)
    for field in (
        "max_position_embeddings", "max_sequence_length", "seq_length", "n_positions",
    ):
        value = getattr(config, field, None)
        if isinstance(value, (int, float)) and 0 < value < 10**9:
            return int(value) if requested is None else min(int(value), requested)
    return requested


@torch.inference_mode()
def _draw_batch(
    sampler_wrapper,
    requests: list[tuple],
    temperatures: list[float],
    *,
    context_length=None,
) -> tuple[list[tuple], dict]:
    """Batch variable prefixes and preserve the original request order.

    Requests are (prefix, length, temperature, seed); return (proposals, work), where each proposal is (token_ids,
    base_logprobs, component_logprobs). The existing HF helper decodes each row for the longest suffix, then trims
    longer-prefix rows to their requested horizon. Count that extra decoding in ``generated_tokens``. If padding plus
    decoding would exceed the loaded context window, split by prefix length so the helper never overflows it. Scores at
    all mixture temperatures reuse raw generation logits and require no additional model calls.
    """
    if not requests:
        return [], {"generation_calls": 0, "generated_tokens": 0}
    if context_length is None:
        context_length = _model_context_length(sampler_wrapper.base_model)
    by_horizon = defaultdict(list)
    for index, request in enumerate(requests):
        prefix, length, _, _ = request
        horizon = len(prefix) + length
        by_horizon[horizon].append((index, request))

    groups = []
    for horizon, items in by_horizon.items():
        padded_horizon = (
            max(len(prefix) for _, (prefix, _, _, _) in items)
            + max(length for _, (_, length, _, _) in items)
        )
        if context_length is None or padded_horizon <= context_length:
            groups.append((horizon, items))
        else:
            by_prefix_length = defaultdict(list)
            for index, request in items:
                prefix, _, _, _ = request
                by_prefix_length[len(prefix)].append((index, request))
            groups.extend((horizon, group) for group in by_prefix_length.values())

    proposals = [None] * len(requests)
    generated_tokens = 0
    for horizon, group in groups:
        sequences, _, target_scores, component_scores = batched_low_temp_proposal_sampling(
            sampler_wrapper,
            contexts=[prefix for _, (prefix, _, _, _) in group],
            seq_len=horizon,
            use_cache=True,
            ignore_eos=True,
            row_temperatures=[temperature for _, (_, _, temperature, _) in group],
            component_temperatures=temperatures,
        )
        generated_tokens += len(group) * max(length for _, (_, length, _, _) in group)
        for row, (index, request) in enumerate(group):
            prefix, _, _, _ = request
            sequence = list(sequences[row])
            token_ids = sequence[len(prefix):]
            base = np.asarray(target_scores[row], dtype=np.float64) / sampler_wrapper.alpha
            components = np.asarray(component_scores[row], dtype=np.float64)
            proposals[index] = (token_ids, base.tolist(), components)
    return proposals, {"generation_calls": len(groups), "generated_tokens": generated_tokens}


def subtree_prefetching_multi_try_sampling(
    sampler_wrapper,
    context,
    mcmc_steps=10,
    max_new_tokens=1024,
    num_of_blocks=8,
    num_tries=4,
    prefetch_budget=16,
    rank_fn="longest_path_first",
    proposal_temperatures=DEFAULT_PROPOSAL_TEMPERATURES,
    stop_on_eos=True,
    print_tree=False,
    verbose=False,
    seed=None,
    rng=None,
    max_model_len=None,
):
    """Return ``(full_prompt_and_response_tokens, prefetch_stats)``.

    ``prefetch_budget`` counts suffix requests; only complete ``num_tries`` bundles are scheduled. This version supports
    uniform cuts and a fixed scalar or whole-suffix temperature mixture. The target exponent stays equal to
    ``sampler_wrapper.alpha`` throughout sampling. EOS truncation is handled by the common engine after fixed-horizon
    refinement. Disabling ``stop_on_eos`` evaluates every block; the returned response is still trimmed at its first
    EOS, as in the existing subtree sampler.
    """
    temperatures = (
        [float(sampler_wrapper.init_temperature)]
        if proposal_temperatures is None else proposal_temperatures
    )
    context_length = _model_context_length(sampler_wrapper.base_model, max_model_len)
    
    (tokens, _, _), stats = sample_subtree_multi_try(
        list(context),
        partial(_draw_batch, sampler_wrapper, context_length=context_length),
        alpha=sampler_wrapper.alpha,
        proposal_temperatures=temperatures,
        num_tries=num_tries,
        mcmc_steps=mcmc_steps,
        num_of_blocks=num_of_blocks,
        max_new_tokens=max_new_tokens,
        prefetch_budget=prefetch_budget,
        rank_fn=rank_fn,
        seed=seed,
        rng=rng,
        eos_token_id=sampler_wrapper.tokenizer.eos_token_id,
        stop_on_eos=stop_on_eos,
        print_tree=print_tree,
        verbose=verbose,
    )
    return list(context) + tokens, stats
