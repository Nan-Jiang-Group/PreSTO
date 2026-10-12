"""Prefetch complete MultiTryMH proposal bundles with the custom vLLM backend.

Run a benchmark on a GPU node with:
    srun -p $PARTITION --quotatype=$QUOTATYPE --gres=gpu:1 --cpus-per-task=24 --time=03:00:00 \
      python -m power_sharpening.runners.vllm.run_subtree_prefetching_multi_try_mh \
      --dataset math500 --algorithm subtree_prefetching_multi_try_mh

The shared scheduler preserves the existing MultiTryMH selection and acceptance rule while speculating complete
candidate bundles on future chain branches. Ye and Lu's paper provides the multiproposal-prefetching precedent. The LLM
shared-prefix construction comes from the accompanying discussion; it retains MTM rather than replacing it with the
paper's proposal-cloud resampling.
"""

import logging
import time

from power_sharpening.backends.vllm.engine_patch import SamplingParams
from power_sharpening.backends.vllm.samplers.multi_try_mh import _draw_proposals
from power_sharpening.common.multi_try import DEFAULT_PROPOSAL_TEMPERATURES
from power_sharpening.common.prefetch_multi_try import sample_subtree_multi_try


logger = logging.getLogger("[subtree MultiTryMH vLLM]")


def subtree_prefetching_multi_try_sampling(
    mh_llm_model,
    prompt_ids,
    sampling_params,
    *,
    num_of_blocks=8,
    max_new_tokens=1024,
    mcmc_steps=10,
    num_tries=4,
    proposal_temperatures=DEFAULT_PROPOSAL_TEMPERATURES,
    scoring_batch_size=128,
    prefetch_budget=16,
    rank_fn="longest_path_first",
    stop_on_eos=True,
    print_tree=False,
    verbose=False,
    rng=None,
    draw_proposals=None,
):
    """Return response token IDs and exact engine-work/trajectory statistics.

    ``prompt_ids`` is one already-formatted prompt encoded as token IDs. The shared scheduler draws each suffix's
    mixture temperature and seed from an ordinary seeded NumPy stream before submitting the batch. Draws are
    reproducible with the same seed and schedule; changing ``prefetch_budget`` or ``rank_fn`` can change those choices
    and the resulting trajectory. ``prefetch_budget`` counts suffix requests, including discarded proposals; each
    complete bundle consumes ``num_tries`` requests. Temperatures stay fixed, cuts are uniform, and EOS is processed
    after block refinement.

    Only alpha, temperature, and seed are inherited from ``sampling_params``. The generation and exact token-ID
    rescoring requests use unconstrained softmax proposals, matching the standalone MultiTryMH implementation.
    ``draw_proposals`` replaces ``multi_try_mh._draw_proposals`` with a callable of the same signature, such as
    ``multi_try_mh_v2._draw_proposals_v2`` (engine-cached scores; needs a wrapper built with ``log_z_temperatures``).
    """
    if draw_proposals is None:
        draw_proposals = _draw_proposals
    temperatures = (
        [float(sampling_params.temperature)]
        if proposal_temperatures is None
        else list(proposal_temperatures)
    )
    temperatures = [float(value) for value in temperatures]
    alpha = float(sampling_params.alpha)
    seed = getattr(sampling_params, "seed", None)
    params = SamplingParams(
        alpha=alpha, temperature=temperatures[0], seed=seed,
        n=1, logprobs=1, ignore_eos=True, detokenize=False,
        top_k=-1, top_p=1.0, min_p=0.0,
    )
    sampling_started = time.perf_counter()
    logger.info(
        "Subtree MultiTryMH started: prompt_tokens=%d alpha=%g num_tries=%d proposal_temperatures=%s "
        "prefetch_budget=%d bundle_capacity=%d rank_fn=%s blocks=%d steps_per_block=%d "
        "max_new_tokens=%d scoring_batch_size=%d",
        len(prompt_ids), alpha, num_tries, temperatures, prefetch_budget, prefetch_budget // num_tries,
        rank_fn, num_of_blocks, mcmc_steps, max_new_tokens, scoring_batch_size,
    )

    def draw_batch(requests, component_temperatures):
        if not requests:
            return [], {"generation_calls": 0, "generated_tokens": 0}
        prefixes, lengths, row_temperatures, seeds = zip(*requests)
        work = {}
        tokens, base_scores, components = draw_proposals(
            mh_llm_model,
            list(prefixes),
            list(lengths),
            component_temperatures,
            params,
            scoring_batch_size,
            None,
            row_temperatures=list(row_temperatures),
            proposal_seeds=list(seeds),
            work_stats=work,
        )
        return list(zip(tokens, base_scores, components)), work

    (tokens, _, _), stats = sample_subtree_multi_try(
        list(prompt_ids), draw_batch,
        alpha=alpha, proposal_temperatures=temperatures,
        num_tries=num_tries, mcmc_steps=mcmc_steps,
        num_of_blocks=num_of_blocks, max_new_tokens=max_new_tokens,
        prefetch_budget=prefetch_budget, rank_fn=rank_fn,
        seed=seed, rng=rng,
        eos_token_id=mh_llm_model.tokenizer.eos_token_id,
        stop_on_eos=stop_on_eos, print_tree=print_tree, verbose=verbose,
    )
    logger.info("Subtree MultiTryMH finished: response_tokens=%d elapsed=%.3fs",
                len(tokens), time.perf_counter() - sampling_started)
    return tokens, stats
