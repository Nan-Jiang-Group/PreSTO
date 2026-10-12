"""Multiple-Try Metropolis v2: REPPS-style proposal scoring for the custom vLLM backend.

Use with a wrapper built for the same temperatures:
    wrapper = vLLM_Wrapper(model, engine_type="custom", log_z_temperatures=[0.25, 0.5, 1.0])
    multi_try_mcmc_power_sampler_v2(wrapper, prompts, sampling_params, num_tries=4,
                                    proposal_temperatures=[0.25, 0.5, 1.0])
or run a benchmark on a GPU node through:
    srun -p $PARTITION --quotatype=$QUOTATYPE --gres=gpu:1 --cpus-per-task=24 --time=03:00:00 \
      python -m power_sharpening.runners.vllm.run_multi_try_mh_v2 --dataset math500 --model_str qwen

Every MTM step (block extension, uniform cut, candidate weights, selection, and acceptance) is exactly
``multi_try_mh.multi_try_mcmc_power_sampler``. Only proposal generation differs, following REPPS
(replica-to-reason, ``sampler_core_vllm.ModelWrapper.sample_continuation_batch_exact``):

* One batched ``llm.generate`` call sends every candidate row with its own proposal temperature.
* While decoding, the patched sampler also returns logZ_T[t] = logsumexp(log p(. | x_<t) / T) over the full vocabulary
  for every configured temperature (``engine_patch/log_normalizers.py``).
* Each suffix token's score under every mixture component is then cached arithmetic,

      log q_T(y_t) = log p(y_t) / T - logZ_T[t],

  so v1's (M - 1) * num_tries * suffix_length one-token ``logprob_token_ids`` scoring requests per step disappear.

The identity holds because the MTM sampler already samples from plain softmax(logits / T): no truncation, penalties, or
constrained decoding. All M components, including the sampled one, come from the same formula, so forward and reverse
proposal scores are computed identically.
"""

from collections import Counter
import logging
import time

import numpy as np
from vllm.inputs import TokensPrompt

from power_sharpening.backends.vllm.engine_patch.log_normalizers import tempered_logprobs
from power_sharpening.backends.vllm.engine_patch.sampling_params import _copy_sampling_params
from power_sharpening.backends.vllm.samplers.multi_try_mh import (
    _extract_logprobs,
    _single_output,
    multi_try_mcmc_power_sampler,
)
from power_sharpening.common.multi_try import DEFAULT_PROPOSAL_TEMPERATURES

logger = logging.getLogger("[Multiple-Try Metropolis v2]")


def _log_z_columns(model, temperatures) -> list[int]:
    """Map each proposal temperature to its column in the engine's logZ rows."""
    engine_temperatures = getattr(model, "log_z_temperatures", None)
    if not engine_temperatures:
        raise ValueError(
            "MultiTryMH v2 needs a wrapper built with log_z_temperatures, e.g. "
            "vLLM_Wrapper(model, engine_type='custom', log_z_temperatures=proposal_temperatures)"
        )
    engine_temperatures = [float(value) for value in engine_temperatures]
    missing = [value for value in temperatures if value not in engine_temperatures]
    if missing:
        raise ValueError(
            f"proposal temperatures {missing} are not in the engine's log_z_temperatures {engine_temperatures}"
        )
    return [engine_temperatures.index(value) for value in temperatures]


def _draw_proposals_v2(
    model, prefixes, lengths, temperatures, sampling_params, scoring_batch_size,
    seed_rng, *, row_temperatures=None, proposal_seeds=None, work_stats=None,
):
    """Drop-in replacement for ``multi_try_mh._draw_proposals`` that needs no scoring requests.

    ``row_temperatures[row]`` is drawn uniformly from ``temperatures`` and stays fixed while generating
    ``token_ids[row]`` after ``prefixes[row]``. Returns the same ``(token_ids, base_logprobs, component_scores)``
    triple; ``component_scores[row]`` has shape (number of temperatures, that row's suffix length) and is computed from
    the engine's logZ rows. ``scoring_batch_size`` is accepted for signature compatibility and unused.
    """
    columns = _log_z_columns(model, temperatures)
    if row_temperatures is None:
        row_temperatures = (
            [temperatures[0]] * len(prefixes) if len(temperatures) == 1
            else np.random.choice(temperatures, size=len(prefixes)).tolist()
        )
    if proposal_seeds is None:
        proposal_seeds = [
            None if seed_rng is None else int(seed_rng.integers(0, 2**63 - 1))
            for _ in prefixes
        ]
    if not prefixes:
        return [], [], []
    params = [
        _copy_sampling_params(
            sampling_params, n=1, max_tokens=length, temperature=temperature,
            # Cloned identical seeds would couple otherwise independent trials.
            seed=seed,
        )
        for length, temperature, seed in zip(lengths, row_temperatures, proposal_seeds)
    ]
    logger.info(
        "Suffix generation started: requests=%d suffix_tokens=%s requests_by_temperature=%s",
        len(prefixes), f"{min(lengths)}..{max(lengths)}", dict(Counter(row_temperatures)),
    )
    generation_started = time.perf_counter()
    # One batched call; each row carries its own proposal temperature, as REPPS batches its replica ladder.
    requests = model.llm.generate(
        [TokensPrompt(prompt_token_ids=prefix) for prefix in prefixes],
        sampling_params=params, use_tqdm=False,
    )
    if work_stats is not None:
        work_stats["generation_calls"] = work_stats.get("generation_calls", 0) + 1
    token_ids, base_logprobs, component_scores = [], [], []
    # Largest disagreement between the formula and vLLM's own score at the sampled temperature; float32 roundoff only.
    max_mismatch = 0.0
    for request, temperature in zip(requests, row_temperatures):
        output = _single_output(request)
        tokens = list(output.token_ids)
        log_z = getattr(output, "log_z_by_temperature", None)
        if log_z is None or len(log_z) != len(tokens):
            raise RuntimeError("custom vLLM output must provide one logZ row per generated token")
        base = _extract_logprobs(output.power_logprobs, tokens)
        scores = tempered_logprobs(
            base, np.asarray(log_z, dtype=np.float64)[:, columns].reshape(len(tokens), len(columns)), temperatures,
        )
        if tokens:
            sampled = scores[temperatures.index(temperature)]
            engine = np.asarray(_extract_logprobs(output.logprobs, tokens))
            max_mismatch = max(max_mismatch, float(np.max(np.abs(sampled - engine))))
        token_ids.append(tokens)
        base_logprobs.append(base)
        component_scores.append(scores)
    del requests
    generated_tokens = sum(map(len, token_ids))
    if work_stats is not None:
        work_stats["generated_tokens"] = work_stats.get("generated_tokens", 0) + generated_tokens
    logger.info(
        "Suffix generation finished: requests=%d generated_tokens=%d elapsed=%.3fs "
        "scoring_requests=0 max_sampled_logq_mismatch=%.3g",
        len(token_ids), generated_tokens, time.perf_counter() - generation_started, max_mismatch,
    )
    return token_ids, base_logprobs, component_scores


def multi_try_mcmc_power_sampler_v2(
    mh_llm_model,
    prompts: str | list[str],
    sampling_params,
    num_of_blocks: int = 8,
    max_new_tokens: int = 3_072,
    mcmc_steps: int = 10,
    num_tries: int = 4,
    proposal_temperatures=DEFAULT_PROPOSAL_TEMPERATURES,
) -> list[str]:
    """Run ``multi_try_mcmc_power_sampler`` with engine-cached proposal scores.

    ``proposal_temperatures`` must be a nonempty list of positive finite values present in the wrapper's
    ``log_z_temperatures``; repeated entries increase that component's mixture mass, as in v1. Temperature schedules
    are unsupported because the engine's temperature list is fixed when it is built. Returns one decoded continuation
    per input prompt, in input order.
    """
    if proposal_temperatures is None:
        raise ValueError("MultiTryMH v2 requires an explicit proposal_temperatures list")
    temperatures = np.asarray(proposal_temperatures, dtype=np.float64).tolist()
    if not temperatures or any(not np.isfinite(value) or value <= 0 for value in temperatures):
        raise ValueError("proposal_temperatures must be a nonempty list of positive finite numbers")
    # Fail before generating anything when the engine cannot score these temperatures.
    _log_z_columns(mh_llm_model, temperatures)
    return multi_try_mcmc_power_sampler(
        mh_llm_model,
        prompts,
        sampling_params,
        num_of_blocks=num_of_blocks,
        max_new_tokens=max_new_tokens,
        mcmc_steps=mcmc_steps,
        num_tries=num_tries,
        temperature_scheduler=None,
        proposal_temperatures=temperatures,
        draw_proposals=_draw_proposals_v2,
    )


__all__ = ["multi_try_mcmc_power_sampler_v2"]
