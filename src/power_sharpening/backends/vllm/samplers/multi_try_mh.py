"""Paper-style Multiple-Try Metropolis for the custom vLLM backend.

Use ``multi_try_mcmc_power_sampler(wrapper, prompts, sampling_params,
num_tries=5, proposal_temperatures=[0.25, 0.5, 1.0])`` with a custom
``vLLM_Wrapper``. Run a GPU driver through:
    srun -p $PARTITION --quotatype=$QUOTATYPE --gres=gpu:1 \
      --cpus-per-task=24 --time=03:00:00 python /absolute/path/to/driver.py

Implements the MTM steps in Algorithm 1 of https://arxiv.org/html/2606.09926v1. For prompt i, the unnormalized power
target in Eq. (2) is ``np.exp(alpha * sum(base_scores[i]))``. This variant refines every block and uses uniform cuts
over all generated tokens.

Code names used below:
    generated[i]: current response, following the fixed prompt_tokens[i]; base_scores[i]: base-model log probability for
    each token in generated[i];
    block_size: number of tokens appended before each block's refinement;
    cuts[i]: first response token to replace for prompt i;
    num_tries: number of candidate suffixes drawn for each prompt;
    start: first candidate's row in the flattened generation batch;
    weights: this prompt's log MH weights from Eq. (6);
    selected: index in range(num_tries), drawn with probabilities proportional to np.exp(weights);
    chosen: start + selected, the accepted candidate's batch row;
    log_acceptance: log acceptance probability from Eq. (7).
The unselected candidates are reused; no extra reference suffixes are drawn.

Like the HF implementation, independently draw one temperature per candidate suffix and average whole-suffix
probabilities over ``temperatures`` in both directions of the MH ratio. vLLM returns only the sampled component's scores
during generation. Additional one-token requests use ``logprob_token_ids`` to score observed tokens under the other
temperatures, with bounded batches and no vocabulary transfer. Their sampled tokens are discarded. Component scores are
cached for subsequent cuts. Requires the repository's pinned custom vLLM backend.
"""

from collections import Counter
import itertools
import logging
import math
import random
import time

import numpy as np
from vllm.inputs import TokensPrompt

from power_sharpening.common.multi_try import (
    DEFAULT_PROPOSAL_TEMPERATURES,
    categorical_from_log_weights as _categorical_from_log_weights,
    log_acceptance_probability as _log_acceptance_probability,
    mixture_logprobs as _mixture_logprobs,
    mtm_edge_probabilities as _mtm_edge_probabilities,
)
from power_sharpening.backends.vllm.engine_patch import SamplingParams
from power_sharpening.backends.vllm.engine_patch.sampling_params import _copy_sampling_params
from power_sharpening.backends.vllm.engine_patch.entropy import suffix_log_acceptance

logger = logging.getLogger("[Multiple-Try Metropolis]")


def _extract_logprobs(logprobs, token_ids) -> list[float]:
    """Extract the requested token's score, independent of dictionary order."""
    return [float(entries[token_id].logprob) for entries, token_id in zip(logprobs, token_ids)]


def compute_acceptance_ratio(
    proposed_logprobs: list[float],
    curr_logprobs: list[float],
    proposed_power_logprobs: list[float],
    curr_power_logprobs: list[float],
    alpha: float,
) -> float:
    """Return the log MH weight in paper Eq. (6), using these arguments:

        alpha * (sum(proposed_power_logprobs) - sum(curr_power_logprobs))
        + sum(curr_logprobs) - sum(proposed_logprobs)

    Both suffixes follow the same retained token prefix, whose target scores cancel. The forward proposal generates the
    proposed suffix from that prefix; the reverse proposal generates the current suffix from it.

    ``proposed_logprobs`` and ``curr_logprobs`` hold per-token proposal log probabilities. ``proposed_power_logprobs``
    and ``curr_power_logprobs`` hold unscaled base-model log probabilities despite their historical names. The helper
    applies alpha to the base-score difference once. Its return value is a log MH weight; Eq. (7) later gives the
    acceptance probability.
    """
    return suffix_log_acceptance(
        proposed_logq=proposed_logprobs,
        current_logq=curr_logprobs,
        proposed_logp=proposed_power_logprobs,
        current_logp=curr_power_logprobs,
        alpha=alpha,
    )


def _single_output(request):
    return request.outputs[0]


def _score_components(
    model, sequences, starts, temperatures, sampling_params, batch_size,
    known_scores=None, *, work_stats=None,
):
    """Score suffix tokens at all temperatures using exact token-ID queries.

    ``known_scores`` optionally supplies each suffix's generation temperature and per-token proposal log probabilities,
    avoiding duplicate scoring. Each query scores ``sequence[position]`` given ``sequence[:position]`` at
    ``temperature``; the query's generated token is not used in the chain. ``work_stats`` optionally accumulates actual
    scoring calls and requests.
    """
    # Each row stores (temperature component, suffix token). Here sequences include their prompts, and starts gives the
    # absolute first suffix index.
    scores = [np.empty((len(temperatures), len(seq) - start), dtype=np.float64)
              for seq, start in zip(sequences, starts)]
    total_requests = sum(
        (len(sequence) - start) * sum(
            known_scores is None or known_scores[row] is None or temperature != known_scores[row][0]
            for temperature in temperatures
        )
        for row, (sequence, start) in enumerate(zip(sequences, starts))
    )
    total_calls = (total_requests + batch_size - 1) // batch_size
    scoring_started = time.perf_counter()
    scoring_calls = scoring_requests = 0
    if total_requests:
        logger.info(
            "Token scoring started: suffixes=%d temperatures=%s requests=%d batches=%d scoring_batch_size=%d",
            len(sequences), temperatures, total_requests, total_calls, batch_size,
        )
    else:
        logger.info("Token scoring: all component scores reused; no extra engine calls")

    def requests():
        for row, (sequence, start) in enumerate(zip(sequences, starts)):
            known = None if known_scores is None else known_scores[row]
            for component, temperature in enumerate(temperatures):
                if known is not None and temperature == known[0]:
                    # known[1] already holds the scores at this temperature. This also reuses scores for repeated
                    # temperature entries.
                    scores[row][component] = known[1]
                    continue
                for position in range(start, len(sequence)):
                    # Score sequence[position] given sequence[:position]. logprob_token_ids returns its score even when
                    # vLLM samples a different token; the sampled token is discarded.
                    params = _copy_sampling_params(
                        sampling_params, n=1, max_tokens=1, temperature=temperature,
                        logprobs=1, logprob_token_ids=[sequence[position]], seed=0,
                    )
                    yield (
                        row, component, position - start, sequence[position],
                        TokensPrompt(prompt_token_ids=sequence[:position]), params,
                    )

    # Bound extra scoring work per engine call instead of submitting every candidate/temperature/token combination at
    # once.
    pending = requests()
    while batch := list(itertools.islice(pending, batch_size)):
        outputs = model.llm.generate(
            [item[4] for item in batch],
            sampling_params=[item[5] for item in batch], use_tqdm=False,
        )
        scoring_calls += 1
        scoring_requests += len(batch)
        if work_stats is not None:
            work_stats["scoring_calls"] = work_stats.get("scoring_calls", 0) + 1
            work_stats["scoring_requests"] = work_stats.get("scoring_requests", 0) + len(batch)
        for (row, component, position, token_id, _, _), request in zip(batch, outputs):
            output = _single_output(request)
            scores[row][component, position] = _extract_logprobs(
                output.logprobs, [token_id]
            )[0]
        if scoring_calls % 10 == 0 and scoring_calls < total_calls:
            logger.info(
                "Token scoring progress: batches=%d/%d requests=%d/%d elapsed=%.3fs",
                scoring_calls, total_calls, scoring_requests, total_requests, time.perf_counter() - scoring_started,
            )
    if total_requests:
        logger.info(
            "Token scoring finished: calls=%d requests=%d elapsed=%.3fs",
            scoring_calls, scoring_requests, time.perf_counter() - scoring_started,
        )
    return scores


def _draw_proposals(
    model, prefixes, lengths, temperatures, sampling_params, scoring_batch_size,
    seed_rng, *, row_temperatures=None, proposal_seeds=None, work_stats=None,
):
    """Draw one suffix per row and score the temperature-list extension.

    ``row_temperatures[row]`` is drawn uniformly from ``temperatures`` and stays fixed while generating
    ``token_ids[row]`` after ``prefixes[row]``.
    Eq. (6) uses the mixture's whole-suffix log probability:

        sum(_mixture_logprobs(component_scores[row])) =
            np.logaddexp.reduce(np.sum(component_scores[row], axis=1))
            - np.log(len(temperatures))

    Sum across suffix tokens before mixing temperatures. The helper evaluates this identity stably in log space.
    Generation supplies the sampled component's scores; extra queries score the other components.

    Prefetch callers may supply ``row_temperatures`` and ``proposal_seeds`` to fix each request's randomness before
    scheduling it. ``work_stats`` counts generation/scoring calls, scoring requests, and generated suffix tokens; the
    one-token scoring completions are tracked only by scoring_requests.
    """
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
    prefix_min, prefix_max = min(map(len, prefixes)), max(map(len, prefixes))
    suffix_min, suffix_max = min(lengths), max(lengths)
    logger.info(
        "Suffix generation started: requests=%d prefix_tokens=%s suffix_tokens=%s "
        "requests_by_temperature=%s",
        len(prefixes),
        prefix_min if prefix_min == prefix_max else f"{prefix_min}..{prefix_max}",
        suffix_min if suffix_min == suffix_max else f"{suffix_min}..{suffix_max}",
        dict(Counter(row_temperatures)),
    )
    generation_started = time.perf_counter()
    # Preserve the retained prefix as token IDs: decoding and re-encoding it could change the prefix on which both
    # directions of Eq. (6) condition.
    requests = model.llm.generate(
        [TokensPrompt(prompt_token_ids=prefix) for prefix in prefixes],
        sampling_params=params, use_tqdm=False,
    )
    if work_stats is not None:
        work_stats["generation_calls"] = work_stats.get("generation_calls", 0) + 1
    token_ids, base_logprobs, known_scores = [], [], []
    for request, temperature in zip(requests, row_temperatures):
        output = _single_output(request)
        tokens = list(output.token_ids)
        token_ids.append(tokens)
        base_logprobs.append(_extract_logprobs(output.power_logprobs, tokens))
        known_scores.append((temperature, _extract_logprobs(output.logprobs, tokens)))
    logger.info(
        "Suffix generation finished: requests=%d generated_tokens=%d elapsed=%.3fs",
        len(token_ids), sum(map(len, token_ids)), time.perf_counter() - generation_started,
    )
    if work_stats is not None:
        work_stats["generated_tokens"] = work_stats.get("generated_tokens", 0) + sum(map(len, token_ids))
    # Release generation outputs before the scoring batches allocate more.
    del requests
    component_scores = _score_components(
        model, [prefix + tokens for prefix, tokens in zip(prefixes, token_ids)],
        [len(prefix) for prefix in prefixes], temperatures, sampling_params,
        scoring_batch_size, known_scores=known_scores, work_stats=work_stats,
    )
    # All returned tokens/scores cover only new suffixes. Each component-score matrix has shape (number of temperatures,
    # that row's suffix length).
    return token_ids, base_logprobs, component_scores


def multi_try_mcmc_power_sampler(
    mh_llm_model,
    prompts: str | list[str],
    sampling_params: SamplingParams,
    num_of_blocks: int = 8,
    max_new_tokens: int = 3_072,
    mcmc_steps: int = 10,
    num_tries: int = 4,
    temperature_scheduler=None,
    proposal_temperatures=DEFAULT_PROPOSAL_TEMPERATURES,
    scoring_batch_size: int = 128,
    *,
    draw_proposals=None,
) -> list[str]:
    """Generate continuations with the paper's MTM rule and uniform cuts.

    ``num_tries`` sets the candidate count independently of the number of entries in ``proposal_temperatures``.
    ``proposal_temperatures`` is a nonempty list of finite positive values; each candidate draws a uniform list entry
    once for its entire suffix. Repeated entries increase that component's mixture mass. It defaults to
    ``DEFAULT_PROPOSAL_TEMPERATURES``. Use no scheduler (or a ConstantScheduler) with an explicit list. Passing ``None``
    instead of a list selects the scalar proposal temperature, which can be scheduled; current-state scores are
    refreshed when it changes, while the target exponent ``sampling_params.alpha`` stays fixed. ``scoring_batch_size``
    bounds extra token-scoring requests per call. ``draw_proposals`` replaces ``_draw_proposals`` with a callable of the
    same signature and return value (see ``multi_try_mh_v2``).

    Proposals use plain softmax(logits / temperature). Sampling truncations, penalties, stop strings, and constrained
    decoding are disabled locally. All steps within a block share its fixed horizon; truncate at the first response EOS
    only after refinement. Returns one decoded continuation per input prompt, in input order. Only alpha, temperature,
    and seed are inherited
    from the caller's sampling parameters.
    """
    if isinstance(prompts, str):
        prompts = [prompts]
    if draw_proposals is None:
        draw_proposals = _draw_proposals

    alpha = float(sampling_params.alpha)
    temperatures = (
        [float(sampling_params.temperature)]
        if proposal_temperatures is None
        else np.asarray(proposal_temperatures, dtype=np.float64).tolist()
    )

    seed = sampling_params.seed
    seed_rng = None if seed is None else np.random.default_rng(seed)
    # Use softmax(logits / temperature), matching the probabilities scored below. Ignore EOS while refining so all
    # candidates share the block's horizon. Fresh parameters also discard cached stop/constraint state on an input
    # object that has previously been submitted to the engine.
    sampling_params = SamplingParams(
        alpha=alpha, temperature=temperatures[0], seed=seed,
        n=1, logprobs=1, ignore_eos=True, detokenize=False,
        top_k=-1, top_p=1.0, min_p=0.0,
    )
    prompt_tokens = [list(mh_llm_model.tokenizer.encode(prompt)) for prompt in prompts]
    # Keep one independent chain per prompt. These tokens and score arrays exclude the prompt; component_scores[i] is
    # (temperature, response token). base_scores[i] stores unscaled base-model log probabilities separately
    # from component_scores[i], which permits fresh mixtures at later cuts.
    generated = [[] for _ in prompts]
    base_scores = [[] for _ in prompts]
    component_scores = [np.empty((len(temperatures), 0)) for _ in prompts]
    active = set(range(len(prompts)))
    block_size = max_new_tokens // num_of_blocks
    verbose = getattr(mh_llm_model, "verbose", False)
    sampling_started = time.perf_counter()
    total_accepted = total_attempted = 0
    logger.info(
        "MultiTryMH started: prompts=%d alpha=%g num_tries=%d proposal_temperatures=%s "
        "blocks=%d tokens_per_block=%d steps_per_block=%d scoring_batch_size=%d",
        len(prompts), alpha, num_tries, temperatures, num_of_blocks, block_size, mcmc_steps, scoring_batch_size,
    )

    def set_temperature(temperature, active_list):
        """Refresh component_scores for Eq. (6) when temperature changes."""
        nonlocal temperatures
        if temperatures == [temperature]:
            return
        temperatures = [temperature]
        existing = [i for i in active_list if generated[i]]
        if existing:
            # Both proposal directions must use temperature. Rescore generated tokens without changing them; base_scores
            # and alpha remain valid.
            rescored = _score_components(
                mh_llm_model, [prompt_tokens[i] + generated[i] for i in existing],
                [len(prompt_tokens[i]) for i in existing], temperatures,
                sampling_params, scoring_batch_size,
            )
            for i, scores in zip(existing, rescored):
                component_scores[i] = scores

    for block in range(num_of_blocks):
        if not active:
            break
        active_list = sorted(active)
        if proposal_temperatures is None and temperature_scheduler is not None:
            temperature_scheduler.step_count = 0
            set_temperature(float(temperature_scheduler.step()), active_list)
        logger.info("block [%d/%d], active prompts=%d, proposal temperatures=%s",
                    block + 1, num_of_blocks, len(active_list), temperatures)
        # 1. Append block_size tokens to generated[i] for each i in active_list
        # (Algorithm 1, line 5). Refinements keep len(generated[i]) fixed.
        tokens, logp, components = draw_proposals(
            mh_llm_model, [prompt_tokens[i] + generated[i] for i in active_list],
            [block_size] * len(active_list), temperatures, sampling_params,
            scoring_batch_size, seed_rng,
        )
        for row, i in enumerate(active_list):
            generated[i].extend(tokens[row])
            base_scores[i].extend(logp[row])
            component_scores[i] = np.concatenate([component_scores[i], components[row]], axis=1)

        for step in range(mcmc_steps):
            step_started = time.perf_counter()
            if proposal_temperatures is None and temperature_scheduler is not None:
                set_temperature(float(temperature_scheduler.step()), active_list)
            # 2. Choose cuts[i] uniformly from range(len(generated[i])).
            # cuts[i] is response-relative; its token is the first regenerated token. Earlier blocks can change. The
            # state-independent cut probability cancels between directions at this fixed horizon.
            cuts = {i: random.randint(0, len(generated[i]) - 1) for i in active_list}
            # 3. Draw num_tries candidates per prompt (Algorithm 1, line 12).
            # Flatten all prompt/candidate pairs into one batch: owners repeats each prompt num_tries times, keeping its
            # candidate rows contiguous.
            owners = [i for i in active_list for _ in range(num_tries)]
            logger.info(
                "MH proposal batch: block=%d/%d step=%d/%d prompts=%d candidates=%d",
                block + 1, num_of_blocks, step + 1, mcmc_steps, len(active_list), len(owners),
            )
            tokens, logp, components = draw_proposals(
                mh_llm_model,
                [prompt_tokens[i] + generated[i][:cuts[i]] for i in owners],
                [len(generated[i]) - cuts[i] for i in owners],
                temperatures, sampling_params, scoring_batch_size, seed_rng,
            )
            accepted_count = 0
            for row, i in enumerate(active_list):
                cut = cuts[i]
                start = row * num_tries
                # Slice component_scores[i][:, cut:] BEFORE mixing: draw the temperature afresh at cut.
                # sum(current_logq) then scores generated[i][cut:] given prompt_tokens[i] + generated[i][:cut].
                current_logq = _mixture_logprobs(component_scores[i][:, cut:])
                # 4. Eq. (6): weights stores this prompt's log MH weights.
                # The comprehension's j indexes the flattened batch; its result goes in weights[j - start] for tokens[j]
                # and components[j].
                weights = [
                    compute_acceptance_ratio(
                        _mixture_logprobs(components[j]), current_logq,
                        logp[j], base_scores[i][cut:], alpha,
                    )
                    for j in range(start, start + num_tries)
                ]
                # 5. Algorithm 1, lines 13-14, for local j in range(num_tries):
                # P(selected == j) = np.exp(weights[j]) / sum(np.exp(weights)).
                # The helper shifts weights before exponentiating to avoid overflow.
                selected = _categorical_from_log_weights(weights)
                # 6. Eq. (7), expressed using the same local variables:
                # np.exp(log_acceptance) = min(
                #     1, sum(np.exp(weights))
                #     / (1 + sum(np.exp(np.delete(weights, selected))))
                # ). The 1 is the current generated[i] state's relative weight; the other terms reuse the unselected
                # candidates. The helper evaluates this stably in log space. For num_tries == 1, the probability reduces
                # to min(1, np.exp(weights[0])).
                log_acceptance = _log_acceptance_probability(weights, selected)
                accepted = np.random.rand() < math.exp(log_acceptance)
                candidate_selection_probabilities, candidate_acceptance_probabilities, _ = (
                    _mtm_edge_probabilities(weights)
                )
                logger.info(
                    "MH decision: prompt_index=%d cut=%d suffix_tokens=%d selected_index=%d candidates=%d "
                    "selected_log_weight=%.4f p_select_cand=[%s] "
                    "A_mtm=[%s] "
                    "mtm_acceptance_probability=%.6g accepted=%s",
                    i, cut, len(generated[i]) - cut, selected, num_tries, weights[selected],
                    ", ".join(f"{probability:.6g}" for probability in candidate_selection_probabilities),
                    ", ".join(f"{probability:.6g}" for probability in candidate_acceptance_probabilities),
                    math.exp(log_acceptance), accepted,
                )
                if verbose:
                    logger.debug("prompt[%d] candidate_log_weights=%s", i, weights)
                if accepted:
                    # chosen = start + selected converts to the batch index.
                    # Replace generated[i][cut:] and both matching score slices together (Algorithm 1, line 16). A
                    # rejection leaves this prompt's state and scores intact.
                    accepted_count += 1
                    chosen = start + selected
                    generated[i][cut:] = tokens[chosen]
                    base_scores[i][cut:] = logp[chosen]
                    component_scores[i][:, cut:] = components[chosen]
            total_accepted += accepted_count
            total_attempted += len(active_list)
            logger.info("MH step %d/%d: accepted %d/%d prompts, elapsed=%.3fs",
                        step + 1, mcmc_steps, accepted_count, len(active_list), time.perf_counter() - step_started)

        # Finish fixed-horizon refinement before truncating at response EOS. Trim caches with the tokens and remove
        # completed prompts from future batches; prompt EOS tokens are outside generated[i].
        eos_id = mh_llm_model.tokenizer.eos_token_id
        for i in active_list:
            if eos_id in generated[i]:
                keep = generated[i].index(eos_id) + 1
                generated[i] = generated[i][:keep]
                base_scores[i] = base_scores[i][:keep]
                component_scores[i] = component_scores[i][:, :keep]
                active.remove(i)
                logger.info("prompt[%d] finished at EOS after block %d/%d: response_tokens=%d",
                            i, block + 1, num_of_blocks, keep)

    # Return response text in the original prompt order, including chains that finished in earlier blocks.
    logger.info(
        "MultiTryMH finished: prompts=%d response_tokens=%d accepted=%d/%d acceptance_rate=%.4f elapsed=%.3fs",
        len(prompts), sum(map(len, generated)), total_accepted, total_attempted,
        total_accepted / total_attempted if total_attempted else 0.0, time.perf_counter() - sampling_started,
    )
    return [mh_llm_model.tokenizer.decode(tokens, skip_special_tokens=True) for tokens in generated]
