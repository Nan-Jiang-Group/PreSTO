"""Subtree-prefetching Metropolis-Hastings with uniform or entropy-based cuts.

Both samplers operate on one prompt in token space and batch suffix proposals through the custom vLLM engine.
The EntropyCut version also maintains entropy estimates and corrects for state-dependent cut probabilities.

"""

import json
import logging
import math
import time

import numpy as np

from power_sharpening.common.cache_eviction import evict_subtree_cache as _evict_subtree_cache

from power_sharpening.backends.vllm.engine_patch.sampling_params import _copy_sampling_params
from power_sharpening.backends.vllm.samplers.proposal_llm_call import (
    batched_proposal_call,
    batched_proposal_call_with_entropy_cut,
    vllm_proposal_sampling,
)
from power_sharpening.common.prefetch_subtree import (
    build_and_sample_subtree,
    collect_batch_cut_indicesv3,
)
from power_sharpening.common.sample_stats import BaseModelDiagnostics, SamplingStats

logger = logging.getLogger("[subtree prefetching MH]")

# Printed request tables flag A >= 1 - eps, A <= eps, and eps < A < 1 - eps in separate columns.
CERTAINTY_EPSILON = 0.01


def _truncate_at_eos(
    generated_seq: list[int],
    proposal_logprobs: list[float],
    power_logprobs: list[float],
    eos_id: int | None,
) -> tuple[list[int], list[float], list[float], bool]:
    """Truncate generated tokens and both score caches after EOS, and report whether EOS was found.

    ``eos_id=None`` disables truncation. Otherwise, the EOS token is retained and later tokens are discarded.
    """
    if eos_id is None or eos_id not in generated_seq:
        return generated_seq, proposal_logprobs, power_logprobs, False
    keep = generated_seq.index(eos_id) + 1
    return (
        generated_seq[:keep],
        proposal_logprobs[:keep],
        power_logprobs[:keep],
        True,
    )


def _print_root_sequence_scores(
    power_logprobs: list[float],
    entropies: list[float] | None = None,
) -> None:
    """Print the subtree root's per-token scores as JSON lists, one line each, for offline parsing.

    Prints base-model log p (length L) and its adjacent difference ``delta[t] = log p[t + 1] - log p[t]`` (length
    L - 1). When ``entropies`` is given, prints them (length L) and their adjacent difference (length L - 1) too.
    """
    def emit(name: str, values: np.ndarray) -> None:
        print(f"root {name} (len={len(values)}): " + json.dumps([round(float(v), 4) for v in values]), flush=True)

    log_p = np.asarray(power_logprobs, dtype=float)
    emit("log_p", log_p)
    emit("delta_log_p", np.diff(log_p))
    if entropies is not None:
        entropy = np.asarray(entropies, dtype=float)
        emit("entropy", entropy)
        emit("delta_entropy", np.diff(entropy))


def _rank(values: np.ndarray) -> np.ndarray:
    """Rank values with ties sharing their average rank."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=float)
    ranks[order] = np.arange(values.size, dtype=float)
    for value in np.unique(values):
        tied = values == value
        ranks[tied] = ranks[tied].mean()
    return ranks


def _print_node_features(
    power_logprobs: list[float],
    entropies: list[float] | None,
    node_to_batch_idx: dict[int, int],
    batch_cuts: list[int],
    log_accept_ratios,
    path: list,
) -> None:
    """Print, for each prefetched request, its pre-proposal features beside its acceptance, then their per-tree
    Spearman correlations with ``log A = min(0, log_acc_prob)``. The ``A>=1-eps``, ``A<=eps``, and ``eps<A<1-eps``
    columns flag the three classes at ``CERTAINTY_EPSILON`` with 1/0; exactly one is 1 per row.

    A request at cut c keeps tokens ``[0, c)`` and resamples from token c, so ``log_p`` and ``entropy`` are the root's
    values at c, ``d_in = v[c] - v[c - 1]`` and ``d_out = v[c + 1] - v[c]`` (NaN past either end). Every cut lies in the
    prefix its decision node shares with the root, so these are also the decision node's own values. The per-tree
    correlations are over at most a batch of requests; pool trees offline before trusting them.
    """
    log_p = np.asarray(power_logprobs, dtype=float)
    entropy = np.asarray(entropies, dtype=float) if entropies is not None else None

    def at(values: np.ndarray, index: int) -> float:
        return float(values[index]) if 0 <= index < values.size else math.nan

    # Recover the node each sampled decision was made at.
    decisions: dict[int, str] = {}
    node_id = 0
    for decision, _, _ in path:
        decisions[node_id] = decision
        node_id = 2 * node_id + 1 if decision == "accept" else 2 * node_id + 2

    columns = ["cut", "log_p", "d_log_p_in", "d_log_p_out"]
    if entropy is not None:
        columns += ["entropy", "d_entropy_in", "d_entropy_out"]
    rows = []
    for child_id, batch_idx in sorted(node_to_batch_idx.items()):
        cut = batch_cuts[batch_idx]
        features = [float(cut), at(log_p, cut), at(log_p, cut) - at(log_p, cut - 1), at(log_p, cut + 1) - at(log_p, cut)]
        if entropy is not None:
            features += [at(entropy, cut), at(entropy, cut) - at(entropy, cut - 1), at(entropy, cut + 1) - at(entropy, cut)]
        log_ratio = float(log_accept_ratios[batch_idx])
        rows.append((child_id, (child_id - 1) // 2, features, log_ratio))

    header = f"{'node':>5} {'parent':>6} " + " ".join(f"{name:>13}" for name in columns)
    header += f" {'log_acc_prob':>13} {'A':>7} {'A>=1-eps':>9} {'A<=eps':>7} {'eps<A<1-eps':>12} {'decision':>8}"
    print("prefetched request features (root scores at each cut):")
    print(header)
    for child_id, parent_id, features, log_ratio in rows:
        values = " ".join(f"{value:>13.4f}" for value in features)
        accept_prob = math.exp(min(0.0, log_ratio))
        certain_accept = int(accept_prob >= 1.0 - CERTAINTY_EPSILON)
        certain_reject = int(accept_prob <= CERTAINTY_EPSILON)
        uncertain = 1 - certain_accept - certain_reject
        print(
            f"{child_id:>5} {parent_id:>6} {values} {log_ratio:>13.4f} "
            f"{accept_prob:>7.4f} {certain_accept:>9} {certain_reject:>7} {uncertain:>12} {decisions.get(parent_id, '-'):>8}"
        )

    log_a = np.array([min(0.0, row[3]) for row in rows])
    correlations = []
    for column, name in enumerate(columns):
        feature = np.array([row[2][column] for row in rows])
        keep = ~np.isnan(feature)
        if keep.sum() < 3 or np.ptp(feature[keep]) == 0 or np.ptp(log_a[keep]) == 0:
            correlations.append(f"{name}=nan")
            continue
        rho = float(np.corrcoef(_rank(feature[keep]), _rank(log_a[keep]))[0, 1])
        correlations.append(f"{name}={rho:+.3f}")
    print(f"per-tree Spearman with log A (n={len(rows)}): " + ", ".join(correlations))
    print("-" * 60, flush=True)


def subtree_prefetching_sampling(
    mh_llm_model,
    prompt_ids: list[int],
    sampling_params,
    mcmc_steps: int = 10,
    max_batch_size: int = 4,
    max_new_tokens: int = 1024,
    num_of_blocks: int = 16,
    rank_fn: str = "longest_path_first",
    stop_on_eos: bool = True,
    print_tree: bool = False,
    temperature_scheduler=None,
    rng: np.random.Generator | None = None,
    verbose: bool = False,
    evict_subtree_cache: bool = False,
) -> tuple[list[int], SamplingStats]:
    """Sample one prompt from p^alpha using subtree-prefetching MH with uniform cuts.

    Each block extends the response, generates a batch of suffix proposals, walks the proposal subtree, and
    updates the accepted token and score caches. Generation ignores EOS until refinement finishes so every
    proposal spans the same horizon as the suffix it replaces.

    Uniform cut probabilities cancel in the MH ratio. The proposal temperature may follow a per-block schedule;
    the target exponent alpha stays fixed.

    Args:
        mh_llm_model: wrapper providing the custom vLLM engine and tokenizer.
        prompt_ids: fixed prompt token IDs, excluded from the returned response.
        sampling_params: generation settings, including proposal temperature and target alpha.
        mcmc_steps: MH transitions per block.
        max_batch_size: maximum proposals per batched subtree traversal.
        max_new_tokens: total generation budget, divisible by ``num_of_blocks``.
        num_of_blocks: number of autoregressive blocks.
        rank_fn: proposal-subtree frontier ranking mode from ``RANK_FNS``.
        stop_on_eos: stop after the first block containing EOS; the final response is always truncated at EOS.
        print_tree: print each prefetched proposal tree.
        temperature_scheduler: optional proposal-temperature schedule, stepped once per block.
        rng: NumPy generator for cut draws and subtree traversal; created if omitted.
        verbose: log proposal and traversal details at the debug level.
        evict_subtree_cache: evict discarded cached prefixes after each completed traversal; requires an enabled wrapper.

    Returns:
        generated_seq: final generated token IDs, excluding the prompt.
        stats: per-block and aggregate sampling statistics.
    """
    if rng is None:
        rng = np.random.default_rng()

    eos_id = mh_llm_model.tokenizer.eos_token_id

    # Keep every proposal at the block horizon; handle EOS after refinement.
    sampling_params = _copy_sampling_params(sampling_params, ignore_eos=True)

    generated_seq: list[int] = []
    proposal_logprobs: list[float] = []   # log q over generated tokens
    power_logprobs: list[float] = []      # log p over generated tokens

    assert max_new_tokens % num_of_blocks == 0
    block_size = max_new_tokens // num_of_blocks

    stats = SamplingStats()

    for block_index in range(num_of_blocks):
        block_start = time.perf_counter()
        logger.info("block [%d/%d]", block_index, num_of_blocks)

        # Advance the proposal temperature while keeping the target exponent fixed.
        if temperature_scheduler is not None:
            new_temp = temperature_scheduler.step()
            sampling_params = _copy_sampling_params(sampling_params, temperature=new_temp)

        # Extend the response by one block.
        extend_start = time.perf_counter()
        context_ids = prompt_ids + generated_seq
        new_suffix, block_proposal_lp, block_power_lp = vllm_proposal_sampling(
            mh_llm_model,
            context_ids=context_ids,
            max_new_tokens=block_size,
            sampling_params=sampling_params,
            return_entropies=False,
            verbose=verbose,
        )
        logger.info(
            "  block extend took %.3f sec: %d prompts, context_len=%s, "
            "new_suffix_len=%s",
            time.perf_counter() - extend_start,
            1,
            [len(context_ids)],
            [len(new_suffix)],
        )
        generated_seq.extend(new_suffix)
        proposal_logprobs.extend(block_proposal_lp)
        power_logprobs.extend(block_power_lp)
        if verbose:
            logger.debug(
                "sequence length: %d, block proposal logprobs: %d, block power logprobs: %d",
                len(generated_seq), len(block_proposal_lp), len(block_power_lp),
            )

        completed_steps = 0

        while completed_steps < mcmc_steps:
            gen_len = len(generated_seq)
            if gen_len < 2:
                # Not enough tokens to define a cut; skip refinement this block.
                break

            step_index = completed_steps
            step_start = time.perf_counter()

            if print_tree:
                _print_root_sequence_scores(power_logprobs)

            # Cuts index the generated response, so the prompt offset is zero.
            parent_to_child, node_to_batch_idx, batch_cuts = collect_batch_cut_indicesv3(
                max_batch_size=max_batch_size,
                prompt_len=0,
                seq_len=gen_len,
                mcmc_steps=mcmc_steps - completed_steps,
                rank=rank_fn,
                rng=rng,
                cut_dist_type="uniform",
            )

            if verbose:
                logger.debug("parent_to_child: %s", parent_to_child)
                logger.debug("node_to_batch_idx: %s", node_to_batch_idx)
                logger.debug("batch_cuts: %s", batch_cuts)

            # Generate and score the suffix proposals in one batch.
            (
                proposals,
                log_accept_ratios,
                cached_proposal_lp,
                cached_power_lp,
            ) = batched_proposal_call(
                mh_llm_model,
                parent_proposal_seq=generated_seq,
                batched_cut_indexes=batch_cuts,
                proposal_logprobs_seq=proposal_logprobs,
                power_logprobs_seq=power_logprobs,
                sampling_params=sampling_params,
                prompt_ids=prompt_ids,
                verbose=verbose,
            )

            if verbose:
                logger.debug("num proposals: %d", len(proposals))
                logger.debug("accept_ratios: %s", np.exp(log_accept_ratios))

            stats.batch_sizes.append(len(proposals))

            # Traverse the batch as sequential MH accept/reject decisions.
            previous_seq = generated_seq
            leaf, last_accepted, path, num_accepted = build_and_sample_subtree(
                subtree_root_proposal=generated_seq,
                sampled_proposals=proposals,
                log_acceptance_ratios=log_accept_ratios,
                node_to_batch_idx=node_to_batch_idx,
                parent_to_accept_child=parent_to_child,
                batch_cut_indices=batch_cuts,
                rng=rng,
                print_tree=print_tree,
                verbose=verbose,
            )

            if verbose:
                logger.debug(
                    "leaf: %s, trajectory: %s, accepted: %d",
                    leaf, [s[0] for s in path], num_accepted,
                )

            if print_tree:
                _print_node_features(
                    power_logprobs, None, node_to_batch_idx, batch_cuts, log_accept_ratios, path,
                )

            completed_steps += len(path)
            stats.acceptances.append(num_accepted)
            stats.walked_steps.append(len(path))

            # Adopt the final state and replace scores from the last accepted cut onward.
            generated_seq = list(leaf.proposal)
            if num_accepted > 0:
                batch_idx = node_to_batch_idx[last_accepted.node_id]
                accepted_cut = batch_cuts[batch_idx]
                proposal_logprobs[accepted_cut:] = cached_proposal_lp[batch_idx].copy()
                power_logprobs[accepted_cut:] = cached_power_lp[batch_idx].copy()

            if evict_subtree_cache:
                _evict_subtree_cache(
                    mh_llm_model, stats,
                    [prompt_ids + seq for seq in [previous_seq, *proposals]],
                    prompt_ids + generated_seq,
                )

            logger.info(
                "  MH step %d/%d took %.3f seconds",
                step_index,
                mcmc_steps,
                time.perf_counter() - step_start,
            )

        # Record the block duration even when EOS ends generation.
        logger.info(
            "block [%d/%d] took %.3f seconds",
            block_index,
            num_of_blocks,
            time.perf_counter() - block_start,
        )

        # Finish refinement before truncating at EOS.
        if stop_on_eos:
            previous_seq = generated_seq
            generated_seq, proposal_logprobs, power_logprobs, saw_eos = _truncate_at_eos(
                generated_seq, proposal_logprobs, power_logprobs, eos_id
            )
            if evict_subtree_cache and len(previous_seq) != len(generated_seq):
                _evict_subtree_cache(mh_llm_model, stats, [prompt_ids + previous_seq], prompt_ids + generated_seq)
            if saw_eos:
                logger.info("stop on eos after block %d", block_index)
                break

    if not stop_on_eos:
        # Even when all blocks run, return only tokens through the first EOS.
        previous_seq = generated_seq
        generated_seq, proposal_logprobs, power_logprobs, _ = _truncate_at_eos(
            generated_seq, proposal_logprobs, power_logprobs, eos_id
        )
        if evict_subtree_cache and len(previous_seq) != len(generated_seq):
            _evict_subtree_cache(mh_llm_model, stats, [prompt_ids + previous_seq], prompt_ids + generated_seq)

    expected_score_count = len(generated_seq)
    if len(power_logprobs) != expected_score_count or len(proposal_logprobs) != expected_score_count:
        raise RuntimeError(
            "final per-token caches are misaligned with the generated "
            f"response: expected {expected_score_count}, got "
            f"{len(proposal_logprobs)} proposal and {len(power_logprobs)} power log-probs"
        )

    return generated_seq, stats


def subtree_prefetching_sampling_with_entropy_cut(
    mh_llm_model,
    prompt_ids: list[int],
    sampling_params,
    mcmc_steps: int = 10,
    max_batch_size: int = 4,
    max_new_tokens: int = 1024,
    num_of_blocks: int = 16,
    rank_fn: str = "longest_path_first",
    cut_power: float | None = 4.0,
    stop_on_eos: bool = True,
    print_tree: bool = False,
    temperature_scheduler=None,
    rng: np.random.Generator | None = None,
    verbose: bool = False,
    evict_subtree_cache: bool = False,
) -> tuple[list[int], SamplingStats]:
    """Sample one prompt from p^alpha using subtree-prefetching MH with entropy cuts.

    Each block extends the response, generates a batch of suffix proposals, walks the proposal subtree, and
    updates the accepted token and score caches. Generation ignores EOS until refinement finishes so every
    proposal spans the same horizon as the suffix it replaces.

    Entropy estimates stay aligned with generated tokens and score caches. Each MH ratio includes the
    reverse/forward cut-probability correction. Proposal temperature must remain fixed.

    Args:
        mh_llm_model: wrapper providing the custom vLLM engine and tokenizer.
        prompt_ids: fixed prompt token IDs, excluded from the returned response.
        sampling_params: generation settings, including proposal temperature and target alpha.
        mcmc_steps: MH transitions per block.
        max_batch_size: maximum proposals per batched subtree traversal.
        max_new_tokens: total generation budget, divisible by ``num_of_blocks``.
        num_of_blocks: number of autoregressive blocks.
        rank_fn: proposal-subtree frontier ranking mode from ``RANK_FNS``.
        cut_power: exponent for the entropy-cut probabilities; None uses 4.0.
        stop_on_eos: stop after the first block containing EOS; the final response is always truncated at EOS.
        print_tree: print each prefetched proposal tree.
        temperature_scheduler: optional constant schedule; annealing is rejected.
        rng: NumPy generator for cut draws and subtree traversal; created if omitted.
        verbose: log proposal and traversal details at the debug level.
        evict_subtree_cache: evict discarded cached prefixes after each completed traversal; requires an enabled wrapper.

    Returns:
        generated_seq: final generated token IDs, excluding the prompt.
        stats: sampling statistics and base-model likelihood/confidence diagnostics.
    """
    if rng is None:
        rng = np.random.default_rng()

    if cut_power is None:
        cut_power = 4.0
    if temperature_scheduler is not None and getattr(temperature_scheduler, "is_anneal", True):
        raise ValueError(
            "EntropyCut caches proposal scores and requires a fixed proposal "
            "temperature; use no schedule or the constant schedule"
        )

    eos_id = mh_llm_model.tokenizer.eos_token_id

    # Keep every proposal at the block horizon; handle EOS after refinement.
    sampling_params = _copy_sampling_params(sampling_params, ignore_eos=True)

    generated_seq: list[int] = []
    proposal_logprobs: list[float] = []   # log q over generated tokens
    power_logprobs: list[float] = []      # log p over generated tokens
    entropies: list[float] = []           # full-vocabulary H[p(. | prefix)] per token

    assert max_new_tokens % num_of_blocks == 0
    block_size = max_new_tokens // num_of_blocks

    stats = SamplingStats()

    for block_index in range(num_of_blocks):
        block_start = time.perf_counter()
        logger.info("block [%d/%d]", block_index, num_of_blocks)

        # A constant schedule may be supplied for runner compatibility; annealing was rejected above.
        if temperature_scheduler is not None:
            new_temp = temperature_scheduler.step()
            sampling_params = _copy_sampling_params(sampling_params, temperature=new_temp)

        # Extend the response by one block.
        extend_start = time.perf_counter()
        context_ids = prompt_ids + generated_seq
        new_suffix, block_proposal_lp, block_power_lp, block_entropies = vllm_proposal_sampling(
            mh_llm_model,
            context_ids=context_ids,
            max_new_tokens=block_size,
            sampling_params=sampling_params,
            return_entropies=True,
            verbose=verbose,
        )
        entropies.extend(block_entropies)
        logger.info(
            "  block extend took %.3f sec: %d prompts, context_len=%s, "
            "new_suffix_len=%s",
            time.perf_counter() - extend_start,
            1,
            [len(context_ids)],
            [len(new_suffix)],
        )
        generated_seq.extend(new_suffix)
        proposal_logprobs.extend(block_proposal_lp)
        power_logprobs.extend(block_power_lp)
        if verbose:
            logger.debug(
                "sequence length: %d, block proposal logprobs: %d, block power logprobs: %d",
                len(generated_seq), len(block_proposal_lp), len(block_power_lp),
            )

        completed_steps = 0

        while completed_steps < mcmc_steps:
            gen_len = len(generated_seq)
            if gen_len < 2:
                # Not enough tokens to define a cut; skip refinement this block.
                break

            step_index = completed_steps
            step_start = time.perf_counter()

            if print_tree:
                _print_root_sequence_scores(power_logprobs, entropies)

            # Cuts index the generated response, so the prompt offset is zero.
            parent_to_child, node_to_batch_idx, batch_cuts = collect_batch_cut_indicesv3(
                max_batch_size=max_batch_size,
                prompt_len=0,
                seq_len=gen_len,
                mcmc_steps=mcmc_steps - completed_steps,
                rank=rank_fn,
                rng=rng,
                cut_dist_type="entropy",
                cut_dist_param=cut_power,
                cut_entropies=entropies,
            )

            if verbose:
                logger.debug("parent_to_child: %s", parent_to_child)
                logger.debug("node_to_batch_idx: %s", node_to_batch_idx)
                logger.debug("batch_cuts: %s", batch_cuts)

            # Generate and score the suffix proposals in one batch.
            (
                proposals,
                log_accept_ratios,
                cached_proposal_lp,
                cached_power_lp,
                cached_entropies,
            ) = batched_proposal_call_with_entropy_cut(
                mh_llm_model,
                parent_proposal_seq=generated_seq,
                batched_cut_indexes=batch_cuts,
                proposal_logprobs_seq=proposal_logprobs,
                power_logprobs_seq=power_logprobs,
                sampling_params=sampling_params,
                prompt_ids=prompt_ids,
                cut_entropies=entropies,
                cut_power=cut_power,
                verbose=verbose,
            )
            if verbose:
                logger.debug("num proposals: %d", len(proposals))
                logger.debug("accept_ratios: %s", np.exp(log_accept_ratios))

            stats.batch_sizes.append(len(proposals))

            # Traverse the batch as sequential MH accept/reject decisions.
            previous_seq = generated_seq
            leaf, last_accepted, path, num_accepted = build_and_sample_subtree(
                subtree_root_proposal=generated_seq,
                sampled_proposals=proposals,
                log_acceptance_ratios=log_accept_ratios,
                node_to_batch_idx=node_to_batch_idx,
                parent_to_accept_child=parent_to_child,
                batch_cut_indices=batch_cuts,
                rng=rng,
                print_tree=print_tree,
                verbose=verbose,
            )

            if verbose:
                logger.debug(
                    "leaf: %s, trajectory: %s, accepted: %d",
                    leaf, [s[0] for s in path], num_accepted,
                )

            if print_tree:
                _print_node_features(
                    power_logprobs, entropies, node_to_batch_idx, batch_cuts, log_accept_ratios, path,
                )

            completed_steps += len(path)
            stats.acceptances.append(num_accepted)
            stats.walked_steps.append(len(path))

            # Adopt the final state and replace scores from the last accepted cut onward.
            generated_seq = list(leaf.proposal)
            if num_accepted > 0:
                batch_idx = node_to_batch_idx[last_accepted.node_id]
                accepted_cut = batch_cuts[batch_idx]
                proposal_logprobs[accepted_cut:] = cached_proposal_lp[batch_idx].copy()
                power_logprobs[accepted_cut:] = cached_power_lp[batch_idx].copy()
                entropies[accepted_cut:] = cached_entropies[batch_idx].copy()

            if evict_subtree_cache:
                _evict_subtree_cache(
                    mh_llm_model, stats,
                    [prompt_ids + seq for seq in [previous_seq, *proposals]],
                    prompt_ids + generated_seq,
                )

            logger.info(
                "  MH step %d/%d took %.3f seconds",
                step_index,
                mcmc_steps,
                time.perf_counter() - step_start,
            )

        # Record the block duration even when EOS ends generation.
        logger.info(
            "block [%d/%d] took %.3f seconds",
            block_index,
            num_of_blocks,
            time.perf_counter() - block_start,
        )

        # Finish refinement before truncating at EOS.
        if stop_on_eos:
            previous_seq = generated_seq
            generated_seq, proposal_logprobs, power_logprobs, saw_eos = _truncate_at_eos(
                generated_seq, proposal_logprobs, power_logprobs, eos_id
            )
            entropies = entropies[:len(generated_seq)]
            if evict_subtree_cache and len(previous_seq) != len(generated_seq):
                _evict_subtree_cache(mh_llm_model, stats, [prompt_ids + previous_seq], prompt_ids + generated_seq)
            if saw_eos:
                logger.info("stop on eos after block %d", block_index)
                break

    if not stop_on_eos:
        # Even when all blocks run, return only tokens through the first EOS.
        previous_seq = generated_seq
        generated_seq, proposal_logprobs, power_logprobs, _ = _truncate_at_eos(
            generated_seq, proposal_logprobs, power_logprobs, eos_id
        )
        entropies = entropies[:len(generated_seq)]
        if evict_subtree_cache and len(previous_seq) != len(generated_seq):
            _evict_subtree_cache(mh_llm_model, stats, [prompt_ids + previous_seq], prompt_ids + generated_seq)

    expected_score_count = len(generated_seq)
    if (
        len(power_logprobs) != expected_score_count
        or len(proposal_logprobs) != expected_score_count
        or len(entropies) != expected_score_count
    ):
        raise RuntimeError(
            "final per-token caches are misaligned with the generated "
            f"response: expected {expected_score_count}, got "
            f"{len(proposal_logprobs)} proposal and {len(power_logprobs)} power "
            f"log-probs and {len(entropies)} entropies"
        )

    if expected_score_count > 0:
        stats.base_diagnostics = BaseModelDiagnostics(
            num_tokens=expected_score_count,
            logprob_sum=math.fsum(power_logprobs),
            neg_entropy_sum=-math.fsum(entropies),
        )

    return generated_seq, stats
