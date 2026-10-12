"""HuggingFace low-temperature proposal sampling and scoring.

Run the focused sampler tests with:
    uv run --project /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src \
      pytest /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/tests/test_hf_low_temp_numerics.py
"""

import logging
from typing import List

import torch
from torch.nn import functional as F

logger = logging.getLogger("[HF low-temp sampler]")


def _selected_token_logprobs(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
) -> List[float]:
    """Gather selected-token log-probabilities in fp32 and require finiteness."""
    selected = torch.gather(
        F.log_softmax(logits, dim=-1, dtype=torch.float32),
        -1,
        token_ids,
    ).reshape(-1)
    return selected.tolist()


def low_temp_sampling(
        sampler_wrapper,
        context: List[int],
        seq_len: int,
        use_cache: bool = True,
        verbose: bool = False,
        ignore_eos: bool = False,
        return_entropies: bool = False,
):
    """Sample from a low-temperature proposal distribution and return log-probs.

    Generates a continuation of ``context`` by sampling from a proposal
    distribution q(x_t | x_{<t}) = softmax(logits / T), where T = 1/alpha
    and the logits come from the base LM.

    The returned log-probabilities are used to compute the Metropolis-Hastings
    acceptance ratio:

        log A = [log pi(x') - log pi(x)] + [log q(x) - log q(x')]

    where pi is the (unnormalized) target distribution p^alpha and q is the proposal.

    Args:
        sampler_wrapper: model wrapper that exposes ``generate()`` with temperature and alpha configuration.
        context: token ids for the prefix/prompt.
        seq_len: total sequence length (context + new tokens).
        verbose: print extra diagnostics.
        ignore_eos: continue decoding through EOS. Fixed-length MH callers must enable this so every proposal and score
            cache has the same horizon.

    Returns:
        proposal_seq: full token sequence (context + generated tokens).
        suffix_proposal_logprob: per-token log q(x_t | x_{<t}), the normalized
            log-probability under the temperature-scaled proposal distribution.
        suffix_target_log_score: per-token alpha * log p(x_t | x_{<t}), the
            unnormalized log-score under the target distribution p^alpha.
    """
    context_len = len(context)

    input_ids = torch.tensor([context], dtype=torch.long, device=sampler_wrapper.device)
    if verbose:
        logger.debug("input_ids: batch=%d, len=%d", len(input_ids), len(input_ids[0]))
        logger.debug("seq_len %d, context_len: %d", seq_len, context_len)

    assert seq_len - context_len > 0, "new suffixes must be non-negative"
    output = sampler_wrapper.generate(
        input_ids,
        seq_len - context_len,
        output_logits=True,
        use_cache=use_cache,
        ignore_eos=ignore_eos,
    )

    if verbose:
        logger.debug("sampled output: %s", vars(output).keys())

    # Strip the input context; keep only newly generated tokens
    unscaled_logits = torch.stack(output.logits, dim=0)
    scaled_logits = torch.stack(output.scores, dim=0)
    # cut the prefix, only preserve the suffix seq.
    suffix_tokens = output.sequences[0][context_len:]
    proposal_seq = output.sequences[0].tolist()
    if ignore_eos and len(proposal_seq) != seq_len:
        raise RuntimeError(
            "fixed-length HF proposal generation returned "
            f"{len(proposal_seq)} tokens instead of {seq_len}; "
            "ignore_eos=True was not honored"
        )

    # log q(x_t | x_{<t}): normalized log-prob under the proposal distribution.
    # output.scores are the temperature-scaled logits actually used for sampling.
    suffix_new_ids = suffix_tokens.view(-1, 1, 1)
    suffix_proposal_logprob = _selected_token_logprobs(
        scaled_logits,
        suffix_new_ids,
    )

    if verbose:
        logger.debug(
            "suffix_proposal_logprob %d, first 10 tokens: %s, last 10 tokens: %s",
            len(suffix_proposal_logprob),
            suffix_proposal_logprob[:10],
            suffix_proposal_logprob[-10:],
        )

    alpha = sampler_wrapper.alpha
    # alpha * log p(x_t | x_{<t}): log-score under the unnormalized target p^alpha.
    # output.logits are the raw (unscaled) LM logits.
    suffix_base_logprob = _selected_token_logprobs(
        unscaled_logits,
        suffix_new_ids,
    )
    suffix_target_log_score = [
        float(alpha) * logprob for logprob in suffix_base_logprob
    ]
    if verbose:
        logger.debug(
            "suffix_target_log_score %d, first 10 tokens: %s, last 10 tokens: %s",
            len(suffix_target_log_score),
            suffix_target_log_score[:10],
            suffix_target_log_score[-10:],
        )

    if return_entropies:
        # Exact base-model predictive entropy per generated position.
        base_logprobs = torch.log_softmax(unscaled_logits.float(), dim=-1)
        entropy = -(base_logprobs.exp() * base_logprobs).sum(dim=-1)
        if not torch.isfinite(entropy).all():
            raise RuntimeError("non-finite base-model predictive entropy")
        suffix_entropies = [
            max(0.0, float(h)) for h in entropy.double().flatten().cpu()
        ]
        if len(suffix_entropies) != len(suffix_proposal_logprob):
            raise RuntimeError(
                "entropy/token misalignment: "
                f"{len(suffix_entropies)} entropies vs "
                f"{len(suffix_proposal_logprob)} tokens"
            )
        return (
            proposal_seq,
            suffix_proposal_logprob,
            suffix_target_log_score,
            suffix_entropies,
        )
    return proposal_seq, suffix_proposal_logprob, suffix_target_log_score


def batched_low_temp_proposal_sampling(
        sampler_wrapper,
        contexts: List[List[int]],
        seq_len: int,
        use_cache: bool = True,
        verbose: bool = False,
        ignore_eos: bool = False,
        return_entropies: bool = False,
        row_temperatures=None,
        component_temperatures=None,
) -> tuple:
    """Batched version of low_temp_proposal_sampling.

    Left-pads all contexts to equal length and runs a single ``generate()`` call, then extracts per-sequence proposal
    tokens and log-probs.

    Args:
        sampler_wrapper: model wrapper with temperature and alpha config.
        contexts: batch of token-id prefixes (variable length).
        seq_len: target total length (context + new tokens) for all sequences.
        verbose: print extra diagnostics.
        ignore_eos: continue decoding through EOS. Fixed-length MH callers must enable this so finished rows are not
            padded while other rows decode.
        return_entropies: also return per-token base-model predictive entropies, needed by state-dependent cut laws
            (EntropyCut). Costs one extra vocabulary-wide softmax reduction per generated position.
        row_temperatures: one sampling temperature per batch row, indexed like ``contexts``. Row ``j`` draws its suffix
            from ``softmax(logits / row_temperatures[j])``. When None every row shares ``sampler_wrapper.temperature``.
        component_temperatures: temperatures the already-sampled suffixes are re-scored under, indexed independently of
            the batch. Reuses the generation logits, so no extra forward pass. Callers building a mixture proposal pass
            the full component list here and a per-row draw from it as ``row_temperatures``.

    Returns:
        proposal_seqs: list of full token sequences (one per context).
        suffix_proposal_logprobs: per-sequence list of per-token log q(x_t | x_{<t}).
        suffix_target_log_scores: per-sequence list of per-token alpha * log p(x_t | x_{<t}).
        suffix_entropies: only when ``return_entropies``; per-sequence list of
            per-token base-model predictive entropies H(p(. | x_{<t})).
        component_logprobs: appended last when ``component_temperatures`` is set; nested lists indexed by batch row,
            then component temperature, then suffix token.
    """
    device = sampler_wrapper.device
    batch_size = len(contexts)
    context_lens = [len(c) for c in contexts]
    max_context_len = max(context_lens)
    min_context_len = min(context_lens)
    max_new_tokens = seq_len - min_context_len

    assert max_new_tokens > 0, "seq_len must exceed every context length"

    # Left-pad shorter contexts so all inputs have equal length.
    pad_id = sampler_wrapper.tokenizer.pad_token_id

    padded_inputs = []
    attention_masks = []
    for ctx in contexts:
        pad_len = max_context_len - len(ctx)
        padded_inputs.append([pad_id] * pad_len + ctx)
        attention_masks.append([0] * pad_len + [1] * len(ctx))

    input_ids = torch.tensor(padded_inputs, dtype=torch.long, device=device)
    attention_mask = torch.tensor(attention_masks, dtype=torch.long, device=device)

    if verbose:
        logger.debug(
            "batched generate: batch_size=%d, context_lens=%s, max_new_tokens=%d",
            batch_size, context_lens, max_new_tokens,
        )

    temperature_kwargs = (
        {} if row_temperatures is None else {"row_temperatures": row_temperatures}
    )
    output = sampler_wrapper.generate(
        input_ids,
        max_new_tokens,
        attention_mask=attention_mask,
        output_logits=True,
        use_cache=use_cache,
        ignore_eos=ignore_eos,
        **temperature_kwargs,
    )

    # output.logits / output.scores: tuples of (batch, vocab), one per step. Stack to (batch, max_new_tokens, vocab).
    unscaled_logits = torch.stack(output.logits, dim=1)
    scaled_logits = torch.stack(output.scores, dim=1)
    alpha = sampler_wrapper.alpha

    proposal_seqs: List[List[int]] = []
    suffix_proposal_logprobs: List[List[float]] = []
    suffix_target_log_scores: List[List[float]] = []
    suffix_entropies: List[List[float]] = []
    component_logprobs = []

    for i in range(batch_size):
        ctx_len = context_lens[i]
        num_new = seq_len - ctx_len
        pad_len = max_context_len - ctx_len

        # Extract tokens: skip left-padding, take context + num_new generated tokens.
        full_seq = output.sequences[i][pad_len: pad_len + seq_len].tolist()
        suffix_tokens = output.sequences[i][max_context_len: max_context_len + num_new]
        if ignore_eos and len(full_seq) != seq_len:
            raise RuntimeError(
                "fixed-length batched HF proposal generation returned "
                f"{len(full_seq)} tokens instead of {seq_len} for batch row {i}; "
                "ignore_eos=True was not honored"
            )

        # Gather log-probs for the first num_new generation steps.
        suffix_ids = suffix_tokens.view(-1, 1)
        seq_scaled = scaled_logits[i, :num_new]
        seq_unscaled = unscaled_logits[i, :num_new]

        suffix_proposal_lp = _selected_token_logprobs(
            seq_scaled,
            suffix_ids,
        )
        suffix_base_lp = _selected_token_logprobs(
            seq_unscaled,
            suffix_ids,
        )
        suffix_target_ls = [
            float(alpha) * logprob for logprob in suffix_base_lp
        ]

        if verbose:
            logger.debug(
                "  >>> proposal[%d]: context_len=%d, new suffix len=%d, full proposal len=%d",
                i, ctx_len, num_new, len(full_seq),
            )

        proposal_seqs.append(full_seq)
        suffix_proposal_logprobs.append(suffix_proposal_lp)
        suffix_target_log_scores.append(suffix_target_ls)

        if component_temperatures is not None:
            component_logprobs.append([
                _selected_token_logprobs(
                    seq_unscaled.float() / temperature,
                    suffix_ids,
                )
                for temperature in component_temperatures
            ])

        if return_entropies:
            # Exact base-model predictive entropy per generated position:
            # H_t = -sum_u p(u) log p(u) over the full vocabulary.
            base_logprobs = torch.log_softmax(seq_unscaled.float(), dim=-1)
            entropy = -(base_logprobs.exp() * base_logprobs).sum(dim=-1)

            # Clamp away the tiny negative values float error can produce.
            suffix_entropies.append(
                [max(0.0, float(h)) for h in entropy.double().cpu()]
            )

    result = (proposal_seqs, suffix_proposal_logprobs, suffix_target_log_scores)
    if return_entropies:
        result += (suffix_entropies,)
    if component_temperatures is not None:
        result += (component_logprobs,)
    return result
