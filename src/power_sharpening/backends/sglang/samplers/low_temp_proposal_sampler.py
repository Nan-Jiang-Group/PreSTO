"""SGLang version of ``power_sharpening.backends.hf.samplers.low_temp_sampler``.

Run through an installed checkout with:
    uv run --project /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src python <script>

For each sampled suffix token it returns the proposal log-prob ``log q`` (temperature-scaled at T = 1/alpha) and the
target log-score ``alpha * log p``. SGLang exposes only one post-temperature logprob stream per request, so it takes
two passes instead of the HF version's single ``generate`` call:

  1. Decode at T = 1/alpha: ``output_token_logprobs`` are ``log q`` for the sampled suffix.
  2. Score the full sequence at T = 1: ``input_token_logprobs`` are ``log p`` (a cheap prefill that reuses SGLang's
     RadixAttention prefix cache).

``custom_logit_processor`` cannot replace the second pass through the public SGLang API. The processor can mutate logits
before sampling, but it has no
return channel in ``meta_info``. A single request can therefore expose either
the proposal stream ``log q`` or the raw stream ``log p`` for sampled tokens, not both; reconstructing ``log q`` from
raw ``log p`` would also require the full-vocabulary normalizer at every decode step.

When ``entropy_top_k`` is set, pass 2 also requests SGLang's ``input_top_logprobs`` and stores an aligned entropy proxy
in the returned metadata. The proxy treats the probability mass outside the top K as one tail bucket, so it is a lower
bound on the full-vocabulary Shannon entropy rather than the exact EntropyCut quantity. Exact entropy requires a
server-side scalar output that SGLang 0.5.2 does not expose through its public ``Engine.generate`` response.
"""

import math
from typing import List

# Per-request sampling params for SGLang. top_p/top_k are neutralised so the proposal is exactly softmax(logits / T)
# (matching the HF temperature-only proposal); change here if you want truncated-sampling proposals.
_NEUTRAL = {"top_p": 1.0, "top_k": -1}
_ENTROPY_MODE = "topk_tail_bucket"


def _unpack_logprob_entry(entry):
    """SGLang logprob entries are ``[logprob, token_id, token_text?]``."""
    logprob = entry[0]
    token_id = entry[1]
    return float(logprob), (int(token_id) if token_id is not None else None)


def _validate_entropy_top_k(entropy_top_k):
    """Validate and normalize the optional top-K entropy request."""
    if entropy_top_k is None:
        return None
    if isinstance(entropy_top_k, bool) or not isinstance(entropy_top_k, int):
        raise TypeError("entropy_top_k must be a positive integer or None")
    if entropy_top_k <= 0:
        raise ValueError("entropy_top_k must be positive")
    return entropy_top_k


def _topk_tail_bucket_entropy(entries):
    """Return entropy after coarsening all non-top-K tokens into one bucket.

    ``entries`` are SGLang top-logprob entries whose log probabilities are normalized over the full vocabulary.
    Coarsening the unreturned tail makes this quantity a deterministic lower bound on the exact Shannon entropy.
    """
    if not entries:
        raise RuntimeError(
            "SGLang returned no input_top_logprobs for an entropy-scored token"
        )

    logprobs = [_unpack_logprob_entry(entry)[0] for entry in entries]
    if any(math.isnan(logprob) or logprob == math.inf for logprob in logprobs):
        raise RuntimeError("SGLang returned an invalid top-K log probability")
    if any(logprob > 1e-6 for logprob in logprobs):
        raise RuntimeError("SGLang returned a positive log probability")

    # Tiny positive values can occur from floating-point roundoff. Clamp those to log(1) before exponentiation.
    logprobs = [min(logprob, 0.0) for logprob in logprobs]
    probabilities = [math.exp(logprob) for logprob in logprobs]
    topk_mass = math.fsum(probabilities)
    if topk_mass > 1.0 + 1e-5:
        raise RuntimeError(
            f"SGLang top-K probabilities sum to {topk_mass:.8f}, which exceeds 1"
        )

    tail_mass = max(0.0, 1.0 - min(topk_mass, 1.0))
    entropy = -math.fsum(
        probability * logprob
        for probability, logprob in zip(probabilities, logprobs)
        if probability > 0.0
    )
    if tail_mass > 0.0:
        entropy -= tail_mass * math.log(tail_mass)
    return entropy


def require_suffix_entropies(meta):
    """Return aligned suffix entropies or raise an actionable configuration error."""
    entropies = meta.get("suffix_entropies")
    if entropies is None:
        raise RuntimeError(
            "Suffix entropy is unavailable. Pass entropy_top_k=<positive integer> "
            "to low_temp_proposal_sampling for the SGLang top-K lower-bound "
            "estimator. Exact full-vocabulary entropy requires a server-side "
            "SGLang extension; SGLang 0.5.2 does not expose it through "
            "Engine.generate."
        )
    return entropies


def low_temp_proposal_sampling(
    sampler_wrapper,
    context: list,
    seq_len: int,
    verbose: bool = False,
    use_cache: bool | None = None,
    entropy_top_k: int | None = None,
    ignore_eos: bool = False,
):
    """Sample a suffix from the proposal q and return its proposal/target log-probs.

    Mirrors ``power_sharpening.backends.hf.samplers.low_temp_sampler.low_temp_sampling`` (same signature and
    return tuple), plus a 4th element carrying the raw decode ``meta_info`` for
    debugging.

    Args:
        sampler_wrapper: an ``AutoRegressiveLMWrapper``
        context: token ids for the prefix.
        seq_len: total target length (context + new tokens).
        verbose: print diagnostics.
        use_cache: accepted for HF-call compatibility; SGLang manages its own RadixAttention cache and this value is
            ignored.
        entropy_top_k: if set, request the top K base-model log probabilities in pass 2 and compute a tail-bucket
            entropy lower bound.
        ignore_eos: continue decoding through EOS to preserve a fixed-length sequence state space.

    Returns:
        proposal_seq: full token sequence (context + sampled suffix).
        suffix_proposal_logprob: per-token log q(x_t | x_{<t}) (post-temperature).
        suffix_target_log_score: per-token alpha * log p(x_t | x_{<t}).
        meta: decode ``meta_info`` augmented with ``suffix_entropies``, ``entropy_mode``, and ``entropy_top_k``.
            Entropies are ``None`` when ``entropy_top_k`` is not requested.
    """
    del use_cache
    entropy_top_k = _validate_entropy_top_k(entropy_top_k)

    engine = sampler_wrapper.engine
    temperature = sampler_wrapper.temperature
    alpha = sampler_wrapper.alpha

    context = list(context)
    context_len = len(context)
    max_new_tokens = seq_len - context_len
    assert max_new_tokens > 0, "seq_len must exceed the context length"

    # --- Pass 1: decode the suffix from the proposal q at temperature T ---------
    gen = engine.generate(
        input_ids=context,
        sampling_params={
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "ignore_eos": ignore_eos,
            **_NEUTRAL,
        },
        return_logprob=True,
    )
    meta = gen["meta_info"]
    out_lp = meta["output_token_logprobs"]  # list of [logprob, token_id, text]

    suffix_tokens: List[int] = []
    suffix_proposal_logprob: List[float] = []
    for entry in out_lp:
        lp, tok = _unpack_logprob_entry(entry)
        suffix_tokens.append(tok)
        suffix_proposal_logprob.append(lp)  # log q (post-temperature)

    proposal_seq = context + suffix_tokens
    n_suffix = len(suffix_tokens)

    if verbose:
        print(f"[decode] context_len={context_len}, sampled suffix len={n_suffix}")

    # --- Pass 2: score the full sequence at T=1 to recover log p over the suffix
    suffix_target_log_score, suffix_entropies = _score_base_distribution(
        engine,
        proposal_seq,
        context_len,
        n_suffix,
        alpha,
        entropy_top_k=entropy_top_k,
        verbose=verbose,
    )

    meta["suffix_entropies"] = suffix_entropies
    meta["entropy_mode"] = _ENTROPY_MODE if entropy_top_k is not None else None
    meta["entropy_top_k"] = entropy_top_k

    return proposal_seq, suffix_proposal_logprob, suffix_target_log_score, meta


def _score_base_logprobs(engine, full_seq, context_len, n_suffix, alpha, verbose=False):
    """Return ``alpha * log p`` for the last ``n_suffix`` tokens of ``full_seq``.

    Scores at temperature 1 so the returned ``input_token_logprobs`` are the raw base-model logprobs (independent of any
    temperature-scaling flag).
    """
    target_log_scores, _ = _score_base_distribution(
        engine,
        full_seq,
        context_len,
        n_suffix,
        alpha,
        entropy_top_k=None,
        verbose=verbose,
    )
    return target_log_scores


def _score_base_distribution(
    engine,
    full_seq,
    context_len,
    n_suffix,
    alpha,
    entropy_top_k=None,
    verbose=False,
):
    """Return target scores and optional aligned top-K entropy lower bounds."""
    if n_suffix == 0:
        return [], ([] if entropy_top_k is not None else None)

    generate_kwargs = dict(
        input_ids=list(full_seq),
        sampling_params={
            "temperature": 1.0,
            "max_new_tokens": 1,
            "ignore_eos": True,
            **_NEUTRAL,
        },
        return_logprob=True,
        # Start early enough to guarantee the whole suffix is covered; we slice the last n_suffix entries below, which
        # sidesteps off-by-one ambiguity in SGLang's logprob_start_len convention.
        logprob_start_len=max(context_len - 1, 0),
    )
    if entropy_top_k is not None:
        generate_kwargs["top_logprobs_num"] = entropy_top_k
    score = engine.generate(**generate_kwargs)
    score_meta = score["meta_info"]
    in_lp = score_meta["input_token_logprobs"]

    # The final n_suffix input-token logprobs correspond to the suffix tokens (each conditioned on all preceding
    # tokens).
    suffix_entries = in_lp[-n_suffix:]
    assert len(suffix_entries) == n_suffix, (
        f"scoring returned {len(suffix_entries)} logprobs for {n_suffix} suffix tokens"
    )
    suffix_base_logprob = [_unpack_logprob_entry(e)[0] for e in suffix_entries]

    suffix_entropies = None
    if entropy_top_k is not None:
        if "input_top_logprobs" not in score_meta:
            raise RuntimeError(
                "SGLang did not return input_top_logprobs after "
                f"top_logprobs_num={entropy_top_k} was requested"
            )
        suffix_topk_entries = score_meta["input_top_logprobs"][-n_suffix:]
        if len(suffix_topk_entries) != n_suffix:
            raise RuntimeError(
                "SGLang returned misaligned input_top_logprobs: "
                f"expected {n_suffix}, got {len(suffix_topk_entries)}"
            )
        suffix_entropies = [
            _topk_tail_bucket_entropy(entries) for entries in suffix_topk_entries
        ]

    if verbose:
        print(f"[score] suffix base logprob count={len(suffix_base_logprob)}")
        if suffix_entropies is not None:
            print(
                f"[score] suffix entropy count={len(suffix_entropies)}, "
                f"mode={_ENTROPY_MODE}, top_k={entropy_top_k}"
            )

    return [alpha * lp for lp in suffix_base_logprob], suffix_entropies


def batched_low_temp_proposal_sampling(
    sampler_wrapper,
    contexts: List[List[int]],
    seq_len: int,
    verbose: bool = False,
    use_cache: bool | None = None,
    ignore_eos: bool = False,
) -> tuple[List[List[int]], List[List[float]], List[List[float]]]:
    """Batched version of :func:`low_temp_proposal_sampling`.

    Issues one batched decode pass and one batched scoring pass over all contexts. Each context may have a different
    length; ``max_new_tokens`` is set per request so every sequence is extended to ``seq_len`` (no left-padding needed).

    Args:
        sampler_wrapper: holds the SGLang engine, temperature and alpha.
        contexts: batch of token-id prefixes (variable length).
        seq_len: target total length (context + new tokens) for all sequences.
        verbose: print diagnostics.
        use_cache: accepted for HF-call compatibility; SGLang manages its own RadixAttention cache and this value is
            ignored.
        ignore_eos: continue decoding through EOS to preserve fixed lengths.

    Returns:
        proposal_seqs: list of full token sequences (one per context).
        suffix_proposal_logprobs: per-sequence per-token log q.
        suffix_target_log_scores: per-sequence per-token alpha * log p.
    """
    del use_cache

    engine = sampler_wrapper.engine
    temperature = sampler_wrapper.temperature
    alpha = sampler_wrapper.alpha

    contexts = [list(c) for c in contexts]
    for c in contexts:
        assert seq_len - len(c) > 0, "seq_len must exceed every context length"

    # --- Batched decode pass (proposal q) --------------------------------------
    decode_params = [
        {
            "temperature": temperature,
            "max_new_tokens": seq_len - len(c),
            "ignore_eos": ignore_eos,
            **_NEUTRAL,
        }
        for c in contexts
    ]
    gens = engine.generate(
        input_ids=contexts, sampling_params=decode_params, return_logprob=True
    )

    proposal_seqs: List[List[int]] = []
    suffix_proposal_logprobs: List[List[float]] = []
    suffix_lens: List[int] = []
    for ctx, gen in zip(contexts, gens):
        out_lp = gen["meta_info"]["output_token_logprobs"]
        toks, qlps = [], []
        for entry in out_lp:
            lp, tok = _unpack_logprob_entry(entry)
            toks.append(tok)
            qlps.append(lp)
        proposal_seqs.append(ctx + toks)
        suffix_proposal_logprobs.append(qlps)
        suffix_lens.append(len(toks))

    if verbose:
        print(f"[batched decode] {len(contexts)} requests, suffix_lens={suffix_lens}")

    # --- Batched scoring pass (target alpha * log p) ---------------------------
    score_params = [
        {
            "temperature": 1.0,
            "max_new_tokens": 1,
            "ignore_eos": True,
            **_NEUTRAL,
        }
        for _ in proposal_seqs
    ]
    start_lens = [max(len(c) - 1, 0) for c in contexts]
    scores = engine.generate(
        input_ids=proposal_seqs,
        sampling_params=score_params,
        return_logprob=True,
        logprob_start_len=start_lens,
    )

    suffix_target_log_scores: List[List[float]] = []
    for n_suffix, sc in zip(suffix_lens, scores):
        if n_suffix == 0:
            suffix_target_log_scores.append([])
            continue
        in_lp = sc["meta_info"]["input_token_logprobs"]
        suffix_entries = in_lp[-n_suffix:]
        assert len(suffix_entries) == n_suffix
        suffix_target_log_scores.append(
            [alpha * _unpack_logprob_entry(e)[0] for e in suffix_entries]
        )

    return proposal_seqs, suffix_proposal_logprobs, suffix_target_log_scores
