"""Compute base-model log-likelihood and confidence for sampled responses.

Run the focused tests with:
    uv run --project /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src \
      pytest /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/tests/test_base_diagnostics.py
"""

import logging
import math
from typing import Sequence

import torch
from torch.nn import functional as F

from power_sharpening.common.sample_stats import BaseModelDiagnostics

logger = logging.getLogger("[HF log-likelihood and confidence]")

# Positions scored per reduction chunk. Bounds the fp32 log-softmax buffer to chunk_size x vocab_size floats regardless
# of how long the response is.
DEFAULT_CHUNK_SIZE = 256


def _eager_module(base_model):
    """Return the uncompiled module behind a ``torch.compile`` wrapper.

    Computing log-likelihood and confidence requires scoring one variable-length sequence per problem. Using the
    compiled graph would trigger a recompile per distinct length for a single forward pass. ``OptimizedModule`` keeps
    the original module on ``_orig_mod``; plain modules are returned unchanged.
    """
    return getattr(base_model, "_orig_mod", base_model)


@torch.inference_mode()
def compute_log_likelihood_and_confidence(
    sampler_wrapper,
    sequence: Sequence[int],
    prompt_len: int,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> BaseModelDiagnostics:
    """Compute log-likelihood and confidence for a terminal MH state under the base model p_0.

    Runs one teacher-forced forward pass over ``sequence`` and reduces the next-token distributions at the response
    positions into

        logprob_sum     = sum_t log p_0(x_t | x_0, x_{<t}),
        neg_entropy_sum = sum_t sum_u p_0(u | x_0, x_{<t}) log p_0(u | x_0, x_{<t}).

    The base model is called directly, so the logits carry neither the proposal temperature nor the target exponent
    alpha: both statistics are measured under p_0 itself and stay comparable across values of alpha.

    Args:
        sampler_wrapper: model wrapper exposing ``base_model`` and ``device``.
        sequence: full terminal token sequence (prompt followed by response).
        prompt_len: number of leading prompt tokens; tokens at and after this index are the response tokens being
            scored.
        chunk_size: number of response positions reduced per fp32 log-softmax.

    Returns:
        A :class:`BaseModelDiagnostics` holding the unnormalized sums plus the length-normalized log-likelihood and
        confidence.
    """
    if prompt_len < 1:
        raise ValueError(
            "scoring the first response token requires at least one prompt "
            f"token to condition on, got prompt_len={prompt_len}"
        )
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    tokens = list(sequence)
    num_response_tokens = len(tokens) - prompt_len
    if num_response_tokens < 1:
        raise ValueError(
            "terminal state has no response tokens to score: sequence length "
            f"{len(tokens)} with prompt_len={prompt_len}"
        )

    device = sampler_wrapper.device
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    output = _eager_module(sampler_wrapper.base_model)(
        input_ids=input_ids,
        use_cache=False,
    )
    # Row t-1 holds the distribution over token t, so response tokens x_{prompt_len:} are predicted by rows prompt_len-1
    # through len-2.
    response_logits = output.logits[0, prompt_len - 1 : -1]
    response_ids = input_ids[0, prompt_len:]
    if response_logits.shape[0] != num_response_tokens:
        raise RuntimeError(
            "base model returned logits for "
            f"{response_logits.shape[0]} response positions instead of "
            f"{num_response_tokens}"
        )

    logprob_chunks = []
    neg_entropy_chunks = []
    for start in range(0, num_response_tokens, chunk_size):
        stop = min(start + chunk_size, num_response_tokens)
        logprobs = F.log_softmax(
            response_logits[start:stop],
            dim=-1,
            dtype=torch.float32,
        )
        selected = torch.gather(
            logprobs,
            -1,
            response_ids[start:stop].unsqueeze(-1),
        ).reshape(-1)
        # sum_u p log p, i.e. the negative entropy of each next-token distribution.
        neg_entropy = (logprobs.exp() * logprobs).sum(dim=-1)

        _require_finite(selected, "selected-token log-probabilities", start)
        _require_finite(neg_entropy, "next-token negative entropies", start)

        logprob_chunks.append(float(selected.double().sum().item()))
        neg_entropy_chunks.append(float(neg_entropy.double().sum().item()))

    metrics = BaseModelDiagnostics(
        num_tokens=num_response_tokens,
        logprob_sum=math.fsum(logprob_chunks),
        neg_entropy_sum=math.fsum(neg_entropy_chunks),
    )
    logger.debug(
        "scored %d response tokens: log_likelihood=%.6f, confidence=%.6f",
        metrics.num_tokens,
        metrics.log_likelihood,
        metrics.confidence,
    )
    return metrics


def _require_finite(values: torch.Tensor, stream_name: str, offset: int) -> None:
    """Fail loudly when a log-probability or negative-entropy reduction is non-finite."""
    invalid = torch.nonzero(~torch.isfinite(values), as_tuple=False).reshape(-1)
    if invalid.numel():
        positions = [offset + int(index) for index in invalid.cpu().tolist()]
        raise RuntimeError(
            f"base model produced non-finite {stream_name} at response "
            f"positions {positions}. This indicates non-finite model logits "
            "or padding being scored as response tokens."
        )
