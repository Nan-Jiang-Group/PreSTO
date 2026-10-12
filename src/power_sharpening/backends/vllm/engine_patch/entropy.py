"""Predictive-entropy and EntropyCut helpers for the vLLM backend.

Run the focused CPU tests with:
    uv run --project /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src \
      pytest /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/tests/test_vllm_entropy_cut.py
"""

from __future__ import annotations

import math

import numpy as np
import torch


def predictive_entropies(logits: torch.Tensor) -> torch.Tensor:
    """Compute full-vocabulary base-model entropy for each decoding position.

    Args:
        logits: Base-model logits of shape ``[num_positions, vocab_size]``,
            before temperature scaling or sampling processors.

    Returns:
        Float32 Shannon entropies in nats, of shape ``[num_positions]`` on
        the input device. Tokens with ``-inf`` logits contribute zero.
    """
    logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float32)
    finite_logprobs = logprobs.masked_fill(torch.isneginf(logprobs), 0.0)
    return (-(logprobs.exp() * finite_logprobs).sum(dim=-1)).clamp_min(0.0)


def suffix_log_acceptance(
    proposed_logq: np.ndarray,
    current_logq: np.ndarray,
    proposed_logp: np.ndarray,
    current_logp: np.ndarray,
    alpha: float,
) -> float:
    """Return the MH log ratio for a fixed-length suffix proposal.

    The target is ``pi(x) proportional to p(x)^alpha`` and the proposal is ``q``. Therefore

    ``log A = alpha * (log p(x') - log p(x))  + log q(x) - log q(x')``.
    """
    target_term = alpha * (
            math.fsum(proposed_logp) - math.fsum(current_logp)
        )
    proposal_term = math.fsum(current_logq) - math.fsum(proposed_logq)
    return target_term + proposal_term


__all__ = [
    "predictive_entropies",
    "suffix_log_acceptance",
]
