"""Shared cut-index distributions for PowerMH and subtree-prefetching.

Run the CPU tests with:
    uv run --project /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src \
      pytest /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/tests/test_cut_distribution.py
"""

from __future__ import annotations

import logging

import numpy as np


logger = logging.getLogger("[cut distribution]")


def compute_entropy_cut_policy(
    cut_entropies: np.ndarray,
    beta: float = 4,
) -> np.ndarray:
    """Return entropy-cut probabilities.
    Given ``[h_0, h_1, ..., h_l]``, compute the positive entropy jumps ``Delta_t(x) = max(0, h_t(x) - h_{t-1}(x))`` and
    sample a cut with probability proportional to ``Delta_t(x)**beta``. If every jump is zero, sample the cut uniformly.

    The result is one entry SHORTER than ``cut_entropies``: ``Delta_t`` needs both ``h_t`` and ``h_{t-1}``, so
    ``Delta_0`` does not exist and ``n`` entropies define only ``n - 1`` jumps. Entry ``j`` is therefore the jump
    between tokens ``j`` and ``j + 1``, i.e. the mass of cut position ``j + 1`` -- callers index it by cut OFFSET, not
    by cut index. See ``draw_entropy_cut_index`` for the offset-to-index conversion and ``entropy_cut_log_ratio`` for
    its inverse.
    """
    delta_by_cut_offset = np.maximum(0.0, np.diff(cut_entropies))
    lambda_beta = delta_by_cut_offset**beta
    lambda_beta_sum = np.sum(lambda_beta)

    if lambda_beta_sum == 0.0:
        return np.full(
            lambda_beta.size,
            1.0 / lambda_beta.size,
            dtype=np.float64,
        )
    return lambda_beta / lambda_beta_sum


def summarize_cut_policy(
    cut_entropies: np.ndarray,
    probabilities: np.ndarray,
    beta: float,
    low: int = 1,
    top_k: int = 5,
) -> str:
    """Return a one-line digest of the entropy jumps that drive the cut law.

    Dumping all ``n - 1`` probabilities is unreadable and mostly zeros. Report the positive entropy jumps instead: these
    are the decision points whose ``jump**beta`` values define the cut probabilities. Cut indices are the values
    returned by ``draw_entropy_cut_index`` (``low + jump offset``).
    """
    entropies = np.asarray(cut_entropies, dtype=float)
    jumps = np.maximum(0.0, np.diff(entropies))
    positive_offsets = np.flatnonzero(jumps > 0.0)
    order = positive_offsets[
        np.argsort(jumps[positive_offsets])[::-1][:top_k]
    ]
    top = ", ".join(
        f"cut {low + int(offset)} (+{jumps[offset]:.3f})"
        for offset in order
    )

    summary = (
        f"The sampler considered {jumps.size} possible cuts. "
        f"Entropy ranged from {entropies.min():.3f} to "
        f"{entropies.max():.3f}, with an average of {entropies.mean():.3f}. "
    )
    if positive_offsets.size == 0:
        return summary + (
            "Entropy did not increase between adjacent positions, so the "
            "cuts are sampled uniformly."
        )
    if positive_offsets.size == 1:
        return summary + (
            f"Entropy increased at 1 position. The largest increase occurred "
            f"at {top}."
        )
    return summary + (
        f"Entropy increased at {positive_offsets.size} positions. "
        f"The largest increases occurred at {top}."
    )


def entropy_cut_log_ratio(
    current_probabilities: np.ndarray,
    proposed_probabilities: np.ndarray,
    cut_offset: int,
) -> float:
    """Return ``log lambda(m; x') - log lambda(m; x)`` for one cut."""
    return np.log(proposed_probabilities[cut_offset]) - \
           np.log(current_probabilities[cut_offset] )



def draw_uniform_cut_index(
    rng: np.random.Generator,
    low: int,
    high: int,
) -> int:
    """Draw one uniformly distributed cut index from ``[low, high]``."""
    # Cast to a Python int: a NumPy scalar leaks into vLLM SamplingParams (max_tokens) and breaks msgpack encoding.
    return int(rng.integers(low, high, endpoint=True))


def draw_entropy_cut_index(
    rng: np.random.Generator,
    cut_entropies: np.ndarray,
    low: int = 1,
    beta: float = 4,
    verbose: bool = False,
) -> int:
    r"""Draw a cut index using Algorithm 2 of Entropy-Cut MH.

    ``compute_entropy_cut_policy`` returns ``n - 1`` masses for ``n`` entropies, so ``rng.choice`` draws a cut OFFSET
    ``j`` in ``[0, n - 2]``; adding ``low`` turns it into the cut index the caller slices with. The default ``low=1`` is
    what makes the two line up: offset ``j`` is the jump between tokens ``j`` and ``j + 1``, which is the jump at
    position
    ``j + 1``. Reachable indices are therefore ``[1, n - 1]``:

    * index 0 is unreachable, so the whole sequence is never resampled -- ``Delta_0`` is undefined, per Definition 1;
    * index ``n - 1`` is reachable, so the shortest proposal is one token.

    That support differs from ``draw_uniform_cut_index`` as PowerMH calls it (``[0, n - 2]``, which can resample
    everything but always resamples at least two tokens), so the two cut laws are not exploring the same set.

    ``entropy_cut_log_ratio`` undoes the shift with ``cut_offset = index - 1``; keep the two in step when changing
    ``low``.

    Set ``verbose`` to log the cut law the draw was taken from.

    See https://arxiv.org/pdf/2605.30327, Algorithm 2 and Definition 1.
    """

    probabilities = compute_entropy_cut_policy(
        cut_entropies,
        beta=beta,
    )

    # Offset in [0, n - 2] over the jumps; `low` re-anchors it to a cut index.
    sample_index = rng.choice(probabilities.size, p=probabilities)
    if verbose:
        cut_index = low + int(sample_index)
        entropy_before = cut_entropies[sample_index]
        entropy_after = cut_entropies[sample_index + 1]
        logger.info(
            "%s Selected cut %d. Entropy changed from %.4f to %.4f at "
            "this cut; the positive increase used by the sampler was %.4f.",
            summarize_cut_policy(
                cut_entropies,
                probabilities,
                beta,
                low=low,
            ),
            cut_index,
            entropy_before,
            entropy_after,
            max(0.0, entropy_after - entropy_before),
        )
    return low + int(sample_index)


__all__ = [
    "compute_entropy_cut_policy",
    "draw_entropy_cut_index",
    "draw_uniform_cut_index",
    "entropy_cut_log_ratio",
    "summarize_cut_policy",
]
