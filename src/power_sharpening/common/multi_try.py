"""Shared NumPy operations for the HF and vLLM paper-style MTM samplers.

Run the CPU checks with:
    python -m pytest /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/tests/test_prefetch_multi_try_tree.py
"""

import math

import numpy as np

# Default suffix-mixture proposal for the MTM samplers, matching the list the MultiTry case-study scripts run. Passing
# ``None`` instead selects the scalar proposal temperature, which the HF/vLLM samplers may anneal on a schedule.
DEFAULT_PROPOSAL_TEMPERATURES = (0.25, 0.5, 1.0)


def log_sum_exp(log_values):
    """Numerically stable log(sum(exp(log_values))), including an empty sum."""
    return float(np.logaddexp.reduce(np.asarray(log_values, dtype=np.float64)))


def categorical_from_log_weights(log_weights, rng=None) -> int:
    """Select one proposed candidate for the MTM acceptance decision.

    An MTM step generates several candidate suffixes, but can move to only one. This function chooses which candidate to
    consider. In the MTM callers, ``log_weights[i]`` is candidate i's log MH ratio relative to the current
    sequence: it combines the target probability ratio with the correction for
    how often the proposal generates each suffix. Higher-weight candidates are more likely to be selected, according to
    the MTM selection rule

        P(selected = i) = w_i / sum_k w_k,
        where w_i = exp(log_weights[i]).

    The returned integer is the selected candidate's index. The caller then passes this index and all trial log weights
    to ``log_acceptance_probability`` to decide whether to move to that candidate or keep the current sequence.
    Selection alone does not update the chain; the acceptance decision completes the MTM step.
    """
    arr = np.asarray(log_weights, dtype=np.float64)
    probs = np.exp(arr - np.max(arr))
    sampler = np.random if rng is None else rng
    return int(sampler.choice(len(probs), p=probs / probs.sum()))


def log_acceptance_probability(log_weights, selected) -> float:
    """Return the log probability of accepting the selected MTM candidate.

    Let w_i = exp(log_weights[i]) be candidate i's MH ratio relative to the current state. Eq. (7) gives

        A = min(1, sum_i w_i / (1 + sum_{i != selected} w_i)).

    The denominator reuses the unselected candidates and includes the current state with weight 1. Return log(A): 0
    means certain acceptance; rejection keeps the current state.
    """
    unselected = np.delete(np.asarray(log_weights, dtype=np.float64), selected)
    log_den = log_sum_exp(np.append(unselected, 0.0))
    return min(0.0, log_sum_exp(log_weights) - log_den)


def suffix_log_weights(state, proposals, cut, alpha):
    """Return candidates' log MH ratios against the current suffix at cut.

    State and proposals are (token_ids, base_logprobs, logprobs_by_temperature).
    """
    _, base, logprobs_by_temperature = state
    current_logq = math.fsum(mixture_logprobs(logprobs_by_temperature[:, cut:]))
    current_logp = math.fsum(base[cut:])
    return np.asarray([
        alpha * (math.fsum(proposal_base) - current_logp)
        + current_logq - math.fsum(mixture_logprobs(proposal_logprobs_by_temperature))
        for _, proposal_base, proposal_logprobs_by_temperature in proposals
    ])


def mtm_edge_probabilities(log_weights):
    """Return selection, conditional acceptance, and K+1 edge probabilities.

    For w_j = exp(log_weights[j]), s_j = w_j / sum_i w_i, and MTM acceptance A_j, edges 0..K-1 accept candidates with
    probability s_j * A_j. Edge K is rejection, with probability 1 - sum_j s_j * A_j. These probabilities condition on
    the current state and generated candidates.
    """
    log_weights = np.asarray(log_weights, dtype=np.float64)
    selection = np.exp(log_weights - np.max(log_weights))
    selection /= selection.sum()
    acceptance = np.array([
        math.exp(log_acceptance_probability(log_weights, j))
        for j in range(len(log_weights))
    ])
    moves = selection * acceptance
    """Clamp a negative rejection probability caused by roundoff."""
    edges = np.append(moves, max(0.0, 1.0 - moves.sum()))
    edges /= edges.sum()
    return selection, acceptance, edges


def mixture_logprobs(component_logprobs):
    """Score a suffix under the temperature-mixture proposal used by MTM.

    Each proposal chooses one of M temperatures uniformly for the whole suffix. For retained prefix c and suffix y, its
    proposal probability is

        q(y | c) = (1 / M) * sum_k prod_t p_{T_k}(y_t | c, y_{<t}).

    ``component_logprobs`` has shape (M, L): each row contains the same L-token suffix's conditional log probabilities
    under one temperature. Returns L
    conditional token log probabilities whose sum is log q(y | c), used in
    the MH proposal correction.

    Slice the input columns at the regeneration cut BEFORE calling: the temperature is chosen afresh there.
    """
    logprobs_by_temperature = np.asarray(component_logprobs, dtype=np.float64)
    cumulative = np.concatenate([
        np.zeros((len(logprobs_by_temperature), 1)),
        np.cumsum(logprobs_by_temperature, axis=1),
    ], axis=1)
    log_mixture = np.logaddexp.reduce(cumulative, axis=0) - np.log(len(logprobs_by_temperature))
    return np.diff(log_mixture).tolist()
