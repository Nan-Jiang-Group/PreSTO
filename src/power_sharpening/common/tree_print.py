"""Proposal-tree printouts shared by the vLLM and SGLang subtree-prefetching (PreSTO) samplers.

Both samplers call these when ``print_tree`` is on; ``case_studies/extract/analysis/acceptance_features.py`` parses the
lines. Import-only module: ``from power_sharpening.common.tree_print import print_root_sequence_scores``.
"""

import json
import math

import numpy as np

# Printed request tables flag A >= 1 - eps, A <= eps, and eps < A < 1 - eps in separate columns.
CERTAINTY_EPSILON = 0.01


def print_root_sequence_scores(
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


def print_node_features(
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
