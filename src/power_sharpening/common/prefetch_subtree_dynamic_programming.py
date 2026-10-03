"""Budget-constrained prefetch-request selection by dynamic programming.

This is the predictive subtree allocation of the paper's Appendix C
(``tex/C.tree-traversal.tex`` and ``tex/C.predictive_dp.tex``): instead of
expanding the prefetching tree in a fixed queue order, choose the
ancestor-closed set of decisions that maximizes the predicted number of
completed MH transitions under a request budget ``N_pf``.

The tree utilities (``Node``, ``build_subtree``, the pathwise sampler, the Rich
display) and the best-first collector live in
``power_sharpening.common.prefetch_subtree`` and are re-exported here, so a
sampler can swap ``collect_batch_cut_indicesv3`` for
``collect_batch_cut_indices_dp`` and change nothing else: both return the same
``(parent_to_child, node_to_batch_idx, batch_cut_indices)`` triple.

Run the demo with:
    uv run --project /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src \
      python /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/examples/prefetch_dp_requests.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count
from typing import Callable, Sequence
import heapq
import logging

import numpy as np

from power_sharpening.common.cut_distribution import (
    compute_entropy_cut_policy,
    draw_uniform_cut_index,
    summarize_cut_policy,
)
from power_sharpening.common.prefetch_subtree import (
    PRINT_MAX_WIDTH,
    Node,
    build_and_sample_subtree,
    build_subtree,
    collect_batch_cut_indicesv3,
    sample_mh_tansition_pathwise,
    subtree_to_rich_tree,
)


logger = logging.getLogger("[subtree prefetching DP]")


# The epsilon the branch model is calibrated at (Definition
# `epsilon-pruned-continuation`). It never enters the recurrence: it only fixes
# what "certain accept" and "certain reject" mean for the predictor, so it is a
# documented constant rather than an argument.
DEFAULT_EPSILON = 0.01

# Ties in the Bellman split are resolved on a second key, so the comparison
# needs a tolerance: the two candidate values are different sums of the same
# floats and agree only to rounding.
_TIE_TOL = 1e-12


# ---------------------------------------------------------------------------
# Predicted branch probabilities
# ---------------------------------------------------------------------------

def midpoint_branch_probability(
    p_certain_reject: float,
    p_certain_accept: float,
) -> float:
    """Return the predicted acceptance-branch probability from its two certainty classes.

    Lemma `midpoint-branch-probability`: the class probabilities only pin
    ``E[h(v_l;eps) | u]`` down to an interval, and the estimate with the
    smallest worst-case error is that interval's midpoint,

        h_hat(v_l; eps) = 1/2 [1 + P_hat(A >= 1 - eps | u) - P_hat(A <= eps | u)].

    The rejection branch carries the complement ``1 - h_hat(v_l; eps)``.
    """
    return 0.5 * (1.0 + float(p_certain_accept) - float(p_certain_reject))


def class_probability_predictor(
    predict_classes: Callable[[int, int, int, int, int], tuple[float, float]],
) -> Callable[[int, int, int, int, int], float]:
    """Adapt a certainty-class model into a branch-probability predictor.

    The paper predicts the two certainty-class probabilities
    ``(P_hat(A <= eps | u), P_hat(A >= 1 - eps | u))`` with a small network fed
    pre-allocation features (token-wise likelihood and entropy from the
    proposal LLM). Wrap such a model with this adapter to obtain the
    ``h_hat(v_l; eps)`` that `BRANCH_PREDICTORS` entries return.
    """
    def predict(node_idx, cut, accept_cut, depth, num_accepts) -> float:
        p_reject, p_accept = predict_classes(
            node_idx, cut, accept_cut, depth, num_accepts
        )
        return midpoint_branch_probability(p_reject, p_accept)

    return predict


# Built-in predictors of the acceptance-branch probability ``h_hat(v_l; eps)``.
# Signature ``(node_idx, cut, accept_cut, depth, num_accepts) -> float in [0, 1]``,
# mirroring `RANK_FNS` in `prefetch_rank`: ``node_idx``, ``cut``, ``depth`` and
# ``num_accepts`` describe the decision u being expanded, and ``accept_cut`` is
# the cut index drawn for the acceptance child v_l -- i.e. the request that
# expanding u pays for. All of it is known before the proposal is generated,
# which is the point: `eq:rounded-branch-probability` needs the acceptance
# probability A(u, v_l), and that is unavailable while the batch is assembled.
BRANCH_PREDICTORS: dict[str, Callable[[int, int, int, int, int], float]] = {
    # The shared-mean baseline of `eq:empirical-hardened-branch-weights`: the
    # pooled rounded acceptance frequency over 6,518 scored LCB V6 edges,
    # (1,523 + 1,303 * 0.3738) / 6,518. No node-specific information.
    "shared_mean": lambda node_idx, cut, accept_cut, depth, num_accepts: 0.3084,
    # The midpoint rule at the pooled certainty classes of
    # `eq:calibrated-certainty-classes` (0.4 certain reject, 0.4 certain
    # accept, 0.2 uncertain). Symmetric classes give exactly 1/2, so this is
    # the "no directional information" reference rather than a second
    # calibration of the same number.
    "calibrated_classes": lambda node_idx, cut, accept_cut, depth, num_accepts:
        midpoint_branch_probability(0.4, 0.4),
    # Both branches equally likely; the DP then allocates on tree structure
    # (feasibility and depth) alone.
    "uniform": lambda node_idx, cut, accept_cut, depth, num_accepts: 0.5,
}


# ---------------------------------------------------------------------------
# Candidate decisions (the set I_max)
# ---------------------------------------------------------------------------

@dataclass
class PrefetchCandidate(object):
    """One decision the budget may be spent on, i.e. one element of ``I_max``.

    Expanding this node issues exactly one request: the proposal for its
    acceptance child ``2 * node_idx + 1`` at cut index `accept_cut`. The
    rejection child ``2 * node_idx + 2`` keeps the parent's state and costs
    nothing.
    """

    node_idx: int  # BFS index in the prefetching tree; root is 0
    cut: int  # this decision's own cut index (the root's is seq_len)
    accept_cut: int  # cut drawn for the acceptance child; the request
    depth: int  # edges from the subtree root
    num_accepts: int  # acceptance edges on the path from the root
    h_left: float  # h_hat(v_l; eps), the predicted acceptance-branch probability
    reach: float  # R_hat(u; eps), the predicted probability of reaching u
    left: int = -1  # slot of the acceptance child, -1 when it is not a candidate
    right: int = -1  # slot of the rejection child, -1 when it is not a candidate


def build_candidate_decisions(
    max_batch_size: int,
    prompt_len: int,
    seq_len: int,
    mcmc_steps: int,
    branch_predictor: str | Callable[[int, int, int, int, int], float] = "shared_mean",
    rng: np.random.Generator | None = None,
    cut_dist_type: str = "uniform",
    cut_dist_param: float = None,
    cut_entropies: np.ndarray = None,
    max_candidate_nodes: int | None = None,
) -> tuple[list[PrefetchCandidate], np.ndarray | None]:
    """Materialize ``I_max`` as a list of candidate decisions, parents first.

    A node joins ``I_max`` when the depth limit permits it and the cut index
    drawn for its acceptance child satisfies Condition `cond:prefetch`
    (``c' <= c``), so that its proposal can be generated from an already
    available prefix. An infeasible node is *not* a decision, and ancestor
    closure then excludes its whole subtree -- unlike the best-first collector
    in `collect_batch_cut_indicesv3`, which keeps expanding below such a node
    even though `build_subtree` cannot attach those requests to the tree.

    Two limits bound the depth. `mcmc_steps` caps the tree itself, so decisions
    live at depths ``0 .. mcmc_steps - 1`` (a decision needs both children to
    exist). Ancestor closure caps it again: selecting a decision at depth ``d``
    also selects its ``d`` ancestors, so ``d <= max_batch_size - 1``.

    Nodes are expanded best-first by predicted reaching weight
    ``R_hat(u; eps)``. Since ``R_hat`` cannot increase along a path, this pops
    them in globally descending ``R_hat`` order and always pops a parent before
    its children, so truncating at `max_candidate_nodes` keeps exactly the
    highest-weight candidates. Any cap of at least `max_batch_size` therefore
    leaves the optimum of `eq:prefetch-selection-objective` inside the returned
    set.

    Returns the candidate list (slot ``i`` is the ``i``-th popped node, so
    parents precede their children) and the entropy-cut law, or ``None`` under
    a uniform cut law.
    """
    if rng is None:
        rng = np.random.default_rng()
    if isinstance(branch_predictor, str):
        predict = BRANCH_PREDICTORS[branch_predictor]
    else:
        predict = branch_predictor

    entropy_cut_probabilities = None
    if cut_dist_type == "entropy":
        if cut_dist_param is None:
            cut_dist_param = 4
        entropy_cut_probabilities = compute_entropy_cut_policy(
            cut_entropies,
            beta=cut_dist_param,
        )
        logger.info(
            "  cut law: %s",
            summarize_cut_policy(
                cut_entropies,
                entropy_cut_probabilities,
                float(cut_dist_param),
                low=prompt_len,
            ),
        )
    elif cut_dist_type != "uniform":
        raise ValueError(
            f"unknown cut_dist_type {cut_dist_type!r}; expected 'uniform' or 'entropy'"
        )

    def draw_cut() -> int:
        if cut_dist_type == "uniform":
            return int(draw_uniform_cut_index(rng, prompt_len, seq_len - 1))
        return prompt_len + int(
            rng.choice(entropy_cut_probabilities.size, p=entropy_cut_probabilities)
        )

    candidates: list[PrefetchCandidate] = []
    max_decision_depth = min(mcmc_steps, max_batch_size) - 1
    if max_decision_depth < 0:
        return candidates, entropy_cut_probabilities
    if max_candidate_nodes is None:
        max_candidate_nodes = max(256, 16 * max_batch_size)

    # Heap entries: (-reach, tie_break, node_idx, cut, depth, num_accepts,
    # parent_slot, took_accept_edge). The negated weight pops the most reachable
    # node first; the counter keeps equal weights in insertion (BFS) order, so a
    # parent still precedes a child whose predicted factor is 1.
    tie_break = count()
    heap: list[tuple[float, int, int, int, int, int, int, bool]] = [
        (-1.0, next(tie_break), 0, seq_len, 0, 0, -1, False)
    ]

    while heap and len(candidates) < max_candidate_nodes:
        neg_reach, _, node_idx, cut, depth, num_accepts, parent_slot, took_accept = (
            heapq.heappop(heap)
        )
        if depth > max_decision_depth:
            # Beyond the tree or unaffordable under ancestor closure: the node
            # still exists as a leaf of the prefetched subtree, but it can never
            # be expanded, so it is not a candidate.
            continue

        # The request this expansion would pay for. Drawing it is what decides
        # feasibility, so it happens before the node becomes a candidate.
        accept_cut = draw_cut()
        if accept_cut > cut:
            # Condition `cond:prefetch` fails: the acceptance child's prefix is
            # not available yet, so this proposal cannot join the batch. The
            # node is not expandable, and ancestor closure drops its subtree.
            continue

        reach = -neg_reach
        h_left = float(predict(node_idx, cut, accept_cut, depth, num_accepts))
        slot = len(candidates)
        candidates.append(
            PrefetchCandidate(
                node_idx=node_idx,
                cut=cut,
                accept_cut=accept_cut,
                depth=depth,
                num_accepts=num_accepts,
                h_left=h_left,
                reach=reach,
            )
        )
        if parent_slot >= 0:
            if took_accept:
                candidates[parent_slot].left = slot
            else:
                candidates[parent_slot].right = slot

        # The acceptance child adopts the resampled proposal and its cut index;
        # the rejection child keeps this node's state, and therefore its cut.
        heapq.heappush(
            heap,
            (
                -(reach * h_left), next(tie_break), 2 * node_idx + 1,
                accept_cut, depth + 1, num_accepts + 1, slot, True,
            ),
        )
        heapq.heappush(
            heap,
            (
                -(reach * (1.0 - h_left)), next(tie_break), 2 * node_idx + 2,
                cut, depth + 1, num_accepts, slot, False,
            ),
        )

    return candidates, entropy_cut_probabilities


# ---------------------------------------------------------------------------
# The budget allocation itself
# ---------------------------------------------------------------------------

def solve_budget_allocation(
    candidates: Sequence[PrefetchCandidate],
    max_batch_size: int,
) -> tuple[list[int], float]:
    """Solve `eq:predictive-dp-bellman` over `candidates` and backtrack the selection.

    ``V_hat(u, n; eps)`` is the largest predicted transition count reachable
    from ``u`` with at most ``n`` requests:

        V_hat(u, n) = 0                                   if u not expandable or n = 0
        V_hat(u, n) = 1 + max_{0 <= m <= n-1} {
                          h_hat(v_l) V_hat(v_l, m)
                        + h_hat(v_r) V_hat(v_r, n - 1 - m) }   otherwise

    One request is reserved for ``u`` itself and the remaining ``n - 1`` are
    split between the two child subtrees before the outcome is known. The value
    is conditional on *starting* at ``u``, so no root-to-node reaching weight
    enters the recurrence -- it enters through the ``h_hat`` factors above.

    Splits are ranked by ``(value, -requests)``, so among equally valuable
    allocations the one issuing fewer proposals wins. That matters under a
    predictor that returns a hard 0 or 1: a branch with ``R_hat(u; eps) = 0``
    adds nothing to the objective while still costing a request, and this rule
    is what stops the budget leaking into it.

    Returns the selected slots and the optimum ``V_hat(v_0, N_pf; eps)``.
    """
    if not candidates or max_batch_size <= 0:
        return [], 0.0

    values: list[np.ndarray] = [None] * len(candidates)
    requests: list[np.ndarray] = [None] * len(candidates)
    splits: list[np.ndarray] = [None] * len(candidates)

    # Slot order is pop order, and a parent is always popped before its
    # children, so walking it backwards evaluates children first.
    for slot in reversed(range(len(candidates))):
        candidate = candidates[slot]
        budget = max_batch_size - candidate.depth
        value = np.zeros(budget + 1, dtype=np.float64)
        request = np.zeros(budget + 1, dtype=np.int64)
        split = np.zeros(budget + 1, dtype=np.int64)

        # A child table covers budgets 0 .. budget - 1, which is exactly the
        # range the split can hand it. A missing child contributes zeros.
        left_value = values[candidate.left] if candidate.left >= 0 else np.zeros(budget)
        left_request = (
            requests[candidate.left] if candidate.left >= 0
            else np.zeros(budget, dtype=np.int64)
        )
        right_value = values[candidate.right] if candidate.right >= 0 else np.zeros(budget)
        right_request = (
            requests[candidate.right] if candidate.right >= 0
            else np.zeros(budget, dtype=np.int64)
        )

        h_left = candidate.h_left
        h_right = 1.0 - h_left
        for n in range(1, budget + 1):
            best_value = -1.0
            best_requests = 0
            best_split = 0
            for m in range(n):
                child_value = (
                    h_left * left_value[m] + h_right * right_value[n - 1 - m]
                )
                child_requests = int(left_request[m] + right_request[n - 1 - m])
                if child_value > best_value + _TIE_TOL or (
                    child_value > best_value - _TIE_TOL
                    and child_requests < best_requests
                ):
                    best_value = child_value
                    best_requests = child_requests
                    best_split = m
            # The leading 1 counts this decision's own transition, which it
            # completes whether its proposal is accepted or rejected.
            value[n] = 1.0 + best_value
            request[n] = 1 + best_requests
            split[n] = best_split

        values[slot] = value
        requests[slot] = request
        splits[slot] = split

    # Backtrack the maximizing splits from (v_0, N_pf).
    selected: list[int] = []
    stack: list[tuple[int, int]] = [(0, max_batch_size)]
    while stack:
        slot, budget = stack.pop()
        if slot < 0 or budget <= 0:
            continue
        budget = min(budget, max_batch_size - candidates[slot].depth)
        if budget <= 0 or values[slot][budget] <= 0.0:
            continue
        selected.append(slot)
        split = int(splits[slot][budget])
        stack.append((candidates[slot].left, split))
        stack.append((candidates[slot].right, budget - 1 - split))

    return selected, float(values[0][max_batch_size])


@dataclass
class PrefetchPlan(object):
    """The selected requests plus what the DP predicted for them."""

    parent_to_child: dict[int, int] = field(default_factory=dict)
    node_to_batch_idx: dict[int, int] = field(default_factory=dict)
    batch_cut_indices: list[int] = field(default_factory=list)
    reaching_weights: list[float] = field(default_factory=list)
    predicted_transitions: float = 0.0
    num_candidates: int = 0

    def as_batch_tuple(self) -> tuple[dict[int, int], dict[int, int], list[int]]:
        """Return the triple `collect_batch_cut_indicesv3` returns."""
        return self.parent_to_child, self.node_to_batch_idx, self.batch_cut_indices


def plan_prefetch_subtree_dp(
    max_batch_size: int,
    prompt_len: int,
    seq_len: int,
    mcmc_steps: int,
    branch_predictor: str | Callable[[int, int, int, int, int], float] = "shared_mean",
    rng: np.random.Generator | None = None,
    cut_dist_type: str = "uniform",
    cut_dist_param: float = None,
    cut_entropies: np.ndarray = None,
    max_candidate_nodes: int | None = None,
) -> PrefetchPlan:
    """Choose the request batch by dynamic programming and report the prediction.

    Builds ``I_max`` with `build_candidate_decisions`, solves
    `eq:prefetch-selection-objective` with `solve_budget_allocation`, and emits
    the selected decisions in BFS order, so ancestors precede descendants in the
    batch as they do under `collect_batch_cut_indicesv3`.
    """
    candidates, entropy_cut_probabilities = build_candidate_decisions(
        max_batch_size=max_batch_size,
        prompt_len=prompt_len,
        seq_len=seq_len,
        mcmc_steps=mcmc_steps,
        branch_predictor=branch_predictor,
        rng=rng,
        cut_dist_type=cut_dist_type,
        cut_dist_param=cut_dist_param,
        cut_entropies=cut_entropies,
        max_candidate_nodes=max_candidate_nodes,
    )
    selected, predicted_transitions = solve_budget_allocation(
        candidates,
        max_batch_size,
    )

    plan = PrefetchPlan(
        predicted_transitions=predicted_transitions,
        num_candidates=len(candidates),
    )
    for slot in sorted(selected, key=lambda i: (candidates[i].depth, candidates[i].node_idx)):
        candidate = candidates[slot]
        accept = 2 * candidate.node_idx + 1
        plan.parent_to_child[candidate.node_idx] = accept
        plan.node_to_batch_idx[accept] = len(plan.batch_cut_indices)
        plan.batch_cut_indices.append(candidate.accept_cut)
        plan.reaching_weights.append(candidate.reach)

    logger.info(
        "  DP selected %d/%d candidate decisions under budget %d; "
        "predicted transitions %.4f (sum of reaching weights %.4f)",
        len(plan.batch_cut_indices),
        len(candidates),
        max_batch_size,
        predicted_transitions,
        float(sum(plan.reaching_weights)),
    )
    if cut_dist_type == "entropy":
        logger.info(
            "  drawn cut indices (batch, BFS order): %s; "
            "sampling probabilities (same order): %s",
            plan.batch_cut_indices,
            [
                float(entropy_cut_probabilities[cut - prompt_len])
                for cut in plan.batch_cut_indices
            ],
        )
    return plan


def collect_batch_cut_indices_dp(
    max_batch_size: int,
    prompt_len: int,
    seq_len: int,
    mcmc_steps: int,
    branch_predictor: str | Callable[[int, int, int, int, int], float] = "shared_mean",
    rng: np.random.Generator | None = None,
    cut_dist_type: str = "uniform",
    cut_dist_param: float = None,
    cut_entropies: np.ndarray = None,
    max_candidate_nodes: int | None = None,
) -> tuple[dict[int, int], dict[int, int], list[int]]:
    """Collect the prefetch batch by dynamic programming.

    Drop-in replacement for `collect_batch_cut_indicesv3`: same triple, same
    cut laws, same feasibility condition. The difference is how the budget is
    spent -- a frontier priority is replaced by the optimum of
    `eq:prefetch-selection-objective`, the ancestor-closed selection maximizing
    the predicted number of completed MH transitions.

    `branch_predictor` names an entry of `BRANCH_PREDICTORS` or is a callable
    ``(node_idx, cut, accept_cut, depth, num_accepts) -> h_hat(v_l; eps)``; wrap
    a certainty-class model with `class_probability_predictor`.

    `cut_dist_type` selects the law for each acceptance child's cut index.
    Uniform cuts span ``[prompt_len, seq_len - 1]``; entropy cuts map adjacent
    entropy jumps to ``[prompt_len, seq_len - 2]``, need `cut_entropies` (one
    base-model predictive entropy per candidate cut), and are state-dependent,
    so callers MUST add the matching `entropy_cut_log_ratio` term to each
    acceptance probability. `cut_dist_param` is the entropy cut power and
    defaults to 4.0.

    The DP draws a cut for every candidate it materializes, not only for the
    requests it keeps, so it consumes more of `rng` than the best-first
    collector does at the same budget.

    Returns:
        parent_to_child:   parent -> accepted left-child node index.
        node_to_batch_idx: accept-child (left) node index -> batch position.
        batch_cut_indices: cut indices in batch (BFS) order.
    """
    return plan_prefetch_subtree_dp(
        max_batch_size=max_batch_size,
        prompt_len=prompt_len,
        seq_len=seq_len,
        mcmc_steps=mcmc_steps,
        branch_predictor=branch_predictor,
        rng=rng,
        cut_dist_type=cut_dist_type,
        cut_dist_param=cut_dist_param,
        cut_entropies=cut_entropies,
        max_candidate_nodes=max_candidate_nodes,
    ).as_batch_tuple()


__all__ = [
    "BRANCH_PREDICTORS",
    "DEFAULT_EPSILON",
    "PRINT_MAX_WIDTH",
    "Node",
    "PrefetchCandidate",
    "PrefetchPlan",
    "build_and_sample_subtree",
    "build_candidate_decisions",
    "build_subtree",
    "class_probability_predictor",
    "collect_batch_cut_indices_dp",
    "collect_batch_cut_indicesv3",
    "midpoint_branch_probability",
    "plan_prefetch_subtree_dp",
    "sample_mh_tansition_pathwise",
    "solve_budget_allocation",
    "subtree_to_rich_tree",
]
