"""Predictive subtree allocation: choose the prefetch requests by dynamic programming.

Implements Appendix C.2 of the paper (``tex/C.2.predictive_dp.tex``):

* "Predictive Subtree Allocation": a classifier predicts, before generation,
  whether each candidate node v is eps-certain accept, eps-certain reject or
  eps-uncertain; the midpoint rule turns those class probabilities into the
  predicted branch probabilities ``h_hat(v^+; eps)`` and ``h_hat(v^-; eps)``
  (`eq:predictive-rounded-branch-probabilities`), and the selection ``I``
  maximizes the predicted transition count ``F_hat_I(v_0; eps)`` within the
  request budget ``N_pf`` (`eq:prefetch-selection-objective`).
* "Optimal Budget Allocation via Dynamic Programming": the value
  ``V(v, n; eps)`` and its Bellman recurrence (`eq:predictive-dp-bellman`),
  whose root value ``V(v_0, N_pf; eps)`` is that optimum.

The classifier is the one Appendix E.2 selects (``tex/E.2.extra-experiment.tex``,
`apx:acceptance-predictors`): a three-class logistic regression with the cut
fraction ``c_j / T`` as the only feature.

Notation (paper -> code):

    v, v^+, v^-                       `CandidateNode`, its `plus` / `minus` child slots
    v_0                               slot 0 of the candidate list
    c_i                               `held_cut`, the cut of the state held at v
    c_j                               `request_cut`, the cut of the request at v^+
    T                                 response length, ``seq_len - prompt_len``
    P_hat(v is eps-certain accept)    `p_certain_accept`
    P_hat(v is eps-certain reject)    `p_certain_reject`
    h_hat(v^+; eps), h_hat(v^-; eps)  `h_plus`, `h_minus`
    R_hat(v; eps)                     `reaching_weight`
    I_max                             `build_candidate_set`
    I                                 the slots `solve_predictive_dp` selects
    N_pf                              `prefetch_budget` (`max_batch_size` in the drop-in API)
    K                                 the depth cap, `mcmc_steps`
    V(v, n; eps), m                   `V[slot][n]`, the maximizing split `best_m[slot][n]`

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


# The eps the certainty classes are defined at: v is eps-certain accept if
# A >= 1 - eps, eps-certain reject if A <= eps, and eps-uncertain otherwise.
# It never enters the recurrence; it only fixes the labels the classifier was
# trained on (Appendix E.2 uses eps = 0.01).
EPSILON = 0.01

# The traversal-rule name samplers accept next to the `RANK_FNS` keys of
# `prefetch_rank`: it selects `collect_batch_cut_indices_dp` with the default
# certainty classifier instead of the best-first collector.
PREDICTIVE_DP_RANK = "predictive_dp"

# Ties in the Bellman split are resolved on a second key, so the comparison
# needs a tolerance: the two candidate values are different sums of the same
# floats and agree only to rounding.
_TIE_TOL = 1e-12


# ---------------------------------------------------------------------------
# Predicted branch probabilities h_hat(v^+; eps), h_hat(v^-; eps)
# ---------------------------------------------------------------------------

# Class order of the classifier's softmax, as in Appendix E.2 (y = 1, 2, 3).
CERTAINTY_CLASSES = ("certain accept", "certain reject", "uncertain")

# Maps the cut fraction c_j / T of the request at v^+ to
# ``(P_hat(v is eps-certain accept), P_hat(v is eps-certain reject))``.
CertaintyClassifier = Callable[[float], tuple[float, float]]


def midpoint_branch_probability(
    p_certain_accept: float,
    p_certain_reject: float,
) -> float:
    """Return ``h_hat(v^+; eps)`` from the two certainty-class probabilities.

    `eq:predictive-rounded-branch-probabilities`:

        h_hat(v^+; eps) = 1/2 [1 + P_hat(v is eps-certain accept)
                                 - P_hat(v is eps-certain reject)],
        h_hat(v^-; eps) = 1 - h_hat(v^+; eps).

    The average of 1 (certain accept), 0 (certain reject) and 1/2 (uncertain,
    the midpoint of [eps, 1 - eps]) weighted by the class probabilities; Lemma
    `midpoint-branch-probability` shows it has the smallest worst-case error.
    """
    return 0.5 * (1.0 + float(p_certain_accept) - float(p_certain_reject))


@dataclass(frozen=True)
class CutFractionLogisticRegression(object):
    """Three-class logistic regression on the cut fraction ``c_j / T``.

    Appendix E.2: a linear layer followed by a softmax over
    `CERTAINTY_CLASSES`,

        P_hat(y = k | v) = softmax_k(intercept_k + slope_k * c_j / T),

    so ``P_hat(certain accept) + P_hat(certain reject) <= 1`` by construction.
    The coefficients act on the raw cut fraction; the standardization used in
    training is folded into them.
    """

    intercept: tuple[float, float, float]
    slope: tuple[float, float, float]

    def predict_class_probabilities(
        self, cut_fraction: float
    ) -> tuple[float, float, float]:
        """Return ``P_hat(y = k | v)`` for k in `CERTAINTY_CLASSES`."""
        logits = np.asarray(self.intercept) + np.asarray(self.slope) * float(cut_fraction)
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()
        return float(probs[0]), float(probs[1]), float(probs[2])

    def __call__(self, cut_fraction: float) -> tuple[float, float]:
        p_certain_accept, p_certain_reject, _ = self.predict_class_probabilities(
            cut_fraction
        )
        return p_certain_accept, p_certain_reject


# The fitted classifiers of Appendix E.2, one per sampler. Each is the
# cut-fraction-only logistic regression (L2 penalty 1 on the standardized
# weight) refit on all scored nodes of its run, whose 5-fold held-out scores
# the paper reports: Qwen3.5-9B on LCB V6, N_pf = 10, alpha = 4, K = 100,
# T = 1,024, eps = 0.01, default BFS traversal, first 20 prompts.
CERTAINTY_CLASSIFIERS: dict[str, CutFractionLogisticRegression] = {
    # PreSTO (uniform cuts), 4,621 nodes; held-out accuracy 0.643, log loss 0.819.
    "power_mh": CutFractionLogisticRegression(
        intercept=(-0.780255, 2.041824, -1.261570),
        slope=(1.128515, -2.920097, 1.791582),
    ),
    # PreSTO-EntropyCut (entropy cut power 4.0), 4,533 nodes; held-out
    # accuracy 0.625, log loss 0.884.
    "entropy_cut": CutFractionLogisticRegression(
        intercept=(-0.822343, 1.783249, -0.960906),
        slope=(0.936869, -2.248380, 1.311511),
    ),
}


def resolve_certainty_classifier(
    certainty_classifier: str | CertaintyClassifier | None,
    cut_dist_type: str,
) -> CertaintyClassifier:
    """Look up a classifier by name; ``None`` picks the one fit under `cut_dist_type`."""
    if certainty_classifier is None:
        certainty_classifier = "entropy_cut" if cut_dist_type == "entropy" else "power_mh"
    if isinstance(certainty_classifier, str):
        return CERTAINTY_CLASSIFIERS[certainty_classifier]
    return certainty_classifier


# ---------------------------------------------------------------------------
# The candidate set I_max
# ---------------------------------------------------------------------------

@dataclass
class CandidateNode(object):
    """One candidate node v in ``I_max``.

    Expanding v issues exactly one request: the proposal at its acceptance
    child ``v^+`` (BFS index ``2 * node_idx + 1``) at cut index `request_cut`.
    The rejection child ``v^-`` (``2 * node_idx + 2``) keeps the state held at
    v and costs nothing.
    """

    node_idx: int  # BFS index of v in the prefetching tree; v_0 is 0
    held_cut: int  # c_i, the cut of the state held at v (seq_len at v_0)
    request_cut: int  # c_j, the cut of the request at v^+
    depth: int  # edges from v_0
    num_accepts: int  # acceptance edges on the path from v_0 to v
    p_certain_accept: float  # P_hat(v is eps-certain accept)
    p_certain_reject: float  # P_hat(v is eps-certain reject)
    h_plus: float  # h_hat(v^+; eps)
    reaching_weight: float  # R_hat(v; eps)
    plus: int = -1  # slot of v^+, -1 when v^+ is not in I_max
    minus: int = -1  # slot of v^-, -1 when v^- is not in I_max

    @property
    def h_minus(self) -> float:
        """``h_hat(v^-; eps) = 1 - h_hat(v^+; eps)``."""
        return 1.0 - self.h_plus


def build_candidate_set(
    prefetch_budget: int,
    prompt_len: int,
    seq_len: int,
    mcmc_steps: int,
    certainty_classifier: str | CertaintyClassifier | None = None,
    rng: np.random.Generator | None = None,
    cut_dist_type: str = "uniform",
    cut_dist_param: float = None,
    cut_entropies: np.ndarray = None,
    max_candidate_nodes: int | None = None,
) -> tuple[list[CandidateNode], np.ndarray | None]:
    """Materialize ``I_max`` as a list of candidate nodes, parents first.

    A node v joins ``I_max`` when the depth cap permits it and the cut ``c_j``
    drawn for its request satisfies Condition `cond:prefetch` (``c_j <= c_i``),
    so that the proposal at v^+ can be generated from an already available
    prefix. A node outside ``I_max`` has ``V = 0`` (the first boundary case of
    `eq:predictive-dp-bellman`), and ancestor closure then excludes its whole
    subtree -- unlike the best-first collector in `collect_batch_cut_indicesv3`,
    which keeps expanding below such a node even though `build_subtree` cannot
    attach those requests to the tree.

    Two limits bound the depth. The depth cap ``K`` (`mcmc_steps`) caps the
    tree itself, so candidate nodes live at depths ``0 .. K - 1`` (expanding v
    needs both children to exist). Ancestor closure caps it again: selecting a
    node at depth ``d`` also selects its ``d`` ancestors, so ``d <= N_pf - 1``.

    For each candidate v the classifier reads the cut fraction ``c_j / T`` of
    its request, with ``T = seq_len - prompt_len`` and cuts measured from the
    end of the prompt, and returns the class probabilities that
    `midpoint_branch_probability` turns into ``h_hat(v^+; eps)``.

    Nodes are expanded best-first by predicted reaching weight
    ``R_hat(v; eps)``. Since ``R_hat`` cannot increase along a path, this pops
    them in globally descending ``R_hat`` order and always pops a parent before
    its children, so truncating at `max_candidate_nodes` keeps exactly the
    highest-weight candidates. Any cap of at least ``N_pf`` therefore leaves the
    optimum of `eq:prefetch-selection-objective` inside the returned set.

    Returns the candidate list (slot ``i`` is the ``i``-th popped node, so
    parents precede their children) and the entropy-cut law, or ``None`` under
    a uniform cut law.
    """
    if rng is None:
        rng = np.random.default_rng()
    classify = resolve_certainty_classifier(certainty_classifier, cut_dist_type)

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

    response_len = max(seq_len - prompt_len, 1)  # T

    candidates: list[CandidateNode] = []
    max_depth = min(mcmc_steps, prefetch_budget) - 1
    if max_depth < 0:
        return candidates, entropy_cut_probabilities
    if max_candidate_nodes is None:
        max_candidate_nodes = max(256, 16 * prefetch_budget)

    # Heap entries: (-R_hat(v), tie_break, node_idx, c_i, depth, num_accepts,
    # parent_slot, is_plus_child). The negated weight pops the most reachable
    # node first; the counter keeps equal weights in insertion (BFS) order, so a
    # parent still precedes a child whose branch probability is 1.
    tie_break = count()
    heap: list[tuple[float, int, int, int, int, int, int, bool]] = [
        (-1.0, next(tie_break), 0, seq_len, 0, 0, -1, False)
    ]

    while heap and len(candidates) < max_candidate_nodes:
        neg_reaching_weight, _, node_idx, held_cut, depth, num_accepts, parent_slot, is_plus = (
            heapq.heappop(heap)
        )
        if depth > max_depth:
            # Beyond the depth cap or unaffordable under ancestor closure: the
            # node still exists as a leaf of the prefetched subtree, but it can
            # never be expanded, so it is not in I_max.
            continue

        # The request that expanding v pays for. Drawing c_j is what decides
        # feasibility, so it happens before v becomes a candidate.
        request_cut = draw_cut()
        if request_cut > held_cut:
            # Condition `cond:prefetch` fails: the prefix of v^+ is not
            # available yet, so this proposal cannot join the batch. v is not
            # in I_max, and ancestor closure drops its subtree.
            continue

        reaching_weight = -neg_reaching_weight
        p_certain_accept, p_certain_reject = classify(
            (request_cut - prompt_len) / response_len
        )
        h_plus = midpoint_branch_probability(p_certain_accept, p_certain_reject)
        slot = len(candidates)
        candidates.append(
            CandidateNode(
                node_idx=node_idx,
                held_cut=held_cut,
                request_cut=request_cut,
                depth=depth,
                num_accepts=num_accepts,
                p_certain_accept=float(p_certain_accept),
                p_certain_reject=float(p_certain_reject),
                h_plus=h_plus,
                reaching_weight=reaching_weight,
            )
        )
        if parent_slot >= 0:
            if is_plus:
                candidates[parent_slot].plus = slot
            else:
                candidates[parent_slot].minus = slot

        # R_hat(v^+) = R_hat(v) h_hat(v^+) and R_hat(v^-) = R_hat(v) h_hat(v^-).
        # v^+ adopts the proposal and its cut c_j; v^- keeps the state held at
        # v, and therefore its cut c_i.
        heapq.heappush(
            heap,
            (
                -(reaching_weight * h_plus), next(tie_break), 2 * node_idx + 1,
                request_cut, depth + 1, num_accepts + 1, slot, True,
            ),
        )
        heapq.heappush(
            heap,
            (
                -(reaching_weight * (1.0 - h_plus)), next(tie_break), 2 * node_idx + 2,
                held_cut, depth + 1, num_accepts, slot, False,
            ),
        )

    return candidates, entropy_cut_probabilities


# ---------------------------------------------------------------------------
# The dynamic program V(v, n; eps)
# ---------------------------------------------------------------------------

def solve_predictive_dp(
    candidates: Sequence[CandidateNode],
    prefetch_budget: int,
) -> tuple[list[int], float]:
    """Solve `eq:predictive-dp-bellman` over ``I_max`` and backtrack the selection ``I``.

    ``V(v, n; eps)`` is the largest predicted transition count attainable from
    v with at most n requests (`eq:predictive-dp-value`):

        V(v, n) = 0                                      if v not in I_max
        V(v, n) = 0                                      if n = 0
        V(v, n) = 1 + max_{0 <= m <= n-1} {
                      h_hat(v^+) V(v^+, m) + h_hat(v^-) V(v^-, n - 1 - m) }   else

    One request expands v itself and the remaining ``n - 1`` are split between
    the two child subtrees before the outcome at v is known. By Proposition
    `predictive-dp-optimality`, ``V(v_0, N_pf; eps)`` is the optimum of
    `eq:prefetch-selection-objective`.

    Splits are ranked by ``(value, -requests)``, so among equally valuable
    splits the one issuing fewer requests wins. That matters when
    ``h_hat`` is exactly 0 or 1: a node with ``R_hat(v; eps) = 0`` contributes
    nothing to the objective but still costs a request, and this rule leaves it
    and its descendants unexpanded.

    Returns the selected slots ``I`` and ``V(v_0, N_pf; eps)``.
    """
    if not candidates or prefetch_budget <= 0:
        return [], 0.0

    V: list[np.ndarray] = [None] * len(candidates)
    num_requests: list[np.ndarray] = [None] * len(candidates)
    best_m: list[np.ndarray] = [None] * len(candidates)

    # Slot order is pop order, and a parent is always popped before its
    # children, so walking it backwards evaluates children first.
    for slot in reversed(range(len(candidates))):
        v = candidates[slot]
        # Selecting v also selects its depth-many ancestors, so at most
        # N_pf - depth requests can land in the subtree rooted at v.
        max_n = prefetch_budget - v.depth
        V_v = np.zeros(max_n + 1, dtype=np.float64)
        num_requests_v = np.zeros(max_n + 1, dtype=np.int64)
        best_m_v = np.zeros(max_n + 1, dtype=np.int64)

        # V(v^+, .) and V(v^-, .) cover n = 0 .. max_n - 1, exactly the range a
        # split can hand a child. A child outside I_max has V = 0.
        V_plus = V[v.plus] if v.plus >= 0 else np.zeros(max_n)
        num_requests_plus = (
            num_requests[v.plus] if v.plus >= 0 else np.zeros(max_n, dtype=np.int64)
        )
        V_minus = V[v.minus] if v.minus >= 0 else np.zeros(max_n)
        num_requests_minus = (
            num_requests[v.minus] if v.minus >= 0 else np.zeros(max_n, dtype=np.int64)
        )

        h_plus = v.h_plus
        h_minus = v.h_minus
        for n in range(1, max_n + 1):
            best_value = -1.0
            best_requests = 0
            best_split = 0
            for m in range(n):
                value = h_plus * V_plus[m] + h_minus * V_minus[n - 1 - m]
                requests = int(num_requests_plus[m] + num_requests_minus[n - 1 - m])
                if value > best_value + _TIE_TOL or (
                    value > best_value - _TIE_TOL and requests < best_requests
                ):
                    best_value = value
                    best_requests = requests
                    best_split = m
            # The leading 1 counts the transition at v, which is completed
            # whether its proposal is accepted or rejected.
            V_v[n] = 1.0 + best_value
            num_requests_v[n] = 1 + best_requests
            best_m_v[n] = best_split

        V[slot] = V_v
        num_requests[slot] = num_requests_v
        best_m[slot] = best_m_v

    # Backtrack the maximizing splits from (v_0, N_pf).
    selection: list[int] = []
    stack: list[tuple[int, int]] = [(0, prefetch_budget)]
    while stack:
        slot, n = stack.pop()
        if slot < 0 or n <= 0:
            continue
        n = min(n, prefetch_budget - candidates[slot].depth)
        if n <= 0 or V[slot][n] <= 0.0:
            continue
        selection.append(slot)
        m = int(best_m[slot][n])
        stack.append((candidates[slot].plus, m))
        stack.append((candidates[slot].minus, n - 1 - m))

    return selection, float(V[0][prefetch_budget])


@dataclass
class PrefetchPlan(object):
    """The selected requests plus what the DP predicted for them."""

    parent_to_child: dict[int, int] = field(default_factory=dict)
    node_to_batch_idx: dict[int, int] = field(default_factory=dict)
    batch_cut_indices: list[int] = field(default_factory=list)
    reaching_weights: list[float] = field(default_factory=list)  # R_hat(v; eps), batch order
    h_plus: list[float] = field(default_factory=list)  # h_hat(v^+; eps), batch order
    predicted_transition_count: float = 0.0  # V(v_0, N_pf; eps) = F_hat_I(v_0; eps)
    num_candidates: int = 0  # |I_max|

    def as_batch_tuple(self) -> tuple[dict[int, int], dict[int, int], list[int]]:
        """Return the triple `collect_batch_cut_indicesv3` returns."""
        return self.parent_to_child, self.node_to_batch_idx, self.batch_cut_indices


def plan_prefetch_subtree_dp(
    prefetch_budget: int,
    prompt_len: int,
    seq_len: int,
    mcmc_steps: int,
    certainty_classifier: str | CertaintyClassifier | None = None,
    rng: np.random.Generator | None = None,
    cut_dist_type: str = "uniform",
    cut_dist_param: float = None,
    cut_entropies: np.ndarray = None,
    max_candidate_nodes: int | None = None,
) -> PrefetchPlan:
    """Choose the request batch by dynamic programming and report the prediction.

    Builds ``I_max`` with `build_candidate_set`, solves
    `eq:prefetch-selection-objective` with `solve_predictive_dp`, and emits
    the selected nodes in BFS order, so ancestors precede descendants in the
    batch as they do under `collect_batch_cut_indicesv3`.
    """
    candidates, entropy_cut_probabilities = build_candidate_set(
        prefetch_budget=prefetch_budget,
        prompt_len=prompt_len,
        seq_len=seq_len,
        mcmc_steps=mcmc_steps,
        certainty_classifier=certainty_classifier,
        rng=rng,
        cut_dist_type=cut_dist_type,
        cut_dist_param=cut_dist_param,
        cut_entropies=cut_entropies,
        max_candidate_nodes=max_candidate_nodes,
    )
    selection, predicted_transition_count = solve_predictive_dp(
        candidates,
        prefetch_budget,
    )

    plan = PrefetchPlan(
        predicted_transition_count=predicted_transition_count,
        num_candidates=len(candidates),
    )
    for slot in sorted(selection, key=lambda i: (candidates[i].depth, candidates[i].node_idx)):
        v = candidates[slot]
        v_plus = 2 * v.node_idx + 1
        plan.parent_to_child[v.node_idx] = v_plus
        plan.node_to_batch_idx[v_plus] = len(plan.batch_cut_indices)
        plan.batch_cut_indices.append(v.request_cut)
        plan.reaching_weights.append(v.reaching_weight)
        plan.h_plus.append(v.h_plus)

    logger.info(
        "  DP selected %d/%d candidate nodes under budget N_pf=%d; "
        "V(v_0, N_pf) = %.4f (sum of reaching weights %.4f)",
        len(plan.batch_cut_indices),
        len(candidates),
        prefetch_budget,
        predicted_transition_count,
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
    certainty_classifier: str | CertaintyClassifier | None = None,
    rng: np.random.Generator | None = None,
    cut_dist_type: str = "uniform",
    cut_dist_param: float = None,
    cut_entropies: np.ndarray = None,
    max_candidate_nodes: int | None = None,
) -> tuple[dict[int, int], dict[int, int], list[int]]:
    """Collect the prefetch batch by dynamic programming.

    Drop-in replacement for `collect_batch_cut_indicesv3`: same triple, same
    cut laws, same feasibility condition, and `max_batch_size` is ``N_pf``. The
    difference is how the budget is spent -- a frontier priority is replaced by
    the optimum of `eq:prefetch-selection-objective`, the ancestor-closed
    selection maximizing the predicted number of completed MH transitions.

    `certainty_classifier` names an entry of `CERTAINTY_CLASSIFIERS` or is a
    callable ``c_j / T -> (P_hat(certain accept), P_hat(certain reject))``.
    ``None`` uses the cut-fraction logistic regression fit under the same cut
    law: ``"entropy_cut"`` for entropy cuts, ``"power_mh"`` otherwise.

    `cut_dist_type` selects the law for each request's cut index. Uniform cuts
    span ``[prompt_len, seq_len - 1]``; entropy cuts map adjacent entropy jumps
    to ``[prompt_len, seq_len - 2]``, need `cut_entropies` (one base-model
    predictive entropy per candidate cut), and are state-dependent, so callers
    MUST add the matching `entropy_cut_log_ratio` term to each acceptance
    probability. `cut_dist_param` is the entropy cut power and defaults to 4.0.

    The DP draws a cut for every candidate it materializes, not only for the
    requests it keeps, so it consumes more of `rng` than the best-first
    collector does at the same budget.

    Returns:
        parent_to_child:   v -> v^+ node index.
        node_to_batch_idx: v^+ node index -> batch position.
        batch_cut_indices: cut indices c_j in batch (BFS) order.
    """
    return plan_prefetch_subtree_dp(
        prefetch_budget=max_batch_size,
        prompt_len=prompt_len,
        seq_len=seq_len,
        mcmc_steps=mcmc_steps,
        certainty_classifier=certainty_classifier,
        rng=rng,
        cut_dist_type=cut_dist_type,
        cut_dist_param=cut_dist_param,
        cut_entropies=cut_entropies,
        max_candidate_nodes=max_candidate_nodes,
    ).as_batch_tuple()


__all__ = [
    "CERTAINTY_CLASSES",
    "CERTAINTY_CLASSIFIERS",
    "CandidateNode",
    "CertaintyClassifier",
    "CutFractionLogisticRegression",
    "EPSILON",
    "PREDICTIVE_DP_RANK",
    "PRINT_MAX_WIDTH",
    "Node",
    "PrefetchPlan",
    "build_and_sample_subtree",
    "build_candidate_set",
    "build_subtree",
    "collect_batch_cut_indices_dp",
    "collect_batch_cut_indicesv3",
    "midpoint_branch_probability",
    "plan_prefetch_subtree_dp",
    "resolve_certainty_classifier",
    "sample_mh_tansition_pathwise",
    "solve_predictive_dp",
    "subtree_to_rich_tree",
]
