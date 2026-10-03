"""Collect prefetch requests with the predictive dynamic program, on sample inputs.

Builds a synthetic MH state (no model, no GPU) and runs
``power_sharpening.common.prefetch_subtree_dynamic_programming`` on it:

1. prints the request batch the DP selects, with each request's cut index,
   depth and predicted reaching weight;
2. checks the DP optimum against brute force over every ancestor-closed
   selection, and against best-first greedy on predicted reaching weight;
3. compares it with the best-first collector `collect_batch_cut_indicesv3`
   at the same budget and the same cut law;
4. sweeps the prefetch budget N_pf from 2 to 20, as in the paper's ablation;
5. shows a hard certainty predictor steering the budget onto the rejection
   spine, and assembles the selected requests into a real proposal subtree.

Run:
    uv run --project /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src \
      python /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/examples/prefetch_dp_requests.py
"""

import logging
import os
import sys

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Make ``src`` importable when running this file directly.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SRC_ROOT = os.path.join(_REPO_ROOT, "src")
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)

from power_sharpening.common.prefetch_subtree import (
    build_subtree,
    collect_batch_cut_indicesv3,
)
from power_sharpening.common.prefetch_subtree_dynamic_programming import (
    build_candidate_decisions,
    class_probability_predictor,
    collect_batch_cut_indices_dp,
    plan_prefetch_subtree_dp,
    solve_budget_allocation,
)


# Sample MH state: the case-study configuration of Table
# `hyperparameter-configuration` shrunk so the whole tree stays printable.
# ``seq_len`` is the current response length, cuts index the response, so the
# prompt offset is zero.
SEQ_LEN = 64
PROMPT_LEN = 0
MCMC_STEPS = 12
PREFETCH_BUDGET = 8
SEED = 10086


def brute_force_optimum(candidates, budget):
    """Return the best ancestor-closed selection of at most `budget` decisions.

    Enumerates every ancestor-closed subset once and scores it with the sum
    form of Lemma `prefetch-expectation`, ``sum_{u in I} R_hat(u; eps)``. Only
    tractable for the small budgets this demo uses; it exists to check the
    Bellman recurrence, not to replace it.
    """
    best_value = 0.0
    best_selection: tuple[int, ...] = ()

    def expand(selected, frontier):
        nonlocal best_value, best_selection
        value = sum(candidates[slot].reach for slot in selected)
        if value > best_value:
            best_value, best_selection = value, tuple(selected)
        if len(selected) == budget:
            return
        # Dropping frontier[:i] when taking frontier[i] fixes one order of
        # additions per subset, so each ancestor-closed set is visited once.
        for i, slot in enumerate(frontier):
            children = [
                child
                for child in (candidates[slot].left, candidates[slot].right)
                if child >= 0
            ]
            expand(selected + [slot], list(frontier[i + 1:]) + children)

    if candidates:
        expand([], [0])
    return best_value, best_selection


def greedy_optimum(candidates, budget):
    """Best-first greedy on predicted reaching weight.

    Reaching weights cannot increase along a path, so repeatedly taking the
    heaviest candidate whose parent is already taken is optimal for the same
    objective. A second, independent reference for the DP.
    """
    if not candidates:
        return 0.0, ()
    frontier = [0]
    selected: list[int] = []
    while frontier and len(selected) < budget:
        frontier.sort(key=lambda slot: -candidates[slot].reach)
        slot = frontier.pop(0)
        selected.append(slot)
        frontier.extend(
            child
            for child in (candidates[slot].left, candidates[slot].right)
            if child >= 0
        )
    return sum(candidates[slot].reach for slot in selected), tuple(selected)


def print_requests(plan, header):
    """Print one request per line: which decision pays for it, and at what cut."""
    logger.info("%s", header)
    logger.info(
        "  %-5s %-10s %-12s %-9s %s",
        "batch", "decision", "accept child", "cut index", "reaching weight",
    )
    inverse = {child: parent for parent, child in plan.parent_to_child.items()}
    for child, slot in sorted(plan.node_to_batch_idx.items(), key=lambda kv: kv[1]):
        logger.info(
            "  %-5d %-10d %-12d %-9d %.6f",
            slot,
            inverse[child],
            child,
            plan.batch_cut_indices[slot],
            plan.reaching_weights[slot],
        )


def demo_dp_batch():
    """Select a batch by DP and verify the optimum two independent ways."""
    logger.info("=" * 78)
    logger.info(
        "1) DP request batch: seq_len=%d, mcmc_steps=%d, N_pf=%d, "
        "predictor='shared_mean' (h_l=0.3084)",
        SEQ_LEN, MCMC_STEPS, PREFETCH_BUDGET,
    )
    plan = plan_prefetch_subtree_dp(
        max_batch_size=PREFETCH_BUDGET,
        prompt_len=PROMPT_LEN,
        seq_len=SEQ_LEN,
        mcmc_steps=MCMC_STEPS,
        branch_predictor="shared_mean",
        rng=np.random.default_rng(SEED),
    )
    print_requests(plan, "selected requests (batch order is BFS order):")

    # The recurrence value and the sum form of Lemma `prefetch-expectation`
    # are two ways of writing the same number.
    weight_sum = float(sum(plan.reaching_weights))
    logger.info(
        "  V_hat(v_0, N_pf) = %.6f, sum of reaching weights = %.6f, "
        "requests issued = %d/%d",
        plan.predicted_transitions,
        weight_sum,
        len(plan.batch_cut_indices),
        PREFETCH_BUDGET,
    )
    assert abs(plan.predicted_transitions - weight_sum) < 1e-9

    # Same candidate tree, same seed: rebuild it and optimize it two other ways.
    candidates, _ = build_candidate_decisions(
        max_batch_size=PREFETCH_BUDGET,
        prompt_len=PROMPT_LEN,
        seq_len=SEQ_LEN,
        mcmc_steps=MCMC_STEPS,
        branch_predictor="shared_mean",
        rng=np.random.default_rng(SEED),
    )
    dp_value = solve_budget_allocation(candidates, PREFETCH_BUDGET)[1]
    brute_value, _ = brute_force_optimum(candidates, PREFETCH_BUDGET)
    greedy_value, _ = greedy_optimum(candidates, PREFETCH_BUDGET)
    logger.info(
        "  candidates in I_max: %d; DP %.6f, brute force %.6f, greedy %.6f",
        len(candidates), dp_value, brute_value, greedy_value,
    )
    assert abs(dp_value - brute_value) < 1e-9, (dp_value, brute_value)
    assert abs(dp_value - greedy_value) < 1e-9, (dp_value, greedy_value)
    logger.info("  DP matches brute force and greedy.")


def demo_against_best_first():
    """Score the best-first collector's batch under the same branch model."""
    logger.info("=" * 78)
    logger.info("2) DP vs the best-first collector at the same budget")

    parent_to_child, _, batch_cuts = collect_batch_cut_indicesv3(
        max_batch_size=PREFETCH_BUDGET,
        prompt_len=PROMPT_LEN,
        seq_len=SEQ_LEN,
        mcmc_steps=MCMC_STEPS,
        rank="bfs_accept_first",
        rng=np.random.default_rng(SEED),
    )

    # Score the BFS selection under the same shared-mean branch model: walk
    # each selected decision's path and multiply 0.3084 per acceptance edge,
    # 0.6916 per rejection edge. Decisions whose ancestors were not selected
    # are unreachable and contribute nothing, which is what makes them waste.
    h_left = 0.3084
    value = 0.0
    wasted = 0
    for node_idx in sorted(parent_to_child):
        path_weight = 1.0
        cursor = node_idx
        while cursor > 0:
            parent = (cursor - 1) // 2
            path_weight *= h_left if cursor == 2 * parent + 1 else 1.0 - h_left
            cursor = parent
        if all(
            ancestor in parent_to_child
            for ancestor in _ancestors(node_idx)
        ):
            value += path_weight
        else:
            wasted += 1

    logger.info(
        "  best-first: %d requests, %d of them below an unexpanded decision "
        "(their proposals cannot be attached by build_subtree); "
        "predicted transitions %.6f",
        len(batch_cuts), wasted, value,
    )

    plan = plan_prefetch_subtree_dp(
        max_batch_size=PREFETCH_BUDGET,
        prompt_len=PROMPT_LEN,
        seq_len=SEQ_LEN,
        mcmc_steps=MCMC_STEPS,
        branch_predictor="shared_mean",
        rng=np.random.default_rng(SEED),
    )
    logger.info(
        "  DP:         %d requests, 0 unreachable; predicted transitions %.6f",
        len(plan.batch_cut_indices), plan.predicted_transitions,
    )


def _ancestors(node_idx):
    """Yield every strict ancestor of `node_idx`, root last."""
    cursor = node_idx
    while cursor > 0:
        cursor = (cursor - 1) // 2
        yield cursor


def demo_budget_sweep():
    """Sweep N_pf from 2 to 20, the range of the paper's ablation."""
    logger.info("=" * 78)
    logger.info("3) predicted transitions vs prefetch budget N_pf")
    logger.info(
        "  %-6s %-12s %-10s %s", "N_pf", "candidates", "requests", "V_hat(v_0, N_pf)",
    )
    for budget in range(2, 21, 2):
        plan = plan_prefetch_subtree_dp(
            max_batch_size=budget,
            prompt_len=PROMPT_LEN,
            seq_len=SEQ_LEN,
            mcmc_steps=MCMC_STEPS,
            branch_predictor="shared_mean",
            rng=np.random.default_rng(SEED),
        )
        logger.info(
            "  %-6d %-12d %-10d %.6f",
            budget,
            plan.num_candidates,
            len(plan.batch_cut_indices),
            plan.predicted_transitions,
        )


def demo_hard_predictor():
    """A certainty-class model that flags long-suffix proposals as certain rejects.

    Resampling a longer suffix diverges further from the current state, so the
    acceptance probability tends to collapse. This stands in for the network of
    Appendix C: it returns the two class probabilities and
    `class_probability_predictor` turns them into h_hat(v_l; eps) through the
    midpoint rule of Lemma `midpoint-branch-probability`.
    """
    logger.info("=" * 78)
    logger.info("4) a hard certainty predictor: long suffixes are certain rejects")

    def predict_classes(node_idx, cut, accept_cut, depth, num_accepts):
        resampled = SEQ_LEN - accept_cut
        if resampled > SEQ_LEN // 2:
            return 1.0, 0.0  # certain reject
        if resampled < SEQ_LEN // 8:
            return 0.0, 1.0  # certain accept
        return 0.4, 0.4  # the pooled uncertain case

    plan = plan_prefetch_subtree_dp(
        max_batch_size=PREFETCH_BUDGET,
        prompt_len=PROMPT_LEN,
        seq_len=SEQ_LEN,
        mcmc_steps=MCMC_STEPS,
        branch_predictor=class_probability_predictor(predict_classes),
        rng=np.random.default_rng(SEED),
    )
    print_requests(plan, "selected requests under the hard predictor:")
    logger.info(
        "  V_hat(v_0, N_pf) = %.6f from %d requests; a branch predicted "
        "unreachable (R_hat = 0) never gets budget.",
        plan.predicted_transitions, len(plan.batch_cut_indices),
    )
    assert all(weight > 0.0 for weight in plan.reaching_weights)


def demo_build_subtree():
    """Feed the DP's batch into build_subtree, as a sampler would."""
    logger.info("=" * 78)
    logger.info("5) the selected requests assemble into a proposal subtree")

    rng = np.random.default_rng(SEED)
    parent_to_child, node_to_batch_idx, batch_cuts = collect_batch_cut_indices_dp(
        max_batch_size=PREFETCH_BUDGET,
        prompt_len=PROMPT_LEN,
        seq_len=SEQ_LEN,
        mcmc_steps=MCMC_STEPS,
        branch_predictor="shared_mean",
        rng=rng,
    )

    # Stand-ins for the batched proposal call: token ids and log-ratios the
    # serving engine would return for these requests.
    root_proposal = list(range(SEQ_LEN))
    proposals = [
        root_proposal[:cut] + [1000 + slot] * (SEQ_LEN - cut)
        for slot, cut in enumerate(batch_cuts)
    ]
    log_accept_ratios = rng.normal(loc=-1.0, scale=2.0, size=len(batch_cuts))

    root = build_subtree(
        subtree_root_proposal=root_proposal,
        sampled_proposals=proposals,
        log_acceptance_ratios=log_accept_ratios,
        node_to_batch_idx=node_to_batch_idx,
        parent_to_accept_child=parent_to_child,
        batch_cut_indices=batch_cuts,
    )
    logger.info("prefetched subtree:\n%s", root.tree_str())

    # Every request ends up on the tree: the DP's selection is ancestor-closed,
    # so no proposal is stranded below an unexpanded decision.
    attached = set()
    stack = [root]
    while stack:
        node = stack.pop()
        attached.add(node.node_id)
        stack.extend(child for child in (node.left, node.right) if child is not None)
    assert set(node_to_batch_idx).issubset(attached)
    logger.info(
        "  all %d requested proposals are attached to the subtree.",
        len(node_to_batch_idx),
    )


if __name__ == "__main__":
    demo_dp_batch()
    demo_against_best_first()
    demo_budget_sweep()
    demo_hard_predictor()
    demo_build_subtree()
