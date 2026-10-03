"""Detailed-balance checks for the state-dependent EntropyCut correction.

Run with:
    uv run --project /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src \
      pytest /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/tests/test_entropy_cut_detailed_balance.py
"""

import math

import numpy as np
import pytest

from power_sharpening.common.cut_distribution import (
    compute_entropy_cut_policy,
    entropy_cut_log_ratio,
)


def _accepted_transition_mass(
    target_from,
    target_to,
    lambda_beta_from,
    lambda_beta_to,
    proposal_to,
    proposal_from,
):
    log_acceptance = (
        math.log(target_to)
        - math.log(target_from)
        + math.log(proposal_from)
        - math.log(proposal_to)
        + math.log(lambda_beta_to)
        - math.log(lambda_beta_from)
    )
    return lambda_beta_from * proposal_to * min(1.0, math.exp(log_acceptance))


def test_state_dependent_cut_ratio_satisfies_detailed_balance():
    target_x, target_y = 0.3, 0.7
    proposal_y, proposal_x = 0.4, 0.6
    lambda_beta_x = compute_entropy_cut_policy(
        [0.0, 1.0, 3.0],
        beta=2.0,
    )
    lambda_beta_y = compute_entropy_cut_policy(
        [0.0, 3.0, 4.0],
        beta=2.0,
    )
    cut_offset = 0

    flow_xy = target_x * _accepted_transition_mass(
        target_x,
        target_y,
        lambda_beta_x[cut_offset],
        lambda_beta_y[cut_offset],
        proposal_y,
        proposal_x,
    )
    flow_yx = target_y * _accepted_transition_mass(
        target_y,
        target_x,
        lambda_beta_y[cut_offset],
        lambda_beta_x[cut_offset],
        proposal_x,
        proposal_y,
    )

    assert flow_xy == pytest.approx(flow_yx)
    assert entropy_cut_log_ratio(
        lambda_beta_x,
        lambda_beta_y,
        cut_offset,
    ) == pytest.approx(
        math.log(lambda_beta_y[cut_offset] / lambda_beta_x[cut_offset])
    )


def test_omitting_cut_ratio_breaks_balance_for_same_example():
    target_x, target_y = 0.3, 0.7
    proposal_y, proposal_x = 0.4, 0.6
    lambda_x = np.array([0.2, 0.8])
    lambda_y = np.array([0.9, 0.1])
    cut_offset = 0

    log_acceptance_xy = (
        math.log(target_y / target_x)
        + math.log(proposal_x / proposal_y)
    )
    log_acceptance_yx = -log_acceptance_xy
    flow_xy = (
        target_x
        * lambda_x[cut_offset]
        * proposal_y
        * min(1.0, math.exp(log_acceptance_xy))
    )
    flow_yx = (
        target_y
        * lambda_y[cut_offset]
        * proposal_x
        * min(1.0, math.exp(log_acceptance_yx))
    )

    assert flow_xy != pytest.approx(flow_yx)


def test_beta_zero_recovers_uniform_cut_law():
    current = compute_entropy_cut_policy(
        [0.0, 0.0, 1.0, 101.0],
        beta=0.0,
    )
    proposed = compute_entropy_cut_policy(
        [0.0, 5.0, 5.0, 7.0],
        beta=0.0,
    )

    np.testing.assert_allclose(current, np.full(3, 1.0 / 3.0))
    assert entropy_cut_log_ratio(current, proposed, 2) == pytest.approx(0.0)
