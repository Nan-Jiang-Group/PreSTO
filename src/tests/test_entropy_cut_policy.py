"""CPU tests for the shared Entropy-Cut policy.

Run with:
    uv run --project /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src pytest /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/tests/test_entropy_cut_policy.py
"""

import math

import numpy as np
import pytest

from power_sharpening.common.cut_distribution import (
    compute_entropy_cut_policy,
    draw_entropy_cut_index,
    entropy_cut_log_ratio,
)


def test_entropy_cut_policy_uses_positive_adjacent_jumps():
    probabilities = compute_entropy_cut_policy(
        [1.0, 3.0, 2.0, 2.5],
        beta=2.0,
    )

    np.testing.assert_allclose(
        probabilities,
        [4.0 / 4.25, 0.0, 0.25 / 4.25],
    )


def test_beta_zero_is_uniform():
    probabilities = compute_entropy_cut_policy(
        [1.0, 3.0, 2.0, 2.5],
        beta=0.0,
    )

    np.testing.assert_array_equal(probabilities, np.full(3, 1.0 / 3.0))


def test_all_zero_jumps_use_uniform_fallback():
    probabilities = compute_entropy_cut_policy(
        [2.0, 2.0, 2.0, 2.0],
        beta=4.0,
    )

    np.testing.assert_array_equal(probabilities, np.full(3, 1.0 / 3.0))


def test_draw_entropy_cut_index_respects_zero_mass_positions():
    rng = np.random.default_rng(7)
    draws = [
        draw_entropy_cut_index(rng, [0.0, 1.0, 1.0, 1.0], low=1)
        for _ in range(10)
    ]

    assert draws == [1] * 10


def test_draw_entropy_cut_index_is_seeded_by_generator():
    first_rng = np.random.default_rng(11)
    second_rng = np.random.default_rng(11)
    entropies = [0.0, 1.0, 3.0]

    first_draws = [
        draw_entropy_cut_index(first_rng, entropies)
        for _ in range(5)
    ]
    second_draws = [
        draw_entropy_cut_index(second_rng, entropies)
        for _ in range(5)
    ]

    assert first_draws == second_draws


def test_entropy_cut_log_ratio_uses_normalized_probabilities():
    current = np.array([0.25, 0.75])
    proposed = np.array([0.5, 0.5])

    assert entropy_cut_log_ratio(current, proposed, 0) == pytest.approx(
        math.log(2.0)
    )
    assert entropy_cut_log_ratio(current, proposed, 1) == pytest.approx(
        math.log(2.0 / 3.0)
    )
