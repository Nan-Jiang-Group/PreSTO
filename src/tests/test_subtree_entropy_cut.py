"""EntropyCut checks for the subtree-prefetching cut law and its MH correction.

Run with:
    uv run --project /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src \
      pytest /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/tests/test_subtree_entropy_cut.py
"""

import collections
import math

import numpy as np
import pytest

from power_sharpening.common.cut_distribution import (
    compute_entropy_cut_policy,
    draw_entropy_cut_index,
    entropy_cut_log_ratio,
    summarize_cut_policy,
)
LOW, HIGH = 100, 110
NUM_CUTS = HIGH - LOW + 1


def _entropies_with_jumps():
    """Flat entropies with a +2 jump at offset 3 and a +4 jump at offset 7."""
    entropies = [1.0] * NUM_CUTS
    entropies[3] = 3.0
    entropies[7] = 5.0
    return entropies


def test_entropy_cut_concentrates_on_positive_delta():
    """Only the two jump positions carry mass, in proportion to jump**beta."""
    rng = np.random.default_rng(0)
    entropies = _entropies_with_jumps()
    draws = collections.Counter(
        draw_entropy_cut_index(
            rng,
            entropies,
            low=LOW,
            beta=4.0,
        )
        for _ in range(20_000)
    )

    assert set(draws) == {LOW + 2, LOW + 6}
    # weights 2**4 = 16 and 4**4 = 256, so the larger jump takes 256/272.
    assert draws[LOW + 6] / 20_000 == pytest.approx(256 / 272, abs=0.01)


def test_entropy_cut_with_zero_power_is_uniform():
    """beta=0 recovers the uniform cut law exactly."""
    rng = np.random.default_rng(1)
    draws = collections.Counter(
        draw_entropy_cut_index(
            rng,
            _entropies_with_jumps(),
            low=LOW,
            beta=0.0,
        )
        for _ in range(40_000)
    )

    assert set(draws) == set(range(LOW, HIGH))
    for count in draws.values():
        assert count / 40_000 == pytest.approx(1 / (NUM_CUTS - 1), abs=0.01)


def test_default_entropy_cut_power_is_four():
    entropies = _entropies_with_jumps()
    default_draw = draw_entropy_cut_index(
        np.random.default_rng(3),
        entropies,
        low=LOW,
    )
    beta_four_draw = draw_entropy_cut_index(
        np.random.default_rng(3),
        entropies,
        low=LOW,
        beta=4,
    )

    assert default_draw == beta_four_draw


def test_cut_summary_reports_entropy_jumps_instead_of_probability_vector():
    entropies = np.array([0.0, 0.5, 0.25, 1.25])
    summary = summarize_cut_policy(
        entropies,
        compute_entropy_cut_policy(entropies, beta=4.0),
        beta=4.0,
        low=10,
    )

    assert summary == (
        "The sampler considered 3 possible cuts. Entropy ranged from 0.000 "
        "to 1.250, with an average of 0.500. Entropy increased at 2 "
        "positions. The largest increases occurred at cut 12 (+1.000), "
        "cut 10 (+0.500)."
    )
    assert "beta" not in summary


def _subtree_log_acceptance(
    current_entropies,
    proposed_suffix_entropies,
    offset,
    target_to,
    target_from,
    proposal_from,
    proposal_to,
    cut_power,
    include_cut_term,
):
    """Reproduce the acceptance expression built in batched_proposal_callv2."""
    current_cut_probabilities = compute_entropy_cut_policy(
        current_entropies,
        beta=cut_power,
    )
    proposed_entropies = list(current_entropies[:offset]) + list(
        proposed_suffix_entropies
    )
    proposed_cut_probabilities = compute_entropy_cut_policy(
        proposed_entropies,
        beta=cut_power,
    )
    cut_term = (
        float(
            entropy_cut_log_ratio(
                current_cut_probabilities,
                proposed_cut_probabilities,
                cut_offset=offset,
            )
        )
        if include_cut_term
        else 0.0
    )
    log_acceptance = (
        math.log(target_to)
        - math.log(target_from)
        + math.log(proposal_from)
        - math.log(proposal_to)
        + cut_term
    )
    forward_cut = float(current_cut_probabilities[offset])
    return forward_cut * proposal_to * min(1.0, math.exp(log_acceptance))


@pytest.mark.parametrize("cut_power", [1.0, 2.0, 4.0])
def test_subtree_cut_correction_satisfies_detailed_balance(cut_power):
    """pi(x) P(x->y) == pi(y) P(y->x) once the cut ratio is included.

    x and y share the prefix before the cut and differ on the suffix, so their EntropyCut laws differ and the cut
    density does not cancel on its own.
    """
    offset = 2
    target_x, target_y = 0.3, 0.7
    proposal_y, proposal_x = 0.4, 0.6
    entropies_x = [1.0, 2.0, 1.0, 4.0, 1.0]
    suffix_y = [1.0, 3.0, 2.0]
    entropies_y = entropies_x[:offset] + suffix_y
    suffix_x = entropies_x[offset:]

    flow_xy = target_x * _subtree_log_acceptance(
        entropies_x, suffix_y, offset,
        target_y, target_x, proposal_x, proposal_y,
        cut_power, include_cut_term=True,
    )
    flow_yx = target_y * _subtree_log_acceptance(
        entropies_y, suffix_x, offset,
        target_x, target_y, proposal_y, proposal_x,
        cut_power, include_cut_term=True,
    )

    assert flow_xy == pytest.approx(flow_yx)


def test_dropping_the_subtree_cut_correction_breaks_detailed_balance():
    """Guard the regression the correction exists to prevent."""
    offset = 2
    target_x, target_y = 0.3, 0.7
    proposal_y, proposal_x = 0.4, 0.6
    cut_power = 2.0

    entropies_x = [1.0, 2.0, 1.0, 4.0, 1.0]
    suffix_y = [1.0, 3.0, 2.0]
    entropies_y = entropies_x[:offset] + suffix_y
    suffix_x = entropies_x[offset:]

    flow_xy = target_x * _subtree_log_acceptance(
        entropies_x, suffix_y, offset,
        target_y, target_x, proposal_x, proposal_y,
        cut_power, include_cut_term=False,
    )
    flow_yx = target_y * _subtree_log_acceptance(
        entropies_y, suffix_x, offset,
        target_x, target_y, proposal_y, proposal_x,
        cut_power, include_cut_term=False,
    )

    assert flow_xy != pytest.approx(flow_yx)
