"""CPU tests for the shared PowerMH/subtree cut distributions.

Run with:
    uv run --project /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src \
      pytest /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/tests/test_cut_distribution.py
"""

from pathlib import Path

import numpy as np
import pytest

from power_sharpening.backends.hf.samplers.power_samp_MH import (
    mcmc_power_sampler,
)
from power_sharpening.common.cut_distribution import (
    draw_uniform_cut_index,
)
from power_sharpening.common.prefetch_subtree import collect_batch_cut_indicesv3
from power_sharpening.config import load_run_config


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_power_mh_config_selects_the_shared_uniform_law_by_default():
    config = load_run_config(
        REPO_ROOT / "src" / "power_sharpening" / "config",
        "math500",
        "power_mcmc",
    )

    assert config["cut_dist_type"] == "uniform"
    assert "cut_dist_param" not in config


def test_numpy_generator_draws_uniform_cuts():
    rng = np.random.default_rng(17)
    draws = [
        draw_uniform_cut_index(rng, 3, 11)
        for _ in range(100)
    ]

    assert all(3 <= cut <= 11 for cut in draws)


def test_shared_uniform_draw_preserves_numpy_generator_sequence():
    direct_rng = np.random.default_rng(314159)
    shared_rng = np.random.default_rng(314159)

    expected = [
        int(direct_rng.integers(4, 19, endpoint=True))
        for _ in range(20)
    ]
    actual = [draw_uniform_cut_index(shared_rng, 4, 19) for _ in range(20)]

    assert actual == expected


def test_uniform_draw_handles_a_single_cut():
    assert (
        draw_uniform_cut_index(
            np.random.default_rng(0),
            7,
            7,
        )
        == 7
    )


def test_seeded_subtree_draws_are_unchanged_after_extraction():
    _, _, batch_cuts = collect_batch_cut_indicesv3(
        8,
        3,
        12,
        5,
        rng=np.random.default_rng(2468),
        cut_dist_type="uniform",
    )

    assert batch_cuts == [5, 7, 6, 10, 6, 3, 9, 7]


@pytest.mark.parametrize(
    "removed_cut_dist_type",
    [
        "local_geometric",
        "global_geometric",
        "local_zipf",
        "beta",
        "truncated_normal",
    ],
)
def test_removed_cut_distributions_are_rejected(removed_cut_dist_type):
    with pytest.raises(ValueError, match="unknown cut_dist_type"):
        collect_batch_cut_indicesv3(
            1,
            0,
            5,
            1,
            rng=np.random.default_rng(0),
            cut_dist_type=removed_cut_dist_type,
        )


def test_invalid_name_and_range_fail_loudly():
    with pytest.raises(ValueError, match="unknown cut_dist_type"):
        collect_batch_cut_indicesv3(
            1,
            0,
            5,
            1,
            rng=np.random.default_rng(0),
            cut_dist_type="missing",
        )
    with pytest.raises(ValueError, match="low > high"):
        draw_uniform_cut_index(np.random.default_rng(0), 5, 4)


def test_power_mh_rejects_unknown_cut_law_before_model_use():
    with pytest.raises(ValueError, match="unknown cut_dist_type"):
        mcmc_power_sampler(
            sampler_wrapper=None,
            context=[],
            mcmc_steps=1,
            cut_dist_type="missing",
        )
