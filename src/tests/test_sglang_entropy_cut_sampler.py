"""CPU tests for the SGLang EntropyCut MH transition.

Run with:
    uv run --project /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src \
      pytest /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/tests/test_sglang_entropy_cut_sampler.py
"""

import importlib.util
import math
import sys
import types

import numpy as np
import pytest

# Importing the SGLang sampler package also imports its wrapper. A minimal stub keeps these model-free tests runnable
# when the optional SGLang extra is not installed; no SGLang API is exercised here.
if (
    "sglang" not in sys.modules
    and importlib.util.find_spec("sglang") is None
):
    sys.modules["sglang"] = types.ModuleType("sglang")

from power_sharpening.backends.sglang.samplers import entropy_cut_mh


class _Wrapper:
    alpha = 4.0
    temperature = 0.25


class _FixedRng:
    def __init__(self, cut_index, uniform_draw):
        self.cut_index = cut_index
        self.uniform_draw = uniform_draw

    def choice(self, size, p):
        assert 0 <= self.cut_index < size
        return self.cut_index

    def random(self):
        return self.uniform_draw


def _proposal(sequence, q_scores, target_scores, entropies):
    return (
        sequence,
        q_scores,
        target_scores,
        {
            "suffix_entropies": entropies,
            "entropy_mode": "topk_tail_bucket",
            "entropy_top_k": 64,
        },
    )


def _queued_proposer(monkeypatch, responses):
    calls = []
    queue = list(responses)

    def fake_proposer(
        sampler_wrapper,
        context,
        seq_len,
        entropy_top_k,
        ignore_eos,
        verbose,
    ):
        calls.append(
            {
                "context": list(context),
                "seq_len": seq_len,
                "entropy_top_k": entropy_top_k,
                "ignore_eos": ignore_eos,
            }
        )
        assert sampler_wrapper is not None
        assert queue, "unexpected extra proposal call"
        return queue.pop(0)

    monkeypatch.setattr(
        entropy_cut_mh,
        "low_temp_proposal_sampling",
        fake_proposer,
    )
    return calls, queue


@pytest.mark.parametrize(
    (
        "uniform_draw",
        "expected_tokens",
        "expected_acceptances",
        "expected_mean_entropy",
    ),
    [
        (0.01, [10, 1, 4, 5], 1, 7.0 / 3.0),
        (0.50, [10, 1, 2, 3], 0, 5.0 / 3.0),
    ],
)
def test_cut_ratio_controls_acceptance_and_state_replacement(
    monkeypatch,
    uniform_draw,
    expected_tokens,
    expected_acceptances,
    expected_mean_entropy,
):
    calls, queue = _queued_proposer(
        monkeypatch,
        [
            _proposal(
                [10, 1, 2, 3],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 2.0, 2.0],
            ),
            _proposal(
                [10, 1, 4, 5],
                [0.0, 0.0],
                [0.0, 0.0],
                [2.0, 4.0],
            ),
        ],
    )

    tokens, stats = entropy_cut_mh.entropy_cut_mh_sampler(
        _Wrapper(),
        context=[10],
        mcmc_steps=1,
        max_new_tokens=3,
        num_of_blocks=1,
        cut_power=4.0,
        entropy_mode="topk",
        entropy_top_k=64,
        rng=_FixedRng(cut_index=0, uniform_draw=uniform_draw),
    )

    assert tokens == expected_tokens
    assert stats["attempts"] == 1
    assert stats["acceptances"] == expected_acceptances
    assert stats["mean_token_entropy"] == pytest.approx(
        expected_mean_entropy
    )
    transition = stats["transitions"][0]
    assert transition["target_term"] == 0.0
    assert transition["proposal_term"] == 0.0
    assert transition["cut_term"] == pytest.approx(-math.log(17.0))
    assert transition["log_acceptance"] == pytest.approx(-math.log(17.0))
    assert transition["accepted"] is bool(expected_acceptances)
    assert calls[0]["context"] == [10]
    assert calls[1]["context"] == [10, 1]
    assert all(call["ignore_eos"] for call in calls)
    assert not queue


def test_initial_generation_without_mh_returns_aligned_fixed_state(monkeypatch):
    calls, queue = _queued_proposer(
        monkeypatch,
        [
            _proposal(
                [10, 1, 2],
                [-0.2, -0.3],
                [-0.4, -0.5],
                [0.7, 0.9],
            ),
        ],
    )

    tokens, stats = entropy_cut_mh.entropy_cut_mh_sampler(
        _Wrapper(),
        context=[10],
        mcmc_steps=0,
        max_new_tokens=2,
        num_of_blocks=1,
        entropy_mode="topk",
        entropy_top_k=64,
    )

    assert tokens == [10, 1, 2]
    assert stats["attempts"] == 0
    assert stats["acceptance_rate"] == 0.0
    assert stats["transitions"] == []
    assert stats["mean_token_entropy"] == pytest.approx(0.8)
    assert len(calls) == 1
    assert not queue


def test_all_zero_jumps_use_uniform_fallback():
    probabilities = entropy_cut_mh.compute_entropy_cut_policy(
        [1.0, 1.0, 1.0],
        beta=4.0,
    )

    np.testing.assert_array_equal(probabilities, [0.5, 0.5])


def test_fixed_length_violation_fails_before_mh(monkeypatch):
    _queued_proposer(
        monkeypatch,
        [
            _proposal(
                [10, 1],
                [-0.2],
                [-0.4],
                [0.7],
            ),
        ],
    )

    with pytest.raises(RuntimeError, match="fixed-length state space"):
        entropy_cut_mh.entropy_cut_mh_sampler(
            _Wrapper(),
            context=[10],
            mcmc_steps=0,
            max_new_tokens=2,
            num_of_blocks=1,
            entropy_mode="topk",
            entropy_top_k=64,
        )


def test_exact_mode_reports_missing_server_entropy(monkeypatch):
    _queued_proposer(
        monkeypatch,
        [
            (
                [10, 1, 2],
                [-0.2, -0.3],
                [-0.4, -0.5],
                {
                    "suffix_entropies": None,
                    "entropy_mode": None,
                    "entropy_top_k": None,
                },
            ),
        ],
    )

    with pytest.raises(RuntimeError, match="Exact full-vocabulary entropy"):
        entropy_cut_mh.entropy_cut_mh_sampler(
            _Wrapper(),
            context=[10],
            mcmc_steps=0,
            max_new_tokens=2,
            num_of_blocks=1,
            entropy_mode="exact",
            entropy_top_k=None,
        )
