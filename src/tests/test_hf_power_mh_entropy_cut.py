"""CPU tests for entropy-weighted cuts in sequential HF PowerMH.

Run with:
    uv run --project /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src \
      pytest /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/tests/test_hf_power_mh_entropy_cut.py
"""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from power_sharpening.backends.hf.samplers import power_samp_MH


class _Wrapper:
    temperature = 0.25
    tokenizer = SimpleNamespace(eos_token_id=-1)

    def init_schedule(self, schedule_type, mcmc_steps):
        assert schedule_type == "const"
        assert mcmc_steps == 1

    def temperature_step(self):
        return None


class _FixedRng:
    def __init__(self, uniform_draw):
        self.uniform_draw = uniform_draw
        self.cut_probabilities = []

    def choice(self, size, p):
        assert size == len(p)
        self.cut_probabilities.append(np.asarray(p))
        return 0

    def random(self):
        return self.uniform_draw


def _queued_proposer(monkeypatch):
    responses = [
        (
            [10, 1, 2, 3],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 2.0],
        ),
        (
            [10, 1, 4, 5],
            [0.0, 0.0],
            [0.0, 0.0],
            [2.0, 4.0],
        ),
    ]
    calls = []

    def fake_proposer(sampler_wrapper, context, seq_len, **kwargs):
        assert sampler_wrapper is not None
        calls.append((list(context), seq_len, kwargs))
        return responses.pop(0)

    monkeypatch.setattr(
        power_samp_MH,
        "low_temp_proposal_sampling",
        fake_proposer,
    )
    return calls, responses


@pytest.mark.parametrize(
    ("uniform_draw", "expected_tokens", "expected_acceptance"),
    [
        (0.01, [10, 1, 4, 5], 1.0),
        (0.50, [10, 1, 2, 3], 0.0),
    ],
)
def test_entropy_cut_ratio_controls_hf_power_mh_transition(
    monkeypatch,
    uniform_draw,
    expected_tokens,
    expected_acceptance,
):
    calls, responses = _queued_proposer(monkeypatch)
    rng = _FixedRng(uniform_draw)

    tokens, proposal_logprobs, target_scores, acceptance_ratio = (
        power_samp_MH.mcmc_power_sampler(
            _Wrapper(),
            context=[10],
            mcmc_steps=1,
            max_new_tokens=3,
            num_of_blocks=1,
            cut_dist_type="entropy",
            cut_power=4.0,
            rng=rng,
        )
    )

    assert tokens == expected_tokens
    assert proposal_logprobs == [0.0, 0.0, 0.0]
    assert target_scores == [0.0, 0.0, 0.0]
    assert acceptance_ratio == expected_acceptance
    assert len(rng.cut_probabilities) == 1
    np.testing.assert_array_equal(rng.cut_probabilities[0], [1.0, 0.0])
    assert calls[0][0] == [10]
    assert calls[1][0] == [10, 1]
    assert all(call[2]["return_entropies"] for call in calls)
    assert all(call[2]["ignore_eos"] for call in calls)
    assert not responses

    expected_cut_term = -math.log(17.0)
    assert (uniform_draw < math.exp(expected_cut_term)) is bool(
        expected_acceptance
    )


def test_entropy_cut_requires_constant_temperature_schedule():
    with pytest.raises(ValueError, match="fixed proposal distribution"):
        power_samp_MH.mcmc_power_sampler(
            _Wrapper(),
            context=[10],
            mcmc_steps=1,
            cut_dist_type="entropy",
            temperature_schedule_type="linear",
        )
