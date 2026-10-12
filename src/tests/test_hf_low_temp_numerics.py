"""CPU regression tests for fixed-length HF proposal generation.

Run with:
    uv run --project /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src \
      pytest /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/tests/test_hf_low_temp_numerics.py
"""

from types import SimpleNamespace

import pytest
import torch

from power_sharpening.backends.hf.samplers.low_temp_sampler import (
    batched_low_temp_proposal_sampling,
    low_temp_sampling,
)
from power_sharpening.backends.hf.samplers.proposal_model_call import (
    _finite_score_sum,
)
from power_sharpening.backends.hf.wrapper import HF_LLM_Wrapper


class _Tokenizer:
    """Minimal tokenizer fields consumed by the HF wrapper."""

    eos_token_id = 2
    pad_token_id = 2


class _FixedLengthModel:
    """Return an EOS followed by real tokens when EOS stopping is disabled."""

    def __init__(self):
        self.last_kwargs = None

    def generate(self, input_ids, **kwargs):
        self.last_kwargs = kwargs
        assert kwargs["eos_token_id"] is None
        assert kwargs["top_k"] == 0
        assert kwargs["top_p"] == 1.0

        suffix_ids = [2, 3, 4][: kwargs["max_new_tokens"]]
        suffix = torch.tensor(
            [suffix_ids] * input_ids.shape[0],
            dtype=torch.long,
        )
        sequences = torch.cat((input_ids.cpu(), suffix), dim=1)
        raw_steps = []
        scaled_steps = []
        for token_id in suffix_ids:
            raw = torch.tensor(
                [[0.0, 1.0, 2.0, 3.0, 4.0]] * input_ids.shape[0]
            )
            scaled = raw / kwargs["temperature"]
            raw_steps.append(raw)
            scaled_steps.append(scaled)
            assert bool(torch.isfinite(scaled[:, token_id]).all())
        return SimpleNamespace(
            sequences=sequences,
            logits=tuple(raw_steps),
            scores=tuple(scaled_steps),
        )


def test_fixed_length_proposal_continues_through_eos_with_finite_scores():
    model = _FixedLengthModel()
    wrapper = HF_LLM_Wrapper(
        model,
        _Tokenizer(),
        device=torch.device("cpu"),
        temperature=0.25,
        alpha=4.0,
    )

    proposal, proposal_logprobs, target_scores = low_temp_sampling(
        wrapper,
        context=[1],
        seq_len=4,
        ignore_eos=True,
    )

    assert proposal == [1, 2, 3, 4]
    assert len(proposal_logprobs) == len(target_scores) == 3
    assert bool(torch.isfinite(torch.tensor(proposal_logprobs)).all())
    assert bool(torch.isfinite(torch.tensor(target_scores)).all())


def test_batched_fixed_length_proposals_do_not_score_post_eos_padding():
    model = _FixedLengthModel()
    wrapper = HF_LLM_Wrapper(
        model,
        _Tokenizer(),
        device=torch.device("cpu"),
        temperature=0.25,
        alpha=4.0,
    )

    proposals, proposal_logprobs, target_scores = (
        batched_low_temp_proposal_sampling(
            wrapper,
            contexts=[[1], [1, 3]],
            seq_len=4,
            ignore_eos=True,
        )
    )

    assert proposals == [[1, 2, 3, 4], [1, 3, 2, 3]]
    assert [len(row) for row in proposal_logprobs] == [3, 2]
    assert [len(row) for row in target_scores] == [3, 2]
    assert all(
        bool(torch.isfinite(torch.tensor(row)).all())
        for row in proposal_logprobs + target_scores
    )


@pytest.mark.parametrize("invalid_value", [float("-inf"), float("inf"), float("nan")])
def test_mh_score_sum_rejects_nonfinite_entries_before_ratio(invalid_value):
    with pytest.raises(ValueError, match="non-finite current proposal"):
        _finite_score_sum(
            [0.0, invalid_value],
            score_name="current proposal log-probability",
            proposal_index=0,
            cut_idx=7,
        )
