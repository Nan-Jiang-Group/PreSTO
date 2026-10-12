"""CPU adapter checks for subtree-prefetched HF Multi-Try MH.

Run with:
    /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/.venv/bin/python -m pytest \
      /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/tests/test_hf_subtree_multi_try_mh.py
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from power_sharpening.backends.hf.samplers import subtree_prefetching_multi_try_mh as sampler
from power_sharpening.backends.hf.wrapper import HF_LLM_Wrapper


class _ScriptedModel:
    """Exercise the real HF logits/scoring adapter without loading a model."""

    def __init__(self, context_limit=64):
        self.config = SimpleNamespace(max_position_embeddings=context_limit)
        self.calls = []
        self.logits = torch.tensor([-0.6, 0.0, 0.2, -0.7, 0.8, -0.1, 0.6])

    def generate(self, input_ids, **kwargs):
        self.calls.append((input_ids.clone(), kwargs))
        assert kwargs["eos_token_id"] is None
        count = kwargs["max_new_tokens"]
        assert input_ids.shape[1] + count <= self.config.max_position_embeddings
        # Identical prefix continuations are independent of left padding or batch position, allowing deterministic
        # coupling through the adapter.
        suffix = (input_ids[:, -1, None] + torch.arange(1, count + 1)) % len(self.logits)
        sequence = torch.cat([input_ids, suffix], dim=1)
        logits, scores = [], []
        for step in range(count):
            raw = self.logits.expand(len(input_ids), -1).clone()
            processed = kwargs["logits_processor"](
                sequence[:, :input_ids.shape[1] + step], raw.clone(),
            )
            logits.append(raw)
            scores.append(processed)
        return SimpleNamespace(sequences=sequence, logits=tuple(logits), scores=tuple(scores))


def _wrapper(context_limit=64, eos=None):
    model = _ScriptedModel(context_limit)
    return HF_LLM_Wrapper(
        model, SimpleNamespace(pad_token_id=0, eos_token_id=eos),
        torch.device("cpu"), temperature=0.5, alpha=3.0,
    )


def _assert_scores(wrapper, proposal, temperatures):
    ids, base, components = proposal
    expected_base = torch.log_softmax(wrapper.base_model.logits, dim=-1)[ids]
    np.testing.assert_allclose(base, expected_base, rtol=1e-6)
    expected_components = [
        torch.log_softmax(wrapper.base_model.logits / value, dim=-1)[ids].tolist()
        for value in temperatures
    ]
    np.testing.assert_allclose(components, expected_components, rtol=1e-6)


def test_variable_prefix_batch_preserves_temperatures_scores_and_actual_decode_work():
    wrapper = _wrapper()
    requests = [
        ([1], 4, 0.25, 11),
        ([1, 2, 3], 2, 1.0, 12),
    ]
    proposals, work = sampler._draw_batch(wrapper, requests, [0.25, 1.0])
    assert [tokens for tokens, _, _ in proposals] == [[2, 3, 4, 5], [4, 5]]
    assert work["generation_calls"] == 1
    assert work.get("scoring_calls", 0) == 0
    assert work["generated_tokens"] == 8  # Two rows decode four tokens, six retained.
    inputs, kwargs = wrapper.base_model.calls[0]
    assert inputs.tolist() == [[0, 0, 1], [1, 2, 3]]
    assert kwargs["attention_mask"].tolist() == [[0, 0, 1], [1, 1, 1]]
    np.testing.assert_allclose(kwargs["logits_processor"][0].temperatures.flatten(), [0.25, 1.0])
    for proposal in proposals:
        _assert_scores(wrapper, proposal, [0.25, 1.0])
    assert wrapper.alpha == 3.0


def test_context_limit_splits_by_prefix_length_and_restores_request_order():
    wrapper = _wrapper(context_limit=5)
    requests = [
        ([1, 2, 3], 2, 1.0, 13),
        ([1], 4, 0.25, 14),
        ([3, 2, 1], 2, 0.5, 15),
    ]
    proposals, work = sampler._draw_batch(wrapper, requests, [0.25, 0.5, 1.0])
    assert [tokens for tokens, _, _ in proposals] == [[4, 5], [2, 3, 4, 5], [2, 3]]
    assert work["generation_calls"] == 2
    assert work["generated_tokens"] == 8
    assert [(len(inputs), kwargs["max_new_tokens"]) for inputs, kwargs in wrapper.base_model.calls] == [(2, 2), (1, 4)]
    for proposal in proposals:
        _assert_scores(wrapper, proposal, [0.25, 0.5, 1.0])


def test_requested_context_cap_is_honored_even_when_model_window_is_larger():
    wrapper = _wrapper(context_limit=32)
    requests = [([1], 4, 0.5, 5), ([1, 2, 3], 2, 0.5, 6)]
    _, work = sampler._draw_batch(wrapper, requests, [0.5], context_length=5)
    assert work["generation_calls"] == 2
    assert work["generated_tokens"] == 6


def test_empty_batch_does_no_work():
    wrapper = _wrapper()
    proposals, work = sampler._draw_batch(wrapper, [], [0.5])
    assert proposals == []
    assert work["generation_calls"] == work["generated_tokens"] == 0
    assert not wrapper.base_model.calls


@pytest.mark.parametrize("budget", [2, 8])
def test_shared_engine_and_hf_adapter_repeat_seed_for_same_budget(budget):
    outputs = []
    for _ in range(2):
        wrapper = _wrapper()
        tokens, stats = sampler.subtree_prefetching_multi_try_sampling(
            wrapper, [1], mcmc_steps=5, max_new_tokens=8, num_of_blocks=2,
            num_tries=2, prefetch_budget=budget, seed=42,
            proposal_temperatures=[0.25, 1.0], stop_on_eos=False,
        )
        assert len(tokens) == 9
        assert tokens[0] == 1
        assert stats.total_walked_steps == 10
        assert stats.total_nfe == len(wrapper.base_model.calls)
        expected = torch.log_softmax(wrapper.base_model.logits, dim=-1)[tokens[1:]]
        np.testing.assert_allclose(stats.final_base_logprobs, expected, rtol=1e-6)
        assert stats.base_diagnostics is None
        outputs.append((tokens, stats.to_json()))
    assert outputs[0] == outputs[1]
