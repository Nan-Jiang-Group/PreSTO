"""CPU contracts for vLLM subtree MultiTryMH and exact scoring-work counts.

Run with:
    python -m pytest /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/tests/test_vllm_subtree_multi_try_mh.py
"""

import copy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from test_vllm_multi_try_mh import (
    _Llm, _Tokenizer, _distribution, _params, _scores, _wrapper,
    sampler as baseline_sampler,
)


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "power_sharpening/backends/vllm/samplers/subtree_prefetching_multi_try_mh.py"
)


@pytest.fixture
def sampler(baseline_sampler, monkeypatch):
    monkeypatch.setitem(
        sys.modules, "power_sharpening.backends.vllm.samplers.multi_try_mh", baseline_sampler,
    )
    spec = importlib.util.spec_from_file_location("_test_vllm_subtree_multitry", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_supplied_temperature_seed_and_exact_work_counts(baseline_sampler):
    wrapper = _wrapper([[[1, 4], [3]]])
    params = SimpleNamespace(
        alpha=3.0, temperature=0.5, n=1, ignore_eos=True,
        detokenize=False, top_k=-1, top_p=1.0, min_p=0.0,
    )
    work = {}
    tokens, base, components = baseline_sampler._draw_proposals(
        wrapper, [[7, 0], [6]], [2, 1], [0.25, 0.5, 0.5, 1.0],
        params, 3, None, row_temperatures=[0.5, 0.25],
        proposal_seeds=[91, 27], work_stats=work,
    )
    assert tokens == [[1, 4], [3]]
    submitted = wrapper.llm.generations[0][1]
    assert [row.temperature for row in submitted] == [0.5, 0.25]
    assert [row.seed for row in submitted] == [91, 27]
    assert work == {
        "generation_calls": 1, "generated_tokens": 3,
        "scoring_calls": 3, "scoring_requests": 7,
    }
    for prefix, row_tokens, row_base, row_components in zip([[7, 0], [6]], tokens, base, components):
        np.testing.assert_allclose(row_base, _scores(prefix, row_tokens, 1.0))
        for temperature, scores in zip([0.25, 0.5, 0.5, 1.0], row_components):
            np.testing.assert_allclose(scores, _scores(prefix, row_tokens, temperature))


def test_adapter_forwards_kernel_requests_and_inherits_only_three_params(sampler, monkeypatch):
    wrapper = _wrapper([[[1, 4], [3]]])
    config = _params(top_k=10, stop=["stop"], repetition_penalty=2.0)
    config.seed = 17
    original = copy.deepcopy(vars(config))
    stats = object()

    def sample(prompt_ids, draw_batch, **kwargs):
        assert prompt_ids == [7, 0]
        assert kwargs["alpha"] == 3.0
        assert kwargs["proposal_temperatures"] == [0.25, 0.5]
        assert kwargs["num_tries"] == 2
        assert kwargs["prefetch_budget"] == 6
        assert kwargs["seed"] == 17
        assert kwargs["stop_on_eos"] is False
        proposals, work = draw_batch([
            ([7, 0], 2, 0.5, 91),
            ([6], 1, 0.25, 27),
        ], kwargs["proposal_temperatures"])
        assert work["generation_calls"] == 1
        assert work["scoring_calls"] == 2
        assert work["scoring_requests"] == 3
        assert work["generated_tokens"] == 3
        tokens, _, _ = proposals[0]
        assert tokens == [1, 4]
        empty, empty_work = draw_batch([], kwargs["proposal_temperatures"])
        assert empty == []
        assert empty_work == {"generation_calls": 0, "generated_tokens": 0}
        assert len(wrapper.llm.generations) == 1
        return proposals[0], stats

    monkeypatch.setattr(sampler, "sample_subtree_multi_try", sample)
    response, actual_stats = sampler.subtree_prefetching_multi_try_sampling(
        wrapper, [7, 0], config, num_tries=2, prefetch_budget=6,
        proposal_temperatures=[0.25, 0.5], scoring_batch_size=2,
        stop_on_eos=False,
    )
    assert response == [1, 4]
    assert actual_stats is stats
    assert vars(config) == original
    assert not wrapper.tokenizer.decoded
    assert all(not hasattr(row, "stop") for _, rows in wrapper.llm.generations for row in rows)


class _SeededLlm(_Llm):
    """Tiny normalized AR engine with independently seeded generation rows."""

    def __init__(self):
        super().__init__([])

    def generate(self, prompts, sampling_params, use_tqdm):
        if not getattr(sampling_params[0], "logprob_token_ids", None):
            suffixes = []
            for prompt, params in zip(prompts, sampling_params):
                prefix = list(prompt["prompt_token_ids"])
                rng = np.random.default_rng(params.seed)
                suffix = []
                for _ in range(params.max_tokens):
                    token = int(rng.choice(8, p=np.exp(_distribution(prefix, params.temperature))))
                    prefix.append(token)
                    suffix.append(token)
                suffixes.append(suffix)
            self.batches.append(suffixes)
        return super().generate(prompts, sampling_params, use_tqdm)


@pytest.mark.parametrize("budget", [2, 8])
def test_seeded_vllm_adapter_repeats_response_for_same_budget(sampler, budget):
    outputs, records = [], []
    for _ in range(2):
        wrapper = SimpleNamespace(tokenizer=_Tokenizer(), llm=_SeededLlm(), verbose=False)
        config = _params()
        config.seed = 173
        response, stats = sampler.subtree_prefetching_multi_try_sampling(
            wrapper, [7, 0], config, num_tries=2, prefetch_budget=budget,
            num_of_blocks=2, max_new_tokens=8, mcmc_steps=5,
            proposal_temperatures=[0.5, 1.0], scoring_batch_size=5,
            stop_on_eos=False,
        )
        outputs.append(response)
        records.append(stats)
        assert stats.generation_calls == len(wrapper.llm.generations)
        assert stats.scoring_calls == len(wrapper.llm.scoring)
        assert stats.total_nfe == len(wrapper.llm.generations) + len(wrapper.llm.scoring)
        assert stats.scoring_requests == sum(len(prompts) for prompts, _ in wrapper.llm.scoring)
        assert stats.total_walked_steps == 10
    assert outputs[0] == outputs[1]
    assert records[0].to_json() == records[1].to_json()
