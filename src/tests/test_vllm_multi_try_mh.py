"""CPU contract tests for vLLM MTM, including exact temperature cross-scoring.

Run with:
    python -m pytest /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/tests/test_vllm_multi_try_mh.py

vLLM/CUDA are stubbed; a small normalized autoregressive model supplies exact probabilities. These tests do not claim a
real vLLM runtime or GPU benchmark.
"""

import copy
import importlib.util
import math
from pathlib import Path
import sys
import types

import numpy as np
import pytest

from power_sharpening.common.temp_scheduler import ConstantScheduler, LinearScheduler

MODULE_PATH = Path(__file__).resolve().parents[1] / 'power_sharpening/backends/vllm/samplers/multi_try_mh.py'


@pytest.fixture
def sampler(monkeypatch):
    vllm = types.ModuleType('vllm')
    inputs = types.ModuleType('vllm.inputs')
    inputs.TokensPrompt = dict
    engine_name = 'power_sharpening.backends.vllm.engine_patch'
    engine = types.ModuleType(engine_name)
    engine.__path__ = [str(MODULE_PATH.parents[1] / 'engine_patch')]
    engine.SamplingParams = types.SimpleNamespace
    params = types.ModuleType(engine_name + '.sampling_params')

    def clone(source, **overrides):
        result = copy.deepcopy(source)
        for key, value in overrides.items():
            setattr(result, key, value)
        return result

    params._copy_sampling_params = clone
    for name, module in [('vllm', vllm), ('vllm.inputs', inputs),
                         (engine_name, engine), (params.__name__, params)]:
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location('_test_vllm_mtm', MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _distribution(prefix, temperature=1.0):
    # Context dependent, so rebuilding prefixes or scoring with a stale prefix changes the expected result. Token 4 is
    # deliberately high probability.
    logits = np.array([-1.8, -0.7, 0.1, -0.2, 0.9, 0.2, -0.1, -0.3])
    logits += ((np.arange(8) + sum(prefix)) % 3) * 0.2
    logits /= temperature
    return logits - np.logaddexp.reduce(logits)


def _scores(prefix, suffix, temperature):
    result = []
    for token in suffix:
        result.append(_distribution(prefix, temperature)[token])
        prefix = prefix + [token]
    return result


class _Tokenizer:
    eos_token_id = 0

    def __init__(self):
        self.encoded, self.decoded = [], []

    def encode(self, prompt):
        self.encoded.append(prompt)
        return {'A': [7, 0], 'B': [6]}[prompt]

    def decode(self, tokens, skip_special_tokens):
        self.decoded.append(list(tokens))
        return ' '.join(map(str, tokens))


class _Llm:
    def __init__(self, generation_batches):
        self.batches = list(generation_batches)
        self.generations, self.scoring = [], []

    def generate(self, prompts, sampling_params, use_tqdm):
        params = sampling_params
        assert len(prompts) == len(params)
        is_scoring = bool(getattr(params[0], 'logprob_token_ids', None))
        if is_scoring:
            self.scoring.append((copy.deepcopy(prompts), copy.deepcopy(params)))
            assert all(sp.logprobs == len(sp.logprob_token_ids) == 1 for sp in params)
            # Generated token 7 is arbitrary and must never enter the chain.
            suffixes = [[7] for _ in prompts]
        else:
            self.generations.append((copy.deepcopy(prompts), copy.deepcopy(params)))
            suffixes = self.batches.pop(0)
        assert len(suffixes) == len(prompts)
        outputs = []
        for prompt, sp, suffix in zip(prompts, params, suffixes):
            assert sp.n == 1
            assert sp.ignore_eos
            assert not sp.detokenize
            assert (sp.top_k, sp.top_p, sp.min_p) == (-1, 1.0, 0.0)
            assert len(suffix) == sp.max_tokens
            prefix = list(prompt['prompt_token_ids'])
            lps, base_lps = [], []
            for token in suffix:
                query = sp.logprob_token_ids[0] if is_scoring else token
                # Deliberately put a different token first; extracting the first dictionary value would use the wrong
                # score.
                ids = [(query + 1) % 8, token, query]
                q, p = _distribution(prefix, sp.temperature), _distribution(prefix)
                lps.append({idx: types.SimpleNamespace(logprob=q[idx]) for idx in ids})
                base_lps.append({idx: types.SimpleNamespace(logprob=p[idx]) for idx in ids})
                prefix.append(token)
            completion = types.SimpleNamespace(token_ids=suffix, logprobs=lps, power_logprobs=base_lps)
            outputs.append(types.SimpleNamespace(outputs=[completion]))
        return outputs


def _wrapper(batches):
    return types.SimpleNamespace(tokenizer=_Tokenizer(), llm=_Llm(batches), verbose=True)


def _params(**kwargs):
    return types.SimpleNamespace(alpha=3.0, temperature=0.5, seed=None, **kwargs)


def test_ratio_matches_hf_and_has_reverse_proposal_sign(sampler):
    from power_sharpening.backends.hf.samplers.multi_try_mh import _log_weight
    proposed_q, current_q = [-0.2, -0.4], [-0.7, -0.8]
    proposed_p, current_p = [-0.3, -0.5], [-0.6, -0.9]
    actual = sampler.compute_acceptance_ratio(proposed_q, current_q, proposed_p, current_p, 4)
    expected = 4 * (sum(proposed_p) - sum(current_p)) + sum(current_q) - sum(proposed_q)
    assert actual == pytest.approx(expected)
    assert actual == pytest.approx(_log_weight(np.array(proposed_p) * 4, np.array(current_p) * 4,
                                               proposed_q, current_q))


@pytest.mark.parametrize('weight', [0.01, 0.8, 1.0, 9.0])
def test_one_try_is_standard_mh(sampler, weight):
    assert math.exp(sampler._log_acceptance_probability([math.log(weight)], 0)) == pytest.approx(min(1, weight))


@pytest.mark.parametrize('draw, expected', [(0.0, '1 0'), (0.999999, '4 4 4')])
def test_k_candidates_uniform_cuts_acceptance_and_eos(sampler, monkeypatch, draw, expected):
    model = _wrapper([[[4, 4, 4]], [[1, 0, 5], [3, 2, 6]]])
    config = _params(top_k=10, stop=['stop'], repetition_penalty=2.0)
    before = copy.deepcopy(vars(config))
    cuts = []

    def cut(low, high):
        cuts.append((low, high))
        return low

    monkeypatch.setattr(sampler.random, 'randint', cut)
    monkeypatch.setattr(sampler, '_categorical_from_log_weights', lambda weights: 0)
    monkeypatch.setattr(np.random, 'rand', lambda: draw)
    actual = sampler.multi_try_mcmc_power_sampler(model, 'A', config, num_of_blocks=1,
                                                 max_new_tokens=3, mcmc_steps=1, num_tries=2,
                                                 proposal_temperatures=None)
    assert actual == [expected]
    assert cuts == [(0, 2)]  # Last generated token is an eligible uniform cut.
    assert [len(batch[0]) for batch in model.llm.generations] == [1, 2]
    assert not model.llm.scoring
    assert model.tokenizer.encoded == ['A']
    assert len(model.tokenizer.decoded) == 1  # Decode only the final response.
    assert vars(config) == before
    assert all(sp.alpha == 3 for _, params in model.llm.generations for sp in params)


def test_mixture_weights_and_later_cut_use_all_components(sampler, monkeypatch):
    model = _wrapper([[[1, 4]], [[4, 5], [1, 3]], [[3], [6]]])
    temperatures = [0.25, 0.6, 1.2]
    draws = iter([[0.6], [1.2, 0.25], [0.6, 1.2]])
    monkeypatch.setattr(np.random, 'choice', lambda choices, size: np.array(next(draws)))
    cuts = iter([0, 1])
    monkeypatch.setattr(sampler.random, 'randint', lambda low, high: next(cuts))
    monkeypatch.setattr(np.random, 'rand', lambda: 0.0)
    observed = []

    def select(weights):
        observed.append(weights)
        return 0

    monkeypatch.setattr(sampler, '_categorical_from_log_weights', select)
    actual = sampler.multi_try_mcmc_power_sampler(
        model, 'A', _params(), num_of_blocks=1, max_new_tokens=2,
        mcmc_steps=2, num_tries=2, proposal_temperatures=temperatures, scoring_batch_size=3,
    )
    assert actual == ['4 3']
    assert [len(batch[0]) for batch in model.llm.generations] == [1, 2, 2]
    assert all(len(prompts) <= 3 for prompts, _ in model.llm.scoring)
    # Only missing components are scored: 2*(2 + 2*2 + 2*1) token queries.
    assert sum(len(prompts) for prompts, _ in model.llm.scoring) == 16

    def weight(prefix, suffix):
        logp = sum(_scores(prefix, suffix, 1.0))
        logq = np.logaddexp.reduce([sum(_scores(prefix, suffix, t)) for t in temperatures]) - np.log(3)
        return 3 * logp - logq

    expected0 = [weight([7, 0], suffix) - weight([7, 0], [1, 4]) for suffix in [[4, 5], [1, 3]]]
    expected1 = [weight([7, 0, 4], suffix) - weight([7, 0, 4], [5]) for suffix in [[3], [6]]]
    np.testing.assert_allclose(observed, [expected0, expected1], atol=1e-12)


def test_prompt_batch_keeps_owners_and_removes_finished_prompts(sampler, monkeypatch):
    model = _wrapper([
        [[4, 4], [1, 2]],
        [[0, 5], [2, 3], [4, 6], [3, 1]],
        [[2, 4]],
        [[1, 2, 3, 4], [2, 3, 4, 5]],
    ])
    monkeypatch.setattr(sampler.random, 'randint', lambda low, high: low)
    monkeypatch.setattr(sampler, '_categorical_from_log_weights', lambda weights: 0)
    monkeypatch.setattr(np.random, 'rand', lambda: 0.0)
    actual = sampler.multi_try_mcmc_power_sampler(model, ['A', 'B'], _params(),
                                                 num_of_blocks=2, max_new_tokens=4,
                                                 mcmc_steps=1, num_tries=2)
    assert actual == ['0', '1 2 3 4']
    assert [len(batch[0]) for batch in model.llm.generations] == [2, 4, 1, 2]
    assert [p['prompt_token_ids'] for p in model.llm.generations[1][0]] == [[7, 0], [7, 0], [6], [6]]
    assert model.llm.generations[2][0][0]['prompt_token_ids'] == [6, 4, 6]


def test_scalar_schedule_rescores_current_scores_without_changing_alpha(sampler, monkeypatch):
    model = _wrapper([[[1, 4]], [[3, 5]]])
    schedule = LinearScheduler(0.9, end_temp=0.3, total_steps=2)
    monkeypatch.setattr(sampler.random, 'randint', lambda low, high: low)
    monkeypatch.setattr(np.random, 'rand', lambda: 0.0)
    weights = []
    monkeypatch.setattr(sampler, '_categorical_from_log_weights', lambda values: weights.append(values) or 0)
    actual = sampler.multi_try_mcmc_power_sampler(model, 'A', _params(), num_of_blocks=1,
                                                 max_new_tokens=2, mcmc_steps=1, num_tries=1,
                                                 temperature_scheduler=schedule,
                                                 proposal_temperatures=None)
    assert actual == ['3 5']
    assert len(model.llm.scoring) == 1
    assert len(model.llm.scoring[0][0]) == 2
    expected = sampler.compute_acceptance_ratio(_scores([7, 0], [3, 5], 0.3),
                                                _scores([7, 0], [1, 4], 0.3),
                                                _scores([7, 0], [3, 5], 1),
                                                _scores([7, 0], [1, 4], 1), 3)
    assert weights[0][0] == pytest.approx(expected)
    assert all(sp.alpha == 3 for _, params in model.llm.generations + model.llm.scoring for sp in params)


def test_duplicate_temperatures_reuse_scores_and_explicit_seeds_are_distinct(sampler, monkeypatch):
    model = _wrapper([[[1, 4]], [[3, 5], [4, 6]]])
    config = _params()
    config.seed = 42
    monkeypatch.setattr(sampler.random, 'randint', lambda low, high: low)
    sampler.multi_try_mcmc_power_sampler(model, 'A', config, num_of_blocks=1,
                                        max_new_tokens=2, mcmc_steps=1, num_tries=2,
                                        proposal_temperatures=[0.5, 0.5],
                                        temperature_scheduler=ConstantScheduler(0.25))
    assert not model.llm.scoring
    seeds = [sp.seed for _, params in model.llm.generations for sp in params]
    assert len(set(seeds)) == 3
    assert config.seed == 42
