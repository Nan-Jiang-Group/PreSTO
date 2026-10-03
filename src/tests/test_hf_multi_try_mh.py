"""CPU tests of the paper MTM transition and temperature-mixture proposals.

Run with:
    uv run --project /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src \
      pytest /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/tests/test_hf_multi_try_mh.py
"""

import itertools
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from power_sharpening.backends.hf.samplers import multi_try_mh as mtm
from power_sharpening.backends.hf.samplers.low_temp_sampler import batched_low_temp_proposal_sampling
from power_sharpening.backends.hf.wrapper import HF_LLM_Wrapper
from power_sharpening.common.multi_try import log_sum_exp


@pytest.mark.parametrize("weight", [0.01, 0.7, 1.0, 9.0])
def test_one_try_reduces_to_standard_mh(weight):
    assert math.exp(mtm._log_acceptance_probability([math.log(weight)], 0)) == pytest.approx(min(1.0, weight))


@pytest.mark.parametrize("selected, expected", [(0, 1.0), (1, 1.0), (2, 3.5 / 4.0)])
def test_paper_acceptance_reuses_unselected_candidates(selected, expected):
    # Eq. (7), W=3.5: replace selected weight with current state's weight 1.
    assert math.exp(mtm._log_acceptance_probability(np.log([2.0, 1.0, 0.5]), selected)) == pytest.approx(expected)


def test_acceptance_is_stable_for_extreme_log_weights():
    assert mtm._log_acceptance_probability([1000.0, -1000.0], 0) == 0.0
    assert mtm._log_acceptance_probability([-1000.0], 0) == -1000.0
    assert log_sum_exp([-np.inf, -np.inf]) == -np.inf


def test_categorical_probabilities_match_paper(monkeypatch):
    observed = []

    def choose(size, p):
        observed.append(p)
        return 2

    monkeypatch.setattr(np.random, "choice", choose)
    assert mtm._categorical_from_log_weights(np.log([1, 2, 3]) + 1000.0) == 2
    np.testing.assert_allclose(observed[0], [1 / 6, 2 / 6, 3 / 6])


def test_mixture_uses_whole_suffix_and_restarts_at_cut():
    components = np.log([[0.9, 0.8], [0.2, 0.3]])
    mixed = mtm._mixture_logprobs(components)
    assert math.exp(sum(mixed)) == pytest.approx((0.9 * 0.8 + 0.2 * 0.3) / 2)
    assert math.exp(mixed[0]) == pytest.approx(0.55)
    cut_mixed = mtm._mixture_logprobs(components[:, 1:])
    assert math.exp(sum(cut_mixed)) == pytest.approx(0.55)
    assert cut_mixed != pytest.approx(mixed[1:])


@pytest.mark.parametrize("num_tries", [1, 2, 3])
def test_finite_state_mixture_kernel_satisfies_detailed_balance(num_tries):
    # Enumerate every candidate tuple, including repeated states, and every selected slot. This checks the actual
    # transition against its power target.
    base = np.array([0.2, 0.3, 0.5])
    target = base ** 3
    target /= target.sum()
    temperatures = [0.4, 1.0, 1.5]
    components = np.array([base ** (1 / temp) for temp in temperatures])
    components /= components.sum(axis=1, keepdims=True)
    proposal = components.mean(axis=0)
    transition = np.zeros((3, 3))
    for current in range(3):
        for candidates in itertools.product(range(3), repeat=num_tries):
            draw_probability = np.prod(proposal[list(candidates)])
            weights = [
                mtm._log_weight([math.log(target[candidate])], [math.log(target[current])],
                                [math.log(proposal[candidate])], [math.log(proposal[current])])
                for candidate in candidates
            ]
            selection = np.exp(weights - np.max(weights))
            selection /= selection.sum()
            for selected, candidate in enumerate(candidates):
                accepted = math.exp(mtm._log_acceptance_probability(weights, selected))
                mass = draw_probability * selection[selected]
                transition[current, candidate] += mass * accepted
                transition[current, current] += mass * (1 - accepted)
    np.testing.assert_allclose(transition.sum(axis=1), 1.0, atol=1e-14)
    flow = target[:, None] * transition
    np.testing.assert_allclose(flow, flow.T, atol=1e-14)


class _QueuedModel:
    """A small causal LM with deterministic scripted generation for control tests."""

    def __init__(self, batches):
        self.batches = list(batches)
        self.calls = []
        self.forward_calls = 0
        self.logits = torch.tensor([0.0, -0.7, 0.2, -1.5, 0.6, -0.4, 0.1, -0.3])

    def generate(self, input_ids, **kwargs):
        self.calls.append((input_ids.clone(), kwargs))
        assert kwargs["eos_token_id"] is None
        suffix = torch.tensor(self.batches.pop(0), dtype=torch.long)
        assert suffix.shape == (len(input_ids), kwargs["max_new_tokens"])
        sequence = torch.cat([input_ids, suffix], dim=1)
        logits, scores = [], []
        for step in range(suffix.shape[1]):
            raw = self.logits.expand(len(input_ids), -1).clone()
            processed = kwargs["logits_processor"](sequence[:, :input_ids.shape[1] + step], raw.clone())
            logits.append(raw)
            scores.append(processed)
        return SimpleNamespace(sequences=sequence, logits=tuple(logits), scores=tuple(scores))

    def __call__(self, input_ids, **kwargs):
        self.forward_calls += 1
        return SimpleNamespace(logits=self.logits.expand(*input_ids.shape, -1))


def _wrapper(model, *, eos=None, temperature=0.5, alpha=3.0):
    return HF_LLM_Wrapper(model, SimpleNamespace(eos_token_id=eos, pad_token_id=0),
                          torch.device("cpu"), temperature=temperature, alpha=alpha)


@pytest.mark.parametrize("draw, expected", [(0.0, [2, 5, 1, 2]), (0.999999, [2, 5, 4, 4, 4])])
def test_sampler_draws_only_k_candidates_and_preserves_eos_scores(monkeypatch, draw, expected):
    # EOS in the prompt must be ignored. On acceptance, truncate response at EOS after the whole fixed-horizon MTM move
    # and retain its acceptance count.
    model = _QueuedModel([[[4, 4, 4]], [[1, 2, 6], [3, 5, 7]]])
    wrapper = _wrapper(model, eos=2)
    monkeypatch.setattr(mtm, "_categorical_from_log_weights", lambda weights: 0)
    monkeypatch.setattr(np.random, "rand", lambda: draw)
    tokens, proposal, target, acceptance = mtm.multi_try_mcmc_power_sampler(
        wrapper, context=[2, 5], mcmc_steps=1, max_new_tokens=3,
        num_of_blocks=1, num_tries=2, given_cut_idx=2,
    )
    assert tokens == expected
    assert acceptance == (1.0 if draw == 0 else 0.0)
    assert len(proposal) == len(target) == len(tokens) - 2
    assert [len(inputs) for inputs, _ in model.calls] == [1, 2]
    assert not model.batches
    assert wrapper.alpha == 3.0  # Must not become 1 / temperature.
    expected_target = torch.log_softmax(model.logits, dim=-1)[tokens[2:]] * 3
    np.testing.assert_allclose(target, expected_target, rtol=1e-6)


def test_temperature_mixture_generation_and_cross_scoring(monkeypatch):
    model = _QueuedModel([[[1, 4]], [[3, 1], [4, 5], [6, 1]]])
    wrapper = _wrapper(model)
    temperatures = [0.25, 0.6, 1.2]
    draws = iter([[0.6], [1.2, 0.25, 1.2]])
    monkeypatch.setattr(np.random, "choice", lambda choices, size: np.array(next(draws)))
    observed_weights = []

    def select(weights):
        observed_weights.append(weights)
        return 1

    monkeypatch.setattr(mtm, "_categorical_from_log_weights", select)
    monkeypatch.setattr(np.random, "rand", lambda: 0.0)
    tokens, proposal, target, acceptance = mtm.multi_try_mcmc_power_sampler(
        wrapper, context=[7], mcmc_steps=1, max_new_tokens=2, num_of_blocks=1,
        num_tries=3, proposal_temperatures=temperatures,
        given_cut_idx=1,
    )
    assert acceptance == 1.0
    np.testing.assert_allclose(
        model.calls[1][1]["logits_processor"][0].temperatures.flatten(), [1.2, 0.25, 1.2]
    )
    component_density = [
        torch.log_softmax(model.logits / temperature, dim=-1)[tokens[1:]].sum().exp().item()
        for temperature in temperatures
    ]
    assert math.exp(sum(proposal)) == pytest.approx(np.mean(component_density), rel=1e-6)
    assert wrapper.alpha == 3.0
    assert len(target) == 2
    base_logprobs = torch.log_softmax(model.logits, dim=-1).numpy()

    def relative_score(suffix):
        component_logprobs = [
            torch.log_softmax(model.logits / temperature, dim=-1)[suffix].sum().item()
            for temperature in temperatures
        ]
        return 3.0 * base_logprobs[suffix].sum() - math.log(np.exp(component_logprobs).mean())

    expected_weights = [
        relative_score(suffix) - relative_score([1, 4])
        for suffix in [[3, 1], [4, 5], [6, 1]]
    ]
    np.testing.assert_allclose(observed_weights[0], expected_weights, atol=2e-6)


def test_later_cut_resets_mixture_after_an_accepted_move(monkeypatch):
    model = _QueuedModel([[[1, 4]], [[4, 5], [1, 3]], [[3], [6]]])
    wrapper = _wrapper(model)
    temperatures = [0.25, 1.2]
    cuts = iter([1, 2])
    monkeypatch.setattr(mtm.random, "randint", lambda low, high: next(cuts))
    monkeypatch.setattr(np.random, "rand", lambda: 0.0)
    observed_weights = []

    def select(weights):
        observed_weights.append(weights)
        return 0

    monkeypatch.setattr(mtm, "_categorical_from_log_weights", select)
    tokens, proposal, _, acceptance = mtm.multi_try_mcmc_power_sampler(
        wrapper, context=[7], mcmc_steps=2, max_new_tokens=2, num_of_blocks=1,
        num_tries=2, proposal_temperatures=temperatures,
    )
    assert tokens == [7, 4, 3]
    assert acceptance == 1.0
    base = torch.log_softmax(model.logits, -1).numpy()
    component = np.array([
        torch.log_softmax(model.logits / temp, -1).numpy() for temp in temperatures
    ])
    q = np.exp(component).mean(axis=0)
    expected_weights = [3 * (base[token] - base[5]) + np.log(q[5] / q[token]) for token in [3, 6]]
    np.testing.assert_allclose(observed_weights[1], expected_weights, atol=2e-6)
    assert math.exp(sum(proposal)) == pytest.approx(np.exp(component[:, [4, 3]].sum(axis=1)).mean(), rel=1e-6)


def test_real_tiny_hf_model_applies_each_row_temperature_and_scores_components():
    torch.manual_seed(7)
    model = GPT2LMHeadModel(GPT2Config(vocab_size=8, n_positions=16, n_embd=8,
                                      n_layer=1, n_head=2, bos_token_id=7,
                                      eos_token_id=None, pad_token_id=0)).eval()
    wrapper = _wrapper(model, alpha=2.7)
    row_temperatures = [0.25, 1.2]
    sequences, proposal, target, components = batched_low_temp_proposal_sampling(
        wrapper, contexts=[[7, 1], [7, 1]], seq_len=5, ignore_eos=True,
        row_temperatures=row_temperatures, component_temperatures=row_temperatures,
    )
    with torch.inference_mode():
        logits = model(torch.tensor(sequences)).logits[:, 1:-1].float()
    for row in range(2):
        token_ids = torch.tensor(sequences[row][2:]).unsqueeze(-1)
        expected_target = torch.log_softmax(logits[row], -1).gather(-1, token_ids).flatten() * 2.7
        np.testing.assert_allclose(target[row], expected_target, atol=1e-6)
        for component, temperature in enumerate(row_temperatures):
            expected_q = torch.log_softmax(logits[row] / temperature, -1).gather(-1, token_ids).flatten()
            np.testing.assert_allclose(components[row][component], expected_q, atol=1e-6)
        np.testing.assert_allclose(proposal[row], components[row][row], atol=1e-6)
    assert wrapper.alpha == 2.7


def test_zero_refinement_steps_extend_blocks_without_mh_draws():
    model = _QueuedModel([[[1, 4]], [[3, 5]]])
    tokens, proposal, target, acceptance = mtm.multi_try_mcmc_power_sampler(
        _wrapper(model), context=[7], mcmc_steps=0, max_new_tokens=4,
        num_of_blocks=2, proposal_temperatures=[0.25],
    )
    assert tokens == [7, 1, 4, 3, 5]
    assert len(proposal) == len(target) == 4
    assert acceptance == 0.0
    assert [len(inputs) for inputs, _ in model.calls] == [1, 1]


def test_runner_passes_and_records_proposal_temperatures(monkeypatch):
    from power_sharpening.runners.hf import run_power_mh as runner

    temperatures = [0.25, 0.5, 1.0]
    tokenizer = SimpleNamespace(
        encode=lambda text, return_tensors: torch.tensor([[7]]),
        decode=lambda ids, skip_special_tokens: "a response",
    )
    monkeypatch.setattr(runner, "hf_load_model_and_tokenizer", lambda name: (
        torch.device("cpu"), tokenizer, object()
    ))
    problem = {"prompt": "a question", "answer": "an answer"}
    benchmark = SimpleNamespace(
        name="mock", dataset_loader=[None],
        evaluate_completions=lambda problems, completions: [
            {"prediction": "an answer", "is_correct": True}
        ],
    )
    monkeypatch.setattr(runner, "_get_prompts_and_problems", lambda *args: (
        ["a prompt"], [problem]
    ))
    calls = []

    def sample(wrapper, prefix, mcmc_steps, **kwargs):
        calls.append(kwargs)
        return [7, 1], [0.0], [0.0], 0.5

    monkeypatch.setattr(runner, "multi_try_mcmc_power_sampler", sample)
    results = runner.multi_try_sampling(
        "mock-model", benchmark, alpha=4.0, mcmc_steps=1, num_blocks=1,
        num_tries=5, proposal_temperatures=temperatures,
    )
    assert calls[0]["proposal_temperatures"] == temperatures
    assert calls[0]["num_tries"] == 5
    assert results[0]["proposal_temperatures"] == temperatures
    assert results[0]["alpha"] == 4.0
