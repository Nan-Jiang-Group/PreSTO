"""CPU tests for the custom-vLLM Entropy-Cut PowerMH driver.

Run with:
    uv run --project /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src \
      pytest /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/tests/test_vllm_entropycut_power_sampler.py
"""

from __future__ import annotations

import importlib.util
import logging
import math
from pathlib import Path
import sys
import types

import numpy as np
import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "power_sharpening/backends/vllm/samplers/entropycut_power_sampling_mh.py"
)


def _load_module(monkeypatch):
    """Load the sampler without requiring a local vLLM installation."""
    vllm_stub = types.ModuleType("vllm")
    logprobs_stub = types.ModuleType("vllm.logprobs")
    logprobs_stub.Logprob = object
    vllm_stub.logprobs = logprobs_stub

    engine_name = "power_sharpening.backends.vllm.engine_patch"
    engine_stub = types.ModuleType(engine_name)
    engine_stub.__path__ = [str(MODULE_PATH.parents[1] / "engine_patch")]
    engine_stub.SamplingParams = object

    sampling_name = f"{engine_name}.sampling_params"
    sampling_stub = types.ModuleType(sampling_name)

    def copy_sampling_params(sampling_params, **overrides):
        values = dict(vars(sampling_params))
        values.update(overrides)
        return types.SimpleNamespace(**values)

    sampling_stub._copy_sampling_params = copy_sampling_params

    monkeypatch.setitem(sys.modules, "vllm", vllm_stub)
    monkeypatch.setitem(sys.modules, "vllm.logprobs", logprobs_stub)
    monkeypatch.setitem(sys.modules, engine_name, engine_stub)
    monkeypatch.setitem(sys.modules, sampling_name, sampling_stub)

    spec = importlib.util.spec_from_file_location(
        "_test_entropycut_power_sampling_mh",
        MODULE_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _power_dict(sampled_token, dist):
    """Return only the sampled token's score; entropy arrives in a separate field."""
    return {sampled_token: types.SimpleNamespace(logprob=math.log(dist[0]))}


def _completion(token_ids, logq, power_dists):
    """Build the minimal custom-vLLM completion shape used by the sampler.

    ``power_dists`` gives the full base-model distribution for each token.
    """
    return types.SimpleNamespace(
        token_ids=list(token_ids),
        logprobs=[
            {token: types.SimpleNamespace(logprob=value)}
            for token, value in zip(token_ids, logq)
        ],
        power_logprobs=[
            _power_dict(token, dist)
            for token, dist in zip(token_ids, power_dists)
        ],
        entropies=[max(0.0, -sum(p * math.log(p) for p in dist if p > 0)) for dist in power_dists],
    )


def _request(completion):
    return types.SimpleNamespace(outputs=[completion])


class _DeterministicRng:
    """Choose the highest-mass cut and accept every finite proposal."""

    def __init__(self):
        self.cut_probabilities = []

    def choice(self, size, p):
        assert size == len(p)
        self.cut_probabilities.append(np.asarray(p))
        return int(np.argmax(p))

    def random(self):
        return 0.0


class _Tokenizer:
    eos_token_id = 0

    @staticmethod
    def decode(tokens, skip_special_tokens=True):
        return " ".join(str(token) for token in tokens)


class _FakeLlm:
    def __init__(self):
        self.calls = []

    def generate(self, prompts, sampling_params, use_tqdm):
        self.calls.append((prompts, sampling_params, use_tqdm))
        if len(self.calls) == 1:
            # Entropies [0, ln2, ln2] -> positive jump then flat, so the cut law concentrates on the first jump (offset
            # 0 -> cut index 1).
            return [
                _request(
                    _completion(
                        [1, 2, 3],
                        [-0.1, -0.1, -0.1],
                        [[1.0], [0.5, 0.5], [0.5, 0.5]],
                    )
                )
            ]
        return [
            _request(
                _completion(
                    [8, 9],
                    [-0.1, -0.1],
                    [[0.5, 0.5], [0.5, 0.5]],
                )
            )
        ]


class _Wrapper:
    verbose = False
    tokenizer = _Tokenizer()

    def __init__(self):
        self.llm = _FakeLlm()
        self.built_prompts = []

    def _build_prompt(self, prompt, generated_tokens):
        built = (prompt, tuple(generated_tokens))
        self.built_prompts.append(built)
        return built


def test_complete_ratio_includes_the_state_dependent_cut_correction(monkeypatch):
    sampler = _load_module(monkeypatch)

    ratio = sampler.compute_acceptance_ratio(
        proposed_logprobs=[-0.2, -0.4],
        curr_logprobs=[-0.7, -0.8],
        proposed_power_logprobs=[-0.3, -0.5],
        curr_power_logprobs=[-0.6, -0.9],
        alpha=4.0,
        current_entropies=[1.0, 2.0, 2.0],
        proposed_suffix_entropies=[2.0, 4.0],
        cut_index=1,
        beta=4.0,
    )

    suffix_ratio = 4.0 * ((-0.3 - 0.5) - (-0.6 - 0.9))
    suffix_ratio += (-0.7 - 0.8) - (-0.2 - 0.4)
    assert ratio == pytest.approx(suffix_ratio - math.log(17.0))


def test_sampler_uses_entropy_jumps_and_splices_all_caches(monkeypatch, caplog):
    sampler = _load_module(monkeypatch)
    wrapper = _Wrapper()
    rng = _DeterministicRng()
    caplog.set_level(logging.INFO, logger="[entropy-cut MH]")

    completions = sampler.mcmc_power_sampler_entropycut(
        wrapper,
        prompts="prompt",
        sampling_params=types.SimpleNamespace(alpha=4.0, temperature=0.25),
        num_of_blocks=1,
        max_new_tokens=3,
        mcmc_steps=1,
        beta=4.0,
        rng=rng,
    )

    assert completions == ["1 8 9"]
    assert wrapper.built_prompts == [("prompt", ()), ("prompt", (1,))]
    assert len(rng.cut_probabilities) == 1
    np.testing.assert_array_equal(rng.cut_probabilities[0], [1.0, 0.0])

    extension_params = wrapper.llm.calls[0][1]
    proposal_params = wrapper.llm.calls[1][1]
    assert extension_params.logprobs == 1
    assert extension_params.ignore_eos is True
    assert extension_params.temperature == pytest.approx(0.25)
    assert proposal_params[0].logprobs == 1
    assert proposal_params[0].max_tokens == 2
    assert proposal_params[0].temperature == pytest.approx(0.25)
    assert (
        "[step 0, prompt 0] Selected cut 1: keep the first 1 token and "
        "regenerate the remaining 2 of 3 tokens. Entropy changed from "
        "0.0000 to 0.6931 at this cut; the positive increase used by the "
        "sampler was 0.6931."
        in caplog.text
    )
    assert "lambda_beta (" not in caplog.text
    assert "entropy vector (" not in caplog.text


def test_scheduler_supplies_the_fixed_proposal_temperature(monkeypatch):
    sampler = _load_module(monkeypatch)
    wrapper = _Wrapper()

    class _Scheduler:
        def __init__(self):
            self.step_count = 7

        def step(self):
            self.step_count += 1
            return 0.5

    sampler.mcmc_power_sampler_entropycut(
        wrapper,
        prompts="prompt",
        sampling_params=types.SimpleNamespace(alpha=4.0, temperature=0.25),
        num_of_blocks=1,
        max_new_tokens=3,
        mcmc_steps=1,
        beta=4.0,
        temperature_scheduler=_Scheduler(),
        rng=_DeterministicRng(),
    )

    assert wrapper.llm.calls[0][1].temperature == pytest.approx(0.5)
    assert wrapper.llm.calls[1][1][0].temperature == pytest.approx(0.5)


class _EosBlockLlm:
    """Return a two-token block whose second token is the EOS token."""

    def __init__(self):
        self.calls = []

    def generate(self, prompts, sampling_params, use_tqdm):
        self.calls.append((prompts, sampling_params, use_tqdm))
        return [
            _request(
                _completion(
                    [1, _Tokenizer.eos_token_id],
                    [-0.1, -0.1],
                    [[0.5, 0.5], [0.5, 0.5]],
                )
            )
            for _ in prompts
        ]


def test_a_block_ending_on_eos_retires_the_prompt(monkeypatch):
    sampler = _load_module(monkeypatch)
    wrapper = _Wrapper()
    wrapper.llm = _EosBlockLlm()

    completions = sampler.mcmc_power_sampler_entropycut(
        wrapper,
        prompts="prompt",
        sampling_params=types.SimpleNamespace(alpha=4.0, temperature=0.25),
        num_of_blocks=2,
        max_new_tokens=4,
        mcmc_steps=0,
        beta=4.0,
        rng=_DeterministicRng(),
    )

    # The second block never runs, so generation stops at the first block.
    assert completions == ["1 0"]
    assert len(wrapper.llm.calls) == 1
