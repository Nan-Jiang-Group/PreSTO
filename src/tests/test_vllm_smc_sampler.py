"""CPU tests for Power-SMC EOS suppression across vLLM generation chunks.

Run with:
    uv run --project /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src \
      pytest /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/tests/test_vllm_smc_sampler.py
"""

import copy
import importlib.util
import math
from pathlib import Path
import sys
import types

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "power_sharpening/backends/vllm/samplers/smc_sampler.py"


@pytest.fixture
def sampler(monkeypatch):
    """Execute the sampler with real CPU Torch and a stubbed vLLM boundary."""
    inputs = types.ModuleType("vllm.inputs")
    inputs.TokensPrompt = dict
    engine_name = "power_sharpening.backends.vllm.engine_patch"
    engine = types.ModuleType(engine_name)
    engine.SamplingParams = types.SimpleNamespace
    params = types.ModuleType(f"{engine_name}.sampling_params")

    def copy_params(original, **overrides):
        cloned = copy.deepcopy(original)
        for name, value in overrides.items():
            setattr(cloned, name, value)
        return cloned

    params._copy_sampling_params = copy_params
    for name, module in (
        ("vllm", types.ModuleType("vllm")),
        ("vllm.inputs", inputs),
        (engine_name, engine),
        (params.__name__, params),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location("_test_smc_sampler", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Tokenizer:
    eos_token_id = 0

    def __init__(self, boxed=False):
        self.boxed = boxed

    def encode(self, prompt, add_special_tokens=False):
        return [9]

    def decode(self, tokens, skip_special_tokens=True):
        if self.boxed and 1 in tokens:
            return r"\boxed{1}"
        return " ".join(str(token) for token in tokens if token != 0)


class _EosEngine:
    """Prefer EOS as soon as allowed; return distinct base/proposal scores."""

    def __init__(self, eos_at=0):
        self.eos_at = eos_at
        self.calls = []

    def generate(self, prompts, sampling_params, use_tqdm=False):
        assert len(prompts) == len(sampling_params)
        self.calls.append((copy.deepcopy(prompts), copy.deepcopy(sampling_params)))
        results = []
        for prompt, params in zip(prompts, sampling_params):
            assert 0 <= params.min_tokens <= params.max_tokens
            assert params.logprobs == 0
            assert params.logprob_token_ids is None
            generated_so_far = len(prompt["prompt_token_ids"]) - 1
            tokens, logp, logq = [], [], []
            for offset in range(params.max_tokens):
                eos_allowed = offset >= params.min_tokens
                token = int(not (
                    eos_allowed and generated_so_far + offset >= self.eos_at
                ))
                tokens.append(token)
                base = math.log(0.8 if token == 0 else 0.2)
                proposal = base if eos_allowed else 0.0
                logp.append({token: types.SimpleNamespace(logprob=base)})
                logq.append({token: types.SimpleNamespace(logprob=proposal)})
                if token == 0:
                    break
            output = types.SimpleNamespace(
                token_ids=tokens, logprobs=logq, power_logprobs=logp,
            )
            results.append(types.SimpleNamespace(outputs=[output]))
        return results


def _run(sampler, *, eos_at=0, boxed=False, **overrides):
    engine = _EosEngine(eos_at=eos_at)
    wrapper = types.SimpleNamespace(llm=engine, tokenizer=_Tokenizer(boxed=boxed))
    params = types.SimpleNamespace(
        alpha=4.0, temperature=0.25, top_k=0, top_p=0.9,
        min_tokens=7, logprobs=5, logprob_token_ids=[2], n=3,
    )
    original = copy.deepcopy(vars(params))
    options = dict(
        max_new_tokens=200, block_size=64, n_particles=2,
        ess_threshold=0.0, alpha_ramp_tokens=1, device="cpu",
    )
    options.update(overrides)
    result = sampler.smc_power_sampler(wrapper, ["prompt"], params, **options)
    assert vars(params) == original
    for _, requests in engine.calls:
        for request in requests:
            assert (request.top_k, request.top_p, request.alpha) == (0, 0.9, 4.0)
    return result, engine


@pytest.mark.parametrize(
    ("minimum", "horizon", "eos_at", "expected_minima", "expected_length"),
    [
        (100, 200, 0, [64, 36], 100),
        (100, 200, 140, [64, 36, 0], 140),
        (64, 200, 0, [64, 0], 64),
        (0, 200, 0, [0], 0),
        (100, 70, 0, [64, 6], 70),
        (100, 20, 0, [20], 20),
    ],
)
def test_eos_minimum_is_global_and_clipped_to_each_chunk(
    sampler, minimum, horizon, eos_at, expected_minima, expected_length,
):
    result, engine = _run(
        sampler, min_new_tokens=minimum, max_new_tokens=horizon, eos_at=eos_at,
    )
    assert [requests[0].min_tokens for _, requests in engine.calls] == expected_minima
    assert all(
        request.min_tokens == expected
        for (_, requests), expected in zip(engine.calls, expected_minima)
        for request in requests
    )
    assert result == [" ".join(["1"] * expected_length)]


def test_default_minimum_preserves_base_scores_in_importance_weights(sampler, monkeypatch):
    final_weights = []
    normalize = sampler._normalize_log_weights

    def capture(log_weights):
        final_weights.append(log_weights.tolist())
        return normalize(log_weights)

    monkeypatch.setattr(sampler, "_normalize_log_weights", capture)
    _run(sampler)
    # 100 forced non-EOS tokens have log q=0 but base log p=log(.2). The first permitted EOS has log p=log q=log(.8).
    expected = 100 * 4 * math.log(0.2) + 3 * math.log(0.8)
    assert final_weights[-1] == pytest.approx([expected, expected], rel=2e-6)


def test_boxed_answer_can_finish_before_eos_minimum(sampler):
    result, engine = _run(sampler, boxed=True, min_new_tokens=100)
    assert result == [r"\boxed{1}"]
    assert len(engine.calls) == 1


@pytest.mark.parametrize("minimum", [-1, 1.5, None])
def test_invalid_minimum_fails_before_using_engine(sampler, minimum):
    with pytest.raises(ValueError, match="min_new_tokens"):
        sampler.smc_power_sampler(None, "prompt", None, min_new_tokens=minimum)
