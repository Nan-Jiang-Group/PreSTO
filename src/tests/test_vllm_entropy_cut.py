"""CPU checks for EntropyCut support in the custom vLLM backend.

Run with:
    uv run --project /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src \
      pytest /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/tests/test_vllm_entropy_cut.py
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys
import types

import pytest
import torch

ENGINE_DIR = Path(__file__).resolve().parents[1] / "power_sharpening/backends/vllm/engine_patch"

# Load the numerical helpers without importing the vLLM engine package.
_entropy_spec = importlib.util.spec_from_file_location(
    "_test_vllm_entropy_helpers", ENGINE_DIR / "entropy.py"
)
_entropy = importlib.util.module_from_spec(_entropy_spec)
_entropy_spec.loader.exec_module(_entropy)
predictive_entropies = _entropy.predictive_entropies
suffix_log_acceptance = _entropy.suffix_log_acceptance


def test_predictive_entropies_cover_the_full_vocabulary_and_zero_mass_tokens():
    probabilities = torch.tensor([
        [0.25, 0.25, 0.25, 0.25],
        [0.75, 0.25, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
    ])
    actual = predictive_entropies(probabilities.log())
    expected_binary = -(0.75 * math.log(0.75) + 0.25 * math.log(0.25))

    assert actual.tolist() == pytest.approx([math.log(4.0), expected_binary, 0.0])
    assert actual.device == probabilities.device


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_predictive_entropies_are_stable_in_float32(dtype):
    logits = torch.tensor([[1000.0, 999.0, -1000.0], [0.0, 0.0, 0.0]], dtype=dtype)
    before = logits.clone()
    expected = torch.distributions.Categorical(logits=logits.to(torch.float64)).entropy()

    actual = predictive_entropies(logits)

    assert actual.dtype == torch.float32
    assert actual.tolist() == pytest.approx(expected.tolist(), abs=1e-6)
    torch.testing.assert_close(logits, before)


def test_suffix_log_acceptance_uses_the_reverse_proposal_density():
    actual = suffix_log_acceptance(
        proposed_logq=[-0.2, -0.4],
        current_logq=[-0.7, -0.8],
        proposed_logp=[-0.3, -0.5],
        current_logp=[-0.6, -0.9],
        alpha=4.0,
    )

    target_term = 4.0 * ((-0.3 - 0.5) - (-0.6 - 0.9))
    proposal_term = (-0.7 - 0.8) - (-0.2 - 0.4)
    assert actual == pytest.approx(target_term + proposal_term)





@pytest.fixture
def proposal_module(monkeypatch):
    """Load the proposal functions and real MH score helper with vLLM imports stubbed."""
    engine_name = "power_sharpening.backends.vllm.engine_patch"
    engine = types.ModuleType(engine_name)
    engine.__path__ = [str(ENGINE_DIR)]
    engine.SamplingParams = types.SimpleNamespace
    sampling = types.ModuleType(engine_name + ".sampling_params")

    def copy_sampling_params(params, **overrides):
        return types.SimpleNamespace(**(vars(params) | overrides))

    sampling._copy_sampling_params = copy_sampling_params
    logprobs = types.ModuleType("vllm.logprobs")
    logprobs.Logprob = object
    for module in (engine, sampling, logprobs):
        monkeypatch.setitem(sys.modules, module.__name__, module)

    sampler_dir = Path(__file__).resolve().parents[1] / "power_sharpening/backends/vllm/samplers"
    for name in ("power_sampling_mh", "proposal_llm_call"):
        module_name = "power_sharpening.backends.vllm.samplers." + name
        spec = importlib.util.spec_from_file_location(module_name, sampler_dir / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, module_name, module)
        spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("use_entropy_cut", [False, True])
def test_batched_proposals_keep_prefixes_and_score_each_cut(proposal_module, monkeypatch, use_entropy_cut):
    module = proposal_module
    suffixes = [[4, 5, 6], [7, 8]]
    logq = [[-0.2, -0.4, -0.6], [-0.7, -0.9]]
    base_probs = [[0.9, 0.6, 0.5], [0.6, 0.5]]
    calls = []

    def binary_entropy(p):
        return -p * math.log(p) - (1 - p) * math.log(1 - p)

    def generate(prompts, sampling_params, use_tqdm):
        calls.append(prompts)
        assert prompts == [{"prompt_token_ids": [10, 11]}, {"prompt_token_ids": [10, 11, 1]}]
        assert [params.max_tokens for params in sampling_params] == [3, 2]
        assert all(params.logprobs == 1 and params.ignore_eos for params in sampling_params)
        outputs = []
        for tokens, scores, probs in zip(suffixes, logq, base_probs):
            completion = types.SimpleNamespace(
                token_ids=tokens,
                logprobs=[{t: types.SimpleNamespace(logprob=q)} for t, q in zip(tokens, scores)],
                power_logprobs=[
                    {t: types.SimpleNamespace(logprob=math.log(p))}
                    for t, p in zip(tokens, probs)
                ],
                entropies=[binary_entropy(p) for p in probs],
            )
            outputs.append(types.SimpleNamespace(outputs=[completion]))
        return outputs

    current_logq = [-0.3, -0.4, -0.8]
    current_logp = [math.log(p) for p in [0.95, 0.8, 0.55]]
    current_entropies = [binary_entropy(p) for p in [0.95, 0.8, 0.55]]
    params = types.SimpleNamespace(alpha=2.0, temperature=0.5, ignore_eos=True, logprobs=7)
    kwargs = dict(
        mh_llm_model=types.SimpleNamespace(llm=types.SimpleNamespace(generate=generate)),
        parent_proposal_seq=[1, 2, 3],
        batched_cut_indexes=[0, 1],
        proposal_logprobs_seq=current_logq,
        power_logprobs_seq=current_logp,
        sampling_params=params,
        prompt_ids=[10, 11],
    )
    if use_entropy_cut:
        result = module.batched_proposal_call_with_entropy_cut(**kwargs, cut_entropies=current_entropies, cut_power=2)
    else:
        def unexpected_entropy(*args, **kwargs):
            pytest.fail("standard proposals must not compute entropy or cut probabilities")

        monkeypatch.setattr(module, "_suffix_entropies", unexpected_entropy)
        monkeypatch.setattr(module, "compute_entropy_cut_policy", unexpected_entropy)
        result = module.batched_proposal_call(**kwargs)

    assert len(calls) == 1
    assert len(result) == (5 if use_entropy_cut else 4)
    proposals, log_ratios, cached_logq, cached_logp = result[:4]
    assert proposals == [[4, 5, 6], [1, 7, 8]]
    assert cached_logq == logq
    assert params.logprobs == 7  # Per-request overrides must leave caller settings intact.

    def cut_probability(entropies, cut):
        weights = [(right - left) ** 2 for left, right in zip(entropies, entropies[1:])]
        return weights[cut] / sum(weights)

    for cut, probs in enumerate(base_probs):
        expected_logp = [math.log(p) for p in probs]
        assert cached_logp[cut] == pytest.approx(expected_logp)
        expected = 2 * (sum(expected_logp) - sum(current_logp[cut:])) + sum(current_logq[cut:]) - sum(logq[cut])
        if use_entropy_cut:
            suffix_entropies = [binary_entropy(p) for p in probs]
            assert result[4][cut] == pytest.approx(suffix_entropies)
            proposed_entropies = current_entropies[:cut] + suffix_entropies
            expected += math.log(cut_probability(proposed_entropies, cut) / cut_probability(current_entropies, cut))
        assert log_ratios[cut] == pytest.approx(expected)


def _load_subtree_module(monkeypatch):
    """Load the vLLM subtree driver with model-free dependency stubs."""
    proposal_name = (
        "power_sharpening.backends.vllm.samplers.proposal_llm_call"
    )
    proposal_stub = types.ModuleType(proposal_name)
    proposal_stub.batched_proposal_call = None
    proposal_stub.batched_proposal_call_with_entropy_cut = None
    proposal_stub.vllm_proposal_sampling = None

    sampling_name = (
        "power_sharpening.backends.vllm.engine_patch.sampling_params"
    )
    sampling_stub = types.ModuleType(sampling_name)

    def copy_sampling_params(sampling_params, **overrides):
        values = dict(vars(sampling_params))
        values.update(overrides)
        return types.SimpleNamespace(**values)

    sampling_stub._copy_sampling_params = copy_sampling_params

    monkeypatch.setitem(sys.modules, proposal_name, proposal_stub)
    monkeypatch.setitem(sys.modules, sampling_name, sampling_stub)

    module_path = (
        Path(__file__).resolve().parents[1]
        / "power_sharpening/backends/vllm/samplers/subtree_prefetching_MH_sampler.py"
    )
    module_name = "_test_vllm_subtree_entropy_cut"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Tokenizer:
    eos_token_id = None


class _Wrapper:
    tokenizer = _Tokenizer()


def test_vllm_subtree_requests_and_reports_entropies(monkeypatch):
    subtree = _load_subtree_module(monkeypatch)
    calls = []

    def proposal(
        model,
        context_ids,
        max_new_tokens,
        sampling_params,
        return_entropies,
        verbose,
    ):
        calls.append(return_entropies)
        return [1, 2], [-0.1, -0.2], [-0.3, -0.4], [0.7, 0.9]

    subtree.vllm_proposal_sampling = proposal
    tokens, stats = subtree.subtree_prefetching_sampling_with_entropy_cut(
        _Wrapper(),
        prompt_ids=[10],
        sampling_params=types.SimpleNamespace(alpha=4.0, temperature=0.25),
        mcmc_steps=0,
        max_new_tokens=2,
        num_of_blocks=1,
    )

    assert calls == [True]
    assert tokens == [1, 2]
    assert stats.base_diagnostics is not None
    assert stats.base_diagnostics.logprob_sum == pytest.approx(-0.7)
    assert stats.base_diagnostics.neg_entropy_sum == pytest.approx(-1.6)


@pytest.mark.parametrize("cut_dist_type", ["uniform", "entropy"])
@pytest.mark.parametrize("evict_cache", [False, True])
def test_vllm_subtree_routes_proposals_and_splices_accepted_caches(monkeypatch, cut_dist_type, evict_cache):
    subtree = _load_subtree_module(monkeypatch)
    forwarded = {"remaining_steps": [], "collect_entropies": [], "score_caches": []}
    use_entropy_cut = cut_dist_type == "entropy"
    decisions = iter([True, False])
    eviction_calls = []

    def proposal(*args, **kwargs):
        assert kwargs["return_entropies"] is use_entropy_cut
        assert kwargs["sampling_params"].ignore_eos
        result = (
            [1, 2, 3],
            [-0.1, -0.2, -0.3],
            [-0.4, -0.5, -0.6],
        )
        return result + ([1.0, 2.0, 2.0],) if use_entropy_cut else result

    def collect(*args, **kwargs):
        assert kwargs["cut_dist_type"] == cut_dist_type
        if not use_entropy_cut:
            assert "cut_entropies" not in kwargs
            assert "cut_dist_param" not in kwargs
        forwarded["remaining_steps"].append(kwargs["mcmc_steps"])
        forwarded["collect_entropies"].append(list(kwargs["cut_entropies"]) if use_entropy_cut else None)
        return {0: 7}, {7: 0}, [1]

    def batch(*args, **kwargs):
        forwarded["score_caches"].append((list(kwargs["proposal_logprobs_seq"]), list(kwargs["power_logprobs_seq"])))
        if use_entropy_cut:
            forwarded["batch_entropies"] = list(kwargs["cut_entropies"])
            forwarded["cut_power"] = kwargs["cut_power"]
        else:
            assert "cut_entropies" not in kwargs
            assert "cut_power" not in kwargs
        result = (
            [[1, 4, 5]],
            [0.0],
            [[-0.7, -0.8]],
            [[-0.9, -1.0]],
        )
        return result + ([[4.0, 5.0]],) if use_entropy_cut else result

    def walk(**kwargs):
        if next(decisions):
            leaf = types.SimpleNamespace(proposal=[1, 4, 5])
            accepted = types.SimpleNamespace(node_id=7)
            return leaf, accepted, [(True,), (False,)], 1
        leaf = types.SimpleNamespace(proposal=kwargs["subtree_root_proposal"])
        return leaf, None, [(False,)], 0

    subtree.vllm_proposal_sampling = proposal
    subtree.collect_batch_cut_indicesv3 = collect
    if use_entropy_cut:
        subtree.batched_proposal_call_with_entropy_cut = batch
    else:
        subtree.batched_proposal_call = batch
    subtree.build_and_sample_subtree = walk

    sampling_fn = (
        subtree.subtree_prefetching_sampling_with_entropy_cut if use_entropy_cut else subtree.subtree_prefetching_sampling
    )
    wrapper = _Wrapper()
    if evict_cache:
        def evict(sequences, keep_token_ids):
            eviction_calls.append((sequences, keep_token_ids))
            return {"evicted_blocks": 2}
        wrapper.evict_subtree_cache = evict
    tokens, stats = sampling_fn(
        wrapper,
        prompt_ids=[10],
        sampling_params=types.SimpleNamespace(alpha=4.0, temperature=0.25),
        mcmc_steps=3,
        max_new_tokens=3,
        num_of_blocks=1,
        evict_subtree_cache=evict_cache,
        **({"cut_power": 4.0} if use_entropy_cut else {}),
    )

    assert tokens == [1, 4, 5]
    assert forwarded["remaining_steps"] == [3, 1]
    assert forwarded["score_caches"] == [
        ([-0.1, -0.2, -0.3], [-0.4, -0.5, -0.6]),
        ([-0.1, -0.7, -0.8], [-0.4, -0.9, -1.0]),
    ]
    assert stats.walked_steps == [2, 1]
    assert stats.acceptances == [1, 0]
    assert len(eviction_calls) == (2 if evict_cache else 0)
    if evict_cache:
        assert eviction_calls == [
            ([[10, 1, 2, 3], [10, 1, 4, 5]], [10, 1, 4, 5]),
            ([[10, 1, 4, 5], [10, 1, 4, 5]], [10, 1, 4, 5]),
        ]
        assert stats.to_json()["cache_evicted_blocks"] == 4
    if use_entropy_cut:
        assert forwarded["collect_entropies"] == [[1.0, 2.0, 2.0], [1.0, 4.0, 5.0]]
        assert forwarded["batch_entropies"] == [1.0, 4.0, 5.0]
        assert forwarded["cut_power"] == 4.0
        assert stats.base_diagnostics is not None
        assert stats.base_diagnostics.logprob_sum == pytest.approx(-2.3)
        assert stats.base_diagnostics.neg_entropy_sum == pytest.approx(-10.0)
    else:
        assert forwarded["collect_entropies"] == [None, None]
        assert stats.base_diagnostics is None


@pytest.mark.parametrize("use_entropy_cut", [False, True])
@pytest.mark.parametrize("stop_on_eos", [False, True])
@pytest.mark.parametrize("evict_cache", [False, True])
def test_split_subtree_samplers_keep_eos_and_diagnostics_aligned(monkeypatch, use_entropy_cut, stop_on_eos, evict_cache):
    subtree = _load_subtree_module(monkeypatch)
    calls = []
    cleanup = []

    def proposal(model, context_ids, max_new_tokens, sampling_params, return_entropies, verbose):
        calls.append(list(context_ids))
        assert sampling_params.ignore_eos
        assert return_entropies is use_entropy_cut
        result = ([1, 0, 2], [-0.1, -0.2, -0.3], [-0.4, -0.5, -0.6])
        return result + ([0.7, 0.8, 0.9],) if use_entropy_cut else result

    subtree.vllm_proposal_sampling = proposal
    sampler = (
        subtree.subtree_prefetching_sampling_with_entropy_cut if use_entropy_cut else subtree.subtree_prefetching_sampling
    )
    wrapper = types.SimpleNamespace(tokenizer=types.SimpleNamespace(eos_token_id=0))
    if evict_cache:
        def evict(sequences, keep_token_ids):
            cleanup.append((sequences, keep_token_ids))
            return {"evicted_blocks": 1}
        wrapper.evict_subtree_cache = evict
    tokens, stats = sampler(
        wrapper,
        prompt_ids=[10],
        sampling_params=types.SimpleNamespace(alpha=4.0, temperature=0.25),
        mcmc_steps=0,
        max_new_tokens=6,
        num_of_blocks=2,
        stop_on_eos=stop_on_eos,
        evict_subtree_cache=evict_cache,
    )
    assert calls == ([[10]] if stop_on_eos else [[10], [10, 1, 0, 2]])
    assert tokens == [1, 0]
    assert len(cleanup) == int(evict_cache)
    if evict_cache:
        assert cleanup[0][1] == [10, 1, 0]
    if use_entropy_cut:
        assert stats.base_diagnostics.num_tokens == 2
        assert stats.base_diagnostics.logprob_sum == pytest.approx(-0.9)
        assert stats.base_diagnostics.neg_entropy_sum == pytest.approx(-1.5)
    else:
        assert stats.base_diagnostics is None


def test_entropy_subtree_rejects_annealing_before_generation(monkeypatch):
    subtree = _load_subtree_module(monkeypatch)
    with pytest.raises(ValueError, match="fixed proposal temperature"):
        subtree.subtree_prefetching_sampling_with_entropy_cut(
            None,
            prompt_ids=[10],
            sampling_params=types.SimpleNamespace(alpha=4.0, temperature=0.25),
            temperature_scheduler=types.SimpleNamespace(is_anneal=True),
        )
