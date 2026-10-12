"""CPU checks for benchmark records used by the vLLM subtree runner.

Run with:
    uv run --project /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src \
      pytest /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/tests/test_vllm_subtree_runner.py
"""

from __future__ import annotations

import importlib.util
from contextlib import nullcontext
from pathlib import Path
import sys
import types

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "power_sharpening/runners/vllm/run_subtree_prefetching_mh.py"


@pytest.fixture
def runner(monkeypatch):
    """Load the runner without requiring a local vLLM installation."""
    engine_name = "power_sharpening.backends.vllm.engine_patch"
    engine_stub = types.ModuleType(engine_name)
    engine_stub.SamplingParams = object

    wrapper_name = "power_sharpening.backends.vllm.wrapper"
    wrapper_stub = types.ModuleType(wrapper_name)
    wrapper_stub.vLLM_Wrapper = object

    sampler_name = (
        "power_sharpening.backends.vllm.samplers."
        "subtree_prefetching_MH_sampler"
    )
    sampler_stub = types.ModuleType(sampler_name)
    sampler_stub.subtree_prefetching_sampling = None
    sampler_stub.subtree_prefetching_sampling_with_entropy_cut = None

    monkeypatch.setitem(sys.modules, engine_name, engine_stub)
    monkeypatch.setitem(sys.modules, wrapper_name, wrapper_stub)
    monkeypatch.setitem(sys.modules, sampler_name, sampler_stub)

    spec = importlib.util.spec_from_file_location(
        "_test_vllm_subtree_runner",
        MODULE_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prompt_helper_keeps_benchmark_normalized_problems(runner):
    """Do not reconstruct collated records after a benchmark normalizes them."""
    normalized = {
        "question": "Which answer is correct?",
        "choices": ["a", "b", "c", "d"],
        "correct_letter": "B",
    }

    class _Benchmark:
        def get_question_and_answer(self, batch, tokenizer):
            del batch, tokenizer
            return ["formatted prompt"], [normalized]

        def batch_to_problems(self, batch):
            raise AssertionError(
                "runner must not discard benchmark-specific reconstruction"
            )

    prompts, problems = runner._get_prompts_and_problems(
        _Benchmark(),
        batch={"choices": "transposed"},
        tokenizer=object(),
    )

    assert prompts == ["formatted prompt"]
    assert problems == [normalized]


@pytest.mark.parametrize(
    ("problem", "formatted_prompt", "expected"),
    [
        ({"prompt": "prompt field"}, "formatted", "prompt field"),
        ({"question": "question field"}, "formatted", "question field"),
        ({"text": "text field"}, "formatted", "text field"),
        ({}, "formatted fallback", "formatted fallback"),
    ],
)
def test_question_text_supports_benchmark_field_names(
    runner,
    problem,
    formatted_prompt,
    expected,
):
    assert runner._question_text(problem, formatted_prompt) == expected


@pytest.mark.parametrize("cut_dist_type", ["uniform", "entropy"])
@pytest.mark.parametrize("evict_cache", [False, True])
def test_runner_selects_the_requested_subtree_sampler(runner, monkeypatch, cut_dist_type, evict_cache):
    calls = []

    class SamplerReached(Exception):
        pass

    def sample(model, **kwargs):
        calls.append(kwargs)
        raise SamplerReached

    name = "subtree_prefetching_sampling_with_entropy_cut" if cut_dist_type == "entropy" else "subtree_prefetching_sampling"
    monkeypatch.setattr(runner, name, sample)
    monkeypatch.setattr(runner, "SamplingParams", types.SimpleNamespace)
    model = types.SimpleNamespace(llm=object(), tokenizer=types.SimpleNamespace(encode=lambda text: [10]))
    def wrapper(**kwargs):
        assert kwargs["enable_subtree_cache_eviction"] is evict_cache
        return nullcontext(model)
    monkeypatch.setattr(runner, "vLLM_Wrapper", wrapper)
    probe = types.SimpleNamespace(start=lambda: None, attach=lambda llm: None, close=lambda: None)
    monkeypatch.setattr(runner, "ResourceProbe", lambda **kwargs: probe)
    benchmark = types.SimpleNamespace(
        name="test",
        dataset_loader=[object()],
        get_question_and_answer=lambda batch, tokenizer: (["prompt"], [{"answer": "answer"}]),
    )
    args = types.SimpleNamespace(
        cut_dist_type=cut_dist_type,
        cut_dist_param=2.5,
        seed=0,
        resource_probe=False,
        prefix_cache=True,
        verbose=False,
        alpha=4.0,
        num_blocks=2,
        max_samples=1,
        max_new_tokens=4,
        mcmc_steps=1,
        prefetch_budget=4,
        rank_fn="accept_first",
        print_tree=False,
        evict_subtree_cache=evict_cache,
    )
    with pytest.raises(SamplerReached):
        runner.run_subtree_prefetching("test-model", benchmark, args, max_model_len=None, init_temperature=0.25)

    assert len(calls) == 1
    assert calls[0]["prompt_ids"] == [10]
    assert calls[0]["max_new_tokens"] == 4
    assert calls[0]["evict_subtree_cache"] is evict_cache
    assert "cut_dist_type" not in calls[0] and "cut_dist_param" not in calls[0]
    if cut_dist_type == "entropy":
        assert calls[0]["cut_power"] == 2.5
    else:
        assert "cut_power" not in calls[0]


@pytest.mark.parametrize("flags,expected", [([], False), (["--evict_subtree_cache"], True), (["--evict_subtree_cache", "--no-evict_subtree_cache"], False)])
def test_cache_eviction_cli_toggle(runner, flags, expected):
    args = runner.build_parser().parse_args(["--dataset", "math500", *flags])
    assert args.evict_subtree_cache is expected


def test_cache_eviction_requires_prefix_caching_before_model_load(runner):
    args = types.SimpleNamespace(evict_subtree_cache=True, prefix_cache=False)
    with pytest.raises(ValueError, match="prefix_cache=true"):
        runner.run_subtree_prefetching("model", None, args, None, 0.25)
