"""CPU test for SGLang EntropyCut runner batching and output boundaries.

Run with:
    uv run --project /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src \
      pytest /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/tests/test_sglang_entropy_cut_runner.py
"""

import importlib.util
import json
import math
import sys
import types
from types import SimpleNamespace

import pytest

if (
    "sglang" not in sys.modules
    and importlib.util.find_spec("sglang") is None
):
    sys.modules["sglang"] = types.ModuleType("sglang")

from power_sharpening.runners.sglang import run_entropy_cut_mh


class _Engine:
    def __init__(self):
        self.shutdown_called = False

    def shutdown(self):
        self.shutdown_called = True


class _Tokenizer:
    eos_token_id = 0

    def encode(self, text):
        return [len(text)]

    def decode(self, token_ids, skip_special_tokens):
        assert skip_special_tokens is True
        assert len(token_ids) == 1
        return f"completion-{token_ids[0]}"


class _Benchmark:
    name = "fake"
    size = 3

    def __init__(self):
        self.dataset_loader = [
            [{"prompt": "p0"}, {"prompt": "p1"}],
            [{"prompt": "p2"}],
        ]
        self.seen = 0
        self.graded_completions = []

    def get_question_and_answer(self, batch, tokenizer):
        del tokenizer
        return [item["prompt"] for item in batch], list(batch)

    def batch_to_problems(self, batch):
        return batch

    def evaluate_completions(self, problems, completions):
        self.graded_completions.extend(completions)
        return [
            {
                "completion": completion,
                "prediction": completion,
                "is_correct": True,
                "justification": "fake",
            }
            for _, completion in zip(problems, completions)
        ]


def _args(max_samples=None):
    return SimpleNamespace(
        seed=17,
        gpu_memory_utilization=0.8,
        alpha=4.0,
        mcmc_steps=1,
        num_blocks=1,
        max_new_tokens=1,
        cut_power=4.0,
        entropy_mode="topk",
        entropy_top_k=64,
        temperature_schedule_type="const",
        dataset="fake",
        verbose=False,
        max_samples=max_samples,
    )


def test_runner_processes_every_prompt_and_decodes_only_continuation(monkeypatch):
    engine = _Engine()
    tokenizer = _Tokenizer()
    load_kwargs = {}
    sampler_prefixes = []

    def fake_loader(model_str, **kwargs):
        assert model_str == "fake-model"
        load_kwargs.update(kwargs)
        return engine, tokenizer

    def fake_sampler(wrapper, prefix, **kwargs):
        del wrapper, kwargs
        sampler_prefixes.append(list(prefix))
        return list(prefix) + [100 + len(sampler_prefixes)], {
            "attempts": 1,
            "acceptances": 1,
            "acceptance_rate": 1.0,
            "mean_cut_position": 0.0,
            "transitions": [],
        }

    monkeypatch.setattr(
        run_entropy_cut_mh,
        "sglang_load_model_and_tokenizer",
        fake_loader,
    )
    monkeypatch.setattr(
        run_entropy_cut_mh,
        "entropy_cut_mh_sampler",
        fake_sampler,
    )

    benchmark = _Benchmark()
    results, stats = run_entropy_cut_mh.run_entropy_cut(
        "fake-model",
        benchmark,
        _args(),
        proposal_temperature=0.25,
    )

    assert load_kwargs == {
        "random_seed": 17,
        "mem_fraction_static": 0.8,
    }
    assert sampler_prefixes == [[2], [2], [2]]
    assert benchmark.graded_completions == [
        "completion-101",
        "completion-102",
        "completion-103",
    ]
    assert len(results) == 3
    assert len(stats) == 3
    assert [row["idx"] for row in stats] == [0, 1, 2]
    assert engine.shutdown_called is True


def test_runner_honors_max_samples_across_batches(monkeypatch):
    engine = _Engine()
    tokenizer = _Tokenizer()

    monkeypatch.setattr(
        run_entropy_cut_mh,
        "sglang_load_model_and_tokenizer",
        lambda *args, **kwargs: (engine, tokenizer),
    )
    monkeypatch.setattr(
        run_entropy_cut_mh,
        "entropy_cut_mh_sampler",
        lambda wrapper, prefix, **kwargs: (
            list(prefix) + [101],
            {"acceptance_rate": 0.0, "transitions": []},
        ),
    )

    results, stats = run_entropy_cut_mh.run_entropy_cut(
        "fake-model",
        _Benchmark(),
        _args(max_samples=1),
        proposal_temperature=0.25,
    )

    assert len(results) == 1
    assert len(stats) == 1
    assert engine.shutdown_called is True


def test_prompt_helper_uses_benchmark_reconstructed_problems():
    reconstructed = {
        "question": "q",
        "choices": ["a", "b", "c", "d"],
        "answer": 1,
    }

    class _ReconstructingBenchmark:
        def get_question_and_answer(self, batch, tokenizer):
            del batch, tokenizer
            return ["formatted"], [reconstructed]

        def batch_to_problems(self, batch):
            raise AssertionError(
                "runner must not discard benchmark-specific reconstruction"
            )

    prompts, problems = run_entropy_cut_mh._get_prompts_and_problems(
        _ReconstructingBenchmark(),
        batch={"choices": "transposed"},
        tokenizer=_Tokenizer(),
    )

    assert prompts == ["formatted"]
    assert problems == [reconstructed]


def test_json_safe_serializes_infinite_rejection_diagnostics():
    safe = run_entropy_cut_mh._json_safe(
        {
            "cut_term": -math.inf,
            "nested": [math.inf, 1.0],
        }
    )

    assert safe == {
        "cut_term": "-inf",
        "nested": ["inf", 1.0],
    }
    json.dumps(safe, allow_nan=False)


def test_generation_budget_reserves_scoring_margin_and_whole_blocks():
    capped = run_entropy_cut_mh._fit_generation_budget(
        prompt_tokens=100,
        requested_tokens=3072,
        num_blocks=16,
        context_length=2048,
    )

    assert capped == 1936
    assert capped % 16 == 0
    assert 100 + capped + 1 < 2048


def test_generation_budget_fails_when_one_block_cannot_fit():
    with pytest.raises(RuntimeError, match="No positive"):
        run_entropy_cut_mh._fit_generation_budget(
            prompt_tokens=100,
            requested_tokens=16,
            num_blocks=16,
            context_length=110,
        )


def test_visible_completion_stops_at_first_eos():
    assert run_entropy_cut_mh._truncate_at_eos(
        [4, 0, 8, 9],
        eos_token_id=0,
    ) == [4, 0]
