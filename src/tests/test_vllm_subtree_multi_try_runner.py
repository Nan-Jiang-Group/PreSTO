"""CPU checks for subtree MultiTryMH runner batching, artifacts, and cleanup.

Run with:
    python -m pytest /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/tests/test_vllm_subtree_multi_try_runner.py
"""

import importlib
import json
import sys
import types

import pytest

from test_vllm_multi_try_runner import (
    PACKAGE_DIR, _Benchmark, _Wrapper, _args, _load_module,
    runner as baseline_runner,
)


@pytest.fixture
def runner(baseline_runner, monkeypatch):
    package = importlib.import_module("power_sharpening.runners.vllm")
    monkeypatch.setattr(package, "run_multi_try_mh", baseline_runner, raising=False)
    name = "power_sharpening.backends.vllm.samplers.subtree_prefetching_multi_try_mh"
    stub = types.ModuleType(name)
    stub.subtree_prefetching_multi_try_sampling = None
    monkeypatch.setitem(sys.modules, name, stub)
    return _load_module(
        "_test_vllm_subtree_multitry_runner",
        PACKAGE_DIR / "runners/vllm/run_subtree_prefetching_multi_try_mh.py",
    )


def _subtree_args(**overrides):
    values = dict(
        algorithm="subtree_prefetching_multi_try_mh", max_samples=None,
        prefetch_budget=15, rank_fn="longest_path_first", print_tree=False,
        stop_on_eos=True,
    )
    values.update(overrides)
    return _args(**values)


class _Stats:
    def to_json(self, **kwargs):
        return {
            "total_nfe": 7, "total_workload": 15, "total_acceptances": 2,
            "total_walked_steps": 3, "generation_calls": 2,
            "scoring_calls": 5, "scoring_requests": 11,
            "batch_sizes": [5, 10], "transitions": [{"accepted": True}],
        }


def _wrapper():
    wrapper = _Wrapper()
    wrapper.tokenizer.encode = lambda prompt: [ord(prompt[-1])]
    wrapper.tokenizer.decode = lambda tokens, **kwargs: " ".join(map(str, tokens))
    return wrapper


@pytest.mark.parametrize("max_samples,expected", [(None, 3), (1, 1), (2, 2)])
def test_every_prompt_and_limit_keep_stats_aligned(runner, monkeypatch, max_samples, expected):
    wrapper = _wrapper()
    monkeypatch.setattr(runner.driver, "vLLM_Wrapper", lambda **kwargs: wrapper)
    calls = []

    def sample(model, prompt_ids, params, **kwargs):
        assert model is wrapper
        calls.append((prompt_ids, params, kwargs))
        return prompt_ids, _Stats()

    monkeypatch.setattr(runner, "subtree_prefetching_multi_try_sampling", sample)
    args = _subtree_args(max_samples=max_samples)
    results, resources = runner.run_subtree_multi_try_sampling(
        _Benchmark(), "fake-model", args, None, runner.validate_args(args),
    )
    assert wrapper.closed
    assert resources == {"enabled": False}
    assert len(results) == len(calls) == expected
    assert [row[0] for row in calls] == [[48], [49], [50]][:expected]
    assert [row[1].seed for row in calls] == [11, 12, 13][:expected]
    assert [row["idx"] for row in results] == list(range(expected))
    assert all(row["method"] == "subtree_multi_try_mh" for row in results)
    assert all(row["total_nfe"] == 7 and row["scoring_requests"] == 11 for row in results)
    assert all(row["prefetch_budget_units"] == "suffix_requests" for row in results)
    assert all(json.loads(row["prefetch_stats"])["transitions"] for row in results)
    assert all(row[2]["prefetch_budget"] == 15 and row[2]["num_tries"] == 5 for row in calls)


def test_failure_closes_shared_engine_and_probe(runner, monkeypatch):
    wrapper = _wrapper()
    monkeypatch.setattr(runner.driver, "vLLM_Wrapper", lambda **kwargs: wrapper)

    def fail(*args, **kwargs):
        raise RuntimeError("sampling failed")

    monkeypatch.setattr(runner, "subtree_prefetching_multi_try_sampling", fail)
    with pytest.raises(RuntimeError, match="sampling failed"):
        runner.run_subtree_multi_try_sampling(_Benchmark(), "fake-model", _subtree_args(), None, 0.25)
    assert wrapper.closed


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"temperature_schedule_type": "linear"}, "const"),
        ({"cut_dist_type": "entropy"}, "uniform"),
        ({"prefetch_budget": 4}, "num_tries"),
        ({"prefetch_budget": True}, "num_tries"),
        ({"rank_fn": "unknown"}, "rank_fn"),
        ({"stop_on_eos": "false"}, "boolean"),
    ],
)
def test_runner_validates_before_engine_construction(runner, overrides, match):
    with pytest.raises(ValueError, match=match):
        runner.validate_args(_subtree_args(**overrides))


def test_main_writes_matching_csv_stats_and_resources(runner, monkeypatch, tmp_path):
    wrapper = _wrapper()
    benchmark = _Benchmark()
    benchmark.uses_max_model_len = False
    monkeypatch.setattr(runner.driver, "vLLM_Wrapper", lambda **kwargs: wrapper)
    monkeypatch.setattr(runner.driver, "build_benchmark", lambda *args: benchmark)
    monkeypatch.setattr(runner.driver, "set_random_seed", lambda seed: None)
    monkeypatch.setattr(runner.driver, "MODEL_MAP", {"fake": "fake/model"})
    monkeypatch.setattr(
        runner, "subtree_prefetching_multi_try_sampling",
        lambda model, prompt_ids, params, **kwargs: (prompt_ids, _Stats()),
    )
    runner.main([
        "--dataset", "math500", "--model_str", "fake", "--save_str", str(tmp_path),
        "--run_name", "subtree-run", "--override", "num_tries=3", "prefetch_budget=12",
        "num_blocks=4", "max_new_tokens=32", "max_samples=3", "resource_probe=true",
        "proposal_temperatures=[0.25,0.5]",
    ])
    assert wrapper.closed
    assert (tmp_path / "subtree-run.csv").exists()
    samples = json.loads((tmp_path / "subtree-run.stats.json").read_text())
    assert len(samples) == 3
    assert all(row["method"] == "subtree_multi_try_mh" for row in samples)
    assert [row["idx"] for row in samples] == [0, 1, 2]
    metadata = json.loads((tmp_path / "subtree-run.resources.json").read_text())
    assert metadata["method"] == "subtree_multi_try_mh"
    assert metadata["num_tries"] == 3
    assert metadata["prefetch_budget"] == 12
    assert metadata["prefetch_bundle_budget"] == 4
    assert metadata["proposal_temperatures"] == [0.25, 0.5]
    assert metadata["enabled"] is True
