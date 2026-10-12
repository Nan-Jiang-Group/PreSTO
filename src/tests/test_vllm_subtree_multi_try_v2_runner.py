"""CPU checks for the subtree MultiTryMH v2 runner: engine-cached scoring wiring, config, and artifacts.

Run with:
    python -m pytest /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/tests/test_vllm_subtree_multi_try_v2_runner.py
"""

import importlib
import json
import sys
import types

import pytest

from test_vllm_multi_try_runner import PACKAGE_DIR, _Benchmark, _load_module, runner as baseline_runner
from test_vllm_subtree_multi_try_runner import _Stats, _subtree_args, _wrapper, runner as subtree_runner


def _draw_proposals_v2(*args, **kwargs):
    raise AssertionError("the sampler stub is replaced in each test")


@pytest.fixture
def runner(baseline_runner, subtree_runner, monkeypatch):
    package = importlib.import_module("power_sharpening.runners.vllm")
    monkeypatch.setattr(package, "run_subtree_prefetching_multi_try_mh", subtree_runner, raising=False)
    name = "power_sharpening.backends.vllm.samplers.multi_try_mh_v2"
    stub = types.ModuleType(name)
    stub._draw_proposals_v2 = _draw_proposals_v2
    stub.multi_try_mcmc_power_sampler_v2 = None
    monkeypatch.setitem(sys.modules, name, stub)
    standalone = _load_module("_test_vllm_multitry_v2_runner", PACKAGE_DIR / "runners/vllm/run_multi_try_mh_v2.py")
    monkeypatch.setattr(package, "run_multi_try_mh_v2", standalone, raising=False)
    return _load_module(
        "_test_vllm_subtree_multitry_v2_runner",
        PACKAGE_DIR / "runners/vllm/run_subtree_prefetching_multi_try_mh_v2.py",
    )


def _v2_args(**overrides):
    values = dict(algorithm="subtree_prefetching_multi_try_mh_v2", proposal_temperatures=[1.0, 0.25, 0.5, 0.25])
    values.update(overrides)
    args = _subtree_args(**values)
    del args.scoring_batch_size
    return args


def test_v2_passes_engine_scoring_and_log_z_temperatures(runner, monkeypatch):
    wrapper = _wrapper()
    wrapper_kwargs = {}

    def build_wrapper(**kwargs):
        wrapper_kwargs.update(kwargs)
        return wrapper

    monkeypatch.setattr(runner.driver, "vLLM_Wrapper", build_wrapper)
    calls = []

    def sample(model, prompt_ids, params, **kwargs):
        calls.append(kwargs)
        return prompt_ids, _Stats()

    monkeypatch.setattr(runner.subtree, "subtree_prefetching_multi_try_sampling", sample)
    args = _v2_args(max_samples=2)
    results, _ = runner.run_subtree_multi_try_v2_sampling(
        _Benchmark(), "fake-model", args, None, runner.validate_args(args),
    )
    # Repeated mixture entries are scored once by the engine, in a stable column order.
    assert wrapper_kwargs["log_z_temperatures"] == [0.25, 0.5, 1.0]
    assert wrapper_kwargs["engine_type"] == "custom"
    assert len(calls) == len(results) == 2
    assert all(call["draw_proposals"] is _draw_proposals_v2 for call in calls)
    assert all(row["method"] == "subtree_multi_try_mh_v2" for row in results)
    assert all(row["proposal_scoring"] == "engine_log_z" for row in results)
    assert all("scoring_batch_size" not in json.loads(row["prefetch_stats"]) for row in results)


def test_v1_runner_keeps_request_scoring(subtree_runner, monkeypatch):
    wrapper_kwargs = {}

    def build_wrapper(**kwargs):
        wrapper_kwargs.update(kwargs)
        return _wrapper()

    monkeypatch.setattr(subtree_runner.driver, "vLLM_Wrapper", build_wrapper)
    calls = []
    monkeypatch.setattr(
        subtree_runner, "subtree_prefetching_multi_try_sampling",
        lambda model, prompt_ids, params, **kwargs: (calls.append(kwargs), (prompt_ids, _Stats()))[1],
    )
    args = _subtree_args(max_samples=1)
    subtree_runner.run_subtree_multi_try_sampling(
        _Benchmark(), "fake-model", args, None, subtree_runner.validate_args(args),
    )
    assert "log_z_temperatures" not in wrapper_kwargs
    assert calls[0]["draw_proposals"] is None


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"proposal_temperatures": None}, "explicit proposal_temperatures"),
        ({"prefetch_budget": 4}, "num_tries"),
        ({"print_tree": "tree"}, "boolean"),
    ],
)
def test_v2_validates_before_engine_construction(runner, overrides, match):
    with pytest.raises(ValueError, match=match):
        runner.validate_args(_v2_args(**overrides))


def test_main_resolves_v2_config_and_writes_artifacts(runner, monkeypatch, tmp_path):
    wrapper = _wrapper()
    benchmark = _Benchmark()
    benchmark.uses_max_model_len = False
    monkeypatch.setattr(runner.driver, "vLLM_Wrapper", lambda **kwargs: wrapper)
    monkeypatch.setattr(runner.driver, "build_benchmark", lambda *args: benchmark)
    monkeypatch.setattr(runner.driver, "set_random_seed", lambda seed: None)
    monkeypatch.setattr(runner.driver, "MODEL_MAP", {"fake": "fake/model"})
    monkeypatch.setattr(
        runner.subtree, "subtree_prefetching_multi_try_sampling",
        lambda model, prompt_ids, params, **kwargs: (prompt_ids, _Stats()),
    )
    # No scoring_batch_size: the v2 config block omits it and the runner must not require it.
    runner.main([
        "--dataset", "math500", "--model_str", "fake", "--save_str", str(tmp_path),
        "--run_name", "subtree-v2-run", "--override", "num_tries=3", "prefetch_budget=12",
        "num_blocks=4", "max_new_tokens=32", "max_samples=3", "resource_probe=true",
    ])
    assert wrapper.closed
    assert (tmp_path / "subtree-v2-run.csv").exists()
    assert len(json.loads((tmp_path / "subtree-v2-run.stats.json").read_text())) == 3
