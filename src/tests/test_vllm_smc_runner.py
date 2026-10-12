"""CPU checks for the patched-vLLM SMC runner configuration and cleanup.

Run with:
    uv run --project /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src \
      pytest /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/tests/test_vllm_smc_runner.py
"""

import importlib.util
import logging
from pathlib import Path
import sys
import types
from types import SimpleNamespace

import pytest

from power_sharpening.config import load_run_config


PACKAGE_DIR = Path(__file__).resolve().parents[1] / "power_sharpening"


@pytest.fixture
def runner(monkeypatch):
    """Import the real runner with its GPU-only dependencies replaced."""
    stubs = {
        "power_sharpening.backends.vllm.engine_patch": {
            "SamplingParams": SimpleNamespace,
        },
        "power_sharpening.backends.vllm.wrapper": {
            "vLLM_Wrapper": object,
        },
        "power_sharpening.backends.vllm.samplers": {
            "smc_power_sampler": None,
        },
    }
    for name, attributes in stubs.items():
        stub = types.ModuleType(name)
        stub.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, stub)

    spec = importlib.util.spec_from_file_location(
        "_test_vllm_smc_runner",
        PACKAGE_DIR / "runners" / "vllm" / "run_power_smc.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.logger = logging.getLogger("_test_smc")
    monkeypatch.setattr(module, "ResourceProbe", _Probe)
    _Probe.instances.clear()
    return module


class _Probe:
    instances = []

    def __init__(self, enabled):
        self.enabled = enabled
        self.marks = 0
        self.closed = False
        _Probe.instances.append(self)

    def start(self):
        pass

    def attach(self, llm):
        self.llm = llm

    def mark(self):
        self.marks += 1
        return {"peak_kv_cache_occupancy": 0.1 * self.marks}

    def close(self):
        self.closed = True

    def summary(self):
        assert self.closed
        return {"enabled": self.enabled, "peak_kv_cache_occupancy": 0.1 * self.marks}

    def summary_line(self):
        return "mock resources"


class _Wrapper:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.llm = object()
        self.tokenizer = SimpleNamespace(pad_token_id=None, eos_token_id=0)
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.closed = True


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
        return [problem["prompt"] for problem in batch], list(batch)

    def batch_to_problems(self, batch):
        return list(batch)

    def evaluate_completions(self, problems, completions):
        assert len(problems) == len(completions)
        self.graded_completions.extend(completions)
        return [
            {
                "completion": completion,
                "prediction": completion,
                "is_correct": True,
                "justification": "fake",
            }
            for completion in completions
        ]


def _args(prefix_cache, **overrides):
    config = load_run_config(PACKAGE_DIR / "config", "math500", "smc")
    config.update(prefix_cache=prefix_cache, min_new_tokens=73, verbose=False, resource_probe=True)
    config.update(overrides)
    return SimpleNamespace(**config)


@pytest.mark.parametrize("prefix_cache", [True, False])
def test_runner_propagates_smc_settings_and_closes_wrapper(
    runner, monkeypatch, caplog, prefix_cache
):
    wrappers = []
    sampler_calls = []

    def make_wrapper(**kwargs):
        wrapper = _Wrapper(**kwargs)
        wrappers.append(wrapper)
        return wrapper

    def sample(wrapper, prompts, **kwargs):
        assert wrapper is wrappers[0]
        sampler_calls.append(kwargs)
        return [f"completion-{prompt}" for prompt in prompts]

    monkeypatch.setattr(runner, "vLLM_Wrapper", make_wrapper)
    monkeypatch.setattr(runner, "smc_power_sampler", sample)
    benchmark = _Benchmark()

    with caplog.at_level(logging.INFO, logger="_test_smc"):
        results, summary = runner.run_power_smc(
            benchmark,
            "fake-model",
            _args(prefix_cache),
            max_model_len=None,
            init_temperature=0.25,
        )

    assert len(wrappers) == 1
    assert wrappers[0].kwargs["engine_type"] == "custom"
    assert wrappers[0].kwargs["enable_prefix_caching"] is prefix_cache
    assert wrappers[0].closed is True
    assert len(sampler_calls) == 2
    assert all(call["min_new_tokens"] == 73 for call in sampler_calls)
    assert len(results) == benchmark.size
    assert all(row["prefix_cache"] is prefix_cache for row in results)
    assert all(row["min_new_tokens"] == 73 for row in results)
    assert benchmark.graded_completions == [
        "completion-p0", "completion-p1", "completion-p2"
    ]
    # One timing line and one resource window per batch, stamped onto that batch's rows.
    took = [r.getMessage() for r in caplog.records if " took " in r.getMessage()]
    assert len(took) == 2
    assert took[1].startswith("smc_power_sampler took ") and took[1].endswith(" seconds for 2-th prompts")
    assert [row["peak_kv_cache_occupancy"] for row in results] == pytest.approx([0.1, 0.1, 0.2])
    assert summary == {"enabled": True, "peak_kv_cache_occupancy": pytest.approx(0.2)}
    assert _Probe.instances[0].llm is wrappers[0].llm
    assert wrappers[0].kwargs["disable_log_stats"] is False


def test_probe_off_keeps_engine_stats_default(runner, monkeypatch):
    wrappers = []

    def make_wrapper(**kwargs):
        wrappers.append(_Wrapper(**kwargs))
        return wrappers[-1]

    monkeypatch.setattr(runner, "vLLM_Wrapper", make_wrapper)
    monkeypatch.setattr(runner, "smc_power_sampler", lambda wrapper, prompts, **kwargs: list(prompts))
    runner.run_power_smc(
        _Benchmark(), "fake-model", _args(True, resource_probe=False), max_model_len=None, init_temperature=0.25
    )
    assert "disable_log_stats" not in wrappers[0].kwargs
    assert _Probe.instances[0].enabled is False


def test_runner_closes_wrapper_when_sampling_fails(runner, monkeypatch):
    wrapper = _Wrapper()
    monkeypatch.setattr(runner, "vLLM_Wrapper", lambda **kwargs: wrapper)

    def fail_sampling(*args, **kwargs):
        raise RuntimeError("sampling failed")

    monkeypatch.setattr(runner, "smc_power_sampler", fail_sampling)

    with pytest.raises(RuntimeError, match="sampling failed"):
        runner.run_power_smc(
            _Benchmark(),
            "fake-model",
            _args(prefix_cache=True),
            max_model_len=None,
            init_temperature=0.25,
        )

    assert wrapper.closed is True
    assert _Probe.instances[0].closed is True


def test_smc_config_enables_prefix_cache_and_suppresses_early_eos():
    config = load_run_config(PACKAGE_DIR / "config", "math500", "smc")

    assert config["min_new_tokens"] == 100
    assert config["prefix_cache"] is True
