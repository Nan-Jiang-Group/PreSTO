"""CPU checks for the vLLM low-temperature / best-of-N runner: max_samples, timing lines, and resource columns.

Run with:
    uv run --project /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src \
      pytest /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/tests/test_vllm_low_temp_runner.py
"""

import importlib.util
import logging
from pathlib import Path
import sys
import types
from types import SimpleNamespace

import pytest


PACKAGE_DIR = Path(__file__).resolve().parents[1] / "power_sharpening"


@pytest.fixture
def runner(monkeypatch):
    """Load run_low_temp.py without importing GPU-only backends."""
    stubs = {
        "power_sharpening.backends.vllm.engine_patch": {"SamplingParams": SimpleNamespace},
        "power_sharpening.backends.vllm.wrapper": {"vLLM_Wrapper": object},
    }
    for name, attributes in stubs.items():
        stub = types.ModuleType(name)
        stub.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, stub)
    spec = importlib.util.spec_from_file_location(
        "_test_vllm_low_temp_runner", PACKAGE_DIR / "runners/vllm/run_low_temp.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.logger = logging.getLogger("_test_low_temp")
    monkeypatch.setattr(module, "ResourceProbe", _Probe)
    monkeypatch.setattr(module, "vLLM_Wrapper", _Wrapper)
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
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.tokenizer = object()
        self.llm = self
        self.calls = []
        _Wrapper.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def generate(self, prompts, sampling_params, use_tqdm):
        self.calls.append(list(prompts))
        n = getattr(sampling_params, "n", 1)
        return [
            SimpleNamespace(outputs=[
                SimpleNamespace(text=f"{prompt}#{i}", cumulative_logprob=float(i)) for i in range(n)
            ])
            for prompt in prompts
        ]


class _Benchmark:
    name = "fake"
    size = 5

    def __init__(self):
        self.dataset_loader = [[{"prompt": "p0"}, {"prompt": "p1"}], [{"prompt": "p2"}, {"prompt": "p3"}],
                               [{"prompt": "p4"}]]
        self.seen = 0

    def get_question_and_answer(self, batch, tokenizer):
        return [problem["prompt"] for problem in batch], list(batch)

    def batch_to_problems(self, batch):
        return list(batch)

    def evaluate_completions(self, problems, completions):
        return [
            {"completion": c, "prediction": c, "is_correct": c.endswith("#0"), "justification": "mock"}
            for c in completions
        ]


def _args(**overrides):
    args = dict(seed=3, temperature=0.25, max_new_tokens=16, prefix_cache=True, verbose=False, dtype="auto",
                gpu_memory_utilization=0.9, resource_probe=True, max_samples=3, best_of_N=[2])
    args.update(overrides)
    return SimpleNamespace(**args)


@pytest.fixture(autouse=True)
def _reset():
    _Probe.instances.clear()
    _Wrapper.instances.clear()


def test_low_temp_caps_prompts_logs_time_and_stamps_resources(runner, caplog):
    with caplog.at_level(logging.INFO, logger="_test_low_temp"):
        results, summary = runner.run_low_temperature(_Benchmark(), "fake-model", _args(), None)

    assert _Wrapper.instances[0].calls == [["p0", "p1"], ["p2"]]
    assert [row["completion"] for row in results] == ["p0#0", "p1#0", "p2#0"]
    assert [row["peak_kv_cache_occupancy"] for row in results] == pytest.approx([0.1, 0.1, 0.2])
    assert summary == {"enabled": True, "peak_kv_cache_occupancy": pytest.approx(0.2)}
    took = [r.getMessage() for r in caplog.records if " took " in r.getMessage()]
    assert len(took) == 2
    assert took[1].startswith("low_temp_sampler took ") and took[1].endswith(" seconds for 2-th prompts")
    kwargs = _Wrapper.instances[0].kwargs
    assert kwargs["gpu_memory_utilization"] == 0.9 and kwargs["disable_log_stats"] is False


def test_best_of_n_logs_time_and_keeps_best_candidate(runner, caplog):
    with caplog.at_level(logging.INFO, logger="_test_low_temp"):
        results, _ = runner.run_best_of_n_with_low_temp(_Benchmark(), "fake-model", _args(max_samples=None), None)

    assert len(results) == 5
    assert {row["completion"] for row in results} == {f"p{i}#1" for i in range(5)}
    assert all(row["best_of_n"] == 2 and "peak_kv_cache_occupancy" in row for row in results)
    took = [r.getMessage() for r in caplog.records if " took " in r.getMessage()]
    assert [m.split(" took ")[0] for m in took] == ["best_of_n_sampler"] * 3


def test_probe_off_keeps_engine_stats_default(runner):
    runner.run_low_temperature(_Benchmark(), "fake-model", _args(resource_probe=False), None)
    assert "disable_log_stats" not in _Wrapper.instances[0].kwargs
    assert _Probe.instances[0].enabled is False


@pytest.mark.parametrize("bad", [0, -1, True, 2.5])
def test_rejects_bad_max_samples(runner, bad):
    with pytest.raises(SystemExit):
        runner.run_low_temperature(_Benchmark(), "fake-model", _args(max_samples=bad), None)
