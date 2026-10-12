"""CPU checks for the vLLM MultiTryMH runner and its shared engine lifecycle.

Run with:
    uv run --project /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src \
      pytest /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/tests/test_vllm_multi_try_runner.py
"""

import importlib
import importlib.util
import json
from pathlib import Path
import sys
import types
from types import SimpleNamespace

import pytest

from power_sharpening.common.temp_scheduler import ConstantScheduler, LinearScheduler
from power_sharpening.config import load_run_config


PACKAGE_DIR = Path(__file__).resolve().parents[1] / "power_sharpening"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def runner(monkeypatch):
    """Use the real benchmark loop without importing GPU-only backends."""
    stubs = {
        "power_sharpening.backends.vllm.engine_patch": {
            "SamplingParams": SimpleNamespace,
        },
        "power_sharpening.backends.vllm.wrapper": {"vLLM_Wrapper": object},
        "power_sharpening.backends.vllm.samplers": {
            "mcmc_power_sampler": None,
            "mcmc_power_sampler_entropycut": None,
        },
        "power_sharpening.backends.vllm.samplers.multi_try_mh": {
            "multi_try_mcmc_power_sampler": None,
        },
    }
    for name, attributes in stubs.items():
        stub = types.ModuleType(name)
        stub.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, stub)

    driver = _load_module(
        "_test_multi_try_shared_driver",
        PACKAGE_DIR / "runners/vllm/run_power_mh.py",
    )
    package = importlib.import_module("power_sharpening.runners.vllm")
    monkeypatch.setattr(package, "run_power_mh", driver, raising=False)
    module = _load_module(
        "_test_vllm_multi_try_runner",
        PACKAGE_DIR / "runners/vllm/run_multi_try_mh.py",
    )
    monkeypatch.setattr(driver, "ResourceProbe", _Probe)
    return module


class _Probe:
    def __init__(self, enabled):
        self.enabled = enabled

    def start(self):
        pass

    def attach(self, engine):
        self.engine = engine

    def mark(self):
        return {}

    def close(self):
        self.closed = True

    def summary(self):
        assert self.closed
        return {"enabled": self.enabled}

    def summary_line(self):
        return "mock resources"


class _Wrapper:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.tokenizer = SimpleNamespace(
            encode=lambda prompt, **kwargs: [0] * 5,
        )
        self.llm = object()
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.closed = True


class _Benchmark:
    name = "fake"
    size = 3
    uses_max_model_len = True

    def __init__(self):
        self.dataset_loader = [[{"prompt": "p0"}, {"prompt": "p1"}], [{"prompt": "p2"}]]
        self.seen = 0
        self.use_chat_template = False

    def should_use_chat_template(self, model_name):
        self.use_chat_template = True
        return True

    def get_question_and_answer(self, batch, tokenizer):
        del tokenizer
        prefix = "chat:" if self.use_chat_template else ""
        return [prefix + problem["prompt"] for problem in batch], list(batch)

    def batch_to_problems(self, batch):
        return list(batch)

    def evaluate_completions(self, problems, completions):
        assert len(problems) == len(completions)
        return [
            {
                "completion": completion,
                "prediction": completion,
                "is_correct": True,
                "justification": "mock",
            }
            for completion in completions
        ]


def _args(**overrides):
    args = dict(
        algorithm="multi_try", seed=11, batch_size=2, alpha=4.0,
        temperature=-1, temperature_schedule_type="const", mcmc_steps=3,
        num_blocks=4, max_new_tokens=32, max_samples=2, num_tries=5,
        scoring_batch_size=7, proposal_temperatures=None, cut_dist_type="uniform",
        resource_probe=False, prefix_cache=True, verbose=False, dtype="auto",
    )
    args.update(overrides)
    return SimpleNamespace(**args)


@pytest.mark.parametrize(
    ("temperatures", "schedule", "scheduler_class"),
    [(None, "linear", LinearScheduler), ([0.25, 0.5, 0.5], "const", ConstantScheduler)],
)
def test_sampler_settings_context_alignment_and_cleanup(
    runner, monkeypatch, temperatures, schedule, scheduler_class,
):
    wrapper = _Wrapper()
    monkeypatch.setattr(runner.driver, "vLLM_Wrapper", lambda **kwargs: wrapper)
    calls = []

    def sample(model, prompts, **kwargs):
        assert model is wrapper
        calls.append(kwargs)
        return [f"completion-{prompt}" for prompt in prompts]

    monkeypatch.setattr(runner, "multi_try_mcmc_power_sampler", sample)
    args = _args(proposal_temperatures=temperatures, temperature_schedule_type=schedule)
    init_temperature = runner.validate_args(args)
    # The shared context helper caps 32 tokens to 30, then the runner aligns to 28.
    results, resources = runner.run_multi_try_sampling(
        _Benchmark(), "fake-model", args, max_model_len=51,
        init_temperature=init_temperature,
    )

    assert wrapper.closed
    assert resources == {"enabled": False}
    assert len(calls) == 1
    assert calls[0]["max_new_tokens"] == 28
    assert calls[0]["num_of_blocks"] == 4
    assert calls[0]["num_tries"] == 5
    assert calls[0]["mcmc_steps"] == 3
    assert calls[0]["scoring_batch_size"] == 7
    assert calls[0]["proposal_temperatures"] == temperatures
    assert isinstance(calls[0]["temperature_scheduler"], scheduler_class)
    assert calls[0]["sampling_params"].temperature == 0.25
    assert calls[0]["sampling_params"].alpha == 4.0
    assert len(results) == 2
    assert all(row["method"] == "multi_try_mh" and row["num_tries"] == 5 for row in results)


@pytest.mark.parametrize("context_limit", [23, None])
def test_sampling_failures_close_engine(runner, monkeypatch, context_limit):
    wrapper = _Wrapper()
    monkeypatch.setattr(runner.driver, "vLLM_Wrapper", lambda **kwargs: wrapper)

    def fail(*args, **kwargs):
        raise RuntimeError("sampling failed")

    monkeypatch.setattr(runner, "multi_try_mcmc_power_sampler", fail)
    expected = ValueError if context_limit else RuntimeError
    with pytest.raises(expected):
        runner.run_multi_try_sampling(
            _Benchmark(), "fake-model", _args(), context_limit, 0.25,
        )
    assert wrapper.closed


def test_zero_mcmc_steps_skip_scalar_annealing(runner, monkeypatch):
    wrapper = _Wrapper()
    monkeypatch.setattr(runner.driver, "vLLM_Wrapper", lambda **kwargs: wrapper)

    def sample(model, prompts, **kwargs):
        assert kwargs["temperature_scheduler"] is None
        assert kwargs["mcmc_steps"] == 0
        return ["completion"] * len(prompts)

    monkeypatch.setattr(runner, "multi_try_mcmc_power_sampler", sample)
    runner.run_multi_try_sampling(
        _Benchmark(), "fake-model",
        _args(mcmc_steps=0, temperature_schedule_type="linear"), None, 0.25,
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"num_tries": True}, "num_tries"),
        ({"scoring_batch_size": 0}, "scoring_batch_size"),
        ({"max_new_tokens": 31}, "divisible"),
        ({"max_samples": 0}, "max_samples"),
        ({"max_model_len": 0}, "max_model_len"),
        ({"alpha": float("inf")}, "alpha"),
        ({"temperature": 0}, "temperature"),
        ({"cut_dist_type": "entropy"}, "uniform"),
        ({"temperature_schedule_type": "cyclic"}, "min_temp and max_temp"),
        ({"temperature_schedule_type": "cyclic", "mcmc_steps": 0}, "min_temp and max_temp"),
        ({"proposal_temperatures": []}, "nonempty list"),
        ({"proposal_temperatures": [0.2, float("nan")]}, "positive finite"),
        ({"proposal_temperatures": [0.2], "temperature_schedule_type": "linear"}, "const"),
    ],
)
def test_validation_rejects_invalid_multi_try_inputs(runner, overrides, message):
    with pytest.raises(ValueError, match=message):
        runner.validate_args(_args(**overrides))


def test_main_uses_multi_try_config_and_matching_artifact_names(runner, monkeypatch, tmp_path):
    wrapper = _Wrapper()
    benchmark = _Benchmark()
    benchmark.uses_max_model_len = False
    calls = []
    monkeypatch.setattr(runner.driver, "vLLM_Wrapper", lambda **kwargs: wrapper)
    monkeypatch.setattr(runner.driver, "build_benchmark", lambda *args: benchmark)
    monkeypatch.setattr(runner.driver, "set_random_seed", lambda seed: None)
    monkeypatch.setattr(runner.driver, "MODEL_MAP", {"fake": "fake/model"})

    def resolve_context(model, requested):
        assert model == "fake/model"
        assert requested == 60
        return 60, 100

    monkeypatch.setattr(runner.driver, "resolve_max_model_len", resolve_context)

    def sample(model, prompts, **kwargs):
        calls.append((prompts, kwargs))
        return ["completion"] * len(prompts)

    monkeypatch.setattr(runner, "multi_try_mcmc_power_sampler", sample)
    runner.main([
        "--dataset", "math500", "--algorithm", "multi_try", "--model_str", "fake",
        "--save_str", str(tmp_path), "--run_name", "chosen-run", "--no-resource_probe",
        "--override", "num_tries=3", "num_blocks=4", "max_new_tokens=32",
        "max_samples=1", "max_model_len=60", "proposal_temperatures=[0.25,0.5]",
        "resource_probe=true",
    ])

    assert wrapper.closed
    assert calls[0][0] == ["chat:p0"]
    assert calls[0][1]["num_tries"] == 3
    assert calls[0][1]["proposal_temperatures"] == [0.25, 0.5]
    assert (tmp_path / "chosen-run.csv").exists()
    metadata = json.loads((tmp_path / "chosen-run.resources.json").read_text())
    assert metadata["run"] == "chosen-run"
    assert metadata["algorithm"] == "multi_try"
    assert metadata["num_tries"] == 3
    assert metadata["proposal_temperatures"] == [0.25, 0.5]
    assert metadata["max_model_len"] == 60
    assert metadata["enabled"] is True


def test_multi_try_defaults_and_resource_flag(runner):
    config = load_run_config(PACKAGE_DIR / "config", "math500", "multi_try")
    assert config["num_tries"] == 4
    assert config["proposal_temperatures"] == [0.25, 0.5, 1.0]
    assert config["scoring_batch_size"] == 128
    parsed = runner.build_parser().parse_args(["--dataset", "math500", "--no-resource_probe"])
    assert parsed.algorithm == "multi_try"
    assert parsed.resource_probe is False
