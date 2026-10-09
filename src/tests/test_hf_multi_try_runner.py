"""CPU checks for the HF Multi-Try MH runner without loading a checkpoint.

Run with:
    /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/.venv/bin/python -m pytest \
      /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/tests/test_hf_multi_try_runner.py
"""

import json
from types import SimpleNamespace

import pytest

from power_sharpening.runners.hf import run_multi_try_mh as runner


class _Tokenizer:
    eos_token_id = 0

    def __init__(self):
        self.encoded = []

    def encode(self, text, add_special_tokens=True):
        self.encoded.append((text, add_special_tokens))
        return [int(text[-1]) + 1] * 3

    def decode(self, ids, skip_special_tokens):
        assert skip_special_tokens is True
        assert len(ids) == 2
        return f"completion-{ids[0]}"


class _Benchmark:
    name = "fake"
    use_chat_template = True

    def __init__(self):
        self.dataset_loader = [[0, 1], [2, 3]]
        self.graded = []
        self.formatted_offsets = []
        self.chat_model = None

    def should_use_chat_template(self, model):
        self.chat_model = model
        return self.use_chat_template

    def get_question_and_answer(self, batch, tokenizer):
        self.formatted_offsets.append(self.seen)
        # Grading data is reconstructed by the benchmark, not the runner.
        return [f"chat-{item}" for item in batch], [
            {"question": f"question-{item}", "correct_letter": "A", "choices": [item]}
            for item in batch
        ]

    def batch_to_problems(self, batch):
        raise AssertionError("Use reconstructed grading problems returned by the benchmark.")

    def evaluate_completions(self, problems, completions):
        self.graded.append((problems[0], completions[0]))
        return [{"prediction": completions[0], "is_correct": True, "completion": completions[0]}]


def _args(**overrides):
    args = dict(
        alpha=4.0, temperature=-1.0, mcmc_steps=2, num_blocks=4,
        max_new_tokens=16, num_tries=3, proposal_temperatures=[0.25, 1.0],
        temperature_schedule_type="const", cut_dist_type="uniform",
        batch_size=2, max_samples=None, dataset="fake", verbose=False,
    )
    args.update(overrides)
    return SimpleNamespace(**args)


def _mock_sampler(monkeypatch):
    tokenizer = _Tokenizer()
    calls = []
    model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=14))
    monkeypatch.setattr(runner, "hf_load_model_and_tokenizer", lambda model_str: ("cpu", tokenizer, model))

    def sample(wrapper, prefix, **kwargs):
        assert wrapper.alpha == 4.0
        assert wrapper.init_temperature == 0.25
        calls.append((list(prefix), kwargs))
        return list(prefix) + [100 + len(calls), 0], [-1.0, -2.0], [-4.0, -8.0], 0.5

    monkeypatch.setattr(runner, "multi_try_mcmc_power_sampler", sample)
    return tokenizer, calls


def test_runner_processes_every_prompt_and_preserves_mixture_and_grading(monkeypatch):
    tokenizer, calls = _mock_sampler(monkeypatch)
    benchmark = _Benchmark()
    results, stats = runner.run_multi_try("fake-model", benchmark, _args(), 0.25)

    assert [prefix for prefix, _ in calls] == [[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]]
    assert tokenizer.encoded == [(f"chat-{idx}", True) for idx in range(4)]
    assert benchmark.formatted_offsets == [0, 2]
    assert [problem["choices"] for problem, _ in benchmark.graded] == [[0], [1], [2], [3]]
    assert [completion for _, completion in benchmark.graded] == [f"completion-{idx}" for idx in range(101, 105)]
    for _, kwargs in calls:
        assert kwargs == {
            "mcmc_steps": 2, "max_new_tokens": 8, "num_of_blocks": 4,
            "num_tries": 3, "temperature_schedule_type": "const",
            "proposal_temperatures": [0.25, 1.0], "verbose": False,
        }
    assert len(results) == len(stats) == 4
    assert [row["idx"] for row in stats] == list(range(4))
    assert results[0]["max_new_tokens"] == 8
    assert results[0]["requested_max_new_tokens"] == 16
    assert results[0]["num_response_tokens"] == 2
    assert results[0]["log_likelihood"] == -3.0
    assert results[0]["num_tries"] == stats[0]["num_tries"] == 3
    assert stats[0]["proposal_temperatures"] == [0.25, 1.0]
    assert stats[0]["acceptance_rate"] == 0.5


@pytest.mark.parametrize("max_samples", [1, 3])
def test_runner_honors_max_samples_within_and_across_batches(monkeypatch, max_samples):
    _, calls = _mock_sampler(monkeypatch)
    results, stats = runner.run_multi_try("fake-model", _Benchmark(), _args(max_samples=max_samples), 0.25)
    assert len(calls) == len(results) == len(stats) == max_samples


def test_context_budget_fails_before_sampling_when_whole_block_cannot_fit(monkeypatch):
    _, calls = _mock_sampler(monkeypatch)
    with pytest.raises(RuntimeError, match="No positive"):
        runner.run_multi_try("fake-model", _Benchmark(), _args(max_model_len=6), 0.25)
    assert calls == []


def test_context_window_uses_smaller_override():
    model = SimpleNamespace(config=SimpleNamespace(n_positions=32))
    assert runner._model_context_length(model, requested=64) == 32
    assert runner._model_context_length(model, requested=16) == 16


def test_raw_prompts_retain_default_special_tokens_despite_chat_flag(monkeypatch):
    tokenizer, _ = _mock_sampler(monkeypatch)

    class _RawBenchmark(_Benchmark):
        use_chat_template = True

        def get_question_and_answer(self, batch, tokenizer):
            prompts, problems = super().get_question_and_answer(batch, tokenizer)
            return [prompt.replace("chat-", "raw-") for prompt in prompts], problems

    runner.run_multi_try("fake-model", _RawBenchmark(), _args(max_samples=1), 0.25)
    assert tokenizer.encoded == [("raw-0", True)]


@pytest.mark.parametrize("overrides, message", [
    ({"alpha": True}, "alpha"),
    ({"alpha": "four"}, "alpha"),
    ({"temperature": False}, "temperature"),
    ({"temperature": "cold"}, "temperature"),
    ({"num_tries": 0}, "num_tries"),
    ({"max_new_tokens": 15}, "divisible"),
    ({"max_samples": 0}, "max_samples"),
    ({"cut_dist_type": "entropy"}, "uniform"),
    ({"temperature_schedule_type": "cyclic", "proposal_temperatures": None}, "unsupported"),
    ({"proposal_temperatures": []}, "nonempty"),
    ({"proposal_temperatures": [0.0, 1.0]}, "finite positive"),
    ({"temperature_schedule_type": "linear"}, "requires"),
])
def test_validation_rejects_unsupported_sampling_configuration(overrides, message):
    with pytest.raises(SystemExit, match=message):
        runner._validate_and_resolve(_args(**overrides))


def test_cli_resolves_yaml_overrides_and_pairs_output_names(monkeypatch, tmp_path):
    benchmark = _Benchmark()
    observed = {}
    monkeypatch.setattr(runner, "build_benchmark", lambda *args: benchmark)
    monkeypatch.setattr(runner, "set_random_seed", lambda seed: None)

    def run(model_str, selected_benchmark, args, proposal_temperature):
        observed.update(vars(args))
        observed["proposal_temperature"] = proposal_temperature
        assert selected_benchmark is benchmark
        return [{"num_tries": args.num_tries}], [{"proposal_temperatures": args.proposal_temperatures}]

    monkeypatch.setattr(runner, "run_multi_try", run)
    runner.main([
        "--dataset", "math500", "--algorithm", "multi_try",
        "--model_str", "qwen-math-small", "--save_str", str(tmp_path),
        "--run_name", "mixture-comparison", "--override", "num_tries=5",
        "proposal_temperatures=[0.25,0.5,1.0]", "max_samples=3",
    ])

    assert observed["num_tries"] == 5
    assert observed["max_samples"] == 3
    assert observed["proposal_temperature"] == 1.0 / observed["alpha"]
    assert benchmark.chat_model == "qwen-math-small"
    assert (tmp_path / "mixture-comparison.csv").read_text() == "num_tries\n5\n"
    assert json.loads((tmp_path / "mixture-comparison.stats.json").read_text()) == [
        {"proposal_temperatures": [0.25, 0.5, 1.0]},
    ]


@pytest.mark.parametrize("override, message", [
    ("cut_dist_type=entropy", "uniform"),
    ("temperature_schedule_type=cyclic", "unsupported"),
])
def test_invalid_cli_settings_fail_before_model_loading(monkeypatch, tmp_path, override, message):
    def load(*args, **kwargs):
        raise AssertionError("Validation should run before loading the model.")

    monkeypatch.setattr(runner, "hf_load_model_and_tokenizer", load)
    with pytest.raises(SystemExit, match=message):
        runner.main([
            "--dataset", "math500", "--save_str", str(tmp_path),
            "--override", override,
        ])
