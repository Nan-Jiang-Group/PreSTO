"""CPU benchmark/CLI checks for the HF subtree Multi-Try runner.

Run with:
    /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/.venv/bin/python -m pytest \
      /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/tests/test_hf_subtree_multi_try_runner.py
"""

import json
from types import SimpleNamespace

import pytest

from power_sharpening.runners.hf import run_subtree_prefetching_multi_try_mh as runner


class _Tokenizer:
    eos_token_id = 0

    def encode(self, text):
        return [1, 2, 3]

    def decode(self, ids, skip_special_tokens):
        assert ids == [4, 0]
        return "sampled answer"


class _Benchmark:
    name = "fake"
    dataset_loader = [[1, 2], [3]]

    def __init__(self):
        self.offsets = []
        self.graded = []

    def should_use_chat_template(self, model):
        return False

    def get_question_and_answer(self, batch, tokenizer):
        self.offsets.append(self.seen)
        return [str(item) for item in batch], [
            {"prompt": f"prompt-{item}", "answer": str(item), "choices": [item]}
            for item in batch
        ]

    def evaluate_completions(self, problems, completions):
        self.graded.append((problems[0], completions[0]))
        return [{"prediction": completions[0], "is_correct": True}]


def _args(**overrides):
    values = dict(
        alpha=4.0, temperature=-1, mcmc_steps=2, num_blocks=4,
        max_new_tokens=16, num_tries=3, proposal_temperatures=[0.25, 1.0],
        temperature_schedule_type="const", cut_dist_type="uniform",
        batch_size=2, max_samples=None, dataset="fake", verbose=False,
        prefetch_budget=12, rank_fn="longest_path_first", stop_on_eos=True,
        print_tree=False, seed=17,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_runner_grades_each_prompt_and_preserves_prefetch_stats(monkeypatch):
    benchmark = _Benchmark()
    calls = []
    model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=14))
    monkeypatch.setattr(
        runner.baseline, "hf_load_model_and_tokenizer",
        lambda model_str: ("cpu", _Tokenizer(), model),
    )

    def sample(wrapper, prefix, **kwargs):
        calls.append((prefix, kwargs))
        stats = SimpleNamespace(
            final_base_logprobs=[-1.0, -2.0], total_acceptances=2,
            total_walked_steps=4, total_workload=18, total_nfe=3,
            to_json=lambda **options: {
                "batch_sizes": [9, 9], "generation_calls": 3,
                "transitions": [{"accepted": True}],
            },
        )
        return prefix + [4, 0], stats

    monkeypatch.setattr(runner, "subtree_prefetching_multi_try_sampling", sample)
    results, stats = runner.run_subtree_multi_try("fake-model", benchmark, _args(), 0.25)
    assert len(calls) == len(results) == len(stats) == 3
    assert benchmark.offsets == [0, 2]
    assert [problem["choices"] for problem, _ in benchmark.graded] == [[1], [2], [3]]
    for index, (prefix, kwargs) in enumerate(calls):
        assert prefix == [1, 2, 3]
        assert kwargs["max_new_tokens"] == 8
        assert kwargs["max_model_len"] == 14
        assert kwargs["seed"] == 17 + index
        assert kwargs["prefetch_budget"] == 12
        assert kwargs["num_tries"] == 3
        assert kwargs["proposal_temperatures"] == [0.25, 1.0]
    assert results[0]["method"] == "subtree_prefetching_multi_try_mh"
    assert results[0]["logprob_sum"] == -3.0
    assert results[0]["log_likelihood"] == -1.5
    assert results[0]["mean_token_probability"] == pytest.approx(0.2231301601)
    assert "confidence" not in results[0]
    assert results[0]["acceptance_ratio"] == 0.5
    assert results[0]["total_nfe"] == 3
    assert stats[0]["generation_calls"] == 3
    assert stats[0]["batch_sizes"] == [9, 9]
    assert stats[0]["prefetch_budget"] == 12
    assert "average_entropy" not in stats[0]


@pytest.mark.parametrize("overrides, message", [
    ({"prefetch_budget": 2}, "complete bundle"),
    ({"prefetch_budget": True}, "prefetch_budget"),
    ({"rank_fn": "unknown"}, "rank_fn"),
    ({"cut_dist_type": "entropy"}, "uniform"),
    ({"temperature_schedule_type": "linear", "proposal_temperatures": None}, "const"),
    ({"stop_on_eos": "true"}, "boolean"),
    ({"print_tree": 1}, "boolean"),
])
def test_invalid_prefetch_config_is_rejected(overrides, message):
    with pytest.raises(SystemExit, match=message):
        runner._validate_and_resolve(_args(**overrides))


def test_cli_resolves_algorithm_and_writes_matching_stats_csv(monkeypatch, tmp_path):
    benchmark = _Benchmark()
    captured = {}
    monkeypatch.setattr(runner, "build_benchmark", lambda *args: benchmark)
    monkeypatch.setattr(runner, "set_random_seed", lambda seed: None)

    def run(model, selected, args, temperature):
        captured.update(vars(args))
        return [{"num_tries": args.num_tries}], [{"prefetch_budget": args.prefetch_budget}]

    monkeypatch.setattr(runner, "run_subtree_multi_try", run)
    runner.main([
        "--dataset", "math500", "--algorithm", "subtree_prefetching_multi_try_mh",
        "--model_str", "qwen-math-small", "--save_str", str(tmp_path),
        "--run_name", "prefetched", "--override", "num_tries=3", "prefetch_budget=12",
    ])
    assert captured["num_tries"] == 3
    assert captured["prefetch_budget"] == 12
    assert (tmp_path / "prefetched.csv").read_text() == "num_tries\n3\n"
    assert json.loads((tmp_path / "prefetched.stats.json").read_text()) == [{"prefetch_budget": 12}]
