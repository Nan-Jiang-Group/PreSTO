"""
LiveCodeBench benchmark implementation, using the requested release's local file.

Fetch the V6-only shard with:
    python /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/data/fetch_data.py livecodebench --lcb-version release_v6

Generated solutions are executed against decoded public and private test
cases using ``power_sharpening.tasks.grader_utils.livecodebench_utils.lcb_run``.

For full evaluation with test execution, also see the official harness:
https://github.com/LiveCodeBench/LiveCodeBench
"""

import copy
import hashlib
import logging
import os
import json
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List


from torch.utils.data import DataLoader
import logging

logger = logging.getLogger("[lcb_benchmark]")
from .grader_utils.livecodebench_utils import (
    compute_metrics_from_results,
    lcb_run,
    map_to_example,
    post_process_code,
    translate_private_test_cases,
)

from .base import Benchmark


_LCB_DATA_DIR = Path(__file__).resolve().parents[3] / "data"


LCB_PROMPT_WITHOUT_STARTER_CODE = """You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the program.

Question: {problem_description}

Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.
```python
  # YOUR CODE HERE
```"""

LCB_PROMPT_WITH_STARTER_CODE = """You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the program.

Question: {problem_description}

You will use the following starter code to write the solution to the problem and enclose your code within delimiters."
```python
{entry_point}
"""


def has_code(response):
    """Extract Python code, falling back to a raw code-only response."""
    pattern = r"```(?:[a-zA-Z0-9_+-]*)[ \t]*\r?\n(.*?)```"
    matches = [
        match.strip()
        for match in re.findall(pattern, response, re.DOTALL)
        if match.strip()
    ]
    if matches:
        return matches

    # Qwen3 may emit reasoning before a raw program, and starter-code prompts already contain the opening fence. Keep
    # the program after </think> and remove any unmatched fence left at either end.
    raw = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    if "</think>" in raw:
        raw = raw.rsplit("</think>", 1)[-1].strip()
    raw = re.sub(
        r"^```(?:[a-zA-Z0-9_+-]*)[ \t]*(?:\r?\n)?", "", raw
    ).strip()
    raw = re.sub(r"```[ \t]*$", "", raw).strip()
    return [raw] if raw else []



def _lcb_collate(examples):
    """Dict-of-lists collate that preserves None and nested structures."""
    if not examples:
        return {}
    keys = examples[0].keys()
    return {k: [ex.get(k) for ex in examples] for k in keys}


class LiveCodeBenchBenchmark(Benchmark):
    """
    LiveCodeBench benchmark for competitive-programming-style code generation.
    Problems come from the explicitly selected local release file.

    After loading, each example exposes the fields produced by
    ``map_to_example`` in ``power_sharpening.tasks.grader_utils.livecodebench_utils``:
    - ``task_id``: unique problem id
    - ``prompt``: full problem statement (used as ``{problem_description}``)
    - ``entry_point``: starter code / function signature (when applicable)
    - ``is_stdin``: True for stdin/stdout problems, False for function-call style (uses starter-code prompt)
    - ``difficulty``: easy / medium / hard
    - ``test``: decoded list of test cases consumed by ``lcb_run``
    """

    # Subclasses override to restrict to a single difficulty (raw-JSONL filter).
    DIFFICULTY: str = "all"

    def __init__(
            self,
            batch_size,
            version: str = "release_v6",
            max_prompt_chars: int = 3500,
    ):
        """
        Args:
            batch_size: Number of prompts per DataLoader batch.
            version: Local release to load; only release_v6 is supported. It uses the
                V6-only file produced by data/fetch_data.py.
            max_prompt_chars: Hard cap on the formatted prompt length in chars. Long problems are middle-truncated to
                fit.
        """
        self.version = version
        self.max_prompt_chars = max_prompt_chars

        self.logger = logging.getLogger(self.__class__.__name__)
        self.dataset_loader = None
        self.load_dataset(batch_size=batch_size)
        self.seen = 0

    @property
    def name(self) -> str:
        return f"LiveCodeBench-{self.version}"

    @property
    def size(self):
        """Total number of problems."""
        return self._num_problems

    def load_dataset(self, batch_size: int) -> DataLoader:
        """Load the requested local release and record its source fingerprint."""
        if self.version not in {"release_v6"}:
            raise ValueError(f"Unsupported LiveCodeBench version: {self.version!r}")
        suffix = self.version.removeprefix("release_")
        path = _LCB_DATA_DIR / f"LiveCodeBench_{suffix}.jsonl"
        if not path.exists():
            raise FileNotFoundError(
                f"dataset {path} not found; populate it with "
                f"`python {_LCB_DATA_DIR / 'fetch_data.py'} livecodebench "
                f"--lcb-version {self.version}`"
            )
        self.source_path = path
        self.source_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        self.logger.info(
            "dataset source: version=%s path=%s sha256=%s",
            self.version, self.source_path, self.source_sha256,
        )

        difficulty = (self.DIFFICULTY or "all").lower()
        with path.open("r", encoding="utf-8") as handle:
            dataset = []
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if (
                    difficulty != "all"
                    and str(row.get("difficulty", "")).lower() != difficulty
                ):
                    continue
                row["private_test_cases"] = translate_private_test_cases(
                    row["private_test_cases"]
                )
                dataset.append(map_to_example(row))

        self._num_problems = len(dataset)
        self.dataset_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
            collate_fn=_lcb_collate,
        )

    def get_question_and_answer(
        self,
        batch: Dict,
        tokenizer,
    ):
        task_ids = list(batch["task_id"])
        prompts = list(batch["prompt"])
        entry_points = list(batch.get("entry_point", [""] * len(task_ids)))
        is_stdins = list(batch.get("is_stdin", [True] * len(task_ids)))
        difficulties = list(batch.get("difficulty", [""] * len(task_ids)))
        tests = list(batch.get("test", [None] * len(task_ids)))

        # Truncate the last partial batch if it would exceed the dataset size.
        if self.seen + len(task_ids) > self.size:
            trunc = self.size - self.seen
            task_ids = task_ids[:trunc]
            prompts = prompts[:trunc]
            entry_points = entry_points[:trunc]
            is_stdins = is_stdins[:trunc]
            difficulties = difficulties[:trunc]
            tests = tests[:trunc]

        input_texts = []

        for tid, p, ep, stdin in zip(task_ids, prompts, entry_points, is_stdins):
            temp = {"task_id": tid, "prompt": p,
                    "entry_point": ep,
                    "is_stdin": stdin}
            input_texts.append(self.format_prompt(temp, tokenizer))

        # The "answer" for each example is the full problem dict so downstream check_correctness can run lcb_run against
        # the decoded test cases.
        answers = []

        for tid, p, ep, stdin, d, t in zip(task_ids, prompts, entry_points, is_stdins, difficulties, tests):
            temp = {"task_id": tid, "prompt": p,
                    "entry_point": ep,
                    "is_stdin": stdin,
                    "difficulty": d,
                    "test": t}
            answers.append(temp)

        self.seen += len(task_ids)
        return input_texts, answers

    def format_prompt(self, problem: Dict, tokenizer) -> str:
        """
        Format a LiveCodeBench problem into a prompt using the Apple LCB templates. Stdin/stdout problems get
        ``LCB_PROMPT_WITHOUT_STARTER_CODE``; function-call problems with a starter signature get
        ``LCB_PROMPT_WITH_STARTER_CODE``. The result is middle-truncated to ``self.max_prompt_chars`` if necessary.
        """
        is_stdin = bool(problem.get("is_stdin", True))
        prompt_text = problem.get("prompt", "") or ""
        entry_point = problem.get("entry_point", "") or ""

        def _build(description: str) -> str:
            if is_stdin:
                return LCB_PROMPT_WITHOUT_STARTER_CODE.format(
                    problem_description=description
                )
            return LCB_PROMPT_WITH_STARTER_CODE.format(
                problem_description=description,
                entry_point=entry_point,
            )

        prompt = _build(prompt_text)
        if len(prompt) > self.max_prompt_chars:
            # Middle-truncate the problem description.
            overhead = len(_build("")) + 32  # +32 for the ellipsis marker
            budget = max(self.max_prompt_chars - overhead, 256)
            if len(prompt_text) > budget:
                head = budget // 2
                tail = budget - head
                prompt_text = (
                        prompt_text[:head]
                        + "\n... [truncated] ...\n"
                        + prompt_text[-tail:]
                )
            prompt = _build(prompt_text)

        if self.use_chat_template:
            # tokenizer_name = str(getattr(tokenizer, "name_or_path", "")).lower()
            messages = [{"role": "user", "content": prompt}]
            chat_template_kwargs = {}
            # if "qwen3" in tokenizer_name:
            #     messages[0]["content"] += "\n/no_think"
            #     chat_template_kwargs["enable_thinking"] = False
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **chat_template_kwargs,
            )
        return prompt

    @staticmethod
    def extract_completion(response: str, problem: Dict) -> str:
        """Extract the last fenced code block from the model response."""
        matches = has_code(response)
        if matches:
            return matches[-1].strip()
        return response.strip()

    def evaluate_single_example(self, example: Dict) -> Dict:
        """
        Evaluate a single example by running its extracted code against the decoded LiveCodeBench test cases.
        ``example`` must carry the keys produced upstream by ``model_answer`` (the extracted code block list) plus the
        canonical ``task_id`` / ``prompt`` / ``entry_point`` / ``is_stdin`` / ``difficulty`` / ``test`` fields.
        """
        try:
            response_entry = {
                "task_id": example.get("task_id"),
                "prompt": example.get("prompt", ""),
                "entry_point": example.get("entry_point", ""),
                "is_stdin": example.get("is_stdin", False),
                "content": example.get("model_answer"),
                "difficulty": example.get("difficulty"),
                "correctness": None,
                "reason": None,
                "test_input": None,
                "test_output": None,
                "test_expected": None,
                "num_tests_passed": 0,
                "num_tests_failed": 0,
                "test_results": [],
            }

            code_filter_result = example.get("model_answer")
            if not code_filter_result or len(code_filter_result) == 0:
                response_entry["correctness"] = False
                response_entry["reason"] = "Does not contain code component."
                return response_entry

            try:
                last_code = code_filter_result[-1]
                problem_to_check = copy.deepcopy(example)

                self.logger.debug(
                    "Evaluating %s problem...", example.get("difficulty")
                )

                curr_res = self.check_correctness(
                    problem=problem_to_check,
                    completion=post_process_code(last_code),
                    timeout=6,
                    is_extracted=not bool(problem_to_check.get("is_stdin", False)),
                )

                self.logger.debug(
                    "Result for %s: %s",
                    example.get("difficulty"), curr_res["all_passed"],
                )

                result_list = curr_res["result_list"]
                test_cases = curr_res["test_cases"] or []

                num_passed = sum(1 for r in result_list if r[0])
                num_failed = len(result_list) - num_passed

                response_entry["test_results"] = [
                    1 if r[0] else 0 for r in result_list
                ]
                response_entry["num_tests_passed"] = num_passed
                response_entry["num_tests_failed"] = num_failed
                response_entry["correctness"] = curr_res["all_passed"]

                if not curr_res["all_passed"]:
                    response_entry["reason"] = "Code is incorrect."
                    for idx, item in enumerate(result_list):
                        passed = item[0]
                        output_value = item[2] if len(item) > 2 else None
                        if not passed and idx < len(test_cases):
                            test_case = test_cases[idx]
                            if isinstance(test_case, dict):
                                response_entry["test_input"] = str(
                                    test_case.get("input", "")
                                )
                                response_entry["test_expected"] = str(
                                    test_case.get("output", "")
                                )
                            response_entry["test_output"] = str(output_value)
                            break
                else:
                    response_entry["reason"] = ""

            except Exception as e:
                self.logger.error(
                    "Error evaluating %s example: %s",
                    example.get("difficulty"), str(e),
                )
                response_entry["correctness"] = False
                response_entry["reason"] = f"Evaluation error: {str(e)}"

            return response_entry

        except Exception as outer_e:
            self.logger.error(
                "Outer error in evaluate_single_example: %s", str(outer_e)
            )
            return {
                "task_id": example.get("task_id"),
                "prompt": example.get("prompt", ""),
                "entry_point": example.get("entry_point", ""),
                "is_stdin": example.get("is_stdin", False),
                "content": example.get("model_answer"),
                "difficulty": example.get("difficulty"),
                "correctness": False,
                "reason": f"Critical error: {str(outer_e)}",
                "test_input": None,
                "test_output": None,
                "test_expected": None,
                "num_tests_passed": 0,
                "num_tests_failed": 0,
                "test_results": [],
            }

    def evaluate(self, examples: List[Dict]) -> Dict:
        """
        Evaluate generated solutions in parallel threads.

        Each example must carry ``model_outputs`` (one completion text) and ``model_answers`` (one extracted code-block
        list from ``has_code``), as produced by the sampler.
        """
        self.logger.info("=" * 80)
        self.logger.info("Evaluating %d examples...", len(examples))
        self.logger.warning(
            "Expect some output leaks from code/test execution into stdout"
        )

        prepared = []
        num_with_code = 0
        for example in examples:
            outputs = example.get("model_outputs", []) or []
            answers = example.get("model_answers", []) or []
            example_copy = example.copy()
            example_copy["model_answer"] = answers[0] if answers else None
            example_copy["model_output"] = outputs[0] if outputs else None
            example_copy.pop("model_outputs", None)
            example_copy.pop("model_answers", None)
            prepared.append(example_copy)
            if example_copy["model_answer"]:
                num_with_code += 1

        self.logger.info(
            "Prepared %d/%d examples with non-empty extracted code blocks",
            num_with_code, len(prepared),
        )
        first = prepared[0]
        first_answer = first.get("model_answer") or []
        self.logger.info(
            "First example: task_id=%s difficulty=%s is_stdin=%s "
            "n_code_blocks=%d output_len=%d",
            first.get("task_id"), first.get("difficulty"),
            first.get("is_stdin"), len(first_answer),
            len(first.get("model_output") or ""),
        )

        num_questions = len(examples)
        results = [None] * len(prepared)
        completed = 0
        with ThreadPoolExecutor(max_workers=32) as executor:
            future_to_example = {
                executor.submit(self.evaluate_single_example, ex): (i, ex)
                for i, ex in enumerate(prepared)
            }
            self.logger.info(
                "Dispatched %d tasks to ThreadPoolExecutor(max_workers=32)",
                len(future_to_example),
            )
            for future in as_completed(future_to_example):
                idx, example = future_to_example[future]
                try:
                    result = future.result()
                    results[idx] = (result, example)
                    completed += 1
                    self.logger.info(
                        "[%d/%d] task_id=%s difficulty=%s correctness=%s "
                        "passed=%d/%d reason=%s",
                        completed, len(prepared),
                        result.get("task_id"), result.get("difficulty"),
                        result.get("correctness"),
                        result.get("num_tests_passed", 0),
                        result.get("num_tests_passed", 0)
                            + result.get("num_tests_failed", 0),
                        result.get("reason"),
                    )
                except Exception as e:
                    self.logger.error(
                        "Future error for example %d: %s", idx, str(e)
                    )
                    results[idx] = (
                        {
                            "task_id": example.get("task_id"),
                            "prompt": example.get("prompt", ""),
                            "entry_point": example.get("entry_point", ""),
                            "is_stdin": example.get("is_stdin", False),
                            "content": example.get("model_answer"),
                            "difficulty": example.get("difficulty"),
                            "correctness": False,
                            "reason": f"Future error: {str(e)}",
                            "test_input": None,
                            "test_output": None,
                            "test_expected": None,
                            "num_tests_passed": 0,
                            "num_tests_failed": 0,
                            "test_results": [],
                        },
                        example,
                    )

        final_metrics: Dict = {}

        results_by_task_id = defaultdict(list)
        results_by_task_id_and_difficulty = defaultdict(lambda: defaultdict(list))

        empty_count = 0
        for result, example in results:
            task_id = result["task_id"]
            difficulty = result["difficulty"]
            test_results = result.get("test_results", [])

            if not test_results:
                empty_count += 1
                num_test_cases = max(len(example.get("test", []) or []), 1)
                self.logger.debug(
                    "Task %s (%s): empty test_results, "
                    "treating as all %d tests failed",
                    task_id, difficulty, num_test_cases,
                )
                test_results = [0] * num_test_cases

            results_by_task_id[task_id].append(test_results)
            results_by_task_id_and_difficulty[difficulty][task_id].append(
                test_results
            )

        self.logger.info(
            "Aggregated %d task_ids across %d difficulty buckets "
            "(%d examples with empty test_results)",
            len(results_by_task_id),
            len(results_by_task_id_and_difficulty),
            empty_count,
        )
        self.logger.info("Computing pass@1 metric")

        if results_by_task_id:
            overall = compute_metrics_from_results(
                dict(results_by_task_id), k_list=[1]
            )["pass@1"]
            final_metrics["pass@1"] = overall
            self.logger.info("Overall pass@1: %.2f%%", overall * 100)

        for difficulty, diff_results in results_by_task_id_and_difficulty.items():
            score = compute_metrics_from_results(
                dict(diff_results), k_list=[1]
            )["pass@1"]
            final_metrics[f"pass@1_{difficulty}"] = score
            self.logger.info(
                "pass@1 %s (n=%d): %.2f%%",
                difficulty, len(diff_results), score * 100,
            )

        final_metrics["examples"] = [result for result, _ in results]
        final_metrics["num_total"] = num_questions
        self.logger.info("=" * 80)

        return final_metrics

    def evaluate_completions(
        self,
        problems: List[Dict],
        completions: List[str],
    ) -> List[Dict]:
        """Extract and grade a batch through the LiveCodeBench evaluator."""
        if len(completions) != len(problems):
            raise RuntimeError(
                f"Sampler returned {len(completions)} completions "
                f"for {len(problems)} prompts"
            )

        examples = []
        for problem, response in zip(problems, completions):
            example = problem.copy()
            example["model_outputs"] = [response]
            example["model_answers"] = [has_code(response)]
            examples.append(example)

        metrics = self.evaluate(examples)
        results = []
        for problem, response, evaluation in zip(
            problems, completions, metrics["examples"]
        ):
            prediction = self.extract_completion(response, problem)
            results.append(
                {
                    "completion": response,
                    "prediction": prediction,
                    "is_correct": bool(evaluation.get("correctness")),
                    "justification": evaluation.get("reason", ""),
                }
            )
        return results

    @staticmethod
    def check_correctness(
            problem: Dict,
            completion: str,
            timeout: float,
            is_extracted: bool = False,
    ) -> Dict:
        """Evaluate functional correctness by running the test suite."""
        result_list = lcb_run(problem, completion, timeout, is_extracted)
        details = [r[0] for r in result_list]
        all_passed = all(details)
        return {
            "all_passed": all_passed,
            "result_list": result_list,
            "test_cases": problem["test"],
        }


class LiveCodeBenchEasyBenchmark(LiveCodeBenchBenchmark):
    """Convenience class filtered to easy problems (raw-JSONL filter)."""

    DIFFICULTY = "easy"


class LiveCodeBenchMediumBenchmark(LiveCodeBenchBenchmark):
    """Convenience class filtered to medium problems (raw-JSONL filter)."""

    DIFFICULTY = "medium"


class LiveCodeBenchHardBenchmark(LiveCodeBenchBenchmark):
    """Convenience class filtered to hard problems (raw-JSONL filter)."""

    DIFFICULTY = "hard"



LCB_CLASSES = {
    "all": LiveCodeBenchBenchmark,
    "easy": LiveCodeBenchEasyBenchmark,
    "medium": LiveCodeBenchMediumBenchmark,
    "hard": LiveCodeBenchHardBenchmark,
}
