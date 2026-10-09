"""
HumanEval benchmark implementation.

HumanEval is a dataset of 164 hand-written Python programming problems used to evaluate functional correctness of code
generation.

The benchmark reads the local ``data/human-eval-v2-20210705.jsonl``.
"""

import json

from pathlib import Path
from typing import Any, Dict

from torch.utils.data import DataLoader

from .base import Benchmark
from .grader_utils.he_grader import extract_code
from .grader_utils.he_execute import check_correctness as he_check_correctness

_HUMANEVAL_DATA_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "human-eval-v2-20210705.jsonl"
)


_prompt_template = """Complete the following Python function. Return the full function definition inside a Python code block.

{}

Format your response like this:
```python
[your completed function here]
```"""

_TULU_RAW_PROMPT_MODELS = frozenset(
    {"tulu", "tulu3-dpo", "tulu3-sft", "tulu3-grpo"}
)


class HumanEvalBenchmark(Benchmark):
    """
    HumanEval benchmark for code generation.

    Dataset: openai/openai_humaneval (164 problems, "test" split)
    - Each problem provides a function signature and docstring.
    - Correctness is evaluated by executing model-generated code against the provided unit tests.

    Each problem includes:
    - task_id: Unique problem identifier (e.g. "HumanEval/0")
    - prompt: Function signature + docstring
    - canonical_solution: Reference solution body
    - test: Unit-test code that defines a check() function
    - entry_point: Name of the function to implement
    """
    name = "HumanEval"
    uses_max_model_len = False

    def __init__(
        self,
        batch_size,
        model_name: str | None = None,
        model_str: str | None = None,
    ):
        self.dataset = None
        self.model_name = model_name or ""
        self.model_str = model_str or ""
        self.load_dataset(batch_size=batch_size)
        self.seen = 0

    @property
    def size(self):
        """Total number of problems."""
        return len(self.dataset_loader.dataset)

    def load_dataset(self, batch_size: int) -> DataLoader:
        """Load the local HumanEval JSONL file."""
        if self.dataset is None:
            with _HUMANEVAL_DATA_PATH.open("r", encoding="utf-8") as handle:
                self.dataset = [
                    json.loads(line) for line in handle if line.strip()
                ]
            self.dataset_loader = DataLoader(
                self.dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                drop_last=False,
            )

    def get_question_and_answer(
        self,
        batch: Dict,
        tokenizer: Any,
    ):
        prompts = list(batch["prompt"])
        task_ids = list(batch["task_id"])
        entry_points = list(batch["entry_point"])
        tests = list(batch["test"])
        canonical = list(batch.get("canonical_solution", [""] * len(prompts)))

        # Truncate the last partial batch if it would exceed maxi_iters.
        if self.seen + len(prompts) > self.size:
            trunc = self.size - self.seen
            prompts = prompts[:trunc]
            task_ids = task_ids[:trunc]
            entry_points = entry_points[:trunc]
            tests = tests[:trunc]
            canonical = canonical[:trunc]

        # The "answer" returned for each example is the full problem dict so that downstream check_correctness can run
        # the unit tests.
        answers = [
            {
                "task_id": tid,
                "prompt": p,
                "entry_point": ep,
                "test": t,
                "canonical_solution": cs,
            }
            for tid, p, ep, t, cs in zip(task_ids, prompts, entry_points, tests, canonical)
        ]
        input_texts = [
            self.format_prompt(
                problem,
                use_chat_template=self.use_chat_template,
                tokenizer=tokenizer,
            )
            for problem in answers
        ]

        self.seen += len(prompts)
        return input_texts, answers

    def format_prompt(
        self,
        problem,
        use_chat_template: bool = False,
        tokenizer: Any | None = None,
    ):
        """
        Format a HumanEval problem into a prompt for the LLM.

        Accepts either a raw prompt string (function signature + docstring) or a problem dict with a "prompt" key.
        """
        if not isinstance(problem, dict):
            prompt = problem
        else:
            question = problem.get("prompt", "")
            if self._uses_raw_humaneval_prompt():
                prompt = question
            else:
                prompt = _prompt_template.format(question)

        if use_chat_template:
            if tokenizer is None:
                raise ValueError(
                    "tokenizer is required when use_chat_template=True"
                )
            messages = [{"role": "user", "content": prompt}]
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        return prompt

    def extract_completion(self, response: str, problem: Dict) -> str:
        """
        Extract the function body from LLM response.

        Uses he_grader.extract_code which handles fenced code blocks and falls back to indenting raw text if no block is
        found.
        """
        entry_point = ""
        if isinstance(problem, dict):
            entry_point = problem.get("entry_point", "")
        if self._uses_raw_humaneval_prompt():
            prompt = problem.get("prompt", "") if isinstance(problem, dict) else ""
            return extract_code(prompt + response, entry_point)
        return extract_code(response, entry_point)

    def _uses_raw_humaneval_prompt(self) -> bool:
        return (
            self.model_name in _TULU_RAW_PROMPT_MODELS
            or "Tulu" in self.model_str
        )

    def should_use_chat_template(self, model_name: str) -> bool:
        self.use_chat_template = self._uses_raw_humaneval_prompt()
        return self.use_chat_template

    @staticmethod
    def check_correctness(
        problem: Dict, completion: str, timeout: float = 3.0
    ) -> tuple[bool, str]:
        """
        Check if the completion is functionally correct by executing the problem's unit tests against `prompt +
        completion`.

        Returns: (passed, result_message)
        """
        try:
            if not completion or not completion.strip():
                return False, "empty completion"

            result = he_check_correctness(problem, completion, timeout=timeout)
            passed = bool(result.get("passed", False))
            message = result.get("result", "")

            if passed:
                return True, f"correct: {message}"
            return False, f"incorrect: {message}"

        except Exception as e:
            return False, f"error during check: {str(e)}"
