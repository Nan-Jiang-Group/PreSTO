"""
AIME benchmark implementation.

AIME (American Invitational Mathematics Examination) is a challenging competition math benchmark. Every answer is an
integer in [0, 999].

The benchmark reads ``data/AIME.json`` and supports the 2024, 2025, or combined 2024-2025 subsets.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict

from torch.utils.data import DataLoader

from .grader_utils.math_grader import grade_answer
from .base import Benchmark


_prompt_template = """Answer the following math problem. The last line of your response should be of the following format: '\\boxed{{$ANSWER}}' (without quotes), where ANSWER is an integer between 0 and 999 inclusive (for example, '\\boxed{{42}}'). Think step by step before answering.

{question}"""


_AIME_YEAR_RANGES = {
    "aime2024": (2024, 2024),
    "aime2025": (2025, 2025),
    "aime2024-2025": (2024, 2025),
}

_AIME_DATA_PATH = Path(__file__).resolve().parents[3] / "data" / "AIME.json"


class AIMEBenchmark(Benchmark):
    """
    AIME benchmark for competition-level mathematical reasoning.

    Supports AIME 2024, AIME 2025, or their combined evaluation set through the ``name`` argument.

    Each problem includes:
    - Problem: The problem statement
    - Answer: Integer ground-truth answer in [0, 999]
    - ID: Problem identifier
    """
    name = "aime2024-2025"

    def __init__(self, batch_size, name: str = "aime2024-2025"):
        """
        Args:
            batch_size: DataLoader batch size.
            name: ``"aime2024"``, ``"aime2025"``, or ``"aime2024-2025"``. Selects the local year range and is used as
                this benchmark's name.
        """
        name = str(name).lower()
        if name not in _AIME_YEAR_RANGES:
            raise ValueError(
                f"Unsupported AIME name '{name}'. Expected one of "
                f"{sorted(_AIME_YEAR_RANGES.keys())}."
            )

        self.name = name
        self.dataset = None
        self.start_year, self.end_year = _AIME_YEAR_RANGES[name]
        self.load_dataset(batch_size=batch_size)
        self.seen = 0

    @property
    def size(self):
        """Total number of problems."""
        return len(self.dataset_loader.dataset)

    def load_dataset(self, batch_size: int) -> DataLoader:
        """Load local AIME data and retain problems in the configured years."""
        if self.dataset is None:
            with _AIME_DATA_PATH.open("r", encoding="utf-8") as handle:
                self.dataset = json.load(handle)

            processed_data = [
                {
                    "id": item["id"],
                    "year": int(item["year"]),
                    "part": item.get("part"),
                    "problem_num": item.get("problem_num"),
                    "prompt": item["prompt"],
                    "answer": str(item["answer"]).strip(),
                    "source_dataset": f"aime{int(item['year'])}",
                }
                for item in self.dataset
                if self.start_year <= int(item["year"]) <= self.end_year
            ]

            self.dataset_loader = DataLoader(
                processed_data,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                drop_last=False,
                collate_fn=lambda b: b,
            )

    def get_question_and_answer(
        self, batch, tokenizer: Any,
    ):
        # batch is a list of problem dicts (pass-through collate).
        if self.seen + len(batch) > self.size:
            trunc = self.size - self.seen
            batch = batch[:trunc]

        input_texts = [
            self.format_prompt(problem, tokenizer=tokenizer)
            for problem in batch
        ]
        self.seen += len(batch)
        return input_texts, batch

    def format_prompt(
        self,
        problem: Dict,
        tokenizer: Any | None = None,
    ) -> str:
        """
        Format an AIME problem into a prompt for the LLM.

        Uses the GPQA-style instruction with an explicit boxed-answer format.
        """
        question = problem["prompt"]
        prompt = _prompt_template.format(question=question)

        if self.use_chat_template:
            if tokenizer is None:
                raise ValueError(
                    "tokenizer is required when use_chat_template=True"
                )
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )

        return prompt

    @staticmethod
    def extract_completion(response: str, problem: Dict = None) -> str:
        """
        Extract the final answer from LLM response.

        Looks for content inside \\boxed{...} (handling nested braces), then falls back to common answer patterns and
        finally the last integer in the response.
        """
        boxed_matches = []
        i = 0
        while i < len(response):
            if response[i:i + 7] == '\\boxed{':
                start = i + 7
                depth = 1
                j = start
                while j < len(response) and depth > 0:
                    if response[j] == '{':
                        depth += 1
                    elif response[j] == '}':
                        depth -= 1
                    j += 1
                if depth == 0:
                    boxed_matches.append(response[start:j - 1])
                i = j
            else:
                i += 1

        if boxed_matches:
            return boxed_matches[-1].strip()

        match = re.search(r'\\boxed\{([^{}]+)\}', response)
        if match:
            return match.group(1).strip()

        match = re.search(r'[Tt]he answer is[:\s]*([^\n.]+)', response)
        if match:
            return match.group(1).strip()

        integers = re.findall(r'-?\d+', response)
        if integers:
            return integers[-1]

        return ""

    def needs_extraction(
        self,
        parsed_answer,
        completion: str | None = None,
        problem: Dict | None = None,
    ) -> bool:
        return not parsed_answer or str(parsed_answer).strip() == ""

    def extract_answer_text(self, completion: str, problem: Dict) -> str | None:
        return "\n\nFinal answer: $\\boxed{"

    def extract_max_tokens(self) -> int:
        return 8

    def extract_stop_tokens(self) -> list[str]:
        return ["}", "$", "\n"]

    def parse_extraction(self, extracted_text: str, problem: Dict) -> str:
        if not extracted_text:
            return ""
        wrapped = "\\boxed{" + extracted_text
        if wrapped.count("}") < wrapped.count("{"):
            wrapped += "}" * (wrapped.count("{") - wrapped.count("}"))
        return self.extract_completion(wrapped, problem)

    @staticmethod
    def check_correctness(problem: Dict, completion: str) -> tuple[bool, str]:
        """
        Check if the completion is correct.

        AIME answers are integers in [0, 999], so we first try a direct integer comparison and fall back to the math
        grader for robustness against extra LaTeX wrapping.

        Returns: (passed, result_message)
        """
        try:
            expected = str(problem.get("answer", "")).strip()

            if not expected:
                return False, "could not extract expected answer"

            if not completion or not completion.strip():
                return False, "empty completion"

            completion_clean = completion.strip()

            # Try strict integer comparison after stripping non-digit chars.
            pred_int_match = re.search(r'-?\d+', completion_clean)
            exp_int_match = re.search(r'-?\d+', expected)
            if pred_int_match and exp_int_match:
                try:
                    if int(pred_int_match.group(0)) == int(exp_int_match.group(0)):
                        return True, f"correct: {expected}"
                except ValueError:
                    pass

            # Fall back to the sympy-based math grader.
            if grade_answer(completion_clean, expected):
                return True, f"correct: {expected}"

            return False, f"incorrect: got '{completion_clean}', expected '{expected}'"

        except Exception as e:
            return False, f"error during check: {str(e)}"


class AIME2024Benchmark(AIMEBenchmark):
    """AIME 2024 convenience subclass."""

    def __init__(self, batch_size):
        super().__init__(batch_size=batch_size, name="aime2024")


class AIME2025Benchmark(AIMEBenchmark):
    """AIME 2025 convenience subclass."""

    def __init__(self, batch_size):
        super().__init__(batch_size=batch_size, name="aime2025")
