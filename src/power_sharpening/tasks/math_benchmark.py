"""
MATH benchmark implementation.

MATH is a dataset of 12,500 challenging competition mathematics problems. This implementation uses the MATH500 subset -
500 representative problems.
"""

import json
import os
import re

from typing import Any, Dict

from torch.utils.data import DataLoader

from .grader_utils.math_grader import grade_answer
from .base import Benchmark


_prompt_template="""Solve the following math problem step by step. Show your work and reasoning clearly.

Problem: {}

Provide your final answer in \\boxed{{}} at the end.

Format your response like this:
[Your step-by-step solution]
Therefore, the answer is \\boxed{{[your final answer]}}"""

_data_filename='MATH500.json'

class MATHBenchmark(Benchmark):
    """
    MATH benchmark for mathematical reasoning.

    Dataset: MATH500 subset (500 problems from the MATH dataset)
    - Problems span algebra, geometry, number theory, calculus, etc.
    - Answers are symbolic/LaTeX formatted in \boxed{}

    Each problem includes:
    - prompt: The math problem statement
    - answer: Ground truth answer (often LaTeX)
    - source: Always "math"
    - id: Original problem identifier
    """
    name="MATH500"

    def __init__(self, batch_size):
        self.dataset = None
        # Data files live in <repo-root>/data, three levels above tasks/.
        self._data_path = os.path.join(
            os.path.dirname(__file__),
            "..", "..", "..",
            "data",
            _data_filename
        )
        self.load_dataset(batch_size=batch_size)
        self.seen = 0

    @property
    def size(self):
        """Total number of problems."""
        return len(self.dataset_loader.dataset)

    def load_dataset(self, batch_size: int) -> DataLoader:
        """Load MATH500 dataset from local JSON file."""
        if self.dataset is None:
            with open(self._data_path, 'r') as f:
                self.dataset = json.load(f)
                self.dataset_loader = DataLoader(
                    self.dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=0,
                    drop_last=False,
                )

    def get_question_and_answer(
        self, batch: Dict, tokenizer: Any | None = None,
    ):
        # The second return value is the full problem dicts so downstream evaluate_completions can grade against the
        # ground truth.
        problems = self.batch_to_problems(batch)

        # Truncate the last partial batch if it would exceed the dataset size.
        if self.seen + len(problems) > self.size:
            problems = problems[:self.size - self.seen]

        input_texts = [self.format_prompt(p) for p in problems]
        self.seen += len(problems)
        return input_texts, problems


    def format_prompt(self, problem) -> str:
        """
        Format a MATH problem into a prompt for the LLM.

        Accepts either a raw question string (as passed from get_question_and_answer) or a problem dict with a "prompt"
        key.
        """
        if isinstance(problem, dict):
            question = problem.get("prompt", "")
        else:
            question = problem

        prompt = _prompt_template.format(question)

        return prompt

    @staticmethod
    def extract_completion(response: str, problem: Dict) -> str:
        """
        Extract the final answer from LLM response.

        Looks for content inside \boxed{...}, handling nested braces. Falls back to various patterns if no boxed answer
        found.
        """
        # Try to find the last \boxed{...} in the response Handle nested braces by counting depth
        boxed_matches = []
        i = 0
        while i < len(response):
            # Look for \boxed{
            if response[i:i+7] == '\\boxed{':
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
                    boxed_matches.append(response[start:j-1])
                i = j
            else:
                i += 1

        if boxed_matches:
            # Return the last boxed answer (typically the final answer)
            return boxed_matches[-1].strip()

        # Fallback: try simpler regex for \boxed{...}
        match = re.search(r'\\boxed\{([^{}]+)\}', response)
        if match:
            return match.group(1).strip()

        # Try "the answer is X" pattern
        match = re.search(r'[Tt]he answer is[:\s]*([^\n.]+)', response)
        if match:
            return match.group(1).strip()

        # Try "= X" at end of lines
        lines = response.strip().split('\n')
        for line in reversed(lines):
            match = re.search(r'=\s*([^\n=]+)\s*$', line)
            if match:
                return match.group(1).strip()

        # Last resort: return empty
        return ""

    @staticmethod
    def check_correctness(problem: Dict, completion: str) -> tuple[bool, str]:
        """
        Check if the completion is correct.

        Uses the sympy-based grade_answer function for robust comparison of mathematical expressions.

        Returns: (passed, result_message)
        """
        try:
            expected = problem.get("answer", "")

            if not expected:
                return False, "could not extract expected answer"

            if not completion or not completion.strip():
                return False, "empty completion"

            # Use the sophisticated math grader
            passed = grade_answer(completion, expected)

            if passed:
                return True, f"correct: {completion}"
            else:
                return False, f"incorrect: got '{completion}', expected '{expected}'"

        except Exception as e:
            return False, f"error during check: {str(e)}"
