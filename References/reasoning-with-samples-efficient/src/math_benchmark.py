"""
MATH benchmark implementation.

MATH is a dataset of 12,500 challenging competition mathematics problems.
This implementation uses the MATH500 subset - 500 representative problems.
"""

import json
import os
import re
import sys
from benchmark_runner import Benchmark
from typing import Dict

# Add parent directory to path for importing grader utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'llm_experiments'))
from grader_utils.math_grader import grade_answer


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

    def __init__(self):
        self.dataset = None
        self._data_path = os.path.join(
            os.path.dirname(__file__),
            '..',
            'llm_experiments',
            'data',
            'MATH500.json'
        )

    def name(self) -> str:
        return "MATH500"

    def load_dataset(self):
        """Load MATH500 dataset from local JSON file."""
        if self.dataset is None:
            with open(self._data_path, 'r') as f:
                self.dataset = json.load(f)

    def get_problem(self, index: int) -> Dict:
        """Get a problem by index."""
        return self.dataset[index]

    def get_num_problems(self) -> int:
        """Return total number of problems."""
        return len(self.dataset)

    def format_prompt(self, problem: Dict) -> str:
        """
        Format a MATH problem into a prompt for the LLM.

        Instructs the model to:
        - Show step-by-step reasoning
        - Provide final answer in \boxed{}
        """
        question = problem.get("prompt", "")

        prompt = f"""Solve the following math problem step by step. Show your work and reasoning clearly.

Problem: {question}

Provide your final answer in \\boxed{{}} at the end.

Format your response like this:
[Your step-by-step solution]
Therefore, the answer is \\boxed{{[your final answer]}}"""

        return prompt

    def extract_completion(self, response: str, problem: Dict) -> str:
        """
        Extract the final answer from LLM response.

        Looks for content inside \boxed{...}, handling nested braces.
        Falls back to various patterns if no boxed answer found.
        """
        # Try to find the last \boxed{...} in the response
        # Handle nested braces by counting depth
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

    def check_correctness(self, problem: Dict, completion: str) -> tuple[bool, str]:
        """
        Check if the completion is correct.

        Uses the sympy-based grade_answer function for robust
        comparison of mathematical expressions.

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

    def format_prediction(self, problem: Dict, completion: str) -> Dict:
        """Format prediction for evaluation."""
        return {
            "problem_id": problem.get("id", ""),
            "problem": problem.get("prompt", ""),
            "predicted_answer": completion,
            "ground_truth": problem.get("answer", "")
        }
