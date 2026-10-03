"""
GPQA benchmark implementation.

GPQA (Graduate-Level Google-Proof Q&A) is a challenging multiple-choice benchmark
designed to test expert-level reasoning. Questions are crafted by domain experts
to be difficult for both search engines and non-experts.

Dataset: https://huggingface.co/datasets/Idavidrein/gpqa
"""

import random
import re
from benchmark_runner import Benchmark
from typing import Dict, List
from datasets import load_dataset


class GPQABenchmark(Benchmark):
    """
    GPQA benchmark for graduate-level reasoning.

    Dataset: Idavidrein/gpqa (gpqa_diamond config)
    - ~198 expert-crafted questions
    - Multiple choice with 4 options (A, B, C, D)
    - Covers biology, physics, chemistry

    Each problem includes:
    - Question: The question text
    - Correct Answer: The correct answer text
    - Incorrect Answer 1/2/3: Three incorrect answer texts
    """

    def __init__(self, seed: int = 42):
        """
        Args:
            seed: Random seed for shuffling answer choices (for reproducibility)
        """
        self.dataset = None
        self.processed_data = []
        self.seed = seed

    def name(self) -> str:
        return "GPQA"

    def load_dataset(self):
        """Load GPQA dataset from HuggingFace and preprocess."""
        if self.dataset is None:
            # Load the gpqa_diamond config (hardest subset)
            self.dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")

            # Preprocess: shuffle options and track correct answer
            random.seed(self.seed)
            self.processed_data = []

            for item in self.dataset:
                # Get all choices
                choices = [
                    item["Incorrect Answer 1"],
                    item["Incorrect Answer 2"],
                    item["Incorrect Answer 3"]
                ]
                random.shuffle(choices)

                # Insert correct answer at random position
                correct_idx = random.randint(0, 3)
                choices.insert(correct_idx, item["Correct Answer"])

                # Store processed problem
                self.processed_data.append({
                    "question": item["Question"],
                    "choices": choices,
                    "correct_idx": correct_idx,
                    "correct_letter": "ABCD"[correct_idx],
                    "correct_answer_text": item["Correct Answer"]
                })

    def get_problem(self, index: int) -> Dict:
        """Get a problem by index."""
        return self.processed_data[index]

    def get_num_problems(self) -> int:
        """Return total number of problems."""
        return len(self.processed_data)

    def format_prompt(self, problem: Dict) -> str:
        """
        Format a GPQA problem into a prompt for the LLM.

        Uses the standard GPQA format with \boxed{} for the answer.
        """
        question = problem["question"]
        choices = problem["choices"]

        prompt = f"""Answer the following multiple choice question. The last line of your response should be of the following format: '\\boxed{{$LETTER}}' (without quotes) where LETTER is one of ABCD (ex. '\\boxed{{A}}'). Think step by step before answering.

{question}

A) {choices[0]}
B) {choices[1]}
C) {choices[2]}
D) {choices[3]}"""

        return prompt

    def extract_completion(self, response: str, problem: Dict) -> str:
        """
        Extract the answer choice from LLM response.

        Looks for:
        - \boxed{A} pattern (preferred)
        - Single letter A-D
        - Various answer patterns
        """
        response_upper = response.upper()

        # Try to find \boxed{X} pattern first
        boxed_match = re.search(r'\\BOXED\{([A-D])\}', response_upper)
        if boxed_match:
            return boxed_match.group(1)

        # Try plain boxed without escaping
        boxed_match = re.search(r'BOXED\{([A-D])\}', response_upper)
        if boxed_match:
            return boxed_match.group(1)

        # Look for "Answer: X" or "The answer is X" patterns
        patterns = [
            r'ANSWER\s*IS\s*([A-D])',
            r'ANSWER\s*:\s*([A-D])',
            r'OPTION\s*([A-D])',
            r'\(([A-D])\)\s*$',  # (A) at end
        ]

        for pattern in patterns:
            match = re.search(pattern, response_upper)
            if match:
                return match.group(1)

        # Find all A-D letters and return the last one (usually the final answer)
        letters = re.findall(r'\b([A-D])\b', response_upper)
        if letters:
            return letters[-1]

        # If nothing found, return empty
        return ""

    def check_correctness(self, problem: Dict, completion: str) -> tuple[bool, str]:
        """
        Check if the completion is correct.

        Compares the extracted answer letter with the correct answer.

        Returns: (passed, result_message)
        """
        try:
            expected = problem["correct_letter"]

            if not completion or not completion.strip():
                return False, "empty completion"

            # Normalize to uppercase single letter
            completion_clean = completion.strip().upper()

            # Extract letter if there's extra text
            if len(completion_clean) > 1:
                match = re.search(r'[A-D]', completion_clean)
                if match:
                    completion_clean = match.group(0)

            passed = completion_clean == expected

            if passed:
                return True, f"correct: {expected}"
            else:
                return False, f"incorrect: got {completion_clean}, expected {expected}"

        except Exception as e:
            return False, f"error during check: {str(e)}"

    def format_prediction(self, problem: Dict, completion: str) -> Dict:
        """Format prediction for evaluation."""
        return {
            "question": problem["question"],
            "choices": problem["choices"],
            "predicted_answer": completion,
            "ground_truth": problem["correct_letter"],
            "correct_answer_text": problem["correct_answer_text"]
        }
