"""
GPQA benchmark implementation.

GPQA (Graduate-Level Google-Proof Q&A) is a challenging multiple-choice benchmark designed to test expert-level
reasoning. Questions are crafted by domain experts to be difficult for both search engines and non-experts.

Dataset: https://huggingface.co/datasets/Idavidrein/gpqa
"""

import json
import random
import re

from pathlib import Path
from typing import Dict

from torch.utils.data import DataLoader

from .base import Benchmark

_GPQA_DATA_PATH = Path(__file__).resolve().parents[3] / "data" / "GPQA.jsonl"


_prompt_template = """Answer the following multiple choice question. The last line of your response should be of the following format: '\\boxed{{$LETTER}}' (without quotes) where LETTER is one of ABCD (ex. '\\boxed{{A}}'). Think step by step before answering.

{question}

A) {a}
B) {b}
C) {c}
D) {d}"""


def _last_boxed_letter(completion: str) -> str | None:
    """Return the letter inside the last boxed answer, if it is A-D."""
    if not completion:
        return None
    starts = [m.end() for m in re.finditer(r"\\boxed\s*\{", completion)]
    for start in reversed(starts):
        depth = 1
        index = start
        chars = []
        while index < len(completion) and depth > 0:
            char = completion[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
            if depth > 0:
                chars.append(char)
            index += 1

        inner = "".join(chars)
        inner = re.sub(r"\\(?:text|mathrm|mathbf|rm|textbf)\s*", "", inner)
        inner = inner.replace("{", "").replace("}", "").replace("$", "")
        inner = inner.strip()
        if re.fullmatch(r"[A-Da-d]", inner):
            return inner.upper()
    return None


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
    name = "GPQA"

    def __init__(self, batch_size):
        """
        Args:
            batch_size: DataLoader batch size.
        """
        self.dataset = None
        self.load_dataset(batch_size=batch_size)
        self.seen = 0

    @property
    def size(self):
        """Total number of problems."""
        return len(self.dataset_loader.dataset)

    def load_dataset(self, batch_size: int) -> DataLoader:
        """Load the local GPQA diamond JSONL, preprocess, and build a loader."""
        if self.dataset is None:
            # data/GPQA.jsonl holds the gpqa_diamond split (hardest subset).
            with _GPQA_DATA_PATH.open("r", encoding="utf-8") as handle:
                self.dataset = [
                    json.loads(line) for line in handle if line.strip()
                ]

            # Preprocess: shuffle options and track the correct answer.
            # Choice placement draws from the global random state, which the runners seed via set_random_seed for
            # reproducibility.
            processed_data = []

            for item in self.dataset:
                choices = [
                    item["Incorrect Answer 1"],
                    item["Incorrect Answer 2"],
                    item["Incorrect Answer 3"],
                ]
                random.shuffle(choices)

                correct_idx = random.randint(0, 3)
                choices.insert(correct_idx, item["Correct Answer"])

                processed_data.append({
                    "question": item["Question"],
                    "choices": choices,
                    "correct_idx": correct_idx,
                    "correct_letter": "ABCD"[correct_idx],
                    "correct_answer_text": item["Correct Answer"],
                })

            # Use a pass-through collate so per-item lists (e.g. "choices") don't get transposed by torch's
            # default_collate.
            self.dataset_loader = DataLoader(
                processed_data,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                drop_last=False,
                collate_fn=lambda b: b,
            )

    def get_question_and_answer(self, batch, tokenizer=None):
        # batch is a list of problem dicts (pass-through collate); the dicts are returned as-is so check_correctness can
        # read correct_letter.
        if self.seen + len(batch) > self.size:
            trunc = self.size - self.seen
            batch = batch[:trunc]

        input_texts = [self.format_prompt(p) for p in batch]
        self.seen += len(batch)
        return input_texts, batch

    def format_prompt(self, problem: Dict) -> str:
        """
        Format a GPQA problem into a prompt for the LLM.

        Uses the standard GPQA format with \\boxed{} for the answer.
        """
        question = problem["question"]
        a, b, c, d = problem["choices"]
        return _prompt_template.format(question=question, a=a, b=b, c=c, d=d)

    @staticmethod
    def extract_completion(response: str, problem: Dict = None) -> str:
        """
        Extract the answer choice from LLM response.

        Looks for:
        - \\boxed{A} pattern (preferred)
        - Single letter A-D
        - Various answer patterns
        """
        response_upper = response.upper()

        boxed_match = re.search(r'\\BOXED\{([A-D])\}', response_upper)
        if boxed_match:
            return boxed_match.group(1)

        boxed_match = re.search(r'BOXED\{([A-D])\}', response_upper)
        if boxed_match:
            return boxed_match.group(1)

        patterns = [
            r'ANSWER\s*IS\s*([A-D])',
            r'ANSWER\s*:\s*([A-D])',
            r'OPTION\s*([A-D])',
            r'\(([A-D])\)\s*$',
        ]

        for pattern in patterns:
            match = re.search(pattern, response_upper)
            if match:
                return match.group(1)

        letters = re.findall(r'\b([A-D])\b', response_upper)
        if letters:
            return letters[-1]

        return ""

    def needs_extraction(
        self,
        parsed_answer,
        completion: str | None = None,
        problem: Dict | None = None,
    ) -> bool:
        if (
            not parsed_answer
            or str(parsed_answer).strip().upper() not in {"A", "B", "C", "D"}
        ):
            return True
        return _last_boxed_letter(completion or "") is None

    def extract_answer_text(self, completion: str, problem: Dict) -> str | None:
        return (
            "\n\nGiven the reasoning above, answer with only the letter "
            "A, B, C, or D.\nFinal answer: \\boxed{"
        )

    def extract_max_tokens(self) -> int:
        return 4

    def extract_stop_tokens(self) -> list[str]:
        return ["}", "\n"]

    def parse_extraction(self, extracted_text: str, problem: Dict) -> str:
        if not extracted_text:
            return ""
        wrapped = "\\boxed{" + extracted_text
        if wrapped.count("}") < wrapped.count("{"):
            wrapped += "}" * (wrapped.count("{") - wrapped.count("}"))
        return self.extract_completion(wrapped, problem)

    @staticmethod
    def check_correctness(problem: Dict, completion: str, verbose=False) -> tuple[bool, str]:
        """
        Check if the completion is correct.

        Compares the extracted answer letter with the correct answer.

        Returns: (passed, result_message)
        """
        try:
            if verbose:
                print(">>>> ", problem)
            expected = problem["correct_letter"]

            if not completion or not completion.strip():
                return False, "empty completion"

            completion_clean = completion.strip().upper()

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
