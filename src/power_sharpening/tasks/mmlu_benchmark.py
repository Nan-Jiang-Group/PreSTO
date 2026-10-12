"""
MMLU benchmark implementation.

MMLU (Massive Multitask Language Understanding) is a benchmark covering 57 subjects across STEM, humanities, social
sciences, and more. Each question is multiple choice with 4 options (A, B, C, D).

Dataset: https://huggingface.co/datasets/cais/mmlu
"""

import re
from typing import Dict, List, Optional

from datasets import load_dataset
from torch.utils.data import DataLoader

from .base import Benchmark


_prompt_template = """Subject: {subject}

Question: {question}

{choices_text}
Answer with only the letter (A, B, C, or D) of the correct answer."""


MMLU_SUBJECTS = [
    # STEM
    "abstract_algebra", "anatomy", "astronomy", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_physics", "computer_security",
    "conceptual_physics", "electrical_engineering", "elementary_mathematics", "high_school_biology",
    "high_school_chemistry", "high_school_computer_science", "high_school_mathematics",
    "high_school_physics", "high_school_statistics", "machine_learning",
    # Humanities
    "formal_logic", "high_school_european_history", "high_school_us_history",
    "high_school_world_history", "international_law", "jurisprudence", "logical_fallacies",
    "moral_disputes", "moral_scenarios", "philosophy", "prehistory", "professional_law",
    "world_religions",
    # Social Sciences
    "econometrics", "high_school_geography", "high_school_government_and_politics",
    "high_school_macroeconomics", "high_school_microeconomics", "high_school_psychology",
    "human_sexuality", "professional_psychology", "public_relations", "security_studies",
    "sociology", "us_foreign_policy",
    # Other
    "business_ethics", "clinical_knowledge", "college_medicine", "global_facts",
    "human_aging", "management", "marketing", "medical_genetics", "miscellaneous",
    "nutrition", "professional_accounting", "professional_medicine", "virology"
]


class MMLUBenchmark(Benchmark):
    """
    MMLU benchmark for multitask language understanding.

    Dataset: https://huggingface.co/datasets/cais/mmlu
    - 57 subjects across STEM, humanities, social sciences
    - Multiple choice questions with 4 options (A, B, C, D)
    - ~14,000 test questions total

    Each problem includes:
    - question: The question text
    - choices: List of 4 answer choices
    - answer: The correct answer index (0-3)
    - subject: The subject category
    """
    name = "MMLU"

    def __init__(self, batch_size, split: str = "test", subjects: Optional[List[str]] = None):
        self.dataset = None
        self.split = split
        self.subjects = subjects or MMLU_SUBJECTS
        self.load_dataset(batch_size=batch_size)
        self.seen = 0

    @property
    def size(self):
        """Total number of problems."""
        return len(self.dataset_loader.dataset)

    def load_dataset(self, batch_size: int) -> DataLoader:
        """Load MMLU dataset from HuggingFace."""
        if self.dataset is None:
            all_data = []
            for subject in self.subjects:
                try:
                    subject_data = load_dataset("cais/mmlu", subject, split=self.split)
                    for example in subject_data:
                        example_dict = dict(example)
                        example_dict["subject"] = subject
                        all_data.append(example_dict)
                except Exception as e:
                    print(f"Warning: Could not load subject '{subject}': {e}")

            if not all_data:
                raise ValueError(f"No data loaded for MMLU with subjects: {self.subjects}")

            self.dataset = all_data
            self.dataset_loader = DataLoader(
                self.dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                drop_last=False,
            )

    @staticmethod
    def batch_to_problems(batch) -> list[Dict]:
        """Rebuild per-example dicts, un-transposing the "choices" field.

        DataLoader's default collate turns each example's 4-element ``choices`` list into 4 tuples of batch_size, so the
        base implementation reads ``choices`` as one entry per example and runs off the end (or, when the batch happens
        to hold 4 examples, silently pairs each example with a choice *position*). Pull ``choices`` out, collate the
        rest normally, and transpose it back.
        """
        if not isinstance(batch, dict) or "choices" not in batch:
            return Benchmark.batch_to_problems(batch)

        choices_per_position = batch["choices"]
        rest = {key: value for key, value in batch.items() if key != "choices"}
        problems = Benchmark.batch_to_problems(rest)
        for index, problem in enumerate(problems):
            problem["choices"] = [
                choices_per_position[c][index]
                for c in range(len(choices_per_position))
            ]
        return problems

    def get_question_and_answer(self, batch: Dict, tokenizer=None):
        questions = list(batch["question"])
        subjects = list(batch["subject"])

        # DataLoader's default collate transposes list-valued fields, so batch["choices"] becomes a list of 4 lists (one
        # per choice position).
        choices_per_position = batch["choices"]
        n = len(questions)
        choices_list = [
            [choices_per_position[c][i] for c in range(len(choices_per_position))]
            for i in range(n)
        ]

        raw_answers = list(batch["answer"])

        # Truncate the last partial batch if it would exceed the dataset size.
        if self.seen + len(questions) > self.size:
            trunc = self.size - self.seen
            questions = questions[:trunc]
            subjects = subjects[:trunc]
            choices_list = choices_list[:trunc]
            raw_answers = raw_answers[:trunc]

        input_texts = [
            self.format_prompt({"question": q, "choices": c, "subject": s})
            for q, c, s in zip(questions, choices_list, subjects)
        ]

        # The "answer" returned for each example is the full problem dict so that downstream extract_completion /
        # check_correctness can run.
        answers = [
            {
                "question": q,
                "subject": s,
                "choices": c,
                "answer": a,
                "correct_letter": self._answer_to_letter(a),
            }
            for q, s, c, a in zip(questions, subjects, choices_list, raw_answers)
        ]

        self.seen += len(questions)
        return input_texts, answers

    def format_prompt(self, problem) -> str:
        """
        Format an MMLU problem into a prompt for the LLM.

        Presents the question with multiple choice options A-D.
        """
        if isinstance(problem, dict):
            question = problem.get("question", "")
            choices = problem.get("choices", [])
            subject = problem.get("subject", "").replace("_", " ").title()
        else:
            question = problem
            choices = []
            subject = ""

        choices_text = ""
        for i, choice in enumerate(choices):
            letter = chr(65 + i)  # 65 is ASCII for 'A'
            choices_text += f"{letter}. {choice}\n"

        return _prompt_template.format(
            subject=subject, question=question, choices_text=choices_text
        )

    @staticmethod
    def extract_completion(response: str, problem: Dict) -> str:
        """
        Extract the answer choice from LLM response.

        Looks for:
        - Single letter A-D
        - "Answer: A" pattern
        - "The answer is A" pattern
        """
        response_clean = response.strip().upper()

        patterns = [
            r'ANSWER\s*IS\s*([A-D])',
            r'ANSWER\s*:\s*([A-D])',
            r'^\s*([A-D])\s*$',
            r'OPTION\s*([A-D])',
            r'CHOICE\s*([A-D])',
            r'\(([A-D])\)',
        ]

        for pattern in patterns:
            match = re.search(pattern, response_clean)
            if match:
                return match.group(1)

        match = re.search(r'[A-D]', response_clean)
        if match:
            return match.group(0)

        return response.strip()

    @staticmethod
    def _answer_to_letter(answer_idx) -> str:
        """Convert a raw answer field (int, tensor, or string) to a letter A-D."""
        if hasattr(answer_idx, "item"):
            try:
                answer_idx = answer_idx.item()
            except Exception:
                pass

        if isinstance(answer_idx, str):
            if answer_idx.upper() in ['A', 'B', 'C', 'D']:
                return answer_idx.upper()
            try:
                answer_idx = int(answer_idx)
            except ValueError:
                return 'A'

        if isinstance(answer_idx, int) and 0 <= answer_idx <= 3:
            return chr(65 + answer_idx)

        return 'A'

    @staticmethod
    def check_correctness(problem: Dict, completion: str) -> tuple[bool, str]:
        """
        Check if the completion is correct.

        Compares the extracted answer letter with the ground truth.

        Returns: (passed, result_message)
        """
        try:
            expected = MMLUBenchmark._answer_to_letter(problem.get("answer", 0))

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


class MMLUSTEMBenchmark(MMLUBenchmark):
    """MMLU benchmark with only STEM subjects."""
    name = "MMLU-STEM"

    STEM_SUBJECTS = [
        "abstract_algebra", "anatomy", "astronomy", "college_biology", "college_chemistry",
        "college_computer_science", "college_mathematics", "college_physics", "computer_security",
        "conceptual_physics", "electrical_engineering", "elementary_mathematics", "high_school_biology",
        "high_school_chemistry", "high_school_computer_science", "high_school_mathematics",
        "high_school_physics", "high_school_statistics", "machine_learning"
    ]

    def __init__(self, batch_size, split: str = "test"):
        super().__init__(batch_size=batch_size, split=split, subjects=self.STEM_SUBJECTS)


class MMLUHumanitiesBenchmark(MMLUBenchmark):
    """MMLU benchmark with only Humanities subjects."""
    name = "MMLU-Humanities"

    HUMANITIES_SUBJECTS = [
        "formal_logic", "high_school_european_history", "high_school_us_history",
        "high_school_world_history", "international_law", "jurisprudence", "logical_fallacies",
        "moral_disputes", "moral_scenarios", "philosophy", "prehistory", "professional_law",
        "world_religions"
    ]

    def __init__(self, batch_size, split: str = "test"):
        super().__init__(batch_size=batch_size, split=split, subjects=self.HUMANITIES_SUBJECTS)


class MMLUSocialSciencesBenchmark(MMLUBenchmark):
    """MMLU benchmark with only Social Sciences subjects."""
    name = "MMLU-SocialSciences"

    SOCIAL_SCIENCES_SUBJECTS = [
        "econometrics", "high_school_geography", "high_school_government_and_politics",
        "high_school_macroeconomics", "high_school_microeconomics", "high_school_psychology",
        "human_sexuality", "professional_psychology", "public_relations", "security_studies",
        "sociology", "us_foreign_policy"
    ]

    def __init__(self, batch_size, split: str = "test"):
        super().__init__(batch_size=batch_size, split=split, subjects=self.SOCIAL_SCIENCES_SUBJECTS)
