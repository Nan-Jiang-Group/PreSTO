"""
MMLU benchmark implementation.

MMLU (Massive Multitask Language Understanding) is a benchmark covering 57 subjects
across STEM, humanities, social sciences, and more. Each question is multiple choice
with 4 options (A, B, C, D).

Dataset: https://huggingface.co/datasets/cais/mmlu
"""

from benchmark_runner import Benchmark
from typing import Dict, List, Optional
from datasets import load_dataset
import re


# MMLU subject categories
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
    
    def __init__(self, split: str = "test", subjects: Optional[List[str]] = None):
        """
        Args:
            split: Dataset split to use ('auxiliary_train', 'dev', 'validation', or 'test')
            subjects: List of specific subjects to include (None = all subjects)
        """
        self.dataset = None
        self.split = split
        self.subjects = subjects or MMLU_SUBJECTS
        self.all_data = []
    
    def name(self) -> str:
        if len(self.subjects) < len(MMLU_SUBJECTS):
            return f"MMLU-{len(self.subjects)}subjects"
        return "MMLU"
    
    def load_dataset(self):
        """Load MMLU dataset from HuggingFace."""
        if self.dataset is None:
            # MMLU is structured by subject
            self.all_data = []
            
            for subject in self.subjects:
                try:
                    # Load specific subject
                    subject_data = load_dataset("cais/mmlu", subject, split=self.split)
                    
                    # Add subject label to each example
                    for example in subject_data:
                        example_dict = dict(example)
                        example_dict["subject"] = subject
                        self.all_data.append(example_dict)
                        
                except Exception as e:
                    print(f"Warning: Could not load subject '{subject}': {e}")
            
            if not self.all_data:
                raise ValueError(f"No data loaded for MMLU with subjects: {self.subjects}")
    
    def get_problem(self, index: int) -> Dict:
        """Get a problem by index."""
        return self.all_data[index]
    
    def get_num_problems(self) -> int:
        """Return total number of problems."""
        return len(self.all_data)
    
    def format_prompt(self, problem: Dict) -> str:
        """
        Format an MMLU problem into a prompt for the LLM.
        
        Presents the question with multiple choice options A-D.
        """
        question = problem.get("question", "")
        choices = problem.get("choices", [])
        subject = problem.get("subject", "").replace("_", " ").title()
        
        # Format choices as A, B, C, D
        choices_text = ""
        for i, choice in enumerate(choices):
            letter = chr(65 + i)  # 65 is ASCII for 'A'
            choices_text += f"{letter}. {choice}\n"
        
        prompt = f"""Subject: {subject}

Question: {question}

{choices_text}
Answer with only the letter (A, B, C, or D) of the correct answer."""
        
        return prompt
    
    def extract_completion(self, response: str, problem: Dict) -> str:
        """
        Extract the answer choice from LLM response.
        
        Looks for:
        - Single letter A-D
        - "Answer: A" pattern
        - "The answer is A" pattern
        """
        # Remove whitespace and convert to uppercase
        response_clean = response.strip().upper()
        
        # Try to find explicit answer patterns
        patterns = [
            r'ANSWER\s*IS\s*([A-D])',
            r'ANSWER\s*:\s*([A-D])',
            r'^\s*([A-D])\s*$',  # Just the letter
            r'OPTION\s*([A-D])',
            r'CHOICE\s*([A-D])',
            r'\(([A-D])\)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response_clean)
            if match:
                return match.group(1)
        
        # Look for first occurrence of A, B, C, or D in response
        match = re.search(r'[A-D]', response_clean)
        if match:
            return match.group(0)
        
        # If nothing found, return original response
        return response.strip()
    
    def _get_correct_answer_letter(self, problem: Dict) -> str:
        """Get the correct answer as a letter (A-D)."""
        answer_idx = problem.get("answer", 0)
        
        # Handle different answer formats
        if isinstance(answer_idx, str):
            # If it's already a letter, return it
            if answer_idx.upper() in ['A', 'B', 'C', 'D']:
                return answer_idx.upper()
            # If it's a number as string, convert to int
            try:
                answer_idx = int(answer_idx)
            except ValueError:
                return 'A'  # Default fallback
        
        # Convert index to letter (0->A, 1->B, 2->C, 3->D)
        if 0 <= answer_idx <= 3:
            return chr(65 + answer_idx)
        
        return 'A'  # Default fallback
    
    def check_correctness(self, problem: Dict, completion: str) -> tuple[bool, str]:
        """
        Check if the completion is correct.
        
        Compares the extracted answer letter with the ground truth.
        
        Returns: (passed, result_message)
        """
        try:
            expected = self._get_correct_answer_letter(problem)
            
            if not completion or not completion.strip():
                return False, "empty completion"
            
            # Normalize completion to single letter
            completion_clean = completion.strip().upper()
            if len(completion_clean) > 1:
                # Try to extract letter if there's extra text
                match = re.search(r'[A-D]', completion_clean)
                if match:
                    completion_clean = match.group(0)
            
            # Check if answer matches
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
            "subject": problem.get("subject", "unknown"),
            "predicted_answer": completion,
            "ground_truth": self._get_correct_answer_letter(problem),
            "choices": problem.get("choices", [])
        }


class MMLUSTEMBenchmark(MMLUBenchmark):
    """MMLU benchmark with only STEM subjects."""
    
    STEM_SUBJECTS = [
        "abstract_algebra", "anatomy", "astronomy", "college_biology", "college_chemistry",
        "college_computer_science", "college_mathematics", "college_physics", "computer_security",
        "conceptual_physics", "electrical_engineering", "elementary_mathematics", "high_school_biology",
        "high_school_chemistry", "high_school_computer_science", "high_school_mathematics",
        "high_school_physics", "high_school_statistics", "machine_learning"
    ]
    
    def __init__(self, split: str = "test"):
        super().__init__(split=split, subjects=self.STEM_SUBJECTS)
    
    def name(self) -> str:
        return "MMLU-STEM"


class MMLUHumanitiesBenchmark(MMLUBenchmark):
    """MMLU benchmark with only Humanities subjects."""
    
    HUMANITIES_SUBJECTS = [
        "formal_logic", "high_school_european_history", "high_school_us_history",
        "high_school_world_history", "international_law", "jurisprudence", "logical_fallacies",
        "moral_disputes", "moral_scenarios", "philosophy", "prehistory", "professional_law",
        "world_religions"
    ]
    
    def __init__(self, split: str = "test"):
        super().__init__(split=split, subjects=self.HUMANITIES_SUBJECTS)
    
    def name(self) -> str:
        return "MMLU-Humanities"


class MMLUSocialSciencesBenchmark(MMLUBenchmark):
    """MMLU benchmark with only Social Sciences subjects."""
    
    SOCIAL_SCIENCES_SUBJECTS = [
        "econometrics", "high_school_geography", "high_school_government_and_politics",
        "high_school_macroeconomics", "high_school_microeconomics", "high_school_psychology",
        "human_sexuality", "professional_psychology", "public_relations", "security_studies",
        "sociology", "us_foreign_policy"
    ]
    
    def __init__(self, split: str = "test"):
        super().__init__(split=split, subjects=self.SOCIAL_SCIENCES_SUBJECTS)
    
    def name(self) -> str:
        return "MMLU-SocialSciences"
