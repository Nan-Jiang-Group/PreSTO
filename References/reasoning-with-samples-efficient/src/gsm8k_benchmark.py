"""
GSM8K benchmark implementation.

GSM8K (Grade School Math 8K) is a dataset of 8.5K high quality grade school math problems.
Each problem takes between 2 and 8 steps to solve, and solutions primarily involve performing
a sequence of elementary calculations using basic arithmetic operations.
"""

from benchmark_runner import Benchmark
from typing import Dict
from datasets import load_dataset
import re


class GSM8KBenchmark(Benchmark):
    """
    GSM8K benchmark for mathematical reasoning.
    
    Dataset: https://huggingface.co/datasets/openai/gsm8k
    - 7,473 training problems
    - 1,319 test problems
    
    Each problem includes:
    - question: The math problem statement
    - answer: Step-by-step solution ending with #### ANSWER
    """
    
    def __init__(self, split: str = "test"):
        """
        Args:
            split: Dataset split to use ('train' or 'test', default 'test')
        """
        self.dataset = None
        self.split = split
    
    def name(self) -> str:
        return "GSM8K"
    
    def load_dataset(self):
        """Load GSM8K dataset from HuggingFace."""
        if self.dataset is None:
            self.dataset = load_dataset("openai/gsm8k", "main", split=self.split)
    
    def get_problem(self, index: int) -> Dict:
        """Get a problem by index."""
        return self.dataset[index]
    
    def get_num_problems(self) -> int:
        """Return total number of problems."""
        return len(self.dataset)
    
    def format_prompt(self, problem: Dict) -> str:
        """
        Format a GSM8K problem into a prompt for the LLM.
        
        Instructs the model to:
        - Show step-by-step reasoning
        - Provide final answer after ####
        """
        question = problem.get("question", "")
        
        prompt = f"""Solve the following math problem step by step. Show your work and reasoning clearly.

Question: {question}

Provide your final numerical answer after #### at the end.

Format your response like this:
[Your step-by-step solution]
#### [Final numerical answer]"""
        
        return prompt
    
    def extract_completion(self, response: str, problem: Dict) -> str:
        """
        Extract the final answer from LLM response.
        
        Looks for the answer after #### marker.
        Also tries to extract from common answer patterns if no marker found.
        """
        # Try to find answer after #### marker
        match = re.search(r'####\s*([0-9,.]+)', response)
        if match:
            answer = match.group(1)
            # Remove commas from numbers
            answer = answer.replace(',', '')
            return answer.strip()
        
        # Try to find "The answer is X" pattern
        match = re.search(r'[Tt]he answer is\s*([0-9,.]+)', response)
        if match:
            answer = match.group(1).replace(',', '')
            return answer.strip()
        
        # Try to find "= X" at the end of lines
        lines = response.strip().split('\n')
        for line in reversed(lines):
            match = re.search(r'=\s*([0-9,.]+)\s*$', line)
            if match:
                answer = match.group(1).replace(',', '')
                return answer.strip()
        
        # Try to find last number in response
        numbers = re.findall(r'([0-9,.]+)', response)
        if numbers:
            answer = numbers[-1].replace(',', '')
            return answer.strip()
        
        return response.strip()
    
    def _extract_ground_truth(self, problem: Dict) -> str:
        """Extract the ground truth answer from the problem."""
        answer_text = problem.get("answer", "")
        
        # GSM8K answers are formatted as: "Step 1\nStep 2\n#### ANSWER"
        match = re.search(r'####\s*([0-9,.]+)', answer_text)
        if match:
            answer = match.group(1).replace(',', '')
            return answer.strip()
        
        # Fallback: try to find last number
        numbers = re.findall(r'([0-9,.]+)', answer_text)
        if numbers:
            answer = numbers[-1].replace(',', '')
            return answer.strip()
        
        return ""
    
    def check_correctness(self, problem: Dict, completion: str) -> tuple[bool, str]:
        """
        Check if the completion is correct.
        
        Compares the numerical answer extracted from completion
        with the ground truth answer from the dataset.
        
        Returns: (passed, result_message)
        """
        try:
            # Extract ground truth
            expected = self._extract_ground_truth(problem)
            
            if not expected:
                return False, "could not extract expected answer"
            
            if not completion or not completion.strip():
                return False, "empty completion"
            
            # Convert to floats for comparison (handles different formats)
            try:
                expected_num = float(expected)
                completion_num = float(completion)
                
                # Check if answers match (with small tolerance for floating point)
                passed = abs(expected_num - completion_num) < 1e-6
                
                if passed:
                    return True, f"correct: {completion}"
                else:
                    return False, f"incorrect: got {completion}, expected {expected}"
                    
            except ValueError:
                # If conversion fails, do string comparison
                passed = completion == expected
                if passed:
                    return True, f"correct: {completion}"
                else:
                    return False, f"incorrect: got '{completion}', expected '{expected}'"
                
        except Exception as e:
            return False, f"error during check: {str(e)}"
    
    def format_prediction(self, problem: Dict, completion: str) -> Dict:
        """Format prediction for evaluation."""
        return {
            "question": problem["question"],
            "predicted_answer": completion,
            "ground_truth": self._extract_ground_truth(problem)
        }


class GSM8KTrainBenchmark(GSM8KBenchmark):
    """Convenience class for GSM8K training set."""
    
    def __init__(self):
        super().__init__(split="train")
    
    def name(self) -> str:
        return "GSM8K-Train"
