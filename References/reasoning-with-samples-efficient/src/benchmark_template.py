"""
Template for adding new benchmarks.

To add a new benchmark:
1. Create a class that inherits from Benchmark
2. Implement all abstract methods
3. Add it to BENCHMARK_REGISTRY in run_benchmark.py

Example: SWEBench implementation
"""

from benchmark_runner import Benchmark
from typing import Dict
from datasets import load_dataset


class SWEBenchBenchmark(Benchmark):
    """
    SWEBench benchmark for code repository tasks.
    (This is a template - actual implementation would need real SWEBench logic)
    """
    
    def __init__(self):
        self.dataset = None
    
    def name(self) -> str:
        return "SWEBench"
    
    def load_dataset(self):
        """Load SWEBench dataset from HuggingFace or other source."""
        if self.dataset is None:
            # Example - adjust to actual dataset
            self.dataset = load_dataset("princeton-nlp/SWE-bench", split="test")
    
    def get_problem(self, index: int) -> Dict:
        """Get a problem by index."""
        return self.dataset[index]
    
    def get_num_problems(self) -> int:
        """Return total number of problems."""
        return len(self.dataset)
    
    def format_prompt(self, problem: Dict) -> str:
        """
        Format a problem into a prompt for the LLM.
        
        For SWEBench, this might include:
        - Issue description
        - Repository context
        - Instructions for generating a patch
        """
        issue = problem.get("problem_statement", "")
        repo = problem.get("repo", "")
        
        prompt = f"""You are working on the {repo} repository.

Issue:
{issue}

Please provide a code patch to fix this issue."""
        
        return prompt
    
    def extract_completion(self, response: str, problem: Dict) -> str:
        """
        Extract the completion from LLM response.
        
        For SWEBench, this might involve:
        - Extracting code patches
        - Parsing diff format
        - Extracting file modifications
        """
        # Example: extract code blocks
        import re
        code_blocks = re.findall(r'```(?:diff|python)?\n(.*?)```', response, re.DOTALL)
        if code_blocks:
            return code_blocks[0].strip()
        return response.strip()
    
    def check_correctness(self, problem: Dict, completion: str) -> tuple[bool, str]:
        """
        Check if the completion is correct.
        
        For SWEBench, this would involve:
        - Applying the patch
        - Running tests
        - Checking if the issue is resolved
        
        Returns: (passed, result_message)
        """
        # This is a placeholder - actual implementation would:
        # 1. Clone the repo
        # 2. Apply the patch
        # 3. Run tests
        # 4. Check test results
        
        # For now, just return a placeholder
        return False, "SWEBench evaluation not implemented"


class MATHBenchmark(Benchmark):
    """
    MATH benchmark for mathematical reasoning.
    (Template implementation)
    """
    
    def __init__(self):
        self.dataset = None
    
    def name(self) -> str:
        return "MATH"
    
    def load_dataset(self):
        if self.dataset is None:
            # Load MATH dataset
            self.dataset = load_dataset("hendrycks/competition_math", split="test")
    
    def get_problem(self, index: int) -> Dict:
        return self.dataset[index]
    
    def get_num_problems(self) -> int:
        return len(self.dataset)
    
    def format_prompt(self, problem: Dict) -> str:
        """Format math problem with instructions."""
        question = problem.get("problem", "")
        prompt = f"""Solve the following math problem. Put your final answer within \\boxed{{}}.

{question}"""
        return prompt
    
    def extract_completion(self, response: str, problem: Dict) -> str:
        """Extract answer from response."""
        import re
        # Look for boxed answer
        boxed = re.findall(r'\\boxed\{(.*?)\}', response)
        if boxed:
            return boxed[-1]  # Return last boxed answer
        return response.strip()
    
    def check_correctness(self, problem: Dict, completion: str) -> tuple[bool, str]:
        """Check if answer matches ground truth."""
        expected = problem.get("solution", "")
        # Extract expected answer from solution
        import re
        expected_boxed = re.findall(r'\\boxed\{(.*?)\}', expected)
        if expected_boxed:
            expected_answer = expected_boxed[-1]
            # Simple string comparison (could be more sophisticated)
            passed = completion.strip() == expected_answer.strip()
            return passed, "correct" if passed else f"expected {expected_answer}"
        return False, "could not extract expected answer"


# To add your benchmark:
# 1. Implement it here or in a separate file
# 2. Import it in run_benchmark.py
# 3. Add to BENCHMARK_REGISTRY:
#
#    BENCHMARK_REGISTRY = {
#        "humaneval": HumanEvalBenchmark,
#        "swebench": SWEBenchBenchmark,
#        "math": MATHBenchmark,
#    }
