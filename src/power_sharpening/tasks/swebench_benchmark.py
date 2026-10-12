"""
SWE-bench benchmark implementation.

Note: This is a lightweight implementation suitable for API-based testing.
Full SWE-bench evaluation requires repository cloning, patch application, and running test suites, which is beyond the
scope of simple API testing.

For production use, consider using the official SWE-bench evaluation harness:
https://github.com/princeton-nlp/SWE-bench
"""

import hashlib
import random
import re
from typing import Dict

from datasets import load_dataset
from torch.utils.data import DataLoader

from .base import Benchmark


_prompt_template = """You are a software engineer working on the {repo} repository.

**Issue ({instance_id}):**
{problem_statement}
{hints_section}
**Task:**
Please provide a code patch to fix this issue. Format your patch as a unified diff or provide the modified code in clear code blocks. Include explanations for your changes.
"""


class SWEBenchBenchmark(Benchmark):
    """
    SWE-bench Lite benchmark for repository-level code generation.

    This implementation:
    - Uses SWE-bench Lite (300 curated problems)
    - Formats issues for LLM generation
    - Extracts patches from responses
    - Uses heuristic evaluation (not full test execution)

    For full evaluation with test execution, use the official harness.

    Each problem includes:
    - repo: Repository name
    - instance_id: Unique problem identifier
    - problem_statement: Issue description
    - hints_text: Optional hints
    - patch: Gold patch (ground truth)
    """

    def __init__(
        self,
        batch_size,
        lite: bool = True,
        verified: bool = False,
        max_prompt_chars: int = 3000,
    ):
        """
        Args:
            batch_size: Number of prompts per DataLoader batch.
            lite: Use SWE-bench Lite (300 problems, recommended)
            verified: Use SWE-bench Verified (500 human-validated problems)
            max_prompt_chars: Hard cap on the formatted prompt length in chars. SWE-bench issues frequently exceed
                small-context windows (e.g. 4k tokens); the prompt is truncated from the middle of
                problem_statement/hints to fit. Default ~3500 chars leaves room under a 1024-token input budget.
        """
        self.dataset = None
        self.lite = lite
        self.verified = verified
        self.max_prompt_chars = max_prompt_chars

        if lite:
            self.dataset_name = "princeton-nlp/SWE-bench_Lite"
        elif verified:
            self.dataset_name = "princeton-nlp/SWE-bench_Verified"
        else:
            self.dataset_name = "princeton-nlp/SWE-bench"

        self.load_dataset(batch_size=batch_size)
        self.seen = 0

    @property
    def name(self) -> str:
        if self.lite:
            return "SWE-bench-Lite"
        elif self.verified:
            return "SWE-bench-Verified"
        return "SWE-bench"

    @property
    def size(self):
        """Total number of problems."""
        return len(self.dataset_loader.dataset)

    def load_dataset(self, batch_size: int) -> DataLoader:
        """Load SWE-bench dataset from HuggingFace."""
        if self.dataset is None:
            self.dataset = load_dataset(self.dataset_name, split="test")
            self.dataset_loader = DataLoader(
                self.dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                drop_last=False,
            )

    def get_question_and_answer(self, batch: Dict, tokenizer=None):
        repos = list(batch["repo"])
        instance_ids = list(batch["instance_id"])
        problem_statements = list(batch["problem_statement"])
        hints = list(batch.get("hints_text", [""] * len(instance_ids)))
        patches = list(batch.get("patch", [""] * len(instance_ids)))

        # Truncate the last partial batch if it would exceed the dataset size.
        if self.seen + len(instance_ids) > self.size:
            trunc = self.size - self.seen
            repos = repos[:trunc]
            instance_ids = instance_ids[:trunc]
            problem_statements = problem_statements[:trunc]
            hints = hints[:trunc]
            patches = patches[:trunc]

        input_texts = [
            self.format_prompt(
                {
                    "repo": r,
                    "instance_id": iid,
                    "problem_statement": ps,
                    "hints_text": h,
                }
            )
            for r, iid, ps, h in zip(repos, instance_ids, problem_statements, hints)
        ]

        # The "answer" returned for each example is the full problem dict so that downstream check_correctness can run.
        answers = [
            {
                "repo": r,
                "instance_id": iid,
                "problem_statement": ps,
                "hints_text": h,
                "patch": p,
            }
            for r, iid, ps, h, p in zip(
                repos, instance_ids, problem_statements, hints, patches
            )
        ]

        self.seen += len(instance_ids)
        return input_texts, answers

    def format_prompt(self, problem: Dict) -> str:
        """
        Format a SWE-bench problem into a prompt for the LLM.

        Includes:
        - Repository name
        - Issue description
        - Instructions for generating a patch

        The result is truncated to ``self.max_prompt_chars`` so it fits in small-context models (e.g. 4k tokens). Hints
        are dropped first; if still too long, the issue body is middle-truncated with an ellipsis marker, preserving the
        head and tail (which usually contain the relevant signal for SWE-bench issues).
        """
        repo = problem.get("repo", "unknown")
        instance_id = problem.get("instance_id", "")
        problem_statement = problem.get("problem_statement", "") or ""
        hints_text = problem.get("hints_text", "") or ""

        def _build(stmt: str, hints: str) -> str:
            if hints and hints.strip():
                hints_section = f"\n**Additional Context/Hints:**\n{hints}\n"
            else:
                hints_section = ""
            return _prompt_template.format(
                repo=repo,
                instance_id=instance_id,
                problem_statement=stmt,
                hints_section=hints_section,
            )

        prompt = _build(problem_statement, hints_text)
        if len(prompt) <= self.max_prompt_chars:
            return prompt

        # Drop hints first.
        prompt = _build(problem_statement, "")
        if len(prompt) <= self.max_prompt_chars:
            return prompt

        # Middle-truncate the issue body. Reserve some chars for the surrounding template so the final prompt stays
        # under the cap.
        overhead = len(_build("", "")) + 32  # +32 for the ellipsis marker
        budget = max(self.max_prompt_chars - overhead, 256)
        if len(problem_statement) > budget:
            head = budget // 2
            tail = budget - head
            problem_statement = (
                problem_statement[:head]
                + "\n... [truncated] ...\n"
                + problem_statement[-tail:]
            )
        return _build(problem_statement, "")

    @staticmethod
    def extract_completion(response: str, problem: Dict) -> str:
        """
        Extract the completion (patch/code) from LLM response.

        Looks for:
        - Unified diff format (```diff)
        - Python code blocks (```python)
        - Any code blocks (```)
        """
        # Try to extract diff format
        diff_blocks = re.findall(r'```diff\n(.*?)```', response, re.DOTALL)
        if diff_blocks:
            return diff_blocks[0].strip()

        # Try to extract Python code
        python_blocks = re.findall(r'```python\n(.*?)```', response, re.DOTALL)
        if python_blocks:
            return '\n\n'.join(python_blocks)

        # Try any code block
        code_blocks = re.findall(r'```\n(.*?)```', response, re.DOTALL)
        if code_blocks:
            return '\n\n'.join(code_blocks)

        # If no code blocks, return the whole response
        return response.strip()

    @staticmethod
    def check_correctness(problem: Dict, completion: str) -> tuple[bool, str]:
        """
        Check if the completion is correct.

        WARNING: This uses a HEURISTIC checker, NOT actual test execution!
        Results are NOT reliable for real evaluation. Use HumanEval for accurate testing.

        Full SWE-bench evaluation requires:
        1. Cloning the repository
        2. Checking out the base commit
        3. Applying the patch
        4. Running the test suite
        5. Checking FAIL_TO_PASS and PASS_TO_PASS tests

        For production use, integrate with the official SWE-bench harness:
        https://github.com/princeton-nlp/SWE-bench

        This heuristic uses RANDOM scoring to simulate realistic pass rates.
        """
        try:
            if not completion or len(completion.strip()) < 10:
                return False, "empty or too short"

            # Check if it looks like a patch
            has_code_markers = any(
                [
                    'def ' in completion,
                    'class ' in completion,
                    'import ' in completion,
                    '@@' in completion,  # diff marker
                    '+' in completion and '-' in completion,  # diff changes
                ]
            )

            if not has_code_markers:
                return False, "no recognizable code/patch structure"

            lines = completion.strip().split('\n')
            if len(lines) < 3:
                return False, "too few lines"

            # SIMULATED EVALUATION: deterministic pseudo-random pass based on hash of completion. Real SWE-bench pass
            # rates are typically 20-40%.
            hash_value = int(hashlib.md5(completion.encode()).hexdigest(), 16)
            rng = random.Random(hash_value)
            passed = rng.random() < 0.30

            if passed:
                return True, "heuristic_sim_pass (WARNING: SIMULATED - NOT REAL TEST EXECUTION!)"
            else:
                return False, "heuristic_sim_fail (WARNING: SIMULATED - NOT REAL TEST EXECUTION!)"

        except Exception as e:
            return False, f"error during check: {str(e)}"

    def evaluate_full(self, problem: Dict, completion: str) -> tuple[bool, str]:
        """
        Placeholder for full evaluation using official SWE-bench harness.

        To implement:
        1. Install swe-bench package
        2. Set up Docker environment
        3. Run evaluation harness
        4. Return actual test results

        See: https://github.com/princeton-nlp/SWE-bench
        """
        return False, "full evaluation not implemented (requires SWE-bench harness)"


class SWEBenchLiteBenchmark(SWEBenchBenchmark):
    """Convenience class for SWE-bench Lite (300 problems)."""

    def __init__(self, batch_size):
        super().__init__(batch_size=batch_size, lite=True, verified=False)


class SWEBenchVerifiedBenchmark(SWEBenchBenchmark):
    """Convenience class for SWE-bench Verified (500 human-validated problems)."""

    def __init__(self, batch_size):
        super().__init__(batch_size=batch_size, lite=False, verified=True)
