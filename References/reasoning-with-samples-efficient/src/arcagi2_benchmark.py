"""
ARC-AGI-2 benchmark implementation.

ARC-AGI-2 (Abstraction and Reasoning Corpus 2) tests abstract reasoning capabilities.
Each task presents a few input-output grid pairs demonstrating a pattern, then asks
the model to predict the output for a new input grid.

Dataset: https://github.com/arcprize/ARC-AGI-2
- 1,000 training tasks
- 120 evaluation tasks

Evaluation is pixel-perfect: the predicted output grid must exactly match the expected grid.
"""

from benchmark_runner import Benchmark
from typing import Dict, List, Optional
import json
import os
import re
import subprocess
from pathlib import Path


class ARCAGI2Benchmark(Benchmark):
    """
    ARC-AGI-2 benchmark for abstract reasoning.

    Each task contains:
    - train: List of input/output demonstration pairs (typically 2-5)
    - test: List of input/output test pairs (typically 1)

    Grids are 2D lists of integers (0-9), where each integer represents a color.
    """

    # Default path to ARC-AGI-2 data
    DEFAULT_DATA_PATH = os.path.expanduser("~/.cache/arc-agi-2/data")
    REPO_URL = "https://github.com/arcprize/ARC-AGI-2.git"

    def __init__(self, split: str = "evaluation", data_path: Optional[str] = None):
        """
        Args:
            split: Dataset split to use ('training' or 'evaluation', default 'evaluation')
            data_path: Custom path to ARC-AGI-2 data directory
        """
        self.split = split
        self.data_path = data_path or self.DEFAULT_DATA_PATH
        self.tasks = None
        self.task_ids = None

    def name(self) -> str:
        suffix = "-train" if self.split == "training" else ""
        return f"ARC-AGI-2{suffix}"

    def _ensure_data_downloaded(self):
        """Download ARC-AGI-2 data from GitHub if not present."""
        data_dir = Path(self.data_path)
        split_dir = data_dir / self.split

        if split_dir.exists() and any(split_dir.glob("*.json")):
            return  # Data already exists

        # Clone the repository
        cache_dir = data_dir.parent
        cache_dir.mkdir(parents=True, exist_ok=True)

        repo_dir = cache_dir / "ARC-AGI-2"
        if not repo_dir.exists():
            print(f"Downloading ARC-AGI-2 dataset from {self.REPO_URL}...")
            subprocess.run(
                ["git", "clone", "--depth", "1", self.REPO_URL, str(repo_dir)],
                check=True
            )

        # Copy data to expected location
        src_data = repo_dir / "data"
        if src_data.exists():
            import shutil
            if data_dir.exists():
                shutil.rmtree(data_dir)
            shutil.copytree(src_data, data_dir)
            print(f"ARC-AGI-2 data installed to {data_dir}")

    def load_dataset(self):
        """Load ARC-AGI-2 tasks from JSON files."""
        if self.tasks is not None:
            return

        self._ensure_data_downloaded()

        split_dir = Path(self.data_path) / self.split
        if not split_dir.exists():
            raise FileNotFoundError(
                f"ARC-AGI-2 {self.split} data not found at {split_dir}. "
                f"Please ensure the dataset is downloaded."
            )

        self.tasks = {}
        self.task_ids = []

        for task_file in sorted(split_dir.glob("*.json")):
            task_id = task_file.stem
            with open(task_file, 'r') as f:
                task_data = json.load(f)
            self.tasks[task_id] = task_data
            self.task_ids.append(task_id)

        if not self.tasks:
            raise ValueError(f"No tasks found in {split_dir}")

        print(f"Loaded {len(self.tasks)} ARC-AGI-2 {self.split} tasks")

    def get_problem(self, index: int) -> Dict:
        """Get a task by index."""
        task_id = self.task_ids[index]
        task = self.tasks[task_id]
        return {
            "task_id": task_id,
            "train": task["train"],
            "test": task["test"]
        }

    def get_num_problems(self) -> int:
        """Return total number of tasks."""
        return len(self.task_ids)

    def _grid_to_string(self, grid: List[List[int]]) -> str:
        """Convert a grid to a human-readable string representation."""
        return "\n".join(" ".join(str(cell) for cell in row) for row in grid)

    def _grid_to_json(self, grid: List[List[int]]) -> str:
        """Convert a grid to JSON string."""
        return json.dumps(grid)

    def format_prompt(self, problem: Dict) -> str:
        """
        Format an ARC-AGI-2 task into a prompt for the LLM.

        Shows all training examples and asks for the test output.
        """
        task_id = problem["task_id"]
        train_examples = problem["train"]
        test_examples = problem["test"]

        prompt_parts = [
            "You are solving an ARC-AGI-2 abstract reasoning task.",
            "",
            "In this task, you will see several input-output grid pairs that demonstrate a pattern or transformation rule.",
            "Your job is to figure out the pattern and apply it to predict the output for a new input grid.",
            "",
            "Grids are 2D arrays of integers (0-9), where each integer represents a different color:",
            "0=black, 1=blue, 2=red, 3=green, 4=yellow, 5=gray, 6=magenta, 7=orange, 8=cyan, 9=brown",
            "",
            "=== TRAINING EXAMPLES ===",
            ""
        ]

        # Add training examples
        for i, example in enumerate(train_examples, 1):
            prompt_parts.append(f"Example {i}:")
            prompt_parts.append(f"Input ({len(example['input'])}x{len(example['input'][0])} grid):")
            prompt_parts.append(self._grid_to_string(example['input']))
            prompt_parts.append(f"\nOutput ({len(example['output'])}x{len(example['output'][0])} grid):")
            prompt_parts.append(self._grid_to_string(example['output']))
            prompt_parts.append("")

        # Add test input
        test_input = test_examples[0]['input']
        prompt_parts.extend([
            "=== TEST ===",
            "",
            f"Now apply the pattern to this new input ({len(test_input)}x{len(test_input[0])} grid):",
            self._grid_to_string(test_input),
            "",
            "Think through the pattern step by step:",
            "1. What transformation is being applied to the input to produce the output?",
            "2. What are the rules governing this transformation?",
            "3. Apply these rules to the test input.",
            "",
            "Provide your answer as a JSON 2D array. Format your final answer like this:",
            "```json",
            "[[row1], [row2], ...]",
            "```",
            "",
            "Your answer must be pixel-perfect - every cell must match exactly."
        ])

        return "\n".join(prompt_parts)

    def extract_completion(self, response: str, problem: Dict) -> str:
        """
        Extract the predicted grid from LLM response.

        Looks for JSON array format in the response.
        Returns the grid as a JSON string.
        """
        # Try to find JSON code block
        json_block_match = re.search(r'```(?:json)?\s*\n(\[[\s\S]*?\])\s*\n```', response)
        if json_block_match:
            return json_block_match.group(1).strip()

        # Try to find a JSON array (2D list) anywhere in the response
        # Look for patterns like [[0,1,2],[3,4,5]]
        array_match = re.search(r'(\[\s*\[[\d,\s\[\]]+\]\s*\])', response)
        if array_match:
            return array_match.group(1).strip()

        # Try to find the last occurrence of something that looks like a grid
        lines = response.strip().split('\n')
        for i in range(len(lines) - 1, -1, -1):
            line = lines[i].strip()
            if line.startswith('[[') and line.endswith(']]'):
                return line

        # Fallback: return the whole response
        return response.strip()

    def _parse_grid(self, completion: str) -> Optional[List[List[int]]]:
        """Parse a grid from completion string."""
        try:
            # Clean up the string
            cleaned = completion.strip()
            # Remove any trailing commas before ] (invalid JSON)
            cleaned = re.sub(r',\s*\]', ']', cleaned)

            grid = json.loads(cleaned)

            # Validate it's a 2D list of integers
            if not isinstance(grid, list):
                return None
            for row in grid:
                if not isinstance(row, list):
                    return None
                for cell in row:
                    if not isinstance(cell, int) or cell < 0 or cell > 9:
                        return None

            return grid
        except (json.JSONDecodeError, TypeError, ValueError):
            return None

    def check_correctness(self, problem: Dict, completion: str) -> tuple[bool, str]:
        """
        Check if the completion is correct.

        ARC-AGI-2 requires pixel-perfect matching - every cell must be correct.

        Returns: (passed, result_message)
        """
        # Parse the predicted grid
        predicted_grid = self._parse_grid(completion)
        if predicted_grid is None:
            return False, "failed to parse grid from completion"

        # Get expected output
        expected_grid = problem["test"][0]["output"]

        # Check dimensions
        if len(predicted_grid) != len(expected_grid):
            return False, f"wrong number of rows: got {len(predicted_grid)}, expected {len(expected_grid)}"

        for i, (pred_row, exp_row) in enumerate(zip(predicted_grid, expected_grid)):
            if len(pred_row) != len(exp_row):
                return False, f"wrong number of columns in row {i}: got {len(pred_row)}, expected {len(exp_row)}"

        # Check pixel-perfect match
        total_cells = 0
        correct_cells = 0
        wrong_cells = []

        for i, (pred_row, exp_row) in enumerate(zip(predicted_grid, expected_grid)):
            for j, (pred_val, exp_val) in enumerate(zip(pred_row, exp_row)):
                total_cells += 1
                if pred_val == exp_val:
                    correct_cells += 1
                else:
                    wrong_cells.append((i, j, pred_val, exp_val))

        if correct_cells == total_cells:
            return True, "pixel-perfect match"
        else:
            accuracy = correct_cells / total_cells * 100
            sample_errors = wrong_cells[:3]  # Show first 3 errors
            error_str = ", ".join(
                f"({r},{c}): got {p}, expected {e}"
                for r, c, p, e in sample_errors
            )
            return False, f"{correct_cells}/{total_cells} cells correct ({accuracy:.1f}%). Errors: {error_str}"

    def format_prediction(self, problem: Dict, completion: str) -> Dict:
        """Format prediction for evaluation."""
        predicted_grid = self._parse_grid(completion)
        return {
            "task_id": problem["task_id"],
            "predicted_output": predicted_grid,
            "expected_output": problem["test"][0]["output"],
            "raw_completion": completion
        }


class ARCAGI2TrainingBenchmark(ARCAGI2Benchmark):
    """Convenience class for ARC-AGI-2 training set."""

    def __init__(self, data_path: Optional[str] = None):
        super().__init__(split="training", data_path=data_path)

    def name(self) -> str:
        return "ARC-AGI-2-Train"
