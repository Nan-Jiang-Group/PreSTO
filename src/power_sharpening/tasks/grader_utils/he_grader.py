"""
HumanEval grader.

Provides extract_code for pulling function bodies from LLM output, and entry_point for running functional correctness
evaluation.
"""
import textwrap
import re

from .he_check import evaluate_functional_correctness


def extract_code(text, entry_point):
    """Extract the code body for a given entry_point function from LLM output."""
    code_block_pattern = re.compile(
        rf"```(?:[Pp]ython\n)?.*?def\s+{entry_point}.*?:\n(.*?)\n```", re.DOTALL
    )
    code_block = code_block_pattern.search(text)
    if code_block is None:
        code_block_pattern = re.compile(
            rf"def\s+{entry_point}.*?:\n(.*?)(?:\n(?!\n*(?:  |\t))|$)", re.DOTALL
        )
        code_block = code_block_pattern.search(text)
    if code_block is None:
        code_block_pattern = re.compile(
            r"def.*?:\n(.*?)(?:\n(?!\n*(?:  |\t))|$)", re.DOTALL
        )
        code_block = code_block_pattern.search(text)

    if code_block is not None:
        return code_block.group(1)

    # if no code block is found, assume the LM is simply filling the code
    return textwrap.indent(text, " " * 4)


def entry_point(
    sample_file: str,
    k: str = "1,2,3,4,5,6,7,8",
    n_workers: int = 4,
    timeout: float = 3.0,
    problem_file: str = "HumanEval.jsonl",
):
    """
    Evaluates the functional correctness of generated samples, and writes results to f"{sample_file}_results.jsonl.gz"
    """
    k = list(map(int, k.split(",")))
    results = evaluate_functional_correctness(sample_file, k, n_workers, timeout, problem_file)
    print(results)
