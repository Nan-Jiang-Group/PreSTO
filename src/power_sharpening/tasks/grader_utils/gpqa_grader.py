"""
GPQA multiple-choice grader.

Provides parse_answer_gpqa for extracting A/B/C/D from model output, and re-exports grade_answer from math_grader for
numeric/symbolic grading.
"""
import re

# from math500.math_grader import grade_answer  # noqa: F401


def parse_answer_gpqa(pred: str) -> str:
    """Extract a single A/B/C/D letter answer from model output."""
    pred = pred.replace("\u043a\u0438", "")
    pred = pred.strip("\n").rstrip(".").rstrip("/").strip(" ").lstrip(":")
    # Clean the answer based on the dataset
    tmp = re.findall(r"\b(A|B|C|D)\b", pred.upper())
    if tmp:
        pred = tmp
    else:
        pred = [pred.strip().strip(".")]
    pred = pred[-1]
    # Remove the period at the end, again!
    pred = pred.rstrip(".").rstrip("/")
    return pred
