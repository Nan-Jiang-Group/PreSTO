"""
Prompt formatting and answer parsing utilities.

Previously duplicated in:
  - baselines/power-sharpening-hf/math_utils.py
  - subtree-prefetching-sampler/math_utils.py
"""

# Prompt fragments for MATH problems. Previously imported from a top-level ``constants`` module that no longer exists;
# inlined here so this utility is self-contained.
PROMPT = "Can you solve the following math problem? "
BASE = " Put your final answer within \\boxed{{}}."
COT = " Please reason step by step, and put your final answer within \\boxed{{}}."


def format_prompt(question, model, tokenizer, cot=True):
    """Build a formatted prompt string, optionally applying a chat template."""
    # Build content string
    content_str = PROMPT + question + (COT if cot else BASE)

    # Models that don't need chat template
    if model in ["qwen", "qwen_math"]:
        return content_str

    # Models that need chat template
    if model in ["qwen_math_grpo", "phi_grpo", "phi", "tulu"]:
        answer_context = [{"role": "user", "content": content_str}]
        return tokenizer.apply_chat_template(
            answer_context,
            tokenize=False,
            add_generation_prompt=True,
        )

    return content_str


def remove_boxed(s):
    """Remove \\boxed{...} wrapper and return inner content."""
    left = "\\boxed{"
    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"
        return s[len(left) : -1]
    except:
        return None


def last_boxed_only(sample):
    """
    Given a (q, a) sample, filter the answers so that they only contain the last \\boxed{...} or \\fbox{...} element.
    """
    q, a = sample
    a = last_boxed_only_string(a)
    if a is None:
        return None
    return (q, a)


def last_boxed_only_string(string):
    """Find the last \\boxed{...} or \\fbox{...} in a string."""
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def parse_answer(input_str):
    """Extract the answer from the last \\boxed{} in the string."""
    return remove_boxed(last_boxed_only_string(input_str))
