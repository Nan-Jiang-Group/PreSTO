#
# For licensing see accompanying LICENSE file. Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""
Part of the codes are from
https://github.com/NovaSky-AI/SkyThought/blob/main/skythought/tools/util/livecodebench/testing_util.py
"""

import ast
import base64
import builtins
import copy
import faulthandler
import io
import json
import multiprocessing
import pickle
import sys
import time
import zlib
from decimal import Decimal
from types import ModuleType
from typing import Dict
from unittest.mock import patch, mock_open
import numpy as np

BASE_IMPORTS = """from itertools import accumulate, chain, combinations, count, permutations, product, groupby, islice, repeat
from copy import deepcopy
from string import ascii_lowercase
from math import floor, log2, log10, sqrt, comb, gcd, ceil, inf, isqrt
from collections import defaultdict, deque, Counter
from bisect import bisect, bisect_left, bisect_right, insort
from heapq import heappush, heappop, heapify, merge
from functools import reduce, cache, lru_cache
from random import randrange, shuffle
from operator import itemgetter, sub
from re import search as re_search  # Assuming 're' refers to a regex search
from os.path import commonprefix
from typing import List, Tuple, Dict, Set, Optional, Union, Any, Callable, Iterable, Iterator, Generator
import copy
import string
import math
import collections
import bisect
import heapq
import functools
import random
import itertools
import operator
import re
import numpy as np
import pandas as pd
from math import log, prod  # 'log' and 'prod' are functions in the math module
from collections import deque, defaultdict, Counter, OrderedDict
from itertools import accumulate, permutations, combinations, product, groupby, islice, chain, repeat, zip_longest, cycle
from functools import lru_cache, reduce, partial
from operator import iand
import sys
"""


# Helper classes for stdin/stdout capture (from official LiveCodeBench testing_util.py)

class Capturing(list):
    """
    Context manager to capture stdout as a list. From official LiveCodeBench testing_util.py:59-70 From
    https://stackoverflow.com/a/16571630/6416660
    """
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = io.StringIO()
        # Make closing the StringIO a no-op
        self._stringio.close = lambda x: 1
        return self

    def __exit__(self, *args):
        self.append(self._stringio.getvalue())
        del self._stringio  # free up some memory
        sys.stdout = self._stdout


class MockBuffer:
    """
    Mock for sys.stdin.buffer with byte string support. From official LiveCodeBench testing_util.py:94-103
    """
    def __init__(self, inputs: str):
        self.inputs = inputs.encode("utf-8")  # Convert to bytes

    def read(self, *args):
        # Return as byte strings that can be split
        return self.inputs

    def readline(self, *args):
        return self.inputs.split(b"\n")[0] + b"\n"


class MockStdinWithBuffer:
    """
    Custom mock for sys.stdin that supports buffer attribute. From official LiveCodeBench testing_util.py:74-91
    """
    def __init__(self, inputs: str):
        self.inputs = inputs
        self._stringio = io.StringIO(inputs)
        self.buffer = MockBuffer(inputs)

    def read(self, *args):
        return self.inputs

    def readline(self, *args):
        return self._stringio.readline(*args)

    def readlines(self, *args):
        return self.inputs.split("\n")

    def __iter__(self):
        # Support `for line in sys.stdin` pattern
        return iter(self._stringio)

    def __next__(self):
        # Support `next(sys.stdin)` pattern
        return next(self._stringio)

    def __getattr__(self, name):
        # Delegate other attributes to StringIO
        return getattr(self._stringio, name)


def get_stripped_lines(val: str):
    """
    Strip the entire value and then strip each line individually. From official LiveCodeBench repo - ensures proper
    whitespace/newline handling.
    """
    ## you don't want empty lines to add empty list after splitlines!
    val = val.strip()
    return [val_line.strip() for val_line in val.split("\n")]


def convert_line_to_decimals(line: str) -> tuple:
    """
    Convert a line of space-separated values to Decimal objects for precise numeric comparison. From official
    LiveCodeBench repo testing_util.py:214-220 Used for stdio tests where outputs are strings.

    Returns:
        (success: bool, decimals: list[Decimal])
    """
    try:
        decimal_line = [Decimal(elem) for elem in line.split()]
    except:
        return False, []
    return True, decimal_line


def compare_strings_with_decimal_fallback(prediction_str: str, expected_str: str) -> bool:
    """
    Compare string outputs with Decimal fallback for numeric values. From official LiveCodeBench repo
    testing_util.py:372-423 (grade_stdio comparison logic)

    Returns True if outputs match (exact or via Decimal comparison).
    """
    # Use official's multi-stage stripping strategy
    stripped_prediction_lines = get_stripped_lines(prediction_str)
    stripped_expected_lines = get_stripped_lines(expected_str)

    # Check if line counts match
    if len(stripped_prediction_lines) != len(stripped_expected_lines):
        return False

    # Line-by-line comparison with exact match first, then decimal fallback
    for stripped_pred_line, stripped_exp_line in zip(stripped_prediction_lines, stripped_expected_lines):
        ## CASE 1: exact match
        if stripped_pred_line == stripped_exp_line:
            continue

        ## CASE 2: element-wise comparison using Decimal for precise float comparison
        success, decimal_pred_line = convert_line_to_decimals(stripped_pred_line)
        if not success:
            return False

        success, decimal_exp_line = convert_line_to_decimals(stripped_exp_line)
        if not success:
            return False

        if decimal_pred_line == decimal_exp_line:
            continue

        # If neither exact match nor decimal match worked, fail
        return False

    # All lines matched
    return True


def reliability_guard():
    """
    This disables various destructive functions and prevents the generated code
    from interfering with the test (e.g. fork bomb, killing other processes,
    removing filesystem files, etc.)

    WARNING
    This function is NOT a security sandbox. Untrusted code, including, model- generated code, should not be blindly
    executed outside of one. See the Codex paper for more information about OpenAI's code sandbox, and proceed with
    caution.
    """

    faulthandler.disable()

    import builtins

    builtins.exit = None
    builtins.quit = None

    import os

    os.environ["OMP_NUM_THREADS"] = "1"

    os.kill = None
    os.system = None
    os.putenv = None
    os.remove = None
    os.removedirs = None
    os.rmdir = None
    os.fchdir = None
    os.setuid = None
    os.fork = None
    os.forkpty = None
    os.killpg = None
    os.rename = None
    os.renames = None
    os.truncate = None
    os.replace = None
    os.unlink = None
    os.fchmod = None
    os.fchown = None
    os.chmod = None
    os.chown = None
    os.chroot = None
    os.fchdir = None
    os.lchflags = None
    os.lchmod = None
    os.lchown = None
    os.getcwd = None
    os.chdir = None

    import shutil

    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None

    import subprocess

    subprocess.Popen = None  # type: ignore

    import sys

    sys.modules["ipdb"] = None
    sys.modules["joblib"] = None
    sys.modules["resource"] = None
    sys.modules["psutil"] = None
    sys.modules["tkinter"] = None


def has_test_type(tests, type):  ## helper to select specific type of problems
    """
    Check if any test in the test list has 'testtype' set to 'type'.
    """
    test_list = json.loads(tests)
    for test in test_list:
        if test.get("testtype") == type:
            return True
    return False


def translate_private_test_cases(encoded_data):
    decoded_data = base64.b64decode(encoded_data)
    decompressed_data = zlib.decompress(decoded_data)
    original_data = pickle.loads(decompressed_data)
    test_cases = json.loads(original_data)
    return test_cases


def map_to_example(row):
    # Parse metadata JSON string to dict (matches official LiveCodeBench code_generation.py:76)
    metadata_raw = row.get("metadata", "{}")
    try:
        metadata = json.loads(metadata_raw) if isinstance(metadata_raw, str) else metadata_raw
    except json.JSONDecodeError:
        metadata = {}

    return {
        "prompt": row["question_content"],
        "test": row["private_test_cases"],
        "entry_point": row["starter_code"],
        "task_id": row["question_id"],
        "is_stdin": has_test_type(row["public_test_cases"], "stdin"),
        "public_test_cases": row["public_test_cases"],
        "difficulty": row["difficulty"],
        "metadata": metadata,
    }


def post_process_code(code):
    code = code.split("</code>")[0]
    code = code.replace("```python", "")
    code = code.split("```")[0]
    code = code.replace("<code>", "")
    return code


def parse_function_name_from_starter_code(starter_code):
    """
    Extract function name from starter code using AST parsing. Based on official LiveCodeBench implementation.

    Args:
        starter_code: Python code string containing function definition

    Returns:
        Function name string, or None if not found
    """
    try:
        # Handle incomplete starter code by adding a pass statement if needed LeetCode-style starter code often has
        # incomplete function definitions
        code_to_parse = starter_code
        if not code_to_parse.strip().endswith(("pass", "...", "return")):
            # Count indentation of last line to add proper pass statement
            lines = code_to_parse.rstrip().split('\n')
            if lines:
                last_line = lines[-1]
                # If last line ends with ':', add indented pass
                if last_line.rstrip().endswith(':'):
                    indent = len(last_line) - len(last_line.lstrip()) + 4
                    code_to_parse = code_to_parse + '\n' + ' ' * indent + 'pass'

        tree = ast.parse(code_to_parse)
        fn = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # For LeetCode-style problems, there should be exactly one function If there are multiple, we take the
                # last one found (which is typically the target)
                fn = node.name
        return fn
    except Exception:
        return None


def clean_if_name(code: str) -> str:
    """
    Remove 'if __name__ == "__main__":' wrapper from code using AST parsing. From official LiveCodeBench
    testing_util.py:106-119

    The runtime doesn't interact well with __name__ == '__main__', so we unwrap it. Extracts the code inside the if
    block and returns it without the if wrapper.
    """
    try:
        astree = ast.parse(code)
        last_block = astree.body[-1]
        if isinstance(last_block, ast.If):
            condition = last_block.test
            if ast.unparse(condition).strip() == "__name__ == '__main__'":
                code = (
                    ast.unparse(astree.body[:-1]) + "\n" + ast.unparse(last_block.body)  # type: ignore
                )
    except:
        pass

    return code


def make_function(code: str) -> str:
    """
    Wrap code inside a function for controlled execution. From official LiveCodeBench testing_util.py:122-151

    Separates imports from other statements and wraps non-import code in wrapped_function(). This allows us to call the
    code as a function with proper stdin mocking.
    """
    try:
        import_stmts = []
        all_other_stmts = []
        astree = ast.parse(code)
        for stmt in astree.body:
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                import_stmts.append(stmt)
            else:
                all_other_stmts.append(stmt)

        function_ast = ast.FunctionDef(
            name="wrapped_function",
            args=ast.arguments(
                posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]
            ),
            body=all_other_stmts,
            decorator_list=[],
            lineno=-1,
        )
        main_code = (
            BASE_IMPORTS
            + "\n"
            + ast.unparse(import_stmts)  # type: ignore
            + "\n"
            + ast.unparse(function_ast)  # type: ignore
        )
        return main_code
    except Exception as e:
        return code


def compile_code(code: str) -> ModuleType:
    """
    Compile code into a module. From official LiveCodeBench testing_util.py:192-211

    Note: Removed signal.alarm() calls since timeout is handled by parent process in lcb_run().
    """
    try:
        tmp_sol = ModuleType("tmp_sol", "")
        exec(code, tmp_sol.__dict__)
        if "class Solution" in code:
            # leetcode wraps solutions in `Solution` this is a hack to check if it is leetcode solution or not currently
            # livecodebench only supports LeetCode but else condition allows future extensibility to other platforms
            compiled_sol = tmp_sol.Solution()
        else:
            # do nothing in the other case since function is accesible
            compiled_sol = tmp_sol

        assert compiled_sol is not None
        return compiled_sol
    except Exception:
        return None


def get_function(compiled_sol, fn_name: str):
    """
    Safely extract function from compiled module. From official LiveCodeBench testing_util.py:184-189
    """
    try:
        assert hasattr(compiled_sol, fn_name)
        return getattr(compiled_sol, fn_name)
    except Exception as e:
        return None


def call_method(method, inputs):
    """
    Call method with comprehensive stdin mocking. From official LiveCodeBench testing_util.py:154-181

    Provides full stdin support including read(), readline(), readlines(), and buffer.
    """
    if isinstance(inputs, list):
        inputs = "\n".join(inputs)

    inputs_line_iterator = iter(inputs.split("\n"))

    # Create custom stdin mock with buffer support
    mock_stdin = MockStdinWithBuffer(inputs)

    @patch("builtins.open", mock_open(read_data=inputs))
    @patch("sys.stdin", mock_stdin)  # Use our custom mock instead of StringIO
    @patch("sys.stdin.readline", lambda *args: next(inputs_line_iterator))
    @patch("sys.stdin.readlines", lambda *args: inputs.split("\n"))
    @patch("sys.stdin.read", lambda *args: inputs)
    def _inner_call_method(_method):
        try:
            return _method()
        except SystemExit as e:
            pass
        finally:
            pass

    return _inner_call_method(method)


def prepare_test_input_output_std(test_case):
    """
    Prepare test input/output for stdin-based tests.

    From official LiveCodeBench testing_util.py:310-425 (grade_stdio flow). The strip() on output is critical for proper
    comparison - see compare_strings_with_decimal_fallback().

    Args:
        test_case: Dict with "input" (str) and "output" (str) keys

    Returns:
        (test_input: str, test_output: str) tuple
    """
    test_input = test_case["input"]
    test_output = test_case["output"].strip()
    return test_input, test_output


def run_test_func(completion, is_extracted, test_input, test_output, func_name):
    """
    Run function-based test with unified comparison logic.

    Difference from run_test_std: Only HOW prediction is obtained (function call vs stdio) Comparison logic: Same as
    run_test_std (string comparison with Decimal fallback)

    Args:
        completion: The code to execute
        is_extracted: Whether inputs are already extracted/parsed
        test_input: Test input data
        test_output: Expected output
        func_name: Name of the function to call (required)
    """
    assert func_name is not None, "func_name must be provided"

    namespace = {}
    exec(completion, namespace)

    # Detect if completion is class-based (e.g., class Solution:)
    is_class_based = "class Solution:" in completion or "class Solution(" in completion

    output = io.StringIO()
    sys.stdout = output

    try:
        # Get the callable (either function or method)
        if is_class_based:
            solution_instance = namespace["Solution"]()
            callable_func = getattr(solution_instance, func_name)
        else:
            callable_func = namespace[func_name]

        # Call the function/method with appropriate arguments
        if not is_extracted:
            if isinstance(test_input, dict):
                prediction = callable_func(**test_input)
            else:
                prediction = callable_func(test_input)
        else:
            prediction = callable_func(*test_input)

        # Don't penalize model if it produces tuples instead of lists
        if isinstance(prediction, tuple):
            prediction = list(prediction)

        # Convert both prediction and expected to strings for unified comparison
        prediction_str = str(prediction) if not isinstance(prediction, str) else prediction
        expected_str = str(test_output) if not isinstance(test_output, str) else test_output

        # Use unified comparison logic (same as run_test_std)
        if compare_strings_with_decimal_fallback(prediction_str, expected_str):
            return True, prediction
        else:
            return False, prediction

    except Exception as e:
        error_msg = f"Error: {str(e)}" if not is_extracted else str(e)
        return False, error_msg

    finally:
        sys.stdout = sys.__stdout__


def run_test_std(completion, test_input, test_output):
    """
    Run stdin-based test using official LiveCodeBench approach. Based on testing_util.py:310-425 (grade_stdio)

    Uses AST-based code transformation and comprehensive stdin mocking.
    """
    # Clean if __name__ == "__main__" wrapper
    completion = clean_if_name(completion)

    # Wrap code in function for controlled execution
    completion = make_function(completion)

    # Compile code into module
    compiled_sol = compile_code(completion)
    if compiled_sol is None:
        return False, "Compilation failed"

    # Get wrapped function
    method = get_function(compiled_sol, "wrapped_function")
    if method is None:
        return False, "Could not find wrapped_function"

    # Execute with captured stdout
    with Capturing() as captured_output:
        try:
            call_method(method, test_input)
        except Exception as e:
            return False, f"Runtime error: {e}"

    prediction = captured_output[0] if captured_output else ""

    # Use unified comparison logic (same as official testing_util.py:372-423)
    if compare_strings_with_decimal_fallback(prediction, test_output):
        return True, prediction.strip()
    else:
        return False, prediction.strip()


def prepare_test_input_output_functional(test_case, is_extracted):
    if not is_extracted:
        # Extract input and expected output from JSON directly
        test_input = test_case["input"]
        test_output = test_case["output"]
        return test_input, test_output
    else:
        # Robustly process complex inputs
        input_str = test_case["input"]
        expected_output = test_case["output"].strip()
        inputs = []

        if "=" in input_str:
            parts = input_str.split(",") if "," in input_str else [input_str]
            for part in parts:
                key, value = map(str.strip, part.split("="))
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        value = value.strip('"')
                inputs.append(value)
        else:
            for line in input_str.split("\n"):
                line = line.strip()
                if not line:
                    continue
                if line.startswith('"') and line.endswith('"'):
                    inputs.append(line.strip('"'))
                    continue
                if line.startswith("[") and line.endswith("]"):
                    inputs.append(json.loads(line))
                    continue
                try:
                    inputs.append(int(line))
                except ValueError:
                    try:
                        inputs.append(float(line))
                    except ValueError:
                        inputs.append(line)

        try:
            expected_output = json.loads(expected_output)
        except json.JSONDecodeError:
            expected_output = expected_output.strip()
        return inputs, expected_output


def run_tests_for_one_example(problem, test_cases, completion, result_list, is_extracted):
    time_elapsed = float("inf")
    test_type = test_cases[0]["testtype"]
    reliability_guard()
    completion = BASE_IMPORTS + "\n" + completion

    # Extract function name from metadata or parse from starter code
    func_name = None
    if test_type == "functional":
        # Try to get func_name from metadata first
        metadata = problem.get("metadata", {})
        func_name = metadata.get("func_name")

        # If not in metadata, parse from starter code
        if not func_name and "entry_point" in problem:
            func_name = parse_function_name_from_starter_code(problem["entry_point"])

    for i, test_case in enumerate(test_cases):
        output_error = ""
        output_value = ""
        try:
            time_start = time.time()
            if test_type == "functional":
                test_input, test_output = prepare_test_input_output_functional(test_case, is_extracted)
                passed, output_value = run_test_func(
                    completion, is_extracted, copy.deepcopy(test_input), copy.deepcopy(test_output), func_name
                )
            else:
                test_input, test_output = prepare_test_input_output_std(test_case)
                passed, output_value = run_test_std(completion, copy.deepcopy(test_input), copy.deepcopy(test_output))
            time_elapsed = time.time() - time_start
            if not passed:
                output_error = (
                    f"For test input: {test_input}. Expected output is: {test_output}, but got: {output_value}."
                )

        except Exception as e:
            passed = False
            output_error = f"For test input: {test_input}. Expected output is: {test_output}, but got error: {e}."
            output_value = f"Error: {e}."
        if output_error == "":
            output_error = f"For test input: {test_input}. Expected output is: {test_output}, your solution correctly passes this test with output {output_value}."
        result_list.append((passed, output_error, output_value, time_elapsed))
        if not passed:
            return


def lcb_run(problem, completion, timeout, is_extracted):
    test_cases = problem["test"]
    manager = multiprocessing.Manager()
    result = manager.list()
    p = multiprocessing.Process(target=run_tests_for_one_example, args=(problem, test_cases, completion, result, is_extracted))
    p.start()
    p.join(timeout=(timeout + 1) * len(test_cases) + 5)
    if p.is_alive():
        p.kill()

    # if len(result) < len(test_cases): failed due to timeout
    for i in range(len(test_cases) - len(result)):
        result.append((False, f"Time out!.", "Error: Time out!", float("inf")))
    return result


def estimate_pass_at_k(num_samples, num_correct, k):
    """Estimates pass@k of each problem and returns them in an array."""

    def estimator(n: int, c: int, k: int) -> float:
        """Calculates 1 - comb(n - c, k) / comb(n, k)."""
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    import itertools

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array(
        [estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)]
    )


def compute_metrics_from_results(results, k_list=[1, 5]):
    total = []
    correct = []
    task_ids = []
    for task_id, res in results.items():
        all_correct = []
        for generation in res:
            gen = np.array(generation)
            all_correct.append(np.all(gen > 0))
        task_ids.append(task_id)
        total.append(len(all_correct))
        correct.append(sum(all_correct))
    total = np.array(total)
    correct = np.array(correct)
    ks = k_list
    detail_pass_at_k = {
        f"pass@{k}": estimate_pass_at_k(total, correct, k).tolist()
        for k in ks
        if (total >= k).all()
    }
    pass_at_k = {
        f"pass@{k}": estimate_pass_at_k(total, correct, k).mean()
        for k in ks
        if (total >= k).all()
    }
    detail_metrics = {k: dict(zip(task_ids, v)) for k, v in detail_pass_at_k.items()}
    pass_at_k["detail"] = detail_metrics
    return pass_at_k