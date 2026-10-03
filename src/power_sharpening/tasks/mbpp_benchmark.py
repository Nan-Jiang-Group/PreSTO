"""MBPP benchmark implementation."""

import json
import multiprocessing
import re
from pathlib import Path
from typing import Any, Dict

from torch.utils.data import DataLoader

from .base import Benchmark
from .grader_utils.he_execute import (
    create_tempdir,
    reliability_guard,
    swallow_io,
    time_limit,
)


_FEW_SHOT_EXAMPLES = [
    {
        "text": (
            "Write a function to find the similar elements from the given "
            "two tuple lists."
        ),
        "tests": [
            "assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)",
            "assert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)",
        ],
        "code": (
            "def similar_elements(test_tup1, test_tup2):\n"
            "    return tuple(set(test_tup1) & set(test_tup2))"
        ),
    },
    {
        "text": "Write a python function to identify non-prime numbers.",
        "tests": [
            "assert is_not_prime(2) == False",
            "assert is_not_prime(10) == True",
        ],
        "code": (
            "import math\n"
            "def is_not_prime(n):\n"
            "    return any(n % i == 0 for i in range(2, int(math.sqrt(n)) + 1))"
        ),
    },
]

_BASE_HEADER_TEMPLATE = (
    "You are an expert Python programmer, and here is your task: "
    "{text} Your code should pass these tests:\n\n{tests}\n"
)
_BASE_SHOT_TEMPLATE = _BASE_HEADER_TEMPLATE + "[BEGIN]\n{code}\n[DONE]\n\n"
_BASE_FINAL_TEMPLATE = _BASE_HEADER_TEMPLATE + "{name_hint}[BEGIN]\n"
_CHAT_TEMPLATE = (
    "You are an expert Python programmer. Solve the following task:\n\n"
    "{text}\n\n"
    "Your function must pass these tests:\n{tests}\n\n"
    "{name_hint}"
    "Respond with the function definition and any required imports wrapped "
    "in a single ```python``` code block. Do not include the tests."
)

_STOP_TOKENS = ("[DONE]", "\nclass ", "\nif __name__")
_CODE_FENCE_RE = re.compile(
    r"```(?:python|py)?[ \t]*\n(.*?)(?:\n```|\Z)",
    re.DOTALL | re.IGNORECASE,
)
_BRACKET_MARKER_RE = re.compile(
    r"\[\s*/\s*[A-Z]+\s*\]|\[\s*[A-Z]{2,}\s*\]"
)
_GRADE_TIMEOUT_SECONDS = 5.0
_MBPP_DATA_PATH = Path(__file__).resolve().parents[3] / "data" / "MBPP.jsonl"


def _canonical_function_name(test_list: list[str]) -> str | None:
    if not test_list:
        return None
    match = re.search(r"\b([A-Za-z_][A-Za-z_0-9]*)\s*\(", test_list[0])
    return match.group(1) if match else None


def _truncate_at_stops(text: str) -> str:
    cut = text or ""
    for stop in _STOP_TOKENS:
        index = cut.find(stop)
        if index >= 0:
            cut = cut[:index]
    marker = _BRACKET_MARKER_RE.search(cut)
    if marker:
        cut = cut[:marker.start()]
    return cut.strip()


def _extract_candidate_code(completion: str) -> str:
    if not completion:
        return ""

    text = completion
    begin = text.find("[BEGIN]")
    done = text.find("[DONE]")
    if begin >= 0 and (done < 0 or begin < done):
        text = text[begin + len("[BEGIN]"):]

    blocks = _CODE_FENCE_RE.findall(text)
    if blocks:
        for block in blocks:
            if re.search(
                r"^\s*(?:def|class|import|from)\s",
                block,
                re.MULTILINE,
            ):
                return _truncate_at_stops(block)
        return _truncate_at_stops(blocks[0])

    return _truncate_at_stops(text)


def _unsafe_execute(candidate_code, test_list, timeout, result_connection):
    with create_tempdir():
        import os
        import shutil

        rmtree = shutil.rmtree
        rmdir = os.rmdir
        chdir = os.chdir

        reliability_guard()
        check_program = candidate_code + "\n\n" + "\n".join(test_list)
        try:
            with swallow_io():
                with time_limit(timeout):
                    exec(check_program, {})
            result_connection.send("passed")
        except BaseException as exc:
            result_connection.send(f"failed: {exc}")
        finally:
            result_connection.close()

        shutil.rmtree = rmtree
        os.rmdir = rmdir
        os.chdir = chdir


def _check_candidate(
    candidate_code: str,
    test_list: list[str],
    timeout: float,
) -> tuple[bool, str]:
    result_connection, child_connection = multiprocessing.Pipe(duplex=False)
    process = multiprocessing.Process(
        target=_unsafe_execute,
        args=(candidate_code, list(test_list), timeout, child_connection),
    )
    process.start()
    child_connection.close()
    process.join(timeout=timeout + 1)
    if process.is_alive():
        process.kill()
        process.join()
    if not result_connection.poll():
        result_connection.close()
        return False, "timed out"
    result = result_connection.recv()
    result_connection.close()
    return result == "passed", result


class MBPPBenchmark(Benchmark):
    """Mostly Basic Python Problems sanitized test benchmark."""

    name = "MBPP"

    def __init__(self, batch_size):
        self.dataset = None
        self.load_dataset(batch_size=batch_size)
        self.seen = 0

    @property
    def size(self):
        """Total number of problems."""
        return len(self.dataset_loader.dataset)

    def should_use_chat_template(self, model_name: str) -> bool:
        from .constants import model_uses_chat_template

        self.use_chat_template = model_uses_chat_template(model_name)
        return self.use_chat_template

    def load_dataset(self, batch_size: int) -> DataLoader:
        if self.dataset is None:
            with _MBPP_DATA_PATH.open("r", encoding="utf-8") as handle:
                self.dataset = [
                    json.loads(line)
                    for line in handle
                    if line.strip()
                ]
            self.dataset_loader = DataLoader(
                self.dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                drop_last=False,
                collate_fn=lambda batch: batch,
            )

    def get_question_and_answer(
        self,
        batch,
        tokenizer: Any,
    ):
        if self.seen + len(batch) > self.size:
            batch = batch[:self.size - self.seen]

        input_texts = [
            self.format_prompt(
                problem,
                use_chat_template=self.use_chat_template,
                tokenizer=tokenizer,
            )
            for problem in batch
        ]
        self.seen += len(batch)
        return input_texts, batch

    def format_prompt(
        self,
        problem: Dict,
        use_chat_template: bool = False,
        tokenizer: Any | None = None,
    ) -> str:
        text = problem.get("text") or problem.get("prompt") or ""
        test_list = problem.get("test_list") or []
        tests = "\n".join(test_list)
        name = _canonical_function_name(test_list)

        if use_chat_template:
            if tokenizer is None:
                raise ValueError(
                    "tokenizer is required when use_chat_template=True"
                )
            name_hint = (
                f"The function must be named exactly `{name}`.\n\n"
                if name else ""
            )
            rendered = _CHAT_TEMPLATE.format(
                text=text,
                tests=tests,
                name_hint=name_hint,
            )
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": rendered}],
                tokenize=False,
                add_generation_prompt=True,
            )

        shots = "".join(
            _BASE_SHOT_TEMPLATE.format(
                text=example["text"],
                tests="\n".join(example["tests"]),
                code=example["code"],
            )
            for example in _FEW_SHOT_EXAMPLES
        )
        name_hint = (
            f"The function must be named exactly `{name}`.\n"
            if name else ""
        )
        return shots + _BASE_FINAL_TEMPLATE.format(
            text=text,
            tests=tests,
            name_hint=name_hint,
        )

    @staticmethod
    def extract_completion(response: str, problem: Dict) -> str:
        return _extract_candidate_code(response)

    def stop_tokens(self) -> list[str]:
        return list(_STOP_TOKENS)

    @staticmethod
    def check_correctness(
        problem: Dict,
        completion: str,
    ) -> tuple[bool, str]:
        test_list = problem.get("test_list") or []
        if not completion:
            return False, "empty completion"
        if not test_list:
            return False, "missing test cases"
        try:
            return _check_candidate(
                completion,
                test_list,
                _GRADE_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            return False, f"error during check: {exc}"
