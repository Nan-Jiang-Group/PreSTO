"""Check LiveCodeBench release selection (release_v6 only) without a GPU or network.

Run with:
    /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/.venv/bin/python -m pytest /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/tests/test_lcb_dataset_version.py
"""

import base64
import hashlib
import json
import pickle
import zlib

import pytest

from power_sharpening.tasks import lcb_benchmark as lcb


def write_release(directory, version, task_id):
    private = base64.b64encode(zlib.compress(pickle.dumps("[]"))).decode()
    row = {
        "question_id": task_id, "question_content": f"Problem {task_id}",
        "starter_code": "", "difficulty": "easy", "metadata": "{}",
        "private_test_cases": private, "public_test_cases": "[]",
    }
    path = directory / f"LiveCodeBench_{version}.jsonl"
    path.write_text(json.dumps(row) + "\n")
    return path


def test_v6_release_loads_its_own_problems(tmp_path, monkeypatch):
    monkeypatch.setattr(lcb, "_LCB_DATA_DIR", tmp_path)
    write_release(tmp_path, "v6", "v6-problem")
    benchmark = lcb.LiveCodeBenchBenchmark(1, version="release_v6")
    assert next(iter(benchmark.dataset_loader))["task_id"] == ["v6-problem"]
    assert benchmark.source_path.name == "LiveCodeBench_v6.jsonl"
    assert benchmark.source_sha256 == hashlib.sha256(benchmark.source_path.read_bytes()).hexdigest()


def test_missing_v6_names_the_fetch_command(tmp_path, monkeypatch):
    monkeypatch.setattr(lcb, "_LCB_DATA_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="--lcb-version release_v6"):
        lcb.LiveCodeBenchBenchmark(1, version="release_v6")


@pytest.mark.parametrize("version", ["release_v5", "release_unknown"])
def test_other_releases_are_rejected(tmp_path, monkeypatch, version):
    monkeypatch.setattr(lcb, "_LCB_DATA_DIR", tmp_path)
    with pytest.raises(ValueError, match="Unsupported LiveCodeBench version"):
        lcb.LiveCodeBenchBenchmark(1, version=version)
