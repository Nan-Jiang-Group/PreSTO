"""Check release selection without a GPU or network.

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


@pytest.mark.parametrize("version", ["v5", "v6"])
def test_requested_release_selects_its_own_problems(tmp_path, monkeypatch, version):
    monkeypatch.setattr(lcb, "_LCB_DATA_DIR", tmp_path)
    for release in ("v5", "v6"):
        write_release(tmp_path, release, f"{release}-problem")
    benchmark = lcb.LiveCodeBenchBenchmark(1, version=f"release_{version}")
    assert next(iter(benchmark.dataset_loader))["task_id"] == [f"{version}-problem"]
    assert benchmark.source_path.name == f"LiveCodeBench_{version}.jsonl"
    assert benchmark.source_sha256 == hashlib.sha256(benchmark.source_path.read_bytes()).hexdigest()


def test_missing_v6_never_falls_back_to_v5(tmp_path, monkeypatch):
    monkeypatch.setattr(lcb, "_LCB_DATA_DIR", tmp_path)
    write_release(tmp_path, "v5", "old-problem")
    with pytest.raises(FileNotFoundError, match="--lcb-version release_v6"):
        lcb.LiveCodeBenchBenchmark(1, version="release_v6")


def test_unknown_release_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(lcb, "_LCB_DATA_DIR", tmp_path)
    with pytest.raises(ValueError, match="Unsupported LiveCodeBench version"):
        lcb.LiveCodeBenchBenchmark(1, version="release_unknown")
