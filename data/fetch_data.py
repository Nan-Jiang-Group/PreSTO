"""Download evaluation datasets into vllm_experiments/data/.

Handles the datasets we want to run end-to-end:

  - MATH500       : already shipped at data/MATH500.json (no-op).
  - HumanEval     : public, fetched from openai/openai_humaneval (test).
  - LiveCodeBench : public, fetched from livecodebench/code_generation_lite.
  - MBPP          : public, fetched from google-research-datasets/mbpp
                    (sanitized config, test split — 257 vetted problems).
  - Alpaca        : public, fetched from tatsu-lab/alpaca_eval (eval split,
                    805 prompts).
  - GPQA          : *gated* (Idavidrein/gpqa, gpqa_diamond split). The
                    script prints exact instructions but does not auto-download — it requires an HF token + license
                    acceptance.

Each dataset is written as JSON Lines (one JSON object per line) — except Alpaca, which is written as a single JSON list
(``ALPACA.json``) to match the upstream ``power_samp_alpaca.py`` loader. Files that already exist are left alone (use
``--force`` to overwrite).

Usage
-----
    python scripts/fetch_data.py                  # all public datasets
    python scripts/fetch_data.py humaneval  # subset
    python scripts/fetch_data.py --force          # re-download
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_DATA_DIR = _THIS_DIR.parent / "data"


def _write_jsonl(path: Path, rows) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def fetch_humaneval(force: bool = False) -> None:
    out_path = _DATA_DIR / "HumanEval.jsonl"
    if out_path.exists() and not force:
        print(f"[humaneval] already present at {out_path} (use --force to redo)")
        return
    print(f"[humaneval] downloading openai/openai_humaneval (test) -> {out_path}")
    from datasets import load_dataset

    ds = load_dataset("openai/openai_humaneval", split="test")
    n = _write_jsonl(out_path, ds)
    print(f"[humaneval] wrote {n} rows")


def fetch_math500(force: bool = False) -> None:
    out_path = _DATA_DIR / "MATH500.json"
    if out_path.exists() and not force:
        print(f"[math500] already present at {out_path}")
        return
    print(
        f"[math500] {out_path} not found. The repo ships this file; "
        "if you deleted it, restore via `git checkout -- "
        "vllm_experiments/data/MATH500.json`."
    )


_LCB_RELEASE_FILES = {
    "release_v1": ["test.jsonl"],
    "release_v2": ["test2.jsonl"],
    "release_v3": ["test3.jsonl"],
    "release_v4": [ "test4.jsonl"],
    "release_v6": [ "test6.jsonl",
    ],
}


def fetch_livecodebench(
    force: bool = False, version_tag: str = "release_v6",
) -> None:
    if version_tag not in _LCB_RELEASE_FILES:
        raise ValueError(
            f"unknown LiveCodeBench version_tag={version_tag!r}; "
            f"expected one of {sorted(_LCB_RELEASE_FILES)}"
        )
    version_suffix = version_tag.removeprefix("release_")  # e.g. "v6"
    out_path = _DATA_DIR / f"LiveCodeBench_{version_suffix}.jsonl"
    if out_path.exists() and not force:
        print(f"[livecodebench] already present at {out_path} (use --force to redo)")
        return
    # The lite repo ships a custom loader script (code_generation_lite.py) that `datasets>=3` refuses to execute.
    # Replicate the loader's behavior by downloading the raw per-release JSONL shards directly and concatenating them
    # line-by-line (matches the upstream _generate_examples implementation).
    from huggingface_hub import hf_hub_download

    shards = _LCB_RELEASE_FILES[version_tag]
    print(
        f"[livecodebench] downloading livecodebench/code_generation_lite "
        f"(version_tag={version_tag}, shards={shards}) -> {out_path}"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as out:
        for shard in shards:
            local = hf_hub_download(
                repo_id="livecodebench/code_generation_lite",
                filename=shard,
                repo_type="dataset",
            )
            with open(local, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    # Re-serialize to normalize formatting (one row per line).
                    out.write(json.dumps(json.loads(line), ensure_ascii=False) + "\n")
                    n += 1
    print(f"[livecodebench] wrote {n} rows")


def fetch_mbpp(force: bool = False) -> None:
    out_path = _DATA_DIR / "MBPP.jsonl"
    if out_path.exists() and not force:
        print(f"[mbpp] already present at {out_path} (use --force to redo)")
        return
    print(
        f"[mbpp] downloading google-research-datasets/mbpp "
        f"(sanitized, test) -> {out_path}"
    )
    from datasets import load_dataset

    # 'sanitized' = ~427 hand-vetted problems; 'test' split has 257 of those (the standard MBPP eval slice). The
    # tasks/mbpp.py adapter reads `text` (problem statement) and `test_list` (asserts).
    ds = load_dataset(
        "google-research-datasets/mbpp", "sanitized", split="test",
    )
    n = _write_jsonl(out_path, ds)
    print(f"[mbpp] wrote {n} rows")


def fetch_alpaca(force: bool = False) -> None:
    out_path = _DATA_DIR / "ALPACA.json"
    if out_path.exists() and not force:
        print(f"[alpaca] already present at {out_path} (use --force to redo)")
        return
    print(
        f"[alpaca] downloading tatsu-lab/alpaca_eval (eval) -> {out_path}"
    )

    # The HF dataset ``tatsu-lab/alpaca_eval`` is backed by a loading script, which newer ``datasets`` versions refuse
    # to execute. The underlying eval-set JSON is shipped alongside the script as a plain file in the repo, so we grab
    # it directly. 805 prompts, AlpacaEval 2.0 evaluation set. Saved as a JSON *list* (not JSONL) to match the upstream
    # loader in llm_experiments/power_samp_alpaca.py.
    from huggingface_hub import hf_hub_download

    cached = hf_hub_download(
        repo_id="tatsu-lab/alpaca_eval",
        filename="alpaca_eval.json",
        repo_type="dataset",
    )
    with open(cached, "r", encoding="utf-8") as handle:
        rows = json.load(handle)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)
    print(f"[alpaca] wrote {len(rows)} rows")


def fetch_mmlu(force: bool = False) -> None:
    out_path = _DATA_DIR / "MMLU.jsonl"
    if out_path.exists() and not force:
        print(f"[mmlu] already present at {out_path} (use --force to redo)")
        return
    print(f"[mmlu] downloading cais/mmlu (all, test) -> {out_path}")
    from datasets import load_dataset

    # The 'all' config bundles every one of the 57 subjects into a single split and carries a `subject` column, which is
    # exactly the shape the MMLUTask adapter expects: {question, choices: [4], answer: int, subject}. The test split is
    # the standard 14,042-question eval set.
    ds = load_dataset("cais/mmlu", "all", split="test")
    n = _write_jsonl(out_path, ds)
    print(f"[mmlu] wrote {n} rows")


def fetch_arc_challenge(force: bool = False) -> None:
    out_path = _DATA_DIR / "ARC_Challenge.jsonl"
    if out_path.exists() and not force:
        print(f"[arc_challenge] already present at {out_path} (use --force to redo)")
        return
    print(
        f"[arc_challenge] downloading allenai/ai2_arc "
        f"(ARC-Challenge, test) -> {out_path}"
    )
    from datasets import load_dataset

    # ARC-Challenge test split = 1,172 grade-school science MCQs. Rows ship as {id, question, choices: {text, label},
    # answerKey} — written verbatim; the ARCChallengeTask adapter normalizes labels at load time.
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    n = _write_jsonl(out_path, ds)
    print(f"[arc_challenge] wrote {n} rows")


def fetch_gpqa(force: bool = False) -> None:
    out_path = _DATA_DIR / "GPQA.jsonl"
    if out_path.exists() and not force:
        print(f"[gpqa] already present at {out_path} (use --force to redo)")
        return
    print(
        "[gpqa] cannot auto-download — Idavidrein/gpqa is a gated "
        "dataset. To populate it:\n"
        "  1. Visit https://huggingface.co/datasets/Idavidrein/gpqa "
        "and accept the license.\n"
        "  2. Authenticate this machine. Pick one:\n"
        "       hf auth login              # interactive (modern)\n"
        "       export HF_TOKEN=hf_xxxx    # env var\n"
        "       --token hf_xxxx            # pass directly to this script\n"
        "  3. Re-run this script with --include-gpqa."
    )


def fetch_gpqa_authenticated(
    force: bool = False, token: str | None = None,
) -> None:
    out_path = _DATA_DIR / "GPQA.jsonl"
    if out_path.exists() and not force:
        print(f"[gpqa] already present at {out_path} (use --force to redo)")
        return
    print(f"[gpqa] downloading Idavidrein/gpqa (gpqa_diamond, train) -> {out_path}")
    from datasets import load_dataset

    # gpqa only ships a 'train' split; that's the standard 198-row eval set.
    kwargs = {"token": token} if token else {}
    ds = load_dataset(
        "Idavidrein/gpqa", "gpqa_diamond", split="train", **kwargs,
    )
    n = _write_jsonl(out_path, ds)
    print(f"[gpqa] wrote {n} rows")


_DISPATCH = {
    "math500": fetch_math500,
    "humaneval": fetch_humaneval,
    "gpqa": fetch_gpqa,
    "livecodebench": fetch_livecodebench,
    "mbpp": fetch_mbpp,
    "alpaca": fetch_alpaca,
    "mmlu": fetch_mmlu,
    "arc_challenge": fetch_arc_challenge,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "datasets",
        nargs="*",
        choices=list(_DISPATCH.keys()) + [],
        help="Datasets to fetch (default: all public).",
    )
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if the output file exists.")
    parser.add_argument("--include-gpqa", action="store_true",
                        help="Attempt the gated GPQA download (requires "
                             "HF auth + accepted license).")
    parser.add_argument("--token", type=str, default=None,
                        help="HF token for gated datasets (overrides "
                             "HF_TOKEN / cached login).")
    parser.add_argument("--lcb-version", type=str, default="release_v6",
                        choices=sorted(_LCB_RELEASE_FILES),
                        help="LiveCodeBench release tag (default release_v6).")
    args = parser.parse_args()

    if args.datasets:
        targets = args.datasets
    else:
        # Default = all public, but GPQA only auto-downloads with the flag.
        targets = [
            "math500", "humaneval", "livecodebench",
            "mbpp", "alpaca", "mmlu", "arc_challenge", "gpqa",
        ]

    for name in targets:
        if name == "gpqa" and args.include_gpqa:
            fetch_gpqa_authenticated(force=args.force, token=args.token)
        elif name == "livecodebench":
            fetch_livecodebench(force=args.force, version_tag=args.lcb_version)
        else:
            _DISPATCH[name](force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
