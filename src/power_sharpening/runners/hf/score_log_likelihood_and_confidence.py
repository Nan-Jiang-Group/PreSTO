"""Re-score saved completions: base-model log-likelihood and confidence under p_0.

For each response x = (x_1, ..., x_n) in a result CSV, with prompt x_0 and p_t(u) = p_0(u | x_0, x_{<t}):

    LL(x)   = (1/n) sum_t log p_t(x_t)
    Conf(x) = (1/n) sum_t sum_{u in V} p_t(u) log p_t(u)      (full vocabulary, temperature 1)

The PreSTO (subtree-prefetching) runner logs both, but the PowerMH runners only save the decoded ``completion``, so
this script scores any result CSV offline with one teacher-forced forward pass per response
(``compute_log_likelihood_and_confidence``). Scoring every method with the same scorer keeps them comparable; rows that
already carry logged values are compared against the rescored ones as a check on the tokenization round trip.

Each run's prompts are rebuilt in the style its own log records ("prompt style for <model>: chat template|raw"). vLLM
PreSTO logs made before the PreSTO runners adopted ``benchmark.should_use_chat_template`` carry no such line and used
raw prompts; SGLang logs made before the SGLang runners logged that line, and runs with no log at all, follow the
current policy, which those runners always applied. The logs also record each prompt's token length (``context_len``
of the first proposal, which spans the whole horizon), and the rebuilt prompt lengths must appear among them in order.
CSV row i is dataset example i, as every runner writes rows in dataset order. SGLang CSVs carry no ``max_new_tokens``
column; the horizon then comes from the resolved configuration in the log.
The response tokens are the re-encoded ``completion`` text, which the samplers decoded with skip_special_tokens=True,
so special tokens are lost from the text:

* PreSTO stops on EOS and counts that EOS in its response, so a PreSTO response shorter than ``max_new_tokens`` gets
  its EOS appended back (``appended_eos``).
* vLLM and HF PowerMH run with ignore_eos over a fixed ``max_new_tokens`` horizon. A chain state that passed through EOS
  continued into a new chat turn whose special tokens were stripped from the text, so its re-encoded response is
  shorter than the horizon and is not the token sequence the chain sampled. Such rows are kept but marked
  ``round_trip_exact=False``.
* SGLang PowerMH and PreSTO runs made before 2026-10-05 decoded the prompt together with the response, so their
  ``completion`` starts with the prompt text; it is stripped before re-encoding.
* SGLang PowerMH truncates its final chain state after the first EOS, so a short response gets its EOS appended back
  as for PreSTO. Its log records no response token count to confirm the rest of the round trip, so only rows that fill
  the horizon are marked exact.

Writes <input stem>.rescored.csv beside each input CSV. Needs a GPU:

    srun -p $PARTITION --quotatype=$QUOTATYPE --gres=gpu:1 --cpus-per-task=24 --time=03:00:00 \
      python -m power_sharpening.runners.hf.score_log_likelihood_and_confidence \
        --dataset lcb_v6 --model_str qwen3.5-9b --csv /abs/path/run1.csv /abs/path/run2.csv
"""

import argparse
import csv
import logging
import math
import re
from pathlib import Path
from statistics import fmean, stdev
from types import SimpleNamespace

import torch
import transformers

from power_sharpening.backends.hf.compute_log_likelihood_and_confidence import (
    compute_log_likelihood_and_confidence,
)
from power_sharpening.config import add_config_selection_args, resolve_config
from power_sharpening.tasks.constants import MODEL_MAP
from power_sharpening.tasks.registry import build_benchmark, set_random_seed

logger = logging.getLogger("[score log-likelihood and confidence]")

LOGGED_COLUMNS = ("num_response_tokens", "log_likelihood", "confidence")
# Runner methods whose samplers stop on the first EOS and keep it in the response, on every backend.
EOS_STOPPING_METHODS = frozenset({"subtree_prefetching_MH"})
# Methods that do so only on SGLang, whose PowerMH sampler truncates the final state after the first EOS.
SGLANG_EOS_STOPPING_METHODS = frozenset({"power_sampling_mcmc"})
# The first proposal of each prompt, logged as "context_len=418, new_suffix_len=1024" (vLLM PreSTO) or with each value
# in brackets (PowerMH, SGLang).
CONTEXT_LEN = r"context_len=\[?(\d+)\]?, new_suffix_len=\[?{horizon}\]?(?!\d)"
csv.field_size_limit(1 << 30)


def read_result_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def check_benchmark_matches_run(path: Path, benchmark) -> None:
    """Refuse to score a run whose log reports a different benchmark size than the rebuilt benchmark.

    Rebuilding from the wrong dataset file pairs each completion with another problem's prompt. The runner logs sit
    beside the CSV as ``<stem>.<method>.<backend>.log`` and print ``benchmark_size`` in their resolved configuration;
    e.g. lcb_v6 runs made before the loader fix read the 880-problem V5 file, which is no longer supported, and cannot be rescored.
    """
    for log in sorted(path.parent.glob(f"{path.stem}.*.log")):
        match = re.search(r"^\s*benchmark_size: (\d+)", log.read_text(errors="replace"), re.M)
        if match and int(match.group(1)) != benchmark.size:
            raise SystemExit(
                f"{log.name} ran on a {match.group(1)}-problem benchmark, but the rebuilt {benchmark.name} has "
                f"{benchmark.size}; pass --override to select the dataset the run used"
            )


def run_log(path: Path) -> tuple[str | None, str | None]:
    """Return the text and backend of the runner log ``<stem>.<method>.<backend>.log`` beside a result CSV, if any."""
    logs = sorted(path.parent.glob(f"{path.stem}.*.log"))
    if not logs:
        return None, None
    return logs[0].read_text(errors="replace"), logs[0].name.rsplit(".", 2)[-2]


def stops_on_eos(method: str, backend: str | None) -> bool:
    """Whether the run's responses end at the first EOS, which they keep."""
    return method in EOS_STOPPING_METHODS or (backend == "sglang" and method in SGLANG_EOS_STOPPING_METHODS)


def run_horizon(rows: list[dict], log_text: str | None, name: str) -> int:
    """Return ``max_new_tokens``: the CSV column (vLLM, HF), else the log's resolved configuration (SGLang)."""
    if rows[0].get("max_new_tokens"):
        return int(rows[0]["max_new_tokens"])
    match = re.search(r"^\s*max_new_tokens: (\d+)", log_text or "", re.M)
    if match is None:
        raise SystemExit(f"{name}: no max_new_tokens in the CSV or in its log")
    return int(match.group(1))


def logged_chat_template(log_text: str | None, method: str, backend: str | None, benchmark, model_alias: str) -> bool:
    """Return whether the run used the chat template, as its log records it."""
    if log_text is not None:
        match = re.search(r"prompt style for \S+: (chat template|raw)", log_text)
        if match:
            return match.group(1) == "chat template"
        if method in EOS_STOPPING_METHODS and backend != "sglang":
            return False
    return benchmark.should_use_chat_template(model_alias)


def check_prompt_lengths(log_text: str | None, prompt_lengths: list[int], horizon: int, name: str) -> None:
    """Check that the rebuilt prompt lengths appear, in order, among the ``context_len`` values the log records.

    Lines also come from MH steps that cut at the prompt, and vLLM PreSTO logs each first proposal twice, so the rebuilt
    lengths need only be a subsequence of the logged ones.
    """
    if log_text is None:
        return
    logged = [int(n) for n in re.findall(CONTEXT_LEN.format(horizon=horizon), log_text)]
    remaining = iter(logged)
    if logged and not all(length in remaining for length in prompt_lengths):
        raise SystemExit(f"{name}: rebuilt prompt lengths {prompt_lengths} are not among the logged {logged}")


def rebuild_prompts(benchmark, tokenizer, count: int, use_chat_template: bool) -> list[str]:
    """Return the first ``count`` prompts in the order the runners iterate the benchmark."""
    benchmark.use_chat_template = use_chat_template
    prompts: list[str] = []
    for batch in benchmark.dataset_loader:
        if len(prompts) >= count:
            break
        benchmark.seen = len(prompts)
        batch_prompts, _ = benchmark.get_question_and_answer(batch, tokenizer)
        prompts.extend(batch_prompts)
    if len(prompts) < count:
        raise RuntimeError(f"benchmark {benchmark.name} has {len(prompts)} prompts, CSV has {count} rows")
    return prompts[:count]


def load_model(model_str: str, dtype: str):
    """Load the base model for teacher-forced scoring (eager module, no compile)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_str, trust_remote_code=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_str,
        dtype=getattr(torch, dtype) if dtype != "auto" else "auto",
        trust_remote_code=True,
    ).to(device)
    model.eval()
    return tokenizer, SimpleNamespace(base_model=model, device=device)


def score_csv(path: Path, benchmark, tokenizer, scorer, model_alias: str) -> Path | None:
    """Score one result CSV and write <stem>.rescored.csv beside it; with ``scorer=None`` only run the checks."""
    rows = read_result_rows(path)
    if not rows:
        raise ValueError(f"{path} has no rows")
    method = rows[0].get("method", "")
    check_benchmark_matches_run(path, benchmark)
    log_text, backend = run_log(path)
    horizon = run_horizon(rows, log_text, path.name)
    eos_stopping = stops_on_eos(method, backend)
    use_chat_template = logged_chat_template(log_text, method, backend, benchmark, model_alias)
    prompts = rebuild_prompts(benchmark, tokenizer, len(rows), use_chat_template)
    check_prompt_lengths(log_text, [len(tokenizer.encode(p)) for p in prompts], horizon, path.name)
    logger.info("%s: method=%s, backend=%s, %d rows, horizon %d, prompt style=%s", path.name, method, backend,
                len(rows), horizon, "chat template" if use_chat_template else "raw")

    output_rows = []
    stripped = 0
    for index, (row, prompt) in enumerate(zip(rows, prompts)):
        prompt_ids = tokenizer.encode(prompt)
        completion = row.get("completion") or ""
        shown_prompt = tokenizer.decode(prompt_ids, skip_special_tokens=True)
        if backend == "sglang" and completion.startswith(shown_prompt):
            completion = completion[len(shown_prompt):]
            stripped += 1
        response_ids = tokenizer.encode(completion, add_special_tokens=False)
        fills_horizon = len(response_ids) == horizon
        appended_eos = eos_stopping and len(response_ids) < horizon
        if appended_eos:
            response_ids.append(tokenizer.eos_token_id)
        scored = {"sample_idx": index, "method": method, "prompt_tokens": len(prompt_ids),
                  "is_correct": row.get("is_correct", ""), "appended_eos": appended_eos,
                  "round_trip_exact": method in EOS_STOPPING_METHODS or fills_horizon}
        if scorer is None:
            output_rows.append(scored)
            continue
        if response_ids:
            diagnostics = compute_log_likelihood_and_confidence(scorer, prompt_ids + response_ids, len(prompt_ids))
            scored.update(diagnostics.to_json())
        else:
            scored.update({key: math.nan for key in ("num_response_tokens", "logprob_sum", "neg_entropy_sum",
                                                      "log_likelihood", "confidence")})
        for key in LOGGED_COLUMNS:
            scored[f"logged_{key}"] = row.get(key, "")
        output_rows.append(scored)
        logger.info("sample %d: prompt %d tokens, response %s tokens, log-likelihood %.6f, confidence %.6f",
                    index, len(prompt_ids), scored["num_response_tokens"], scored["log_likelihood"],
                    scored["confidence"])

    if stripped:
        logger.info("%s: stripped the prompt text from %d/%d completions", path.name, stripped, len(rows))
    if scorer is None:
        inexact = [row["sample_idx"] for row in output_rows if not row["round_trip_exact"]]
        logger.info("%s: checks passed; %d/%d rows round-trip exactly%s", path.name, len(rows) - len(inexact),
                    len(rows), f" (inexact: {inexact})" if inexact else "")
        return None

    output_path = path.with_name(path.stem + ".rescored.csv")
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    summarize(path.name, output_rows)
    return output_path


def summarize(name: str, rows: list[dict]) -> None:
    """Log mean +/- sd of both metrics, and the rescored-vs-logged gap where logged values exist."""
    for key in ("log_likelihood", "confidence", "num_response_tokens"):
        for label, subset in (("all", rows), ("round-trip exact", [r for r in rows if r["round_trip_exact"]])):
            values = [float(row[key]) for row in subset if not math.isnan(float(row[key]))]
            if not values:
                continue
            spread = stdev(values) if len(values) > 1 else 0.0
            logger.info("%s [%s]: %s = %.6f +/- %.6f (n=%d)", name, label, key, fmean(values), spread, len(values))
    for key in LOGGED_COLUMNS:
        pairs = [(float(row[key]), float(row[f"logged_{key}"])) for row in rows if row[f"logged_{key}"] not in ("", None)]
        if pairs:
            gaps = [abs(rescored - logged) for rescored, logged in pairs]
            logger.info("%s: |rescored - logged| %s: mean %.6f, max %.6f", name, key, fmean(gaps), max(gaps))


def build_parser():
    parser = argparse.ArgumentParser(description="Re-score result CSVs with base-model log-likelihood and confidence.")
    add_config_selection_args(
        parser,
        algorithm_default="power_mcmc",
        algorithm_choices=["power_mcmc", "subtree_prefetching_mh"],
        save_str_default="",
    )
    parser.add_argument("--csv", type=Path, nargs="+", required=True, help="Result CSVs to score.")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32", "auto"],
                        help="Scoring dtype; the log-softmax itself always runs in fp32.")
    parser.add_argument("--check-only", action="store_true",
                        help="Load only the tokenizer and run the dataset, prompt, and round-trip checks; write nothing.")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    logging.basicConfig(format="%(name)s %(levelname)s: %(message)s", level=logging.INFO)
    task = resolve_config(args, args.algorithm, ("task", "batch_size"))
    model_str = MODEL_MAP[args.model_str]
    # Seed exactly as the runners do right before build_benchmark: GPQA places its answer choices with the global RNG
    # while loading, so an unseeded rebuild gives each problem a different prompt.
    set_random_seed(args.seed)
    benchmark = build_benchmark(task, args, model_str)
    if args.check_only:
        tokenizer, scorer = transformers.AutoTokenizer.from_pretrained(model_str, trust_remote_code=True), None
    else:
        tokenizer, scorer = load_model(model_str, args.dtype)
    for csv_path in args.csv:
        output_path = score_csv(csv_path, benchmark, tokenizer, scorer, args.model_str)
        if output_path is not None:
            print(output_path, flush=True)
