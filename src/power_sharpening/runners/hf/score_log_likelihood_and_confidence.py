"""Re-score saved completions: base-model log-likelihood and confidence under p_0.

For each response x = (x_1, ..., x_n) in a result CSV, with prompt x_0 and p_t(u) = p_0(u | x_0, x_{<t}):

    LL(x)   = (1/n) sum_t log p_t(x_t)
    Conf(x) = (1/n) sum_t sum_{u in V} p_t(u) log p_t(u)      (full vocabulary, temperature 1)

The PreSTO (subtree-prefetching) runner logs both, but the PowerMH runners only save the decoded ``completion``, so
this script scores any result CSV offline with one teacher-forced forward pass per response
(``compute_log_likelihood_and_confidence``). Scoring every method with the same scorer keeps them comparable; rows that
already carry logged values are compared against the rescored ones as a check on the tokenization round trip.

Each run's prompts are rebuilt in the style its own log records ("prompt style for <model>: chat template|raw"). PreSTO
logs made before the PreSTO runners adopted ``benchmark.should_use_chat_template`` carry no such line and used raw
prompts; with no log at all, the current policy applies. PreSTO logs also record each prompt's token length
(``context_len``), and the rebuilt prompts must match it. CSV row i is dataset example i, as both runners write rows in
dataset order.
The response tokens are the re-encoded ``completion`` text, which the samplers decoded with skip_special_tokens=True,
so special tokens are lost from the text:

* PreSTO stops on EOS and counts that EOS in its response, so a PreSTO response shorter than ``max_new_tokens`` gets
  its EOS appended back (``appended_eos``).
* PowerMH runs with ignore_eos over a fixed ``max_new_tokens`` horizon. A chain state that passed through EOS continued
  into a new chat turn whose special tokens were stripped from the text, so its re-encoded response is shorter than the
  horizon and is not the token sequence the chain sampled. Such rows are kept but marked ``round_trip_exact=False``.

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
# Runner methods whose samplers stop on the first EOS and keep it in the response.
EOS_STOPPING_METHODS = frozenset({"subtree_prefetching_MH"})
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


def run_log(path: Path) -> str | None:
    """Return the text of the runner log beside a result CSV, if there is one."""
    logs = sorted(path.parent.glob(f"{path.stem}.*.log"))
    return logs[0].read_text(errors="replace") if logs else None


def logged_chat_template(log_text: str | None, method: str, benchmark, model_alias: str) -> bool:
    """Return whether the run used the chat template, as its log records it."""
    if log_text is not None:
        match = re.search(r"prompt style for \S+: (chat template|raw)", log_text)
        if match:
            return match.group(1) == "chat template"
        if method in EOS_STOPPING_METHODS:
            return False
    return benchmark.should_use_chat_template(model_alias)


def check_prompt_lengths(log_text: str | None, prompt_lengths: list[int], horizon: int, name: str) -> None:
    """Compare rebuilt prompt lengths with the per-sample ``context_len`` a PreSTO log records."""
    if log_text is None:
        return
    logged = [int(n) for n in re.findall(rf"context_len=(\d+), new_suffix_len={horizon}\b", log_text)]
    if logged and logged[:len(prompt_lengths)] != prompt_lengths:
        raise SystemExit(f"{name}: rebuilt prompt lengths {prompt_lengths} differ from the logged {logged}")


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
    log_text = run_log(path)
    use_chat_template = logged_chat_template(log_text, method, benchmark, model_alias)
    prompts = rebuild_prompts(benchmark, tokenizer, len(rows), use_chat_template)
    check_prompt_lengths(log_text, [len(tokenizer.encode(p)) for p in prompts], int(rows[0]["max_new_tokens"]),
                         path.name)
    logger.info("%s: method=%s, %d rows, prompt style=%s", path.name, method, len(rows),
                "chat template" if use_chat_template else "raw")

    output_rows = []
    for index, (row, prompt) in enumerate(zip(rows, prompts)):
        prompt_ids = tokenizer.encode(prompt)
        response_ids = tokenizer.encode(row.get("completion") or "", add_special_tokens=False)
        horizon = int(row["max_new_tokens"])
        appended_eos = method in EOS_STOPPING_METHODS and len(response_ids) < horizon
        if appended_eos:
            response_ids.append(tokenizer.eos_token_id)
        scored = {"sample_idx": index, "method": method, "prompt_tokens": len(prompt_ids),
                  "is_correct": row.get("is_correct", ""), "appended_eos": appended_eos,
                  "round_trip_exact": method in EOS_STOPPING_METHODS or len(response_ids) == horizon}
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
