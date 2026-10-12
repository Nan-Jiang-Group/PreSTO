# Top‑k‑Constrained Beam Search (k‑CBS) — Implementation and Usage

This mini‑module adds a deterministic Top‑k‑Constrained Beam Search (k‑CBS) on top of vLLM. It enumerates likely continuations under top‑k decoding by:
- Expanding each active beam only with its per‑step top‑k tokens (from vLLM logprobs)
- Renormalizing within that k‑set
- Pruning globally to beam width B=k

It’s intended to compare against existing baselines in this repo:
- Vanilla beam search: `run_vllm_beam.sh` / `vllm_beam_math.py`
- Best‑of‑N sampling: `run_vllm_sampling_bestof.sh` / `vllm_passk_math.py`


## Files

- `reasoning-with-sampling/llm_experiments/topk_constrained_beam.py`
  - Core k‑CBS implementation.
  - Uses vLLM `SamplingParams(logprobs=topk, max_tokens=1)` to read per‑step top‑k logprobs (topk mask).
  - Vectorized inner loop (NumPy) for renormalization and pruning.
  - Returns final beams and effective `topk_eff` (min(topk, cap)).

- `reasoning-with-sampling/llm_experiments/vllm_kcbs_math.py`
  - Dataset runner over MATH500 mirroring other scripts.
  - Formats prompts (CoT by default), runs k‑CBS per problem, grades with existing math grader.
  - Saves CSV (per‑problem completions + correctness flags) and a JSON summary.

- `run_vllm_kcbs.sh`
  - Orchestrates the k‑CBS run with logging and plotting.
  - Produces a Pass@k curve via `eval_vllm_passk.py` that’s comparable to the other baselines.

- `codex_implement_top_k_beam_search.md`
  - Design notes and code skeleton (kept for reference).


## How to Run

Prereqs: same as the rest of the repo (vLLM installed, GPU available). From repository root:

1) Run the k‑CBS experiment

```
./run_vllm_kcbs.sh
```

- Defaults: model=`qwen_math`, k (beam width)=10, topk (per‑step logprobs)=40, length_penalty=1.0.
- Output directory: `reasoning-with-sampling/llm_experiments/kcbs_results/<model>/`.

2) Inspect results

- CSV: `kcbs_results/<model>/<model>_vllm_kcbs_bw_<k>_topk_<topk>_seed_0.csv`
- Plot: `kcbs_results/<model>/<model>_vllm_kcbs_bw_<k>_topk_<topk>_plot.png`
- Summary JSON: `kcbs_results/<model>/<model>_vllm_kcbs_bw_<k>_topk_<topk>_summary_seed_0.json`

You can also call the runner directly:

```
cd reasoning-with-sampling/llm_experiments
python vllm_kcbs_math.py \
  --model qwen_math \
  --k 10 \            # number of sequences (beam width)
  --topk 40 \         # per-step top-k next-token logprobs to consider
  --length_penalty 1.0 \
  --logprobs_cap 10 \
  --save_str kcbs_results/ \
  --seed 0
```


## What to Expect

- Deterministic outputs at fixed config (no randomness) — good for repeatable comparisons.
- Typically less diversity than sampling, more coverage of high‑probability region than greedy.
- Pass@k curve comparable but not identical to sampling/beam baselines:
  - k‑CBS renormalizes within the per‑step top‑k set; vanilla beam uses raw logits (often with temperature 0).
  - Recommended to set `topk >= k` so early steps have enough candidates; if your vLLM build cannot serve the requested `topk`, it will raise an error — lower `topk` accordingly.


## Notes and Tips

- Performance: The dominant cost is model inference; the NumPy pruning/renorm is negligible. The code batches all active beams per step into a single vLLM call.
- EOS handling: If EOS is in the top‑k set, the “ended” beam remains eligible during pruning but won’t be expanded further.
- Distance metrics: If you later need near‑verbatim probability bounds, the search can be extended with Hamming/Levenshtein pruning, as outlined in the design doc.
- Plots: `eval_vllm_passk.py` reads the per‑problem `correct_flags` list and plots Pass@j for j=1..k.


## Troubleshooting

- OOM or slow runs: lower `--max_new_tokens`, reduce `k`, or set `gpu_memory_utilization` in the LLM init (already set to 0.85 in the runner).
- Logprobs cap: most vLLM builds cap `SamplingParams.logprobs` (often 10–32). The implementation uses `--logprobs_cap` to stay within limits.
- Model support: mirrors the same model aliases used by the existing scripts.
