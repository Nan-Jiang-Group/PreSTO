<h1>
  <img src="logo.svg" alt="PreSTO logo" height="48" align="absmiddle">
  PreSTO: Predictive Subtree Prefetching for Fast LLM Power Sampling
</h1>

This repository implements PreSTO together with Metropolis–Hastings (MH) and related samplers for
power-sharpened LLM distributions, proportional to `p(x)^α`, with HuggingFace,
vLLM, and SGLang backends.

**Subtree prefetching** batches proposals for possible future MH states. A
single-proposal MH step has accept and reject branches, so the sampler can
prepare a tree of continuations before knowing which branch the chain will
visit. It then follows the realized MH decisions and discards unused work.
The prefetch budget and node-ranking policy control this tradeoff between
speculative computation, cache use, and sequential model calls. Prefetching
preserves the underlying MH transition rule; a finite run still has the usual
MCMC convergence limitations.

## Project website

The canonical repository is [Nan-Jiang-Group/PreSTO](https://github.com/Nan-Jiang-Group/PreSTO). The project website source and its history are included under `website/`; no website submodule is required.

[Paper](https://arxiv.org/xxxx) · [Current website](https://step-tree-explorer.jiangnanhugo.chatgpt.site/)

Build and preview the website with Node.js and Python 3:

```bash
cd /absolute/path/to/PreSTO/website
npm ci
npm test
npm run build
npm start
```

Open http://localhost:4173/. The exported experiment data is included; research logs are not needed to build the site. Build output is written to `website/dist/`.

The `Website` Actions workflow tests and builds changes under `website/`. Its Pages deployment job is enabled only for a public repository; private source continues to use the existing Sites hosting. When Pages is enabled, the project URL is https://nan-jiang-group.github.io/PreSTO/. Repository visibility and hosting access are separate settings; the workflow does not change either.

## Installation

Python 3.10 or newer is required. Set the absolute checkout path once; all
commands below use it. On a cluster, change it to the cluster checkout.

```bash
export REPO_ROOT=/Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening

uv sync --project "$REPO_ROOT/src"                # HF, benchmark tasks, and pytest
uv sync --project "$REPO_ROOT/src" --extra vllm   # add vLLM
# Alternative serving backend:
# uv sync --project "$REPO_ROOT/src" --extra sglang
```

The [package manifest](/Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/pyproject.toml)
pins **vLLM 0.27.1** and declares **SGLang ≥0.5.0**. HuggingFace dependencies
(`torch` and `transformers`) are part of the base installation; there is no
separate `hf` extra. The custom vLLM engine and patches must stay aligned with
the pinned version. Optional cache eviction has stricter requirements below.

Plain pip is also supported:

```bash
python -m pip install -e "$REPO_ROOT/src"          # HF and benchmark tasks
python -m pip install -e "$REPO_ROOT/src[vllm]"    # with vLLM
# Extras also include sglang and all.
```

For figures, install the drawing requirements into the same environment used
by the drawing scripts:

```bash
uv pip install --python "$REPO_ROOT/src/.venv/bin/python" \
  -r "$REPO_ROOT/case_studies/draw/requirements.txt"
```

PDF export requires `qpdf` on `PATH`. Proposal-certainty figures also require
a working LaTeX installation; Graphviz tree renderers require the `dot`
executable. [draw_all_logs.sh](/Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/draw/draw_all_logs.sh)
uses `$REPO_ROOT/src/.venv/bin/python` directly.

## Samplers and runner interfaces

Runner modules live under `power_sharpening.runners.<backend>`.

| Method | HF module | vLLM module | SGLang module |
| --- | --- | --- | --- |
| PowerMH | `run_power_mh` | `run_power_mh` | `run_power_sample_mh` |
| Subtree-prefetching MH | `run_subtree_prefetching_mh` | `run_subtree_prefetching_mh` | `run_subtree_prefetching_mh` |
| Multiple-Try Metropolis (MTM) | `run_multi_try_mh` | `run_multi_try_mh` | — |
| Subtree-prefetching MTM | `run_subtree_prefetching_multi_try_mh` | `run_subtree_prefetching_multi_try_mh` | — |

HF and custom vLLM PowerMH and subtree MH support both **uniform** and
**entropy-based** suffix cuts. Entropy cuts include the forward/reverse
cut-probability correction in the MH acceptance calculation. SGLang has a
separate `run_entropy_cut_mh` runner; its public-backend entropy mode uses a
top-K approximation. The legacy SGLang PowerMH and subtree runners do not
expose the same entropy-cut interface.

The [vLLM runners](/Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/power_sharpening/runners/vllm)
also include low-temperature / best-of-N sampling, SMC, and Calderhead
multiple-proposal MH.

**The CLIs differ across runners.** The HF subtree and dedicated MTM runners,
the vLLM runners listed in the table, and SGLang's `run_entropy_cut_mh` use
`--dataset` and `--override key=value`. HF's `run_power_mh` and SGLang's
`run_power_sample_mh` / `run_subtree_prefetching_mh` retain `--task` and
explicit hyperparameter flags.

### Configuration

The config-driven runners merge settings in this order, with later values
taking precedence:

1. `default` in [algorithms.yaml](/Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/power_sharpening/config/algorithms.yaml).
2. The selected algorithm block.
3. `defaults` in [dataset.yaml](/Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/power_sharpening/config/dataset.yaml).
4. The selected dataset block.
5. CLI `--override key=value` entries.

Overrides accept YAML values, including lists, booleans, and `null`.
Use `--model_str` to select a model alias, `--save_str` for the output
directory, and `--run_name` to set a common output prefix. The case-study
helpers use that prefix to pair logs, results, statistics, and resource data.

| Setting | Meaning |
| --- | --- |
| `alpha` | Exponent of the power-sharpened target. |
| `temperature` | Proposal temperature; `-1` selects `1/alpha`. |
| `mcmc_steps`, `num_blocks` | MH transitions per block and number of generation blocks. |
| `prefetch_budget` | Proposal budget for subtree methods. |
| `rank_fn` | Priority for expanding the speculative tree. |
| `cut_dist_type` | `uniform` or `entropy` for single-proposal HF/vLLM MH. |
| `cut_power` / `cut_dist_param` | Entropy-cut exponent: PowerMH uses `cut_power`; subtree MH uses `cut_dist_param`. |
| `print_tree` | Write proposal trees to the log, needed for tree-based figures. |
| `max_samples`, `max_new_tokens` | Dataset sample limit and generation-token limit. |

The [rank registry](/Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/power_sharpening/common/prefetch_rank.py)
contains `accept_first`, `reject_first`, `longest_first`,
`smallest_cut_diff`, `longest_path_first`, `bfs_accept_first`, and
`bfs_reject_first`. The default subtree rank is `longest_path_first`;
individual launch scripts can override it.

### Run a small subtree experiment

GPU runs should use a Slurm allocation. Set `PARTITION` and `QUOTATYPE`
for the cluster, adapting scheduler flags where necessary:

```bash
srun -p "$PARTITION" --quotatype="$QUOTATYPE" --gres=gpu:1 --cpus-per-task=24 --time=03:00:00 \
  "$REPO_ROOT/src/.venv/bin/python" \
  -m power_sharpening.runners.vllm.run_subtree_prefetching_mh \
  --dataset math500 --algorithm subtree_prefetching_mh --model_str qwen3-4b \
  --save_str /tmp/subtree-prefetch --run_name math500-subtree-smoke \
  --override alpha=4.0 temperature=-1 temperature_schedule_type=const \
  mcmc_steps=10 num_blocks=1 prefetch_budget=20 rank_fn=bfs_accept_first \
  max_samples=2 max_new_tokens=512 print_tree=true
```

The HF subtree runner uses the same configuration interface; change
`runners.vllm` to `runners.hf`. For entropy-based subtree cuts, add
`cut_dist_type=entropy cut_dist_param=4.0` to the overrides. For vLLM
sequential PowerMH, use `run_power_mh --algorithm power_mcmc` and
`cut_dist_type=entropy cut_power=4.0`, omitting subtree-only settings.

### Multiple-Try Metropolis

Use `run_multi_try_mh --algorithm multi_try` on HF or vLLM.
`num_tries` controls the number of candidate suffixes per transition.
The current YAML defaults use four tries and the fixed whole-suffix
temperature mixture `[0.25, 0.5, 1.0]`. Override with
`proposal_temperatures=null` to use the scalar proposal temperature;
an explicit temperature list requires a constant schedule.

For prefetched MTM, use `run_subtree_prefetching_multi_try_mh` with
`--algorithm subtree_prefetching_multi_try_mh`. It supports uniform
cuts and a constant proposal distribution. Its budget counts **suffix
requests**: `num_tries=4 prefetch_budget=16` fits four complete bundles;
the budget must be at least `num_tries`.

The [MultiTry case-study scripts](/Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/scripts/MultiTry-family)
and [implementation note](/Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/scripts/MultiTry-family/IMPLEMENTATION.md)
describe the comparison workflow and implementation.

## Models and benchmarks

Model aliases and chat-template routing live in
[constants.py](/Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/power_sharpening/tasks/constants.py).
To add a model, update `MODEL_MAP` and, where appropriate,
`CHAT_TEMPLATE_MODELS`, then include the alias in the launcher.

The current all-dataset TACC sweep uses these aliases:

| Alias | Model identifier in `MODEL_MAP` |
| --- | --- |
| `qwen3.5-4b` | `Qwen/Qwen3.5-4B` |
| `qwen3-4b` | `Qwen/Qwen3-4B` |
| `qwen3-8b` | `Qwen/Qwen3-8B` |
| `gemma-12b-it` | `google/gemma-4-12B-it` |

Available benchmark configuration keys include `math500`, `aime`,
`mbpp`, `gpqa`, `human_eval`, `mmlu`, and
`lcb_v6`. AIME defaults to the combined 2024–2025 set. Loaders in
[tasks](/Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/power_sharpening/tasks)
use local files under [data](/Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/data) or dataset downloads,
depending on the benchmark.

`lcb_v6` loads the V6-only file written by `data/fetch_data.py livecodebench
--lcb-version release_v6`, never a V5 fallback. The `swebench` YAML entry does
not yet have a registered task handler.

## Case studies and figures

### Submit the model/rank sweep

Each vLLM PowerMH / PreSTO case study has its own launcher, listed in the
[scripts README](/Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/scripts/README.md).
Run the entry point for your cluster: `<name>.interactive.sh`,
`<name>.punakha.sbatch.sh`, or `<name>.tacc.sbatch.sh`. Pass grid settings such
as `DATASETS=...`, `MODELS=...`, `RANKS=...`, or `BUDGETS=...` as arguments after
the script name; they override the settings block at the top of the entry point.

The command below runs the matched resource-probe sweep PreSTO-only, with
budget 20. It submits **28 jobs**: seven datasets × four
models with the `bfs_accept_first` rank. Each job uses `alpha=4`, 100 MH steps,
one block, 20 samples, and at most 1,024 generated tokens, with tree logging and
resource probing enabled. Add `DRY_RUN=1` to print the `sbatch` commands
first.

```bash
bash "$REPO_ROOT/case_studies/scripts/subtree_prefetch.long_mh_steps+resource_prob_all_dataset.vllm.tacc.sbatch.sh" \
  METHODS=presto BUDGETS=20
```

[run_power_mh_case_study.sh](/Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/scripts/run_power_mh_case_study.sh)
and [run_subtree_prefetching_mh_case_study.sh](/Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/scripts/run_subtree_prefetching_mh_case_study.sh)
are the shared vLLM helpers. Their `--log-mode file` default writes to
the run log; `--log-mode tee` also streams output to the terminal.
`--override` forwards the remaining `key=value` settings to the runner.

Uniform-cut runs go under
`$REPO_ROOT/case_studies/logs/<dataset>/<date>/vllm`.
The TACC sweep's Slurm output goes under
`$REPO_ROOT/case_studies/logs/all-datasets/<date>/vllm`.
The [EntropyCut scripts](/Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/scripts/EntropyCut-family)
use [logs-entropycut](/Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/logs-entropycut);
the MultiTry scripts use
[logs-MultiTry](/Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/logs-MultiTry).

| Artifact | Contents |
| --- | --- |
| `.log` | Runner output and printed proposal trees, when enabled. |
| `.csv` | Benchmark outputs and scores. |
| `.stats.json` | Sampler counters and timings; optional cache-cleanup statistics. |
| `.resources.json` | Resource-probe measurements, when enabled. |

The vLLM MH runners enable resource probing by default;
`--no-resource_probe` disables it. Exact cache hooks require
`VLLM_ENABLE_V1_MULTIPROCESSING=0`; the shared launchers configure this
when using `--kv-cache-mode hooks`. Sampled occupancy and exact
allocation/eviction counters are different measurements.

### Draw proposal-certainty figures using all tree edges

[draw_dataset_proposal_certainty.py](/Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/draw/draw_dataset_proposal_certainty.py)
selects logs by model, rank, date, and sampling settings. Set `RUN_DATE`
to the date of completed runs; the example matches the sweep above.

```bash
RUN_DATE=2026-09-16
"$REPO_ROOT/src/.venv/bin/python" \
  "$REPO_ROOT/case_studies/draw/draw_dataset_proposal_certainty.py" \
  --log-root "$REPO_ROOT/case_studies/logs" --run-date "$RUN_DATE" \
  --model qwen3-4b --prefetch-budget 20 --rank-fn bfs_accept_first \
  --alpha 4.0 --mcmc-steps 100 --num-blocks 1 \
  --max-samples 20 --max-new-tokens 1024 --seed 10086 \
  --datasets math500 aime mbpp gpqa human_eval mmlu lcb_v6 \
  --all-edges --hatch
```

**Keep `--all-edges` to count every acceptance and rejection edge in the
logged trees**, including branches the chain did not visit. Each scored
proposal contributes its acceptance probability `A` and rejection
probability `1-A`, including zero-probability edges. This is an equally
weighted distribution over logged tree edges, not the realized chain's
acceptance rate. Without the flag, the script counts only `A` once per
proposal.

The command writes a PDF and source CSV under
`$REPO_ROOT/case_studies/logs/all-datasets/<date>/vllm`.
All-edge filenames include `.all-edges`; `--output` and `--csv-output`
override the destinations. Change `--model` and `--rank-fn` to generate
other panels from matching runs. For the historical 2026-09-04 collection,
the plotter has an explicit GPQA rerun mapping to 2026-09-05.

For the broader per-run and aggregate drawing pipeline:

```bash
bash "$REPO_ROOT/case_studies/draw/draw_all_logs.sh" \
  "$REPO_ROOT/case_studies/logs"

"$REPO_ROOT/src/.venv/bin/python" \
  "$REPO_ROOT/case_studies/draw/draw_entropycut_analysis.py" \
  --directory "$REPO_ROOT/case_studies/logs-entropycut" --aggregate-only
```

### Case-study code layout

`case_studies` is an importable package. Log parsers live in
`case_studies/extract/` and figure modules in `case_studies/draw/figures/<topic>/`
(model calls, cache, proposals, likelihood, EntropyCut, MultiTry). The scripts
named above, such as `draw_dataset_proposal_certainty.py`, stay at their
`case_studies/draw/` paths as thin wrappers, so the commands in this README
work unchanged. From the repository root the same figure also runs as
`python -m case_studies.draw.figures.proposals.draw_dataset_proposal_certainty`.

## Optional subtree cache eviction

The vLLM and SGLang subtree MH runners accept
`--evict_subtree_cache` / `--no-evict_subtree_cache`, default **off**.
Eviction recycles idle entries exclusive to discarded sequences while
preserving selected/shared prefixes. It does not shrink the preallocated GPU
cache pool.

vLLM requires the pinned 0.27.1 custom engine, prefix caching, and
`data_parallel_size=1`; pass `--override prefix_cache=true`.
SGLang requires 0.5.2 with ordinary RadixCache and
`dp_size=pp_size=nnodes=1`; tensor parallelism is supported. Direct wrapper
users must set `enable_subtree_cache_eviction=True`. The vLLM runner also
accepts `--override evict_subtree_cache=true`.

Cleanup timings and counts appear under `cache_evictions` in
`.stats.json`, separate from allocation-driven eviction metrics.
CPU tests cover ownership and transport behavior; GPU performance and
numerical equivalence require validation on the target serving setup.

## Repository layout and tests

| Path | Purpose |
| --- | --- |
| [src/power_sharpening](/Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/power_sharpening) | Installable package: shared utilities, backend wrappers/samplers, tasks, configs, and runners. |
| [src/tests](/Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/tests) | Unit and integration tests; runnable demos are in [examples](/Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/examples). |
| [experiments](/Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/experiments) | Backend/benchmark launch scripts, cluster settings, and shared environment setup. |
| [examples](/Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/examples) | CPU-only demo of the prefetch dynamic program (`prefetch_dp_requests.py`). |
| [data](/Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/data) | Local benchmark inputs. |
| [case_studies](/Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies) | Python package `case_studies`: launchers (`scripts/`), log parsers (`extract/`), figure code (`draw/`), and dated case-study outputs (`logs*`). See [draw/README.md](/Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/draw/README.md). |
| [result](/Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/result) | Benchmark results and statistical analyses. |
| [References](/Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/References) | Upstream reference implementations, kept read-only. |

Run focused CPU checks from the checkout:

```bash
cd "$REPO_ROOT"
"$REPO_ROOT/src/.venv/bin/python" -m pytest \
  "$REPO_ROOT/src/tests/test_proposal_tree.py" \
  "$REPO_ROOT/src/tests/test_cut_distribution.py" \
  "$REPO_ROOT/src/tests/test_entropy_cut_detailed_balance.py" \
  "$REPO_ROOT/src/tests/test_prefetch_multi_try.py"
```

The full suite also includes backend/model integration tests.
[test_low_temp_sampler.py](/Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/tests/test_low_temp_sampler.py)
loads a real model and needs a GPU. Its batched/scalar equivalence check uses
float32, eager attention, and greedy decoding; preserve those settings when
running it through Slurm.

## Related repositories

- [reasoning-with-sampling forks](https://github.com/aakaran/reasoning-with-sampling/forks)
- [inference_rl sampling experiments](https://github.com/ahmeda14960/inference_rl/tree/main/reasoning-with-sampling/llm_experiments)
- [reasoning-with-samples-efficient](https://github.com/Lev-Stambler/reasoning-with-samples-efficient/)
- [Large-Language-Models-reasoning-with-sampling](https://github.com/SelimLali/Large-Language-Models-reasoning-with-sampling)
- [power-sampling-test](https://github.com/tomo0530/power-sampling-test)
- [mh-llm](https://github.com/maxzuo/mh-llm)
- [Abiel136/reasoning-with-sampling](https://github.com/Abiel136/reasoning-with-sampling/)
- [jeffhernandez1995/reasoning-with-sampling](https://github.com/jeffhernandez1995/reasoning-with-sampling) and [notes](https://jeffhernandez1995.github.io/sampling,/commentary,/english/2026/03/17/vibe-research/)
- [Power-SMC](https://github.com/ArminAzizi98/Power-SMC) ([local reference](/Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/References/Power-SMC))
- [IsaacRe/reasoning-with-sampling](https://github.com/IsaacRe/reasoning-with-sampling)
- [ahabedsoltan vLLM power sampler](https://github.com/ahabedsoltan/reasoning-with-sampling/blob/main/llm_experiments/power_samp_utils_vllm.py#L89)
- [flashSpeculation](https://github.com/maxwell-gao/flashSpeculation)
