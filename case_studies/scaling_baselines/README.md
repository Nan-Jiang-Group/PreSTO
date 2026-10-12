# Scaling baselines

Accuracy of three baselines on LCB V6 with Qwen3.5-9B by default (first 20 prompts, one prompt at a time, 1,024
new tokens; pass `DATASETS`, `MODELS`, `MAX_SAMPLES`, `BATCH_SIZE`, `MAX_NEW_TOKENS` to change), each swept along the
axis that is natural for it. These are the standard comparisons for the power-sharpened samplers (PowerMH and friends).

| Method | Runner | Scaling axis (default) | Script |
| --- | --- | --- | --- |
| Low-temperature | `run_low_temp --algorithm low_temp` | temperature: 0.1, 0.25, 0.5, 0.75, 1.0 | `low_temp.temperature_scaling.punakha.sh` |
| Best-of-N | `run_low_temp --algorithm best_of_n` | samples N: 1, 2, 4, ..., 64 (at temperature 1.0) | `best_of_n.sample_scaling.punakha.sh` |
| Power-SMC | `run_power_smc --algorithm smc` | particles: 1, 2, 4, 8, 16, 32 | `power_smc.particle_scaling.punakha.sh` |

All paths below are relative to
`/Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/scaling_baselines`.

## Launch

```bash
bash power_smc.particle_scaling.punakha.sh DATASETS="math500 gpqa" MODELS=qwen3-8b
bash best_of_n.sample_scaling.punakha.sh tacc BEST_OF_N="1 4 16 64" TEMPERATURES="0.25 1.0"
bash low_temp.temperature_scaling.punakha.sh interactive      # runs on the current GPU node
```

The scripts submit to `punakha`; a leading `interactive` or `tacc` argument overrides that. The rest are `KEY=value`
settings. `DATASETS`, `MODELS`, `MAX_SAMPLES` (20), `BATCH_SIZE` (1), `SEED`, `RUN_DATE`, and `MAX_NEW_TOKENS` (1024)
are common; each script adds its axis (`TEMPERATURES`, `BEST_OF_N` and `TEMPERATURES`, `N_PARTICLES`). Runs go to
`logs/<dataset>/<date>/` as `<run_name>.csv` (one row per prompt) plus its `.log`; a run whose CSV exists is skipped, so
repeating a command resumes the sweep.

Notes on the settings:

- `MAX_NEW_TOKENS` defaults to 1,024 here, matching the total-time runs; `MAX_NEW_TOKENS=` (empty) keeps the
  per-dataset budget in `src/power_sharpening/config/dataset.yaml` (3,072 for most, 8,092 for LCB V6, 20,480 for
  AIME). 1,024 tokens truncate long reasoning, which lowers accuracy.
- Best-of-N draws `max(BEST_OF_N)` samples once and reuses the first `n` for every `n`, ranking by sequence
  log-probability, so one run covers the ladder. At temperature 0.25 the candidates are nearly identical; hence 1.0.
- Cost grows with N: best-of-N at 64 and Power-SMC at 32 are the expensive cells, especially on AIME and LCB V6.
  Narrow `DATASETS` / `N_PARTICLES` first.
- Power-SMC uses `batch_size` from `dataset.yaml` (500 for MATH500); lower it with a `batch_size=` entry in the
  script's overrides if the particle count does not fit in memory.

## Total time on the standard configuration

`total_time.standard_config.punakha.sh` times four methods on the same prompts, one prompt at a time:

- the low-temperature baseline;
- best-of-N at the same low temperature;
- PowerMH;
- PreSTO-PowerMH;
- with `METHODS=power_smc` / `power_smc_standard`, Power-SMC on the custom engine / on stock vLLM (particles 1 to 128).

All of them use the same settings: 1,024 new tokens, 8 blocks, 10 MH steps per block, alpha 4, temperature 0.25
(= 1/alpha). PreSTO uses rank `bfs_accept_first` with prefetch budget 10.

```bash
bash total_time.standard_config.punakha.sh                               # lcb_v6, qwen3.5-9b, first 20 prompts
bash total_time.standard_config.punakha.sh tacc DATASETS="math500 lcb_v6" MAX_SAMPLES=50 BEST_OF_N="8 32"
```

Settings: `METHODS` (`low_temp best_of_n power_mh presto`), `BEST_OF_N` (`4 8 16 32`), `DATASETS`, `MODELS`,
`MAX_SAMPLES`, `RANKS`, `BUDGETS`, and the shared knobs `MAX_NEW_TOKENS`, `MCMC_STEPS`, `NUM_BLOCKS`, `ALPHA`,
`TEMPERATURE`, `BATCH_SIZE`.

- **Best-of-N:** one run per N, so each run's time is the cost of that N alone.
- **PowerMH and PreSTO:** they run through `case_studies/scripts/_launch_common.sh`, with tree printing off.
- **KV cache:** all four methods run the same resource probe, so they pay the same measurement cost. It writes
  `<run>.resources.json` and per-prompt columns in the CSV. Two settings control it, for every method at once:
  - `RESOURCE_PROBE` (`on` or `off`; default `on`).
  - `KV_CACHE_MODE` (default `hooks`). `hooks` runs vLLM in-process, which gives exact peak KV occupancy and eviction
    counts. `inherit` keeps the engine in a child process: occupancy is only polled and eviction is not measured.
- **Logs:** every run writes to `logs_total_time/<dataset>/<date>/vllm/`.

`collect_total_time.py` reads these logs and writes `total_time.csv`, one row per run. Each row has:

- **Time:** total and per-prompt sampling time, and the ratio to the PowerMH total (`vs_power_mh`). The time is the sum
  of the per-prompt `... took <s> seconds` log lines, so it excludes model loading and grading.
- **Accuracy.**
- **KV cache:**
  - `peak_kv_cache_occupancy` and `peak_kv_cache_gib`: the run's peak use of the KV-block pool, as a fraction and in GiB.
  - `kv_pool_gib`: the pool size.
  - `mean_prompt_peak_kv`: the mean over prompts of each prompt's peak occupancy.
  - `kv_cache_eviction_rate` and `prefix_cache_hit_rate`.
  - `peak_demand_gib`: peak GPU memory without the idle part of the preallocated pool.
  - `kv_source`: `scheduler-hook` means exact; `metrics-poll` means sampled, so short peaks can be missed.

```bash
python -m case_studies.scaling_baselines.collect_total_time   # -> scaling_baselines/total_time.csv
```

## Collect and draw

```bash
cd /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening
export PYTHONPATH=$PWD
python -m case_studies.scaling_baselines.collect_accuracy    # -> scaling_baselines/accuracy_scaling.csv
python -m case_studies.scaling_baselines.draw_accuracy       # -> scaling_baselines/figures/*.pdf
```

`accuracy_scaling.csv` has one row per method, dataset, model, and scaling value (accuracy, graded prompts, binomial
standard error, source CSV). Figures: one per method (a panel per dataset, a line per base LLM) and `methods.pdf`,
which compares best-of-N and Power-SMC against N on the mean accuracy over datasets, with the best low-temperature
accuracy as a reference. Drawing needs the dependencies in `case_studies/draw/requirements.txt` and LaTeX.
