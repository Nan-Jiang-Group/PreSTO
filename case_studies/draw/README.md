# Case-study figures and log extraction

Layout of the `case_studies` Python package
(`/Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies`):

| Path | Holds |
|---|---|
| `paths.py` | Absolute repo, logs, draw, and extract directories |
| `plot_config.py` | Shared palette, method labels, and SciencePlots style |
| `extract/run_naming.py` | Parse and build run-configuration file names |
| `extract/logs/` | Log parsers: `proposal_log`, `model_call_traces`, `vllm_engine_metrics`, `resource_summary` |
| `extract/analysis/` | Derived tables: `rank_analysis`, `acceptance_features`, `branch_uncertainty` |
| `draw/common/` | Figure helpers: `figure_io`, `plot_helpers`, `rank_plotting` |
| `draw/figures/model_calls/` | Model calls and empirical time, PowerMH vs PreSTO |
| `draw/figures/endpoints/` | Endpoint metrics: mean MH transitions, peak KV cache |
| `draw/figures/cache/` | KV cache, prefix cache, vLLM engine metrics, resource usage |
| `draw/figures/proposals/` | Proposal certainty, acceptance, realized transitions, predictors |
| `draw/figures/likelihood/` | Likelihood and confidence (raw and rescored) |
| `draw/figures/entropycut/` | EntropyCut analysis and sweeps |
| `draw/figures/multitry/` | MultiTry-MH figures |

Every figure module keeps its historical file name. The files directly in
`/Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/draw`
and `.../extract` are thin wrappers, so existing commands and launch scripts still work:

```bash
draw/.venv/bin/python /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/draw/draw_model_calls_step.py --help
```

Equivalent module form, from the repository root:
`python -m case_studies.draw.figures.model_calls.draw_model_calls_step --help`.

To add a figure, put the module in the matching `draw/figures/<topic>/` package and, if it is a CLI, add a
three-line wrapper that calls `_entry.run("<dotted.module.path>")`.
