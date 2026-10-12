# Agent Guide

Repo root: `/Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening`

`/Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/README.md` is the project guide (install, runners, configuration, testing, layout, contributing rules). Read it before making changes. This file only adds what applies to agents.

- Use absolute paths in commands, code references, and Markdown.
- Never edit `References/`.
- Do **not** switch `src/tests/test_low_temp_sampler.py` to bf16; its batched and scalar paths match only in fp32.

## Testing

- Run `pytest src` from the repo root (or a bare `pytest` from `src/`) with the package installed; config is in `src/pytest.ini`.
- `src/tests/test_low_temp_sampler.py` needs a GPU. It loads a real model in **float32 + eager attention** with **greedy** decoding; batched and scalar paths match only in fp32. Do **not** switch it to bf16.
- `src/examples/prefetch_dp_requests.py` is a CPU-only sanity check of the prefetch DP.


## Case studies and figures

`case_studies/` holds the paper's launchers, log parsers, figure code, and dated logs (see `case_studies/README.md`). The vLLM grids are in `case_studies/scripts/` (see its README). Each has `.interactive.sh`, `.punakha.sbatch.sh`, and `.tacc.sbatch.sh` entry points; pass `DATASETS=`, `MODELS=`, `RANKS=`, `BUDGETS=` after the script name.

```bash
bash "$REPO_ROOT/case_studies/scripts/subtree_prefetch.long_mh_steps+resource_prob_all_dataset.vllm.tacc.sbatch.sh" \
  METHODS=presto BUDGETS=20
```

Outputs per run: `.log`, `.csv` (scores), `.stats.json` (counters, timings), `.resources.json` (resource probe).

Draw proposal-certainty figures (`--all-edges` counts every logged accept/reject edge, including unvisited branches):

```bash
"$REPO_ROOT/src/.venv/bin/python" \
  "$REPO_ROOT/case_studies/draw/draw_dataset_proposal_certainty.py" \
  --log-root "$REPO_ROOT/case_studies/logs" --run-date 2026-09-16 \
  --model qwen3-4b --prefetch-budget 20 --rank-fn bfs_accept_first \
  --alpha 4.0 --mcmc-steps 100 --num-blocks 1 --max-samples 20 \
  --max-new-tokens 1024 --seed 10086 \
  --datasets math500 aime mbpp gpqa human_eval mmlu lcb_v6 --all-edges --hatch
```

Drawing dependencies: `uv pip install --python "$REPO_ROOT/src/.venv/bin/python" -r "$REPO_ROOT/case_studies/draw/requirements.txt"`. PDF export needs `qpdf`; some figures need LaTeX and Graphviz `dot`. All drawing options: `case_studies/draw/README.md`.
