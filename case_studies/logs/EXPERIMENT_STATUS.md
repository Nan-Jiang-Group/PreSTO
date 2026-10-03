# Experiment Run Status

Snapshot: 2026-08-19

This tracker covers the current vLLM long-MH-step grid defined by
`case_studies/scripts/power_mh.long_mh_steps.punakha.sbatch.sh` and
`case_studies/scripts/subtree_prefetch.long_mh_steps.vllm.punakha.sbatch.sh`.

- Shared configuration: `alpha=4.0`, `mcmc_steps=100`, `num_blocks=1`,
  `max_samples=20`, `max_new_tokens=1024`, and `seed=10086`.
- PowerMH expects one run per dataset/model pair.
- SubtreeMH expects 54 runs per dataset/model pair: nine prefetch budgets
  (`4, 6, 8, 10, 12, 14, 16, 18, 20`) times six ranking methods
  (`accept_first`, `reject_first`, `longest_first`, `smallest_cut_diff`,
  `longest_path_first`, `bfs_reject_first`).
- **Complete** means the matching `.log` exists and contains
  `INFO: output saved to`. **Incomplete** means the log exists without that
  completion record. Derived CSV, JSON, and PDF files are not counted as runs.
- In compact entries, `pN/rank` means prefetch budget `N` with that ranking
  method.

## Overall

| Experiment family | Complete | Incomplete | Not run | Expected |
|---|---:|---:|---:|---:|
| PowerMH | 57 | 1 | 6 | 64 |
| SubtreeMH | 134 | 27 | 3,295 | 3,456 |

## MATH500

| Model | PowerMH | SubtreeMH complete | SubtreeMH incomplete | SubtreeMH not run |
|---|---|---:|---|---|
| `qwen` | Complete | 44/54 | 10: `p4-p18/longest_first`; `p20/{longest_first, smallest_cut_diff}` | 0 |
| `qwen3-4b` | Complete | 45/54 | 9: `p6/{accept_first, reject_first}`; `p8-p20/accept_first` | 0 |
| `qwen3.5-4b` | Complete | 45/54 | 8: `p4/{accept_first, reject_first}`; `p6/{accept_first, reject_first}`; `p8/accept_first`; `p10/{accept_first, reject_first}`; `p12/accept_first` | 1: `p8/reject_first` |
| `qwen3-8b` | Complete | 0/54 | 0 | 54: all SubtreeMH configurations |
| `qwen3.5-9b` | Complete | 0/54 | 0 | 54: all SubtreeMH configurations |
| `qwen-math-medium` | Complete | 0/54 | 0 | 54: all SubtreeMH configurations |
| `tulu` | Complete | 0/54 | 0 | 54: all SubtreeMH configurations |
| `phi3.5` | Complete | 0/54 | 0 | 54: all SubtreeMH configurations |

## AIME

| Model | PowerMH | SubtreeMH complete | SubtreeMH incomplete | SubtreeMH not run |
|---|---|---:|---:|---|
| `qwen` | Complete | 0/54 | 0 | 54: all SubtreeMH configurations |
| `qwen3-4b` | Complete | 0/54 | 0 | 54: all SubtreeMH configurations |
| `qwen3.5-4b` | Complete | 0/54 | 0 | 54: all SubtreeMH configurations |
| `qwen3-8b` | Complete | 0/54 | 0 | 54: all SubtreeMH configurations |
| `qwen3.5-9b` | Complete | 0/54 | 0 | 54: all SubtreeMH configurations |
| `qwen-math-medium` | Complete | 0/54 | 0 | 54: all SubtreeMH configurations |
| `tulu` | Complete | 0/54 | 0 | 54: all SubtreeMH configurations |
| `phi3.5` | Complete | 0/54 | 0 | 54: all SubtreeMH configurations |


## MBPP

| Model | PowerMH | SubtreeMH complete | SubtreeMH incomplete | SubtreeMH not run |
|---|---|---:|---:|---|
| `qwen` | Complete | 0/54 | 0 | 54: all SubtreeMH configurations |
| `qwen3-4b` | Complete | 0/54 | 0 | 54: all SubtreeMH configurations |
| `qwen3.5-4b` | Complete | 0/54 | 0 | 54: all SubtreeMH configurations |
| `qwen3-8b` | Complete | 0/54 | 0 | 54: all SubtreeMH configurations |
| `qwen3.5-9b` | Complete | 0/54 | 0 | 54: all SubtreeMH configurations |
| `qwen-math-medium` | Complete | 0/54 | 0 | 54: all SubtreeMH configurations |
| `tulu` | Complete | 0/54 | 0 | 54: all SubtreeMH configurations |
| `phi3.5` | Complete | 0/54 | 0 | 54: all SubtreeMH configurations |

## GPQA

| Model | PowerMH | SubtreeMH complete | SubtreeMH incomplete | SubtreeMH not run |
|---|---|---:|---:|---|
| `qwen` | Complete | 0/54 | 0 | 54: all SubtreeMH configurations |
| `qwen3-4b` | Complete | 0/54 | 0 | 54: all SubtreeMH configurations |
| `qwen3.5-4b` | Complete | 0/54 | 0 | 54: all SubtreeMH configurations |
| `qwen3-8b` | Complete | 0/54 | 0 | 54: all SubtreeMH configurations |
| `qwen3.5-9b` | Complete | 0/54 | 0 | 54: all SubtreeMH configurations |
| `qwen-math-medium` | Complete | 0/54 | 0 | 54: all SubtreeMH configurations |
| `tulu` | Complete | 0/54 | 0 | 54: all SubtreeMH configurations |
| `phi3.5` | Complete | 0/54 | 0 | 54: all SubtreeMH configurations |

## HumanEval

| Model | PowerMH | SubtreeMH complete | SubtreeMH incomplete | SubtreeMH not run |
|---|---|---:|---:|---|
| `qwen` | Complete | 0/54 | 0 | 54: all SubtreeMH configurations |
| `qwen3-4b` | Complete | 0/54 | 0 | 54: all SubtreeMH configurations |
| `qwen3.5-4b` | Complete | 0/54 | 0 | 54: all SubtreeMH configurations |
| `qwen3-8b` | Complete | 0/54 | 0 | 54: all SubtreeMH configurations |
| `qwen3.5-9b` | Complete | 0/54 | 0 | 54: all SubtreeMH configurations |
| `qwen-math-medium` | Complete | 0/54 | 0 | 54: all SubtreeMH configurations |
| `tulu` | Complete | 0/54 | 0 | 54: all SubtreeMH configurations |
| `phi3.5` | Complete | 0/54 | 0 | 54: all SubtreeMH configurations |

## LiveCodeBench (`lcb`)

| Model | PowerMH | SubtreeMH complete | SubtreeMH incomplete | SubtreeMH not run |
|---|---|---:|---:|---|
| `qwen` | Complete | 0/54 | 0 | 54: all SubtreeMH configurations |
| `qwen3-4b` | Complete | 0/54 | 0 | 54: all SubtreeMH configurations |
| `qwen3.5-4b` | Complete | 0/54 | 0 | 54: all SubtreeMH configurations |
| `qwen3-8b` | Complete | 0/54 | 0 | 54: all SubtreeMH configurations |
| `qwen3.5-9b` | Complete | 0/54 | 0 | 54: all SubtreeMH configurations |
| `qwen-math-medium` | Complete | 0/54 | 0 | 54: all SubtreeMH configurations |
| `tulu` | Complete | 0/54 | 0 | 54: all SubtreeMH configurations |
| `phi3.5` | Complete | 0/54 | 0 | 54: all SubtreeMH configurations |

## MMLU

| Model | PowerMH | SubtreeMH complete | SubtreeMH incomplete | SubtreeMH not run |
|---|---|---:|---:|---|
| `qwen` | Complete | 0/54 | 0 | 54: all SubtreeMH configurations |
| `qwen3-4b` | Incomplete: log ends at MH step 95/100 without saving output | 0/54 | 0 | 54: all SubtreeMH configurations |
| `qwen3.5-4b` | Not run | 0/54 | 0 | 54: all SubtreeMH configurations |
| `qwen3-8b` | Not run | 0/54 | 0 | 54: all SubtreeMH configurations |
| `qwen3.5-9b` | Not run | 0/54 | 0 | 54: all SubtreeMH configurations |
| `qwen-math-medium` | Not run | 0/54 | 0 | 54: all SubtreeMH configurations |
| `tulu` | Not run | 0/54 | 0 | 54: all SubtreeMH configurations |
| `phi3.5` | Not run | 0/54 | 0 | 54: all SubtreeMH configurations |
