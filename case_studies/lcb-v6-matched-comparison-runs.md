# LCB V6 only: PowerMH versus PreSTO-PowerMH

The requested figures require fresh matched GPU runs. No jobs have been submitted.
The previous loader always selected `LiveCodeBench_v5.jsonl`, including when the
configuration and log header said `lcb_v6`. Those old figures are excluded from
this V6-only comparison. Merely adding a 1,024-token baseline would not resolve
the dataset mismatch.

The loader now selects the requested release file and logs its path and SHA-256.
The downloaded V6-only shard has **175 problems**, dated 2025-01-04 through
2025-04-06, with **zero overlapping (platform, question_id) pairs** with the
local 880-problem V5 file. Its normalized JSONL SHA-256 is
`e92fbf266ad18eaa6995a03104792d494b342f1a3c2524bbd99d67afc35b6f21`.

The [official loader](https://huggingface.co/datasets/livecodebench/code_generation_lite/blob/main/code_generation_lite.py)
distinguishes the single V6 shard (`v6`, `test6.jsonl`) from the cumulative
`release_v6` collection, which also includes earlier shards. This project uses
the single shard for the requested **V6 only, omit V5** experiment.
Its existing downloader spells that selection `--lcb-version release_v6`.

## Prepared runs

[The TACC launcher](/Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/scripts/lcb_v6_power_mh_vs_presto.tacc.sbatch.sh)
prepares six model jobs: Qwen2.5-7B, Qwen3-4B, Qwen3-8B, Qwen3.5-4B,
Qwen3.5-9B, and Gemma-4-12B. Each job runs PowerMH and PreSTO-PowerMH
sequentially on the same allocated GPU.

Shared settings: V6-only data, 1,024 new tokens, alpha 4, 100 MH transitions,
one block, 20 problems, seed 10086, uniform cuts, constant proposal temperature
0.25, batch size 1, prefix caching, resource probes and cache hooks enabled.
PreSTO uses BFS accept-first and prefetch budget 20. Use the same vLLM environment
for both methods; the old subtree runs used vLLM 0.27.1.

Run on TACC after syncing this checkout, including the loader fix:

```bash
repo_root="$HOME/WORK/data/Subtree-Prefetching-Power-Sharpening"
source "$HOME/WORK/miniconda3/etc/profile.d/conda.sh"
conda activate cuda130
python "$repo_root/data/fetch_data.py" livecodebench --lcb-version release_v6
bash "$repo_root/case_studies/scripts/lcb_v6_power_mh_vs_presto.tacc.sbatch.sh"
```

Set `MODELS="qwen3-4b qwen3-8b qwen3.5-4b"` to restrict the jobs to the three
models originally missing 1,024-token baselines. Both methods still need reruns
on the verified V6-only data. `RUN_DATE` defaults to `YYYY-MM-DD-v6-only`;
existing per-run logs are never overwritten. The job checks the V6-only source
fingerprint before either sampler starts.

## Generate the paired PDFs after the jobs finish

Use the run's actual date below. On this machine, `repo_root` is the absolute
path shown here; on the cluster, set it to the synced checkout's absolute path.
The existing renderer rejects mismatched configurations and uses only complete traces.
Only the six new paired PDFs enter the ZIP; all PDFs sit directly in one folder.

```bash
set -euo pipefail
repo_root=/Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening
run_date=2026-09-19-v6-only
logs="$repo_root/case_studies/logs/lcb_v6/$run_date/vllm"
figures="$repo_root/case_studies/lcb-v6-verified-power-mh-vs-step"
mkdir -p "$figures"
export MPLCONFIGDIR=/private/tmp/power-sharpening-mplconfig
export XDG_CACHE_HOME=/private/tmp/power-sharpening-xdg-cache
for model in qwen qwen3-4b qwen3-8b qwen3.5-4b qwen3.5-9b gemma-12b; do
    prefix="dataset-lcb_v6.model-$model.alpha4.0.steps100.blocks-1"
    suffix=samples20.maxnew1024.seed10086
    power="$logs/$prefix.$suffix.powerMH.vllm.log"
    subtree="$logs/$prefix.prefetch-budget-20.rank-bfs_accept_first.$suffix.subtreePrefetch.vllm.log"
    for log in "$power" "$subtree"; do
        rg -q 'dataset source: version=release_v6 path=.*LiveCodeBench_v6.jsonl sha256=e92fbf266ad18eaa6995a03104792d494b342f1a3c2524bbd99d67afc35b6f21$' "$log"
    done
    "$repo_root/src/.venv/bin/python" \
        "$repo_root/case_studies/draw/draw_accept_first_power_mh_model_calls_and_empirical_time.py" \
        --directory "$logs" --subtree-log "$subtree" --power-mh-log "$power" \
        --traces-per-rank 20 \
        --output "$figures/dataset-lcb_v6.model-$model.power-mh-vs-step.pdf"
done
"$repo_root/src/.venv/bin/python" - "$figures" <<'PY'
from pathlib import Path
import sys
from zipfile import ZipFile, ZIP_DEFLATED
directory = Path(sys.argv[1])
models = ('qwen', 'qwen3-4b', 'qwen3-8b', 'qwen3.5-4b', 'qwen3.5-9b', 'gemma-12b')
pdfs = [directory / f'dataset-lcb_v6.model-{model}.power-mh-vs-step.pdf' for model in models]
assert all(path.is_file() for path in pdfs), 'All six paired PDFs must exist'
archive = directory.with_suffix('.zip')
with ZipFile(archive, 'w', ZIP_DEFLATED) as bundle:
    for path in pdfs:
        bundle.write(path, f'{directory.name}/{path.name}')
print(archive)
PY
```

Inspect the resulting PDFs visually before using them in the paper. No verified
V6-only timing PDFs have been generated yet, because the GPU measurements are
still pending.
