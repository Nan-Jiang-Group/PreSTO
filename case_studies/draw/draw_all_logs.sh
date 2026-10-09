#!/usr/bin/env bash
# Render the case-study figures for every subtree-prefetch log under a directory.
#
# Per-log drawing scripts name their output after the run configuration and write it beside the log they read. Aggregate
# renderers separately combine compatible vLLM logs by subtree batch size and *.resources.json files within each result
# directory. Their shared parser rejoins soft-wrapped Node(...) records in memory, so logs need no preprocessing.
#
# Every log is re-rendered on each run. The figures depend on plot_config.py and on the drawing scripts, not only on the
# log, so skipping by timestamp would quietly leave stale PDFs behind after a style change.
#
# A failing log does not stop the sweep -- a truncated or empty log is a normal thing to find in a directory of runs.
# Failures are collected and reported at the end, and the script exits non-zero if there were any.
#
# Usage, from anywhere:
#
#     case_studies/draw/draw_all_logs.sh [LOG_ROOT]
#
# LOG_ROOT defaults to case_studies/logs/math500 and is searched recursively for Hugging Face, vLLM, and SGLang
# subtree-prefetch logs. Python runs directly from the repository's src/.venv environment. Create it from the
# repository root with:
#
#     uv sync --project src

set -uo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")" && pwd)
repo_root=$(cd -- "${script_dir}/../.." && pwd)
drawing_dir="${script_dir}"
python_bin="${repo_root}/src/.venv/bin/python"
log_glob='*.subtreePrefetch.*.log'
figure_cache_root="${TMPDIR:-/tmp}/power-sharpening-cache"
if [[ -z ${MPLCONFIGDIR:-} ]]; then
    export MPLCONFIGDIR="${figure_cache_root}/matplotlib"
fi
if [[ -z ${XDG_CACHE_HOME:-} ]]; then
    export XDG_CACHE_HOME="${figure_cache_root}"
fi
mkdir -p -- "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}/fontconfig"

log_root=""
for arg in "$@"; do
    case ${arg} in
        -h|--help) sed -n '2,22p' "${BASH_SOURCE[0]:-$0}"; exit 0 ;;
        -*) echo "unknown option: ${arg}" >&2; exit 2 ;;
        *) log_root=${arg} ;;
    esac
done
log_root=${log_root:-${repo_root}/case_studies/logs/math500}

# Edit this list to render only some of the figures. Each entry is a script in the draw directory that takes --log and
# writes beside it.
drawing_scripts=(
    "draw_resample_vs_ratio.py"
    "draw_acceptance_panels.py"
)
vllm_drawing_scripts=(
    "draw_vllm_engine_metrics.py"
)

if [[ ! -x ${python_bin} ]]; then
    echo "missing drawing Python: ${python_bin}" >&2
    echo "create it and install: ${drawing_dir}/requirements.txt" >&2
    exit 1
fi

if [[ ! -d ${log_root} ]]; then
    echo "no such directory: ${log_root}" >&2
    exit 1
fi
log_root=$(cd -- "${log_root}" && pwd)

# Collect once so every drawing script operates on exactly the same logs. Recursive discovery covers layouts such as
# <date>/vllm/<run>.log.
logs=()
while IFS= read -r -d '' log; do
    logs+=("${log}")
done < <(find "${log_root}" -type f -name "${log_glob}" -print0 | sort -z)

if [[ ${#logs[@]} -eq 0 ]]; then
    echo "no ${log_glob} files under ${log_root}" >&2
    exit 1
fi

total=0
failures=0
failed_names=()
aggregate_failures=0
started=$SECONDS

run_directory_scripts() {
    local label=$1
    local directory=$2
    shift 2
    echo "==> ${label}"
    for script in "$@"; do
        if ! (cd -- "${repo_root}" &&
              "${python_bin}" "${drawing_dir}/${script}" \
                  --directory "${directory}"); then
            echo "    FAILED: ${script}" >&2
            aggregate_failures=1
        fi
    done
}

for log in "${logs[@]}"; do
    total=$((total + 1))
    name=$(basename -- "${log}")
    echo "==> ${name}"
    log_started=$SECONDS
    ok=1
    scripts_for_log=("${drawing_scripts[@]}")
    if [[ ${log} == *.subtreePrefetch.vllm.log ]]; then
        scripts_for_log+=("${vllm_drawing_scripts[@]}")
    fi
    for script in "${scripts_for_log[@]}"; do
        if ! (cd -- "${repo_root}" &&
              "${python_bin}" "${drawing_dir}/${script}" --log "${log}"); then
            echo "    FAILED: ${script}" >&2
            ok=0
        fi
    done
    if [[ ${ok} -eq 0 ]]; then
        failures=$((failures + 1))
        failed_names+=("${name}")
    fi
    echo "    ($((SECONDS - log_started))s)"
done

vllm_log_count=$(find "${log_root}" -type f -name '*.subtreePrefetch.vllm.log' | wc -l | tr -d ' ')
if [[ ${vllm_log_count} -gt 0 ]]; then
    run_directory_scripts \
        "aggregate vLLM engine metrics (${vllm_log_count} logs)" \
        "${log_root}" \
        "draw_vllm_engine_metrics.py"
fi

direct_vllm_log_count=$(find "${log_root}" -maxdepth 1 -type f -name '*.subtreePrefetch.vllm.log' | wc -l | tr -d ' ')
if [[ ${direct_vllm_log_count} -gt 0 ]]; then
    run_directory_scripts \
        "aggregate proposal, transition, and time figures (${direct_vllm_log_count} logs)" \
        "${log_root}" \
        "draw_rank_proposal_certainty.py" \
        "draw_rank_realized_mh_transitions.py" \
        "draw_model_calls_step.py" \
        "draw_mean_mh_transitions_and_empirical_time.py"
fi

# PowerMH baselines are not subtree logs, so the per-log sweep above never sees them. Each one gets its own timing
# figure, written beside the log it reads.
power_mh_log_count=$(find "${log_root}" -type f -name '*.powerMH.*.log' | wc -l | tr -d ' ')
if [[ ${power_mh_log_count} -gt 0 ]]; then
    run_directory_scripts \
        "PowerMH baseline timing (${power_mh_log_count} logs)" \
        "${log_root}" \
        "draw_power_mh_time.py"
fi

resource_json_count=$(find "${log_root}" -type f -name '*resources.json' | wc -l | tr -d ' ')
if [[ ${resource_json_count} -gt 0 ]]; then
    run_directory_scripts \
        "aggregate resource usage (${resource_json_count} JSON files)" \
        "${log_root}" \
        "draw_resource_usage.py" \
        "draw_peak_kv_cache.py"
fi

echo
echo "rendered $((total - failures))/${total} log(s) in $((SECONDS - started))s"
if [[ ${failures} -gt 0 ]]; then
    echo "failed:" >&2
    printf '  %s\n' "${failed_names[@]}" >&2
fi
if [[ ${failures} -gt 0 || ${aggregate_failures} -gt 0 ]]; then
    exit 1
fi
