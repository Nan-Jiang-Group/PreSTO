#!/bin/bash
# Shared vLLM entry point for EntropyCut MH and PreSTO + EntropyCutMH. Run through a caller in this directory, or supply
# --method entropycut-mh or
# --method presto-entropycut-mh, --cut-power, and the standard case-study job options.
# Everything after --override is forwarded as Python key=value settings.

set -eo pipefail

# Slurm runs a spooled copy of this script, so its location cannot identify the checkout: the launchers export
# REPO_ROOT, and a direct run outside Slurm falls back to the checkout holding this script.
repo_root=${REPO_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)}
export REPO_ROOT="$repo_root"

method=""
cut_power=4.0
job_args=()
overrides=()
while (( $# > 0 )); do
    option="$1"
    shift
    case "$option" in
        --method|--cut-power)
            if [[ $# -eq 0 || -z "$1" || "$1" == --* ]]; then
                echo "Missing value for $option" >&2
                exit 2
            fi
            value="$1"
            shift
            case "$option" in
                --method) method="$value" ;;
                --cut-power) cut_power="$value" ;;
            esac
            ;;
        --override)
            overrides=("$@")
            break
            ;;
        *)
            # The existing job body validates datasets, grids, Conda, and log modes.
            job_args+=("$option")
            ;;
    esac
done

case "$method" in
    entropycut-mh)
        job_script="$repo_root/case_studies/scripts/run_power_mh_case_study.sh"
        cut_override="cut_power=$cut_power"
        ;;
    # step-entropycut-mh is the former name, still accepted so jobs spooled before the rename keep dispatching.
    presto-entropycut-mh|step-entropycut-mh)
        method=presto-entropycut-mh
        job_script="$repo_root/case_studies/scripts/run_subtree_prefetching_mh_case_study.sh"
        cut_override="cut_dist_param=$cut_power"
        ;;
    *)
        echo "Supply --method entropycut-mh or --method presto-entropycut-mh." >&2
        exit 2
        ;;
esac

# Both arms must use entropy cuts and a fixed proposal temperature. The two Python runners name the cut exponent
# differently; --cut-power maps it above.
for setting in "${overrides[@]}"; do
    case "$setting" in
        cut_dist_type=*)
            if [[ "$setting" != cut_dist_type=entropy ]]; then
                echo "EntropyCut comparisons require cut_dist_type=entropy." >&2
                exit 2
            fi
            ;;
        temperature_schedule_type=*)
            if [[ "$setting" != temperature_schedule_type=const ]]; then
                echo "EntropyCut requires temperature_schedule_type=const." >&2
                exit 2
            fi
            ;;
        cut_power=*|cut_dist_param=*)
            echo "Use --cut-power before --override to set the exponent for either arm." >&2
            exit 2
            ;;
    esac
done

# Reuse environment setup, sweep loops, resource probing, logging, and failure handling from the existing job bodies
# instead of duplicating them here.
exec bash "$job_script" "${job_args[@]}" \
    --override "${overrides[@]}" \
    cut_dist_type=entropy "$cut_override" temperature_schedule_type=const
