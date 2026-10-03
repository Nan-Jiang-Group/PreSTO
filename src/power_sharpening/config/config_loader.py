"""Shared YAML-config loading for the power_sharpening runners.

Every runner (``run_custom_smc.py``, ``run_power_mh.py``, ``run_calderhead_mh.py``, ``run_low_temp.py``) selects a
benchmark with ``--dataset`` and a sampler with ``--algorithm``, then reads its hyperparameters
from ``src/power_sharpening/config/{dataset,algorithms}.yaml``. ``--override KEY=VALUE``
tokens change any merged value from the CLI.

Merge precedence (low -> high): the algorithm ``default`` block (shared alpha / temperature / max_new_tokens), the
selected algorithm block, dataset ``defaults`` (seed / prefix_cache), then the selected dataset entry (task / batch_size
/ dataset options). ``--override`` is applied last. See ``src/power_sharpening/config/algorithms.yaml`` for the
authoritative description.
"""
from pathlib import Path

import yaml

from power_sharpening.common.temp_scheduler import SCHEDULE_TYPES
from power_sharpening.tasks.constants import MAX_NEW_TOKENS
from power_sharpening.tasks.registry import TASKS

# The config files live next to this module.
CONFIG_DIR_DEFAULT = Path(__file__).resolve().parent


def parse_overrides(items):
    """Parse ``KEY=VALUE`` override tokens into a typed dict.

    Values pass through ``yaml.safe_load`` so ints/floats/bools/None/strings and inline lists are inferred (e.g.
    ``n_particles=32`` -> int, ``prefix_cache=false`` -> bool, ``best_of_N=[4,2,1]`` -> list).
    """
    overrides = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(
                f"--override expects KEY=VALUE tokens, got {item!r}."
            )
        key, raw = item.split("=", 1)
        overrides[key.strip()] = yaml.safe_load(raw)
    return overrides


def load_run_config(config_dir, dataset, algorithm):
    """Merge dataset.yaml + algorithms.yaml into one config dict."""
    config_dir = Path(config_dir)
    with open(config_dir / "dataset.yaml") as handle:
        dataset_doc = yaml.safe_load(handle)
    with open(config_dir / "algorithms.yaml") as handle:
        algorithm_doc = yaml.safe_load(handle)

    datasets = dataset_doc.get("datasets", {})
    if dataset not in datasets:
        raise SystemExit(
            f"Unknown dataset {dataset!r}. Choose from: {sorted(datasets)}."
        )
    if algorithm not in algorithm_doc or algorithm == "default":
        choices = sorted(k for k in algorithm_doc if k != "default")
        raise SystemExit(
            f"Unknown algorithm {algorithm!r}. Choose from: {choices}."
        )

    config = {}
    config.update(algorithm_doc.get("default", {}))  # shared alpha/temp/tokens
    config.update(algorithm_doc[algorithm])          # algorithm hyperparameters
    config.update(dataset_doc.get("defaults", {}))   # seed / prefix_cache
    config.update(datasets[dataset])                 # task / batch_size / opts
    return config


def add_config_selection_args(
    parser,
    *,
    algorithm_default,
    algorithm_choices,
    save_str_default,
):
    """Register the CLI args every config-driven runner shares."""
    selection = parser.add_argument_group("config selection")
    selection.add_argument(
        "--dataset",
        required=True,
        help="Dataset key in dataset.yaml (e.g. math500, aime, lcb).",
    )
    selection.add_argument(
        "--algorithm",
        default=algorithm_default,
        choices=algorithm_choices,
        help="Algorithm key selecting the sampler and its config block.",
    )
    selection.add_argument(
        "--config_dir",
        type=Path,
        default=CONFIG_DIR_DEFAULT,
        help="Directory holding dataset.yaml and algorithms.yaml.",
    )

    model_group = parser.add_argument_group("model and output")
    # --model_str is a MODEL_MAP alias; None falls back to the task's default.
    model_group.add_argument("--model_str", default=None)
    model_group.add_argument("--save_str", default=save_str_default)
    model_group.add_argument(
        "--run_name",
        default=None,
        help=(
            "Basename for this run's outputs (<run_name>.csv, .stats.json, "
            ".resources.json). Pass the stem the launcher gives the run's .log "
            "so every artifact of a run shares one prefix -- the case-study "
            "plotters pair files that way. Defaults to a name derived from the "
            "resolved config, which no log matches."
        ),
    )

    override_group = parser.add_argument_group("overrides")
    override_group.add_argument(
        "--override",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Override merged config values, e.g. "
            "--override alpha=3.0 temperature=0.3 max_new_tokens=4096."
        ),
    )
    override_group.add_argument("--verbose", action="store_true")


def resolve_config(args, config_algorithm, required_keys):
    """Load + override + project config onto ``args``; validate; return task.

    ``config_algorithm`` is the algorithms.yaml key to read (which may differ
    from ``args.algorithm`` -- e.g. the low-temp runner maps both ``low_temp``
    and ``best_of_n`` onto the ``low_temp`` block). ``required_keys`` are the config keys the caller needs present after
    the merge.
    """
    config = load_run_config(args.config_dir, args.dataset, config_algorithm)
    config.update(parse_overrides(args.override))
    # Project the merged config onto the args namespace read by the runner and by build_benchmark (task / batch_size /
    # aime_dataset / difficulty / ...).
    for key, value in config.items():
        setattr(args, key, value)

    task = args.task
    if task not in TASKS:
        raise SystemExit(
            f"Dataset {args.dataset!r} maps to task {task!r}, unsupported by "
            f"this runner. Supported tasks: {sorted(TASKS)}."
        )
    # build_benchmark reads max_prompt_chars for lcb; keep the prior default.
    if not hasattr(args, "max_prompt_chars"):
        args.max_prompt_chars = 3500
    # Generation budget falls back to the historical constant if unset.
    if getattr(args, "max_new_tokens", None) is None:
        args.max_new_tokens = MAX_NEW_TOKENS
    # prefix_cache is optional (off unless dataset.yaml / --override sets it).
    args.prefix_cache = bool(getattr(args, "prefix_cache", False))

    missing = [k for k in required_keys if getattr(args, k, None) is None]
    if missing:
        raise SystemExit(
            f"Missing config values after merge: {missing}. Check "
            f"{args.config_dir}/dataset.yaml and algorithms.yaml."
        )
    schedule = getattr(args, "temperature_schedule_type", None)
    if schedule is not None and schedule not in SCHEDULE_TYPES:
        raise SystemExit(
            f"temperature_schedule_type={schedule!r} not in "
            f"{sorted(SCHEDULE_TYPES)}."
        )

    # Fall back to the benchmark class's default model when --model_str is omitted.
    args.model_str = args.model_str or TASKS[task].default_model
    return task
