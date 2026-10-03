"""Config-driven runner setup: YAML merging and context-window helpers.

``dataset.yaml`` and ``algorithms.yaml`` live in this directory; every runner selects a benchmark with ``--dataset`` and
a sampler with ``--algorithm`` and merges its hyperparameters from these files (see ``config_loader``).
"""

from .config_loader import (
    CONFIG_DIR_DEFAULT,
    add_config_selection_args,
    load_run_config,
    parse_overrides,
    resolve_config,
)

__all__ = [
    "CONFIG_DIR_DEFAULT",
    "add_config_selection_args",
    "load_run_config",
    "parse_overrides",
    "resolve_config",
]
