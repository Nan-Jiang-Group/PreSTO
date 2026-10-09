"""Read run settings out of subtree-prefetch filenames and name the outputs.

Every run in this case study encodes its configuration in a dot-delimited filename, so grouping, ordering, and output
naming are all filename questions. Keeping them in one module means the drawing CLIs agree on what a rank token looks
like and where a figure lands.

This library is not run directly; invoke a sibling drawing CLI instead.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from pathlib import Path

LOG_GLOB = "*.subtreePrefetch.*.log"
VLLM_LOG_GLOB = "*.subtreePrefetch.vllm.log"
BACKEND_LOG_SUFFIXES = (
    ".subtreePrefetch.hf.log",
    ".subtreePrefetch.sglang.log",
    ".subtreePrefetch.vllm.log",
)
BACKEND_DISPLAY_NAMES = {
    "hf": "HF",
    "sglang": "SGLang",
    "vllm": "vLLM",
}
# Figure labels for models whose HuggingFace repository name reads poorly in a title. Keyed by the ``model_of`` alias,
# so a figure says ``Gemma4-12B-it`` rather than the repository name. Aliases absent here keep the
# repository's own basename, which already reads as intended for the Qwen releases.
MODEL_DISPLAY_NAMES = {
    "gemma-12b-it": "Gemma4-12B-it",
}

# Tokens a filename ends with that describe the file, not the run.
NON_CONFIG_NAME_TOKENS = frozenset(
    {"log", "out", "txt", "hf", "sglang", "vllm", "subtreeprefetch", "powermh"}
)

RANK_TOKEN_PREFIX = "rank-"
# Runs and figures alike are named with the manuscript's prefetch budget. Runs launched before the sampler's
# ``subtree_batch_size`` was renamed spell it ``subtree_batch_size-<n>``; that spelling is still read wherever a name is
# parsed, and never written.
PREFETCH_BUDGET_TOKEN_PREFIX = "prefetch-budget-"
LEGACY_BATCH_SIZE_TOKEN_PREFIX = "subtree_batch_size-"
BATCH_SIZE_TOKEN_PREFIXES = (
    PREFETCH_BUDGET_TOKEN_PREFIX,
    LEGACY_BATCH_SIZE_TOKEN_PREFIX,
)
# The tokens that identify a comparison point rather than its configuration.
COMPARISON_TOKEN_PREFIXES = (RANK_TOKEN_PREFIX, *BATCH_SIZE_TOKEN_PREFIXES)
# The output tag for a figure that combines every prefetch budget it found.
ALL_PREFETCH_BUDGETS_TAG = "all-prefetch-budget"

RANK_TOKEN_PATTERN = re.compile(rf"(?:^|\.){RANK_TOKEN_PREFIX}(?P<rank>[^.]+)(?:\.|$)")
LEGACY_BATCH_SIZE_TOKEN = re.compile(r"^batch(?P<batch_size>\d+)$")

# Display and iteration order shared by every rank-comparison figure.
RANK_ORDER = {
    "accept_first": 0,
    "reject_first": 1,
    "longest_first": 2,
    "smallest_cut_diff": 3,
    "longest_path_first": 4,
    "bfs_accept_first": 5,
    "bfs_reject_first": 6,
    # The predictive dynamic program (power_sharpening.common.prefetch_subtree_dynamic_programming), not a RANK_FNS rule.
    "predictive_dp": 7,
}

# Names used before the traversal rules were renamed; logs under them survive.
LEGACY_RANK_NAMES = {
    "left_first": "accept_first",
    "right_first": "reject_first",
}

# Traversal rules kept out of the comparison figures. longest_first costs about
# 2.5x the next slowest rule per transition, so it does not merely read badly:
# it sets the shared axis limits and squeezes the other five rules into a corner of every panel. Its logs, resource
# files, and per-log figures are all still produced; only the side-by-side comparisons drop it. Empty this set to bring
# it back -- nothing else needs changing.
EXCLUDED_RANKS = frozenset({"longest_first"})


def _name_of(log: Path | str) -> str:
    return log.name if isinstance(log, Path) else log


def rank_of(log: Path | str) -> str:
    """Read the traversal-rank function from a run filename."""
    name = _name_of(log)
    match = RANK_TOKEN_PATTERN.search(name)
    if match is None:
        raise ValueError(f"no 'rank-<fn>' token in {name!r}")
    return match.group("rank")


def canonical_rank(rank_fn: str) -> str:
    """Map a historical traversal-rank name onto its current name."""
    return LEGACY_RANK_NAMES.get(rank_fn, rank_fn)


def batch_size_in_token(token: str) -> int | None:
    """Return the batch size a token encodes, in any spelling, or None."""
    for prefix in BATCH_SIZE_TOKEN_PREFIXES:
        if token.startswith(prefix):
            value = token.removeprefix(prefix)
            return int(value) if value.isdigit() else None
    legacy = LEGACY_BATCH_SIZE_TOKEN.match(token)
    return None if legacy is None else int(legacy.group("batch_size"))


def prefetch_budget_token(batch_size: int) -> str:
    """Spell one prefetch budget the way every output file names it."""
    return f"{PREFETCH_BUDGET_TOKEN_PREFIX}{batch_size}"


def rename_budget_token(name: str) -> str:
    """Respell a run name's batch-size token as an output prefetch budget."""
    return ".".join(
        token
        if (batch_size := batch_size_in_token(token)) is None
        else prefetch_budget_token(batch_size)
        for token in name.split(".")
    )


def batch_size_in(name: str) -> int | None:
    """Return the prefetch budget a dotted name encodes, or None."""
    for token in name.split("."):
        if (batch_size := batch_size_in_token(token)) is not None:
            return batch_size
    return None


DATASET_TOKEN_PATTERN = re.compile(
    r"(?:^|\.)dataset-(?P<dataset>.+?)\.model-"
)
MODEL_TOKEN_PATTERN = re.compile(r"(?:^|\.)model-(?P<model>[^.]+(?:\.[^.]*[a-z]\w*)*?)\.alpha")


def dataset_of(log: Path | str) -> str:
    """Read the benchmark dataset from a run filename."""
    name = _name_of(log)
    match = DATASET_TOKEN_PATTERN.search(name)
    if match is None:
        raise ValueError(f"no 'dataset-<name>.model-' token in {name!r}")
    return match.group("dataset")


def model_of(log: Path | str) -> str:
    """Read the base model from a run filename.

    Some aliases carry a dot of their own (``qwen3.5-4b``), so the token runs up to the ``.alpha`` that always follows
    it rather than to the first dot.
    """
    name = _name_of(log)
    match = MODEL_TOKEN_PATTERN.search(name)
    if match is None:
        raise ValueError(f"no 'model-<name>.alpha' token in {name!r}")
    return match.group("model")


def model_display_name(log: Path | str, resolved: str) -> str:
    """Title-case the base model for a figure, preferring an explicit label over the repository basename.

    ``resolved`` is the name read from the run log, which is the HuggingFace repository's basename. It is returned
    unchanged unless the run's alias has an entry in ``MODEL_DISPLAY_NAMES``.
    """
    try:
        alias = model_of(log)
    except ValueError:
        return resolved
    return MODEL_DISPLAY_NAMES.get(alias, resolved)


def dataset_model_prefix(log: Path | str) -> str:
    """Return the output stem naming the experiment a figure summarizes.

    Figures that pool every prefetch budget or every traversal rule have no one budget or rule to name themselves after,
    so they are named for the dataset and base model they hold fixed instead. One directory holding two such experiments
    needs more than this to stay unique; those callers fall back to the full run configuration.
    """
    return f"dataset-{dataset_of(log)}.model-{model_of(log)}"


def backend_of(log: Path | str) -> str:
    """Read and format the serving backend from a run filename."""
    name = _name_of(log)
    for backend, display_name in BACKEND_DISPLAY_NAMES.items():
        if name.endswith(f".{backend}.log"):
            return display_name
    raise ValueError(f"no supported '.<backend>.log' suffix in {name!r}")


def shared_run_config_of(log: Path | str) -> str:
    """Return settings PowerMH and SubTreeMH must share for comparison.

    The sampler, backend, rank, and prefetch-budget tokens describe how a run was executed or where it is plotted.
    Removing only those tokens leaves the dataset, model, and common experimental settings that must match before a
    timing trajectory can be used as a baseline.
    """
    config_name = config_output_prefix(Path(_name_of(log))).name
    return strip_tokens(config_name)


def prefetch_budget_of(log: Path | str) -> int:
    """Read the prefetch budget from a run filename."""
    name = _name_of(log)
    batch_size = batch_size_in(name)
    if batch_size is None:
        raise ValueError(
            f"no 'prefetch-budget-<n>' or 'subtree_batch_size-<n>' token in {name!r}"
        )
    return batch_size


def is_excluded_rank(rank_fn: str) -> bool:
    """Report whether a traversal rule is kept out of the comparison figures."""
    return canonical_rank(rank_fn) in EXCLUDED_RANKS


def excluded_rank_log(log: Path | str) -> bool:
    """Report whether a run filename names an excluded traversal rule."""
    return is_excluded_rank(rank_of(log))


def sort_rank_names(rank_fns: Iterable[str]) -> list[str]:
    """Order traversal ranks the way every comparison figure presents them."""
    return sorted(
        {rank_fn for rank_fn in rank_fns if not is_excluded_rank(rank_fn)},
        key=lambda rank_fn: (RANK_ORDER.get(rank_fn, len(RANK_ORDER)), rank_fn),
    )


def strip_tokens(name: str, prefixes: Sequence[str] = COMPARISON_TOKEN_PREFIXES) -> str:
    """Drop every dot-delimited token carrying one of ``prefixes``.

    What survives is the run configuration two logs must share before their measurements may be drawn against each
    other.
    """
    return ".".join(
        token for token in name.split(".") if not token.startswith(tuple(prefixes))
    )


def strip_batch_size(name: str) -> str:
    """Drop the prefetch-budget token in any spelling.

    Prefix stripping is not enough here: the legacy ``batch16`` spelling shares no prefix with the current
    ``prefetch-budget-16`` one.
    """
    return ".".join(
        token for token in name.split(".") if batch_size_in_token(token) is None
    )


def config_output_prefix(log_path: Path) -> Path:
    """Return an output prefix containing only run-configuration tokens."""
    tokens = log_path.name.split(".")
    while len(tokens) > 1 and tokens[-1].lower() in NON_CONFIG_NAME_TOKENS:
        tokens.pop()
    return log_path.parent / rename_budget_token(".".join(tokens))


def strip_backend_suffix(signature: str) -> str:
    """Drop the backend and ``.log`` tail from a signature used as a tag."""
    for suffix in BACKEND_LOG_SUFFIXES:
        if signature.endswith(suffix):
            return signature.removesuffix(suffix)
    return signature.removesuffix(".log")


def output_path(output_prefix: Path, suffix: str) -> Path:
    """Append a descriptive suffix without replacing a dotted prefix."""
    return output_prefix.parent / f"{output_prefix.name}{suffix}"


def latex_text(value: str) -> str:
    """Escape underscores in identifiers rendered through LaTeX."""
    return value.replace("_", r"\_")


def rank_label(rank_fn: str) -> str:
    """Render a traversal-rank name as readable, LaTeX-safe figure text."""
    return latex_text(rank_fn.replace("_", " "))
