#!/usr/bin/env python3
"""Turn every scored edge of a prefetch log into one row of pre-proposal features and its acceptance target.

A row is one decision node u whose acceptance child was requested at cut c. The target is the acceptance probability
``A = min(1, exp(log_acc_prob))`` printed on that edge, plus its two easy cases at a threshold epsilon (default
0.01): certain accept ``A >= 1 - epsilon`` and certain reject ``A <= epsilon``, and the uncertain rest
``epsilon < A < 1 - epsilon``. The
features are only what is known before the proposal exists: the cut, the tree position, and the subtree root's
per-token scores around the cut. The root scores come from the ``root log_p / delta_log_p / entropy / delta_entropy``
lines the sampler prints before each step when ``print_tree`` is on; logs without them yield only the cut and tree
features.

Cut c keeps tokens ``[0, c)`` and resamples from token c, so ``log_p[c]`` scores the first token being replaced and
``entropy[c]`` is the entropy of the distribution it is redrawn from. ``delta[t] = v[t + 1] - v[t]``, so the jump into
the cut is ``delta[c - 1]`` and the jump out of it is ``delta[c]``.

Every cut below an accepted node lies in the prefix that node shares with the root, so the point and delta features
at the cut are the held state's own values (at ``c`` equal to the held cut, only the entropy is). The suffix means run
past the held cut and are the root's, not the held state's; ``held_is_root`` marks the rows where they coincide.

Run from the repository root:

    uv run --project case_studies/draw \
        python case_studies/extract/acceptance_features.py \
        --log /absolute/path/to/run.subtreePrefetch.vllm.log \
        --output /absolute/path/to/edges.csv \
        --epsilon 0.01
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

import numpy as np
import pandas as pd

from case_studies.extract.logs.proposal_log import (
    NODE_PATTERN,
    TreeNode,
    _is_root_node_line,
    parse_log,
    unwrap_node_lines,
)
from case_studies.extract.run_naming import strip_backend_suffix

ROOT_SCORES_PATTERN = re.compile(
    r"^root (?P<name>log_p|delta_log_p|entropy|delta_entropy) \(len=(?P<length>\d+)\): (?P<values>\[.*\])\s*$"
)
QUESTION_PATTERN = re.compile(r"^IDX (?P<idx>\d+)\s")
DEFAULT_EPSILON = 0.01

# Columns a predictor may use, grouped by what they need from the log.
CUT_FEATURES = ("cut", "cut_frac", "suffix_len", "held_cut", "held_cut_gap", "depth", "num_accepts")
LIKELIHOOD_FEATURES = ("log_p_at_cut", "delta_log_p_in", "delta_log_p_out", "suffix_log_p_mean")
ENTROPY_FEATURES = ("entropy_at_cut", "delta_entropy_in", "delta_entropy_out", "suffix_entropy_mean")
FEATURES = CUT_FEATURES + LIKELIHOOD_FEATURES + ENTROPY_FEATURES


@dataclass(frozen=True)
class RootScores:
    """The subtree root's per-token scores printed before one MH step."""

    log_p: np.ndarray
    delta_log_p: np.ndarray
    entropy: np.ndarray | None = None
    delta_entropy: np.ndarray | None = None


def parse_root_scores(
    path: Path, *, max_lines: int | None = None
) -> tuple[dict[int, RootScores], dict[int, int]]:
    """Map each printed tree to the root scores printed before it and to its question index.

    Trees are counted exactly as ``parse_log`` counts them, so the returned keys are its ``tree_index`` values.
    """
    scores: dict[int, RootScores] = {}
    questions: dict[int, int] = {}
    pending: dict[str, np.ndarray] = {}
    tree_index = -1
    question: int | None = None

    with path.open(errors="replace") as handle:
        for line in unwrap_node_lines(islice(handle, max_lines)):
            question_match = QUESTION_PATTERN.match(line)
            if question_match:
                question = int(question_match.group("idx"))
                continue

            scores_match = ROOT_SCORES_PATTERN.match(line)
            if scores_match:
                values = np.asarray(json.loads(scores_match.group("values")), dtype=float)
                if values.size != int(scores_match.group("length")):
                    raise ValueError(f"{path}: truncated root {scores_match.group('name')} line")
                pending[scores_match.group("name")] = values
                continue

            match = NODE_PATTERN.search(line)
            if match and _is_root_node_line(line, match.group("id")):
                tree_index += 1
                if question is not None:
                    questions[tree_index] = question
                if "log_p" in pending:
                    scores[tree_index] = RootScores(
                        log_p=pending["log_p"],
                        delta_log_p=pending["delta_log_p"],
                        entropy=pending.get("entropy"),
                        delta_entropy=pending.get("delta_entropy"),
                    )
                pending = {}

    return scores, questions


def _at(values: np.ndarray | None, index: int) -> float:
    """Return ``values[index]``, or NaN when the scores are absent or the index falls off either end."""
    if values is None or not 0 <= index < values.size:
        return math.nan
    return float(values[index])


def _suffix_mean(values: np.ndarray | None, start: int) -> float:
    if values is None or start >= values.size:
        return math.nan
    return float(values[start:].mean())


def edge_features(node: TreeNode, root: RootScores | None) -> dict[str, float]:
    """Features of the request at ``node``'s acceptance child, all known before that proposal is generated."""
    cut = node.proposed_cut_idx
    seq_len = node.seq_len
    row = {
        "cut": cut,
        "cut_frac": cut / seq_len,
        "suffix_len": seq_len - cut,
        "held_cut": node.cut_idx,
        "held_cut_gap": node.cut_idx - cut if node.cut_idx is not None else math.nan,
        "depth": node.depth,
        "num_accepts": node.num_accepts,
    }
    if root is not None and root.log_p.size != seq_len:
        raise ValueError(
            f"tree {node.tree_index}: root log_p has {root.log_p.size} tokens but the tree reports seq_len={seq_len}"
        )
    log_p = root.log_p if root is not None else None
    delta_log_p = root.delta_log_p if root is not None else None
    entropy = root.entropy if root is not None else None
    delta_entropy = root.delta_entropy if root is not None else None
    row.update(
        log_p_at_cut=_at(log_p, cut),
        delta_log_p_in=_at(delta_log_p, cut - 1),
        delta_log_p_out=_at(delta_log_p, cut),
        suffix_log_p_mean=_suffix_mean(log_p, cut),
        entropy_at_cut=_at(entropy, cut),
        delta_entropy_in=_at(delta_entropy, cut - 1),
        delta_entropy_out=_at(delta_entropy, cut),
        suffix_entropy_mean=_suffix_mean(entropy, cut),
    )
    return row


def certainty_classes(accept_prob, epsilon: float = DEFAULT_EPSILON):
    """Return ``(A >= 1 - epsilon, A <= epsilon, epsilon < A < 1 - epsilon)``; exactly one holds for every edge."""
    if not 0.0 < epsilon < 0.5:
        raise ValueError(f"epsilon must lie in (0, 0.5), got {epsilon}")
    certain_accept = accept_prob >= 1.0 - epsilon
    certain_reject = accept_prob <= epsilon
    return certain_accept, certain_reject, ~(certain_accept | certain_reject)


def edge_table(path: Path, *, epsilon: float = DEFAULT_EPSILON, max_lines: int | None = None) -> pd.DataFrame:
    """One row per scored edge in ``path``: identifiers, pre-proposal features, and acceptance targets."""
    parsed = parse_log(path, max_lines=max_lines)
    scores, questions = parse_root_scores(path, max_lines=max_lines)
    run = strip_backend_suffix(path.name)

    rows = []
    for node in parsed.nodes:
        if node.proposed_cut_idx is None:
            continue
        accept_prob = math.exp(node.log_a)
        certain_accept, certain_reject, uncertain = certainty_classes(np.float64(accept_prob), epsilon)
        rows.append(
            {
                "run": run,
                "question": questions.get(node.tree_index, -1),
                "tree_index": node.tree_index,
                "node_id": node.node_id,
                "held_is_root": node.num_accepts == 0,
                "has_root_scores": node.tree_index in scores,
                **edge_features(node, scores.get(node.tree_index)),
                "log_ratio": node.log_ratio,
                "accept_prob": accept_prob,
                "certain_accept": certain_accept,
                "certain_reject": certain_reject,
                "uncertain": uncertain,
                "sampled_decision": node.sampled_decision or "",
            }
        )
    return pd.DataFrame(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write one CSV row of pre-proposal features and acceptance targets per scored prefetch edge."
    )
    parser.add_argument("--log", type=Path, nargs="+", required=True, help="Sampler logs to parse.")
    parser.add_argument("--output", type=Path, required=True, help="CSV to write.")
    parser.add_argument(
        "--epsilon", type=float, default=DEFAULT_EPSILON,
        help="Easy-case threshold: certain accept A >= 1 - epsilon, certain reject A <= epsilon.",
    )
    parser.add_argument("--max-lines", type=int, default=None, help="Stop reading each log after this many lines.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    tables = []
    for log in args.log:
        if not log.is_file():
            raise SystemExit(f"missing log {log}")
        table = edge_table(log, epsilon=args.epsilon, max_lines=args.max_lines)
        print(f"{log.name}: {len(table)} edges, {int(table['has_root_scores'].sum()) if len(table) else 0} with root scores")
        tables.append(table)
    edges = pd.concat(tables, ignore_index=True)
    if edges.empty:
        raise SystemExit("no scored edges with a cut index found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    edges.to_csv(args.output, index=False)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
