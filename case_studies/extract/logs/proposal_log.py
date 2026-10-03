"""Parse subtree-prefetch sampler logs into reusable typed records.

This library is not run directly; invoke a sibling drawing CLI instead.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path

import numpy as np

from case_studies import paths

NEAR_CERTAIN_ACCEPT_PROB = 0.99
EFFECTIVELY_ZERO_ACCEPT_PROB = 0.01
NEAR_CERTAIN_NATS = math.log(NEAR_CERTAIN_ACCEPT_PROB)
EFFECTIVELY_ZERO_NATS = math.log(EFFECTIVELY_ZERO_ACCEPT_PROB)
RATIO_HIST_CLIP = 25.0

# Descending in acceptance probability A.
BUCKET_NAMES = (
    "effectively accepted",
    "genuinely stochastic",
    "effectively rejected",
)
BUCKET_EDGES_ASC = (
    -RATIO_HIST_CLIP,
    EFFECTIVELY_ZERO_NATS,
    NEAR_CERTAIN_NATS,
    0.0,
)
BUCKET_BIN_COUNTS_ASC = (2, 7, 2)

# The ratio field is optional: old logs carried it inside the node record; current logs annotate the accept edge
# instead. EntropyCut logs ``-inf`` for a cut with zero proposal probability (A = 0).
NODE_PATTERN = re.compile(
    r"Node\(id=(?P<id>\d+)"
    r"(?:, log_acc_(?:prob|ratio)=(?P<ratio>None|-inf|-?[\d.]+))?"
    r"(?:, log_A=(?P<log_a>-?[\d.]+))?"
    r"(?:, cut_idx=(?P<cut>-?\d+))?"
    r", seq_len=(?P<seq_len>\d+)\)"
)
EDGE_RATIO_PATTERN = re.compile(
    r"\[accept, log_acc_prob=(?P<ratio>None|-inf|-?[\d.]+)\]"
)
LEAF_ROW_PATTERN = re.compile(
    r"^\s+(?P<leaf>\d+)\s+(?P<exact>[\d.]+)\s+(?P<pathwise>[\d.]+)\s+"
    r"(?P<leafwise>[\d.]+)\s*$"
)
SAMPLED_PATH_PATTERN = re.compile(
    r"^sampled path(?: in the prefetched subtree)?:\s*(?P<walk>\d+.*)$"
)
PATH_STEP_PATTERN = re.compile(
    r"-\[(?P<decision>accept|reject)(?:[^\]]*)\]->\s*(?P<node_id>\d+)"
)
BLOCK_PATTERN = re.compile(r"block \[(?P<block>\d+)/(?P<num_blocks>\d+)\]")
NODE_OPEN_PATTERN = re.compile(r"Node\(id=\d+")
TREE_GUIDE_CHARS = " │├└─"
MAX_WRAP_LINES = 3


@dataclass
class TreeNode:
    """One scored node in a printed proposal tree."""

    tree_index: int
    node_id: int
    log_ratio: float
    log_a: float
    cut_idx: int | None
    seq_len: int
    proposed_cut_idx: int | None = None
    sampled_decision: str | None = None
    block_index: int | None = None

    @property
    def on_sampled_path(self) -> bool:
        return self.sampled_decision is not None

    @property
    def acceptance_bucket(self) -> str:
        """Return the acceptance-certainty bucket for this node."""
        if self.log_a > NEAR_CERTAIN_NATS:
            return "effectively accepted"
        if self.log_a > EFFECTIVELY_ZERO_NATS:
            return "genuinely stochastic"
        return "effectively rejected"

    @property
    def proposed_suffix_len(self) -> int | None:
        """Tokens resampled after the cut, or None without cut data."""
        if self.proposed_cut_idx is None:
            return None
        return self.seq_len - self.proposed_cut_idx

    @property
    def depth(self) -> int:
        return (self.node_id + 1).bit_length() - 1

    @property
    def num_rejects(self) -> int:
        return bin(self.node_id + 1).count("1") - 1

    @property
    def num_accepts(self) -> int:
        return self.depth - self.num_rejects

    @property
    def is_accept_child(self) -> bool:
        """Odd ids are accept children; the root counts as neither."""
        return self.node_id > 0 and self.node_id % 2 == 1


@dataclass
class SampledPath:
    """One realized root-to-leaf walk."""

    tree_index: int
    node_ids: list[int]
    decisions: list[str]
    block_index: int | None = None

    @property
    def num_transitions(self) -> int:
        return len(self.decisions)

    @property
    def num_accepts(self) -> int:
        return sum(1 for decision in self.decisions if decision == "accept")

    @property
    def leaf_id(self) -> int:
        return self.node_ids[-1]

    @property
    def visited_internal_ids(self) -> list[int]:
        """Nodes where a decision was made; the terminal leaf scores nothing."""
        return self.node_ids[:-1]


@dataclass(frozen=True)
class LeafProbability:
    """One optional leaf-probability-table row."""

    tree_index: int
    leaf_id: int
    exact: float
    pathwise: float
    leafwise: float


@dataclass
class ParsedLog:
    """All proposal-tree records parsed from one sampler log."""

    nodes: list[TreeNode]
    leaves: list[LeafProbability]
    paths: list[SampledPath]
    tree_sizes: dict[int, int]
    tree_node_ids: dict[int, set[int]] = field(default_factory=dict)


@dataclass(frozen=True)
class AcceptanceHistogram:
    """Raw histogram counts and bucket boundaries in ascending A order."""

    counts: np.ndarray
    bounds: tuple[int, ...]
    shares: np.ndarray


def _is_root_node_line(line: str, node_id: str) -> bool:
    """Whether a node line opens a new printed tree."""
    tree_prefixes = (" ", "│", "├", "└")
    return node_id == "0" and not line.startswith(tree_prefixes)


def unwrap_node_lines(lines: Iterable[str]) -> Iterator[str]:
    """Rejoin Node records soft-wrapped when an old log was written.

    The opening line keeps its prefix for root detection. Only continuation guides are stripped. An incomplete record is
    emitted unchanged, so a truncated log parses no worse than before.
    """
    pending: list[str] = []

    def flush() -> list[str]:
        held = list(pending)
        pending.clear()
        return held

    for raw in lines:
        line = raw.rstrip("\n")
        if pending:
            candidate = pending[0] + "".join(
                part.lstrip(TREE_GUIDE_CHARS) for part in pending[1:] + [line]
            )
            if NODE_PATTERN.search(candidate):
                pending.clear()
                yield candidate
                continue
            if len(pending) <= MAX_WRAP_LINES and not NODE_OPEN_PATTERN.search(line):
                pending.append(line)
                continue
            yield from flush()
        if NODE_OPEN_PATTERN.search(line) and not NODE_PATTERN.search(line):
            pending.append(line)
            continue
        yield line
    yield from flush()


def parse_log(path: Path, *, max_lines: int | None = None) -> ParsedLog:
    """Parse proposal records, optionally stopping at a complete-question boundary."""
    nodes: list[TreeNode] = []
    leaves: list[LeafProbability] = []
    paths: list[SampledPath] = []
    cuts: dict[tuple[int, int], int] = {}
    edge_ratios: dict[tuple[int, int], float] = {}
    node_records: list[tuple[int, int, re.Match[str]]] = []
    tree_block: dict[int, int] = {}
    tree_sizes: dict[int, int] = {}
    tree_node_ids: dict[int, set[int]] = {}
    tree_index = -1
    block_index: int | None = None

    with path.open(errors="replace") as handle:
        for line in unwrap_node_lines(islice(handle, max_lines)):
            block_match = BLOCK_PATTERN.search(line)
            if block_match:
                block_index = int(block_match.group("block"))
                continue

            match = NODE_PATTERN.search(line)
            if match:
                node_id_text = match.group("id")
                if _is_root_node_line(line, node_id_text):
                    tree_index += 1
                    if block_index is not None:
                        tree_block[tree_index] = block_index
                if tree_index < 0:
                    continue

                tree_sizes[tree_index] = tree_sizes.get(tree_index, 0) + 1
                node_id = int(node_id_text)
                tree_node_ids.setdefault(tree_index, set()).add(node_id)
                cut_text = match.group("cut")
                if cut_text is not None:
                    cuts[(tree_index, node_id)] = int(cut_text)
                edge_match = EDGE_RATIO_PATTERN.search(line)
                if edge_match and edge_match.group("ratio") != "None":
                    parent_id = (node_id - 1) // 2
                    edge_ratios[(tree_index, parent_id)] = float(
                        edge_match.group("ratio")
                    )
                node_records.append((tree_index, node_id, match))
                continue

            path_match = SAMPLED_PATH_PATTERN.match(line.rstrip("\n"))
            if path_match and tree_index >= 0:
                steps = PATH_STEP_PATTERN.findall(path_match.group("walk"))
                paths.append(
                    SampledPath(
                        tree_index=tree_index,
                        node_ids=[0] + [int(node_id) for _, node_id in steps],
                        decisions=[decision for decision, _ in steps],
                        block_index=tree_block.get(tree_index),
                    )
                )
                continue

            leaf_match = LEAF_ROW_PATTERN.match(line.rstrip("\n"))
            if leaf_match and tree_index >= 0:
                leaves.append(
                    LeafProbability(
                        tree_index=tree_index,
                        leaf_id=int(leaf_match.group("leaf")),
                        exact=float(leaf_match.group("exact")),
                        pathwise=float(leaf_match.group("pathwise")),
                        leafwise=float(leaf_match.group("leafwise")),
                    )
                )

    for record_tree_index, node_id, match in node_records:
        ratio_text = match.group("ratio")
        if ratio_text is not None and ratio_text != "None":
            log_ratio = float(ratio_text)
        else:
            edge_ratio = edge_ratios.get((record_tree_index, node_id))
            if edge_ratio is None:
                continue
            log_ratio = edge_ratio

        log_a_text = match.group("log_a")
        cut_text = match.group("cut")
        nodes.append(
            TreeNode(
                tree_index=record_tree_index,
                node_id=node_id,
                log_ratio=log_ratio,
                log_a=(
                    float(log_a_text)
                    if log_a_text is not None
                    else min(0.0, log_ratio)
                ),
                cut_idx=int(cut_text) if cut_text is not None else None,
                seq_len=int(match.group("seq_len")),
                block_index=tree_block.get(record_tree_index),
            )
        )

    for node in nodes:
        node.proposed_cut_idx = cuts.get((node.tree_index, 2 * node.node_id + 1))

    walked = {
        (walk.tree_index, node_id): decision
        for walk in paths
        for node_id, decision in zip(walk.visited_internal_ids, walk.decisions)
    }
    for node in nodes:
        node.sampled_decision = walked.get((node.tree_index, node.node_id))

    return ParsedLog(
        nodes=nodes,
        leaves=leaves,
        paths=paths,
        tree_sizes=tree_sizes,
        tree_node_ids=tree_node_ids,
    )


def bucket_shares(nodes: list[TreeNode]) -> np.ndarray:
    """Share of nodes in each BUCKET_NAMES bucket, in that order."""
    if not nodes:
        raise ValueError("cannot compute acceptance-bucket shares without nodes")
    counts = {name: 0 for name in BUCKET_NAMES}
    for node in nodes:
        counts[node.acceptance_bucket] += 1
    return np.array([counts[name] for name in BUCKET_NAMES], dtype=float) / len(nodes)


def acceptance_histogram(nodes: list[TreeNode]) -> AcceptanceHistogram:
    """Bin log acceptance probabilities within the three certainty buckets."""
    shares = bucket_shares(nodes)
    members = {name: [] for name in BUCKET_NAMES}
    for node in nodes:
        members[node.acceptance_bucket].append(node.log_a)

    counts: list[int] = []
    bounds = [0]
    for index, name in enumerate(reversed(BUCKET_NAMES)):
        left, right = BUCKET_EDGES_ASC[index : index + 2]
        bin_count = BUCKET_BIN_COUNTS_ASC[index]
        values = np.clip(np.asarray(members[name], dtype=float), left, right)
        bucket_counts, _ = np.histogram(
            values,
            bins=np.linspace(left, right, bin_count + 1),
        )
        counts.extend(bucket_counts.tolist())
        bounds.append(bounds[-1] + bin_count)
    return AcceptanceHistogram(
        counts=np.asarray(counts, dtype=float),
        bounds=tuple(bounds),
        shares=shares,
    )
