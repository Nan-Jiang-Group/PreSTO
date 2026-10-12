"""Sampling statistics for subtree-prefetched Multi-Try MH.

Run the CPU checks with:
    python -m pytest /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/tests/test_prefetch_multi_try.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math


def _record_draw(
    stats, work, requests, *,
    num_candidate_pools=None, num_prefetched_nodes=0, limit_hit=False,
):
    """Record backend work and, when pool counts are supplied, prefetch stats.

    Block-extension draws omit the pool counts and record only backend work.
    """
    stats.generation_calls += work.get("generation_calls", 1)
    stats.scoring_calls += work.get("scoring_calls", 0)
    stats.scoring_requests += work.get("scoring_requests", 0)
    requested_tokens = sum(length for _, length, _, _ in requests)
    stats.generated_tokens += work.get("generated_tokens", requested_tokens)
    if num_candidate_pools is not None:
        stats.batch_sizes.append(len(requests))
        stats.bundle_batch_sizes.append(num_candidate_pools)
        stats.proposal_suffix_tokens += requested_tokens
        stats.planned_nodes += num_prefetched_nodes
        stats.deduplicated_bundles += num_prefetched_nodes - num_candidate_pools
        stats.planner_limit_hits += int(limit_hit)


@dataclass
class MultiTryPrefetchStats:
    """Keep transition bundles, suffix requests, and actual model calls distinct."""

    num_tries: int
    prefetch_budget: int
    batch_sizes: list[int] = field(default_factory=list)
    bundle_batch_sizes: list[int] = field(default_factory=list)
    acceptances: list[int] = field(default_factory=list)
    walked_steps: list[int] = field(default_factory=list)
    generation_calls: int = 0
    scoring_calls: int = 0
    scoring_requests: int = 0
    generated_tokens: int = 0
    proposal_suffix_tokens: int = 0
    extension_tokens: int = 0
    extension_suffixes: int = 0
    planned_nodes: int = 0
    deduplicated_bundles: int = 0
    planner_limit_hits: int = 0
    transitions: list[dict] = field(default_factory=list)
    final_base_logprobs: list[float] = field(default_factory=list)
    base_diagnostics: object | None = None

    @property
    def total_workload(self):
        """Refinement suffix requests; excludes block extensions and scoring."""
        return sum(self.batch_sizes)

    @property
    def total_bundles(self):
        return sum(self.bundle_batch_sizes)

    @property
    def total_nfe(self):
        """Actual backend generate calls, including extensions and rescoring."""
        return self.generation_calls + self.scoring_calls

    @property
    def total_acceptances(self):
        return sum(self.acceptances)

    @property
    def total_walked_steps(self):
        return sum(self.walked_steps)

    def acceptance_rate(self, mcmc_steps=None, num_blocks=None):
        # EOS may finish before all configured blocks; use attempted updates.
        return self.total_acceptances / self.total_walked_steps if self.total_walked_steps else 0.0

    def to_json(self, mcmc_steps=None, num_blocks=None):
        logprob_sum = math.fsum(self.final_base_logprobs)
        n_tokens = len(self.final_base_logprobs)
        return {
            "num_tries": self.num_tries,
            "prefetch_budget": self.prefetch_budget,
            "prefetch_budget_units": "suffix_requests",
            "bundle_capacity": self.prefetch_budget // self.num_tries,
            "batch_sizes": self.batch_sizes,
            "bundle_batch_sizes": self.bundle_batch_sizes,
            "acceptances": self.acceptances,
            "walked_steps": self.walked_steps,
            "total_workload": self.total_workload,
            "total_bundles": self.total_bundles,
            "total_nfe": self.total_nfe,
            "total_acceptances": self.total_acceptances,
            "total_walked_steps": self.total_walked_steps,
            "acceptance_rate": self.acceptance_rate(),
            "prefetch_batches": len(self.batch_sizes),
            "generation_calls": self.generation_calls,
            "scoring_calls": self.scoring_calls,
            "scoring_requests": self.scoring_requests,
            "generated_tokens": self.generated_tokens,
            "proposal_suffix_tokens": self.proposal_suffix_tokens,
            "extension_tokens": self.extension_tokens,
            "extension_suffixes": self.extension_suffixes,
            "unused_suffixes": self.total_workload - self.num_tries * self.total_walked_steps,
            "planned_nodes": self.planned_nodes,
            "deduplicated_bundles": self.deduplicated_bundles,
            "planner_limit_hits": self.planner_limit_hits,
            "num_response_tokens": n_tokens,
            "logprob_sum": logprob_sum,
            "log_likelihood": logprob_sum / n_tokens if n_tokens else None,
            "transitions": self.transitions,
        }
