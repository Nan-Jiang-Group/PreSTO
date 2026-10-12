"""Shared sampler utilities.

Run repository scripts with:
    PYTHONPATH=/Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src python <script>
"""

from typing import List

import numpy as np
from dataclasses import dataclass, field


@dataclass(frozen=True)
class BaseModelDiagnostics:
    """Likelihood and confidence of one response under the base model p_0.

    Both statistics are measured under p_0 rather than the sharpened target p_0^alpha, so they stay comparable across
    values of alpha, and both are normalized by response length so responses of different lengths remain comparable.

    Attributes:
        num_tokens: n_K, the number of response tokens scored (the terminal state minus the prompt).
        logprob_sum: sum_t log p_0(x_t | x_0, x_{<t}) over the response.
        neg_entropy_sum: sum_t sum_u p_0(u | x_0, x_{<t}) log p_0(u | x_0, x_{<t}),
            i.e. the negative entropy of the base next-token distributions summed along the response.
    """

    num_tokens: int
    logprob_sum: float
    neg_entropy_sum: float

    def __post_init__(self):
        if self.num_tokens <= 0:
            raise ValueError(
                "base-model diagnostics need at least one scored response "
                f"token, got num_tokens={self.num_tokens}"
            )

    @property
    def log_likelihood(self) -> float:
        """Length-normalized log-likelihood (1/n_K) log p_0(x^(K) | x_0)."""
        return self.logprob_sum / self.num_tokens

    @property
    def confidence(self) -> float:
        """Average negative entropy of the base next-token distributions."""
        return self.neg_entropy_sum / self.num_tokens

    def to_json(self) -> dict:
        return {
            "num_response_tokens": self.num_tokens,
            "logprob_sum": self.logprob_sum,
            "neg_entropy_sum": self.neg_entropy_sum,
            "log_likelihood": self.log_likelihood,
            "confidence": self.confidence,
        }


@dataclass
class SamplingStats:
    """Per-block and aggregate statistics for subtree-prefetching MH sampling."""

    # Per-block accumulators
    batch_sizes: List[int] = field(default_factory=list)
    acceptances: List[int] = field(default_factory=list)
    walked_steps: List[int] = field(default_factory=list)
    base_diagnostics: BaseModelDiagnostics | None = None
    cache_evictions: List[dict] = field(default_factory=list)

    @property
    def total_workload(self) -> int:
        """Total number of proposals generated across all batched calls."""
        return int(np.sum(self.batch_sizes))

    @property
    def total_nfe(self) -> int:
        """Total number of forward evaluations (batched LLM calls)."""
        return len(self.batch_sizes)

    @property
    def total_acceptances(self) -> int:
        return int(np.sum(self.acceptances))

    @property
    def total_walked_steps(self) -> int:
        """Total length of the sampled trajectory across all batched calls."""
        return int(np.sum(self.walked_steps))

    def acceptance_rate(self, mcmc_steps: int, num_blocks: int):
        """Overall MH acceptance rate."""
        return float(np.sum(self.acceptances) / (mcmc_steps * num_blocks))

    def to_json(self, mcmc_steps: int | None = None, num_blocks: int | None = None) -> dict:
        """Serialise statistics to a JSON-compatible dict.

        Args:
            mcmc_steps: If provided (along with *num_blocks*), includes the overall acceptance rate in the output.
            num_blocks: See *mcmc_steps*.
        """
        data = {
            "batch_sizes": self.batch_sizes,
            "acceptances": self.acceptances,
            "total_workload": self.total_workload,
            "total_nfe": self.total_nfe,
            "total_acceptances": self.total_acceptances,
            "total_walked_steps": self.total_walked_steps,
        }
        if mcmc_steps is not None and num_blocks is not None:
            data["acceptance_rate"] = self.acceptance_rate(mcmc_steps, num_blocks)
        if self.base_diagnostics is not None:
            data.update(self.base_diagnostics.to_json())
        if self.cache_evictions:
            data["cache_evictions"] = self.cache_evictions
            data["cache_eviction_seconds"] = sum(item["seconds"] for item in self.cache_evictions)
            for key in ("evicted_blocks", "evicted_tokens"):
                if any(key in item for item in self.cache_evictions):
                    data["cache_" + key] = sum(item.get(key, 0) for item in self.cache_evictions)
        return data

    def __repr__(self) -> str:
        diagnostics = self.base_diagnostics
        likelihood_confidence = (
            "log_likelihood=None, confidence=None"
            if diagnostics is None
            else (
                f"log_likelihood={diagnostics.log_likelihood}, "
                f"confidence={diagnostics.confidence}"
            )
        )
        return (
            f"SamplingStats(total_workload={self.total_workload}, "
            f"total_nfe={self.total_nfe}, "
            f"total_acceptances={self.total_acceptances}, "
            f"total_walked_steps={self.total_walked_steps}, "
            f"{likelihood_confidence})"
        )
