"""vLLM-backed samplers.

Run Power-SMC with ``python -m power_sharpening.runners.vllm.run_power_smc --help``.

Two families live here:

* MH/SMC samplers driving the patched engine in
  ``power_sharpening.backends.vllm.engine_patch`` (formerly ``mh_vllm``):
  ``mcmc_power_sampler``, ``mcmc_power_sampler_entropycut``, ``multi_try_mcmc_power_sampler``, and
  ``smc_power_sampler``.
* Samplers running on a stock ``vllm.LLM`` (formerly ``standard_vllm``):
  ``batched_low_temp_sampling`` and ``low_temp_proposal_sampling``.

The subtree-prefetching samplers are deliberately not re-exported: they pull in the proposal-tree machinery, which no
importer of this package needs by default. Import them from ``subtree_prefetching_MH_sampler`` or
``subtree_prefetching_multi_try_mh``.
"""

# Patched-engine samplers.
from .entropycut_power_sampling_mh import mcmc_power_sampler_entropycut
from .multi_try_mh import multi_try_mcmc_power_sampler
from .power_sampling_mh import mcmc_power_sampler
from .smc_sampler import smc_power_sampler

# Stock ``vllm.LLM`` samplers.
from .standard_low_temp_sampler import (
    batched_low_temp_sampling,
    low_temp_proposal_sampling,
)


__all__ = [
    "mcmc_power_sampler",
    "mcmc_power_sampler_entropycut",
    "multi_try_mcmc_power_sampler",
    "smc_power_sampler",
    "batched_low_temp_sampling",
    "low_temp_proposal_sampling",
]
