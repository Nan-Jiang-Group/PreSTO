"""SGLang reimplementation of the transformers-based MH samplers (now ``power_sharpening.backends.hf.samplers``).

The sampling *algorithms* (power-sharpening MCMC and subtree-prefetching MH) are unchanged; only the model backend
differs. The HF ``model.generate`` (which returned both temperature-scaled `scores` and raw `logits` in one call) is
replaced by an SGLang ``Engine``. SGLang exposes only one logprob stream per
pass, so the two streams the algorithm needs are obtained with two passes:

  * proposal  log q(x_t | x_{<t})        -> decode pass at temperature T=1/alpha
                                            (post-temperature `output_token_logprobs`)
  * target    alpha * log p(x_t | x_{<t}) -> scoring pass at temperature 1
                                            (raw `input_token_logprobs`)
"""

from power_sharpening.backends.sglang.wrapper import (
    SGL_LLM_Wrapper,
    sglang_load_model_and_tokenizer,
)
from .low_temp_proposal_sampler import (
    low_temp_proposal_sampling,
    batched_low_temp_proposal_sampling,
)
from .entropy_cut_mh import entropy_cut_mh_sampler

__all__ = [
    "SGL_LLM_Wrapper",
    "sglang_load_model_and_tokenizer",
    "low_temp_proposal_sampling",
    "batched_low_temp_proposal_sampling",
    "entropy_cut_mh_sampler",
]
