"""Patched vLLM engine (vendored overrides of vLLM internals).

Extends `SamplingParams` with an `alpha` exponent and patches the engine so
it can sample from p^alpha. Selected via `vLLM_Wrapper(engine_type='custom')`.
Do not edit casually: files here mirror vLLM internals and pin to
vllm>=0.18.0.
"""

from .llm import LLM
from .sampling_params import SamplingParams

__all__ = ["LLM", "SamplingParams"]
