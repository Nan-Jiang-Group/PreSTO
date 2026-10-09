"""
Extended vLLM output dataclasses that carry MH scoring metadata.

Each class mirrors a corresponding vLLM base class and adds fields for raw
base-model log-probabilities and optional full-vocabulary predictive entropies.
These extended outputs flow through the same pipeline as standard vLLM outputs
(ModelRunner -> EngineCore -> final CompletionOutput) so downstream MCMC code
can access the scalars without a second forward pass.

Run the CPU wire-format tests with:
    python -m pytest /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/tests/test_vllm_prefill_stats.py
"""

import time
from dataclasses import dataclass
from typing import Any, Mapping

import msgspec
import numpy as np
import torch

from vllm.outputs import CompletionOutput as BaseCompletionOutput
from vllm.v1.outputs import (
    LogprobsLists,
    LogprobsTensors,
    ModelRunnerOutput as BaseModelRunnerOutput,
)
from vllm.v1.worker.gpu.sample.output import SamplerOutput as BaseSamplerOutput
from vllm.logprobs import SampleLogprobs
from vllm.v1.engine import (
    EngineCoreEvent,
    FinishReason,
    UtilityOutput,
)
from vllm.v1.metrics.stats import SchedulerStats

try:
    from vllm.v1.metrics.stats import PrefillStats
except ImportError:
    # vLLM 0.18 reports prompt counts without a separate PrefillStats object.
    PrefillStats = Any


@dataclass
class CompletionOutput(BaseCompletionOutput):
    """Final per-request completion returned to the caller.

    ``power_logprobs`` holds raw base-model token log-probabilities.
    ``entropies`` holds full-vocabulary base-model entropy in nats per token.
    ``log_z_by_temperature[t][m]`` holds logsumexp(log p / T_m) at token t when the engine was built with
    ``log_z_temperatures``.
    """
    power_logprobs: list[SampleLogprobs] | None = None
    entropies: list[float] | None = None
    log_z_by_temperature: list[list[float]] | None = None


@dataclass
class SamplerOutput(BaseSamplerOutput):
    """GPU-side sampler output with power log-probs still in tensor form.

    The custom field remains on the GPU until the model-runner output is built.
    """
    power_logprobs_tensors: LogprobsTensors | None = None
    entropies_tensors: torch.Tensor | None = None
    log_z_tensors: torch.Tensor | None = None


@dataclass
class ModelRunnerOutput(BaseModelRunnerOutput):
    """Per-step output from the model runner, after GPU -> CPU transfer.

    ``entropies`` follows the flattened rows of ``power_logprobs`` and uses
    its ``cu_num_generated_tokens`` offsets when requests have multiple rows.
    """
    power_logprobs: LogprobsLists | None = None
    entropies: list[float] | None = None
    log_z_by_temperature: list[list[float]] | None = None


# Custom EngineCoreOutput that mirrors vLLM's wire output and adds
# custom MH fields. It is defined explicitly instead of subclassing the upstream
# msgspec.Struct because vLLM dev builds have changed which fields are accepted
# by BaseEngineCoreOutput.
class EngineCoreOutput(
    msgspec.Struct,
    array_like=True,
    omit_defaults=True,
    gc=False,
):
    """Wire-format output from the engine core (msgspec-serialisable).

    Custom fields carry only newly generated positions since the last output.
    """
    request_id: str
    new_token_ids: list[int]

    new_logprobs: LogprobsLists | None = None
    new_power_logprobs: LogprobsLists | None = None
    new_prompt_logprobs_tensors: LogprobsTensors | None = None

    pooling_output: torch.Tensor | None = None

    finish_reason: FinishReason | None = None
    stop_reason: int | str | None = None
    events: list[EngineCoreEvent] | None = None
    kv_transfer_params: dict[str, Any] | None = None
    ec_transfer_params: dict[str, Any] | None = None

    trace_headers: Mapping[str, str] | None = None
    num_cached_tokens: int = 0
    num_external_computed_tokens: int = 0
    routed_experts: np.ndarray | None = None
    num_nans_in_logits: int = 0
    # Keep the upstream type so msgspec restores attributes across engine IPC.
    prefill_stats: PrefillStats | None = None
    new_entropies: list[float] | None = None
    new_log_z_by_temperature: list[list[float]] | None = None

    @property
    def finished(self) -> bool:
        return self.finish_reason is not None


# EngineCoreOutputs is defined from scratch (NOT subclassing the base)
# so that its ``outputs`` field type references our custom EngineCoreOutput.
# This is necessary because msgspec resolves type annotations at class
# creation time, and a subclass cannot override a parent's field type.
class EngineCoreOutputs(
    msgspec.Struct,
    array_like=True,
    omit_defaults=True,
    gc=False,
):
    """Batch of ``EngineCoreOutput`` items returned from a single engine step."""
    engine_index: int = 0
    outputs: list[EngineCoreOutput] = []
    scheduler_stats: SchedulerStats | None = None
    timestamp: float = 0.0
    utility_output: UtilityOutput | None = None
    finished_requests: set[str] | None = None
    wave_complete: int | None = None
    start_wave: int | None = None

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.monotonic()


# Sentinel used when the model runner produces no outputs (e.g. prefill-only step).
EMPTY_MODEL_RUNNER_OUTPUT = ModelRunnerOutput(
    req_ids=[],
    req_id_to_index={},
)
