"""
Extended output processor that threads MH log-probs and predictive entropies
through the vLLM output pipeline.

The base vLLM ``OutputProcessor`` builds ``RequestState`` objects, which in
turn build ``CompletionOutput`` objects.  Neither knows about power log-probs.
This module subclasses both ``RequestState`` and ``OutputProcessor`` and uses
runtime monkey-patching (via ``mh_patch``) to inject the custom subclasses
into the base vLLM call chain so that power log-probs are carried end-to-end
without forking the upstream code.

**MH-LLM additions are marked with ``# [MH-LLM]`` comments below.**

Run through ``python -m power_sharpening.runners.vllm.run_power_mh --help``.
"""

import logging
import os

import numpy as np

mh_logger = logging.getLogger("vllm_output_processor")

from vllm.sampling_params import RequestOutputKind
from vllm.tokenizers import TokenizerLike
from vllm.v1.engine import EngineCoreRequest, FinishReason
from vllm.v1.engine.output_processor import (
    OutputProcessor as BaseOutputProcessor,
    RequestOutputCollector,
    RequestState as BaseRequestState,
)
from vllm.v1.engine.parallel_sampling import ParentRequest

from .logprobs import LogprobsProcessor
from .outputs import CompletionOutput
from .utils import patch as mh_patch


class RequestState(BaseRequestState):
  """Per-request state that knows how to build ``CompletionOutput`` with
  custom MH metadata attached."""

  def _new_completion_output(
      self,
      token_ids: list[int],
      finish_reason: FinishReason | None,
      stop_reason: int | str | None,
      routed_experts: np.ndarray | None = None,
  ) -> CompletionOutput:
    """Build a ``CompletionOutput`` that includes custom MH metadata.

    This overrides the base implementation to:
    1. Read ``power_logprobs`` from our extended ``LogprobsProcessor``.
       # [MH-LLM]
    2. Return our extended ``CompletionOutput`` (which has a
       ``power_logprobs`` field) instead of the base class.  # [MH-LLM]
    """

    assert self.detokenizer is not None
    assert self.logprobs_processor is not None
    finished = finish_reason is not None
    delta = self.output_kind == RequestOutputKind.DELTA

    # --- text / token_ids (unchanged from base) ---
    text = self.detokenizer.get_next_output_text(finished, delta)
    if not delta:
      token_ids = self.detokenizer.output_token_ids

    # --- logprobs (unchanged from base) ---
    logprobs = self.logprobs_processor.logprobs

    # [MH-LLM] Retrieve power-distribution log-probs from the extended
    # LogprobsProcessor.  These are populated during decoding by
    # LogprobsProcessor._update_sample_logprobs().
    power_logprobs = self.logprobs_processor.power_logprobs
    entropies = self.logprobs_processor.entropies
    log_z_by_temperature = getattr(self.logprobs_processor, "log_z_by_temperature", None)

    if delta and logprobs:
      logprobs = logprobs[-len(token_ids):]
      # [MH-LLM] Slice power_logprobs in the same way as regular logprobs
      # so delta outputs stay aligned.
      if power_logprobs is not None:
        power_logprobs = power_logprobs[-len(token_ids):]

    if entropies is not None:
      entropies = entropies[:len(self.detokenizer.output_token_ids)]
      if delta:
        entropies = entropies[-len(token_ids):] if token_ids else []

    if log_z_by_temperature is not None:
      log_z_by_temperature = log_z_by_temperature[:len(self.detokenizer.output_token_ids)]
      if delta:
        log_z_by_temperature = log_z_by_temperature[-len(token_ids):] if token_ids else []

    mh_logger.debug("_new_completion_output: logprobs=%s, power_logprobs=%s",
                     len(logprobs) if logprobs else None,
                     len(power_logprobs) if power_logprobs else None)

    # [MH-LLM] Return our extended CompletionOutput which carries the
    # extra ``power_logprobs`` field alongside the standard fields.
    return CompletionOutput(
        index=self.request_index,
        text=text,
        token_ids=token_ids,
        routed_experts=routed_experts,
        logprobs=logprobs,
        power_logprobs=power_logprobs,          # [MH-LLM] new field
        entropies=entropies,
        log_z_by_temperature=log_z_by_temperature,
        cumulative_logprob=self.logprobs_processor.cumulative_logprob,
        finish_reason=str(finish_reason) if finished else None,
        stop_reason=stop_reason if finished else None,
    )

  @classmethod
  def from_new_request(
      cls,
      tokenizer: TokenizerLike | None,
      request: EngineCoreRequest,
      prompt: str | None,
      parent_req: ParentRequest | None,
      request_index: int,
      queue: RequestOutputCollector | None,
      log_stats: bool,
      stream_interval: int,
  ) -> "RequestState":
    """Create a new ``RequestState``.

    # [MH-LLM] We monkey-patch the ``LogprobsProcessor`` class inside the
    # base vLLM module so that when ``super().from_new_request()`` internally
    # calls ``LogprobsProcessor.from_new_request()``, it instantiates *our*
    # extended ``LogprobsProcessor`` (which tracks power log-probs) instead
    # of the base one.  The patch is scoped to this call only.
    """
    with mh_patch([{
        'module': 'vllm.v1.engine.output_processor',
        'class': LogprobsProcessor,
    }]):
      return super().from_new_request(
          tokenizer,
          request,
          prompt,
          parent_req,
          request_index,
          queue,
          log_stats,
          stream_interval,
      )


class OutputProcessor(BaseOutputProcessor):
  """Extended output processor that injects our ``RequestState`` into the
  base vLLM pipeline."""

  def add_request(
      self,
      request: EngineCoreRequest,
      prompt: str | None,
      parent_req: ParentRequest | None = None,
      request_index: int = 0,
      queue: RequestOutputCollector | None = None,
  ) -> None:
    """Register a new request with the output processor.

    # [MH-LLM] We monkey-patch ``RequestState`` in the base vLLM module so
    # that ``super().add_request()`` creates *our* ``RequestState`` (which
    # builds ``CompletionOutput`` objects with power log-probs) instead of
    # the base one.  The patch is scoped to this call only.
    """
    with mh_patch([{
        'module': 'vllm.v1.engine.output_processor',
        'class': RequestState,
    }]):
      super().add_request(
          request,
          prompt,
          parent_req,
          request_index,
          queue,
      )
