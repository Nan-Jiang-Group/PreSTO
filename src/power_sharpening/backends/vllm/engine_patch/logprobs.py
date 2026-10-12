"""Accumulate token scores, engine-computed entropy, and optional logZ_T for each request.

Run through ``python -m power_sharpening.runners.vllm.run_power_mh --help``.
"""

import itertools
import logging
from dataclasses import dataclass
from functools import partial

from vllm.logger import init_logger
from vllm.logprobs import SampleLogprobs, create_sample_logprobs
from vllm.tokenizers import TokenizerLike
from vllm.tokenizers.detokenizer_utils import (
    convert_ids_list_to_tokens,
)
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.outputs import LogprobsLists

mh_logger = logging.getLogger("vllm_logprobs")

from vllm.v1.engine.logprobs import (
    LogprobsProcessor as BaseLogprobsProcessor,
    append_logprobs_for_next_position,
)

from .outputs import EngineCoreOutput

logger = init_logger(__name__)

NONES = itertools.repeat(None)
_WARNED_MISSING_POWER_LOGPROBS = False


@dataclass
class LogprobsProcessor(BaseLogprobsProcessor):
  power_logprobs: SampleLogprobs | None = None
  entropies: list[float] | None = None
  # Created by the first output that carries it; None when the engine was built without log_z_temperatures.
  log_z_by_temperature: list[list[float]] | None = None
  verbose: bool = False

  @classmethod
  def from_new_request(
      cls,
      tokenizer: TokenizerLike | None,
      request: EngineCoreRequest,
  ) -> "LogprobsProcessor":
    sampling_params = request.sampling_params
    assert sampling_params is not None
    num_logprobs = sampling_params.logprobs
    cls = partial(
        cls,
        power_logprobs=(
            None if num_logprobs is None
            else create_sample_logprobs(sampling_params.flat_logprobs)
        ),
        entropies=None if num_logprobs is None else [],
        verbose=bool(getattr(sampling_params, '_mh_verbose', False)),
    )
    return BaseLogprobsProcessor.from_new_request.__func__(
        cls,
        tokenizer,
        request,
    )

  def _update_sample_logprobs(
      self,
      logprobs_lists: LogprobsLists,
      power_logprobs_lists: LogprobsLists | None = None,
  ) -> None:
    """Update with sample logprobs from EngineCore.

    Outer lists are only of len > 1 if EngineCore made
    >1 tokens in prior step (e.g. in spec decoding).

    Args:
      logprobs_lists: the lists of logprob tokens, logprobs, and ranks.
      power_logprobs_lists: the lists of power logprob tokens, logprobs,
        and ranks.
    """

    assert self.num_logprobs is not None
    assert self.logprobs is not None
    assert self.cumulative_logprob is not None

    token_ids_lst, logprobs_lst, ranks_lst, _ = logprobs_lists

    power_logprobs_lst = None
    if power_logprobs_lists is not None:
      _, power_logprobs_lst, _, _ = power_logprobs_lists

    for pos, (rank_np, logprobs_np, token_ids_np) in enumerate(zip(
        ranks_lst,
        logprobs_lst,
        token_ids_lst,
    )):
      rank = rank_np.tolist()
      logprobs = logprobs_np.tolist()
      token_ids = token_ids_np.tolist()

      decoded_tokens = NONES if self.tokenizer is None else \
          self._verify_tokens(
              decoded_tokens_list=convert_ids_list_to_tokens(
                  self.tokenizer, token_ids),
              tokens=token_ids,
          )

      # Sampler puts the sampled logprob in first.
      sampled_token_logprob = logprobs[0]
      self.cumulative_logprob += sampled_token_logprob

      # Update with the Logprob dictionary for this pos.
      append_logprobs_for_next_position(
          self.logprobs,
          token_ids,
          logprobs,
          decoded_tokens,
          rank,
          self.num_logprobs,
      )
      if self.power_logprobs is None:
        continue

      power_logprobs_np = None
      if power_logprobs_lst is not None and pos < len(power_logprobs_lst):
        power_logprobs_np = power_logprobs_lst[pos]

      if power_logprobs_np is None:
        global _WARNED_MISSING_POWER_LOGPROBS
        if not _WARNED_MISSING_POWER_LOGPROBS:
          power_len = (
              None if power_logprobs_lst is None
              else len(power_logprobs_lst)
          )
          logger.warning(
              "custom vLLM output did not provide power_logprobs for at "
              "least one generated token; falling back to regular logprobs "
              "for this chunk so SMC can continue. "
              "regular_positions=%d power_positions=%s first_missing_pos=%d",
              len(logprobs_lst),
              power_len,
              pos,
          )
          _WARNED_MISSING_POWER_LOGPROBS = True
        power_lp = logprobs
      else:
        power_lp = power_logprobs_np.tolist()

      append_logprobs_for_next_position(
          self.power_logprobs,
          token_ids,
          power_lp,
          decoded_tokens,
          rank,
          self.num_logprobs,
      )

  def update_from_output(self, output: EngineCoreOutput) -> None:
    # mh_logger.debug("update_from_output: output type=%s, has new_power_logprobs=%s, "
    #                  "new_logprobs=%s, new_power_logprobs=%s",
    #                  type(output).__name__,
    #                  hasattr(output, "new_power_logprobs"),
    #                  "SET" if output.new_logprobs else None,
    #                  "SET" if getattr(output, "new_power_logprobs", None) else None)
    if output.new_logprobs is not None:
      self._update_sample_logprobs(
          output.new_logprobs,
          output.new_power_logprobs,
      )
      if self.entropies is not None:
        if (output.new_entropies is None
            or len(output.new_entropies) != len(output.new_logprobs.logprobs)):
          raise RuntimeError("custom vLLM output must provide one entropy per logprob position")
        self.entropies.extend(output.new_entropies)
      new_log_z = getattr(output, "new_log_z_by_temperature", None)
      if new_log_z is not None:
        if len(new_log_z) != len(output.new_logprobs.logprobs):
          raise RuntimeError("custom vLLM output must provide one logZ row per logprob position")
        if self.log_z_by_temperature is None:
          if len(self.logprobs) != len(new_log_z):
            raise RuntimeError("custom vLLM output omitted logZ rows for earlier positions")
          self.log_z_by_temperature = []
        self.log_z_by_temperature.extend(list(row) for row in new_log_z)
      elif self.log_z_by_temperature is not None:
        raise RuntimeError("custom vLLM output omitted logZ rows for a logprob chunk")
    if output.new_prompt_logprobs_tensors is not None:
      self._update_prompt_logprobs(output.new_prompt_logprobs_tensors)
