"""Compute proposal log-probabilities, base-model log-probabilities, entropy, and optional logZ_T.

Run the vLLM Power MH entry point with
``python -m power_sharpening.runners.vllm.run_power_mh --help``.
"""

import logging
import torch

from vllm.config.model import PROCESSED_LOGPROBS_MODES
from vllm.v1.worker.gpu.input_batch import (
    InputBatch,
    get_num_sampled_and_rejected,
)
from vllm.v1.worker.gpu.metrics.logits import get_num_nans
from vllm.v1.worker.gpu.sample.logprob import compute_topk_scores
from vllm.v1.worker.gpu.sample.sampler import Sampler as BaseSampler
from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS

from .outputs import SamplerOutput
from .entropy import predictive_entropies
from .log_normalizers import temperature_log_normalizers

logger = logging.getLogger("vllm_sampler")


class Sampler(BaseSampler):

  def __init__(
      self,
      max_num_reqs: int,
      vocab_size: int,
      *args,
      verbose: bool = False,
      log_z_temperatures: tuple[float, ...] | None = None,
      **kwargs,
  ):
    super().__init__(max_num_reqs, vocab_size, *args, **kwargs)
    self.verbose = verbose
    # Temperatures whose full-vocabulary normalizers accompany scored tokens.
    self.log_z_temperatures = log_z_temperatures

  def __call__(
      self,
      logits: torch.Tensor,
      input_batch: InputBatch,
  ) -> SamplerOutput:
    expanded_idx_mapping = input_batch.expanded_idx_mapping
    idx_mapping_np = input_batch.idx_mapping_np
    cu_num_logits_np = input_batch.cu_num_logits_np
    expanded_local_pos = input_batch.expanded_local_pos
    pos = input_batch.positions[input_batch.logits_indices]
    input_ids = input_batch.input_ids[input_batch.logits_indices]

    # MCMC needs the proposal/model logprobs before temperature, penalties,
    # grammar, or other sampling processors alter the logits.
    power_logits = logits.to(torch.float32).clone()

    # Match vLLM's log-probability request handling. In 0.27+, callers may
    # request explicit token IDs even when top-k logprobs are disabled.
    num_nans = get_num_nans(logits) if self.compute_nans else None
    max_num_logprobs = self.sampling_states.max_num_logprobs(idx_mapping_np)
    max_per_req_token_ids = self.logprob_token_ids_state.max_num_token_ids(
        idx_mapping_np
    )
    return_logprobs = max_num_logprobs != NO_LOGPROBS or max_per_req_token_ids > 0

    sampled, processed_logits = self.sample(
        logits,
        expanded_idx_mapping,
        idx_mapping_np,
        pos,
        input_ids,
        expanded_local_pos,
        return_logprobs=return_logprobs,
    )

    if return_logprobs:
      if self.logprobs_mode in PROCESSED_LOGPROBS_MODES:
        logits = processed_logits
      expanded_logits = logits.shape[0] != idx_mapping_np.shape[0]
      cu_num_logits = cu_num_logits_np.tolist() if expanded_logits else None
      num_logprobs = max_num_logprobs if max_num_logprobs != NO_LOGPROBS else 0
      score_kwargs = {
          "logprob_token_ids_state": self.logprob_token_ids_state,
          "expanded_idx_mapping": expanded_idx_mapping,
          "max_per_req_token_ids": max_per_req_token_ids,
      }
      logprobs_tensors = compute_topk_scores(
          logits,
          num_logprobs,
          sampled,
          cu_num_logits,
          logits_mode=self.logprobs_mode in ("raw_logits", "processed_logits"),
          **score_kwargs,
      )
      power_logprobs_tensors = compute_topk_scores(
          power_logits,
          num_logprobs,
          sampled,
          cu_num_logits,
          logits_mode=False,
          **score_kwargs,
      )
      entropies_tensors = predictive_entropies(power_logits)
      log_z_tensors = (
          temperature_log_normalizers(power_logits, self.log_z_temperatures)
          if getattr(self, "log_z_temperatures", None) else None
      )
    else:
      logprobs_tensors = None
      power_logprobs_tensors = None
      entropies_tensors = None
      log_z_tensors = None

    num_sampled, num_rejected = get_num_sampled_and_rejected(
        input_batch.seq_lens.new_ones(input_batch.num_reqs),
        input_batch.seq_lens,
        input_batch.cu_num_logits,
        input_batch.idx_mapping,
        self.req_states.prefill_len.gpu,
    )

    sampler_output = SamplerOutput(
        sampled_token_ids=sampled.view(-1, 1),
        logprobs_tensors=logprobs_tensors,
        num_nans=num_nans,
        num_sampled=num_sampled,
        num_rejected=num_rejected,
        power_logprobs_tensors=power_logprobs_tensors,
        entropies_tensors=entropies_tensors,
        log_z_tensors=log_z_tensors,
    )
    logger.debug("max_num_logprobs=%s, logprobs_tensors=%s, "
                 "power_logprobs_tensors=%s",
                 max_num_logprobs,
                 "SET" if logprobs_tensors else None,
                 "SET" if power_logprobs_tensors else None)
    return sampler_output
