"""Thread custom MH metadata through vLLM's scheduler wire outputs.

Run through a custom-vLLM entry point such as:
    python -m power_sharpening.runners.vllm.run_subtree_prefetching_mh --help
"""

import logging
import os

from .outputs import ModelRunnerOutput, EngineCoreOutput, EngineCoreOutputs



from collections import defaultdict
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.utils import remove_all
from vllm.v1.request import Request, RequestStatus
from vllm.v1.spec_decode.metrics import SpecDecodingStats

from vllm.v1.core.sched.scheduler import Scheduler as BaseScheduler

mh_logger = logging.getLogger("vllm_scheduler")
logger = init_logger(__name__)


class Scheduler(BaseScheduler):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    config = getattr(self, "vllm_config", None)
    if (getattr(config, "additional_config", None) or {}).get("subtree_cache_eviction", False):
      # Install in EngineCore's process too, including multiprocessing spawn.
      from power_sharpening.backends.vllm.cache_eviction import install_engine_core_eviction
      install_engine_core_eviction()

  def update_from_output(
      self,
      scheduler_output: SchedulerOutput,
      model_runner_output: ModelRunnerOutput,
  ) -> dict[int, EngineCoreOutputs]:
    verbose = os.environ.get('MH_LLM_VERBOSE', '0') == '1'

    sampled_token_ids = model_runner_output.sampled_token_ids
    logprobs = model_runner_output.logprobs
    power_logprobs = getattr(model_runner_output, 'power_logprobs', None)
    entropies = getattr(model_runner_output, 'entropies', None)
    log_z_by_temperature = getattr(
        model_runner_output, 'log_z_by_temperature', None)
    prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict
    num_scheduled_tokens = scheduler_output.num_scheduled_tokens
    pooler_outputs = model_runner_output.pooler_output
    num_nans_in_logits = model_runner_output.num_nans_in_logits
    kv_connector_output = model_runner_output.kv_connector_output
    cudagraph_stats = model_runner_output.cudagraph_stats

    # mh_logger.debug("model_runner_output type=%s, logprobs=%s, power_logprobs=%s",
    #                  type(model_runner_output).__name__,
    #                  "SET" if logprobs else None,
    #                  "SET" if power_logprobs else None)

    perf_stats = None
    if self.perf_metrics and self.perf_metrics.is_enabled():
      perf_stats = self.perf_metrics.get_step_perf_stats_per_gpu(
          scheduler_output)

    outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)
    spec_decoding_stats: SpecDecodingStats | None = None
    kv_connector_stats = (kv_connector_output.kv_connector_stats
                          if kv_connector_output else None)
    if kv_connector_stats and self.connector:
      kv_stats = self.connector.get_kv_connector_stats()
      if kv_stats:
        kv_connector_stats = kv_connector_stats.aggregate(kv_stats)

    failed_kv_load_req_ids = None
    if kv_connector_output and kv_connector_output.invalid_block_ids:
      failed_kv_load_req_ids = self._handle_invalid_blocks(
          kv_connector_output.invalid_block_ids)

    # NOTE(woosuk): As len(num_scheduled_tokens) can be up to 1K or more,
    # the below loop can be a performance bottleneck. We should do our best
    # to avoid expensive operations inside the loop.
    stopped_running_reqs: set[Request] = set()
    stopped_preempted_reqs: set[Request] = set()
    for req_id, num_tokens_scheduled in num_scheduled_tokens.items():
      assert num_tokens_scheduled > 0
      if failed_kv_load_req_ids and req_id in failed_kv_load_req_ids:
        continue
      request = self.requests.get(req_id)
      if request is None or request.is_finished():
        continue

      req_index = model_runner_output.req_id_to_index[req_id]
      generated_token_ids = sampled_token_ids[
          req_index] if sampled_token_ids else []
      num_generated_logprob_positions = len(generated_token_ids)

      scheduled_spec_token_ids = (
          scheduler_output.scheduled_spec_decode_tokens.get(req_id))
      if scheduled_spec_token_ids and generated_token_ids:
        num_draft_tokens = len(scheduled_spec_token_ids)
        num_accepted = len(generated_token_ids) - 1
        num_rejected = num_draft_tokens - num_accepted
        if request.num_computed_tokens > 0:
          request.num_computed_tokens -= num_rejected
        if request.num_output_placeholders > 0:
          request.num_output_placeholders -= num_rejected
        spec_decoding_stats = self.make_spec_decoding_stats(
            spec_decoding_stats,
            num_draft_tokens=num_draft_tokens,
            num_accepted_tokens=num_accepted,
            num_invalid_spec_tokens=scheduler_output.num_invalid_spec_tokens,
            request_id=req_id,
        )

      stopped = False
      new_logprobs = None
      new_power_logprobs = None
      new_entropies = None
      new_log_z_by_temperature = None
      new_token_ids = generated_token_ids
      pooler_output = pooler_outputs[req_index] if pooler_outputs else None
      kv_transfer_params = None
      ec_transfer_params = None
      status_before_stop = request.status

      # Check for stop and update request status.
      if new_token_ids:
        new_token_ids, stopped = self._update_request_with_output(
            request,
            new_token_ids,
        )
      elif request.pooling_params and pooler_output is not None:
        request.status = RequestStatus.FINISHED_STOPPED
        stopped = True

      # vLLM counts prompt tokens from the first output's prefill metadata.
      # Take it once, before a completed request releases its cache blocks.
      prefill_stats = None
      if new_token_ids or pooler_output is not None or stopped:
        take_prefill_stats = getattr(request, "take_prefill_stats", None)
        if take_prefill_stats is not None:
          prefill_stats = take_prefill_stats()
          finalize = getattr(prefill_stats, "finalize", None)
          if finalize is not None:
            finalize(self.kv_cache_manager.estimate_cached_tokens(request))

      routed_experts = None
      finish_reason = None
      if stopped:
        get_routed_experts = getattr(self, "_get_routed_experts", None)
        if get_routed_experts is not None:
          routed_experts = get_routed_experts(request)
        finish_reason = request.get_finished_reason()
        finished = self._handle_stopped_request(request)
        if finished:
          kv_transfer_params, ec_transfer_params = self._free_request(request)

        if status_before_stop == RequestStatus.RUNNING:
          stopped_running_reqs.add(request)
        else:
          stopped_preempted_reqs.add(request)

      # Extract sample logprobs if needed.
      if request.sampling_params is not None \
          and request.sampling_params.logprobs is not None and logprobs:
        new_logprobs = logprobs.slice_request(
            req_index,
            num_generated_logprob_positions,
        )
        new_power_logprobs = power_logprobs.slice_request(
            req_index,
            num_generated_logprob_positions,
        ) if power_logprobs is not None else None
        if entropies is not None and power_logprobs is not None:
          offsets = power_logprobs.cu_num_generated_tokens
          start = offsets[req_index] if offsets is not None else req_index
          new_entropies = entropies[start:start + num_generated_logprob_positions]
        if log_z_by_temperature is not None and power_logprobs is not None:
          offsets = power_logprobs.cu_num_generated_tokens
          start = offsets[req_index] if offsets is not None else req_index
          new_log_z_by_temperature = log_z_by_temperature[
              start:start + num_generated_logprob_positions]
        # mh_logger.debug("req_id=%s, new_logprobs=%s, new_power_logprobs=%s",
        #                  req_id,
        #                  "SET" if new_logprobs else None,
        #                  "SET" if new_power_logprobs else None)

      if new_token_ids and self.structured_output_manager.should_advance(
          request):
        struct_output_request = request.structured_output_request
        assert struct_output_request is not None
        assert struct_output_request.grammar is not None
        ok = struct_output_request.grammar.accept_tokens(
            req_id, new_token_ids)
        if not ok:
          logger.warning(
              "Unexpected: grammar rejected tokens %s for request %s.",
              new_token_ids,
              req_id,
          )

      if num_nans_in_logits is not None and req_id in num_nans_in_logits:
        request.num_nans_in_logits = num_nans_in_logits[req_id]

      # Get prompt logprobs for this request.
      prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)
      if new_token_ids or pooler_output is not None \
          or kv_transfer_params or ec_transfer_params or stopped:

        # Add EngineCoreOutput for this Request.
        outputs[request.client_index].append(
            EngineCoreOutput(
                request_id=req_id,
                new_token_ids=new_token_ids,
                finish_reason=finish_reason,
                new_logprobs=new_logprobs,
                new_power_logprobs=new_power_logprobs,
                new_entropies=new_entropies,
                new_log_z_by_temperature=new_log_z_by_temperature,
                new_prompt_logprobs_tensors=prompt_logprobs_tensors,
                pooling_output=pooler_output,
                stop_reason=request.stop_reason,
                events=request.take_events(),
                prefill_stats=prefill_stats,
                kv_transfer_params=kv_transfer_params,
                ec_transfer_params=ec_transfer_params,
                trace_headers=request.trace_headers,
                num_cached_tokens=getattr(request, "num_cached_tokens", 0),
                num_external_computed_tokens=getattr(
                    request, "num_external_computed_tokens", 0),
                routed_experts=routed_experts,
                num_nans_in_logits=request.num_nans_in_logits,
            ))
      else:
        # Invariant: EngineCore returns no partial prefill outputs.
        assert not prompt_logprobs_tensors

    # Remove the stopped requests from the running and waiting queues.
    if stopped_running_reqs:
      self.running = remove_all(self.running, stopped_running_reqs)
    if stopped_preempted_reqs:
      # This is a rare case and unlikely to impact performance.
      self.waiting.remove_requests(stopped_preempted_reqs)

    if failed_kv_load_req_ids and not self.recompute_kv_load_failures:
      requests = [
          self.requests[req_id] for req_id in failed_kv_load_req_ids
      ]
      self.finish_requests(failed_kv_load_req_ids,
                           RequestStatus.FINISHED_ERROR)
      for request in requests:
        outputs[request.client_index].append(
            EngineCoreOutput(
                request_id=request.request_id,
                new_token_ids=[],
                finish_reason=request.get_finished_reason(),
                events=request.take_events(),
                trace_headers=request.trace_headers,
                num_cached_tokens=getattr(request, "num_cached_tokens", 0),
            ))

    # KV Connector: update state for finished KV Transfers.
    if kv_connector_output:
      self._update_from_kv_xfer_finished(kv_connector_output)

    # collect KV cache events from KV cache manager
    events = self.kv_cache_manager.take_events()

    # collect KV cache events from connector
    if self.connector is not None:
      connector_events = self.connector.take_events()
      if connector_events:
        if events is None:
          events = list(connector_events)
        else:
          events.extend(connector_events)

    # publish collected KV cache events
    if events:
      from vllm.distributed.kv_events import KVEventBatch
      import time
      batch = KVEventBatch(ts=time.time(), events=events)
      self.kv_event_publisher.publish(batch)

    # Create EngineCoreOutputs for all clients that have requests with
    # outputs in this step.
    # for client_index, outs in outputs.items():
    #   for o in outs:
    #     mh_logger.debug("EngineCoreOutput type=%s, new_power_logprobs=%s",
    #                      type(o).__name__,
    #                      "SET" if o.new_power_logprobs else None)
    engine_core_outputs = {
        client_index: EngineCoreOutputs(outputs=outs)
        for client_index, outs in outputs.items()
    }
    # for ci, eco in engine_core_outputs.items():
    #   mh_logger.debug("EngineCoreOutputs type=%s, num_outputs=%d",
    #                    type(eco).__name__, len(eco.outputs))

    finished_req_ids = self.finished_req_ids_dict
    if finished_req_ids:
      for client_index, finished_set in finished_req_ids.items():
        if (eco := engine_core_outputs.get(client_index)) is not None:
          eco.finished_requests = finished_set
        else:
          engine_core_outputs[client_index] = EngineCoreOutputs(
              finished_requests=finished_set)
      finished_req_ids.clear()

    if (stats := self.make_stats(
        spec_decoding_stats,
        kv_connector_stats,
        cudagraph_stats,
        perf_stats,
    )) is not None:
      if (eco := next(iter(engine_core_outputs.values()), None)) is None:
        engine_core_outputs[0] = eco = EngineCoreOutputs()
      eco.scheduler_stats = stats

    return engine_core_outputs
