"""CPU regression tests for prompt-throughput metadata in the custom scheduler.

Run with:
    uv run --project /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src \
      pytest /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/tests/test_vllm_prefill_stats.py

The wire round-trip test requires msgspec, a dependency of vLLM.
"""

from dataclasses import dataclass
import importlib.util
import logging
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest


PATCH_DIR = (
    Path(__file__).resolve().parents[1]
    / "power_sharpening" / "backends" / "vllm" / "engine_patch"
)
PACKAGE = "_test_vllm_prefill_patch"


def _stub_module(monkeypatch, name, **attributes):
    module = ModuleType(name)
    module.__dict__.update(attributes)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _load_file(monkeypatch, name, filename):
    spec = importlib.util.spec_from_file_location(name, PATCH_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def _load_scheduler(monkeypatch):
    """Execute the real scheduler, replacing only its unavailable vLLM imports."""
    _stub_module(monkeypatch, PACKAGE, __path__=[str(PATCH_DIR)])
    _stub_module(
        monkeypatch, f"{PACKAGE}.outputs", ModelRunnerOutput=SimpleNamespace,
        EngineCoreOutput=SimpleNamespace, EngineCoreOutputs=SimpleNamespace,
    )
    _stub_module(monkeypatch, "vllm.logger", init_logger=logging.getLogger)
    _stub_module(monkeypatch, "vllm.v1.core.sched.output", SchedulerOutput=object)
    _stub_module(
        monkeypatch, "vllm.v1.core.sched.utils",
        remove_all=lambda items, removed: [item for item in items if item not in removed],
    )
    _stub_module(
        monkeypatch, "vllm.v1.request", Request=object,
        RequestStatus=SimpleNamespace(RUNNING="running", FINISHED_STOPPED="stopped"),
    )
    _stub_module(monkeypatch, "vllm.v1.spec_decode.metrics", SpecDecodingStats=object)
    _stub_module(monkeypatch, "vllm.v1.core.sched.scheduler", Scheduler=object)
    return _load_file(monkeypatch, f"{PACKAGE}.scheduler", "scheduler.py")


class _Request:
    request_id = "request-1"
    client_index = 0
    status = "running"
    pooling_params = None
    sampling_params = None
    stop_reason = None
    trace_headers = None
    num_nans_in_logits = 0
    num_cached_tokens = 32
    num_external_computed_tokens = 8

    def __init__(self, stats, *, legacy=False):
        self.stats = stats
        self.take_calls = 0
        self.cache_freed = False
        if legacy:
            self.take_prefill_stats = None

    def is_finished(self):
        return self.status != "running"

    def get_finished_reason(self):
        return "stop" if self.is_finished() else None

    def take_events(self):
        return None

    def take_prefill_stats(self):
        self.take_calls += 1
        stats, self.stats = self.stats, None
        return stats


def _make_scheduler(module, request, *, stop=False):
    scheduler = module.Scheduler()
    scheduler.requests = {request.request_id: request}
    scheduler.running = [request]
    scheduler.perf_metrics = None
    scheduler.connector = None
    scheduler.finished_req_ids_dict = {}
    scheduler.structured_output_manager = SimpleNamespace(should_advance=lambda req: False)
    scheduler.make_stats = lambda *args: None
    scheduler.cache_estimates = []

    def estimate_cached_tokens(req):
        assert not req.cache_freed, "Prefill accounting must precede cache release"
        scheduler.cache_estimates.append(req.request_id)
        return req.num_cached_tokens

    scheduler.kv_cache_manager = SimpleNamespace(
        take_events=lambda: None, estimate_cached_tokens=estimate_cached_tokens,
    )

    def update_request(req, token_ids):
        if stop:
            req.status = "stopped"
        return token_ids, stop

    def free_request(req):
        req.cache_freed = True
        return None, None

    scheduler._update_request_with_output = update_request
    scheduler._handle_stopped_request = lambda req: True
    scheduler._free_request = free_request
    return scheduler


def _step(scheduler, token_ids, *, pooler_output=None):
    scheduled = SimpleNamespace(
        num_scheduled_tokens={"request-1": 1}, scheduled_spec_decode_tokens={},
    )
    runner_output = SimpleNamespace(
        sampled_token_ids=[token_ids], req_id_to_index={"request-1": 0},
        logprobs=None, prompt_logprobs_dict={}, pooler_output=[pooler_output],
        num_nans_in_logits=None, kv_connector_output=None, cudagraph_stats=None,
    )
    return scheduler.update_from_output(scheduled, runner_output)


@pytest.mark.parametrize("cached_tokens", [0, 32])
def test_first_output_forwards_prefill_counts_once(monkeypatch, cached_tokens):
    scheduler_module = _load_scheduler(monkeypatch)
    stats = SimpleNamespace(
        num_prompt_tokens=128, num_computed_tokens=128 - cached_tokens,
        num_cached_tokens=cached_tokens, finalized_with=[],
    )
    stats.finalize = stats.finalized_with.append
    request = _Request(stats)
    request.num_cached_tokens = cached_tokens
    scheduler = _make_scheduler(scheduler_module, request)

    first = _step(scheduler, [7])[0].outputs[0]
    second = _step(scheduler, [8])[0].outputs[0]

    assert first.prefill_stats is stats
    assert first.prefill_stats.num_computed_tokens == 128 - cached_tokens
    assert first.prefill_stats.num_cached_tokens == cached_tokens
    assert stats.finalized_with == [cached_tokens]
    assert scheduler.cache_estimates == [request.request_id]
    assert second.prefill_stats is None


def test_partial_prefill_retains_stats_until_first_output(monkeypatch):
    scheduler_module = _load_scheduler(monkeypatch)
    stats = SimpleNamespace(num_prompt_tokens=128)
    request = _Request(stats)
    scheduler = _make_scheduler(scheduler_module, request)

    assert _step(scheduler, []) == {}
    assert request.take_calls == 0
    assert request.stats is stats
    assert _step(scheduler, [7])[0].outputs[0].prefill_stats is stats


@pytest.mark.parametrize("pooling", [False, True])
def test_first_output_collects_stats_before_finished_cache_is_freed(monkeypatch, pooling):
    scheduler_module = _load_scheduler(monkeypatch)
    finalized_with = []
    stats = SimpleNamespace(finalize=finalized_with.append)
    request = _Request(stats)
    request.pooling_params = object() if pooling else None
    scheduler = _make_scheduler(scheduler_module, request, stop=not pooling)

    output = _step(
        scheduler, [] if pooling else [7], pooler_output=object() if pooling else None,
    )[0].outputs[0]

    assert output.prefill_stats is stats
    assert output.finish_reason == "stop"
    assert finalized_with == [32]
    assert request.cache_freed
    assert scheduler.running == []


def test_older_prefill_stats_without_finalize_are_forwarded(monkeypatch):
    scheduler_module = _load_scheduler(monkeypatch)
    stats = SimpleNamespace(num_prompt_tokens=128, num_computed_tokens=96)
    request = _Request(stats)
    scheduler = _make_scheduler(scheduler_module, request)

    assert _step(scheduler, [7])[0].outputs[0].prefill_stats is stats
    assert scheduler.cache_estimates == []


def test_legacy_request_retains_existing_cache_counters(monkeypatch):
    scheduler_module = _load_scheduler(monkeypatch)
    request = _Request(None, legacy=True)
    scheduler = _make_scheduler(scheduler_module, request)

    output = _step(scheduler, [7])[0].outputs[0]

    assert output.prefill_stats is None
    assert output.num_cached_tokens == 32
    assert output.num_external_computed_tokens == 8
    assert scheduler.cache_estimates == []


def test_prefill_stats_survive_actual_output_wire_roundtrip(monkeypatch):
    msgspec = pytest.importorskip("msgspec")

    @dataclass
    class PrefillStats:
        num_prompt_tokens: int
        num_computed_tokens: int
        num_cached_tokens: int
        num_local_cached_tokens: int
        num_external_cached_tokens: int

    @dataclass
    class ModelRunnerOutput:
        req_ids: list
        req_id_to_index: dict

    _stub_module(monkeypatch, PACKAGE, __path__=[str(PATCH_DIR)])
    _stub_module(monkeypatch, "vllm.outputs", CompletionOutput=object)
    _stub_module(
        monkeypatch, "vllm.v1.outputs", LogprobsLists=Any, LogprobsTensors=Any,
        ModelRunnerOutput=ModelRunnerOutput,
    )
    _stub_module(monkeypatch, "vllm.v1.worker.gpu.sample.output", SamplerOutput=object)
    _stub_module(monkeypatch, "vllm.logprobs", SampleLogprobs=Any)
    _stub_module(
        monkeypatch, "vllm.v1.engine", EngineCoreEvent=Any, FinishReason=Any,
        UtilityOutput=Any,
    )
    _stub_module(
        monkeypatch, "vllm.v1.metrics.stats", SchedulerStats=Any, PrefillStats=PrefillStats,
    )
    outputs = _load_file(monkeypatch, f"{PACKAGE}.outputs", "outputs.py")
    original = outputs.EngineCoreOutputs(outputs=[outputs.EngineCoreOutput(
        request_id="request-1", new_token_ids=[7],
        prefill_stats=PrefillStats(128, 96, 32, 24, 8),
        new_entropies=[0.7],
    )])

    restored = msgspec.msgpack.decode(
        msgspec.msgpack.encode(original), type=outputs.EngineCoreOutputs,
    )

    stats = restored.outputs[0].prefill_stats
    assert restored.outputs[0].new_entropies == [0.7]
    assert isinstance(stats, PrefillStats)
    assert stats.num_prompt_tokens == 128
    assert stats.num_computed_tokens == 96
    assert stats.num_cached_tokens == 32
    assert stats.num_local_cached_tokens == 24
    assert stats.num_external_cached_tokens == 8
