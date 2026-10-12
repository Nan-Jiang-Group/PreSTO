"""CPU checks for generation-only GPU and KV-eviction resource metrics.

Run with:
    uv run --project /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src \
      pytest /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/tests/test_resource_probe.py
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from power_sharpening.common.resource_probe import ResourceProbe, _GIB


_ENGINE_LOGGER = logging.getLogger("vllm.v1.metrics.loggers")
_ENGINE_STATS = (
    "Engine %03d: Avg prompt throughput: %.1f tokens/s, "
    "Avg generation throughput: %.1f tokens/s, Running: %d reqs, "
    "Waiting: %d reqs, GPU KV cache usage: %.1f%%, "
    "Prefix cache hit rate: %.1f%%"
)
_ENGINE_STATS_ARGS = (0, 12.0, 283.4, 1, 0, 0.3, 82.8)


class _FakeLoggerManager:
    def __init__(self, engine_indexes=(0,)):
        self.engine_indexes = list(engine_indexes)

    def log(self, *, emit_stats=True, fail=False, repeats=1):
        _ENGINE_LOGGER.info("Unrelated engine diagnostic: %s", "ready")
        if fail:
            raise RuntimeError("logging failed")
        if emit_stats:
            for _ in range(repeats):
                _ENGINE_LOGGER.info(_ENGINE_STATS, *_ENGINE_STATS_ARGS)
        return "logged"


class _FakeNvmlSampler:
    """Deterministic stand-in for the background NVML sampler."""

    def __init__(self):
        self.peak_bytes = 18 * _GIB
        self.window_peak_bytes = 18 * _GIB
        self.device_peak_bytes = 20 * _GIB
        self.min_bytes = 9 * _GIB
        self.device_min_bytes = 11 * _GIB
        self.current_bytes = 10 * _GIB
        self.loaded_bytes = None
        self.generation_peak_bytes = 0
        self.window_generation_peak_bytes = 0
        self.generation_calls = 0
        self.window_generation_calls = 0
        self.in_generation = False
        self.error = None
        self.stopped = False

    def observe(self, value):
        self.current_bytes = value
        self.peak_bytes = max(self.peak_bytes, value)
        self.window_peak_bytes = max(self.window_peak_bytes, value)
        if self.in_generation:
            self.generation_peak_bytes = max(self.generation_peak_bytes, value)
            self.window_generation_peak_bytes = max(
                self.window_generation_peak_bytes, value
            )

    def reset_after_load(self):
        self.loaded_bytes = self.current_bytes
        self.window_peak_bytes = 0
        self.window_generation_peak_bytes = 0
        self.window_generation_calls = 0
        self.min_bytes = None
        self.device_min_bytes = None

    def begin_generation(self):
        self.in_generation = True
        self.generation_calls += 1
        self.window_generation_calls += 1
        self.observe(self.current_bytes)

    def end_generation(self):
        self.observe(self.current_bytes)
        self.in_generation = False

    def take_window(self):
        peak = self.window_peak_bytes
        self.window_peak_bytes = 0
        return peak

    def take_generation_window(self):
        values = (
            self.window_generation_peak_bytes,
            self.window_generation_calls,
        )
        self.window_generation_peak_bytes = 0
        self.window_generation_calls = 0
        return values

    def stop(self):
        self.stopped = True


class _Block:
    def __init__(self, cached):
        self.cached = cached


class _FakeBlockPool:
    def __init__(self):
        self.evictions_enabled = True

    def _maybe_evict_cached_block(self, block):
        return block.cached and self.evictions_enabled

    def get_new_blocks(self, count):
        blocks = [_Block(index % 2 == 0) for index in range(count)]
        for block in blocks:
            self._maybe_evict_cached_block(block)
        return blocks


class _FakeScheduler:
    def __init__(self):
        self.kv_cache_manager = SimpleNamespace(
            usage=0.75,
            block_pool=_FakeBlockPool(),
        )

    def schedule(self):
        return "scheduled"

    def update_from_output(self):
        return "updated"


class _FakeLlm:
    def __init__(self, nvml, *, fail=False, engine_indexes=(0,), with_scheduler=True):
        self._nvml = nvml
        self._fail = fail
        scheduler = _FakeScheduler()
        core = SimpleNamespace(scheduler=scheduler)
        client = SimpleNamespace(engine_core=core)
        self.llm_engine = SimpleNamespace(
            engine_core=client if with_scheduler else None,
            logger_manager=_FakeLoggerManager(engine_indexes),
        )

    def generate(self, *args, **kwargs):
        self._nvml.observe(12 * _GIB)
        if self._fail:
            raise RuntimeError("generation failed")
        return ["completion"]


def _attached_probe(*, fail=False, engine_indexes=(0,), with_scheduler=True):
    nvml = _FakeNvmlSampler()
    probe = ResourceProbe()
    probe._nvml = nvml
    probe._nvml_ok = True
    llm = _FakeLlm(
        nvml, fail=fail, engine_indexes=engine_indexes, with_scheduler=with_scheduler
    )
    probe.attach(llm)
    return probe, llm, nvml


def test_explicit_subtree_eviction_is_separate_from_allocation_evictions():
    probe, llm, _ = _attached_probe()
    pool = llm.llm_engine.engine_core.engine_core.scheduler.kv_cache_manager.block_pool
    try:
        pool.subtree_eviction_in_progress = True
        pool._maybe_evict_cached_block(_Block(True))
        assert probe._kv.evicted_blocks == 0
        pool.subtree_eviction_in_progress = False
        pool.get_new_blocks(2)
        assert probe._kv.evicted_blocks == 1
    finally:
        probe.close()


def test_generation_delta_excludes_loaded_model_and_startup_window():
    probe, llm, nvml = _attached_probe()

    assert nvml.window_peak_bytes == 0
    assert llm.generate() == ["completion"]
    scheduler = llm.llm_engine.engine_core.engine_core.scheduler
    assert scheduler.schedule() == "scheduled"
    blocks = scheduler.kv_cache_manager.block_pool.get_new_blocks(4)
    assert len(blocks) == 4

    window = probe.mark()
    assert window["peak_gpu_memory_gib"] == pytest.approx(12.0)
    assert window["loaded_gpu_memory_gib"] == pytest.approx(10.0)
    assert window["peak_llm_generate_gpu_memory_gib"] == pytest.approx(12.0)
    assert window["peak_llm_generate_gpu_memory_delta_gib"] == pytest.approx(2.0)
    assert window["llm_generate_calls"] == 1
    assert window["kv_cache_allocated_blocks"] == 4
    assert window["kv_cache_evicted_blocks"] == 2
    assert window["kv_cache_eviction_rate"] == pytest.approx(0.5)

    empty_window = probe.mark()
    assert empty_window["peak_llm_generate_gpu_memory_gib"] is None
    assert empty_window["llm_generate_calls"] == 0
    assert empty_window["kv_cache_allocated_blocks"] == 0
    assert empty_window["kv_cache_evicted_blocks"] == 0
    assert empty_window["kv_cache_eviction_rate"] is None

    summary = probe.summary()
    assert summary["peak_gpu_memory_gib"] == pytest.approx(18.0)
    assert summary["loaded_gpu_memory_gib"] == pytest.approx(10.0)
    assert summary["peak_llm_generate_gpu_memory_gib"] == pytest.approx(12.0)
    assert summary["peak_llm_generate_gpu_memory_delta_gib"] == pytest.approx(2.0)
    assert summary["llm_generate_calls"] == 1
    assert summary["kv_cache_allocated_blocks"] == 4
    assert summary["kv_cache_evicted_blocks"] == 2
    assert summary["kv_cache_eviction_rate"] == pytest.approx(0.5)
    assert summary["generation_gpu_source"] == "llm.generate-nvml-sampled"
    assert summary["eviction_source"] == "block-pool-hook"
    probe.close()


@pytest.fixture
def attached_probes():
    probes = []

    def attach(**kwargs):
        probe, llm, nvml = _attached_probe(**kwargs)
        probes.append(probe)
        return probe, llm, nvml

    yield attach
    for probe in reversed(probes):
        probe.close()


def _block_pool(llm):
    return llm.llm_engine.engine_core.engine_core.scheduler.kv_cache_manager.block_pool


def _throughput_records(caplog):
    return [
        record
        for record in caplog.records
        if record.name == _ENGINE_LOGGER.name
        and "Avg prompt throughput:" in record.getMessage()
    ]


def test_engine_logging_has_independent_eviction_windows(attached_probes, caplog):
    caplog.set_level(logging.INFO, logger=_ENGINE_LOGGER.name)
    probe, llm, _ = attached_probes()
    pool = _block_pool(llm)
    manager = llm.llm_engine.logger_manager

    pool.get_new_blocks(4)
    first_mark = probe.mark()
    assert first_mark["kv_cache_allocated_blocks"] == 4
    assert first_mark["kv_cache_evicted_blocks"] == 2
    assert manager.log() == "logged"
    first_record = _throughput_records(caplog)[-1]
    assert first_record.getMessage() == (
        _ENGINE_STATS % _ENGINE_STATS_ARGS
        + ", KV cache eviction rate: 50.000000% (2/4 blocks)"
    )
    assert first_record.pathname == __file__
    assert first_record.funcName == "log"
    assert all(
        "KV cache eviction rate:" not in record.getMessage()
        for record in caplog.records
        if "Unrelated engine diagnostic:" in record.getMessage()
    )

    pool.get_new_blocks(1)
    manager.log(emit_stats=False)
    with caplog.at_level(logging.WARNING, logger=_ENGINE_LOGGER.name):
        manager.log()
    # Calls without an emitted throughput record must not drain the log window.
    manager.log()
    assert _throughput_records(caplog)[-1].getMessage().endswith(
        "KV cache eviction rate: 100.000000% (1/1 blocks)"
    )
    second_mark = probe.mark()
    assert second_mark["kv_cache_allocated_blocks"] == 1
    assert second_mark["kv_cache_evicted_blocks"] == 1
    summary = probe.summary()
    assert summary["kv_cache_allocated_blocks"] == 5
    assert summary["kv_cache_evicted_blocks"] == 3

    manager.log()
    assert _throughput_records(caplog)[-1].getMessage().endswith(
        "KV cache eviction rate: n/a (0/0 blocks)"
    )


def test_engine_logging_distinguishes_zero_evictions_from_empty_window(
    attached_probes, caplog
):
    caplog.set_level(logging.INFO, logger=_ENGINE_LOGGER.name)
    _, llm, _ = attached_probes()
    pool = _block_pool(llm)
    pool.evictions_enabled = False
    pool.get_new_blocks(3)

    llm.llm_engine.logger_manager.log()
    assert _throughput_records(caplog)[-1].getMessage().endswith(
        "KV cache eviction rate: 0.000000% (0/3 blocks)"
    )


def test_engine_logging_repeated_records_share_one_window(attached_probes, caplog):
    caplog.set_level(logging.INFO, logger=_ENGINE_LOGGER.name)
    _, llm, _ = attached_probes()
    _block_pool(llm).get_new_blocks(4)

    llm.llm_engine.logger_manager.log(repeats=2)
    records = _throughput_records(caplog)
    assert len(records) == 2
    assert records[0].getMessage() == records[1].getMessage()
    assert records[0].getMessage().endswith(
        "KV cache eviction rate: 50.000000% (2/4 blocks)"
    )
    llm.llm_engine.logger_manager.log()
    assert _throughput_records(caplog)[-1].getMessage().endswith(
        "KV cache eviction rate: n/a (0/0 blocks)"
    )


@pytest.mark.parametrize(
    "probe_options", [{"with_scheduler": False}, {"engine_indexes": (0, 1)}]
)
def test_engine_logging_reports_unavailable_when_counts_cannot_be_attributed(
    attached_probes, caplog, probe_options
):
    caplog.set_level(logging.INFO, logger=_ENGINE_LOGGER.name)
    _, llm, _ = attached_probes(**probe_options)
    if probe_options.get("with_scheduler", True):
        _block_pool(llm).get_new_blocks(4)

    llm.llm_engine.logger_manager.log()
    assert _throughput_records(caplog)[-1].getMessage().endswith(
        "KV cache eviction rate: n/a"
    )


def test_engine_logging_scopes_probes_with_the_same_engine_index(
    attached_probes, caplog
):
    caplog.set_level(logging.INFO, logger=_ENGINE_LOGGER.name)
    first_probe, first_llm, _ = attached_probes()
    _, second_llm, _ = attached_probes()
    _block_pool(first_llm).get_new_blocks(4)
    _block_pool(second_llm).get_new_blocks(1)

    # The engine index alone cannot identify which live engine owns a record.
    _ENGINE_LOGGER.info(_ENGINE_STATS, *_ENGINE_STATS_ARGS)
    first_llm.llm_engine.logger_manager.log()
    second_llm.llm_engine.logger_manager.log()
    records = _throughput_records(caplog)
    assert records[0].getMessage() == _ENGINE_STATS % _ENGINE_STATS_ARGS
    assert records[1].getMessage().endswith(
        "KV cache eviction rate: 50.000000% (2/4 blocks)"
    )
    assert records[2].getMessage().endswith(
        "KV cache eviction rate: 100.000000% (1/1 blocks)"
    )
    assert all(
        record.getMessage().count("KV cache eviction rate:") == 1
        for record in records[1:]
    )

    first_probe.close()
    second_llm.llm_engine.logger_manager.log()
    assert _throughput_records(caplog)[-1].getMessage().endswith(
        "KV cache eviction rate: n/a (0/0 blocks)"
    )


def test_engine_logging_restores_method_filter_and_context_after_failure(caplog):
    caplog.set_level(logging.INFO, logger=_ENGINE_LOGGER.name)
    original_filters = list(_ENGINE_LOGGER.filters)
    nvml = _FakeNvmlSampler()
    llm = _FakeLlm(nvml)
    manager = llm.llm_engine.logger_manager
    original_log = manager.log
    probe = ResourceProbe()
    try:
        probe.attach(llm)
        _block_pool(llm).get_new_blocks(4)
        with pytest.raises(RuntimeError, match="logging failed"):
            manager.log(fail=True)
        _ENGINE_LOGGER.info(_ENGINE_STATS, *_ENGINE_STATS_ARGS)
        assert _throughput_records(caplog)[-1].getMessage() == (
            _ENGINE_STATS % _ENGINE_STATS_ARGS
        )
        manager.log()
        assert _throughput_records(caplog)[-1].getMessage().endswith(
            "KV cache eviction rate: 50.000000% (2/4 blocks)"
        )
    finally:
        probe.close()

    assert manager.log == original_log
    assert _ENGINE_LOGGER.filters == original_filters
    manager.log()
    assert _throughput_records(caplog)[-1].getMessage() == (
        _ENGINE_STATS % _ENGINE_STATS_ARGS
    )


def test_disabled_probe_leaves_engine_logging_untouched(caplog):
    caplog.set_level(logging.INFO, logger=_ENGINE_LOGGER.name)
    llm = _FakeLlm(_FakeNvmlSampler())
    manager = llm.llm_engine.logger_manager
    original_log = manager.log
    original_filters = list(_ENGINE_LOGGER.filters)
    probe = ResourceProbe(enabled=False)

    probe.attach(llm)
    manager.log()
    probe.close()

    assert manager.log == original_log
    assert _ENGINE_LOGGER.filters == original_filters
    assert _throughput_records(caplog)[-1].getMessage() == (
        _ENGINE_STATS % _ENGINE_STATS_ARGS
    )


@pytest.mark.parametrize(
    ("module_name", "function_name"),
    [
        ("run_power_mh", "_run_mcmc_benchmark"),
        ("run_subtree_prefetching_mh", "run_subtree_prefetching"),
    ],
)
def test_runner_closes_resource_probe_when_model_construction_fails(
    module_name, function_name
):
    # Execute the actual runner function without importing its GPU dependencies.
    source_path = (
        Path(__file__).resolve().parents[1]
        / "power_sharpening"
        / "runners"
        / "vllm"
        / f"{module_name}.py"
    )
    source = ast.parse(source_path.read_text(), filename=str(source_path))
    isolated = ast.Module(
        body=[
            node
            for node in source.body
            if (
                isinstance(node, ast.FunctionDef) and node.name == function_name
            )
            or (
                isinstance(node, ast.ImportFrom) and node.module == "contextlib"
            )
        ],
        type_ignores=[],
    )
    probe = SimpleNamespace(start=Mock(), close=Mock())
    namespace = {
        "ResourceProbe": Mock(return_value=probe),
        "subtree_prefetching_sampling": Mock(),
        "subtree_prefetching_sampling_with_entropy_cut": Mock(),
        "vLLM_Wrapper": Mock(side_effect=RuntimeError("model construction failed")),
        "logger": logging.getLogger(__name__),
        "time": SimpleNamespace(time=lambda: 0),
        "np": SimpleNamespace(random=SimpleNamespace(default_rng=lambda seed: None)),
    }
    exec(compile(isolated, str(source_path), "exec"), namespace)
    kwargs = {
        "model_str": "test-model",
        "benchmark": SimpleNamespace(name="test", size=1),
        "args": SimpleNamespace(
            seed=0,
            prefix_cache=True,
            verbose=False,
            cut_dist_type="uniform",
            max_samples=1,
            resource_probe=True,
        ),
        "max_model_len": None,
        "init_temperature": 0.5,
    }
    if module_name == "run_power_mh":
        kwargs.update(method="power_mcmc", cut_power=None, sample_batch=None)

    with pytest.raises(RuntimeError, match="model construction failed"):
        namespace[function_name](**kwargs)

    namespace["ResourceProbe"].assert_called_once_with(enabled=True)
    namespace["vLLM_Wrapper"].assert_called_once()
    probe.start.assert_called_once_with()
    probe.close.assert_called_once_with()


def test_generate_window_closes_on_exception_and_original_is_restored():
    probe, llm, nvml = _attached_probe(fail=True)
    original_generate = probe._original_generate
    block_pool = probe._block_pool
    original_allocate = probe._original_get_new_blocks
    original_evict = probe._original_maybe_evict_cached_block

    with pytest.raises(RuntimeError, match="generation failed"):
        llm.generate()

    assert not nvml.in_generation
    assert nvml.generation_calls == 1
    probe.close()
    assert llm.generate == original_generate
    assert block_pool.get_new_blocks == original_allocate
    assert block_pool._maybe_evict_cached_block == original_evict
    assert nvml.stopped


def test_eviction_is_unavailable_without_an_in_process_block_pool():
    nvml = _FakeNvmlSampler()
    probe = ResourceProbe()
    probe._nvml = nvml
    probe._nvml_ok = True
    llm = SimpleNamespace(generate=lambda: None, llm_engine=SimpleNamespace())

    probe.attach(llm)
    summary = probe.summary()

    assert summary["kv_cache_allocated_blocks"] is None
    assert summary["kv_cache_evicted_blocks"] is None
    assert summary["kv_cache_eviction_rate"] is None
    assert summary["eviction_source"] == "unavailable"
    probe.close()
