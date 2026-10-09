"""Hardware-resource measurement for vLLM runs.

Run the CPU checks with:
    python -m pytest /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/tests/test_resource_probe.py

Reports the resource metrics the paper needs:

  * ``peak_gpu_memory_gib`` -- peak total GPU memory held by this run's processes, and ``min_gpu_memory_gib``, the floor
    it drops back to once the engine is up. Both are reported for the process tree and, as ``device_*``, for the whole
    card. Preallocation keeps the two close: the gap between them, not the peak itself, is what generation actually
    moves.
  * ``peak_llm_generate_gpu_memory_gib`` -- peak process-tree GPU memory sampled strictly inside ``llm.generate`` calls.
    The companion ``peak_llm_generate_gpu_memory_delta_gib`` subtracts the post-load idle baseline, excluding the base
    model and preallocated KV pool.
  * ``peak_kv_cache_occupancy`` -- peak fraction of the KV-block pool that is occupied, i.e. ``max_t (1 - free_blocks_t
    / total_blocks)``.
  * ``kv_cache_eviction_rate`` -- cached physical blocks evicted divided by physical blocks allocated. This is exact
    when the scheduler is in-process; out-of-process vLLM exposes no unsampled eviction counter, so it is ``None``.
  * ``prefix_cache_hit_rate`` -- cumulative ``hits / queries`` in *tokens*, as counted by vLLM's prefix-cache
    statistics.

The engine's periodic throughput line includes the eviction rate and physical block counts since the previous logged
update. This interval is independent of ``mark()`` windows and the run summary. Missing exact counters, ambiguous
multi-engine attribution, and intervals without allocations are reported as ``n/a``.

Four backend facts shape the implementation:

  1. ``torch.cuda.max_memory_allocated()`` is useless here. vLLM v1 runs the EngineCore (and therefore the model) in a
     *child* process unless ``VLLM_ENABLE_V1_MULTIPROCESSING=0``, so the driver process's torch allocator sees ~nothing.
     We use NVML and attribute memory per PID over our own process tree, which is also immune to co-tenants on a shared
     node.
  2. vLLM preallocates the KV pool to ``gpu_memory_utilization`` x capacity, so the raw device peak is pinned by that
     flag and does *not* separate algorithms. ``summary()`` therefore also reports ``peak_demand_gib``: the raw peak
     with the idle part of the preallocated pool subtracted out (pool bytes replaced by ``peak_occupancy x pool
     bytes``). Report whichever the paper defines, but report the flag alongside it.
  3. Occupancy peaks *inside* an engine step -- after the scheduler allocates blocks and before finished requests free
     them -- so an after-the-fact read of the gauge always shows a drained pool. When the engine is in-process we wrap
     ``Scheduler.schedule`` / ``update_from_output`` and get the exact per-step maximum; otherwise we fall back to
     polling ``LLM.get_metrics()``, which samples the gauge and can miss short spikes. ``kv_source`` in the summary says
     which one produced the number, so an approximate figure never silently ends up in a table.
  4. A cached-block eviction is not an occupancy decrease: completed requests free blocks without evicting reusable
     prefixes. For an in-process engine we therefore wrap the block pool's allocation and eviction methods and count
     only evictions that vLLM itself reports as successful.

Everything degrades to ``None`` rather than raising: a missing pynvml or a renamed vLLM internal costs a metric, not the
run.
"""

import logging
import os
import threading
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps

logger = logging.getLogger("[resource_probe]")

_GIB = float(1 << 30)
_VLLM_LOG_CONTEXT = ContextVar("resource_probe_vllm_log", default=None)


def _nvml_handles():
    """NVML handles for the GPUs this process may use, honouring CUDA_VISIBLE_DEVICES."""
    import pynvml

    pynvml.nvmlInit()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or visible.strip() == "":
        count = pynvml.nvmlDeviceGetCount()
        return [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(count)]

    handles = []
    for entry in visible.split(","):
        entry = entry.strip()
        if not entry:
            continue
        # Slurm hands out plain indices; some launchers hand out GPU-<uuid>.
        if entry.startswith("GPU-") or entry.startswith("MIG-"):
            handles.append(pynvml.nvmlDeviceGetHandleByUUID(entry.encode()))
        else:
            handles.append(pynvml.nvmlDeviceGetHandleByIndex(int(entry)))
    return handles


def _process_tree_pids(root_pid: int) -> set[int]:
    """``root_pid`` plus its descendants -- the EngineCore lives in a child."""
    pids = {root_pid}
    try:
        import psutil

        proc = psutil.Process(root_pid)
        pids.update(child.pid for child in proc.children(recursive=True))
    except Exception:
        pass
    return pids


class _NvmlSampler:
    """Background thread tracking peak GPU bytes used by our process tree."""

    def __init__(self, poll_interval: float):
        self.poll_interval = poll_interval
        self.peak_bytes = 0
        self.window_peak_bytes = 0
        self.device_peak_bytes = 0  # whole-device, i.e. including co-tenants
        self.current_bytes = 0
        self.device_current_bytes = 0
        self.loaded_bytes = None
        self.device_loaded_bytes = None
        self.generation_peak_bytes = 0
        self.window_generation_peak_bytes = 0
        self.generation_calls = 0
        self.window_generation_calls = 0
        self._generation_depth = 0
        # Floors start at None, not 0: "no sample yet" and "a real zero-byte reading" have to stay distinguishable, or
        # an unmeasurable run would report a 0 GiB floor as if it were measured.
        self.min_bytes = None
        self.device_min_bytes = None
        self._handles = None
        self._thread = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.error = None

    def start(self) -> bool:
        try:
            self._handles = _nvml_handles()
        except Exception as exc:  # no pynvml, no driver, CPU-only box
            self.error = f"{type(exc).__name__}: {exc}"
            logger.warning("GPU memory probe disabled (%s)", self.error)
            return False
        self._thread = threading.Thread(
            target=self._run, name="resource-probe-nvml", daemon=True
        )
        self._thread.start()
        return True

    def _sample_once(self):
        import pynvml

        pids = _process_tree_pids(os.getpid())
        ours = 0
        device_total = 0
        for handle in self._handles:
            try:
                device_total += pynvml.nvmlDeviceGetMemoryInfo(handle).used
            except Exception:
                pass
            try:
                procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            except Exception:
                continue
            for proc in procs:
                used = getattr(proc, "usedGpuMemory", None)
                # None under MIG/vGPU, where per-process accounting is off.
                if used and proc.pid in pids:
                    ours += used
        with self._lock:
            self.current_bytes = ours
            self.device_current_bytes = device_total
            self.peak_bytes = max(self.peak_bytes, ours)
            self.window_peak_bytes = max(self.window_peak_bytes, ours)
            self.device_peak_bytes = max(self.device_peak_bytes, device_total)
            if self._generation_depth:
                self.generation_peak_bytes = max(
                    self.generation_peak_bytes, ours
                )
                self.window_generation_peak_bytes = max(
                    self.window_generation_peak_bytes, ours
                )
            # Zero is skipped rather than minimised over. Our own total reads 0 before the engine has allocated and on
            # nodes where per-process accounting is off (see usedGpuMemory above), and either would otherwise pin the
            # floor at 0 for the whole run.
            if ours:
                self.min_bytes = min(
                    ours, self.min_bytes if self.min_bytes is not None else ours
                )
            if device_total:
                self.device_min_bytes = min(
                    device_total,
                    (
                        self.device_min_bytes
                        if self.device_min_bytes is not None
                        else device_total
                    ),
                )

    def _run(self):
        while not self._stop.is_set():
            try:
                self._sample_once()
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc}"
                return
            self._stop.wait(self.poll_interval)

    def take_window(self) -> int:
        """Peak since the previous call, then reset the window."""
        with self._lock:
            peak = self.window_peak_bytes
            self.window_peak_bytes = 0
        return peak

    def take_generation_window(self) -> tuple[int, int]:
        """Generation-only peak and call count since the previous mark."""
        with self._lock:
            peak = self.window_generation_peak_bytes
            calls = self.window_generation_calls
            self.window_generation_peak_bytes = 0
            self.window_generation_calls = 0
        return peak, calls

    def reset_min(self):
        """Drop the floors recorded so far.

        Sampling starts before the engine exists, so the samples up to that point walk up through weight loading and
        KV-pool allocation. Called once the engine is up, this keeps the floor a statement about the generation phase
        instead of the smallest reading during startup.
        """
        with self._lock:
            self.min_bytes = None
            self.device_min_bytes = None

    def reset_after_load(self):
        """Capture the loaded-engine baseline and discard startup windows."""
        try:
            self._sample_once()
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
        with self._lock:
            self.loaded_bytes = self.current_bytes or None
            self.device_loaded_bytes = self.device_current_bytes or None
            self.window_peak_bytes = 0
            self.window_generation_peak_bytes = 0
            self.window_generation_calls = 0
            self.min_bytes = None
            self.device_min_bytes = None

    def begin_generation(self):
        """Open an NVML window immediately before one ``llm.generate`` call."""
        with self._lock:
            self._generation_depth += 1
            self.generation_calls += 1
            self.window_generation_calls += 1
        try:
            self._sample_once()
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"

    def end_generation(self):
        """Take a final sample and close one ``llm.generate`` window."""
        try:
            self._sample_once()
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            with self._lock:
                self._generation_depth = max(0, self._generation_depth - 1)

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


def _kv_usage(kv_cache_manager) -> float | None:
    """Occupied fraction of the block pool, across vLLM naming variants."""
    usage = getattr(kv_cache_manager, "usage", None)
    if isinstance(usage, (int, float)):
        return float(usage)
    pool = getattr(kv_cache_manager, "block_pool", None)
    if pool is None:
        return None
    try:
        free = pool.get_num_free_blocks()
        total = getattr(pool, "num_gpu_blocks", None)
        if not total:
            return None
        return 1.0 - free / total
    except Exception:
        return None


@dataclass
class _KvCounters:
    """Occupancy, prefix-cache, allocation, and eviction counters."""

    peak_usage: float = 0.0
    window_peak_usage: float = 0.0
    queries: int = 0
    hits: int = 0
    window_queries: int = 0
    window_hits: int = 0
    allocated_blocks: int = 0
    evicted_blocks: int = 0
    window_allocated_blocks: int = 0
    window_evicted_blocks: int = 0
    steps: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def record_usage(self, usage: float | None):
        if usage is None:
            return
        with self.lock:
            self.peak_usage = max(self.peak_usage, usage)
            self.window_peak_usage = max(self.window_peak_usage, usage)
            self.steps += 1

    def record_prefix(self, queries: int, hits: int):
        with self.lock:
            self.queries += queries
            self.hits += hits
            self.window_queries += queries
            self.window_hits += hits

    def record_allocation(self, blocks: int):
        if blocks <= 0:
            return
        with self.lock:
            self.allocated_blocks += blocks
            self.window_allocated_blocks += blocks

    def record_eviction(self):
        with self.lock:
            self.evicted_blocks += 1
            self.window_evicted_blocks += 1


class ResourceProbe:
    """Measure generation GPU memory, KV behavior, and prefix-cache hits.

    Usage mirrors the lifetime of the engine::

        probe = ResourceProbe()
        probe.start()                # before the engine allocates anything
        with vLLM_Wrapper(...) as llm:
            probe.attach(llm.llm)    # after the engine exists
            for sample in ...:
                ...
                row = probe.mark()   # per-sample window
        summary = probe.summary()
        probe.close()
    """

    def __init__(self, poll_interval: float = 0.05, enabled: bool = True):
        self.enabled = enabled
        self._nvml = _NvmlSampler(poll_interval) if enabled else None
        self._nvml_ok = False
        self._kv = _KvCounters()
        self._llm = None
        self._scheduler = None
        self._metrics_poller = None
        self._stop_metrics = threading.Event()
        self._poll_interval = poll_interval
        self.kv_source = "unavailable"
        self.eviction_source = "unavailable"
        self.prefix_source = "unavailable"
        self.generation_gpu_source = "unavailable"
        self.baseline_bytes = 0
        self.kv_pool_bytes = None
        self.gpu_memory_utilization = None
        self.num_gpu_blocks = None
        self._original_generate = None
        self._block_pool = None
        self._original_get_new_blocks = None
        self._original_maybe_evict_cached_block = None
        self._logger_manager = None
        self._original_stats_log = None
        self._stats_log_filter = None
        self._single_log_engine = False
        self._last_logged_allocated_blocks = 0
        self._last_logged_evicted_blocks = 0

    # -- lifecycle ---------------------------------------------------------

    def start(self):
        """Begin sampling. Call *before* the engine is constructed."""
        if not self.enabled:
            return
        self.baseline_bytes = self._device_used_bytes()
        self._nvml_ok = self._nvml.start()

    def attach(self, llm):
        """Hook the live engine. Call once the ``LLM`` object exists."""
        if not self.enabled:
            return
        self._llm = llm
        # The engine has allocated by now, so anything sampled before this is startup ramp rather than the level the run
        # holds while generating.
        if self._nvml_ok:
            self._nvml.reset_after_load()
            self._wrap_generate(llm)
        self._read_static_config(llm)
        self._scheduler = self._find_scheduler(llm)
        if self._scheduler is not None:
            self._wrap_scheduler(self._scheduler)
            self.kv_source = "scheduler-hook"
        elif hasattr(llm, "get_metrics"):
            # Engine is in a child process: the gauge is all we can reach, and only at the rate we poll it.
            self._start_metrics_poller()
            self.kv_source = "metrics-poll"
        else:
            logger.warning(
                "no KV-cache probe available: engine is out-of-process and "
                "LLM.get_metrics() is missing"
            )
        self._wrap_engine_logging(llm)

    def close(self):
        if not self.enabled:
            return
        if self._stats_log_filter is not None:
            logging.getLogger("vllm.v1.metrics.loggers").removeFilter(
                self._stats_log_filter
            )
            self._stats_log_filter = None
        if self._original_stats_log is not None:
            self._logger_manager.log = self._original_stats_log
            self._original_stats_log = None
        self._stop_metrics.set()
        if self._metrics_poller is not None:
            self._metrics_poller.join(timeout=2.0)
        if self._original_generate is not None and self._llm is not None:
            try:
                self._llm.generate = self._original_generate
            except Exception:
                pass
        if self._block_pool is not None:
            try:
                self._block_pool.get_new_blocks = self._original_get_new_blocks
                self._block_pool._maybe_evict_cached_block = (
                    self._original_maybe_evict_cached_block
                )
            except Exception:
                pass
        if self._nvml is not None:
            self._nvml.stop()

    # -- engine introspection ---------------------------------------------

    def _device_used_bytes(self) -> int:
        try:
            import pynvml

            total = 0
            for handle in _nvml_handles():
                total += pynvml.nvmlDeviceGetMemoryInfo(handle).used
            return total
        except Exception:
            return 0

    def _read_static_config(self, llm):
        """Pull KV-pool size and the preallocation flag off the engine config."""
        config = getattr(getattr(llm, "llm_engine", None), "vllm_config", None)
        cache_config = getattr(config, "cache_config", None)
        if cache_config is None:
            return
        self.gpu_memory_utilization = getattr(
            cache_config, "gpu_memory_utilization", None
        )
        self.num_gpu_blocks = getattr(cache_config, "num_gpu_blocks", None)
        # vLLM records the profiled pool size here once the engine is up; on versions that leave it unset the pool bytes
        # stay None and peak_demand_gib is simply not reported.
        pool_bytes = getattr(cache_config, "kv_cache_memory_bytes", None)
        if isinstance(pool_bytes, (int, float)) and pool_bytes > 0:
            self.kv_pool_bytes = int(pool_bytes)

    def _find_scheduler(self, llm):
        """The in-process scheduler, or None when EngineCore is a subprocess."""
        client = getattr(getattr(llm, "llm_engine", None), "engine_core", None)
        # InprocClient holds the EngineCore directly; the MP clients hold a socket.
        core = getattr(client, "engine_core", None)
        return getattr(core, "scheduler", None)

    def _wrap_generate(self, llm):
        """Measure only time spent inside the public ``llm.generate`` call."""
        original_generate = getattr(llm, "generate", None)
        if original_generate is None:
            logger.warning("LLM has no generate method; generation GPU probe disabled")
            return
        nvml = self._nvml

        @wraps(original_generate)
        def generate(*args, **kwargs):
            nvml.begin_generation()
            try:
                return original_generate(*args, **kwargs)
            finally:
                nvml.end_generation()

        try:
            llm.generate = generate
        except Exception as exc:
            logger.warning(
                "could not wrap LLM.generate; generation GPU probe disabled (%s)",
                exc,
            )
            return
        self._original_generate = original_generate
        self.generation_gpu_source = "llm.generate-nvml-sampled"

    def _wrap_scheduler(self, scheduler):
        """Sample occupancy at both extremes of an engine step.

        ``schedule()`` returns just after blocks are allocated (the high-water mark) and ``update_from_output()`` just
        after finished requests free theirs, so taking the max over both brackets the step exactly.
        """
        kv_manager = getattr(scheduler, "kv_cache_manager", None)
        counters = self._kv
        original_schedule = scheduler.schedule
        original_update = scheduler.update_from_output
        original_make_stats = getattr(scheduler, "make_stats", None)

        def schedule(*args, **kwargs):
            out = original_schedule(*args, **kwargs)
            counters.record_usage(_kv_usage(kv_manager))
            return out

        def update_from_output(*args, **kwargs):
            out = original_update(*args, **kwargs)
            counters.record_usage(_kv_usage(kv_manager))
            return out

        scheduler.schedule = schedule
        scheduler.update_from_output = update_from_output

        self._wrap_block_pool(kv_manager)

        # Prefix-cache counters are drained every time make_stats() collects them, so read them from the stats object
        # rather than from the manager.
        if original_make_stats is not None:

            def make_stats(*args, **kwargs):
                stats = original_make_stats(*args, **kwargs)
                prefix = getattr(stats, "prefix_cache_stats", None)
                if prefix is not None:
                    counters.record_prefix(
                        int(getattr(prefix, "queries", 0)),
                        int(getattr(prefix, "hits", 0)),
                    )
                return stats

            scheduler.make_stats = make_stats
            self.prefix_source = "scheduler-hook"

    def _wrap_block_pool(self, kv_manager):
        """Count exact physical block allocations and cached-block evictions."""
        pool = getattr(kv_manager, "block_pool", None)
        original_allocate = getattr(pool, "get_new_blocks", None)
        original_evict = getattr(pool, "_maybe_evict_cached_block", None)
        if original_allocate is None or original_evict is None:
            logger.warning(
                "no exact KV-cache eviction probe available: block-pool APIs missing"
            )
            return
        counters = self._kv

        @wraps(original_evict)
        def maybe_evict(*args, **kwargs):
            evicted = original_evict(*args, **kwargs)
            if evicted and not getattr(pool, "subtree_eviction_in_progress", False):
                counters.record_eviction()
            return evicted

        @wraps(original_allocate)
        def get_new_blocks(*args, **kwargs):
            blocks = original_allocate(*args, **kwargs)
            counters.record_allocation(len(blocks))
            return blocks

        try:
            pool._maybe_evict_cached_block = maybe_evict
            pool.get_new_blocks = get_new_blocks
        except Exception as exc:
            logger.warning(
                "could not wrap KV block pool; eviction probe disabled (%s)",
                exc,
            )
            return
        self._block_pool = pool
        self._original_get_new_blocks = original_allocate
        self._original_maybe_evict_cached_block = original_evict
        self.eviction_source = "block-pool-hook"

    # -- periodic engine logging ------------------------------------------

    def _wrap_engine_logging(self, llm):
        """Extend this engine's existing stats line without replacing vLLM's logger."""
        manager = getattr(getattr(llm, "llm_engine", None), "logger_manager", None)
        original_log = getattr(manager, "log", None)
        if original_log is None:
            return
        self._single_log_engine = len(getattr(manager, "engine_indexes", ())) == 1

        @wraps(original_log)
        def log(*args, **kwargs):
            # All throughput records in one manager call share one snapshot. Delay taking it until a record is emitted
            # (INFO may be disabled).
            rate = None

            def eviction_rate():
                nonlocal rate
                if rate is None:
                    rate = self._eviction_rate_for_log()
                return rate

            token = _VLLM_LOG_CONTEXT.set((self, eviction_rate))
            try:
                return original_log(*args, **kwargs)
            finally:
                _VLLM_LOG_CONTEXT.reset(token)

        try:
            manager.log = log
        except Exception as exc:
            logger.warning("could not extend vLLM eviction logging (%s)", exc)
            return
        self._logger_manager = manager
        self._original_stats_log = original_log
        self._stats_log_filter = self._extend_vllm_log
        logging.getLogger("vllm.v1.metrics.loggers").addFilter(self._stats_log_filter)

    def _extend_vllm_log(self, record):
        """Append counters only while the attached engine is logging its stats."""
        context = _VLLM_LOG_CONTEXT.get()
        if context is None or context[0] is not self:
            return True
        message = record.getMessage()
        if "Avg prompt throughput:" not in message or \
                "Avg generation throughput:" not in message:
            return True
        record.msg = f"{message}, KV cache eviction rate: {context[1]()}"
        record.args = ()
        return True

    def _eviction_rate_for_log(self) -> str:
        """Read interval deltas without draining per-sample or run counters."""
        if self.eviction_source == "unavailable" or not self._single_log_engine:
            return "n/a"
        with self._kv.lock:
            allocated = self._kv.allocated_blocks - self._last_logged_allocated_blocks
            evicted = self._kv.evicted_blocks - self._last_logged_evicted_blocks
            self._last_logged_allocated_blocks = self._kv.allocated_blocks
            self._last_logged_evicted_blocks = self._kv.evicted_blocks
        rate = format(evicted / allocated, ".6%") if allocated else "n/a"
        return f"{rate} ({evicted}/{allocated} blocks)"

    # -- metrics fallback (out-of-process engine) --------------------------

    @staticmethod
    def _metric_value(metrics, predicate):
        for metric in metrics:
            name = getattr(metric, "name", "")
            if predicate(name):
                value = getattr(metric, "value", None)
                if value is not None:
                    return float(value)
                values = getattr(metric, "values", None)
                if values:
                    return float(sum(values))
        return None

    def _read_metrics(self):
        """(usage_gauge, queries, hits) from the Prometheus snapshot."""
        try:
            metrics = self._llm.get_metrics()
        except Exception:
            return None, None, None
        usage = self._metric_value(metrics, lambda n: n.endswith("cache_usage_perc"))
        queries = self._metric_value(
            metrics,
            lambda n: "prefix_cache" in n and "queries" in n,
        )
        hits = self._metric_value(
            metrics,
            lambda n: "prefix_cache" in n and "hits" in n,
        )
        return usage, queries, hits

    def _start_metrics_poller(self):
        seen = {"queries": 0.0, "hits": 0.0}

        def run():
            while not self._stop_metrics.is_set():
                usage, queries, hits = self._read_metrics()
                self._kv.record_usage(usage)
                if queries is not None and hits is not None:
                    # Prometheus counters are cumulative; record the delta.
                    self._kv.record_prefix(
                        int(queries - seen["queries"]),
                        int(hits - seen["hits"]),
                    )
                    seen["queries"], seen["hits"] = queries, hits
                    self.prefix_source = "metrics-counter"
                self._stop_metrics.wait(self._poll_interval)

        self._metrics_poller = threading.Thread(
            target=run, name="resource-probe-metrics", daemon=True
        )
        self._metrics_poller.start()

    # -- reporting ---------------------------------------------------------

    def mark(self) -> dict:
        """Close the current window and return its per-sample metrics."""
        if not self.enabled:
            return {}
        with self._kv.lock:
            peak_usage = self._kv.window_peak_usage
            queries = self._kv.window_queries
            hits = self._kv.window_hits
            allocated_blocks = self._kv.window_allocated_blocks
            evicted_blocks = self._kv.window_evicted_blocks
            self._kv.window_peak_usage = 0.0
            self._kv.window_queries = 0
            self._kv.window_hits = 0
            self._kv.window_allocated_blocks = 0
            self._kv.window_evicted_blocks = 0
        peak_bytes = self._nvml.take_window() if self._nvml_ok else 0
        generation_peak_bytes, generation_calls = (
            self._nvml.take_generation_window() if self._nvml_ok else (0, 0)
        )
        loaded_bytes = self._nvml.loaded_bytes if self._nvml_ok else None
        generation_delta_bytes = (
            max(0, generation_peak_bytes - loaded_bytes)
            if generation_peak_bytes and loaded_bytes is not None
            else None
        )
        eviction_measured = self.eviction_source != "unavailable"
        return {
            "peak_gpu_memory_gib": (peak_bytes / _GIB) if peak_bytes else None,
            "loaded_gpu_memory_gib": (
                loaded_bytes / _GIB if loaded_bytes is not None else None
            ),
            "peak_llm_generate_gpu_memory_gib": (
                generation_peak_bytes / _GIB if generation_peak_bytes else None
            ),
            "peak_llm_generate_gpu_memory_delta_gib": (
                generation_delta_bytes / _GIB
                if generation_delta_bytes is not None
                else None
            ),
            "llm_generate_calls": (
                generation_calls
                if self.generation_gpu_source != "unavailable"
                else None
            ),
            "peak_kv_cache_occupancy": peak_usage or None,
            "kv_cache_allocated_blocks": (
                allocated_blocks if eviction_measured else None
            ),
            "kv_cache_evicted_blocks": (
                evicted_blocks if eviction_measured else None
            ),
            "kv_cache_eviction_rate": (
                evicted_blocks / allocated_blocks
                if eviction_measured and allocated_blocks
                else None
            ),
            "prefix_cache_hit_rate": (hits / queries) if queries else None,
            "prefix_cache_queried_tokens": queries or None,
        }

    def summary_line(self) -> str:
        """One-line run summary at a precision that survives small values.

        vLLM's own stats line rounds the prefix-cache rate to 0.1%, which cannot tell a cold cache from a genuinely
        unused one. The raw hit/query token counts are printed alongside the rate so the ratio is auditable rather than
        merely displayed.
        """
        stats = self.summary()
        if not stats.get("enabled"):
            return "resource probe disabled"

        def number(value, spec: str) -> str:
            return "n/a" if value is None else format(value, spec)

        return (
            f"peak_gpu={number(stats['peak_gpu_memory_gib'], '.4f')} GiB, "
            f"min_gpu={number(stats['min_gpu_memory_gib'], '.4f')} GiB, "
            f"llm_generate_peak="
            f"{number(stats['peak_llm_generate_gpu_memory_gib'], '.4f')} GiB, "
            f"llm_generate_delta="
            f"{number(stats['peak_llm_generate_gpu_memory_delta_gib'], '.4f')} "
            f"GiB over loaded="
            f"{number(stats['loaded_gpu_memory_gib'], '.4f')} GiB, "
            f"peak_demand={number(stats['peak_demand_gib'], '.4f')} GiB, "
            f"peak_kv_occupancy={number(stats['peak_kv_cache_occupancy'], '.6%')} "
            f"({number(stats['peak_kv_cache_gib'], '.4f')} GiB of "
            f"{number(stats['kv_pool_gib'], '.4f')} GiB pool), "
            f"kv_eviction_rate="
            f"{number(stats['kv_cache_eviction_rate'], '.6%')} "
            f"({stats['kv_cache_evicted_blocks']}/"
            f"{stats['kv_cache_allocated_blocks']} blocks), "
            f"prefix_hit_rate={number(stats['prefix_cache_hit_rate'], '.6%')} "
            f"({stats['prefix_cache_hit_tokens']}/"
            f"{stats['prefix_cache_queried_tokens']} tokens), "
            f"steps={stats['engine_steps_sampled']}, "
            f"kv_source={stats['kv_source']}, "
            f"eviction_source={stats['eviction_source']}, "
            f"generation_gpu_source={stats['generation_gpu_source']}, "
            f"prefix_source={stats['prefix_source']}"
        )

    def summary(self) -> dict:
        """Run-level metrics, with the provenance of each number."""
        if not self.enabled:
            return {"enabled": False}
        peak_bytes = self._nvml.peak_bytes if self._nvml_ok else 0
        peak_gib = (peak_bytes / _GIB) if peak_bytes else None
        loaded_bytes = self._nvml.loaded_bytes if self._nvml_ok else None
        generation_peak_bytes = (
            self._nvml.generation_peak_bytes if self._nvml_ok else 0
        )
        generation_delta_bytes = (
            max(0, generation_peak_bytes - loaded_bytes)
            if generation_peak_bytes and loaded_bytes is not None
            else None
        )
        pool_gib = (self.kv_pool_bytes / _GIB) if self.kv_pool_bytes else None
        with self._kv.lock:
            peak_usage = self._kv.peak_usage or None
            queries = self._kv.queries
            hits = self._kv.hits
            allocated_blocks = self._kv.allocated_blocks
            evicted_blocks = self._kv.evicted_blocks
            steps = self._kv.steps
        eviction_measured = self.eviction_source != "unavailable"

        # Preallocation makes the raw peak a property of the flag, not the algorithm; charge only the blocks that were
        # actually occupied.
        demand_gib = None
        if peak_gib is not None and pool_gib is not None and peak_usage is not None:
            demand_gib = peak_gib - pool_gib * (1.0 - peak_usage)

        def gib(value) -> float | None:
            return None if not self._nvml_ok or value is None else value / _GIB

        return {
            "enabled": True,
            "peak_gpu_memory_gib": peak_gib,
            "min_gpu_memory_gib": gib(self._nvml.min_bytes),
            "loaded_gpu_memory_gib": gib(loaded_bytes),
            "peak_llm_generate_gpu_memory_gib": gib(generation_peak_bytes or None),
            "peak_llm_generate_gpu_memory_delta_gib": gib(
                generation_delta_bytes
            ),
            "llm_generate_calls": (
                self._nvml.generation_calls
                if self._nvml_ok
                and self.generation_gpu_source != "unavailable"
                else None
            ),
            "peak_demand_gib": demand_gib,
            "device_peak_gpu_memory_gib": gib(self._nvml.device_peak_bytes),
            "device_min_gpu_memory_gib": gib(self._nvml.device_min_bytes),
            "baseline_gpu_memory_gib": self.baseline_bytes / _GIB,
            "kv_pool_gib": pool_gib,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "num_gpu_blocks": self.num_gpu_blocks,
            "peak_kv_cache_occupancy": peak_usage,
            "peak_kv_cache_gib": (
                pool_gib * peak_usage if pool_gib and peak_usage else None
            ),
            "kv_cache_allocated_blocks": (
                allocated_blocks if eviction_measured else None
            ),
            "kv_cache_evicted_blocks": (
                evicted_blocks if eviction_measured else None
            ),
            "kv_cache_eviction_rate": (
                evicted_blocks / allocated_blocks
                if eviction_measured and allocated_blocks
                else None
            ),
            "prefix_cache_hit_rate": hits / queries if queries else None,
            "prefix_cache_queried_tokens": queries,
            "prefix_cache_hit_tokens": hits,
            "engine_steps_sampled": steps,
            # "scheduler-hook" is exact; "metrics-poll" samples a gauge and can miss spikes shorter than the poll
            # interval.
            "kv_source": self.kv_source,
            # Exact only for an in-process scheduler. Occupancy changes and sampled residency histograms are
            # deliberately not treated as physical cached-block eviction counts.
            "eviction_source": self.eviction_source,
            "generation_gpu_source": self.generation_gpu_source,
            "prefix_source": self.prefix_source,
            "nvml_error": None if self._nvml is None else self._nvml.error,
        }
