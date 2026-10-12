"""Record optional subtree cache cleanup without changing MH decisions.

Use through either backend's subtree runner with ``--evict_subtree_cache``.
"""

import logging
import time

logger = logging.getLogger(__name__)


def evict_subtree_cache(wrapper, stats, sequences, keep_token_ids):
    """Prune discarded prefixes and record cleanup cost separately from generation.

    Args:
        wrapper: Backend wrapper exposing ``evict_subtree_cache``.
        stats: SamplingStats receiving this cleanup's counters.
        sequences: Old state and proposals, each including prompt tokens.
        keep_token_ids: Selected sequence, including prompt tokens.

    Returns:
        None. Cache state and ``stats.cache_evictions`` are updated.
    """
    start = time.perf_counter()
    counts = wrapper.evict_subtree_cache(sequences, keep_token_ids)
    record = {**counts, "seconds": time.perf_counter() - start}
    stats.cache_evictions.append(record)
    logger.info("subtree cache eviction: %s", record)
