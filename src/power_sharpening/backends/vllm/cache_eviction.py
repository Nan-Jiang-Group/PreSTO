"""Evict completed subtree suffixes through the vLLM 0.27.1 EngineCore.

Run with the vLLM subtree runner's ``--evict_subtree_cache`` flag. Only
token-ID requests without LoRA, multimodal inputs, or cache salts are supported.
"""

from types import SimpleNamespace


def evict_cached_prefixes(pool, discarded_hashes, keep_hashes, group_ids):
    """Remove idle blocks exclusive to discarded prefixes and reuse them first.

    Args:
        pool: vLLM BlockPool, owned by the calling scheduler process.
        discarded_hashes: Chained prefix hashes for the old state and proposals.
        keep_hashes: Chained prefix hashes for the selected sequence.
        group_ids: Cache groups to inspect, including hybrid-model groups.

    Returns:
        Counts of evicted blocks and candidates skipped because they are in use.
    """
    from vllm.v1.core.kv_cache_utils import (
        get_block_hash,
        make_block_hash_with_group_id,
    )

    discarded_hashes = set(discarded_hashes) - set(keep_hashes)
    # Read the pinned implementation's hash buckets to include duplicate physical
    # blocks. get_one_block() alone can hide an idle copy behind an active copy.
    buckets = pool.cached_block_hash_to_block._cache
    candidates = {}
    for prefix_hash in discarded_hashes:
        for group_id in group_ids:
            bucket = buckets.get(make_block_hash_with_group_id(prefix_hash, group_id))
            if bucket is not None:
                blocks = bucket.values() if isinstance(bucket, dict) else (bucket,)
                candidates.update((block.block_id, block) for block in blocks)

    evictable = []
    active = 0
    for block in candidates.values():
        if block.is_null:
            continue
        if block.ref_cnt != 0:
            active += 1
            continue
        # Hybrid/partial cache entries can alias the same physical block. Keep
        # the entire block if ANY alias belongs to a prefix outside this cleanup.
        aliases = set(pool.cached_block_hashes_by_block.get(block.block_id, ()))
        if block.block_hash is not None:
            aliases.add(block.block_hash)
        if aliases and all(get_block_hash(key) in discarded_hashes for key in aliases):
            evictable.append(block)

    # free_blocks() would decrement references a second time. These blocks are
    # already free; only remove cache identities and change their queue order.
    previous_marker = getattr(pool, "subtree_eviction_in_progress", False)
    pool.subtree_eviction_in_progress = True
    try:
        pool.evict_blocks({block.block_id for block in evictable})
    finally:
        pool.subtree_eviction_in_progress = previous_marker
    for block in evictable:
        pool.free_block_queue.remove(block)
    pool.free_block_queue.prepend_n(evictable)
    return {"evicted_blocks": len(evictable), "skipped_active_blocks": active}


def _engine_evict_subtree_cache(self, sequences: list[list[int]], keep_token_ids: list[int]) -> dict:
    """Resolve token-prefix identities and prune them inside EngineCore.

    Args:
        self: EngineCore receiving the synchronous utility call.
        sequences: Old chain state and completed proposals, including the prompt.
        keep_token_ids: Selected chain state, including the prompt.

    Returns:
        Counts returned by ``evict_cached_prefixes``.
    """
    manager = self.scheduler.kv_cache_manager
    if not manager.enable_caching or self.request_block_hasher is None:
        raise ValueError("subtree cache eviction requires enable_prefix_caching=True")

    def hashes(token_ids):
        # Use the engine's own hasher, including its process-local initial hash.
        # This is a hash-only request view: it is never submitted for generation.
        request = SimpleNamespace(
            block_hashes=[], num_tokens=len(token_ids), all_token_ids=token_ids,
            mm_features=[], lora_request=None, cache_salt=None, prompt_embeds=None,
        )
        return self.request_block_hasher(request)

    keep_hashes = set(hashes(keep_token_ids))
    discarded_hashes = {value for tokens in sequences for value in hashes(tokens)}
    return evict_cached_prefixes(
        manager.block_pool, discarded_hashes, keep_hashes,
        range(manager.num_kv_cache_groups),
    )


def install_engine_core_eviction():
    """Install the utility in the process constructing the custom scheduler."""
    from vllm.v1.engine.core import EngineCore

    EngineCore.evict_subtree_cache = _engine_evict_subtree_cache


def call_cache_eviction(llm, sequences, keep_token_ids):
    """Send one acknowledged cleanup to an in-process or multiprocess engine.

    Args:
        llm: Custom vLLM LLM instance.
        sequences: Completed sequence token IDs, including the prompt.
        keep_token_ids: Selected sequence token IDs, including the prompt.

    Returns:
        Scheduler-reported eviction counts.
    """
    client = llm.llm_engine.engine_core
    core = getattr(client, "engine_core", None)
    if core is not None:
        return core.evict_subtree_cache(sequences, keep_token_ids)
    return client.call_utility("evict_subtree_cache", sequences, keep_token_ids)
