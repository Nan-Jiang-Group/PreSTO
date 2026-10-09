"""Selective radix-cache eviction for SGLang 0.5.2 subtree sampling.

Use the SGLang subtree runner's ``--evict_subtree_cache`` flag. The feature
supports the ordinary RadixCache on one node with data/pipeline parallelism 1.
Tensor parallelism uses SGLang's existing collective scheduler RPC transport.
"""

from contextlib import contextmanager
import json


def evict_radix_paths(cache, sequences, keep_token_ids):
    """Free unlocked leaves belonging only to discarded sequence prefixes.

    Args:
        cache: Ordinary SGLang RadixCache, owned by the scheduler.
        sequences: Old state and completed proposals, including prompt tokens.
        keep_token_ids: Selected sequence, including prompt tokens.

    Returns:
        Counts of freed tokens and radix nodes.
    """
    if cache.disable:
        raise ValueError("subtree cache eviction requires the radix cache")
    # Do all splitting before collecting node references. Otherwise a later
    # match could split a candidate node and invalidate the retained ancestor set.
    keys = [list(tokens[:len(tokens) // cache.page_size * cache.page_size]) for tokens in sequences]
    keep = list(keep_token_ids[:len(keep_token_ids) // cache.page_size * cache.page_size])
    for tokens in keys + [keep]:
        cache.match_prefix(tokens)

    def ancestors(tokens):
        node = cache.match_prefix(tokens).last_device_node
        path = []
        while node is not cache.root_node:
            path.append(node)
            node = node.parent
        return path

    retained = set(ancestors(keep))
    candidates = {node for tokens in keys for node in ancestors(tokens)} - retained

    def depth(node):
        result = 0
        while node is not cache.root_node:
            result += 1
            node = node.parent
        return result

    evicted_tokens = 0
    evicted_nodes = 0
    for node in sorted(candidates, key=depth, reverse=True):
        # A child outside the candidate set protects its ancestors too. Never
        # free shared storage while any running request still holds its lock.
        if node.children or node.lock_ref != 0:
            continue
        cache.token_to_kv_pool_allocator.free(node.value)
        evicted_tokens += len(node.value)
        evicted_nodes += 1
        cache._delete_leaf(node)
        cache._record_remove_event(node)
    return {"evicted_tokens": evicted_tokens, "evicted_nodes": evicted_nodes}


def _handle_cache_rpc(self, request):
    """Handle cache cleanup without logging full proposal token arrays."""
    if request.method != "evict_subtree_cache":
        return self._subtree_original_handle_rpc(request)
    from sglang.srt.managers.scheduler import RpcReqOutput, barrier
    from sglang.srt.mem_cache.radix_cache import RadixCache

    try:
        if type(self.tree_cache) is not RadixCache:
            raise ValueError("subtree cache eviction supports ordinary RadixCache only")
        counts = evict_radix_paths(self.tree_cache, **request.parameters)
        response = RpcReqOutput(True, json.dumps(counts))
    except Exception as exc:
        response = RpcReqOutput(False, str(exc))
    barrier()
    return response


def run_scheduler_with_cache_eviction(*args, **kwargs):
    """Install the handler in each spawned scheduler, then run its normal loop."""
    from sglang.srt.managers import scheduler

    scheduler.Scheduler._subtree_original_handle_rpc = scheduler.Scheduler.handle_rpc_request
    scheduler.Scheduler.handle_rpc_request = _handle_cache_rpc
    scheduler.run_scheduler_process(*args, **kwargs)


@contextmanager
def cache_eviction_launch():
    """Launch scheduler processes with the picklable cache-eviction entry point."""
    from sglang.srt.entrypoints import engine

    original = engine.run_scheduler_process
    engine.run_scheduler_process = run_scheduler_with_cache_eviction
    try:
        yield
    finally:
        engine.run_scheduler_process = original


def call_cache_eviction(engine, sequences, keep_token_ids):
    """Perform an acknowledged collective cleanup and return scheduler counts.

    Args:
        engine: SGLang Engine launched with ``cache_eviction_launch``.
        sequences: Old state and proposals, each including the prompt.
        keep_token_ids: Selected sequence, including the prompt.

    Returns:
        Evicted token/node counts for the reporting tensor-parallel rank.
    """
    from sglang.srt.managers.io_struct import RpcReqInput, RpcReqOutput

    engine.send_to_rpc.send_pyobj(RpcReqInput(
        method="evict_subtree_cache",
        parameters={"sequences": sequences, "keep_token_ids": keep_token_ids},
    ))
    response = engine.send_to_rpc.recv_pyobj()
    if not isinstance(response, RpcReqOutput) or not response.success:
        raise RuntimeError(f"SGLang subtree cache eviction failed: {response}")
    return json.loads(response.message)
