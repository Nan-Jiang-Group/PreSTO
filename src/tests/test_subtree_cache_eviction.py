"""CPU cache-ownership tests.

Run: pytest /Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/tests/test_subtree_cache_eviction.py
"""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from power_sharpening.backends.vllm.cache_eviction import (
    _engine_evict_subtree_cache, call_cache_eviction, evict_cached_prefixes,
    install_engine_core_eviction,
)
from power_sharpening.backends.sglang.cache_eviction import evict_radix_paths


class Block:
    def __init__(self, block_id, prefix_hash, group=0, ref_cnt=0):
        self.block_id = block_id
        self.block_hash = (prefix_hash, group)
        self.ref_cnt = ref_cnt
        self.is_null = False


class Queue:
    def __init__(self, blocks):
        self.blocks = [block for block in blocks if block.ref_cnt == 0]

    def remove(self, block):
        self.blocks.remove(block)

    def prepend_n(self, blocks):
        assert not set(blocks) & set(self.blocks), "a free block was inserted twice"
        self.blocks[:0] = blocks


class Pool:
    def __init__(self, blocks, aliases=None):
        self.blocks = blocks
        self.cached_block_hashes_by_block = aliases or {}
        buckets = {}
        for block in blocks:
            for key in [block.block_hash, *self.cached_block_hashes_by_block.get(block.block_id, ())]:
                buckets.setdefault(key, {})[block.block_id] = block
        self.cached_block_hash_to_block = SimpleNamespace(_cache=buckets)
        self.free_block_queue = Queue(blocks)

    def evict_blocks(self, ids):
        assert self.subtree_eviction_in_progress
        for block in self.blocks:
            if block.block_id not in ids:
                continue
            assert block.ref_cnt == 0 and not block.is_null
            for key in [block.block_hash, *self.cached_block_hashes_by_block.pop(block.block_id, ())]:
                bucket = self.cached_block_hash_to_block._cache[key]
                del bucket[block.block_id]
                if not bucket:
                    del self.cached_block_hash_to_block._cache[key]
            block.block_hash = None


@pytest.fixture
def hash_api(monkeypatch):
    module = ModuleType("vllm.v1.core.kv_cache_utils")
    module.get_block_hash = lambda key: key[0]
    module.make_block_hash_with_group_id = lambda value, group: (value, group)
    monkeypatch.setitem(sys.modules, module.__name__, module)


def test_vllm_preserves_shared_active_unrelated_and_aliased_blocks(hash_api):
    blocks = [
        Block(0, "prompt"), Block(1, "old"), Block(2, "chosen"),
        Block(3, "discarded"), Block(4, "discarded", ref_cnt=1),
        Block(5, "unrelated"), Block(6, "discarded", group=1),
        Block(7, "discarded", group=1),
    ]
    pool = Pool(blocks, {7: {("chosen", 1)}})
    before_free = len(pool.free_block_queue.blocks)
    counts = evict_cached_prefixes(pool, {"prompt", "old", "discarded", "chosen"}, {"prompt", "chosen"}, range(2))

    assert counts == {"evicted_blocks": 3, "skipped_active_blocks": 1}
    assert {b.block_id for b in blocks if b.block_hash is None} == {1, 3, 6}
    assert {b.block_id for b in pool.free_block_queue.blocks[:3]} == {1, 3, 6}
    assert len(pool.free_block_queue.blocks) == before_free
    assert [b.ref_cnt for b in blocks] == [0, 0, 0, 0, 1, 0, 0, 0]
    assert not pool.subtree_eviction_in_progress
    assert evict_cached_prefixes(pool, {"old", "discarded"}, {"chosen"}, range(2))["evicted_blocks"] == 0


def test_vllm_all_reject_keeps_old_state_and_skips_recycled_ids(hash_api):
    # Block 1 previously held a proposal; its current identity is unrelated.
    pool = Pool([Block(0, "old"), Block(1, "recycled"), Block(2, "proposal")])
    assert evict_cached_prefixes(pool, {"old", "proposal"}, {"old"}, [0])["evicted_blocks"] == 1
    assert pool.blocks[0].block_hash == ("old", 0)
    assert pool.blocks[1].block_hash == ("recycled", 0)


def test_vllm_uses_engine_hashes_for_full_prompt_and_only_existing_entries(hash_api):
    pool = Pool([Block(0, (10, 11)), Block(1, (10, 11, 1, 2)), Block(2, (10, 11, 1, 3))])
    hashed = []

    def hasher(request):
        hashed.append(list(request.all_token_ids))
        assert request.block_hashes == [] and request.cache_salt is None
        return [tuple(request.all_token_ids[:end]) for end in range(2, request.num_tokens + 1, 2)]

    core = SimpleNamespace(
        request_block_hasher=hasher,
        scheduler=SimpleNamespace(kv_cache_manager=SimpleNamespace(
            enable_caching=True, block_pool=pool, num_kv_cache_groups=1,
        )),
    )
    counts = _engine_evict_subtree_cache(core, [[10, 11, 1, 2, 7], [10, 11, 1, 3, 9]], [10, 11, 1, 2])
    assert counts["evicted_blocks"] == 1
    assert hashed[0] == [10, 11, 1, 2]
    assert pool.blocks[0].block_hash and pool.blocks[1].block_hash
    core.scheduler.kv_cache_manager.enable_caching = False
    with pytest.raises(ValueError, match="prefix_caching"):
        _engine_evict_subtree_cache(core, [], [])


@pytest.mark.parametrize("multiprocess", [False, True])
def test_vllm_cleanup_reaches_scheduler_and_returns_acknowledgement(multiprocess):
    calls = []

    def utility(*args):
        calls.append(args)
        return {"evicted_blocks": 2}

    client = SimpleNamespace(call_utility=utility) if multiprocess else SimpleNamespace(
        engine_core=SimpleNamespace(evict_subtree_cache=utility),
    )
    llm = SimpleNamespace(llm_engine=SimpleNamespace(engine_core=client))
    assert call_cache_eviction(llm, [[1, 2]], [1, 3]) == {"evicted_blocks": 2}
    expected = [("evict_subtree_cache", [[1, 2]], [1, 3])] if multiprocess else [([[1, 2]], [1, 3])]
    assert calls == expected


def test_vllm_installs_the_utility_on_the_engine_core_base_class(monkeypatch):
    module = ModuleType("vllm.v1.engine.core")
    module.EngineCore = type("EngineCore", (), {})
    process_class = type("EngineCoreProc", (module.EngineCore,), {})
    monkeypatch.setitem(sys.modules, module.__name__, module)
    install_engine_core_eviction()
    assert process_class.evict_subtree_cache is _engine_evict_subtree_cache


class Node:
    def __init__(self, key, parent=None, lock_ref=0):
        self.key = list(key)
        self.value = list(key)
        self.parent = parent
        self.children = {}
        self.lock_ref = lock_ref
        if parent is not None:
            parent.children[tuple(key)] = self


class Radix:
    """Small compressed trie fixture with split-on-match and allocator accounting."""

    def __init__(self, page_size=1):
        self.root_node = Node([])
        self.disable = False
        self.page_size = page_size
        self.freed = []
        self.events = []
        self.token_to_kv_pool_allocator = SimpleNamespace(free=lambda value: self.freed.extend(value))

    def match_prefix(self, key):
        node = self.root_node
        key = list(key)
        while key:
            child = next((c for c in node.children.values() if c.key[:self.page_size] == key[:self.page_size]), None)
            if child is None:
                break
            count = 0
            for left, right in zip(child.key, key):
                if left != right:
                    break
                count += 1
            count = count // self.page_size * self.page_size
            if count < len(child.key):
                del node.children[tuple(child.key)]
                prefix = Node(child.key[:count], node, child.lock_ref)
                child.key = child.key[count:]
                child.value = child.value[count:]
                child.parent = prefix
                prefix.children[tuple(child.key)] = child
                child = prefix
            node = child
            key = key[count:]
        return SimpleNamespace(last_device_node=node)

    def _delete_leaf(self, node):
        assert not node.children and node.lock_ref == 0
        del node.parent.children[tuple(node.key)]

    def _record_remove_event(self, node):
        self.events.append(node)


def test_sglang_prunes_discarded_leaves_but_keeps_shared_and_locked_prefixes():
    cache = Radix()
    prompt = Node([10], cache.root_node)
    chosen = Node([1, 2], prompt)
    discarded = Node([3], prompt)
    Node([4], discarded)
    locked = Node([5, 6], prompt, lock_ref=1)
    unrelated = Node([7, 8], prompt)
    sequences = [[10, 1, 2], [10, 3, 4], [10, 5, 6], [10, 3, 4]]

    counts = evict_radix_paths(cache, sequences, [10, 1, 2])
    assert counts == {"evicted_tokens": 2, "evicted_nodes": 2}
    assert cache.freed == [4, 3]
    assert set(prompt.children.values()) == {chosen, locked, unrelated}
    assert evict_radix_paths(cache, sequences, [10, 1, 2])["evicted_tokens"] == 0


@pytest.mark.parametrize("page_size", [1, 2])
def test_sglang_splits_retained_prefix_before_deleting_eos_tail(page_size):
    cache = Radix(page_size)
    Node([10, 11, 12, 13, 14, 15], cache.root_node)
    counts = evict_radix_paths(cache, [[10, 11, 12, 13, 14, 15]], [10, 11, 12])
    expected = [13, 14, 15] if page_size == 1 else [12, 13, 14, 15]
    assert cache.freed == expected
    assert counts["evicted_tokens"] == len(expected)
    # The unaligned retained fragment may be recomputed; its preceding pages survive.
    assert cache.root_node.children


def test_sglang_preserves_a_discarded_prefix_needed_by_an_unrelated_descendant():
    cache = Radix()
    prefix = Node([1, 2], cache.root_node)
    Node([3, 4], prefix)
    assert evict_radix_paths(cache, [[1, 2]], [9])["evicted_tokens"] == 0


def test_sglang_rpc_returns_counts_and_rejects_other_cache_implementations(monkeypatch):
    from power_sharpening.backends.sglang.cache_eviction import _handle_cache_rpc

    scheduler_module = ModuleType("sglang.srt.managers.scheduler")

    class Response:
        def __init__(self, success, message):
            self.success, self.message = success, message

    barriers = []
    scheduler_module.RpcReqOutput = Response
    scheduler_module.barrier = lambda: barriers.append(True)
    radix_module = ModuleType("sglang.srt.mem_cache.radix_cache")
    radix_module.RadixCache = Radix
    for module in (scheduler_module, radix_module):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    cache = Radix()
    Node([1, 2], cache.root_node)
    scheduler = SimpleNamespace(tree_cache=cache, _subtree_original_handle_rpc=lambda req: "original")
    request = SimpleNamespace(method="evict_subtree_cache", parameters={"sequences": [[1, 2]], "keep_token_ids": [3]})
    response = _handle_cache_rpc(scheduler, request)
    assert response.success and '"evicted_tokens": 2' in response.message
    scheduler.tree_cache = object()
    assert not _handle_cache_rpc(scheduler, request).success
    assert barriers == [True, True]
    assert _handle_cache_rpc(scheduler, SimpleNamespace(method="other")) == "original"


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("accept", [False, True])
def test_sglang_sampler_cleans_up_only_after_scoring_and_mh(monkeypatch, enabled, accept):
    directory = Path(__file__).resolve().parents[1] / "power_sharpening/backends/sglang/samplers"
    for name, attr in [("low_temp_proposal_sampler", "low_temp_proposal_sampling"), ("proposal_model_call", "batched_proposal_callv2")]:
        stub = ModuleType("power_sharpening.backends.sglang.samplers." + name)
        setattr(stub, attr, None)
        monkeypatch.setitem(sys.modules, stub.__name__, stub)
    spec = importlib.util.spec_from_file_location("_sglang_subtree_cache_test", directory / "subtree_prefetching_sampler.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    events = []
    module.low_temp_proposal_sampling = lambda *a, **k: ([10, 1, 2], [-1., -1.], [-2., -2.], {})
    module.collect_batch_cut_indicesv3 = lambda *a, **k: ({0: 1}, {1: 0}, [1])

    def batch(*args, **kwargs):
        events.append("scored")
        return [[10, 3, 4]], [0.], [[-1., -1.]], [[-2., -2.]]

    def walk(**kwargs):
        events.append("walked")
        # The sampler walks the tree in response coordinates: prompt [10] stripped, cut 1 shifted to 0.
        assert kwargs["subtree_root_proposal"] == [1, 2]
        assert kwargs["sampled_proposals"] == [[3, 4]]
        assert kwargs["batch_cut_indices"] == [0]
        return SimpleNamespace(proposal=[3, 4] if accept else [1, 2]), SimpleNamespace(node_id=1), [(accept,)], int(accept)

    def evict(sequences, keep_token_ids):
        assert events == ["scored", "walked"]
        events.append("evicted")
        assert sequences == [[10, 1, 2], [10, 3, 4]]
        assert keep_token_ids == ([10, 3, 4] if accept else [10, 1, 2])
        return {"evicted_tokens": 2}

    module.batched_proposal_callv2 = batch
    module.build_and_sample_subtree = walk
    wrapper = SimpleNamespace(tokenizer=SimpleNamespace(eos_token_id=None))
    if enabled:
        wrapper.evict_subtree_cache = evict
    tokens, stats = module.subtree_prefetching_sampling(wrapper, [10], mcmc_steps=1, max_new_tokens=2, num_of_blocks=1, evict_subtree_cache=enabled)
    assert tokens == ([10, 3, 4] if accept else [10, 1, 2])
    assert len(stats.cache_evictions) == int(enabled)
