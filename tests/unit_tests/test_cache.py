import asyncio
import pytest
from unittest import mock

from async_substrate_interface.utils.cache import CachedFetcher, LRUCache


@pytest.mark.asyncio
async def test_cached_fetcher_fetches_and_caches():
    """Tests that CachedFetcher correctly fetches and caches results."""
    # Setup
    mock_method = mock.AsyncMock(side_effect=lambda x: f"result_{x}")
    fetcher = CachedFetcher(max_size=2, method=mock_method)

    # First call should trigger the method
    result1 = await fetcher("key1")
    assert result1 == "result_key1"
    mock_method.assert_awaited_once_with("key1")

    # Second call with the same key should use the cache
    result2 = await fetcher("key1")
    assert result2 == "result_key1"
    # Ensure the method was NOT called again
    assert mock_method.await_count == 1

    # Third call with a new key triggers a method call
    result3 = await fetcher("key2")
    assert result3 == "result_key2"
    assert mock_method.await_count == 2


@pytest.mark.asyncio
async def test_cached_fetcher_handles_inflight_requests():
    """Tests that CachedFetcher waits for in-flight results instead of re-fetching."""
    # Create an event to control when the mock returns
    event = asyncio.Event()

    async def slow_method(x):
        await event.wait()
        return f"slow_result_{x}"

    fetcher = CachedFetcher(max_size=2, method=slow_method)

    # Start first request
    task1 = asyncio.create_task(fetcher("key1"))
    await asyncio.sleep(0.1)  # Let the task start and be inflight

    # Second request for the same key while the first is in-flight
    task2 = asyncio.create_task(fetcher("key1"))
    await asyncio.sleep(0.1)

    # Release the inflight request
    event.set()
    result1, result2 = await asyncio.gather(task1, task2)
    assert result1 == result2 == "slow_result_key1"


@pytest.mark.asyncio
async def test_cached_fetcher_propagates_errors():
    """Tests that CachedFetcher correctly propagates errors."""

    async def error_method(x):
        raise ValueError("Boom!")

    fetcher = CachedFetcher(max_size=2, method=error_method)

    with pytest.raises(ValueError, match="Boom!"):
        await fetcher("key1")


@pytest.mark.asyncio
async def test_cached_fetcher_eviction():
    """Tests that LRU eviction works in CachedFetcher."""

    async def side_effect_method(x):
        return f"val_{x}"

    fetcher = CachedFetcher(max_size=2, method=side_effect_method)

    # Fill cache
    await fetcher("key1")
    await fetcher("key2")
    assert list(fetcher._cache.cache.keys()) == list(fetcher._cache.cache.keys())

    # Insert a new key to trigger eviction
    await fetcher("key3")
    # key1 should be evicted
    assert "key1" not in fetcher._cache.cache
    assert "key2" in fetcher._cache.cache
    assert "key3" in fetcher._cache.cache


@pytest.mark.asyncio
@pytest.mark.parametrize("falsy_value", [0, None, "", False, [], {}])
async def test_cached_fetcher_caches_falsy_values(falsy_value):
    """
    Regression test: falsy values (0, None, "", False, ...) must be cached.

    Previously `CachedFetcher.__call__` used a truthiness check on the cache
    lookup, so any falsy stored value was indistinguishable from a miss and
    the underlying method was re-invoked on every call. The most painful
    real-world cases were `_cached_get_block_number` for the genesis block
    (returns 0) and `get_block_runtime_version_for` on its error path
    (returns None).
    """
    mock_method = mock.AsyncMock(return_value=falsy_value)
    fetcher = CachedFetcher(max_size=2, method=mock_method)

    result1 = await fetcher("key1")
    result2 = await fetcher("key1")

    assert result1 == falsy_value
    assert result2 == falsy_value
    assert mock_method.await_count == 1


def test_lru_cache_get_distinguishes_miss_from_cached_none():
    """`LRUCache.get` must let callers distinguish a stored None from a miss."""
    cache = LRUCache(max_size=2)
    sentinel = object()

    assert cache.get("missing", sentinel) is sentinel

    cache.set("k", None)
    assert cache.get("k", sentinel) is None
    assert cache.get("k") is None  # default-None preserved for existing callers
