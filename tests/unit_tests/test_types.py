from async_substrate_interface.types import Runtime, RuntimeCache
from async_substrate_interface.async_substrate import DiskCachedAsyncSubstrateInterface
from async_substrate_interface.utils import cache

import sqlite3
import os
import pickle
import pytest
from unittest.mock import patch


def test_runtime_cache():
    fake_block = 2
    fake_hash = "0xignore"
    fake_version = 271

    new_fake_block = 3
    newer_fake_block = 4

    new_fake_hash = "0xnewfakehash"

    runtime = Runtime("", None, None)
    runtime_cache = RuntimeCache()
    # insert our Runtime object into the cache with a set block, hash, and version
    runtime_cache.add_item(runtime, fake_block, fake_hash, fake_version)

    assert runtime_cache.retrieve(fake_block) is not None
    # cache does not yet know that new_fake_block has the same runtime
    assert runtime_cache.retrieve(new_fake_block) is None
    assert (
        runtime_cache.retrieve(
            new_fake_block, new_fake_hash, runtime_version=fake_version
        )
        is not None
    )
    # after checking the runtime with the new block, it now knows this runtime should also map to this block
    assert runtime_cache.retrieve(new_fake_block) is not None
    assert runtime_cache.retrieve(newer_fake_block) is None
    assert runtime_cache.retrieve(newer_fake_block, fake_hash) is not None
    assert runtime_cache.retrieve(newer_fake_block) is not None
    assert runtime_cache.retrieve(fake_block, block_hash=new_fake_hash) is not None
    assert runtime_cache.retrieve(block_hash=new_fake_hash) is not None


@pytest.mark.asyncio
async def test_runtime_cache_from_disk():
    test_db_location = "/tmp/async-substrate-interface-test-cache"
    fake_chain = "ws://fake.com"
    fake_block = 1
    fake_hash = "0xignore"
    new_fake_block = 2
    new_fake_hash = "0xnewfakehash"

    if os.path.exists(test_db_location):
        os.remove(test_db_location)
    with patch.object(cache, "CACHE_LOCATION", test_db_location):
        substrate = DiskCachedAsyncSubstrateInterface(fake_chain, _mock=True)
        # Needed to avoid trying to initialize on the network during `substrate.initialize()`
        substrate.initialized = True

        # runtime cache should be completely empty
        assert len(substrate.runtime_cache.block_hashes.cache) == 0
        assert len(substrate.runtime_cache.blocks.cache) == 0
        assert len(substrate.runtime_cache.versions.cache) == 0
        await substrate.initialize()

        # after initialization, runtime cache should still be completely empty
        assert len(substrate.runtime_cache.block_hashes.cache) == 0
        assert len(substrate.runtime_cache.blocks.cache) == 0
        assert len(substrate.runtime_cache.versions.cache) == 0
        await substrate.close()

        # ensure we have created the SQLite DB during initialize()
        assert os.path.exists(test_db_location)

        # insert some fake data into our DB
        conn = sqlite3.connect(test_db_location)
        conn.execute(
            "INSERT INTO RuntimeCache_blocks_v4 (key, value, chain) VALUES (?, ?, ?)",
            (fake_block, pickle.dumps(fake_hash), fake_chain),
        )
        conn.commit()
        conn.close()

        substrate.initialized = True
        await substrate.initialize()
        assert substrate.runtime_cache.blocks.cache == {fake_block: fake_hash}
        # add an item to the cache
        substrate.runtime_cache.add_item(
            runtime=None, block_hash=new_fake_hash, block=new_fake_block
        )
        await substrate.close()

        # verify that our added item is now in the DB
        conn = sqlite3.connect(test_db_location)
        cursor = conn.cursor()
        cursor.execute("SELECT key, value, chain FROM RuntimeCache_blocks_v4")
        query = cursor.fetchall()
        cursor.close()
        conn.close()

        first_row = query[0]
        assert first_row[0] == fake_block
        assert pickle.loads(first_row[1]) == fake_hash
        assert first_row[2] == fake_chain

        second_row = query[1]
        assert second_row[0] == new_fake_block
        assert pickle.loads(second_row[1]) == new_fake_hash
        assert second_row[2] == fake_chain
