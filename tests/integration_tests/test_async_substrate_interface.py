import asyncio
import logging
import os.path
import time
import threading
import socket

import bittensor_wallet
import pytest
from scalecodec import ss58_encode

from async_substrate_interface.async_substrate import AsyncSubstrateInterface, logger
from async_substrate_interface.types import ScaleObj
from tests.helpers.settings import ARCHIVE_ENTRYPOINT, LATENT_LITE_ENTRYPOINT
from tests.helpers.proxy_server import ProxyServer


@pytest.mark.asyncio
async def test_legacy_decoding():
    print("Testing test_legacy_decoding")
    # roughly 4000 blocks before metadata v15 was added
    pre_metadata_v15_block = 3_010_611

    async with AsyncSubstrateInterface(ARCHIVE_ENTRYPOINT) as substrate:
        block_hash = await substrate.get_block_hash(pre_metadata_v15_block)
        events = await substrate.get_events(block_hash)
        assert isinstance(events, list)

        query_map_result = await substrate.query_map(
            module="SubtensorModule",
            storage_function="NetworksAdded",
            block_hash=block_hash,
        )
        async for key, value in query_map_result:
            assert isinstance(key, int)
            assert isinstance(value, ScaleObj)

        timestamp = await substrate.query(
            "Timestamp",
            "Now",
            block_hash=block_hash,
        )
        assert timestamp.value == 1716358476004
    print("test_legacy_decoding succeeded")


@pytest.mark.asyncio
async def test_ss58_conversion():
    print("Testing test_ss58_conversion")
    async with AsyncSubstrateInterface(
        LATENT_LITE_ENTRYPOINT, ss58_format=42
    ) as substrate:
        block_hash = await substrate.get_chain_finalised_head()

        qm = await substrate.query_map(
            "SubtensorModule",
            "OwnedHotkeys",
            block_hash=block_hash,
        )
        for key, value in qm.records:
            assert isinstance(key, str)
            assert isinstance(value, ScaleObj)
            assert isinstance(value.value, list)
            if len(value.value) > 0:
                for decoded_key in value.value:
                    assert isinstance(decoded_key, str)
    print("test_ss58_conversion succeeded")


@pytest.mark.asyncio
async def test_fully_exhaust_query_map():
    print("Testing test_fully_exhaust_query_map")
    async with AsyncSubstrateInterface(LATENT_LITE_ENTRYPOINT) as substrate:
        block_hash = await substrate.get_chain_finalised_head()
        non_fully_exhauster_start = time.time()
        non_fully_exhausted_qm = await substrate.query_map(
            "SubtensorModule",
            "CRV3WeightCommits",
            block_hash=block_hash,
        )
        initial_records_count = len(non_fully_exhausted_qm.records)
        assert initial_records_count <= 100  # default page size
        exhausted_records_count = 0
        async for _ in non_fully_exhausted_qm:
            exhausted_records_count += 1
        non_fully_exhausted_time = time.time() - non_fully_exhauster_start

        assert len(non_fully_exhausted_qm.records) >= initial_records_count
        fully_exhausted_start = time.time()
        fully_exhausted_qm = await substrate.query_map(
            "SubtensorModule",
            "CRV3WeightCommits",
            block_hash=block_hash,
            fully_exhaust=True,
        )

        fully_exhausted_time = time.time() - fully_exhausted_start
        initial_records_count_fully_exhaust = len(fully_exhausted_qm.records)
        assert fully_exhausted_time <= non_fully_exhausted_time, (
            f"Fully exhausted took longer than non-fully exhausted with "
            f"{len(non_fully_exhausted_qm.records)} records in non-fully exhausted "
            f"in {non_fully_exhausted_time} seconds, and {initial_records_count_fully_exhaust} in fully exhausted"
            f" in {fully_exhausted_time} seconds. This could be caused by the fact that on this specific block, "
            f"there are fewer records than take up a single page. This difference should still be small."
        )
        fully_exhausted_records_count = 0
        async for _ in fully_exhausted_qm:
            fully_exhausted_records_count += 1
        assert fully_exhausted_records_count == initial_records_count_fully_exhaust
        assert initial_records_count_fully_exhaust == exhausted_records_count
    print("test_fully_exhaust_query_map succeeded")


@pytest.mark.asyncio
async def test_get_events_proper_decoding():
    print("Testing test_get_events_proper_decoding")
    # known block/hash pair that has the events we seek to decode
    block = 5846788
    block_hash = "0x0a1c45063a59b934bfee827caa25385e60d5ec1fd8566a58b5cc4affc4eec412"

    async with AsyncSubstrateInterface(ARCHIVE_ENTRYPOINT) as substrate:
        all_events = await substrate.get_events(block_hash=block_hash)
        event = all_events[1]
        assert event["attributes"] == (
            "5G1NjW9YhXLadMWajvTkfcJy6up3yH2q1YzMXDTi6ijanChe",
            30,
            "0xa6b4e5c8241d60ece0c25056b19f7d21ae845269fc771ad46bf3e011865129a5",
        )
    print("test_get_events_proper_decoding succeeded")


@pytest.mark.asyncio
async def test_query_multiple():
    print("Testing test_query_multiple")
    block = 6153277
    cks = [
        "5FH9AQM4kqbkdC9jyV5FrdEWVYt41nkhFstop7Vhyfb9ZsXt",
        "5GQxLKxjZWNZDsghmYcw7P6ahC7XJCjx1WD94WGh92quSycx",
        "5EcaPiDT1cv951SkCFsvdHDs2yAEUWhJDuRP9mHb343WnaVn",
    ]
    async with AsyncSubstrateInterface(ARCHIVE_ENTRYPOINT) as substrate:
        block_hash = await substrate.get_block_hash(block_id=block)
        assert await substrate.query_multiple(
            params=cks,
            module="SubtensorModule",
            storage_function="OwnedHotkeys",
            block_hash=block_hash,
        )
    print("test_query_multiple succeeded")


@pytest.mark.asyncio
async def test_reconnection():
    print("Testing test_reconnection")
    async with AsyncSubstrateInterface(
        ARCHIVE_ENTRYPOINT, ss58_format=42, retry_timeout=8.0
    ) as substrate:
        await asyncio.sleep(9)  # sleep for longer than the retry timeout
        bh = await substrate.get_chain_finalised_head()
        assert isinstance(bh, str)
        assert isinstance(await substrate.get_block_number(bh), int)
    print("test_reconnection succeeded")


@pytest.mark.asyncio
async def test_query_map_with_odd_number_of_params():
    print("Testing test_query_map_with_odd_number_of_params")
    async with AsyncSubstrateInterface(ARCHIVE_ENTRYPOINT, ss58_format=42) as substrate:
        qm = await substrate.query_map(
            "SubtensorModule",
            "Alpha",
            ["5CoZxgtfhcJKX2HmkwnsN18KbaT9aih9eF3b6qVPTgAUbifj"],
        )
        first_record = qm.records[0]
        assert len(first_record) == 2
        assert len(first_record[0]) == 2
    print("test_query_map_with_odd_number_of_params succeeded")


@pytest.mark.skip("Weird issue with the GitHub Actions runner")
@pytest.mark.asyncio
async def test_improved_reconnection():
    def get_free_port():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))  # Bind to port 0 = OS picks free port
            s.listen(1)
            port_ = s.getsockname()[1]
        return port_

    print("Testing test_improved_reconnection")
    ws_logger_path = "/tmp/websockets-proxy-test"
    ws_logger = logging.getLogger("websockets.proxy")
    if os.path.exists(ws_logger_path):
        os.remove(ws_logger_path)
    ws_logger.setLevel(logging.INFO)
    ws_logger.addHandler(logging.FileHandler(ws_logger_path))

    asi_logger_path = "/tmp/async-substrate-interface-test"
    if os.path.exists(asi_logger_path):
        os.remove(asi_logger_path)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.FileHandler(asi_logger_path))
    port = get_free_port()
    print(f"Testing using server on port {port}")
    proxy = ProxyServer("wss://archive.sub.latent.to", 10, 20, port=port)

    server_thread = threading.Thread(target=proxy.connect_and_serve, daemon=True)
    server_thread.start()
    await asyncio.sleep(3)  # give the server start up time
    async with AsyncSubstrateInterface(
        f"ws://localhost:{port}",
        ss58_format=42,
        chain_name="Bittensor",
        retry_timeout=10.0,
        ws_shutdown_timer=None,
    ) as substrate:
        blocks_to_check = [
            5215000,
            5215001,
            5215002,
            5215003,
            5215004,
            5215005,
            5215006,
        ]
        tasks = []
        for block in blocks_to_check:
            block_hash = await substrate.get_block_hash(block_id=block)
            tasks.append(
                substrate.query_map(
                    "SubtensorModule", "TotalHotkeyShares", block_hash=block_hash
                )
            )
        records = await asyncio.gather(*tasks)
        assert len(records) == len(blocks_to_check)
        await substrate.close()
        with open(ws_logger_path, "r") as f:
            assert "Pausing" in f.read()
        with open(asi_logger_path, "r") as f:
            assert "Timeout/ConnectionClosed occurred." in f.read()
    shutdown_thread = threading.Thread(target=proxy.close, daemon=True)
    shutdown_thread.start()
    shutdown_thread.join(timeout=5)
    server_thread.join(timeout=5)
    print("test_improved_reconnection succeeded")


@pytest.mark.asyncio
async def test_get_payment_info():
    print("Testing test_get_payment_info")
    alice_coldkey = bittensor_wallet.Keypair.create_from_uri("//Alice")
    bob_coldkey = bittensor_wallet.Keypair.create_from_uri("//Bob")
    async with AsyncSubstrateInterface(
        LATENT_LITE_ENTRYPOINT, ss58_format=42, chain_name="Bittensor"
    ) as substrate:
        block_hash = await substrate.get_chain_head()
        call = await substrate.compose_call(
            "Balances",
            "transfer_keep_alive",
            {"dest": bob_coldkey.ss58_address, "value": 100_000},
            block_hash,
        )
        payment_info = await substrate.get_payment_info(
            call=call,
            keypair=alice_coldkey,
        )
        partial_fee_no_era = payment_info["partial_fee"]
        assert partial_fee_no_era > 0
        payment_info_era = await substrate.get_payment_info(
            call=call, keypair=alice_coldkey, era={"period": 64}
        )
        partial_fee_era = payment_info_era["partial_fee"]
        assert partial_fee_era > partial_fee_no_era

        payment_info_all_options = await substrate.get_payment_info(
            call=call,
            keypair=alice_coldkey,
            era={"period": 64},
            nonce=await substrate.get_account_nonce(alice_coldkey.ss58_address),
            tip=5_000_000,
            tip_asset_id=64,
        )
        partial_fee_all_options = payment_info_all_options["partial_fee"]
        assert partial_fee_all_options > partial_fee_no_era
        assert partial_fee_all_options > partial_fee_era
    print("test_get_payment_info succeeded")


@pytest.mark.asyncio
async def test_concurrent_rpc_requests():
    """
    Test that multiple concurrent RPC requests on a shared connection work correctly.

    This test verifies the fix for the issue where multiple concurrent tasks
    re-initializing the WebSocket connection caused requests to hang.
    """
    print("Testing test_concurrent_rpc_requests")

    async def concurrent_task(substrate_, task_id):
        """Make multiple RPC calls from a single task."""
        for i in range(5):
            result = await substrate_.get_block_number(None)
            assert isinstance(result, int)
            assert result > 0

    async with AsyncSubstrateInterface(LATENT_LITE_ENTRYPOINT) as substrate:
        # Run 5 concurrent tasks, each making 5 RPC calls (25 total)
        # This tests that the connection is properly shared without re-initialization
        tasks = [concurrent_task(substrate, i) for i in range(5)]
        await asyncio.gather(*tasks)

    print("test_concurrent_rpc_requests succeeded")


@pytest.mark.asyncio
async def test_wait_for_block():
    async def handler(_):
        return True

    substrate = AsyncSubstrateInterface(
        LATENT_LITE_ENTRYPOINT, ss58_format=42, chain_name="Bittensor"
    )
    await substrate.initialize()
    current_block = await substrate.get_block_number(None)
    result = await substrate.wait_for_block(
        current_block + 3, result_handler=handler, task_return=False
    )
    assert result is True


@pytest.mark.asyncio
async def test_old_runtime_calls():
    from bittensor import SubtensorApi

    sub = SubtensorApi(
        network=ARCHIVE_ENTRYPOINT, legacy_methods=True, async_subtensor=True
    )
    await sub.initialize()
    # will pass
    assert sub.get_stake_info_for_coldkey(
        "5CQ6dMW8JZhKCZX9kWsZRqa3kZRKmNHxbPPVFEt6FgyvGv2G", 4943592
    )
    # needs to use legacy
    assert sub.get_stake_info_for_coldkey(
        "5CQ6dMW8JZhKCZX9kWsZRqa3kZRKmNHxbPPVFEt6FgyvGv2G", 4670227
    )
    await sub.close()
